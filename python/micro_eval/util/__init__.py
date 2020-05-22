from __future__ import annotations
from collections import Iterable, OrderedDict
import json
import logging
import os
import subprocess
import typing

import numpy as np

import tvm
import topi
from tvm import autotvm, relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw, schedule_conv2d_nhwc
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline
from topi.testing import conv2d_nchw_python

REPO_ROOT = None
def get_repo_root():
    global REPO_ROOT
    if REPO_ROOT is None:
        REPO_ROOT = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                                cwd=os.path.dirname(__file__)), 'utf-8').strip('\n')
    return REPO_ROOT


CMSIS_NN_PATH = f'{get_repo_root()}/3rdparty/CMSIS_5'
CMSIS_ST_PATH = f'{get_repo_root()}/3rdparty/STM32CubeF7/Drivers/CMSIS'


if os.environ.get('CMSIS_ST_PATH', None) not in (None, CMSIS_ST_PATH):
    print('NOTE: overriding environment variable CMSIS_ST_PATH', file=sys.stderr)
    print(f' - new value: {CMSIS_ST_PATH}', file=sys.stderr)
os.environ['CMSIS_ST_PATH'] = CMSIS_ST_PATH


def get_logger(log_file_name):
    logger = logging.getLogger('micro_eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s:%(lineno)d %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class LabelledTensor:
    """A tensor with labelled axes."""

    def __init__(self, data: np.ndarray, shape: LabelledShape):
        """Wraps an ndarray with a given string layout.

        Parameters
        ----------
        data : np.ndarray
            The ndarray containing the underlying data.

        layout : str
            A string naming the axes of `data`. Each char gives its corresponding axis a
            single-char name.
        """
        self.data = data
        self.shape = shape
        assert isinstance(shape, LabelledShape)

    def resize(self, other: LabelledShape):
        """Resize dimensions (per numpy.resize) and potentially transpose.

        Parameters
        ----------
        other : LabelledShape
            The new shape to take on. Per numpy.transpose, newly-created tensor entries will be
            0-padded.

        Returns
        -------
        LabelledTensor : The resized tensor.
        """
        assert other.dtype == self.shape.dtype
        intermediate_shape = LabelledShape.from_dims_and_layout(self.shape, other.layout, dtype=self.shape.dtype)
        intermediate = self.transpose(intermediate_shape)
        zeroes = np.zeros(other.shape, dtype=other.dtype)
        assign_shape = tuple(slice(0,s) for s in intermediate.data.shape)
        zeroes[assign_shape] = intermediate.data
        return LabelledTensor(zeroes, other) #np.resize(intermediate.data, other.shape), other)

    def transpose(self, other: LabelledShape):
        mapping = self.shape.make_transpose_mapping(other)
        return LabelledTensor(np.transpose(self.data, mapping), other)

    def with_layout(self, layout):
        new_shape_dims = OrderedDict([(l, self.shape.dims[l]) for l in layout])
        return self.transpose(LabelledShape(dims=new_shape_dims, dtype=self.shape.dtype))


class LabelledShape:

    @classmethod
    def from_dims_and_layout(cls, dims: typing.Dict[str, int], layout: str, dtype: str):
        return cls(dim_iter=((l, dims[l]) for l in layout), dtype=dtype)

    def __init__(self,
                 dims: typing.Union[OrderedDict, NoneType] = None,
                 dim_iter: typing.Union[typing.Iterable[typing.Tuple[str, int]], NoneType] = None,
                 dtype: str = None,
                 **kw):
        if dims is not None:
            self.dims = OrderedDict(dims)
        elif dim_iter is not None:
            self.dims = OrderedDict(list(dim_iter))
        else:
            self.dims = OrderedDict(kw.items())

        self.dtype = dtype

    def __getitem__(self, k):
        return self.dims[k]

    def __repr__(self):
        dims = ', '.join(f'{k}={v}' for k, v in self.dims.items())
        return f'{self.__class__.__name__}(dtype={self.dtype!r}, {dims})'

    def serialize(self):
        """Serialize to an AutoTVM style spec (e.g., `('TENSOR', (1, 2, 3), 'int8')`)."""
        return ('TENSOR', self.shape, self.dtype)

    def gen_rand_tensor(self, low, high) -> LabelledTensor:
        if 'int' in self.dtype:
            data_np = np.random.randint(low, high, size=self.shape, dtype=self.dtype)
        elif 'float' in self.dtype:
            data_np = np.random.uniform(low, high, size=self.shape, dtype=self.dtype)
        else:
            assert False, 'unknown dtype'

        return LabelledTensor(data_np, self)

    def gen_zero_tensor(self) -> LabelledTensor:
        return LabelledTensor(np.zeros(self.shape, dtype=self.dtype), self)

    @property
    def size(self):
        prod = 1
        for x in self.dims.values():
            prod *= x
        return prod

    def with_layout(self, layout):
        assert len(self.dims) == len(layout)
        return LabelledShape(dim_iter=((l, self.dims[l]) for l in layout), dtype=self.dtype)

    def make_transpose_mapping(self, other: LabelledShape):
        assert self.size == other.size
        assert len(self.dims) == len(other.dims)

        indices = []
        self_dim_keys = list(self.dims.keys())
        for dim_name, dim_value in other.dims.items():
            idx = self_dim_keys.index(dim_name)
            assert idx != -1
            indices.append(idx)
            assert dim_name in self.dims

        return indices

    def as_template_for(self, **new_dims):
        return LabelledShape(dims=((k, new_dims.get(k, self.dims[k])) for k in self.dims.keys()),
                             dtype=self.dtype)

    @property
    def shape(self):
        return tuple(self.dims.values())

    @property
    def layout(self):
        return ''.join(self.dims.keys())


def get_axis_len(iter_var):
    axis_range = iter_var.dom
    low = axis_range.min.value
    high = axis_range.extent.value
    return high - low


def get_op_output_shape(op):
    return list(map(get_axis_len, op.axis))


def print_c_source(sched, arg_bufs):
    print(tvm.build(sched, arg_bufs, target='c').get_source())


def show_c_source(sched, arg_bufs):
    input(tvm.build(sched, arg_bufs, target='c').get_source())


DEBUG_MODE = False


def relay_micro_build(func, dev_config, target, params=None, lib_headers=None, lib_include_paths=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : TODO
        TODO

    target : TODO
        TODO

    params : dict
        input parameters that do not change during inference

    lib_headers : TODO
        TODO

    lib_include_paths : TODO
        TODO

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device

    """
    with tvm.target.build_config(opt_level=3, disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target=target, params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config, lib_headers=lib_headers, lib_include_paths=lib_include_paths)
    ctx = tvm.micro_dev(0)
    if DEBUG_MODE:
        dump_root = f'{get_repo_root()}/debug/micro'
        mod = debug_runtime.create(graph, micro_mod, ctx, dump_root=dump_root)
    else:
        mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def gen_workload_desc_from_task(task):
    if 'conv2d' not in task[1]:
        return None
    workload = ['conv2d']
    args = task[2]
    for arg in args:
        if isinstance(arg, list) and arg[0] == 'TENSOR':
            res = list(arg[1])
            res.append(arg[2])
        else:
            res = arg
        workload.append(res)
    return workload


def deep_tuple(elem):
    """Recursively convert all lists in `elem` into tuples."""
    if isinstance(elem, list):
        return tuple(list(map(deep_tuple, elem)))
    else:
        return elem


def custom_pick_best(in_log_file_name, out_log_file_name, top_k=1):
    workload_to_best = {}
    with open(in_log_file_name, 'r') as f:
        for line in f:
            entry = json.loads(line)
            workload = gen_workload_desc_from_task(entry['input'])
            entry['input'][4] = workload
            hashable_workload = deep_tuple(workload)
            if hashable_workload not in workload_to_best:
                workload_to_best[hashable_workload] = []

            if len(workload_to_best[hashable_workload]) < top_k:
                workload_to_best[hashable_workload].append(entry)
            else:
                worst_entry = workload_to_best[hashable_workload][0]
                worst_entry_idx = 0
                for i, top_entry in enumerate(workload_to_best[hashable_workload]):
                    if top_entry['r'][0][0] > worst_entry['r'][0][0]:
                        worst_entry = top_entry
                        worst_entry_idx = i
                if entry['r'][0][0] < worst_entry['r'][0][0]:
                    workload_to_best[hashable_workload][worst_entry_idx] = entry

    with open(out_log_file_name, 'w') as f:
        for entries in workload_to_best.values():
            for entry in entries:
                f.write(json.dumps(entry) + '\n')


if 'MICRO_GDB_INIT_DIR' in os.environ:
    MICRO_GDB_DEBUG_PATH = os.environ['MICRO_GDB_INIT_DIR']
else:
    MICRO_GDB_DEBUG_PATH = os.environ['MICRO_GDB_INIT_DIR'] = f'{get_repo_root()}/debug/micro'


def reset_gdbinit(dev_config):
    if 'server_port' not in dev_config:
        return
    gdb_init_dir = MICRO_GDB_DEBUG_PATH
    with open(f'{gdb_init_dir}/.gdbinit', 'w') as f:
        gdb_port = dev_config['server_port'] - 3333
        gdbinit_contents = (
f"""#layout src
target remote localhost:{gdb_port}
set $pc = UTVMInit
break UTVMDone

define print_utvm_args
    set $i = 0
    while $i < utvm_num_tasks
        set $j = 0
        eval "print \\"TASK %d ARGS\\"", $i
        eval "set $num_task_args = utvm_tasks[$i].num_args"
        print "num_args: %d", $num_task_args
        while $j < $num_task_args
            eval "set $num_bits = ((DLTensor*) utvm_tasks[0].arg_values[0].v_handle)->dtype.bits"
            if $num_bits == 8
                print "dtype: int8"
                eval "p/d *((int8_t*) ((DLTensor*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            if $num_bits == 32
                print "dtype: int32"
                eval "p/d *((int32_t*) ((DLTensor*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            set $j = $j + 1
        end
        set $i = $i + 1
    end
end

print_utvm_args
""")
        f.write(gdbinit_contents)


def get_comm_overhead(dev_config, num_trials=1):
    """Get communication overhead by executing an empty kernel."""
    class EmptyCMod:
        def __init__(self):
            pass

        def export_library(self, out_obj_path, fcompile=None):
            assert fcompile is not None
            fcompile(out_obj_path, f'{os.path.dirname(__file__)}/empty.c')

    # do multiple trials, then calc the average comm overhead
    results = []
    with micro.Session(dev_config) as sess:
        micro_mod = create_micro_mod(EmptyCMod(), dev_config)
        micro_func = micro_mod['empty']
        for _ in range(num_trials):
            results.append(benchmark_micro_func(sess, micro_func, [], 1, 0.0))
    return sum(results) / len(results)


def benchmark_micro_func(sess, micro_func, args, num_trials):
    ctx = tvm.micro_dev(0)
    # sync before and after to ensure these are the only tasks in the queue
    ctx.sync()
    sess.get_last_batch_time()
    for _ in range(num_trials):
        micro_func(*args)
    ctx.sync()
    return (sess.get_last_batch_time() / num_trials)


def check_conv2d_output(
        data_tensor: LabelledTensor, kernel_tensor: LabelledTensor,
        micro_output_tensor: Labelled_Tensor, strides, padding):
    data_nchw_np = data_tensor.with_layout('NCHW').data
    kernel_oihw_np = kernel_tensor.with_layout('OIHW').data
    micro_output_nchw_np = micro_output_tensor.with_layout('NCHW').data

    topi_output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, strides, padding)
    tvm.testing.assert_allclose(micro_output_nchw_np.shape, topi_output_np.shape)
    for i in range(micro_output_nchw_np.shape[0]):
        tvm.testing.assert_allclose(micro_output_nchw_np[i], topi_output_np[i])
        print('ok', micro_output_nchw_np[i])


def calc_param_mem_footprint(func, units='KB'):
    """Calculate how many bytes are required to store the parameters of `func`.

    Report is in `units` format.
    """
    def _sizeof(dtype):
        if dtype in ('int8', 'uint8'):
            return 1
        elif dtype == 'float32':
            return 4
        else:
            assert False
    result = 0
    for arg in func.params:
        typ = arg.type_annotation
        shape = get_const_tuple(typ.shape)
        prod = 1
        for dim in shape:
            prod *= dim
        result += prod * _sizeof(typ.dtype)

    if units == 'B':
        return result
    elif units == 'KB':
        return result * 1e-3
    elif units == 'MB':
        return result * 1e-6
    else:
        assert False
