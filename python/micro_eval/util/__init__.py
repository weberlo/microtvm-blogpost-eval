from collections import Iterable, OrderedDict
import json
import logging
import os
from typing import *

import numpy as np

import tvm
import topi
from tvm import autotvm, relay
from tvm.relay import create_executor
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

if 'CMSIS_PATH' not in os.environ:
    raise RuntimeError('must have "CMSIS_PATH" in environment')
CMSIS_PATH = os.environ['CMSIS_PATH']
CMSIS_HEADERS = [
    'cmsis_gcc.h',
    'arm_math.h',
    'arm_nnsupportfunctions.h'
]
CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include'
]

def get_logger(log_file_name):
    logger = logging.getLogger('micro_eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class NamedTensor:
    def __init__(self, data: np.ndarray, layout: str):
        self.typ = BakedType(zip(layout, data.shape), dtype=str(data.dtype))
        self.data = data

    def with_layout(self, layout):
        indices = []
        for dim in layout:
            idx = self.typ.layout.index(dim)
            assert idx != -1
            indices.append(idx)
        return NamedTensor(self.data.transpose(tuple(indices)), layout)

    def resize(self, **new_dims):
        for dim_name, dim_size in new_dims.items():
            self.typ.dims[dim_name] = dim_size
        self.data.resize(self.typ.shape)


class NamedType:
    def __init__(self, dim_dict: Dict[str, int], dtype=None):
        self.dim_dict = dim_dict
        if dtype is not None:
            self._dtype = dtype

    def with_layout(self, layout):
        layout_list = []
        for dim_name in layout:
            dim_size = self.dim_dict[dim_name]
            layout_list.append((dim_name, dim_size))
        return BakedType(layout_list, dtype=getattr(self, '_dtype', None))

    def gen_rand_tensor(self, low, high) -> NamedTensor:
        """Create a tensor with random entries between `low` and `high`.

        Useful for testing multiple ops with different input layouts.
        """
        # create a baked type with an arbitrary layout to use its random tensor generation method
        return self.with_layout(list(self.dim_dict.keys())).gen_rand_tensor(low, high)

    def gen_empty(self) -> NamedTensor:
        # create a baked type with an arbitrary layout to use its empty tensor generation method
        return self.with_layout(list(self.dim_dict.keys())).gen_zero_tensor()

    @property
    def dtype(self):
        assert hasattr(self, '_dtype'), 'no dtype has been set'
        return self._dtype


class BakedType:
    def __init__(self, dim_iter: Iterable[Tuple[str, int]], dtype=None):
        # layout = []
        # shape = []
        # assert isinstance(dim_iter, Iterable)
        # for dim_name, dim_size in dim_iter:
        #     # assert len(dim_name) == 1, 'dimension names must be single characters'
        #     # setattr(self, dim_name, dim_size)
        #     layout.append(dim_name)
        #     shape.append(dim_size)
        # self.layout = layout
        # self.shape = tuple(shape)
        self.dims = OrderedDict(list(dim_iter))
        if dtype is not None:
            self._dtype = dtype

    def serialize(self):
        """Serialize to an AutoTVM style spec (e.g., `('TENSOR', (1, 2, 3), 'int8')`)."""
        return ('TENSOR', self.shape, self.dtype)

    def gen_rand_tensor(self, low, high) -> NamedTensor:
        if 'int' in self.dtype:
            data_np = np.random.randint(low, high, size=self.shape, dtype=self.dtype)
        elif 'float' in self.dtype:
            data_np = np.random.uniform(low, high, size=self.shape, dtype=self.dtype)
        else:
            assert False, 'unknown dtype'
        return NamedTensor(data_np, self.layout)

    def gen_zero_tensor(self) -> NamedTensor:
        return NamedTensor(np.zeros(self.shape, dtype=self.dtype), self.layout)

    @property
    def shape(self):
        return tuple(self.dims.values())

    @property
    def layout(self):
        return tuple(self.dims.keys())

    @property
    def dtype(self):
        assert hasattr(self, '_dtype'), 'no dtype has been set'
        return self._dtype


# def transform_data_layout(data_np, from_layout, to_layout):
#     indices = []
#     for dim in to_layout:
#         idx = from_layout.index(dim)
#         assert idx != -1
#         indices.append(idx)
#     return data_np.transpose(tuple(indices))


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
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        with tvm.build_config(disable_vectorize=True):
            graph, c_mod, params = relay.build(func, target=target, params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config, lib_headers=lib_headers, lib_include_paths=lib_include_paths)
    ctx = tvm.micro_dev(0)
    if DEBUG_MODE:
        dump_root = f'{os.path.dirname(__file__)}/../../debug/micro'
        mod = debug_runtime.create(graph, micro_mod, ctx, dump_root=dump_root)
    else:
        mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def eval_relay_intrp(mod, args):
    main_gv = relay.GlobalVar('main')
    mod = relay.Module({main_gv: mod['main']})
    intrp = create_executor("debug", mod)
    f = intrp.evaluate(main_gv)
    return f(*args).data.asnumpy()


def eval_cpu_graph_runtime(mod, params, input_dict):
    graph, op_mod, params = relay.build(mod['main'], target="llvm", params=params)
    if DEBUG_MODE:
        graph_mod = debug_runtime.create(graph, op_mod, tvm.cpu(0), dump_root='/home/lweber/microtvm-blogpost-eval/debug/cpu')
    else:
        graph_mod = graph_runtime.create(graph, op_mod, tvm.cpu(0))
    graph_mod.set_input(**params)
    graph_mod.run(**input_dict)
    return graph_mod.get_output(0).asnumpy()


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
            workload = gen_workload_desc_from_task(entry['i'])
            entry['i'][4] = workload
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


def reset_gdbinit(dev_config):
    if 'server_port' not in dev_config:
        return
    if 'MICRO_GDB_INIT_DIR' not in os.environ:
        print('WARNING: `MICRO_GDB_INIT_DIR` not set. GDB debugging will not be smooth, yo.')
        return
    gdb_init_dir = os.environ['MICRO_GDB_INIT_DIR']
    with open(f'{gdb_init_dir}/.gdbinit', 'w') as f:
        gdb_port = dev_config['server_port'] - 3333
        gdbinit_contents = (
f"""layout src
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
            eval "set $num_bits = ((TVMArray*) utvm_tasks[0].arg_values[0].v_handle)->dtype.bits"
            if $num_bits == 8
                print "dtype: int8"
                eval "p/d *((int8_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            if $num_bits == 32
                print "dtype: int32"
                eval "p/d *((int32_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
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


def benchmark_micro_func(sess, micro_func, args, num_trials, time_overhead):
    ctx = tvm.micro_dev(0)
    # sync before and after to ensure these are the only tasks in the queue
    ctx.sync()
    sess.get_last_batch_time()
    for _ in range(num_trials):
        micro_func(*args)
    ctx.sync()
    return (sess.get_last_batch_time() / num_trials) - time_overhead


class MockCMod:
    def __init__(self, src_path):
        self.src_path = src_path

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, self.src_path)


def check_conv2d_output(
        data_nt: NamedTensor, kernel_nt: NamedTensor, micro_output_nt: NamedTensor,
        strides, padding):
    data_nchw_np = data_nt.with_layout('NCHW').data
    kernel_oihw_np = kernel_nt.with_layout('OIHW').data
    micro_output_nchw_np = micro_output_nt.with_layout('NCHW').data

    topi_output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, strides, padding)
    tvm.testing.assert_allclose(micro_output_nchw_np, topi_output_np)


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
