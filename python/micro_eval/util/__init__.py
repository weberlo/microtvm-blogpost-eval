import json
import os

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
CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include'
]

class NamedShape:
    def __init__(self, *args, **shape_dict):
        if len(args) == 1:
            [dtype] = args
            assert isinstance(dtype, str)
            shape_iter = shape_dict.items()
            self.dtype = dtype
        elif len(args) == 2:
            [layout, shape] = args
            assert isinstance(layout, str)
            assert isinstance(shape, tuple)
            assert len(shape_dict) == 0, 'cannot provide both layout/shape args and layout/shape kwargs'
            shape_iter = zip(layout, shape)
        elif len(args) == 3:
            [layout, shape, dtype] = args
            assert isinstance(layout, str)
            assert isinstance(shape, tuple)
            assert isinstance(dtype, str)
            assert len(shape_dict) == 0, 'cannot provide both layout/shape/dtype args and layout/shape kwargs'
            shape_iter = zip(layout, shape)
            self.dtype = dtype
        elif len(shape_dict) != 0:
            shape_iter = shape_dict.items()
        else:
            assert False

        for dim_name, dim_size in shape_iter:
            assert len(dim_name) == 1, 'dimension names must be single characters'
            setattr(self, dim_name, dim_size)

    def get_shape(self, layout):
        shape = []
        for dim_name in layout:
            assert hasattr(self, dim_name)
            shape.append(getattr(self, dim_name))
        return tuple(shape)

    def get_spec(self, layout):
        """Create an AutoTVM style spec (e.g., `("TENSOR", (1, 2, 3), 'int8')`)."""
        assert hasattr(self, 'dtype'), 'must specify dtype to generate a spec'
        return ('TENSOR', self.get_shape(layout), self.dtype)


def transform_data_layout(data_np, from_layout, to_layout):
    indices = []
    for dim in to_layout:
        idx = from_layout.index(dim)
        assert idx != -1
        indices.append(idx)
    return data_np.transpose(tuple(indices))


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

def relay_micro_build(func, dev_config, target, params=None, lib_include_paths=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : TODO
        TODO

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device

    """
    #with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #    with tvm.build_config(disable_vectorize=True):
    #        graph, c_mod, params = relay.build(func, target=target, params=params)
    with relay.build_config(opt_level=3):
        graph, c_mod, params = relay.build(func, target=target, params=params)
    input(c_mod.get_source())
    micro_mod = micro.create_micro_mod(c_mod, dev_config, lib_include_paths=lib_include_paths)
    ctx = tvm.micro_dev(0)
    if DEBUG_MODE:
        mod = debug_runtime.create(graph, micro_mod, ctx, dump_root='/home/lweber/microtvm-blogpost-eval/debug/micro')
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


def tuplify(elem):
    """Recursively convert all lists in `elem` into tuples."""
    if isinstance(elem, list):
        return tuple(list(map(tuplify, elem)))
    else:
        return elem


def custom_pick_best(in_log_file_name, out_log_file_name, top_k=1):
    workload_to_best = {}
    with open(in_log_file_name, 'r') as f:
        for line in f:
            entry = json.loads(line)
            workload = gen_workload_desc_from_task(entry['i'])
            entry['i'][4] = workload
            hashable_workload = tuplify(workload)
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
    with open('/home/lweber/gdb-conf/.gdbinit', 'w') as f:
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


def get_comm_overhead(dev_config):
    """Get communication overhead by executing an empty kernel."""
    class EmptyCMod:
        def __init__(self):
            pass

        def export_library(self, out_obj_path, fcompile=None):
            assert fcompile is not None
            fcompile(out_obj_path, f'{os.path.dirname(__file__)}/../src/empty.c')

    with micro.Session(dev_config) as sess:
        micro_mod = create_micro_mod(EmptyCMod(), dev_config)
        micro_func = micro_mod['empty']
        ctx = tvm.micro_dev(0)
        ctx.sync()
        sess.get_last_batch_time()
        sess.get_last_batch_cycles()
        micro_func()
        ctx.sync()
        exec_time = sess.get_last_batch_time()
        exec_cycles = sess.get_last_batch_cycles()
        return exec_time, exec_cycles


def benchmark_micro_func(sess, micro_func, args, num_trials):
    ctx = tvm.micro_dev(0)
    # sync before and after to ensure these are the only tasks in the queue
    ctx.sync()
    sess.get_last_batch_time()
    sess.get_last_batch_cycles()
    for _ in range(num_trials):
        micro_func(*args)
    ctx.sync()
    return sess.get_last_batch_time(), sess.get_last_batch_cycles()


class MockCMod:
    def __init__(self, src_path):
        self.src_path = src_path

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, self.src_path)


def check_conv2d_output(data_np, kernel_np, micro_output_np, data_layout, kernel_layout, strides, padding):
    data_nchw_np = transform_data_layout(data_np, data_layout, 'NCHW')
    kernel_oihw_np = transform_data_layout(kernel_np, kernel_layout, 'OIHW')
    micro_output_nchw_np = transform_data_layout(micro_output_np, data_layout, 'NCHW')

    topi_output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, strides, padding)
    tvm.testing.assert_allclose(micro_output_nchw_np, topi_output_np)
