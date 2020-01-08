import json
import os

import numpy as np

import tvm
import topi
from tvm import autotvm, relay
from tvm.relay import create_executor
from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args
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

# init autotvm env to register uTVM ops
TaskExtractEnv()

def _conv2d_arm_micro_nhwc_compute(data, kernel, stride, padding, dilation, out_dtype):
    return conv


@autotvm.register_topi_compute(conv2d, 'micro_dev', ['direct'])
def conv2d_arm_micro_nhwc(cfg, data, kernel, stride, padding, dilation, layout, out_dtype):
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    assert layout == 'NHWC'

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_size, in_height, in_width, in_channels = data.shape
    kernel_h, kernel_w, out_channels, _ = kernel.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")
    rc = tvm.reduce_axis((0, in_channels), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    conv = tvm.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: tvm.sum(
            padded_data[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            kernel[ry, rx, ff, rc].astype(out_dtype), axis=[ry, rx, rc]),
        name="conv2d", tag="conv2d_nhwc")

    ###########################
    # Config Space Definition #
    ###########################
    #N, H, W, CI = get_const_tuple(data.shape)
    #KH, KW, CO, _  = get_const_tuple(kernel.shape)
    #n, f, y, x = s[conv].op.axis
    #rc, ry, rx = s[conv].op.reduce_axis
    #SH, SW = stride if isinstance(stride, (tuple, list)) else (strides, strides)
    #PH, PW, _, _ = get_pad_tuple(padding, kernel)

    #data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    #OH = (H - KH + 2 * PH) // SH + 1
    #OW = (W - KW + 2 * PW) // SW + 1

    n, oh, ow, co = cfg.axis(batch_size.value), cfg.axis(out_height.value), cfg.axis(out_width.value), cfg.axis(out_channels.value)
    kh, kw, ci = cfg.reduce_axis(kernel_h.value), cfg.reduce_axis(kernel_w.value), cfg.reduce_axis(in_channels.value)

    if in_channels.value % 4 == 0:
        owo, owi = cfg.define_split("tile_ow", ow, policy='factors', num_outputs=2)
        cio, cii = cfg.define_split("tile_ci", ci, policy='factors', num_outputs=2, filter=lambda x: x.size[-1] % 4 == 0)
        coo, coi = cfg.define_split("tile_co", co, policy='factors', num_outputs=2)

        cfg.define_reorder("reorder_0_simd",
                [n, oh, owo, owi, coo, coi, kh, kw, cio, cii],
                policy='candidate', candidate=[
                    [n, oh, kh, kw, owo, coo, cio, owi, coi, cii],
                    [n, oh, kh, kw, coo, owo, cio, owi, coi, cii],
                    [n, kh, kw, oh, owo, coo, cio, owi, coi, cii],
                    [n, kh, kw, oh, coo, owo, cio, owi, coi, cii]])
    else:
        # TODO should we add a max_factor attr to these splits?
        co, vc = cfg.define_split("tile_co", co, num_outputs=2)
        oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
        ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)

        #cfg.define_reorder("reorder_0",
        #        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        #        policy='all')
        cfg.define_reorder("reorder_0",
                [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                policy='candidate', candidate=[
                    [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                    [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
                    [n, co, oh, ow, ci, vh, vw, vc, kh, kw],
                    [n, co, oh, ow, ci, vc, vh, vw, kh, kw]])

        cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
        cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll')

    #cfg.define_knob("auto_unroll_max_step", [0, 256, 512, 1024])
    #cfg.define_knob("auto_unroll_max_step", [0, 16, 32, 64, 128, 256])
    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    cfg.define_knob("unroll_explicit", [0, 1])

    return conv


@autotvm.register_topi_schedule(schedule_conv2d_nhwc, 'micro_dev', ['direct'])
def schedule_conv2d_arm_micro_nhwc(cfg, outs):
    sched = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc" not in op.tag:
            return

        ### extract tensors ###
        output = op.output(0)
        conv = op
        data_vec = conv.input_tensors[0]
        kernel = conv.input_tensors[1]
        last = outs[0]

        # tile reduction axes
        n, oh, ow, co = sched[conv].op.axis
        kh, kw, ci = sched[conv].op.reduce_axis

        if 'reorder_0_simd' in cfg:
            M = cfg['tile_ow'].size[-1]
            K = cfg['tile_ci'].size[-1]
            N = cfg['tile_co'].size[-1]

            owo, owi = cfg['tile_ow'].apply(sched, conv, ow)
            cio, cii = cfg['tile_ci'].apply(sched, conv, ci)
            coo, coi = cfg['tile_co'].apply(sched, conv, co)

            cfg["reorder_0_simd"].apply(sched, conv, [n, oh, owo, owi, coo, coi, kh, kw, cio, cii])

            gemm = intrin_gemm_MxKxN(M, K, N, data_vec.dtype, output.dtype)
            sched[output].tensorize(owi, gemm)
            sched[output].pragma(n, "import_c", gemm_MxKxN_impl(M, K, N))
        elif 'reorder_0' in cfg:
            # we can't inline data padding in the SIMD path, because it
            # introduces conditionals in the inner loop.
            data_pad = data_vec.op
            sched[data_pad].compute_inline()

            co, vc = cfg['tile_co'].apply(sched, conv, co)
            oh, vh = cfg['tile_oh'].apply(sched, conv, oh)
            ow, vw = cfg['tile_ow'].apply(sched, conv, ow)
            cfg["reorder_0"].apply(sched, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
            cfg["ann_reduce"].apply(sched, conv, [kh, kw],
                                    axis_lens=[get_const_int(kh.dom.extent),
                                               get_const_int(kw.dom.extent)],
                                    max_unroll=8,
                                    cfg=cfg)
            cfg["ann_spatial"].apply(sched, conv, [vh, vw, vc],
                                     axis_lens=[cfg['tile_oh'].size[-1],
                                                cfg['tile_ow'].size[-1],
                                                cfg['tile_co'].size[-1]],
                                     max_unroll=8,
                                     cfg=cfg)
        else:
            assert False, "no reordering found in config"


        kernel_scope = n  # this is the scope to attach global config inside this kernel

        # tune unroll
        sched[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
        sched[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(sched, outs[0].op, _callback)
    return sched


def _conv2d_arm_micro_nhwc_template(data, kernel, strides, padding, dilation, layout, out_dtype):
    data_typ, data_shape, data_dtype = data
    kernel_typ, kernel_shape, kernel_dtype = kernel
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)
    cfg = autotvm.get_config()
    conv = conv2d_arm_micro_nhwc(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    sched = schedule_conv2d_arm_micro_nhwc(cfg, [conv])
    return sched, [data, kernel, conv]


@autotvm.template
def conv2d_arm_micro_nhwc_template(*args, **kwargs):
    return _conv2d_arm_micro_nhwc_template(*args, **kwargs)


@autotvm.task.register("topi_nn_conv2d", override=True)
def conv2d_arm_micro_nhwc_topi_task(*args, **kwargs):
    return _conv2d_arm_micro_nhwc_template(*args, **kwargs)


#def register_micro_dev_tuning_tasks():
#
#    autotvm.register_topi_compute(
#            conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#    autotvm.register_topi_schedule(
#            schedule_conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#
#    #autotvm.template(conv2d_arm_micro_nchw_template)
#    autotvm.task.register(conv2d_arm_micro_nchw_template, "topi_nn_conv2d", override=True)

def gen_conv2d(data_layout, kernel_layout):
    assert data_layout == "NHWC"
    assert kernel_layout == "HWOI"
    mod = relay.fromtext(f"""
    v0.0.4
    def @main(
        %data: Tensor[(1, 16, 16, 32), int8],
        %kernel: Tensor[(5, 5, 32, 32), int8]) {{
      %0 = nn.conv2d(
        %data,
        %kernel,
        padding=[2, 2],
        channels=32,
        kernel_size=[5, 5],
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype="int32");
      %1 = right_shift(%0, 9);
      cast(%1, "int8")
    }}
    """)

    # generate random params
    params = {}
    for param in mod['main'].params[1:]:
        shape = list(map(lambda x: x.value, param.checked_type.shape))
        dtype = param.checked_type.dtype
        if 'kernel' in param.name_hint:
            result = tvm.nd.array(np.random.randint(-30, 30, size=shape, dtype=dtype), tvm.cpu(0))
        else:
            assert False
        params[param.name_hint] = result

    return mod, params


class NamedShape:
    def __init__(self, *args, **shape_dict):
        if len(args) == 2:
            layout = args[0]
            shape = args[1]
            shape_iter = zip(layout, shape)
        elif len(shape_dict) != 0:
            shape_iter = shape_dict.items()
        else:
            assert False

        for dim_name, dim_size in shape_iter:
            setattr(self, dim_name, dim_size)

    def get_shape(self, layout):
        shape = []
        for dim_name in layout:
            assert hasattr(self, dim_name)
            shape.append(getattr(self, dim_name))
        return tuple(shape)


def transform_data_layout(data_np, from_layout, to_layout):
    indices = []
    for dim in to_layout:
        idx = from_layout.index(dim)
        assert idx != -1
        indices.append(idx)
    return data_np.transpose(tuple(indices))


def gen_cifar10_cnn(data_layout, kernel_layout, use_random_params=False):
    # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
    # TODO discrepancies between outputs might be a result of the bias_add op
    # not matching the semantics of the CMSIS bias add.
    data_shape = NamedShape(N=1, C=3, H=32, W=32).get_shape(data_layout)
    conv0_kernel_shape = NamedShape(O=32, I=3, H=5, W=5).get_shape(kernel_layout)
    conv1_kernel_shape = NamedShape(O=32, I=32, H=5, W=5).get_shape(kernel_layout)
    conv2_kernel_shape = NamedShape(O=64, I=32, H=5, W=5).get_shape(kernel_layout)
    bias_add_axis = data_layout.index('C')
    mod = relay.fromtext(f"""
    v0.0.4
    def @main(%data: Tensor[{data_shape}, uint8],
        %mean_data: Tensor[{data_shape}, uint8],
        %conv0_weight: Tensor[{conv0_kernel_shape}, int8],
        %conv0_bias: Tensor[(32), int8],
        %conv1_weight: Tensor[{conv1_kernel_shape}, int8],
        %conv1_bias: Tensor[(32), int8],
        %conv2_weight: Tensor[{conv2_kernel_shape}, int8],
        %conv2_bias: Tensor[(64), int8],
        %dense0_weight: Tensor[(10, 1024), int8],
        %dense0_bias: Tensor[(10), int8]) {{
      %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
      %1 = nn.conv2d(
             %0,
             %conv0_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layout}",
             out_dtype="int32");
      %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"), axis={bias_add_axis});
      %3 = right_shift(%2, 9);
      %4 = cast(%3, "int8");
      %5 = nn.max_pool2d(%4,
             pool_size=[3, 3],
             strides=[2, 2],
             layout="{data_layout}",
             ceil_mode=True);
      %6 = nn.relu(%5);
      %7 = nn.conv2d(
             %6,
             %conv1_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layout}",
             out_dtype="int32");
      %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"), axis={bias_add_axis});
      %9 = right_shift(%8, 9);
      %10 = cast(%9, "int8");
      %11 = nn.relu(%10);
      %12 = nn.avg_pool2d(%11,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %13 = nn.conv2d(%12,
              %conv2_weight,
              padding=[2, 2],
              channels=64,
              kernel_size=[5, 5],
              data_layout="{data_layout}",
              kernel_layout="{kernel_layout}",
              out_dtype="int32");
      %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"), axis={bias_add_axis});
      %15 = right_shift(%14, 9);
      %16 = cast(%15, "int8");
      %17 = nn.relu(%16);
      %18 = nn.avg_pool2d(%17,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %19 = nn.batch_flatten(%18);
      %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
      %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
      %22 = right_shift(%21, 5);
      cast(%22, "int8")
    }}
    """)

    if use_random_params:
        # generate random params
        params = {}
        for param in mod['main'].params[1:]:
            shape = list(map(lambda x: x.value, param.checked_type.shape))
            dtype = param.checked_type.dtype
            if 'bias' in param.name_hint:
                result = tvm.nd.array(np.random.randint(-3, 3, size=shape, dtype=dtype), tvm.cpu(0))
            elif 'weight' in param.name_hint:
                result = tvm.nd.array(np.random.randint(-30, 30, size=shape, dtype=dtype), tvm.cpu(0))
            elif 'mean' in param.name_hint:
                result = tvm.nd.array(np.random.randint(130, 140, size=shape, dtype=dtype), tvm.cpu(0))
            else:
                assert False
            params[param.name_hint] = result
    else:
        with open('cifar10_cnn_params.json', 'r') as f:
            params = json.load(f)
        for formal_param in mod['main'].params[1:]:
            param_shape = list(map(lambda x: x.value, formal_param.checked_type.shape))
            dtype = formal_param.checked_type.dtype
            name = formal_param.name_hint

            orig_np = np.array(params[name]).astype(dtype)

            if name == 'mean_data':
                shape = NamedShape(data_layout, param_shape)
                cmsis_data_layout = 'NHWC'
                cmsis_shape = shape.get_shape(cmsis_data_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_data_layout, data_layout)
            elif 'conv' in name and 'weight' in name:
                shape = NamedShape(kernel_layout, param_shape)
                cmsis_kernel_layout = 'IHWO'
                cmsis_shape = shape.get_shape(cmsis_kernel_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_kernel_layout, kernel_layout)
            elif 'dense' in name and 'weight' in name:
                dense_layout = 'OI'
                shape = NamedShape(dense_layout, param_shape)
                # TODO they might be doing matmul weight reordering (figure 6 in their paper)
                cmsis_dense_layout = 'IO'
                cmsis_shape = shape.get_shape(cmsis_dense_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_dense_layout, dense_layout)
            else:
                assert name in ['conv0_bias', 'conv1_bias', 'conv2_bias', 'dense0_bias']
                relay_np = orig_np
            params[name] = tvm.nd.array(relay_np, tvm.cpu(0))
    return mod, params


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


class EmptyCMod:
    def __init__(self):
        pass

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, f'{os.path.dirname(__file__)}/../src/empty.c')


def get_comm_overhead(dev_config):
    """Get communication overhead by executing an empty kernel."""
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


##########################
# MxKxN MatMul Intrinsic #
##########################

# NOTE this is transposed matmul (A * B^T)
def intrin_gemm_MxKxN(M, K, N, in_dtype, out_dtype):
    assert K % 4 == 0
    assert in_dtype == 'int8'
    assert out_dtype == 'int32'
    A = tvm.placeholder((M, K), name='a', dtype=in_dtype)
    B = tvm.placeholder((N, K), name='b', dtype=in_dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda i, j: tvm.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k), name='c')
    A_buf = tvm.decl_buffer(
            A.shape, A.dtype,
            name="A",
            offset_factor=1,
            strides=[tvm.var("A_s"), 1])
    B_buf = tvm.decl_buffer(
            B.shape, B.dtype,
            name="B",
            offset_factor=1,
            strides=[tvm.var("B_s"), 1])
    C_buf = tvm.decl_buffer(
            C.shape, C.dtype,
            name="C",
            offset_factor=1,
            strides=[tvm.var("C_s"), 1])
    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        def _body():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_update",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    cc.access_ptr("w"),
                                    aa.strides[0],
                                    bb.strides[0],
                                    cc.strides[0]))
            return ib.get()
        def _reduce_reset():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_reset",
                                    cc.access_ptr("w"),
                                    cc.strides[0]))
            return ib.get()
        def _reduce_update():
            return _body()
        return _body(), _reduce_reset(), _reduce_update()
    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: A_buf, B: B_buf, C: C_buf})


def gemm_MxKxN_impl(M, K, N):
    aa_pad_size = M * K
    bb_pad_size = N * K
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_update(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t aa_pad[{aa_pad_size}];
  int16_t bb_pad[{bb_pad_size}];

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*{K} + j*4], (int32_t*) &aa_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {N}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*{K} + j*4], (int32_t*) &bb_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K} / 2; l++) {{
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*{K} + l*2]),
          *((int32_t*) &bb_pad[j*{K} + l*2]),
          sum);
      }}
      cc[i*C_stride + j] += sum;
    }}
  }}

  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_reset(int32_t *cc, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      cc[i*C_stride + j] = 0;
    }}
  }}
  return 0;
}}
    """
    return cc_code

