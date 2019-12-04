import json

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
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

# init autotvm env to register uTVM ops
TaskExtractEnv()

@autotvm.register_topi_compute(conv2d, 'micro_dev', ['direct'])
def conv2d_arm_micro_nchw(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)

    N, CI, H, W = get_const_tuple(data.shape)
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    #n, f, y, x = s[conv].op.axis
    #rc, ry, rx = s[conv].op.reduce_axis
    SH, SW = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    PH, PW, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'

    #data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    OH = (H - KH + 2 * PH) // SH + 1
    OW = (W - KW + 2 * PW) // SW + 1

    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    # TODO should we add a max_factor attr to these splits?
    co, vc = cfg.define_split("tile_co", co, num_outputs=2)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)

    #cfg.define_reorder("reorder_0",
    #        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
    #        policy='all')
    #cfg.define_reorder("reorder_0",
    #        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
    #        policy='candidate', candidate=[
    #            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
    #            [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
    #            [n, co, oh, ow, ci, vh, vw, vc, kh, kw],
    #            [n, co, oh, ow, ci, vc, vh, vw, kh, kw]])

    #cfg.define_knob("auto_unroll_max_step", [0, 256, 512, 1024])
    cfg.define_knob("auto_unroll_max_step", [0, 16, 32, 64, 128, 256])
    cfg.define_knob("unroll_explicit", [0, 1])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll')

    return conv


@autotvm.register_topi_schedule(schedule_conv2d_nchw, 'micro_dev', ['direct'])
def schedule_conv2d_arm_micro_nchw(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nchw" not in op.tag:
            return

        ### extract tensors ###
        output = op.output(0)
        conv = op
        #import pdb; pdb.set_trace()
        data_vec = conv.input_tensors[0]
        #data_pad = data_vec.op.input_tensors[0]
        data_pad = data_vec.op
        kernel = conv.input_tensors[1]
        last = outs[0]

        s[data_pad].compute_inline()

        # tile reduction axes
        n, co, oh, ow = s[conv].op.axis
        ci, kh, kw = s[conv].op.reduce_axis
        co, vc = cfg['tile_co'].apply(s, conv, co)
        oh, vh = cfg['tile_oh'].apply(s, conv, oh)
        ow, vw = cfg['tile_ow'].apply(s, conv, ow)

        #cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
        cfg["ann_reduce"].apply(s, conv, [kh, kw],
                                axis_lens=[get_const_int(kh.dom.extent),
                                           get_const_int(kw.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=8,
                                 cfg=cfg)
        #s[conv].compute_at(s[last], ow)

        kernel_scope = n  # this is the scope to attach global config inside this kernel

        # tune unroll
        s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
        s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _conv2d_arm_micro_nchw_template(data, kernel, strides, padding, dilation, layout, out_dtype):
    data_typ, data_shape, data_dtype = data
    kernel_typ, kernel_shape, kernel_dtype = kernel
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)
    cfg = autotvm.get_config()
    conv = conv2d_arm_micro_nchw(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    sched = schedule_conv2d_arm_micro_nchw(cfg, [conv])
    return sched, [data, kernel, conv]


@autotvm.template
def conv2d_arm_micro_nchw_template(*args, **kwargs):
    return _conv2d_arm_micro_nchw_template(*args, **kwargs)


@autotvm.task.register("topi_nn_conv2d", override=True)
def conv2d_arm_micro_nchw_topi_task(*args, **kwargs):
    return _conv2d_arm_micro_nchw_template(*args, **kwargs)


#def register_micro_dev_tuning_tasks():
#
#    autotvm.register_topi_compute(
#            conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#    autotvm.register_topi_schedule(
#            schedule_conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#
#    #autotvm.template(conv2d_arm_micro_nchw_template)
#    autotvm.task.register(conv2d_arm_micro_nchw_template, "topi_nn_conv2d", override=True)


def gen_cifar10_cnn(use_random_params=False):
    # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
    # TODO discrepancies between outputs might be a result of the bias_add op
    # not matching the semantics of the CMSIS bias add.
    mod = relay.fromtext("""
    v0.0.4
    def @main(%data: Tensor[(1, 3, 32, 32), uint8],
        %mean_data: Tensor[(1, 3, 32, 32), uint8],
        %conv0_weight: Tensor[(32, 3, 5, 5), int8],
        %conv0_bias: Tensor[(32), int8],
        %conv1_weight: Tensor[(32, 32, 5, 5), int8],
        %conv1_bias: Tensor[(32), int8],
        %conv2_weight: Tensor[(64, 32, 5, 5), int8],
        %conv2_bias: Tensor[(64), int8],
        %dense0_weight: Tensor[(10, 1024), int8],
        %dense0_bias: Tensor[(10), int8]) {
      %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
      %1 = nn.conv2d(%0, %conv0_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
      %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"));
      %3 = right_shift(%2, 9);
      %4 = cast(%3, "int8");
      %5 = nn.max_pool2d(%4, pool_size=[3, 3], strides=[2, 2], ceil_mode=True);
      %6 = nn.relu(%5);
      %7 = nn.conv2d(%6, %conv1_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
      %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"));
      %9 = right_shift(%8, 9);
      %10 = cast(%9, "int8");
      %11 = nn.relu(%10);
      %12 = nn.avg_pool2d(%11, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
      %13 = nn.conv2d(%12, %conv2_weight, padding=[2, 2], channels=64, kernel_size=[5, 5], out_dtype="int32");
      %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"));
      %15 = right_shift(%14, 9);
      %16 = cast(%15, "int8");
      %17 = nn.relu(%16);
      %18 = nn.avg_pool2d(%17, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
      %19 = nn.batch_flatten(%18);
      %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
      %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
      %22 = right_shift(%21, 5);
      cast(%22, "int8")
    }
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
            shape = list(map(lambda x: x.value, formal_param.checked_type.shape))
            dtype = formal_param.checked_type.dtype
            name = formal_param.name_hint
            # NCHW -> NHWC
            orig_np = np.array(params[name]).astype(dtype)
            print(name)
            print(orig_np.shape)
            if name == 'mean_data':
                # NCHW
                # N == 0
                # C == 1
                # H == 2
                # W == 3
                # NHWC (0, 2, 3, 1)

                # NHWC
                # N == 0
                # H == 1
                # W == 2
                # C == 3
                # NCHW (0, 3, 1, 2)
                orig_np = orig_np.reshape((shape[0], shape[2], shape[3], shape[1]))
                print(orig_np.shape)
                orig_np = orig_np.transpose(0, 3, 1, 2)
                print(orig_np.shape)
            elif name == 'conv0_weight':
                # CO, CI, KW, KH
                # CO == 0
                # CI == 1
                # KW == 2
                # KH == 3
                # CI, KW, KH, CO (1, 2, 3, 0)

                # CI, KW, KH, CO
                # CI == 0
                # KW == 1
                # KH == 2
                # CO == 3
                # CO, CI, KW, KH (3, 0, 1, 2)
                orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
                print(orig_np.shape)
                orig_np = orig_np.transpose(3, 0, 1, 2)
                print(orig_np.shape)
            elif name == 'conv0_bias':
                pass
            elif name == 'conv1_weight':
                orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
                print(orig_np.shape)
                orig_np = orig_np.transpose(3, 0, 1, 2)
                print(orig_np.shape)
            elif name == 'conv1_bias':
                pass
            elif name == 'conv2_weight':
                orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
                print(orig_np.shape)
                orig_np = orig_np.transpose(3, 0, 1, 2)
                print(orig_np.shape)
            elif name == 'conv2_bias':
                pass
            elif name == 'dense0_weight':
                orig_np = orig_np.reshape((shape[1], shape[0]))
                print(orig_np.shape)
                orig_np = orig_np.transpose(1, 0)
                print(orig_np.shape)
            elif name == 'dense0_bias':
                pass
            else:
                assert False
            params[name] = tvm.nd.array(orig_np, tvm.cpu(0))
    return mod, params


DEBUG_MODE = False

def relay_micro_build(func, dev_config, target, params=None):
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
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        with tvm.build_config(disable_vectorize=True):
            graph, c_mod, params = relay.build(func, target=target, params=params)
    print(c_mod.get_source())
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
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
