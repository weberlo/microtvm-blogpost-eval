import tvm
import topi
from tvm import autotvm, relay
from tvm.relay import create_executor
from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.contrib import graph_runtime, util, download

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
    #data = tvm.placeholder((N, CI, H, W), name='data')
    #kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    #conv = conv2d_nchw(data, kernel, strides, padding, dilation, layout, out_dtype)
    #sched = schedule_conv2d_nchw(data, kernel, conv)
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
    cfg.define_reorder("reorder_0",
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            policy='candidate', candidate=[
                [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
                [n, co, oh, ow, ci, vh, vw, vc, kh, kw],
                [n, co, oh, ow, ci, vc, vh, vw, kh, kw]])

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

        cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
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


#@autotvm.task.register("topi_nn_conv2d", override=True)
#def conv2d_arm_micro_nchw_topi_task(*args, **kwargs):
#    return _conv2d_arm_micro_nchw_template(*args, **kwargs)


#def register_micro_dev_tuning_tasks():
#
#    autotvm.register_topi_compute(
#            conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#    autotvm.register_topi_schedule(
#            schedule_conv2d_arm_micro_nchw, 'micro_dev', ['direct'])
#
#    #autotvm.template(conv2d_arm_micro_nchw_template)
#    autotvm.task.register(conv2d_arm_micro_nchw_template, "topi_nn_conv2d", override=True)


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
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def eval_relay_intrp(mod, args):
    main_gv = relay.GlobalVar('main')
    mod = relay.Module({main_gv: mod['main']})
    intrp = create_executor("debug", mod)
    f = intrp.evaluate(main_gv)
    return f(*args)


def eval_cpu_graph_runtime(mod, params, input_dict):
    graph, op_mod, params = relay.build(mod['main'], target="llvm", params=params)
    graph_mod = graph_runtime.create(graph, op_mod, tvm.cpu(0))
    graph_mod.set_input(**params)
    graph_mod.run(**input_dict)
    return graph_mod.get_output(0).asnumpy()
