import tvm
from topi.util import simplify, get_const_tuple, traverse_inline
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.nn.conv2d import conv2d_nchw, conv2d_nhwc

from micro_eval.micro_topi import (
    op_decl, register_compute, register_schedule
)

@op_decl(in_tensors=['data', 'kernel'])
def conv2d_direct(
        compute_func, schedule_func,
        cfg,
        data, kernel, strides, padding, dilation, out_dtype):
    # TODO (how should / should) we accomodate the code shape below? can we
    # architect it so autotvm templates and regular schedules fall into the
    # same structure?
    #   cfg = autotvm.get_config()
    #   conv = conv2d_arm_micro_nhwc(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    #   sched = schedule_conv2d_arm_micro_nhwc(cfg, [conv])
    #
    # could do `cfg=None` and `if cfg is None: cfg = autotvm.get_config()`
    #
    data, kernel, conv = compute_func(cfg, data, kernel, strides, padding, dilation, out_dtype)
    sched = schedule_func(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


@register_compute(conv2d_direct, data='NCHW', kernel='OIHW')
def conv2d_direct_nchw_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)

    cfg.define_knob('auto_unroll_max_step', [0, 2, 4, 8, 16, 32])
    cfg.define_knob('unroll_explicit', [0, 1])

    return data, kernel, conv


@register_schedule(conv2d_direct, data='NCHW', kernel='OIHW')
def conv2d_direct_nchw_schedule(cfg, outs):
    # use default schedule
    sched = tvm.create_schedule([x.op for x in outs])

    conv = outs[-1].op
    output = conv.output(0)
    data_vec = conv.input_tensors[0]
    data_pad = data_vec.op
    sched[data_pad].compute_inline()

    # TODO add more schedule opts (similar to the NHWC template)

    n, _, _, _ = sched[conv].op.axis
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    # tune unroll
    sched[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    sched[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    return sched


@register_compute(conv2d_direct, data='NHWC', kernel='HWIO')
def conv2d_direct_nhwc_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    conv = conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)

    N, H, W, CI = get_const_tuple(data.shape)
    KH, KW, _, CO = get_const_tuple(kernel.shape)
    n, oh, ow, co = cfg.axis(N), cfg.axis(H), cfg.axis(W), cfg.axis(CO)
    kh, kw, ci = cfg.reduce_axis(KH), cfg.reduce_axis(KW), cfg.reduce_axis(CI)

    # TODO should we add a max_factor attr to these splits?
    co, vc = cfg.define_split('tile_co', co, num_outputs=2)
    oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
    ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)

    cfg.define_reorder('reorder_0',
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            policy='candidate', candidate=[
                [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
                [n, co, oh, ow, ci, vh, vw, vc, kh, kw],
                [n, co, oh, ow, ci, vc, vh, vw, kh, kw]])

    cfg.define_annotate('ann_reduce', [kh, kw], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [vh, vw, vc], policy='try_unroll')

    cfg.define_knob('auto_unroll_max_step', [0, 2, 4, 8, 16, 32])
    cfg.define_knob('unroll_explicit', [0, 1])

    return data, kernel, conv


@register_schedule(conv2d_direct, data='NHWC', kernel='HWIO')
def conv2d_direct_nhwc_schedule(cfg, outs):
    sched = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_nhwc' not in op.tag:
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
        # NOTE we can't inline data padding in the SIMD path, because it
        # introduces conditionals in the inner loop.
        data_pad = data_vec.op
        sched[data_pad].compute_inline()

        co, vc = cfg['tile_co'].apply(sched, conv, co)
        oh, vh = cfg['tile_oh'].apply(sched, conv, oh)
        ow, vw = cfg['tile_ow'].apply(sched, conv, ow)
        cfg['reorder_0'].apply(sched, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
        cfg['ann_reduce'].apply(sched, conv, [kh, kw],
                                axis_lens=[get_const_int(kh.dom.extent),
                                           get_const_int(kw.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg['ann_spatial'].apply(sched, conv, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=8,
                                 cfg=cfg)

        kernel_scope = n  # this is the scope to attach global config inside this kernel

        # tune unroll
        sched[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
        sched[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(sched, outs[0].op, _callback)
    return sched


# def default_conv2d(data_spec, kernel_spec, data_layout, strides, padding, dilation, out_dtype):
#     data_typ, data_shape, data_dtype = data_spec
#     kernel_typ, kernel_shape, kernel_dtype = kernel_spec
#     assert data_typ == 'TENSOR'
#     assert kernel_typ == 'TENSOR'
#     data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
#     kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

#     if data_layout == 'NCHW':
#     elif data_layout == 'NHWC':
#         conv = conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)
#     else:
#         assert False

#     sched = tvm.create_schedule([conv.op])
#     return sched, [data, kernel, conv]
