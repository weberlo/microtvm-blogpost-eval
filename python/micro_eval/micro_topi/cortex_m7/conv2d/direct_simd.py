import tvm
from topi.util import simplify, get_const_tuple, traverse_inline
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple

from micro_eval.micro_topi import (
        op_decl, register_compute, register_schedule
)

@op_decl(in_tensors=['data', 'kernel'])
def conv2d_direct_simd(
        compute_func, schedule_func,
        cfg,
        data, kernel, strides, padding, dilation, out_dtype):
    # TODO (how should / should) we accomodate the code shape below? can we
    # architect it so autotvm templates and regular schedules fall into the
    # same structure?
    #   cfg = autotvm.get_config()
    #   conv = conv2d_arm_micro_nhwc(cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    #   sched = schedule_conv2d_arm_micro_nhwc(cfg, [conv])
    data, kernel, conv = compute_func(cfg, data, kernel, strides, padding, dilation, out_dtype)
    # sched = schedule_conv2d_arm_micro_nhwc(cfg, [conv])
    sched = schedule_func(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


@register_compute(conv2d_direct_simd, data='NHWC', kernel='HWOI')
def conv2d_direct_simd_nhwc_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

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
    padded_data = pad(data, pad_before, pad_after, name='padded_data')

    rc = tvm.reduce_axis((0, in_channels), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    conv = tvm.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: tvm.sum(
            padded_data[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            kernel[ry, rx, ff, rc].astype(out_dtype), axis=[ry, rx, rc]),
        name='conv2d', tag='conv2d_nhwc')

    ###########################
    # Config Space Definition #
    ###########################
    n, oh, ow, co = cfg.axis(batch_size.value), cfg.axis(out_height.value), cfg.axis(out_width.value), cfg.axis(out_channels.value)
    kh, kw, ci = cfg.reduce_axis(kernel_h.value), cfg.reduce_axis(kernel_w.value), cfg.reduce_axis(in_channels.value)

    assert in_channels.value % 4 == 0
    owo, owi = cfg.define_split('tile_ow', ow, policy='factors', num_outputs=2)
    cio, cii = cfg.define_split('tile_ci', ci, policy='factors', num_outputs=2, filter=lambda x: x.size[-1] % 4 == 0)
    coo, coi = cfg.define_split('tile_co', co, policy='factors', num_outputs=2)

    cfg.define_reorder('reorder_0_simd',
            [n, oh, owo, owi, coo, coi, kh, kw, cio, cii],
            policy='candidate', candidate=[
                [n, oh, kh, kw, owo, coo, cio, owi, coi, cii],
                [n, oh, kh, kw, coo, owo, cio, owi, coi, cii],
                [n, kh, kw, oh, owo, coo, cio, owi, coi, cii],
                [n, kh, kw, oh, coo, owo, cio, owi, coi, cii]])

    cfg.define_knob('auto_unroll_max_step', [0, 2, 4, 8, 16, 32])
    cfg.define_knob('unroll_explicit', [0, 1])

    return data, kernel, conv


@register_schedule(conv2d_direct_simd, data='NHWC', kernel='HWOI')
def conv2d_direct_simd_nhwc_schedule(cfg, outs):
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

        M = cfg['tile_ow'].size[-1]
        K = cfg['tile_ci'].size[-1]
        N = cfg['tile_co'].size[-1]

        owo, owi = cfg['tile_ow'].apply(sched, conv, ow)
        cio, cii = cfg['tile_ci'].apply(sched, conv, ci)
        coo, coi = cfg['tile_co'].apply(sched, conv, co)

        cfg['reorder_0_simd'].apply(sched, conv, [n, oh, owo, owi, coo, coi, kh, kw, cio, cii])

        gemm = intrin_gemm_MxKxN(M, K, N, data_vec.dtype, output.dtype)
        sched[output].tensorize(owi, gemm)
        sched[output].pragma(n, 'import_c', gemm_MxKxN_impl(M, K, N))
        # elif 'reorder_0' in cfg:
        #     # NOTE we can't inline data padding in the SIMD path, because it
        #     # introduces conditionals in the inner loop.
        #     data_pad = data_vec.op
        #     sched[data_pad].compute_inline()

        #     co, vc = cfg['tile_co'].apply(sched, conv, co)
        #     oh, vh = cfg['tile_oh'].apply(sched, conv, oh)
        #     ow, vw = cfg['tile_ow'].apply(sched, conv, ow)
        #     cfg['reorder_0'].apply(sched, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
        #     cfg['ann_reduce'].apply(sched, conv, [kh, kw],
        #                             axis_lens=[get_const_int(kh.dom.extent),
        #                                        get_const_int(kw.dom.extent)],
        #                             max_unroll=8,
        #                             cfg=cfg)
        #     cfg['ann_spatial'].apply(sched, conv, [vh, vw, vc],
        #                              axis_lens=[cfg['tile_oh'].size[-1],
        #                                         cfg['tile_ow'].size[-1],
        #                                         cfg['tile_co'].size[-1]],
        #                              max_unroll=8,
        #                              cfg=cfg)
        # else:
        #     assert False, 'no reordering found in config'

        kernel_scope = n  # this is the scope to attach global config inside this kernel

        # tune unroll
        sched[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
        sched[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(sched, outs[0].op, _callback)
    return sched


# @autotvm.template
# def conv2d_arm_micro_nhwc_template(*args, **kwargs):
#     return _conv2d_arm_micro_nhwc_template(*args, **kwargs)


# @autotvm.task.register('topi_nn_conv2d', override=True)
# def conv2d_arm_micro_nhwc_topi_task(*args, **kwargs):
#    return _conv2d_arm_micro_nhwc_template(*args, **kwargs)
