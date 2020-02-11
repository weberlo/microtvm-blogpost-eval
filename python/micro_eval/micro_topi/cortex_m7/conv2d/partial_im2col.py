import tvm
import topi
from topi.util import simplify, get_const_tuple
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple

from micro_eval.util import get_op_output_shape
from micro_eval.micro_topi import (
        op_decl, register_compute, register_schedule
)
from micro_eval.micro_topi.cortex_m7.micro_kernel.gemm import (
        intrin_gemm_MxKxN, gemm_MxKxN_impl,
)

@op_decl(in_tensors=['data', 'kernel'])
def conv2d_partial_im2col(
        compute_func, schedule_func,
        data, kernel, strides, padding, out_dtype, im2col_batch_size):
    data, kernel, reshaped_conv = compute_func(
        data, kernel, strides, padding, out_dtype, im2col_batch_size)
    sched = schedule_func(data, kernel, reshaped_conv)
    return sched, [data, kernel, reshaped_conv]


# TODO can we phrase `im2col_batch_size` as an axis split in the schedule,
# rather than baking it into the compute?
@register_compute(conv2d_partial_im2col, data='NHWC', kernel='OIHW')
def _compute(data, kernel, strides, padding, out_dtype, im2col_batch_size):
    assert isinstance(strides, int) or len(strides) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    N, HI, WI, CI = data.shape
    CO, _, KH, KW = kernel.shape

    # compute the output shape
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (KH, KW))
    HO = simplify((HI - KH + pad_top + pad_down) // stride_h + 1)
    WO = simplify((WI - KW + pad_left + pad_right) // stride_w + 1)

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name='padded_data')

    _, PDH, PDW, _ = padded_data.shape

    assert ((HO.value * WO.value) % im2col_batch_size) == 0, 'im2col batch size must be a factor of width x height'
    num_im2col_batches = (HO * WO) // im2col_batch_size
    K = CI * KH * KW

    # TODO reshapes fuck everything, so we need to manually fold in reshape
    # into our index calculations. figure out how to fix that, because that is
    # *shit* functionality.

    #im2col_data = tvm.compute(
    #        (N, HO, WO, CI, KH, KW),
    #        lambda nn, yy, xx, cc, ky, kx:
    #            padded_data[nn, yy + ky, xx + kx, cc],
    #        name='im2col_data'
    #        )
    #reshaped_im2col_data = topi.transform.reshape(
    #        im2col_data,
    #        (N, num_im2col_batches, im2col_batch_size, K))

    # yy = ((i2c_batch * im2col_batch_size + i2c_batch_idx) // WO)
    # xx = ((i2c_batch * im2col_batch_size + i2c_batch_idx) % WO)
    # cc = (kk // (KH * KW))
    # ky = ((kk % (KH * KW)) // KW)
    # kx = ((kk % (KH * KW)) % KW)

    # TODO simultaneously expand to int16 for SIMD stuff
    im2col_data = tvm.compute(
            (N, num_im2col_batches, im2col_batch_size, K),
            lambda nn, i2c_batch, i2c_batch_idx, kk:
                padded_data[
                    nn,
                    ((i2c_batch * im2col_batch_size + i2c_batch_idx) // WO) + ((kk % (KH * KW)) // KW),
                    ((i2c_batch * im2col_batch_size + i2c_batch_idx) % WO) + ((kk % (KH * KW)) % KW),
                    kk // (KH * KW)],
            name='im2col_data')

    reshaped_kernel = topi.transform.reshape(kernel, (CO, K))

    k = tvm.reduce_axis((0, K), 'k')
    conv = tvm.compute(
            (N, num_im2col_batches, im2col_batch_size, CO),
            lambda nn, i2c_batch, i2c_batch_idx, cc:
                tvm.sum(
                    im2col_data[nn, i2c_batch, i2c_batch_idx, k].astype(out_dtype) * reshaped_kernel[cc, k].astype(out_dtype),
                    axis=k),
            name='conv2d',
            tag='conv2d_nhwc')
    reshaped_conv = topi.transform.reshape(conv, (N, HO, WO, CO))

    # i2c_batch = (yy * WO + xx) // num_im2col_batches
    # i2c_batch_idx = (yy * WO + xx) % im2col_batch_size

    #conv = tvm.compute(
    #        (N, HO, WO, CO),
    #        lambda nn, yy, xx, cc:
    #            tvm.sum(
    #                im2col_data[
    #                    nn,
    #                    (yy * WO + xx) // num_im2col_batches,
    #                    (yy * WO + xx) % im2col_batch_size,
    #                    k].astype(out_dtype) * reshaped_kernel[cc, k].astype(out_dtype),
    #                axis=k),
    #        name='conv2d',
    #        tag='conv2d_nhwc')

    return data, kernel, reshaped_conv


@register_schedule(conv2d_partial_im2col, data='NHWC', kernel='OIHW')
def _schedule(data, kernel, reshaped_conv):
    sched = tvm.create_schedule(reshaped_conv.op)

    # unpack intermediate ops
    conv = reshaped_conv.op.input_tensors[0]
    im2col_data = conv.op.input_tensors[0]
    padded_data = im2col_data.op.input_tensors[0]

    # NOTE: If we try to compute the conv or kernel reshape inline, then tensorization fails.
    # NOTE it gets worse though. because now the nonreshaped conv is generated
    # in a workspace, before being copied over to the reshaped tensor.

    #sched[reshaped_kernel].compute_inline()
    sched[padded_data].compute_inline()
    sched[im2col_data].compute_at(sched[conv], conv.op.axis[1])

    # tile reduction axes
    n, i2c_batch, i2c_batch_idx, co = sched[conv].op.axis
    k, = sched[conv].op.reduce_axis
    sched[conv].reorder(n, i2c_batch, i2c_batch_idx, co, k)

    _, _, im2col_batch_size, CO = get_op_output_shape(conv.op)
    _, _, _, K = get_op_output_shape(im2col_data.op)

    # tensorize inner matmul with SIMD microkernel
    gemm = intrin_gemm_MxKxN(im2col_batch_size, K, CO, data.dtype, conv.dtype)
    sched[conv].tensorize(i2c_batch_idx, gemm)
    sched[conv].pragma(n, 'import_c', gemm_MxKxN_impl(im2col_batch_size, K, CO))

    return sched

