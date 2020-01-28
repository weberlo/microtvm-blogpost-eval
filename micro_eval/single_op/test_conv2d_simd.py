# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
from collections import OrderedDict
from enum import Enum

import numpy as np
import topi
import tvm
from tvm.contrib import graph_runtime, util
from tvm import autotvm, relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.relay.testing import resnet
from tvm.relay import transform
from tvm.relay import create_executor

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw, conv2d_nhwc
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline
from topi.testing import conv2d_nchw_python, conv2d_nhwc_python

from micro_eval.util import (
        NamedShape,
        transform_data_layout,
        print_c_source,
        relay_micro_build, reset_gdbinit,
        intrin_gemm_MxKxN, gemm_MxKxN_impl,
        get_comm_overhead, benchmark_micro_func)

from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

################
# CMSIS CONFIG #
################
if 'CMSIS_PATH' not in os.environ:
    raise RuntimeError('must have "CMSIS_PATH" in environment')
CMSIS_PATH = os.environ['CMSIS_PATH']

CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include'
]

###############
# µTVM CONFIG #
###############
#DEV_CONFIG = tvm.micro.device.host.default_config()
DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6666)
DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
    ('text', (14000, MemConstraint.ABSOLUTE_BYTES)),
    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
    #('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('bss', (1024, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (8096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (50.0, MemConstraint.WEIGHT)),
    ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    ]))

TARGET = tvm.target.create('c -device=micro_dev')

###############
# CONV CONFIG #
###############
#N, H, W, CO, CI = 1, 16, 16, 32, 32
N, H, W, CO, CI = 1, 16, 16, 4, 4
KH, KW = 5, 5
#STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
STRIDES, PADDING, DILATION = (1, 1), 2, 1
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'

DATA_SHAPE = NamedShape(IN_DTYPE, N=N, C=CI, H=H, W=W)
KERNEL_SHAPE = NamedShape(IN_DTYPE, H=KH, W=KW, I=CI, O=CO)
OUTPUT_SHAPE = NamedShape(IN_DTYPE, N=N, C=CO, H=H, W=W)
BIAS_SHAPE = (CO,)

################
# TRIAL CONFIG #
################
NUM_TRIALS = 15
NUM_BATCH_SIZE_CANDIDATES = 5

#########################
# COMPUTE/SCHEDULE DEFS #
#########################

def default_conv2d(data_spec, kernel_spec, data_layout, strides, padding, dilation, out_dtype):
    data_typ, data_shape, data_dtype = data_spec
    kernel_typ, kernel_shape, kernel_dtype = kernel_spec
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    if data_layout == 'NCHW':
        conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    elif data_layout == 'NHWC':
        conv = conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)
    else:
        assert False

    sched = tvm.create_schedule([conv.op])
    return sched, [data, kernel, conv]


def conv2d_arm_micro_nhwc_direct(Input, Filter, stride, padding, dilation, out_dtype):
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, num_filter, channel = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: tvm.sum(
            PaddedInput[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[ry, rx, ff, rc].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output


def conv2d_arm_micro_nhwc_template(data_spec, kernel_spec, strides, padding, dilation, out_dtype, tens_config):
    data_typ, data_shape, data_dtype = data_spec
    kernel_typ, kernel_shape, kernel_dtype = kernel_spec
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    #conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    conv = conv2d_arm_micro_nhwc_direct(data, kernel, strides, padding, dilation, out_dtype)

    sched = tvm.create_schedule([conv.op])

    #data_vec = conv.op.input_tensors[0]
    #data_pad = data_vec.op
    #sched[data_pad].compute_inline()

    # assign axes of the default schedule to variables
    n, oh, ow, co = sched[conv].op.axis
    kh, kw, ci = sched[conv].op.reduce_axis

    #sched[conv].reorder(n, oh, kh, kw, ow, co, ci)
    if tens_config is None:
        return sched, [data, kernel, conv]
    else:
        (M, K, N) = tens_config

    owo, owi = sched[conv].split(ow, factor=M)
    cio, cii = sched[conv].split(ci, factor=K)
    coo, coi = sched[conv].split(co, factor=N)
    sched[conv].reorder(n, oh, kh, kw, owo, coo, cio, owi, coi, cii)

    gemm = intrin_gemm_MxKxN(M, K, N, data.dtype, conv.dtype)
    sched[conv].tensorize(owi, gemm)
    sched[conv].pragma(n, "import_c", gemm_MxKxN_impl(M, K, N))

    return sched, [data, kernel, conv]


def conv2d_arm_micro_nhwc_partial_im2col(data, kernel, stride, padding, layout, out_dtype, im2col_batch_size):
    data_typ, data_shape, data_dtype = data
    kernel_typ, kernel_shape, kernel_dtype = kernel
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    assert isinstance(stride, int) or len(stride) == 2
    assert layout == 'NHWC'

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

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

    arg_bufs = [data, kernel, reshaped_conv]
    sched = tvm.create_schedule(reshaped_conv.op)

    ################################################################################

    # NOTE: If we try to compute the reshape inline, then tensorization fails.
    # NOTE it gets worse though. because now the nonreshaped conv is generated
    # in a workspace, before being copied over to the reshaped tensor.

    # there might also be a bug with tensorization though.
    #sched[reshaped_kernel].compute_inline()
    sched[padded_data].compute_inline()
    sched[im2col_data].compute_at(sched[conv], conv.op.axis[1])

    #hw = sched[conv].fuse(ho, wo)
    #i2c_batch, i2c_batch_idx = sched[conv].split(hw, factor=im2col_batch_size)

    # tile reduction axes
    n, i2c_batch, i2c_batch_idx, co = sched[conv].op.axis
    k, = sched[conv].op.reduce_axis
    sched[conv].reorder(n, i2c_batch, i2c_batch_idx, co, k)

    gemm = intrin_gemm_MxKxN(im2col_batch_size, K, CO, data.dtype, conv.dtype)
    sched[conv].tensorize(i2c_batch_idx, gemm)
    sched[conv].pragma(n, 'import_c', gemm_MxKxN_impl(im2col_batch_size, K, CO))

    return sched, arg_bufs


##############
# EVALUATION #
##############

def check_output(data_np, kernel_np, micro_output_np, data_layout, kernel_layout):
    data_nchw_np = transform_data_layout(data_np, data_layout, 'NCHW')
    kernel_oihw_np = transform_data_layout(kernel_np, kernel_layout, 'OIHW')
    micro_output_nchw_np = transform_data_layout(micro_output_np, data_layout, 'NCHW')

    topi_output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, STRIDES, PADDING)
    tvm.testing.assert_allclose(micro_output_nchw_np, topi_output_np)


def eval_micro(sess, sched, arg_bufs, data_layout, kernel_layout):
    [data, kernel, conv] = arg_bufs
    c_mod = tvm.build(sched, arg_bufs, target=TARGET, name='conv2d')

    from topi.util import get_const_tuple
    data_np = np.random.randint(-10, 10, size=get_const_tuple(data.shape), dtype=data.dtype)
    kernel_np = np.random.randint(-3, 3, size=get_const_tuple(kernel.shape), dtype=kernel.dtype)

    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['conv2d']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    output_tvm = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx=ctx)

    batch_time, _ = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, output_tvm], 10)

    micro_output_np = output_tvm.asnumpy()
    assert np.sum(micro_output_np) != 0

    print('checking result against topi oracle...')
    check_output(data_np, kernel_np, micro_output_np, data_layout, kernel_layout)

    return batch_time


def main():
    reset_gdbinit(DEV_CONFIG)

    #time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)
    time_overhead, cycle_overhead = 0.0, 0

    with micro.Session(DEV_CONFIG) as sess:
        # default schedules
        default_results = []
        for data_layout, kernel_layout in [('NCHW', 'OIHW'), ('NHWC', 'HWIO')]:
            sched, arg_bufs = default_conv2d(
                    DATA_SHAPE.get_spec(data_layout), KERNEL_SHAPE.get_spec(kernel_layout),
                    data_layout,
                    STRIDES, PADDING, DILATION, OUT_DTYPE)
            time = eval_micro(sess, sched, arg_bufs, data_layout, kernel_layout)
            time -= time_overhead
            default_results.append(((data_layout, kernel_layout), time))

        # SIMD on direct convolution
        direct_simd_results = []
        for microkernel_shape in [(1, 4, 1), (2, 4, 2), (4, 4, 4)]:
            sched, arg_bufs = conv2d_arm_micro_nhwc_template(
                    DATA_SHAPE.get_spec('NHWC'), KERNEL_SHAPE.get_spec('HWOI'),
                    STRIDES, PADDING, DILATION, OUT_DTYPE, microkernel_shape)
            time = eval_micro(sess, sched, arg_bufs, 'NHWC', 'HWOI')
            time -= time_overhead
            direct_simd_results.append((microkernel_shape, time))

        # SIMD on partial im2col convolution
        im2col_simd_results = []
        max_batch_size = H * W
        # try factors of the max batch size
        batch_sizes = [i for i in range(1, max_batch_size+1) if max_batch_size % i == 0]
        if len(batch_sizes) > NUM_BATCH_SIZE_CANDIDATES:
            batch_sizes = batch_sizes[-NUM_BATCH_SIZE_CANDIDATES:]
        for i2c_batch_size in batch_sizes:
            sched, arg_bufs = conv2d_arm_micro_nhwc_partial_im2col(
                    DATA_SHAPE.get_spec('NHWC'), KERNEL_SHAPE.get_spec('OIHW'),
                    STRIDES, PADDING, 'NHWC', OUT_DTYPE,
                    i2c_batch_size)
            print_c_source(sched, arg_bufs)
            time = eval_micro(sess, sched, arg_bufs, 'NHWC', 'OIHW')
            time -= time_overhead
            im2col_simd_results.append((i2c_batch_size, time))

        [default_nchw_time, default_nhwc_time] = default_results
        print()
        print('###########')
        print('# DEFAULT #')
        print('###########')
        print(f'  NCHW time: {default_nchw_time}')
        print(f'  NHWC time: {default_nhwc_time}')
        print()
        [small_direct_simd_time, medium_direct_simd_time, large_direct_simd_time] = direct_simd_results
        print('######################')
        print('# DIRECT CONV + SIMD #')
        print('######################')
        print(f'  small time: {small_direct_simd_time}')
        print(f'  medium time: {medium_direct_simd_time}')
        print(f'  large time: {large_direct_simd_time}')
        print()
        print('######################')
        print('# IM2COL CONV + SIMD #')
        print('######################')
        for (batch_size, i2c_simd_time) in im2col_simd_results:
            print(f'  batch of {batch_size} time: {i2c_simd_time}')
        print()


if __name__ == "__main__":
    main()
