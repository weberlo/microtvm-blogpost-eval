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
    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (8096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (50.0, MemConstraint.WEIGHT)),
    ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    ]))

TARGET = tvm.target.create('c -device=micro_dev')

###############
# CONV CONFIG #
###############
N, H, W, CO, CI = 1, 16, 16, 32, 32
#N, H, W, CO, CI = 1, 4, 4, 4, 4
KH, KW = 5, 5
#STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
STRIDES, PADDING, DILATION = (1, 1), 2, 1
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
NCHW_DATA_SHAPE = (N, CI, H, W)
NHWC_DATA_SHAPE = (N, H, W, CI)
HWIO_KERNEL_SHAPE = (KH, KW, CI, CO)
HWOI_KERNEL_SHAPE = (KH, KW, CO, CI)
OIHW_KERNEL_SHAPE = (CO, CI, KH, KW)

NCHW_DATA_SPEC = ('TENSOR', NCHW_DATA_SHAPE, IN_DTYPE)
OIHW_KERNEL_SPEC = ('TENSOR', OIHW_KERNEL_SHAPE, IN_DTYPE)

NHWC_DATA_SPEC = ('TENSOR', NHWC_DATA_SHAPE, IN_DTYPE)
HWIO_KERNEL_SPEC = ('TENSOR', HWIO_KERNEL_SHAPE, IN_DTYPE)

HWOI_KERNEL_SPEC = ('TENSOR', HWOI_KERNEL_SHAPE, IN_DTYPE)

BIAS_SHAPE = (CO,)
NHWC_OUTPUT_SHAPE = (N, H, W, CO)
NCHW_OUTPUT_SHAPE = (N, CO, H, W)

################
# TRIAL CONFIG #
################
NUM_TRIALS = 15

#########################
# COMPUTE/SCHEDULE DEFS #
#########################
def conv2d_arm_micro_nhwc(Input, Filter, stride, padding, dilation, out_dtype):
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
    conv = conv2d_arm_micro_nhwc(data, kernel, strides, padding, dilation, out_dtype)

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


def default_conv2d_nchw(data_spec, kernel_spec, strides, padding, dilation, out_dtype):
    data_typ, data_shape, data_dtype = data_spec
    kernel_typ, kernel_shape, kernel_dtype = kernel_spec
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
    sched = tvm.create_schedule([conv.op])
    return sched, [data, kernel, conv]


def default_conv2d_nhwc(data_spec, kernel_spec, strides, padding, dilation, out_dtype):
    data_typ, data_shape, data_dtype = data_spec
    kernel_typ, kernel_shape, kernel_dtype = kernel_spec
    assert data_typ == 'TENSOR'
    assert kernel_typ == 'TENSOR'
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    conv = conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)
    sched = tvm.create_schedule([conv.op])
    return sched, [data, kernel, conv]


def check_output(data_np, kernel_np, micro_output_np, data_layout, kernel_layout):
    # convert the data layout to NCHW
    data_nchw_np = data_np.transpose(
            data_layout.index('N'),
            data_layout.index('C'),
            data_layout.index('H'),
            data_layout.index('W'))

    # convert the kernel layout to OIHW
    kernel_oihw_np = kernel_np.transpose(
            kernel_layout.index('O'),
            kernel_layout.index('I'),
            kernel_layout.index('H'),
            kernel_layout.index('W'))

    # convert the output layout to NCHW
    micro_output_nchw_np = micro_output_np.transpose(
            data_layout.index('N'),
            data_layout.index('C'),
            data_layout.index('H'),
            data_layout.index('W'))

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
        default_nchw_sched, default_nchw_arg_bufs = default_conv2d_nchw(
                NCHW_DATA_SPEC, OIHW_KERNEL_SPEC, STRIDES, PADDING, DILATION, OUT_DTYPE)
        default_nchw_time = eval_micro(sess, default_nchw_sched, default_nchw_arg_bufs, 'NCHW', 'OIHW')
        default_nchw_time -= time_overhead

        default_nhwc_sched, default_nhwc_arg_bufs = default_conv2d_nhwc(
                NHWC_DATA_SPEC, HWIO_KERNEL_SPEC, STRIDES, PADDING, DILATION, OUT_DTYPE)
        default_nhwc_time = eval_micro(sess, default_nhwc_sched, default_nhwc_arg_bufs, 'NHWC', 'HWIO')
        default_nhwc_time -= time_overhead

        small_simd_sched, small_simd_arg_bufs = conv2d_arm_micro_nhwc_template(
                NHWC_DATA_SPEC, HWOI_KERNEL_SPEC, STRIDES, PADDING, DILATION, OUT_DTYPE, (1, 4, 1))
        small_simd_time = eval_micro(sess, small_simd_sched, small_simd_arg_bufs, 'NHWC', 'HWOI')
        small_simd_time -= time_overhead

        medium_simd_sched, medium_simd_arg_bufs = conv2d_arm_micro_nhwc_template(
                NHWC_DATA_SPEC, HWOI_KERNEL_SPEC, STRIDES, PADDING, DILATION, OUT_DTYPE, (2, 4, 2))
        medium_simd_time = eval_micro(sess, medium_simd_sched, medium_simd_arg_bufs, 'NHWC', 'HWOI')
        medium_simd_time -= time_overhead

        large_simd_sched, large_simd_arg_bufs = conv2d_arm_micro_nhwc_template(
                NHWC_DATA_SPEC, HWOI_KERNEL_SPEC, STRIDES, PADDING, DILATION, OUT_DTYPE, (4, 4, 4))
        large_simd_time = eval_micro(sess, large_simd_sched, large_simd_arg_bufs, 'NHWC', 'HWOI')
        large_simd_time -= time_overhead

        print()
        print(f'default NCHW time: {default_nchw_time}')
        print(f'default NHWC time: {default_nhwc_time}')
        print(f'small SIMD time: {small_simd_time}')
        print(f'medium SIMD time: {medium_simd_time}')
        print(f'large SIMD time: {large_simd_time}')


if __name__ == "__main__":
    main()
