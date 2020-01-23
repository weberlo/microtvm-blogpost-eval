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

from topi.testing import conv2d_nchw_python
from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw, schedule_conv2d_nhwc
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

from micro_eval.util import (
        show_c_source,
        transform_data_layout,
        relay_micro_build, reset_gdbinit,
        intrin_gemm_MxKxN, gemm_MxKxN_impl,
        get_comm_overhead, benchmark_micro_func)

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
N, H, W, CO, CI, KH, KW = 1, 8, 8, 4, 4, 3, 3
STRIDES, PADDING = (1, 1), 1
LAYOUT = 'NHWC'
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
DATA_SPEC = ('TENSOR', (N, H, W, CI), IN_DTYPE)
# NOTE OIHW layout is better for im2col, because you can do transposed matmul
# via reshape (i.e., (CO, CI * KH * KW))
KERNEL_SPEC = ('TENSOR', (CO, CI, KH, KW), IN_DTYPE)

def conv2d_arm_micro_nhwc_im2col(data, kernel, stride, padding, layout, out_dtype):
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

    IM2COL_BATCH_SIZE = 4
    assert ((HO.value * WO.value) % IM2COL_BATCH_SIZE) == 0, 'im2col batch size must be a factor of width x height'
    NUM_IM2COL_BATCHES = (HO * WO) // IM2COL_BATCH_SIZE
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
    #        (N, NUM_IM2COL_BATCHES, IM2COL_BATCH_SIZE, K))

    # yy = ((i2c_batch * IM2COL_BATCH_SIZE + i2c_batch_idx) // WO)
    # xx = ((i2c_batch * IM2COL_BATCH_SIZE + i2c_batch_idx) % WO)
    # cc = (kk // (KH * KW))
    # ky = ((kk % (KH * KW)) // KW)
    # kx = ((kk % (KH * KW)) % KW)

    # TODO simultaneously expand to int16 for SIMD stuff
    im2col_data = tvm.compute(
            (N, NUM_IM2COL_BATCHES, IM2COL_BATCH_SIZE, K),
            lambda nn, i2c_batch, i2c_batch_idx, kk:
                padded_data[
                    nn,
                    ((i2c_batch * IM2COL_BATCH_SIZE + i2c_batch_idx) // WO) + ((kk % (KH * KW)) // KW),
                    ((i2c_batch * IM2COL_BATCH_SIZE + i2c_batch_idx) % WO) + ((kk % (KH * KW)) % KW),
                    kk // (KH * KW)],
            name='im2col_data')

    reshaped_kernel = topi.transform.reshape(kernel, (CO, K))

    k = tvm.reduce_axis((0, K), 'k')
    conv = tvm.compute(
            (N, NUM_IM2COL_BATCHES, IM2COL_BATCH_SIZE, CO),
            lambda nn, i2c_batch, i2c_batch_idx, cc:
                tvm.sum(
                    im2col_data[nn, i2c_batch, i2c_batch_idx, k].astype(out_dtype) * reshaped_kernel[cc, k].astype(out_dtype),
                    axis=k),
            name='conv2d',
            tag='conv2d_nhwc')
    reshaped_conv = topi.transform.reshape(conv, (N, HO, WO, CO))

    # i2c_batch = (yy * WO + xx) // NUM_IM2COL_BATCHES
    # i2c_batch_idx = (yy * WO + xx) % IM2COL_BATCH_SIZE

    #conv = tvm.compute(
    #        (N, HO, WO, CO),
    #        lambda nn, yy, xx, cc:
    #            tvm.sum(
    #                im2col_data[
    #                    nn,
    #                    (yy * WO + xx) // NUM_IM2COL_BATCHES,
    #                    (yy * WO + xx) % IM2COL_BATCH_SIZE,
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
    sched[im2col_data].compute_at(sched[conv], conv.op.axis[1])

    #hw = sched[conv].fuse(ho, wo)
    #i2c_batch, i2c_batch_idx = sched[conv].split(hw, factor=IM2COL_BATCH_SIZE)

    # tile reduction axes
    n, i2c_batch, i2c_batch_idx, co = sched[conv].op.axis
    k, = sched[conv].op.reduce_axis
    sched[conv].reorder(n, i2c_batch, i2c_batch_idx, co, k)

    gemm = intrin_gemm_MxKxN(IM2COL_BATCH_SIZE, K, CO, data.dtype, conv.dtype)
    sched[conv].tensorize(i2c_batch_idx, gemm)
    sched[conv].pragma(n, 'import_c', gemm_MxKxN_impl(IM2COL_BATCH_SIZE, K, CO))

    return sched, arg_bufs


def im2col(data_np, kernel_shape, padding):
    H, W, CI = data_np.shape
    CO, _, KH, KW = kernel_shape
    PH = PW = padding
    OH = H - (KH - 1) + PH + PH
    OW = W - (KW - 1) + PW + PW

    result = []
    for y in range(KH // 2 - PH, H - KH // 2 + PH):
        for x in range(KW // 2 - PW, W - KW // 2 + PW):
            data_run = []
            for c in range(CI):
                for ky in range(-(KH // 2), (KH // 2) + 1):
                    for kx in range(-(KW // 2), (KW // 2) + 1):
                        yy = y + ky
                        xx = x + kx
                        if yy >= 0 and yy < H and xx >= 0 and xx < W:
                            data_run.append(data_np[yy, xx, c])
                        else:
                            assert ky >= -PH or ky < H + PH
                            assert kx >= -PW or kx < W + PW
                            data_run.append(0)
            result.append(data_run)
    return np.array(result)


def run_conv2d_oracle(data_np, kernel_np, data_layout, kernel_layout, strides, padding):
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

    output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, strides, padding)
    return transform_data_layout(output_np, 'NCHW', data_layout)


def test_py_impl():
    H, W, CI = 3, 3, 1
    CO, KH, KW = 1, 3, 3

    data_np = np.random.randint(-30, 30, size=(H, W, CI), dtype=np.int8)
    kernel_np = np.random.randint(-2, 2, size=(CO, CI, KH, KW), dtype=np.int8)
    print('========')
    print('= DATA =')
    print('========')
    print(data_np)
    print(data_np.shape)
    print()
    print('==========')
    print('= KERNEL =')
    print('==========')
    print(kernel_np)
    print(kernel_np.shape)
    print()
    im2col_data_np = im2col(data_np, kernel_np.shape, PADDING)
    print('==========')
    print('= IM2COL =')
    print('==========')
    print(im2col_data_np)
    print(im2col_data_np.shape)
    print()
    print('==========')
    print('= MATMUL =')
    print('==========')
    print(im2col_data_np.shape, kernel_np.reshape((CO, CI * KH * KW)).T.shape)
    print()
    output_np = np.matmul(im2col_data_np, kernel_np.reshape((CO, CI * KH * KW)).T)
    output_np = output_np.reshape((1, H, W, CO))
    print('==========')
    print('= OUTPUT =')
    print('==========')
    print(output_np)
    print(output_np.shape)
    exp_output_np = run_conv2d_oracle(
            data_np.reshape((1, H, W, CI)),
            kernel_np,
            'NHWC',
            'OIHW',
            STRIDES,
            PADDING
            )
    tvm.testing.assert_allclose(output_np, exp_output_np)


def eval_micro(sess, sched, arg_bufs):
    [data, kernel, output] = arg_bufs
    c_mod = tvm.build(sched, [data, kernel, output], target=TARGET, name='conv2d')
    input(c_mod.get_source())

    from topi.util import get_const_tuple
    data_np = np.random.randint(-30, 30, size=get_const_tuple(data.shape)).astype(data.dtype)
    kernel_np = np.random.randint(-2, 2, size=get_const_tuple(kernel.shape)).astype(kernel.dtype)

    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['conv2d']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    output_tvm = tvm.nd.array(np.zeros(get_const_tuple(output.shape), dtype=output.dtype), ctx)

    batch_time, _ = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, output_tvm], 1)

    data_post_np = data_tvm.asnumpy()
    kernel_post_np = kernel_tvm.asnumpy()
    output_np = output_tvm.asnumpy()
    tvm.testing.assert_allclose(data_np, data_post_np, rtol=1e-3)
    tvm.testing.assert_allclose(kernel_np, kernel_post_np, rtol=1e-3)
    assert np.sum(output_np) != 0

    print('verifying output')
    exp_output_np = run_conv2d_oracle(data_np, kernel_np, 'NHWC', 'OIHW', STRIDES, PADDING)
    tvm.testing.assert_allclose(output_np, exp_output_np)

    return batch_time


def main():
    reset_gdbinit(DEV_CONFIG)

    #time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)
    time_overhead, cycle_overhead = 0.0, 0

    with micro.Session(DEV_CONFIG) as sess:
        sched, arg_bufs = conv2d_arm_micro_nhwc_im2col(DATA_SPEC, KERNEL_SPEC, STRIDES, PADDING, LAYOUT, OUT_DTYPE)
        time = eval_micro(sess, sched, arg_bufs)
        time -= time_overhead

    print(f'exec time: {time}')


if __name__ == "__main__":
    main()
