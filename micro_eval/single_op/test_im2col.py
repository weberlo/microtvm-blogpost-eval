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

from micro_eval.util import show_c_source

N, H, W, CO, CI, KH, KW = 1, 5, 5, 1, 1, 3, 3
STRIDES, PADDING = (1, 1), 1
LAYOUT = 'NHWC'
DTYPE = 'int8'
DATA_SPEC = ('TENSOR', (N, H, W, CI), DTYPE)
# NOTE OIHW layout is better for im2col, because you can do transposed matmul
# via reshape (i.e., (CO, CI * KH * KW))
KERNEL_SPEC = ('TENSOR', (CO, CI, KH, KW), DTYPE)

def main():
    sched, arg_bufs = conv2d_arm_micro_nhwc_im2col(DATA_SPEC, KERNEL_SPEC, STRIDES, PADDING, LAYOUT, DTYPE)
    show_c_source(sched, arg_bufs)
    target = 'llvm'
    ctx = tvm.context(target, 0)
    func = tvm.build(sched, arg_bufs, target=target, name='conv2d')

    data_shape, kernel_shape, output_shape = tuple(map(lambda x: get_const_tuple(x.shape), arg_bufs))
    data_tvm = tvm.nd.array(np.random.randint(-30, 30, size=data_shape, dtype=DTYPE), ctx)
    kernel_tvm = tvm.nd.array(np.random.randint(-2, 2, size=kernel_shape, dtype=DTYPE), ctx)
    output_tvm = tvm.nd.array(np.zeros(output_shape, dtype=DTYPE), ctx)
    func(data_tvm, kernel_tvm, output_tvm)

    check_conv2d_output(data_tvm.asnumpy(), kernel_tvm.asnumpy(), output_tvm.asnumpy(), 'NHWC', 'OIHW', STRIDES, PADDING)
    print('All good!')


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
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    IM2COL_BATCH_SIZE = 1
    NUM_IM2COL_BATCHES = (HO * WO) // IM2COL_BATCH_SIZE

    _, PDH, PDW, _ = padded_data.shape
    K = CI * KH * KW

    im2col_data = tvm.compute(
            (N, HO, WO, CI, KH, KW),
            lambda nn, yy, xx, cc, ky, kx:
                padded_data[nn, yy + ky, xx + kx, cc],
            name='im2col_data')
    reshaped_im2col_data = topi.transform.reshape(
            im2col_data,
            (N, NUM_IM2COL_BATCHES, IM2COL_BATCH_SIZE, K))

    reshaped_kernel = topi.transform.reshape(kernel, (CO, K))

    k = tvm.reduce_axis((0, K), 'k')
    conv = tvm.compute(
            (N, NUM_IM2COL_BATCHES, IM2COL_BATCH_SIZE, CO),
            lambda nn, i2c_batch, i2c_batch_idx, cc:
                tvm.sum(
                    reshaped_im2col_data[nn, i2c_batch, i2c_batch_idx, k] * reshaped_kernel[cc, k],
                    axis=k),
            name='conv2d',
            tag='conv2d_nhwc')
    reshaped_conv = topi.transform.reshape(conv, (N, HO, WO, CO))

    arg_bufs = [data, kernel, reshaped_conv]
    sched = tvm.create_schedule(reshaped_conv.op)

    # GOAL: change schedules so im2col_data can be computed within each batch iter of `conv`
    #   sched[im2col_data].compute_at(sched[conv], conv.op.axis[???])
    assert False, 'look at goal above'

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


def check_conv2d_output(data_np, kernel_np, output_np, data_layout, kernel_layout, strides, padding):
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
    output_nchw_np = output_np.transpose(
            data_layout.index('N'),
            data_layout.index('C'),
            data_layout.index('H'),
            data_layout.index('W'))

    topi_output_np = conv2d_nchw_python(data_nchw_np, kernel_oihw_np, strides, padding)
    tvm.testing.assert_allclose(output_nchw_np, topi_output_np)


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
    output_np = output_np.reshape((H, W, CO))
    print('==========')
    print('= OUTPUT =')
    print('==========')
    print(output_np)
    print(output_np.shape)
    check_conv2d_output(
            data_np.reshape((1, H, W, CI)),
            kernel_np,
            output_np.reshape((1, H, W, CO)),
            'NHWC',
            'OIHW',
            STRIDES,
            PADDING
            )


if __name__ == "__main__":
    main()
