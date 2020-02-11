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
    CMSIS_INCLUDE_PATHS
    show_c_source,
    transform_data_layout,
    get_cmsis_include_paths,
    relay_micro_build, reset_gdbinit,
    get_comm_overhead, benchmark_micro_func)
from micro_eval.micro_topi.cortex_m7.micro_kernel.gemm import (
    intrin_gemm_MxKxN, gemm_MxKxN_impl,
)

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


def py_im2col(data_np, kernel_shape, padding):
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


def main():
    H, W, CI = 3, 3, 1
    CO, KH, KW = 1, 3, 3

    data_np = np.random.randint(-30, 30, size=(H, W, CI), dtype=np.int8)
    kernel_np = np.random.randint(-2, 2, size=(CO, CI, KH, KW), dtype=np.int8)
    print('########')
    print('# DATA #')
    print('########')
    print(data_np)
    print(data_np.shape)
    print()
    print('##########')
    print('# KERNEL #')
    print('##########')
    print(kernel_np)
    print(kernel_np.shape)
    print()
    im2col_data_np = py_im2col(data_np, kernel_np.shape, PADDING)
    print('##########')
    print('# IM2COL #')
    print('##########')
    print(im2col_data_np)
    print(im2col_data_np.shape)
    print()
    print('##########')
    print('# MATMUL #')
    print('##########')
    print(im2col_data_np.shape, kernel_np.reshape((CO, CI * KH * KW)).T.shape)
    print()
    output_np = np.matmul(im2col_data_np, kernel_np.reshape((CO, CI * KH * KW)).T)
    output_np = output_np.reshape((1, H, W, CO))
    print('##########')
    print('# OUTPUT #')
    print('##########')
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


if __name__ == "__main__":
    main()
