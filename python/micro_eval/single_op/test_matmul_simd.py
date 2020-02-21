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
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

from micro_eval.util import relay_micro_build, reset_gdbinit, intrin_gemm_MxKxN, gemm_MxKxN_impl, get_comm_overhead, benchmark_micro_func

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
    ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
    ]))

TARGET = tvm.target.create('c -device=micro_dev')

#################
# MATMUL CONFIG #
#################
#M, K, N = 16, 32, 16
M, K, N = 64, 64, 64
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
A_SHAPE = (M, K)
B_SHAPE = (N, K)
A_TENSOR = ('TENSOR', A_SHAPE, IN_DTYPE)
B_TENSOR = ('TENSOR', B_SHAPE, IN_DTYPE)

################
# TRIAL CONFIG #
################
NUM_TRIALS = 15

# NOTE this is transposed matmul
# NOTE only works for reduction axes that are multiples of 4
def matmul_arm_micro_template(A, B, out_dtype, tens_config):
    A_typ, A_shape, A_dtype = A
    B_typ, B_shape, B_dtype = B
    assert A_typ == 'TENSOR'
    assert B_typ == 'TENSOR'
    assert A_shape[1] == B_shape[1]
    M, K, N = A_shape[0], A_shape[1], B_shape[0]

    # Algorithm
    k = tvm.reduce_axis((0, A_shape[1]), 'k')
    A = tvm.placeholder(A_shape, name='A', dtype=A_dtype)
    B = tvm.placeholder(B_shape, name='B', dtype=B_dtype)
    C = tvm.compute(
               (M, N),
               lambda i, j: tvm.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k),
               name='C')

    # Default schedule
    sched = tvm.create_schedule(C.op)

    if tens_config is None:
        return sched, [A, B, C]
    else:
        (M, K, N) = tens_config

    x, y = C.op.axis
    z, = C.op.reduce_axis

    xo, xi = sched[C].split(x, factor=M)
    zo, zi = sched[C].split(z, factor=K)
    yo, yi = sched[C].split(y, factor=N)
    sched[C].reorder(xo, yo, zo, xi, yi, zi)

    gemm = intrin_gemm_MxKxN(M, K, N, A.dtype, C.dtype)
    sched[C].tensorize(xi, gemm)
    sched[C].pragma(xo, 'import_c', gemm_MxKxN_impl(M, K, N))

    return sched, [A, B, C]


def eval_micro(sess, sched, arg_bufs):
    [A, B, C] = arg_bufs
    c_mod = tvm.build(sched, [A, B, C], target=TARGET, name='gemm')

    from topi.util import get_const_tuple
    A_np = np.random.randint(-30, 30, size=get_const_tuple(A.shape)).astype(A.dtype)
    B_np = np.random.randint(-30, 30, size=get_const_tuple(B.shape)).astype(B.dtype)

    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['gemm']
    ctx = tvm.micro_dev(0)

    A_tvm = tvm.nd.array(A_np, ctx=ctx)
    B_tvm = tvm.nd.array(B_np, ctx=ctx)
    C_tvm = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)

    batch_time, _ = benchmark_micro_func(sess, micro_func, [A_tvm, B_tvm, C_tvm], 10)

    C_np = C_tvm.asnumpy()
    assert np.sum(C_np) != 0
    tvm.testing.assert_allclose(C_np, np.dot(A_np.astype(C.dtype), B_np.T.astype(C.dtype)), rtol=1e-3)
    return batch_time


def main():
    reset_gdbinit(DEV_CONFIG)

    time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)

    with micro.Session(DEV_CONFIG) as sess:
        default_sched, default_arg_bufs = matmul_arm_micro_template(A_TENSOR, B_TENSOR, OUT_DTYPE, None)
        default_time = eval_micro(sess, default_sched, default_arg_bufs)
        default_time -= time_overhead

        small_simd_sched, small_simd_arg_bufs = matmul_arm_micro_template(A_TENSOR, B_TENSOR, OUT_DTYPE, (1, 4, 1))
        small_simd_time = eval_micro(sess, small_simd_sched, small_simd_arg_bufs)
        small_simd_time -= time_overhead

        large_simd_sched, large_simd_arg_bufs = matmul_arm_micro_template(A_TENSOR, B_TENSOR, OUT_DTYPE, (2, 4, 2))
        large_simd_time = eval_micro(sess, large_simd_sched, large_simd_arg_bufs)
        large_simd_time -= time_overhead
        print()
        print(f'default time: {default_time}')
        print(f'small SIMD time: {small_simd_time}')
        print(f'large SIMD time: {large_simd_time}')


if __name__ == '__main__':
    main()
