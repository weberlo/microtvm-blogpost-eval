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
from tvm.autotvm.task.dispatcher import FallbackContext
from tvm.autotvm.task.space import (
    FallbackConfigEntity, ReorderEntity, SplitEntity, OtherOptionEntity
)

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw, conv2d_nhwc
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from micro_eval.util import (
    CMSIS_INCLUDE_PATHS,
    NamedShape,
    transform_data_layout,
    print_c_source,
    relay_micro_build, reset_gdbinit,
    get_comm_overhead, benchmark_micro_func,
    check_conv2d_output
)
from micro_eval.micro_topi.cortex_m7.micro_kernel.gemm import (
        intrin_gemm_MxKxN, gemm_MxKxN_impl,
)
from micro_eval.micro_topi.cortex_m7.conv2d.direct import conv2d_direct
from micro_eval.micro_topi.cortex_m7.conv2d.direct_simd import conv2d_direct_simd
from micro_eval.micro_topi.cortex_m7.conv2d.partial_im2col import conv2d_partial_im2col

from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

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

# TIME_OVERHEAD, CYCLE_OVERHEAD = get_comm_overhead(DEV_CONFIG)
TIME_OVERHEAD, CYCLE_OVERHEAD = 0.0, 0

###############
# CONV CONFIG #
###############
#N, H, W, CO, CI = 1, 16, 16, 32, 32
N, H, W, CO, CI = 1, 4, 4, 4, 4
KH, KW = 3, 3
STRIDES, PADDING, DILATION = (1, 1), 1, 1
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

##############
# EVALUATION #
##############

def eval_micro(sess, sched, arg_bufs, data_layout, kernel_layout):
    [data, kernel, conv] = arg_bufs
    c_mod = tvm.build(sched, arg_bufs, target=TARGET, name='conv2d')

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
    check_conv2d_output(data_np, kernel_np, micro_output_np, data_layout, kernel_layout, STRIDES, PADDING)

    return batch_time


def eval_direct(sess):
    results = []
    for data_layout, kernel_layout in [('NCHW', 'OIHW'), ('NHWC', 'HWIO')]:
        sched, arg_bufs = conv2d_direct(
                FallbackConfigEntity(),
                (DATA_SHAPE, data_layout), (KERNEL_SHAPE, kernel_layout),
                STRIDES, PADDING, DILATION, OUT_DTYPE)
        time = eval_micro(sess, sched, arg_bufs, data_layout, kernel_layout)
        time -= TIME_OVERHEAD
        results.append(((data_layout, kernel_layout), time))
    return results


def eval_direct_simd(sess):
    results = []
    def gen_direct_simd_cfg(M, K, N):
        cfg = FallbackConfigEntity()
        cfg.template_key = 'direct_simd'
        cfg.is_fallback = False
        cfg['tile_ow'] = SplitEntity([-1, M])
        cfg['tile_ci'] = SplitEntity([-1, K])
        cfg['tile_co'] = SplitEntity([-1, N])
        # TODO we shouldn't need to mirror the order of the axes
        # specified in the config space definition to mock a reordering
        # here
        reorder_base = ['n', 'oh', 'owo', 'owi', 'coo', 'coi', 'kh', 'kw', 'cio', 'cii']
        reorder_target = ['n', 'oh', 'kh', 'kw', 'owo', 'coo', 'cio', 'owi', 'coi', 'cii']
        cfg['reorder_0_simd'] = ReorderEntity(
            [reorder_base.index(axis) for axis in reorder_target])
        cfg['auto_unroll_max_step'] = OtherOptionEntity(0)
        cfg['unroll_explicit'] = OtherOptionEntity(0)
        return cfg

    for microkernel_shape in [(1, 4, 1), (2, 4, 2), (4, 4, 4)]:
        sched, arg_bufs = conv2d_direct_simd(
            gen_direct_simd_cfg(*microkernel_shape),
            (DATA_SHAPE, 'NHWC'), (KERNEL_SHAPE, 'HWOI'),
            STRIDES, PADDING, DILATION, OUT_DTYPE)
        time = eval_micro(sess, sched, arg_bufs, 'NHWC', 'HWOI')
        time -= TIME_OVERHEAD
        results.append((microkernel_shape, time))
    return results


def eval_partial_im2col_simd(sess):
    results = []
    max_batch_size = H * W
    # try factors of the max batch size
    batch_sizes = [i for i in range(1, max_batch_size+1) if max_batch_size % i == 0]
    if len(batch_sizes) > NUM_BATCH_SIZE_CANDIDATES:
        batch_sizes = batch_sizes[-NUM_BATCH_SIZE_CANDIDATES:]
    for i2c_batch_size in batch_sizes:
        sched, arg_bufs = conv2d_partial_im2col(
                (DATA_SHAPE, 'NHWC'), (KERNEL_SHAPE, 'OIHW'),
                STRIDES, PADDING, OUT_DTYPE,
                i2c_batch_size)
        print_c_source(sched, arg_bufs)
        time = eval_micro(sess, sched, arg_bufs, 'NHWC', 'OIHW')
        time -= TIME_OVERHEAD
        results.append((i2c_batch_size, time))
    return results


def main():
    reset_gdbinit(DEV_CONFIG)

    with micro.Session(DEV_CONFIG) as sess:
        # direct w/o SIMD
        default_results = eval_direct(sess)
        # direct w/ SIMD
        direct_simd_results = eval_direct_simd(sess)
        # partial im2col w/ SIMD
        partial_im2col_simd_results = eval_partial_im2col_simd(sess)

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
        for (batch_size, i2c_simd_time) in partial_im2col_simd_results:
            print(f'  batch of {batch_size} time: {i2c_simd_time}')
        print()


if __name__ == "__main__":
    main()
