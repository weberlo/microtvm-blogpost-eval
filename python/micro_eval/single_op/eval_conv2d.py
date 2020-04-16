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
from tvm.autotvm.task.space import ReorderEntity, SplitEntity, OtherOptionEntity

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw, conv2d_nhwc
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from micro_eval.util import (
    CMSIS_NN_PATH, CMSIS_INCLUDE_PATHS,
    MockCMod,
    NamedTensor, NamedType, BakedType,
    print_c_source,
    relay_micro_build, reset_gdbinit,
    get_comm_overhead, benchmark_micro_func,
    check_conv2d_output
)
from micro_eval.micro_topi.cortex_m7.micro_kernel.gemm import (
    intrin_gemm_MxKxN, gemm_MxKxN_impl
)
from micro_eval.micro_topi.cortex_m7.conv2d.direct import conv2d_direct
from micro_eval.micro_topi.cortex_m7.conv2d.direct_simd import conv2d_direct_simd
from micro_eval.micro_topi.cortex_m7.conv2d.partial_im2col import conv2d_partial_im2col
from micro_eval.micro_topi import ManualConfigContext, ManualConfigSpace

from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

###############
# µTVM CONFIG #
###############
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
    ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
    ]))

TARGET = tvm.target.create('c -device=micro_dev')

# TIME_OVERHEAD = get_comm_overhead(DEV_CONFIG)
TIME_OVERHEAD = 0.0

###############
# CONV CONFIG #
###############
N, H, W, CO, CI = 1, 16, 16, 32, 32
KH, KW = 5, 5
STRIDES, PADDING, DILATION = (1, 1), 2, 1
# N, H, W, CO, CI = 1, 8, 8, 4, 4
# KH, KW = 3, 3
# STRIDES, PADDING, DILATION = (1, 1), 1, 1
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'

DATA_TYPE = NamedType(dict(N=N, C=CI, H=H, W=W), dtype=IN_DTYPE)
KERNEL_TYPE = NamedType(dict(H=KH, W=KW, I=CI, O=CO), dtype=IN_DTYPE)
OUT_TYPE = NamedType(dict(N=N, C=CO, H=H, W=W), dtype=OUT_DTYPE)

# CMSIS out dtype is the same as in dtype
CMSIS_OUT_TYPE = NamedType(dict(N=N, C=CO, H=H, W=W), dtype=IN_DTYPE)
CMSIS_BIAS_LSHIFT = 0
CMSIS_OUT_RSHIFT = 9  # original param
# CMSIS_OUT_RSHIFT = 6

BIAS_TYPE = BakedType([('B', CO)], dtype=IN_DTYPE)

################
# TRIAL CONFIG #
################
NUM_TRIALS = 15
# NUM_TRIALS = 1

##############
# EVALUATION #
##############

def eval_micro(
        sess, c_mod,
        data_nt: NamedTensor,
        kernel_nt: NamedTensor,
        out_type: BakedType,
        lib_src_paths=None):
    # data_np = np.random.randint(-10, 10, size=data_type.shape, dtype=data_type.dtype)
    # kernel_np = np.random.randint(-3, 3, size=kernel_type.shape, dtype=kernel_type.dtype)

    micro_mod = create_micro_mod(
        c_mod, DEV_CONFIG, lib_src_paths=lib_src_paths, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['conv2d']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_nt.data, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_nt.data, ctx=ctx)
    output_tvm = tvm.nd.array(np.zeros(out_type.shape, dtype=out_type.dtype), ctx=ctx)

    batch_time = benchmark_micro_func(
        sess, micro_func,
        [data_tvm, kernel_tvm, output_tvm],
        NUM_TRIALS, TIME_OVERHEAD)

    output_np = output_tvm.asnumpy()
    assert np.sum(output_np) != 0

    print('checking result against topi oracle...')
    check_conv2d_output(
        data_nt,
        kernel_nt,
        NamedTensor(output_np, out_type.layout),
        STRIDES, PADDING)

    return batch_time


# def eval_cmsis(sess, data_np, kernel_np, bias_np):
def eval_cmsis(sess, data_nt, kernel_nt, out_type):
    DATA_LAYOUT = 'NHWC'
    KERNEL_LAYOUT = 'IHWO'
    data_nt = data_nt.with_layout(DATA_LAYOUT)
    kernel_nt = kernel_nt.with_layout(KERNEL_LAYOUT)
    out_type = out_type.with_layout(DATA_LAYOUT)

    CMSIS_CONV_SRC_PATH = f'{os.path.dirname(__file__)}/../../../cmsis_src/cmsis_fast_conv2d.c'
    CMSIS_SRC_PATHS = [
        f'{CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
        f'{CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
        f'{CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c'
    ]

    # data_np = np.random.randint(-10, 10, size=data_type.shape, dtype=data_type.dtype)
    # kernel_np = np.random.randint(-3, 3, size=kernel_type.shape, dtype=kernel_type.dtype)
    # bias_np = np.random.randint(-3, 3, size=BIAS_TYPE.shape, dtype=BIAS_TYPE.dtype)
    bias_nt = BIAS_TYPE.gen_zero_tensor()
    # encode conv params in a tensor
    metadata_np = np.array(
        [PADDING, STRIDES[0], CMSIS_BIAS_LSHIFT, CMSIS_OUT_RSHIFT],
        dtype=np.uint16)
    out_nt = out_type.gen_zero_tensor()

    micro_mod = create_micro_mod(
        MockCMod(CMSIS_CONV_SRC_PATH),
        DEV_CONFIG,
        lib_src_paths=CMSIS_SRC_PATHS,
        lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['arm_fast_conv2d_wrapper']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_nt.data, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_nt.data, ctx=ctx)
    bias_tvm = tvm.nd.array(bias_nt.data, ctx=ctx)
    metadata_tvm = tvm.nd.array(metadata_np, ctx=ctx)
    output_tvm = tvm.nd.array(out_nt.data, ctx=ctx)

    batch_time = benchmark_micro_func(
        sess, micro_func,
        [data_tvm, kernel_tvm, bias_tvm, metadata_tvm, output_tvm],
        NUM_TRIALS, TIME_OVERHEAD)
    output_np = output_tvm.asnumpy()

    # TODO not sure how to interpret their output...
    # print('checking result against topi oracle...')
    # check_conv2d_output(
    #     NamedTensor(data_np, data_type.layout),
    #     NamedTensor(kernel_np, kernel_type.layout),
    #     NamedTensor(output_np, out_type.layout),
    #     STRIDES, PADDING)
    assert output_np.sum() != 0

    return batch_time


def eval_direct(sess, data_nt, kernel_nt):
    results = []
    for data_layout, kernel_layout in [('NCHW', 'OIHW'), ('NHWC', 'HWIO')]:
        data_nt = data_nt.with_layout(data_layout)
        kernel_nt = kernel_nt.with_layout(kernel_layout)
        out_type = OUT_TYPE.with_layout(data_layout)
        sched, arg_bufs = conv2d_direct(
                data_nt.typ.serialize(), kernel_nt.typ.serialize(),
                STRIDES, PADDING, DILATION, data_layout, out_type.dtype)
        c_mod = tvm.build(sched, arg_bufs, target=TARGET, name='conv2d')
        time = eval_micro(sess, c_mod, data_nt, kernel_nt, out_type)
        time -= TIME_OVERHEAD
        results.append(((data_layout, kernel_layout), time))
    return results


def eval_direct_simd(sess, data_nt, kernel_nt):
    def gen_direct_simd_cfg(M, K, N):
        cfg = ManualConfigSpace()
        cfg.template_key = 'direct_simd'
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

    results = []
    DATA_LAYOUT = 'NHWC'
    KERNEL_LAYOUT = 'HWOI'

    data_nt = data_nt.with_layout(DATA_LAYOUT)
    kernel_nt = kernel_nt.with_layout(KERNEL_LAYOUT)
    out_type = OUT_TYPE.with_layout(DATA_LAYOUT)
    # TODO autogen candidates?
    for (M, K, N) in [(1, 4, 1), (2, 4, 2), (4, 4, 4)]:
        with ManualConfigContext(gen_direct_simd_cfg(M, K, N)), TARGET:
            sched, arg_bufs = conv2d_direct_simd(
                data_nt.typ.serialize(), kernel_nt.typ.serialize(),
                STRIDES, PADDING, DILATION, DATA_LAYOUT, OUT_DTYPE)
        c_mod = tvm.build(sched, arg_bufs, target=TARGET, name='conv2d')
        time = eval_micro(sess, c_mod, data_nt, kernel_nt, out_type)
        time -= TIME_OVERHEAD
        results.append(((M, K, N), time))
    return results


def eval_partial_im2col(sess, data_nt, kernel_nt):
    def gen_partial_im2col_cfg(im2col_batch_size):
        cfg = ManualConfigSpace()
        cfg.template_key = 'partial_im2col'
        cfg['im2col_batch_size'] = OtherOptionEntity(im2col_batch_size)
        return cfg

    NUM_BATCH_SIZE_CANDIDATES = 3
    max_batch_size = H * W
    # use the first NUM_BATCH_SIZE_CANDIDATES factors of the max batch size
    batch_sizes = [i for i in range(1, max_batch_size+1) if max_batch_size % i == 0]
    if len(batch_sizes) > NUM_BATCH_SIZE_CANDIDATES:
        batch_sizes = batch_sizes[:NUM_BATCH_SIZE_CANDIDATES]

    DATA_LAYOUT = 'NHWC'
    KERNEL_LAYOUT = 'OIHW'
    data_nt = data_nt.with_layout(DATA_LAYOUT)
    kernel_nt = kernel_nt.with_layout(KERNEL_LAYOUT)
    out_type = OUT_TYPE.with_layout(DATA_LAYOUT)

    results = []
    for i2c_batch_size in batch_sizes:
        with ManualConfigContext(gen_partial_im2col_cfg(i2c_batch_size)), TARGET:
            sched, arg_bufs = conv2d_partial_im2col(
                    data_nt.typ.serialize(), kernel_nt.typ.serialize(),
                    STRIDES, PADDING, DILATION, DATA_LAYOUT, OUT_DTYPE)
        c_mod = tvm.build(sched, arg_bufs, target=TARGET, name='conv2d')
        time = eval_micro(sess, c_mod, data_nt, kernel_nt, out_type)
        time -= TIME_OVERHEAD
        results.append((i2c_batch_size, time))
    return results


def main():
    reset_gdbinit(DEV_CONFIG)

    data_nt = DATA_TYPE.gen_rand_tensor(-10, 10)
    kernel_nt = KERNEL_TYPE.gen_rand_tensor(-3, 3)
    all_results = []
    with micro.Session(DEV_CONFIG) as sess:
        # CMSIS-NN
        cmsis_results = eval_cmsis(sess, data_nt, kernel_nt, CMSIS_OUT_TYPE)
        all_results.append(
f"""
############
# CMSIS-NN #
############
{cmsis_results}
"""
        )

        # direct w/o SIMD
        [default_nchw_time, default_nhwc_time] = eval_direct(sess, data_nt, kernel_nt)
        all_results.append(
f"""
##########
# DIRECT #
##########
  NCHW time: {default_nchw_time}
  NHWC time: {default_nhwc_time}
""")

        # direct w/ SIMD
        direct_simd_results = eval_direct_simd(sess, data_nt, kernel_nt)
        [small_direct_simd_time, medium_direct_simd_time, large_direct_simd_time] = direct_simd_results
        all_results.append(
f"""
#################
# DIRECT + SIMD #
#################
  small time: {small_direct_simd_time}
  medium time: {medium_direct_simd_time}
  large time: {large_direct_simd_time}
""")

        # partial im2col w/ SIMD
        partial_im2col_results = eval_partial_im2col(sess, data_nt, kernel_nt)
        i2c_report = [
f"""
######################
# IM2COL CONV + SIMD #
######################""".strip()]
        for (batch_size, i2c_simd_time) in partial_im2col_results:
            i2c_report.append(f'  batch of {batch_size} time: {i2c_simd_time}')
        i2c_report = '\n'.join(i2c_report)
        all_results.append(i2c_report)

        print('\n\n'.join(map(lambda s: s.strip(), all_results)))

        # TODO write function that gives a 2d heatmap of mismatches (showing
        # percentage of entries that don't match) by choosing two view axes and
        # reducing other axes



if __name__ == "__main__":
    main()
