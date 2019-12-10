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

from micro_eval.util import relay_micro_build, reset_gdbinit, get_comm_overhead

if 'CMSIS_PATH' not in os.environ:
    raise RuntimeError('must have "CMSIS_PATH" in environment')
CMSIS_PATH = os.environ['CMSIS_PATH']

class DummyCMod:
    def __init__(self):
        pass

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, f'{os.path.dirname(__file__)}/../../src/cmsis_rgb_conv2d.c')


CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include'
]

CMSIS_SRC_PATHS = [
    f'{CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c'
]

from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint
DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6667)
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

N, H, W, CO, CI = 1, 32, 32, 32, 3
KH, KW = 5, 5
STRIDES, PADDING, DILATION = (1, 1), (2, 2), (1, 1)
KERNEL_SIZE = (KH, KW)
DTYPE = 'int8'

TARGET = tvm.target.create('c -device=micro_dev')

CMSIS_DATA_SHAPE = (N, H, W, CI)
CMSIS_KERNEL_SHAPE = (CI, KH, KW, CO)
CMSIS_BIAS_SHAPE = (CO,)
CMSIS_OUTPUT_SHAPE = (N, H, W, CO)
CMSIS_LAYOUT = 'NHWC'

TVM_DATA_SHAPE = (N, CI, H, W)
TVM_KERNEL_SHAPE = (CO, CI, KH, KW)
TVM_BIAS_SHAPE = (CO,)
TVM_OUTPUT_SHAPE = (N, CO, H, W)
TVM_LAYOUT = 'NCHW'

NUM_TRIALS = 15

USE_TUNED_SCHEDULES = True

def benchmark_micro_func(sess, micro_func, args, num_trials=NUM_TRIALS):
    ctx = tvm.micro_dev(0)
    # sync before and after to ensure these are the only tasks in the queue
    ctx.sync()
    sess.get_last_batch_time()
    sess.get_last_batch_cycles()
    for _ in range(NUM_TRIALS):
        micro_func(*args)
    ctx.sync()
    return sess.get_last_batch_time(), sess.get_last_batch_cycles()


def run_cmsis_conv2d(sess, time_overhead, cycle_overhead, data_np, kernel_np, bias_np):
    micro_mod = create_micro_mod(
        DummyCMod(),
        DEV_CONFIG,
        lib_src_paths=CMSIS_SRC_PATHS,
        lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['arm_rgb_conv2d_wrapper']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    output_tvm = tvm.nd.array(np.zeros(CMSIS_OUTPUT_SHAPE, dtype=DTYPE), ctx=ctx)

    batch_time, batch_cycles = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, bias_tvm, output_tvm])
    batch_time -= time_overhead
    batch_cycles -= cycle_overhead

    return output_tvm.asnumpy(), batch_time, batch_cycles


def run_micro_conv2d(sess, time_overhead, cycle_overhead, data_np, kernel_np, bias_np):
    mod = build_conv2d_relay()
    #with tvm.build_config(disable_vectorize=True):
    #    #graph, c_mod, params = relay.build(mod, target="c")
    #    graph, c_mod, params = relay.build(mod, target=TARGET)

    #with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #    with tvm.build_config(disable_vectorize=True):
    #        graph, c_mod, params = relay.build(mod['main'], target=TARGET, params={})
    #        input(c_mod.get_source())

    params = {
        'conv0_weight': tvm.nd.array(kernel_np, ctx=tvm.cpu(0)),
        'conv0_bias': tvm.nd.array(bias_np, ctx=tvm.cpu(0))
    }
    if USE_TUNED_SCHEDULES:
        DEVICE_ID = 'arm.stm32f746xx'
        E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'
        with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
            with TARGET:
                graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, TARGET, params=params)
    else:
        graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, TARGET, params=params)

    ctx = tvm.micro_dev(0)
    ctx.sync()
    sess.get_last_batch_time()
    sess.get_last_batch_cycles()
    graph_mod.set_input(data=data_np)
    for _ in range(NUM_TRIALS):
        graph_mod.run()
    ctx.sync()
    batch_time = sess.get_last_batch_time()
    batch_cycles = sess.get_last_batch_cycles()
    batch_time -= time_overhead
    batch_cycles -= cycle_overhead

    #micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
    ##micro_func = micro_mod['fused_nn_conv2d_add_right_shift_cast']
    #micro_func = micro_mod['fused_nn_conv2d_add_right_shift_cast']
    #ctx = tvm.micro_dev(0)

    #data_tvm = tvm.nd.array(data_np, ctx=ctx)
    #kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    #bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    #output_tvm = tvm.nd.array(np.zeros(TVM_OUTPUT_SHAPE, dtype=DTYPE), ctx=ctx)

    #batch_time = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, bias_tvm, output_tvm])

    return graph_mod.get_output(0).asnumpy(), batch_time, batch_cycles


def run_intrp_conv2d(data_np, kernel_np, bias_np):
    mod = build_conv2d_relay()
    intrp = create_executor('debug')
    result = intrp.evaluate(mod['main'])(data_np, kernel_np, bias_np).data
    return result


def build_conv2d_relay():
    # Construct Relay program (used for micro and interpreter eval).
    #data_var = relay.var("data", shape=TVM_DATA_SHAPE, dtype=DTYPE)
    #kernel_var = relay.var("kernel", shape=TVM_KERNEL_SHAPE, dtype=DTYPE)
    #bias_var = relay.var("bias", shape=TVM_BIAS_SHAPE, dtype=DTYPE)
    #conv_expr = relay.nn.conv2d(
    #        data_var, kernel_var,
    #        kernel_size=KERNEL_SIZE,
    #        strides=STRIDES,
    #        padding=PADDING,
    #        dilation=DILATION,
    #        channels=CO,
    #        data_layout=TVM_LAYOUT,
    #        out_layout=TVM_LAYOUT)
    #bias_add_expr = relay.nn.bias_add(conv_expr, bias_var, axis=1)
    #func = relay.Function(relay.analysis.free_vars(bias_add_expr), bias_add_expr)
    #mod = relay.Module.from_expr(func)
    #mod = transform.InferType()(mod)

    mod = relay.fromtext(f"""
    v0.0.4
    def @main(%data: Tensor[(1, {CI}, {H}, {W}), int8],
        %conv0_weight: Tensor[({CO}, {CI}, {KH}, {KW}), int8],
        %conv0_bias: Tensor[({CO}), int8]) {{
      %0 = nn.conv2d(%data, %conv0_weight, padding=[2, 2], channels={CO}, kernel_size=[5, 5], out_dtype="int32");
      %1 = nn.bias_add(%0, cast(%conv0_bias, "int32"));
      %2 = right_shift(%1, 9);
      cast(%2, "int8")
    }}
    """)
    return mod


def main():
    reset_gdbinit(DEV_CONFIG)

    time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)

    with micro.Session(DEV_CONFIG) as sess:
        # gen CMSIS tensors
        data_np = np.random.randint(-10, 10, size=CMSIS_DATA_SHAPE, dtype=DTYPE)
        kernel_np = np.random.randint(-3, 3, size=CMSIS_KERNEL_SHAPE, dtype=DTYPE)
        bias_np = np.random.randint(-3, 3, size=CMSIS_BIAS_SHAPE, dtype=DTYPE)

        cmsis_output_np, cmsis_time, cmsis_cycles = run_cmsis_conv2d(sess, time_overhead, cycle_overhead, data_np, kernel_np, bias_np)
        assert np.sum(cmsis_output_np) != 0

        # gen TVM tensors
        data_np = np.random.randint(-10, 10, size=TVM_DATA_SHAPE, dtype=DTYPE)
        kernel_np = np.random.randint(-3, 3, size=TVM_KERNEL_SHAPE, dtype=DTYPE)
        bias_np = np.random.randint(-3, 3, size=TVM_BIAS_SHAPE, dtype=DTYPE)

        micro_output_np, micro_time, micro_cycles = run_micro_conv2d(sess, time_overhead, cycle_overhead, data_np, kernel_np, bias_np)
        assert np.sum(micro_output_np) != 0

        #intrp_output_np = run_intrp_conv2d(data_np, kernel_np, bias_np)

        print(f'Time overhead was {time_overhead}')
        print('[CMSIS]')
        print(f'Total Batch Time: {cmsis_time}')
        #print(f'Total Batch Cycles: {cmsis_cycles}')
        print(f'Time Per Trial: {cmsis_time / NUM_TRIALS}')
        #print(f'Cycles Per Trial: {cmsis_cycles / NUM_TRIALS}')
        print('[MicroTVM]')
        print(f'Total Batch Time: {micro_time}')
        #print(f'Total Batch Cycles: {micro_cycles}')
        print(f'Time Per Trial: {micro_time / NUM_TRIALS}')
        #print(f'Cycles Per Trial: {micro_cycles / NUM_TRIALS}')
        print('[MicroTVM Speedup]')
        print(f'Time: {cmsis_time / micro_time}')
        #print(f'Cycles: {cmsis_cycles / micro_cycles}')
        #assert np.array_equal(micro_output_np, intrp_output_np)


if __name__ == "__main__":
    main()
