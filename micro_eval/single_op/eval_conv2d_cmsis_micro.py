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

import numpy as np
import topi
import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.relay.testing import resnet
from tvm.relay import transform
from tvm.relay import create_executor

if 'CMSIS_PATH' not in os.environ:
    raise RuntimeError('must have "CMSIS_PATH" in environment')
CMSIS_PATH = os.environ['CMSIS_PATH']

def reset_gdbinit():
    with open('/home/lweber/gdb-conf/.gdbinit', 'w') as f:
        gdbinit_contents = (
"""
layout asm
target remote localhost:3333
set $pc = UTVMInit
break UTVMDone
""")
        f.write(gdbinit_contents)


class DummyCMod:
    def __init__(self):
        pass

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, f'{os.path.dirname(__file__)}/cmsis_conv2d.c')


CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include'
]

CMSIS_SRC_PATHS = [
    f'{CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c'
]

DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)

N, H, W, CO, CI = 1, 16, 16, 32, 32
KH, KW = 5, 5
STRIDES, PADDING, DILATION = (1, 1), (2, 2), (1, 1)
KERNEL_SIZE = (KH, KW)
DTYPE = 'int8'

CMSIS_DATA_SHAPE = (N, H, W, CI)
CMSIS_KERNEL_SHAPE = (CO, CI, KH, KW)
CMSIS_BIAS_SHAPE = (CO,)
CMSIS_OUTPUT_SHAPE = (N, H, W, CO)
CMSIS_LAYOUT = 'NHWC'

TVM_DATA_SHAPE = (N, CI, H, W)
TVM_KERNEL_SHAPE = (CO, CI, KH, KW)
TVM_BIAS_SHAPE = (CO,)
TVM_OUTPUT_SHAPE = (N, H, W, CO)
TVM_LAYOUT = 'NCHW'

NUM_TRIALS = 15

def benchmark_micro_func(sess, micro_func, args, num_trials=NUM_TRIALS):
    ctx = tvm.micro_dev(0)
    # sync before and after to ensure these are the only tasks in the queue
    ctx.sync()
    for _ in range(NUM_TRIALS):
        micro_func(*args)
    ctx.sync()
    return sess.get_last_batch_time()


def run_cmsis_conv2d(sess, data_np, kernel_np, bias_np):
    micro_mod = create_micro_mod(
        DummyCMod(),
        DEV_CONFIG,
        lib_src_paths=CMSIS_SRC_PATHS,
        lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['arm_conv2d_wrapper']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    output_tvm = tvm.nd.empty(CMSIS_OUTPUT_SHAPE, ctx=ctx, dtype=DTYPE)

    batch_time = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, bias_tvm, output_tvm])

    return output_tvm.asnumpy(), batch_time


def run_micro_conv2d(sess, data_np, kernel_np, bias_np):
    mod = build_conv2d_relay()
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(mod, target="c")
    micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
    micro_func = micro_mod['fused_nn_conv2d_nn_bias_add']
    ctx = tvm.micro_dev(0)

    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
    bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    output_tvm = tvm.nd.empty(TVM_OUTPUT_SHAPE, ctx=ctx, dtype=DTYPE)

    batch_time = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, bias_tvm, output_tvm])

    return output_tvm.asnumpy(), batch_time


def run_intrp_conv2d(data_np, kernel_np, bias_np):
    mod = build_conv2d_relay()
    intrp = create_executor('debug')
    result = intrp.evaluate(mod['main'])(data_np, kernel_np, bias_np).data
    return result


def build_conv2d_relay():
    #assert False, "we might need to use NCHW for micro and interp, because the bias add causes problems"
    # Construct Relay program (used for micro and interpreter eval).
    data_var = relay.var("data", shape=TVM_DATA_SHAPE, dtype=DTYPE)
    kernel_var = relay.var("kernel", shape=TVM_KERNEL_SHAPE, dtype=DTYPE)
    bias_var = relay.var("bias", shape=TVM_BIAS_SHAPE, dtype=DTYPE)
    conv_expr = relay.nn.conv2d(
            data_var, kernel_var,
            kernel_size=KERNEL_SIZE,
            strides=STRIDES,
            padding=PADDING,
            dilation=DILATION,
            channels=CO,
            data_layout=TVM_LAYOUT,
            out_layout=TVM_LAYOUT)
    bias_add_expr = relay.nn.bias_add(conv_expr, bias_var, axis=1)
    func = relay.Function(relay.analysis.free_vars(bias_add_expr), bias_add_expr)
    mod = relay.Module.from_expr(func)
    mod = transform.InferType()(mod)
    return mod


def main():
    reset_gdbinit()

    with micro.Session(DEV_CONFIG) as sess:
        # gen CMSIS tensors
        data_np = np.random.randint(-10, 10, size=CMSIS_DATA_SHAPE, dtype=DTYPE)
        kernel_np = np.random.randint(-3, 3, size=CMSIS_KERNEL_SHAPE, dtype=DTYPE)
        bias_np = np.random.randint(-3, 3, size=CMSIS_BIAS_SHAPE, dtype=DTYPE)

        #cmsis_output_np, cmsis_time = run_cmsis_conv2d(sess, data_np, kernel_np, bias_np)
        cmsis_time = 0.4
        #assert np.sum(cmsis_output_np) != 0

        # gen TVM tensors
        data_np = np.random.randint(-10, 10, size=TVM_DATA_SHAPE, dtype=DTYPE)
        kernel_np = np.random.randint(-3, 3, size=TVM_KERNEL_SHAPE, dtype=DTYPE)
        bias_np = np.random.randint(-3, 3, size=TVM_BIAS_SHAPE, dtype=DTYPE)

        micro_output_np, micro_time = run_micro_conv2d(sess, data_np, kernel_np, bias_np)
        assert np.sum(micro_output_np) != 0
        intrp_output_np = run_intrp_conv2d(data_np, kernel_np, bias_np)

        print('[CMSIS]')
        print(f'Time: {cmsis_time}')
        print('[MicroTVM]')
        print(f'Time: {micro_time}')
        print('[MicroTVM Speedup]')
        print(f'Time: {cmsis_time / micro_time}')
        assert np.array_equal(micro_output_np, intrp_output_np)


# N, H, W, CO, CI, KH, KW = 1, 16, 16, 32, 32, 5, 5
# [CMSIS]
# Cycles: 10918149.0
# Time: 0.23316740989685059
# [MicroTVM]
# Cycles: 95281.0
# Time: 0.16225314140319824
# [MicroTVM Speedup]
# Cycles: 114.58894218154721
# Time: 1.437059448466583

# N, H, W, CO, CI, KH, KW = 1, 18, 18, 32, 32, 5, 5
# [CMSIS]
# Cycles: 13792940.0
# Time: 0.24947428703308105
# [MicroTVM]
# Cycles: 109095.0
# Time: 0.16560769081115723
# [MicroTVM Speedup]
# Cycles: 126.43054218800128
# Time: 1.5064172793614825

# N, H, W, CO, CI, KH, KW = 1, 20, 20, 32, 32, 5, 5
# [CMSIS]
# Cycles: 245489.0
# Time: 0.26293063163757324
# [MicroTVM]
# Cycles: 131617.0
# Time: 0.16651177406311035
# [MicroTVM Speedup]
# Cycles: 1.8651769908142566
# Time: 1.5790512900181988

# N, H, W, CO, CI, KH, KW = 1, 24, 24, 32, 32, 5, 5
# [CMSIS]
# Cycles: 7707967.0
# Time: 0.2836883068084717
# [MicroTVM]
# Cycles: 185840.0
# Time: 0.17332792282104492
# [MicroTVM Speedup]
# Cycles: 41.47636138613861
# Time: 1.6367143977221146

# N, H, W, CO, CI, KH, KW = 1, 28, 28, 32, 32, 5, 5
# [CMSIS]
# Cycles: 16546962.0
# Time: 0.33393096923828125
# [MicroTVM]
# Cycles: 2710118.0
# Time: 0.18435430526733398
# [MicroTVM Speedup]
# Cycles: 6.105624183153648
# Time: 1.811354330749394

# N, H, W, CO, CI, KH, KW = 1, 30, 30, 32, 32, 5, 5
# [CMSIS]
# Cycles: 4694608.0
# Time: 0.3659372329711914
# [MicroTVM]
# Cycles: 4411234.0
# Time: 0.2863960266113281
# [MicroTVM Speedup]
# Cycles: 1.0642391675435943
# Time: 1.2777315289635973

# N, H, W, CO, CI, KH, KW = 1, 31, 31, 32, 32, 5, 5
# [CMSIS]
# Cycles: 7286871.0
# Time: 0.37795591354370117
# [MicroTVM]
# Cycles: 3704344.0
# Time: 0.3617746829986572
# [MicroTVM Speedup]
# Cycles: 1.9671150951423517
# Time: 1.0447273712215623

# N, H, W, CO, CI, KH, KW = 1, 32, 32, 32, 32, 5, 5
# [CMSIS]
# Cycles: 9957309.0
# Time: 0.3922536373138428
# [MicroTVM]
# Cycles: 13185781.0
# Time: 0.49291563034057617
# [MicroTVM Speedup]
# Cycles: 0.7551550416315879
# Time: 0.7957825095601416


if __name__ == "__main__":
    main()
