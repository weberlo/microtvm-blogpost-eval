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

from micro_eval.util import (
        MockCMod,
        show_c_source,
        transform_data_layout,
        get_cmsis_path, get_cmsis_include_paths,
        relay_micro_build, reset_gdbinit,
        get_comm_overhead, benchmark_micro_func)

DENSE_SRC_PATH = f'{os.path.dirname(__file__)}/../../../cmsis_src/cmsis_dense.c'

CMSIS_INCLUDE_PATHS = get_cmsis_include_paths()

CMSIS_PATH = get_cmsis_path()
CMSIS_LIB_SRC_PATHS = [
    f'{CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c'
]

# TODO host device isn't working
# DEV_CONFIG = micro.device.host.default_config()
DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)

N, H, W, C = 1, 3, 3, 64
NUM_OUTPUTS = 10
LAYOUT = 'NHWC'
DTYPE = 'int8'
NUM_TRIALS = 15

def run_cmsis_dense(sess, data_tvm, weight_tvm, bias_tvm):
    micro_mod = create_micro_mod(
        MockCMod(DENSE_SRC_PATH),
        DEV_CONFIG,
        lib_src_paths=CMSIS_LIB_SRC_PATHS,
        lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['arm_dense_wrapper']
    ctx = tvm.micro_dev(0)

    output_tvm = tvm.nd.empty([N, NUM_OUTPUTS], ctx=ctx, dtype=DTYPE)
    batch_time, _ = benchmark_micro_func(
        sess, micro_func, [data_tvm, weight_tvm, bias_tvm, output_tvm], NUM_TRIALS)

    return output_tvm.asnumpy(), batch_time


def run_micro_dense(sess, data_tvm, weight_tvm, bias_tvm):
    func_name = 'dense'
    data_ph = tvm.placeholder((N, H * W * C), name='data')
    weight_ph = tvm.placeholder((NUM_OUTPUTS, H * W * C), name='weight')
    bias_ph = tvm.placeholder((NUM_OUTPUTS,), name='bias')
    output_ph = tvm.placeholder((NUM_OUTPUTS,), name='output')
    dense = topi.nn.dense(data_ph, weight_ph, bias=bias_ph, out_dtype=DTYPE)
    sched = tvm.create_schedule([dense.op])
    with tvm.build_config(disable_vectorize=True):
        c_mod = tvm.build(sched, [data_ph, weight_ph, bias_ph, output_ph], target='c', name=func_name)

    ctx = tvm.micro_dev(0)
    # TODO we shouldn't need to include CMSIS headers if we're not using it.
    # change C codegen to only include it when it's necessary.
    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod[func_name]

    output_tvm = tvm.nd.empty([N, NUM_OUTPUTS], ctx=ctx, dtype=DTYPE)
    batch_time, _ = benchmark_micro_func(
        sess, micro_func, [data_tvm, weight_tvm, bias_tvm, output_tvm], NUM_TRIALS)

    return output_tvm.asnumpy(), batch_time


def run_intrp_dense(data_np, weight_np, bias_np):
    from tvm.relay import create_executor
    from tvm.relay import transform

    # Construct Relay program.
    data_var = relay.var("data", shape=(N, H * W * C), dtype=DTYPE)
    weight_var = relay.var("weight", shape=(NUM_OUTPUTS, H * W * C), dtype=DTYPE)
    bias_var = relay.var("bias", shape=(NUM_OUTPUTS,), dtype=DTYPE)
    dense_expr = relay.nn.dense(
            data_var, weight_var, out_dtype=DTYPE)
    bias_expr = relay.nn.bias_add(dense_expr, bias_var)
    func = relay.Function(relay.analysis.free_vars(bias_expr), bias_expr)
    mod = relay.Module.from_expr(func)
    mod = transform.InferType()(mod)

    data_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    weight_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    bias_shape = list(map(lambda x: x.value, mod['main'].params[2].checked_type.shape))
    out_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))

    intrp = create_executor('debug')
    intrp_output = intrp.evaluate(mod['main'])(data_np, weight_np, bias_np).data

    return intrp_output


def main():
    reset_gdbinit(DEV_CONFIG)

    data_np = np.random.randint(-10, 10, size=(N, H * W * C), dtype=DTYPE)
    weight_np = np.random.randint(-1, 1, size=(NUM_OUTPUTS, H * W * C), dtype=DTYPE)
    # bias_np = np.random.randint(-3, 3, size=(NUM_OUTPUTS,), dtype=DTYPE)
    bias_np = np.zeros((NUM_OUTPUTS,), dtype=DTYPE)
    assert False, 'figure out source of CMSIS <-> MICRO discrepancies. keep bias at zero to simplify things'

    with micro.Session(DEV_CONFIG) as sess:
        ctx = tvm.micro_dev(0)
        data_tvm = tvm.nd.array(data_np, ctx=ctx)
        weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
        bias_tvm = tvm.nd.array(bias_np, ctx=ctx)

        cmsis_output_np, cmsis_time = run_cmsis_dense(sess, data_tvm, weight_tvm, bias_tvm)

        print("verifying input tensors weren't corrupted...")
        tvm.testing.assert_allclose(data_tvm.asnumpy(), data_np)
        tvm.testing.assert_allclose(weight_tvm.asnumpy(), weight_np)
        tvm.testing.assert_allclose(bias_tvm.asnumpy(), bias_np)

        micro_output_np, micro_time = run_micro_dense(sess, data_tvm, weight_tvm, bias_tvm)

        print("verifying input tensors weren't corrupted...")
        tvm.testing.assert_allclose(data_tvm.asnumpy(), data_np)
        tvm.testing.assert_allclose(weight_tvm.asnumpy(), weight_np)
        tvm.testing.assert_allclose(bias_tvm.asnumpy(), bias_np)

    intrp_output_np = run_intrp_dense(data_np, weight_np, bias_np)

    print('[CMSIS]')
    print(f'Time: {cmsis_time}')
    print('[MicroTVM]')
    print(f'Time: {micro_time}')
    print('[MicroTVM Speedup]')
    print(f'Time: {cmsis_time / micro_time}')
    assert np.sum(cmsis_output_np) != 0
    tvm.testing.assert_allclose(cmsis_output_np, micro_output_np)
    tvm.testing.assert_allclose(micro_output_np, intrp_output_np)
    assert False, "use topi.testing.conv2d_nhwc_python to verify!"


if __name__ == "__main__":
    main()
