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

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from micro_eval.util import relay_micro_build, reset_gdbinit, get_comm_overhead, benchmark_micro_func

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

TARGET = tvm.target.create('c -device=micro_dev')

###############
# CONV CONFIG #
###############
N, H, W, CO, CI = 1, 16, 16, 32, 32
KH, KW = 5, 5
STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
LAYOUT = 'NCHW'
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
DATA_SHAPE = (N, CI, H, W)
KERNEL_SHAPE = (CO, CI, KH, KW)
DATA_TENSOR = ('TENSOR', DATA_SHAPE, IN_DTYPE)
KERNEL_TENSOR = ('TENSOR', KERNEL_SHAPE, IN_DTYPE)

BIAS_SHAPE = (CO,)
OUTPUT_SHAPE = (N, CO, H, W)

#################
# MATMUL CONFIG #
#################
#M, K, N = 32, 16, 32
#IN_DTYPE = 'int8'
#OUT_DTYPE = 'int32'
#A_SHAPE = (M, K)
#B_SHAPE = (K, N)
#A_TENSOR = ('TENSOR', A_SHAPE, IN_DTYPE)
#B_TENSOR = ('TENSOR', B_SHAPE, IN_DTYPE)

#N, M, L = 1024, 512, 64
#M, K, N = 2, 4, 2
M, K, N = 16, 32, 16
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

USE_TUNED_SCHEDULES = False

#def conv2d_arm_micro_nchw_template(data, kernel, strides, padding, dilation, layout, out_dtype):
#    data_typ, data_shape, data_dtype = data
#    kernel_typ, kernel_shape, kernel_dtype = kernel
#    assert data_typ == 'TENSOR'
#    assert kernel_typ == 'TENSOR'
#    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
#    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)
#    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)
#    sched = tvm.create_schedule([conv.op])
#
#    #data_vec = conv.op.input_tensors[0]
#    #data_pad = data_vec.op
#    #sched[data_pad].compute_inline()
#
#    ## assign axes of the default schedule to variables
#    #n, co, oh, ow = sched[conv].op.axis
#    #ci, kh, kw = sched[conv].op.reduce_axis
#
#    #co, vc = sched[conv].split(co, factor=2)
#    #oh, vh = sched[conv].split(oh, factor=2)
#    #ow, vw = sched[conv].split(ow, factor=2)
#
#    #sched[conv].reorder(n, co, oh, ow, ci, kh, kw, vh, vw, vc)
#    
#    #sched[conv].vectorize(vc)
#
#    return sched, [data, kernel, conv]


#def matmul_arm_micro_template(A, B, out_dtype):
#    A_typ, A_shape, A_dtype = A
#    B_typ, B_shape, B_dtype = B
#    assert A_typ == 'TENSOR'
#    assert B_typ == 'TENSOR'
#
#    # Algorithm
#    k = tvm.reduce_axis((0, A_shape[1]), 'k')
#    A = tvm.placeholder(A_shape, name='A', dtype=A_dtype)
#    B = tvm.placeholder(B_shape, name='B', dtype=B_dtype)
#    C = tvm.compute(
#               (M, N),
#               lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
#               name='C')
#
#    # Default schedule
#    sched = tvm.create_schedule(C.op)
#
#    bn = 8
#    xo, yo, xi, yi = sched[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
#    k = sched[C].op.reduce_axis[0]
#    #ko, ki = sched[C].split(k, factor=4)
#
#    #sched[C].reorder(xo, yo, ko, ki, xi, yi)
#    sched[C].reorder(xo, yo, k, xi, yi)
#    sched[C].vectorize(yi)
#
#    # TODO use out_dtype
#
#    return sched, [A, B, C]


# NOTE this is transposed matmul
# NOTE only works for column sizes that are multiples of 4
def matmul_arm_micro_template(A, B, out_dtype):
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

    x, y = C.op.axis
    z, = C.op.reduce_axis

    xo, xi = sched[C].split(x, factor=2)
    yo, yi = sched[C].split(y, factor=2)
    zo, zi = sched[C].split(z, factor=4)
    sched[C].reorder(xo, yo, zo, xi, yi, zi)

    # # NOTE we need to create an outer dummy axis to attach the "import_c"
    # # pragma, because we can't attach pragmas to any tensorized axes
    # dummy_axis, x = sched[C].split(x, factor=(x.dom.extent - x.dom.min))
    gemm = intrin_gemm_2x4x2(K, N, A.dtype, C.dtype)
    sched[C].tensorize(xi, gemm)
    sched[C].pragma(xo, "import_c", gemm_2x4x2_impl())

    input(tvm.lower(sched, [A, B, C], simple_mode=True))

    return sched, [A, B, C]


def intrin_gemm_2x4x2(K, N, in_dtype, out_dtype):
    A = tvm.placeholder((2, 4), name='a', dtype=in_dtype)
    B = tvm.placeholder((2, 4), name='b', dtype=in_dtype)
    k = tvm.reduce_axis((0, 4), name='k')
    C = tvm.compute((2, 2), lambda i, j: tvm.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k), name='c')
    A_buf = tvm.decl_buffer(
            A.shape, A.dtype,
            name="A",
            offset_factor=1,
            strides=[tvm.var("A_s"), 1])
    B_buf = tvm.decl_buffer(
            B.shape, B.dtype,
            name="B",
            offset_factor=1,
            strides=[tvm.var("B_s"), 1])
    C_buf = tvm.decl_buffer(
            C.shape, C.dtype,
            name="C",
            offset_factor=1,
            strides=[tvm.var("C_s"), 1])
    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        def _body():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", "gemm_2x4x2_update",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    cc.access_ptr("w"),
                                    aa.strides[0],
                                    bb.strides[0],
                                    cc.strides[0]))
            return ib.get()
        def _reduce_reset():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", "gemm_2x4x2_reset",
                                    cc.access_ptr("w"),
                                    cc.strides[0]))
            return ib.get()
        def _reduce_update():
            return _body()
        return _body(), _reduce_reset(), _reduce_update()
    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: A_buf, B: B_buf, C: C_buf})


def gemm_2x4x2_impl():
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = """
#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_2x4x2_update(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  q7_t *pA = aa;
  q7_t *pA2 = aa + A_stride;
  q7_t *pB = bb;
  q7_t *pB2 = bb + B_stride;

  q31_t sum11 = 0;
  q31_t sum12 = 0;
  q31_t sum21 = 0;
  q31_t sum22 = 0;

  q31_t inA11, inA12, inA21, inA22;
  q31_t inB11, inB12, inB21, inB22;
  pA = read_and_pad(pA, &inA11, &inA12);
  pA2 = read_and_pad(pA2, &inA21, &inA22);
  pB = read_and_pad(pB, &inB11, &inB12);
  pB2 = read_and_pad(pB2, &inB21, &inB22);
  
  sum11 = __SMLAD(inA11, inB11, sum11);
  sum11 = __SMLAD(inA12, inB12, sum11);
  
  sum12 = __SMLAD(inA11, inB21, sum12);
  sum12 = __SMLAD(inA12, inB22, sum12);
  
  sum21 = __SMLAD(inA21, inB11, sum21);
  sum21 = __SMLAD(inA22, inB12, sum21);
  
  sum22 = __SMLAD(inA21, inB21, sum22);
  sum22 = __SMLAD(inA22, inB22, sum22);

  cc[0] += sum11;
  cc[1] += sum12;
  cc[C_stride] += sum21;
  cc[C_stride+1] += sum22;
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_2x4x2_reset(int32_t *cc, int C_stride) {
  cc[0] = 0;
  cc[1] = 0;
  cc[C_stride] = 0;
  cc[C_stride+1] = 0;
  return 0;
}
    """
    #from tvm.contrib import util, clang
    #temp = util.tempdir()
    #ll_path = temp.relpath("temp.ll")
    ## Create LLVM ir from c source code
    #ll_code = clang.create_llvm(cc_code, output=ll_path)
    #return ll_code
    return cc_code


#def run_micro_conv2d(sess, time_overhead, cycle_overhead):
#    data_np = np.random.randint(-10, 10, size=DATA_SHAPE, dtype=IN_DTYPE)
#    kernel_np = np.random.randint(-3, 3, size=KERNEL_SHAPE, dtype=IN_DTYPE)
#
#    # compile with default schedule
#    with TARGET:
#        sched, arg_bufs = conv2d_arm_micro_nchw_template(
#                DATA_TENSOR, KERNEL_TENSOR, STRIDES, PADDING, DILATION, LAYOUT, OUT_DTYPE)
#        c_mod = tvm.build(sched, arg_bufs, name='conv2d_simd')
#        input(c_mod.get_source())
#
#    micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
#    micro_func = micro_mod['conv2d_simd']
#    ctx = tvm.micro_dev(0)
#
#    data_tvm = tvm.nd.array(data_np, ctx=ctx)
#    kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
#    output_tvm = tvm.nd.array(np.zeros(OUTPUT_SHAPE, dtype=OUT_DTYPE), ctx=ctx)
#
#    batch_time, batch_cycles = benchmark_micro_func(sess, micro_func, [data_tvm, kernel_tvm, output_tvm], 1)
#    batch_time -= time_overhead
#    batch_cycles -= cycle_overhead
#
#    micro_output_np = output_tvm.asnumpy()
#    assert np.sum(micro_output_np) != 0


def run_micro_matmul(sess, time_overhead, cycle_overhead):
    sched, [A, B, C] = matmul_arm_micro_template(A_TENSOR, B_TENSOR, OUT_DTYPE)
    #func = tvm.build(sched, arg_bufs, target="llvm", name="gemv")
    with TARGET:
        c_mod = tvm.build(sched, [A, B, C], name="gemm")
    input(c_mod.get_source())

    from topi.util import get_const_tuple
    A_np = np.random.randint(-30, 30, size=get_const_tuple(A.shape)).astype(A.dtype)
    B_np = np.random.randint(-30, 30, size=get_const_tuple(B.shape)).astype(B.dtype)

    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['gemm']
    ctx = tvm.micro_dev(0)

    A_tvm = tvm.nd.array(A_np, ctx=ctx)
    B_tvm = tvm.nd.array(B_np, ctx=ctx)
    C_tvm = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)

    batch_time, batch_cycles = benchmark_micro_func(sess, micro_func, [A_tvm, B_tvm, C_tvm], 1)
    batch_time -= time_overhead
    batch_cycles -= cycle_overhead

    C_np = C_tvm.asnumpy()
    assert np.sum(C_np) != 0
    tvm.testing.assert_allclose(C_np, np.dot(A_np.astype(C.dtype), B_np.T.astype(C.dtype)), rtol=1e-3)


def main():
    reset_gdbinit(DEV_CONFIG)

    #time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)
    time_overhead, cycle_overhead = 0.0, 0

    with micro.Session(DEV_CONFIG) as sess:
        #run_micro_conv2d(sess, time_overhead, cycle_overhead)
        run_micro_matmul(sess, time_overhead, cycle_overhead)


if __name__ == "__main__":
    main()
