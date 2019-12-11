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
N, M, L = 32, 32, 32
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
A_SHAPE = (N, L)
B_SHAPE = (M, L)
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


# NOTE this is transposed (i.e., A * B^T)
def matmul_arm_micro_template(A, B, out_dtype):
    A_typ, A_shape, A_dtype = A
    B_typ, B_shape, B_dtype = B
    assert A_typ == 'TENSOR'
    assert B_typ == 'TENSOR'
    assert A_shape[1] == B_shape[1]
    N, M, L = A_shape[0], B_shape[0], A_shape[1]

    # Algorithm
    k = tvm.reduce_axis((0, A_shape[1]), 'k')
    A = tvm.placeholder(A_shape, name='A', dtype=A_dtype)
    B = tvm.placeholder(B_shape, name='B', dtype=B_dtype)
    C = tvm.compute(
               (N, M),
               lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k),
               name='C')

    # Default schedule
    sched = tvm.create_schedule(C.op)

    factor = 16
    x, y = C.op.axis
    z, = C.op.reduce_axis
    yo, yi = sched[C].split(y, factor=factor)
    sched[C].reorder(x, yo, yi, z)

    gemv = intrin_gemv(factor, A_shape[1], A.dtype, out_dtype)
    sched[C].tensorize(yi, gemv)
    sched[C].pragma(x, "import_c", gemv_impl())

    return sched, [A, B, C]


def intrin_gemv(m, l, in_dtype, out_dtype):
    a = tvm.placeholder((l,), name='a', dtype=in_dtype)
    b = tvm.placeholder((m, l), name='b', dtype=in_dtype)
    k = tvm.reduce_axis((0, l), name='k')
    c = tvm.compute((m,), lambda i: tvm.sum(a[k] * b[i, k], axis=k), name='c')
    Ab = tvm.decl_buffer(a.shape, a.dtype,
                         name="A",
                         offset_factor=1,
                         strides=[1])
    Bb = tvm.decl_buffer(b.shape, b.dtype,
                         name="B",
                         offset_factor=1,
                         strides=[tvm.var("s1"), 1])
    Cb = tvm.decl_buffer(c.shape, c.dtype,
                         name="C",
                         offset_factor=1,
                         strides=[1])
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(tvm.call_extern("int32", "gemv_update",
                                cc.access_ptr("w"),
                                aa.access_ptr("r"),
                                bb.access_ptr("r"),
                                m, l, bb.strides[0]))
        return ib.get()
    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def gemv_impl():
    cc_code = """
#ifdef __cplusplus
extern "C"
#endif
int32_t gemv_update(int8_t *cc, int8_t *aa, int8_t *bb, int m, int l, int stride) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < l; ++j) {
      cc[i] += aa[j] * bb[i * stride + j];
    }
  }
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
        c_mod = tvm.build(sched, [A, B, C], name="gemv")
    input(c_mod.get_source())

    from topi.util import get_const_tuple
    dtype = A.dtype
    A_np = np.random.randint(-30, 30, size=get_const_tuple(A.shape)).astype(dtype)
    B_np = np.random.randint(-30, 30, size=get_const_tuple(B.shape)).astype(dtype)

    micro_mod = create_micro_mod(c_mod, DEV_CONFIG, lib_include_paths=CMSIS_INCLUDE_PATHS)
    micro_func = micro_mod['gemv']
    ctx = tvm.micro_dev(0)

    A_tvm = tvm.nd.array(A_np, ctx=ctx)
    B_tvm = tvm.nd.array(B_np, ctx=ctx)
    C_tvm = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), ctx)

    batch_time, batch_cycles = benchmark_micro_func(sess, micro_func, [A_tvm, B_tvm, C_tvm], 1)
    batch_time -= time_overhead
    batch_cycles -= cycle_overhead

    tvm.testing.assert_allclose(C_tvm.asnumpy(), np.dot(A_np, B_np.T), rtol=1e-3)


def main():
    reset_gdbinit(DEV_CONFIG)

    time_overhead, cycle_overhead = get_comm_overhead(DEV_CONFIG)

    with micro.Session(DEV_CONFIG) as sess:
        #run_micro_conv2d(sess, time_overhead, cycle_overhead)
        run_micro_matmul(sess, time_overhead, cycle_overhead)


if __name__ == "__main__":
    main()
