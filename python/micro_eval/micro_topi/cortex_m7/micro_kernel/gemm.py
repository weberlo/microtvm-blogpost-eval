import random
import string

import tvm

##########################
# MxKxN MatMul Intrinsic #
##########################

# NOTE this is transposed matmul (A * B^T)
def intrin_gemm_MxKxN(M, K, N, in_dtype, out_dtype):
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = ''.join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    if isinstance(M, tvm.expr.IntImm):
        M = M.value
    if isinstance(K, tvm.expr.IntImm):
        K = K.value
    if isinstance(N, tvm.expr.IntImm):
        N = N.value
    assert K % 4 == 0
    # TODO support more dtypes?
    assert in_dtype == 'int8'
    assert out_dtype == 'int32'
    A = tvm.placeholder((M, K), name='a', dtype=in_dtype)
    B = tvm.placeholder((N, K), name='b', dtype=in_dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda i, j: tvm.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k), name='c')
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
        def _reduce_update():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_update_{uniq_id}",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    cc.access_ptr("w"),
                                    aa.strides[0],
                                    bb.strides[0],
                                    cc.strides[0]))
            return ib.get()
        def _reduce_reset():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_reset_{uniq_id}",
                                    cc.access_ptr("w"),
                                    cc.strides[0]))
            return ib.get()
        def _body():
            ib = tvm.ir_builder.create()
            # # NOTE we need the reset in the body for cases where the buffer
            # # we're accumulating into is uninitialized (e.g., if it's the
            # # result of a workspace allocation, because there are no guarantees
            # # on the contents).
            # ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_reset",
            #                         cc.access_ptr("w"),
            #                         cc.strides[0]))
            # ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_update",
            #                         aa.access_ptr("r"),
            #                         bb.access_ptr("r"),
            #                         cc.access_ptr("w"),
            #                         aa.strides[0],
            #                         bb.strides[0],
            #                         cc.strides[0]))
            ib.emit(tvm.call_extern("int32", f"gemm_{M}x{K}x{N}_body_{uniq_id}",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    cc.access_ptr("w"),
                                    aa.strides[0],
                                    bb.strides[0],
                                    cc.strides[0]))
            return ib.get()
        return _body(), _reduce_reset(), _reduce_update()
    with tvm.build_config(offset_factor=1):
        intrin_decl = tvm.decl_tensor_intrin(
            C.op, intrin_func, binds={A: A_buf, B: B_buf, C: C_buf})
        return intrin_decl, uniq_id


def gemm_MxKxN_impl(M, K, N, uniq_id):
    # TODO are there any SIMD tricks to zero out arrays quickly?
    aa_pad_size = M * K
    bb_pad_size = N * K
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = f"""
#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_body_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t aa_pad[{aa_pad_size}];
  int16_t bb_pad[{bb_pad_size}];

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*{K} + j*4], (int32_t*) &aa_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {N}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*{K} + j*4], (int32_t*) &bb_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K} / 2; l++) {{
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*{K} + l*2]),
          *((int32_t*) &bb_pad[j*{K} + l*2]),
          sum);
      }}
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }}
  }}

  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_update_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t aa_pad[{aa_pad_size}];
  int16_t bb_pad[{bb_pad_size}];

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*{K} + j*4], (int32_t*) &aa_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {N}; i++) {{
    for (int j = 0; j < {K} / 4; j++) {{
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*{K} + j*4], (int32_t*) &bb_pad[i*{K} + j*4 + 2]);
    }}
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K} / 2; l++) {{
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*{K} + l*2]),
          *((int32_t*) &bb_pad[j*{K} + l*2]),
          sum);
      }}
      cc[i*C_stride + j] += sum;
    }}
  }}

  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_{M}x{K}x{N}_reset_{uniq_id}(int32_t *cc, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      cc[i*C_stride + j] = 0;
    }}
  }}
  return 0;
}}
    """
    return cc_code
