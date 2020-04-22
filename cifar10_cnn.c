#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
extern void* __tvm_module_ctx = NULL;

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_8x4x64_body_UNXKVHVV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t aa_pad[32];
  int16_t bb_pad[256];

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*4 + j*4], (int32_t*) &aa_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 64; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*4 + l*2]),
          *((int32_t*) &bb_pad[j*4 + l*2]),
          sum);
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_8x4x64_update_UNXKVHVV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t aa_pad[32];
  int16_t bb_pad[256];

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*4 + j*4], (int32_t*) &aa_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 64; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*4 + l*2]),
          *((int32_t*) &bb_pad[j*4 + l*2]),
          sum);
      }
      cc[i*C_stride + j] += sum;
    }
  }

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_8x4x64_reset_UNXKVHVV(int32_t *cc, int C_stride) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 64; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_16x4x32_body_ZIAJTWWA(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t aa_pad[64];
  int16_t bb_pad[128];

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*4 + j*4], (int32_t*) &aa_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 32; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*4 + l*2]),
          *((int32_t*) &bb_pad[j*4 + l*2]),
          sum);
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_16x4x32_update_ZIAJTWWA(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t aa_pad[64];
  int16_t bb_pad[128];

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad[i*4 + j*4], (int32_t*) &aa_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 4 / 4; j++) {
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);
    }
  }

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 32; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(
          *((int32_t*) &aa_pad[i*4 + l*2]),
          *((int32_t*) &bb_pad[j*4 + l*2]),
          sum);
      }
      cc[i*C_stride + j] += sum;
    }
  }

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_16x4x32_reset_ZIAJTWWA(int32_t *cc, int C_stride) {
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 32; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}
    #ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_cast_subtract_cast( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  uint8_t* placeholder = (uint8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int16_t* placeholder1 = (int16_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  int8_t* T_cast = (int8_t*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      if (ax3_inner < 3) {
        T_cast[((ax0_ax1_fused_ax2_fused * 3) + ax3_inner)] = ((int8_t)(((int16_t)placeholder[((ax0_ax1_fused_ax2_fused * 3) + ax3_inner)]) - placeholder1[((ax0_ax1_fused_ax2_fused * 3) + ax3_inner)]));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_right_shift_cast( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* placeholder1 = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  int32_t* placeholder2 = (int32_t*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  int8_t* T_cast = (int8_t*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* Conv2dOutput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)131072, 0, 32);
  if (Conv2dOutput == NULL) {
    return -1;
  }
  for (int32_t ff_outer = 0; ff_outer < 2; ++ff_outer) {
    for (int32_t yy_outer = 0; yy_outer < 16; ++yy_outer) {
      for (int32_t xx_outer = 0; xx_outer < 8; ++xx_outer) {
        for (int32_t ff_inner_init = 0; ff_inner_init < 16; ++ff_inner_init) {
          (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init)] = 0;
        }
        for (int32_t ff_inner_init1 = 0; ff_inner_init1 < 16; ++ff_inner_init1) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init1) + 32)] = 0;
        }
        for (int32_t ff_inner_init2 = 0; ff_inner_init2 < 16; ++ff_inner_init2) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init2) + 64)] = 0;
        }
        for (int32_t ff_inner_init3 = 0; ff_inner_init3 < 16; ++ff_inner_init3) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init3) + 96)] = 0;
        }
        for (int32_t ff_inner_init4 = 0; ff_inner_init4 < 16; ++ff_inner_init4) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init4) + 1024)] = 0;
        }
        for (int32_t ff_inner_init5 = 0; ff_inner_init5 < 16; ++ff_inner_init5) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init5) + 1056)] = 0;
        }
        for (int32_t ff_inner_init6 = 0; ff_inner_init6 < 16; ++ff_inner_init6) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init6) + 1088)] = 0;
        }
        for (int32_t ff_inner_init7 = 0; ff_inner_init7 < 16; ++ff_inner_init7) {
          (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner_init7) + 1120)] = 0;
        }
        for (int32_t rc = 0; rc < 3; ++rc) {
          for (int32_t ff_inner = 0; ff_inner < 16; ++ff_inner) {
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 198)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 96)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 192)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 288)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 384)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 102)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 480)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 99)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 576)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 672)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 768)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 864)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 6)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 960)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 3)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1056)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1152)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1248)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1344)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1440)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1536)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1632)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1728)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1824)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 1920)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 2016)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 2112)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 2208)])));
            (( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] = ((( int32_t*)Conv2dOutput)[((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner) + 2304)])));
          }
          for (int32_t ff_inner1 = 0; ff_inner1 < 16; ++ff_inner1) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 195)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner1)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 183)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 99)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 3)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner1) + 32)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner1) + 2304)])));
          }
          for (int32_t ff_inner2 = 0; ff_inner2 < 16; ++ff_inner2) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 192)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner2)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 183)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 180)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 84)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 12)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 108)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner2) + 64)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 204)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner2) + 2304)])));
          }
          for (int32_t ff_inner3 = 0; ff_inner3 < 16; ++ff_inner3) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 189)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner3)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 183)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 180)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 177)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 84)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 81)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 12)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 15)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 108)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 111)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 204)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner3) + 96)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 207)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner3) + 2304)])));
          }
          for (int32_t ff_inner4 = 0; ff_inner4 < 16; ++ff_inner4) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 102)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner4)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 99)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 6)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 3)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 186)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 282)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 285)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 288)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 291)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner4) + 1024)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 294)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner4) + 2304)])));
          }
          for (int32_t ff_inner5 = 0; ff_inner5 < 16; ++ff_inner5) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)(((1 <= yy_outer) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 99)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner5)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 3)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((1 <= xx_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 189)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)(((yy_outer < 15) && (1 <= xx_outer)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 285)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 288)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 291)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 294)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner5) + 1056)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 297)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner5) + 2304)])));
          }
          for (int32_t ff_inner6 = 0; ff_inner6 < 16; ++ff_inner6) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 96)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner6)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 84)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[(((yy_outer * 192) + (xx_outer * 12)) + rc)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 12)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 96)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 108)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 192)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 204)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 288)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 291)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 294)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 297)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner6) + 1088)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 300)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner6) + 2304)])));
          }
          for (int32_t ff_inner7 = 0; ff_inner7 < 16; ++ff_inner7) {
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 93)] : (int8_t)0)) * ((int32_t)placeholder1[(((rc * 32) + (ff_outer * 16)) + ff_inner7)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 90)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 96)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((1 <= yy_outer) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 87)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 192)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 84)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 288)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((1 <= yy_outer) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) - 81)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 384)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 3)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 480)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 6)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 576)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 9)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 672)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 12)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 768)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 15)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 864)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 99)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 960)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 102)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1056)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 105)]) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1152)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 108)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1248)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((xx_outer < 7) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 111)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1344)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 195)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1440)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 198)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1536)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 201)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1632)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 204)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1728)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 207)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1824)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 291)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 1920)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 294)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 2016)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)((yy_outer < 15) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 297)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 2112)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 300)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 2208)])));
            (( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] = ((( int32_t*)Conv2dOutput)[(((((yy_outer * 2048) + (xx_outer * 128)) + (ff_outer * 16)) + ff_inner7) + 1120)] + (((int32_t)(((yy_outer < 15) && (xx_outer < 7)) ? placeholder[((((yy_outer * 192) + (xx_outer * 12)) + rc) + 303)] : (int8_t)0)) * ((int32_t)placeholder1[((((rc * 32) + (ff_outer * 16)) + ff_inner7) + 2304)])));
          }
        }
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 32; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 32; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        T_cast[(((ax1 * 1024) + (ax2 * 32)) + ax3)] = ((int8_t)(((( int32_t*)Conv2dOutput)[(((ax1 * 1024) + (ax2 * 32)) + ax3)] + placeholder2[ax3]) >> 9));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, Conv2dOutput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_max_pool2d_nn_relu( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* T_relu = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)34848, 0, 8);
  if (pad_temp == NULL) {
    return -1;
  }
  void* tensor = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)8192, 0, 8);
  if (tensor == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 33; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 33; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        (( int8_t*)pad_temp)[(((ax1 * 1056) + (ax2 * 32)) + ax3)] = (((ax1 < 32) && (ax2 < 32)) ? placeholder[(((ax1 * 1024) + (ax2 * 32)) + ax3)] : (int8_t)-128);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 16; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 16; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 32; ++ax31) {
        (( int8_t*)tensor)[(((ax11 * 512) + (ax21 * 32)) + ax31)] = (int8_t)-128;
        for (int32_t rv = 0; rv < 3; ++rv) {
          for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
            int8_t _1 = (( int8_t*)tensor)[(((ax11 * 512) + (ax21 * 32)) + ax31)];
            int8_t _2 = (( int8_t*)pad_temp)[(((((ax11 * 2112) + (rv * 1056)) + (ax21 * 64)) + (rv1 * 32)) + ax31)];
            (( int8_t*)tensor)[(((ax11 * 512) + (ax21 * 32)) + ax31)] = ((_1) > (_2) ? (_1) : (_2));
          }
        }
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 16; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 16; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 32; ++ax32) {
        int8_t _3 = (( int8_t*)tensor)[(((ax12 * 512) + (ax22 * 32)) + ax32)];
        int8_t _4 = (int8_t)0;
        T_relu[(((ax12 * 512) + (ax22 * 32)) + ax32)] = ((_3) > (_4) ? (_3) : (_4));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_avg_pool2d( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int32_t* tensor = (int32_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)20736, 0, 32);
  if (pad_temp == NULL) {
    return -1;
  }
  void* tensor1 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4096, 0, 32);
  if (tensor1 == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 9; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 9; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
        (( int32_t*)pad_temp)[(((ax1 * 576) + (ax2 * 64)) + ax3)] = (((ax1 < 8) && (ax2 < 8)) ? ((int32_t)placeholder[(((ax1 * 512) + (ax2 * 64)) + ax3)]) : 0);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 4; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 4; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 64; ++ax31) {
        (( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 64)) + ax31)] = 0;
        for (int32_t rv = 0; rv < 3; ++rv) {
          for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
            (( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 64)) + ax31)] = ((( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 64)) + ax31)] + (( int32_t*)pad_temp)[(((((ax11 * 1152) + (rv * 576)) + (ax21 * 128)) + (rv1 * 64)) + ax31)]);
          }
        }
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 4; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 4; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 64; ++ax32) {
        tensor[(((ax12 * 256) + (ax22 * 64)) + ax32)] = ((( int32_t*)tensor1)[(((ax12 * 256) + (ax22 * 64)) + ax32)] / 9);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor1) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_batch_flatten( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* tensor = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax1_outer = 0; ax1_outer < 128; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      tensor[((ax1_outer * 8) + ax1_inner)] = placeholder[((ax1_outer * 8) + ax1_inner)];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_right_shift_cast_nn_relu( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* placeholder1 = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  int32_t* placeholder2 = (int32_t*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  int8_t* T_relu = (int8_t*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* padded_data = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4608, 0, 8);
  if (padded_data == NULL) {
    return -1;
  }
  void* conv2d = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16384, 0, 32);
  if (conv2d == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 12; ++i1) {
    for (int32_t i2 = 0; i2 < 12; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        (( int8_t*)padded_data)[(((i1 * 384) + (i2 * 32)) + i3)] = (((((2 <= i1) && (i1 < 10)) && (2 <= i2)) && (i2 < 10)) ? placeholder[((((i1 * 256) + (i2 * 32)) + i3) - 576)] : (int8_t)0);
      }
    }
  }
  for (int32_t yy = 0; yy < 8; ++yy) {
    (void)gemm_8x4x64_reset_UNXKVHVV(((int32_t *)conv2d + (yy * 512)), 64);
    for (int32_t ry = 0; ry < 5; ++ry) {
      for (int32_t rx = 0; rx < 5; ++rx) {
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + (((yy * 384) + (ry * 384)) + (rx * 32))), ((int8_t *)placeholder1 + ((ry * 10240) + (rx * 2048))), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 4)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 4)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 8)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 8)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 12)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 12)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 16)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 16)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 20)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 20)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 24)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 24)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
        (void)gemm_8x4x64_update_UNXKVHVV(((int8_t *)padded_data + ((((yy * 384) + (ry * 384)) + (rx * 32)) + 28)), ((int8_t *)placeholder1 + (((ry * 10240) + (rx * 2048)) + 28)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 64);
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
        int8_t _1 = (int8_t)(((( int32_t*)conv2d)[(((ax1 * 512) + (ax2 * 64)) + ax3)] + placeholder2[ax3]) >> 9);
        int8_t _2 = (int8_t)0;
        T_relu[(((ax1 * 512) + (ax2 * 64)) + ax3)] = ((_1) > (_2) ? (_1) : (_2));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, conv2d) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, padded_data) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_avg_pool2d_1( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int32_t* tensor = (int32_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)36992, 0, 32);
  if (pad_temp == NULL) {
    return -1;
  }
  void* tensor1 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)8192, 0, 32);
  if (tensor1 == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 17; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 17; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        (( int32_t*)pad_temp)[(((ax1 * 544) + (ax2 * 32)) + ax3)] = (((ax1 < 16) && (ax2 < 16)) ? ((int32_t)placeholder[(((ax1 * 512) + (ax2 * 32)) + ax3)]) : 0);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 8; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 8; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 32; ++ax31) {
        (( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 32)) + ax31)] = 0;
        for (int32_t rv = 0; rv < 3; ++rv) {
          for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
            (( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 32)) + ax31)] = ((( int32_t*)tensor1)[(((ax11 * 256) + (ax21 * 32)) + ax31)] + (( int32_t*)pad_temp)[(((((ax11 * 1088) + (rv * 544)) + (ax21 * 64)) + (rv1 * 32)) + ax31)]);
          }
        }
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 8; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 8; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 32; ++ax32) {
        tensor[(((ax12 * 256) + (ax22 * 32)) + ax32)] = ((( int32_t*)tensor1)[(((ax12 * 256) + (ax22 * 32)) + ax32)] / 9);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, tensor1) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_add_right_shift_cast( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* placeholder1 = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  int32_t* placeholder2 = (int32_t*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  int8_t* T_cast = (int8_t*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
   int32_t T_dense[10];
  for (int32_t j = 0; j < 10; ++j) {
    T_dense[j] = 0;
    for (int32_t k = 0; k < 1024; ++k) {
      T_dense[j] = (T_dense[j] + (((int32_t)placeholder[k]) * ((int32_t)placeholder1[((j * 1024) + k)])));
    }
  }
  for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
    T_dense[ax1] = (T_dense[ax1] + placeholder2[ax1]);
  }
  for (int32_t ax11 = 0; ax11 < 10; ++ax11) {
    T_dense[ax11] = (T_dense[ax11] >> 5);
  }
  for (int32_t ax12 = 0; ax12 < 10; ++ax12) {
    T_cast[ax12] = ((int8_t)T_dense[ax12]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_right_shift_cast_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  int8_t* placeholder = (int8_t*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  int8_t* placeholder1 = (int8_t*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  int32_t* placeholder2 = (int32_t*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  int8_t* T_relu = (int8_t*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  void* padded_data = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)12800, 0, 8);
  if (padded_data == NULL) {
    return -1;
  }
  void* conv2d = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)32768, 0, 32);
  if (conv2d == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 20; ++i1) {
    for (int32_t i2 = 0; i2 < 20; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        (( int8_t*)padded_data)[(((i1 * 640) + (i2 * 32)) + i3)] = (((((2 <= i1) && (i1 < 18)) && (2 <= i2)) && (i2 < 18)) ? placeholder[((((i1 * 512) + (i2 * 32)) + i3) - 1088)] : (int8_t)0);
      }
    }
  }
  for (int32_t yy_init = 0; yy_init < 16; ++yy_init) {
    (void)gemm_16x4x32_reset_ZIAJTWWA(((int32_t *)conv2d + (yy_init * 512)), 32);
  }
  for (int32_t ry = 0; ry < 5; ++ry) {
    for (int32_t rx = 0; rx < 5; ++rx) {
      for (int32_t yy = 0; yy < 16; ++yy) {
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + (((yy * 640) + (ry * 640)) + (rx * 32))), ((int8_t *)placeholder1 + ((ry * 5120) + (rx * 1024))), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 4)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 4)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 8)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 8)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 12)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 12)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 16)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 16)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 20)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 20)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 24)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 24)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
        (void)gemm_16x4x32_update_ZIAJTWWA(((int8_t *)padded_data + ((((yy * 640) + (ry * 640)) + (rx * 32)) + 28)), ((int8_t *)placeholder1 + (((ry * 5120) + (rx * 1024)) + 28)), ((int32_t *)conv2d + (yy * 512)), 32, 32, 32);
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        int8_t _1 = (int8_t)(((( int32_t*)conv2d)[(((ax1 * 512) + (ax2 * 32)) + ax3)] + placeholder2[ax3]) >> 9);
        int8_t _2 = (int8_t)0;
        T_relu[(((ax1 * 512) + (ax2 * 32)) + ax3)] = ((_1) > (_2) ? (_1) : (_2));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, conv2d) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, padded_data) != 0) {
    return -1;
  }
  return 0;
}
