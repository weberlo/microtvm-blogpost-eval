#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

#include <arm_math.h>
#include <arm_nnfunctions.h>

// original params
// #define IP1_BIAS_LSHIFT 3
// #define IP1_OUT_RSHIFT 5

// use zeros to simplify comparison
#define IP1_BIAS_LSHIFT 0
#define IP1_OUT_RSHIFT 0

int32_t arm_dense_wrapper(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  void* data_handle = ((TVMValue*)arg_values)[0].v_handle;
  void* weight_handle = ((TVMValue*)arg_values)[1].v_handle;
  void* bias_handle = ((TVMValue*)arg_values)[2].v_handle;
  void* output_handle = ((TVMValue*)arg_values)[3].v_handle;

  int32_t dev_type = (((TVMArray*)data_handle)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)data_handle)[0].ctx.device_id);

  int8_t* data = (int8_t*)(((TVMArray*)data_handle)[0].data);
  int64_t* data_shape = (int64_t*)(((TVMArray*)data_handle)[0].shape);
  int64_t* data_strides = (int64_t*)(((TVMArray*)data_handle)[0].strides);

  int8_t* weight = (int8_t*)(((TVMArray*)weight_handle)[0].data);
  int64_t* weight_shape = (int64_t*)(((TVMArray*)weight_handle)[0].shape);
  int64_t* weight_strides = (int64_t*)(((TVMArray*)weight_handle)[0].strides);

  int8_t* bias = (int8_t*)(((TVMArray*)bias_handle)[0].data);
  int64_t* bias_shape = (int64_t*)(((TVMArray*)bias_handle)[0].shape);
  int64_t* bias_strides = (int64_t*)(((TVMArray*)bias_handle)[0].strides);

  int8_t* output = (int8_t*)(((TVMArray*)output_handle)[0].data);
  int64_t* output_shape = (int64_t*)(((TVMArray*)output_handle)[0].shape);
  int64_t* output_strides = (int64_t*)(((TVMArray*)output_handle)[0].strides);

  // TODO calculate ws size
  //void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, 6400, 2, 8);
  void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, data_shape[3] * 100, 2, 8);
  if (col_buffer == NULL) {
    return UTVM_ERR_WS_OUT_OF_SPACE;
  }

  arm_fully_connected_q7_opt(
      data,
      weight,
      weight_shape[1],
      weight_shape[0],
      IP1_BIAS_LSHIFT,
      IP1_OUT_RSHIFT,
      bias,
      output,
      (q15_t*) col_buffer
      );

  int32_t res = TVMBackendFreeWorkspace(1, dev_id, col_buffer);
  if (res != 0) {
    return res;
  }
  return 0;
}
