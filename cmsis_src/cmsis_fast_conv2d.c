#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

#include <arm_math.h>
#include <arm_nnfunctions.h>

#define PAD 2
#define STRIDE 1

#define BIAS_LSHIFT 0
#define OUT_RSHIFT 9

int32_t arm_fast_conv2d_wrapper(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  void* data_handle = (((TVMValue*)arg_values)[0].v_handle);
  void* kernel_handle = (((TVMValue*)arg_values)[1].v_handle);
  void* bias_handle = (((TVMValue*)arg_values)[2].v_handle);
  void* metadata_handle = (((TVMValue*)arg_values)[3].v_handle);
  void* output_handle = (((TVMValue*)arg_values)[4].v_handle);

  int32_t dev_type = (((DLTensor*)data_handle)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)data_handle)[0].ctx.device_id);

  int8_t* data = (int8_t*)(((DLTensor*)data_handle)[0].data);
  int64_t* data_shape = (int64_t*)(((DLTensor*)data_handle)[0].shape);
  int64_t* data_strides = (int64_t*)(((DLTensor*)data_handle)[0].strides);

  int8_t* kernel = (int8_t*)(((DLTensor*)kernel_handle)[0].data);
  int64_t* kernel_shape = (int64_t*)(((DLTensor*)kernel_handle)[0].shape);
  int64_t* kernel_strides = (int64_t*)(((DLTensor*)kernel_handle)[0].strides);

  int8_t* bias = (int8_t*)(((DLTensor*)bias_handle)[0].data);
  int64_t* bias_shape = (int64_t*)(((DLTensor*)bias_handle)[0].shape);
  int64_t* bias_strides = (int64_t*)(((DLTensor*)bias_handle)[0].strides);

  // HACK: to pass parameters (e.g., padding and stride) to CMSIS-NN, we include a tensor that
  // encodes these values
  int16_t* metadata = (int16_t*)(((DLTensor*)metadata_handle)[0].data);
  int64_t* metadata_shape = (int64_t*)(((DLTensor*)metadata_handle)[0].shape);
  int64_t* metadata_strides = (int64_t*)(((DLTensor*)metadata_handle)[0].strides);
  if (metadata_shape[0] != 4) {
    // TODO set an error code
    return -1;
  }
  uint16_t padding = metadata[0];
  uint16_t stride = metadata[1];
  uint16_t bias_left_shift = metadata[2];
  uint16_t out_right_shift = metadata[3];

  int8_t* output = (int8_t*)(((DLTensor*)output_handle)[0].data);
  int64_t* output_shape = (int64_t*)(((DLTensor*)output_handle)[0].shape);
  int64_t* output_strides = (int64_t*)(((DLTensor*)output_handle)[0].strides);

  uint16_t dim_im_in = data_shape[1];
  uint16_t in_ch = data_shape[3];
  uint16_t dim_im_out = output_shape[1];
  uint16_t out_ch = output_shape[3];
  uint16_t kernel_size = kernel_shape[2];

  // void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, (uint64_t) 6400, 2, 8);
  void* col_buffer = TVMBackendAllocWorkspace(
    1, dev_id, (uint64_t) (2 * in_ch * kernel_size * kernel_size * sizeof(q15_t)), 2, 8);
  if (col_buffer == NULL) {
    return -1;
  }

  int32_t res = arm_convolve_HWC_q7_fast(
    /* Im_in      */  data,
    /* dim_im_in  */  dim_im_in,
    /* ch_im_in   */  in_ch,
    /* wt         */  kernel,
    /* ch_im_out  */  out_ch,
    /* dim_kernel */  kernel_size,
    /* padding    */  padding,
    /* stride     */  stride,
    /* bias       */  bias,
    /* bias_shift */  bias_left_shift,
    /* out_shift  */  out_right_shift,
    /* Im_out     */  output,
    /* dim_im_out */  dim_im_out,
    /* bufferA    */  (q15_t*)col_buffer,
    /* bufferB    */  NULL);
  if (res != 0) {
    return -1;
  }

  if (TVMBackendFreeWorkspace(1, (uint64_t) dev_id, col_buffer) != 0) {
    return -1;
  }
  return 0;
}
