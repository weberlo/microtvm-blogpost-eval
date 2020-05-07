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

int32_t arm_rgb_conv2d_wrapper(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  DLTensor* data_handle = arg_values[0].v_handle;
  DLTensor* kernel_handle = arg_values[1].v_handle;
  DLTensor* bias_handle = arg_values[2].v_handle;
  DLTensor* metadata_handle = arg_values[3].v_handle;
  DLTensor* output_handle = arg_values[4].v_handle;

  int32_t dev_type = data_handle->ctx.device_type;
  int32_t dev_id = data_handle->ctx.device_id;

  int8_t* data = (int8_t*) data_handle->data;
  int64_t* data_shape = data_handle->shape;
  int64_t* data_strides = data_handle->strides;

  int8_t* kernel = (int8_t*) kernel_handle->data;
  int64_t* kernel_shape = kernel_handle->shape;
  int64_t* kernel_strides = kernel_handle->strides;

  int8_t* bias = (int8_t*) bias_handle->data;
  int64_t* bias_shape = bias_handle->shape;
  int64_t* bias_strides = bias_handle->strides;

  uint16_t* metadata_data = (uint16_t*) metadata_handle->data;

  int8_t* output = (int8_t*) output_handle->data;
  int64_t* output_shape = output_handle->shape;
  int64_t* output_strides = output_handle->strides;

  uint16_t padding = metadata_data[0];
  uint16_t stride = metadata_data[1];
  uint16_t bias_shift = metadata_data[2];
  uint16_t out_shift = metadata_data[3];

  uint16_t dim_im_in = data_shape[1];
  uint16_t in_ch = data_shape[3];
  uint16_t dim_im_out = output_shape[1];
  uint16_t out_ch = output_shape[3];
  uint16_t kernel_size = kernel_shape[2];

  void* col_buffer = TVMBackendAllocWorkspace(1, 0, ((uint64_t) 2) * in_ch * kernel_size * kernel_size * sizeof(q15_t), 2, 8);
  if (col_buffer == NULL) {
    return -1;
  }

  int32_t res = arm_convolve_HWC_q7_RGB(
    /* Im_in      */  data,
    /* dim_im_in  */  dim_im_in,
    /* ch_im_in   */  in_ch,
    /* wt         */  kernel,
    /* ch_im_out  */  out_ch,
    /* dim_kernel */  kernel_size,
    /* padding    */  padding,
    /* stride     */  stride,
    /* bias       */  bias,
    /* bias_shift */  bias_shift,
    /* out_shift  */  out_shift,
    /* Im_out     */  output,
    /* dim_im_out */  dim_im_out,
    /* bufferA    */  (q15_t*)col_buffer,
    /* bufferB    */  NULL);
  if (res != 0) {
    return -1;
  }

  if (TVMBackendFreeWorkspace(1, 0, col_buffer) != 0) {
    return -1;
  }
  return 0;
}
