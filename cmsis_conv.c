#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

#include <arm_math.h>
#include <arm_nnfunctions.h>

#define IN_DIM 16
#define IN_CH 32
#define KER_DIM 5
#define PAD 2
#define STRIDE 1
#define OUT_CH 32
#define OUT_DIM 16

#define BIAS_LSHIFT 0
#define OUT_RSHIFT 9

#define BIAS {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
static q7_t conv2_bias[OUT_CH] = BIAS;

int32_t arm_conv_wrapper(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  void* data_handle = (((TVMValue*)arg_values)[0].v_handle);
  void* kernel_handle = (((TVMValue*)arg_values)[1].v_handle);
  void* output_handle = (((TVMValue*)arg_values)[2].v_handle);

  int32_t dev_type = (((TVMArray*)data_handle)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)data_handle)[0].ctx.device_id);

  int8_t* data = (int8_t*)(((TVMArray*)data_handle)[0].data);
  int64_t* data_shape = (int64_t*)(((TVMArray*)data_handle)[0].shape);
  int64_t* data_strides = (int64_t*)(((TVMArray*)data_handle)[0].strides);
  int8_t* kernel = (int8_t*)(((TVMArray*)kernel_handle)[0].data);
  int64_t* kernel_shape = (int64_t*)(((TVMArray*)kernel_handle)[0].shape);
  int64_t* kernel_strides = (int64_t*)(((TVMArray*)kernel_handle)[0].strides);
  int8_t* output = (int8_t*)(((TVMArray*)output_handle)[0].data);
  int64_t* output_shape = (int64_t*)(((TVMArray*)output_handle)[0].shape);
  int64_t* output_strides = (int64_t*)(((TVMArray*)output_handle)[0].strides);

  //void* buffer1 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(32768 * sizeof(int8_t)), 2, 8);
  //void* buffer2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(8192 * sizeof(int8_t)), 2, 8);

  uint16_t dim_im_in = data_shape[1];
  uint16_t in_ch = data_shape[3];
  uint16_t dim_im_out = output_shape[1];
  uint16_t out_ch = output_shape[3];
  uint16_t kernel_size = kernel_shape[2];
  //uint16_t dim_im_in = IN_DIM;
  //uint16_t in_ch = IN_CH;
  //uint16_t dim_im_out = OUT_DIM;
  //uint16_t out_ch = OUT_CH;
  //uint16_t kernel_size = KER_DIM;

  //void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(6400 * sizeof(int8_t)), 2, 8);
  uint64_t ws_size = 2 * in_ch * kernel_size * kernel_size * sizeof(int16_t);
  void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, ws_size, 2, 8);
  if (col_buffer == NULL) {
    return UTVM_ERR_ALLOC_TOO_LARGE;
  }

  arm_convolve_HWC_q7_fast(
    /* Im_in      */  data,
    /* dim_im_in  */  dim_im_in,
    /* ch_im_in   */  in_ch,
    /* wt         */  kernel,
    /* ch_im_out  */  out_ch,
    /* dim_kernel */  kernel_size,
    /* padding    */  PAD,
    /* stride     */  STRIDE,
    /* bias       */  conv2_bias,
    /* bias_shift */  BIAS_LSHIFT,
    /* out_shift  */  OUT_RSHIFT,
    /* Im_out     */  output,
    /* dim_im_out */  dim_im_out,
    /* bufferA    */  (q15_t*)col_buffer,
    /* bufferB    */  NULL);

  int32_t res = TVMBackendFreeWorkspace(1, dev_id, col_buffer);
  if (res != 0) {
    return res;
  }
  return 0;
}
