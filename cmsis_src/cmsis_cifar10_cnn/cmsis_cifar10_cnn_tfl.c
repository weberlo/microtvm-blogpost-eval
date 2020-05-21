#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

#include <arm_math.h>
#include <arm_nnfunctions.h>

#include "nn.h"

static uint8_t mean[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM] = MEAN_DATA;

static q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static int32_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static q31_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IN_CH*CONV3_KER_DIM*CONV3_KER_DIM*CONV3_OUT_CH] = CONV3_WT;
static q31_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static q31_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS_SHIFTED;

static int32_t output_shift[64] = {CONV1_OUT_RSHIFT};
static int32_t output_mult[64] = {1};

void mean_subtract(q7_t* image_data) {
  for(int i=0; i<DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM; i++) {
    image_data[i] = (q7_t)__SSAT( ((int)(*((uint8_t*)&image_data[i]) - mean[i]) >> DATA_RSHIFT), 8);
  }
}

int32_t cifar10(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  DLTensor* input = (DLTensor*) arg_values[0].v_handle;   // NHWC
  DLTensor* output = (DLTensor*) arg_values[1].v_handle;

  int64_t input_h = input->shape[1];
  int64_t input_w = input->shape[2];
  int64_t input_ch = input->shape[3];


  // NOTE: the op that needs the largest col_buffer is the last average pool (2
  // * 64 * 5 * 5 * sizeof(q15_t))
  void* col_buffer = TVMBackendAllocWorkspace(1, 0, (uint64_t) 6400, 2, 8);
  if (col_buffer == NULL) {
    return -1;
  }
  void* buffer1 = TVMBackendAllocWorkspace(1, 0, (uint64_t) 32768, 2, 8);
  if (buffer1 == NULL) {
    return -1;
  }
  void* buffer2 = TVMBackendAllocWorkspace(1, 0, (uint64_t) 8192, 2, 8);
  if (buffer2 == NULL) {
    return -1;
  }

  for (int i = 0; i < 64; i++) {
    output_shift[i] = -CONV1_OUT_RSHIFT;
    output_mult[i] = 0x7fffffff;
  }

  mean_subtract(input->data);
  arm_status st = arm_convolve_s8((q7_t*) input->data, input_w, input_h, input_ch, 1  /* input_batches */,
                                  conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_KER_DIM,
                                  CONV1_PAD, CONV1_PAD, CONV1_STRIDE, CONV1_STRIDE,
//                                  conv1_bias, (q7_t*) output->data, output_shift, output_mult,
                                  conv1_bias, buffer1, output_shift, output_mult,
                                  0  /* output_offset */, 0  /* input_offset */, -128, 127,
                                  CONV1_OUT_DIM, CONV1_OUT_DIM, (q15_t*) col_buffer);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  st = arm_max_pool_s8(POOL1_IN_DIM, POOL1_IN_DIM, POOL1_IN_DIM, POOL1_OUT_DIM, POOL1_STRIDE, POOL1_STRIDE, POOL1_KER_DIM, POOL1_KER_DIM, POOL1_PAD, POOL1_PAD, -128, 127, POOL1_IN_CH, (q7_t*) buffer1, (q15_t*) col_buffer, output->data);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  arm_relu_q7(output->data, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  for (int i = 0; i < 64; i++) {
    output_shift[i] = -CONV1_OUT_RSHIFT;
    output_mult[i] = 0x7fffffff;
  }
  st = arm_convolve_s8(buffer2, CONV2_IN_DIM, CONV2_IN_DIM, CONV2_IN_CH, 1  /* input_batches */,
                       conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_KER_DIM,
                       CONV2_PAD, CONV2_PAD, CONV2_STRIDE, CONV2_STRIDE,
                       conv2_bias, buffer1, output_shift, output_mult,
                       0  /* output_offset */, 0  /* input_offset */,
                       -128, 127, CONV2_OUT_DIM, CONV2_OUT_DIM, (q15_t*) col_buffer);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  arm_relu_q7(buffer1, RELU2_OUT_DIM*RELU2_OUT_DIM*RELU2_OUT_CH);
  arm_avgpool_s8(POOL2_IN_DIM, POOL2_IN_DIM, POOL2_OUT_DIM, POOL2_OUT_DIM, POOL2_STRIDE, POOL2_STRIDE,
                 POOL2_KER_DIM, POOL2_KER_DIM, POOL2_PAD, POOL2_PAD, -128, 127, POOL2_IN_CH, buffer1, (q15_t*) col_buffer, buffer2);

  for (int i = 0; i < 64; i++) {
    output_shift[i] = -CONV1_OUT_RSHIFT;
    output_mult[i] = 0x7fffffff;
  }
  st = arm_convolve_s8(buffer2, CONV3_IN_DIM, CONV3_IN_DIM, CONV3_IN_CH, 1  /* input_batches */,
                       conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_KER_DIM,
                       CONV3_PAD, CONV3_PAD, CONV3_STRIDE, CONV3_STRIDE,
                       conv3_bias, buffer1, output_shift, output_mult,
                       0  /* output_offset */, 0  /* input_offset */,
                       -128, 127, CONV3_OUT_DIM, CONV3_OUT_DIM, col_buffer);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  arm_relu_q7(buffer1, RELU3_OUT_DIM*RELU3_OUT_DIM*RELU3_OUT_CH);

  st = arm_avgpool_s8(POOL3_IN_DIM, POOL3_IN_DIM, POOL3_OUT_DIM, POOL3_OUT_DIM, POOL3_STRIDE, POOL3_STRIDE,
                      POOL2_KER_DIM, POOL2_KER_DIM, POOL2_PAD, POOL2_PAD, -128, 127, POOL2_IN_CH, buffer1, col_buffer, buffer2);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  st = arm_fully_connected_s8(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, 1  /* nb_batches */,
                              0  /* input_offset */, 0  /* filter_offset */, 0x7fffffff  /* out_mult */,  -IP1_OUT_RSHIFT  /* out_shift */,
                              0  /* output_offset */, ip1_bias, output->data, -128, 127, (q15_t*)col_buffer);
  if (st != ARM_MATH_SUCCESS) {
    return -1;
  }

  if (TVMBackendFreeWorkspace(1, 0, buffer2) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, buffer1) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, col_buffer) != 0) {
    return -1;
  }
  return 0;
}




/*   return 0; */
/* } */
