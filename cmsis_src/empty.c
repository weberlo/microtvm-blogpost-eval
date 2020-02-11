#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

int32_t empty(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  return 0;
}
