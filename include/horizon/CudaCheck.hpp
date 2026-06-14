#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace horizon {

inline void throwOnCudaError(cudaError_t error, const char* expression, const char* file, int line) {
  if (error == cudaSuccess) {
    return;
  }

  throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                           " while evaluating `" + expression + "`: " +
                           cudaGetErrorString(error));
}

}  // namespace horizon

#define HORIZON_CUDA_CHECK(expr) ::horizon::throwOnCudaError((expr), #expr, __FILE__, __LINE__)
