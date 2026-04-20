#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <string>

// Thrown by check_cuda on any non-success CUDA status. main() catches this
// and returns a non-zero exit code after destructors run, instead of
// exit(99) which skips cleanup and leaves GL/CUDA state in an unclean shape.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, std::string msg)
        : std::runtime_error(std::move(msg)), code(code) {}
    cudaError_t code;
};

#define check_cuda_errors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);