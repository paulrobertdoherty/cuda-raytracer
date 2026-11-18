#include "cuda_errors.h"

#include "cuda_runtime.h"
#include <sstream>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::ostringstream oss;
        oss << "CUDA error " << static_cast<unsigned int>(result)
            << " (" << cudaGetErrorString(result) << ") at "
            << file << ":" << line << " '" << func << "'";
        // Keep the context alive while resource owners unwind to main().
        throw CudaError(result, oss.str());
    }
}
