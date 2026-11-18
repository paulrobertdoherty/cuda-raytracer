#include "cuda_errors.h"
#include "raytracer/Material.h"

#include <cstdio>

__device__ bool all_passed;

__device__ bool expect(bool result, const char* description) {
    if (!result)
        printf("Failed: %s\n", description);
    return result;
}

__global__ void check_math() {
    AABB box(glm::vec3(-1.0f), glm::vec3(1.0f));
    Ray boundary(glm::vec3(1.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, -1.0f));
    Ray outside(glm::vec3(2.0f, 0.0f, 3.0f), boundary.direction);

    bool passed = expect(box.hit(boundary, 0.0f, 10.0f), "parallel ray on a slab boundary hits");
    passed &= expect(!box.hit(outside, 0.0f, 10.0f), "parallel ray outside the slab misses");
    passed &= expect(!box.hit(boundary, 0.0f, 1.0f), "hit beyond the ray interval is rejected");
    passed &= expect(fabsf(schlick_reflectance(1.0f, 1.5f) - 0.04f) < 1e-6f,
                     "glass reflects 4% of light at normal incidence");
    passed &= expect(fabsf(schlick_reflectance(0.0f, 1.5f) - 1.0f) < 1e-6f,
                     "glass reflects all light at grazing incidence");
    all_passed = passed;
}

int main() {
    try {
        check_math<<<1, 1>>>();
        check_cuda_errors(cudaGetLastError());
        bool passed = false;
        check_cuda_errors(cudaMemcpyFromSymbol(&passed, all_passed, sizeof(passed)));
        return passed ? 0 : 1;
    } catch (const CudaError& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
