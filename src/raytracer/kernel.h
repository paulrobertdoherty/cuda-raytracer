#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "World.h"

#include <thrust/device_ptr.h>

#include <cstdint>
#include <memory>
#include <vector>

class Scene;
class BVHNode;
struct SceneObjectSlot;
struct SceneObject;
struct DeviceObjectDesc;
struct UploadedMesh;
struct UploadedImage;

// Default tile size for progressive final renders.
constexpr int DEFAULT_TILE_SIZE = 64;

struct KernelInfo {

    thrust::device_ptr<Camera*> d_camera;
    thrust::device_ptr<RandState> d_rand_state;
    // High-water mark for d_rand_state. Resize only reallocates+reinits when
    // growing past this; shrinks keep the buffer (pixels reuse whatever
    // Philox state happens to live at index j*width+i, which is fine for
    // stochastic path tracing).
    int d_rand_state_capacity = 0;

    thrust::device_ptr<World*> d_world;

    CameraInfo camera_info;

    int nx, ny;
    int samples;
    int max_depth;
    int seed;

    // Headless rendering support
    bool headless = false;
    uint32_t* d_headless_buffer = nullptr;

    KernelInfo() {}
    ~KernelInfo();
    KernelInfo(const KernelInfo&) = delete;
    KernelInfo& operator=(const KernelInfo&) = delete;
    KernelInfo(cudaGraphicsResource_t resources, int nx, int ny, int samples, int max_depth, float fov, int seed = 1984);
    // Headless constructor — no OpenGL, allocates its own device buffer
    KernelInfo(int nx, int ny, int samples, int max_depth, float fov, int seed = 1984);
    void set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void render(bool camera_moving, int pixelate = 1);
    void resize(int nx, int ny);

    // Repopulate the device-side World and rebuild its BVH from the host scene.
    // Re-uploads mesh buffers as needed.
    void rebuild_world(const Scene& scene);
    // Refit the edited leaf when it can reuse its assets. A changed asset
    // requires a rebuild so every object using it receives the new view.
    void update_object(const Scene& scene, int scene_index);

    // Headless: render the full image and return RGBA pixel data
    void render_to_buffer(std::vector<uint8_t>& output_rgba);

private:
    // Quad owns the GL buffer and controls when CUDA can access it.
    friend class Quad;
    void map_pbo();
    void unmap_pbo();
    void render_tile(int tile_offset_x, int tile_offset_y, int tile_w, int tile_h, int spp);
    cudaGraphicsResource_t resources = nullptr; // Borrowed from the owning Quad.
    uint32_t* d_mapped_pixels = nullptr; // Valid between map_pbo() and unmap_pbo().

    // d_slots owns a single CUDA allocation. The remaining pointers view
    // per-object descriptors, BVH nodes and three arrays of geometry pointers.
    DeviceObjectDesc* d_desc_buffer = nullptr;
    SceneObjectSlot* d_slots = nullptr;
    BVHNode* d_bvh_nodes = nullptr;
    Hittable** d_object_ptr_buffer = nullptr;
    Hittable** d_sorted_objects = nullptr;
    Hittable** d_lights = nullptr;
    size_t d_scene_capacity = 0;

    // Rebuilds stage a second cache. Both caches share the unchanged assets
    // until the new device world is ready and the previous cache is released.
    struct AssetCache {
        std::vector<std::shared_ptr<const UploadedMesh>> meshes;
        std::vector<std::shared_ptr<const UploadedImage>> images;

        AssetCache() = default;
        AssetCache(const Scene& scene, const AssetCache& current);
        bool matches_object_assets(const Scene& scene, const SceneObject& object) const;
        bool describe_object(const Scene& scene, const SceneObject& object, DeviceObjectDesc& desc) const;
    };
    AssetCache assets;
    std::vector<int> descriptor_indices;  // -1 for mesh objects omitted from the world
};
