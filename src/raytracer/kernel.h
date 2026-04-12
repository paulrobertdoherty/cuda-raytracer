#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "World.h"
#include "FrameBuffer.h"

#include <thrust/device_ptr.h>

#include <vector>

class Scene;

// Default tile size for progressive tiled rendering (RENDER_FINAL mode)
constexpr int DEFAULT_TILE_SIZE = 64;

// ---------------------------------------------------------------------------
// POD descriptors used to rebuild the CUDA world from host state. These are
// passed as a contiguous device buffer to build_world_from_desc; the kernel
// `new`s the appropriate Hittable/Material for each entry.
// ---------------------------------------------------------------------------

enum class DeviceObjectKind : int {
    Sphere = 0,
    Triangle = 1,
    Rect = 2,
    Mesh = 3
};

enum class DeviceMaterialKind : int {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    Emissive = 3
};

struct DeviceObjectDesc {
    DeviceObjectKind kind;
    DeviceMaterialKind material;
    glm::vec3 albedo;
    float fuzz;
    float ior;
    glm::vec3 emission;
    int is_light; // 0/1

    // Sphere
    glm::vec3 center;
    float radius;

    // Triangle
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    // Rect
    glm::vec3 rect_Q;
    glm::vec3 rect_u;
    glm::vec3 rect_v;

    // Mesh: pointers into device buffers owned by KernelInfo
    glm::vec3* d_mesh_vertices;
    int* d_mesh_indices;
    int mesh_vcount;
    int mesh_icount;
    glm::vec3 mesh_translate;
    float mesh_scale;
    glm::vec3 mesh_aabb_min;
    glm::vec3 mesh_aabb_max;
};

struct KernelInfo {

    thrust::device_ptr<Camera*> d_camera;
    thrust::device_ptr<RandState> d_rand_state;

    thrust::device_ptr<World*> d_world;

    cudaGraphicsResource_t resources;
    CameraInfo camera_info;
    FrameBuffer* frame_buffer;

    int nx, ny;
    int samples;
    int max_depth;

    // Device-side buffers owned by this KernelInfo, keyed by mesh index in
    // the Scene. Freed in the destructor. Kept in sync across rebuilds so
    // untouched meshes don't get re-uploaded.
    std::vector<glm::vec3*> d_mesh_vertex_buffers;
    std::vector<int*> d_mesh_index_buffers;
    std::vector<int> d_mesh_vcounts;
    std::vector<int> d_mesh_icounts;

    KernelInfo() {}
    ~KernelInfo();
    KernelInfo(cudaGraphicsResource_t resources, int nx, int ny, int samples, int max_depth, float fov);
    void set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void render(bool camera_moving, int pixelate = 1);
    void resize(int nx, int ny);

    // Destroy and rebuild the device-side World from the given host scene.
    // Re-uploads mesh buffers as needed.
    void rebuild_world(const Scene& scene);

    // Tiled rendering support: map/unmap PBO separately, render individual tiles
    void map_pbo();
    void unmap_pbo();
    void render_tile(int tile_offset_x, int tile_offset_y, int tile_w, int tile_h, int spp);
};