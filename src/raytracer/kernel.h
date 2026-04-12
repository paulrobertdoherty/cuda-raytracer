#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "World.h"
#include "FrameBuffer.h"

#include <thrust/device_ptr.h>

#include <vector>

class Scene;
struct MeshBVHNode;

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
    Mesh = 3,
    Disc = 4
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

    // Disc
    glm::vec3 disc_center;
    glm::vec3 disc_normal;
    float     disc_radius;

    // Checker texture (Lambertian surfaces)
    int       use_checker;
    glm::vec3 checker_color1;
    glm::vec3 checker_color2;

    // Mesh: pointers into device buffers owned by KernelInfo
    glm::vec3* d_mesh_vertices;
    glm::vec2* d_mesh_uvs;       // may be nullptr
    int* d_mesh_indices;
    int mesh_vcount;
    int mesh_icount;
    glm::vec3 mesh_translate;
    float mesh_scale;
    glm::vec3 mesh_aabb_min;
    glm::vec3 mesh_aabb_max;

    // Mesh BVH: device pointers owned by KernelInfo
    MeshBVHNode* d_mesh_bvh_nodes;
    int          mesh_bvh_node_count;
    int*         d_mesh_reordered_tri_ids;
    int          mesh_tri_id_count;

    // Image texture for mesh (device pointer owned by KernelInfo)
    unsigned char* d_texture_pixels; // nullptr if no texture
    int            tex_width;
    int            tex_height;
    int            tex_channels;
    int            has_texture;      // 0/1
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

    // Per-mesh BVH node buffers (device pointers, one per scene mesh).
    std::vector<MeshBVHNode*> d_mesh_bvh_buffers;
    std::vector<int>          d_mesh_bvh_node_counts;
    std::vector<int*>         d_mesh_tri_id_buffers;
    std::vector<int>          d_mesh_tri_id_counts;

    // Per-mesh UV buffers (device pointers, one per scene mesh).
    std::vector<glm::vec2*> d_mesh_uv_buffers;

    // Per-texture pixel buffers (device pointers, one per scene texture).
    std::vector<unsigned char*> d_texture_pixel_buffers;
    std::vector<int>            d_texture_widths;
    std::vector<int>            d_texture_heights;
    std::vector<int>            d_texture_channels;

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