#pragma once

#include "Camera.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "World.h"
#include "FrameBuffer.h"

#include <thrust/device_ptr.h>

// Default tile size for progressive tiled rendering (RENDER_FINAL mode)
constexpr int DEFAULT_TILE_SIZE = 64;

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

    KernelInfo() {}
    ~KernelInfo();
    KernelInfo(cudaGraphicsResource_t resources, int nx, int ny, int samples, int max_depth, float fov);
    void set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void render(bool camera_moving, int pixelate = 1);
    void resize(int nx, int ny);

    // Tiled rendering support: map/unmap PBO separately, render individual tiles
    void map_pbo();
    void unmap_pbo();
    void render_tile(int tile_offset_x, int tile_offset_y, int tile_w, int tile_h, int spp);
};