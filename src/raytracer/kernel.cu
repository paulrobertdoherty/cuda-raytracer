
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include <glm/glm.hpp>
#include <curand_kernel.h>

#include "Window.h"
#include "cuda_errors.h"
#include "FrameBuffer.h"
#include "Ray.h"
#include "Sphere.h"
#include "Rectangle.h"
#include "Triangle.h"
#include "World.h"
#include "Camera.h"
#include "BVHNode.h"

#include <thrust/device_new.h>
#include <thrust/device_free.h>

#include "raytracer/kernel.h"

__global__ void raytrace(FrameBuffer fb, thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera, thrust::device_ptr<RandState> rand_state, int samples, int tile_offset_x, int tile_offset_y) {

	// X AND Y coordinates with tile offset
	int i = threadIdx.x + blockIdx.x * blockDim.x + tile_offset_x;
	int j = threadIdx.y + blockIdx.y * blockDim.y + tile_offset_y;

	// return early if we're outside of the frame buffer
	if ((i >= fb.width) || (j >= fb.height)) return;

	int pixel_idx = j * fb.width + i;

	RandState local_rand_state = rand_state[pixel_idx];

	glm::vec3 col = glm::vec3(0.0f, 0.0f, 0.0f);

	for (int s = 0; s < samples; s++) {
		// normalized screen coordinates
		float u = float(i + curand_uniform(&local_rand_state)) / float(fb.width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(fb.height);
		Ray r = ((Camera*)(*d_camera))->get_ray(u, v);
		col += fb.color(r, *world, &local_rand_state);
	}
	rand_state[pixel_idx] = local_rand_state;
	col /= float(samples);
	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);

	fb.writePixel(i, j, glm::vec4(col, 1.0f));
}

__global__ void raytrace_pixelated(FrameBuffer fb, thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera, thrust::device_ptr<RandState> rand_state, int pixelate) {

	int ci = threadIdx.x + blockIdx.x * blockDim.x;
	int cj = threadIdx.y + blockIdx.y * blockDim.y;

	int x0 = ci * pixelate;
	int y0 = cj * pixelate;

	if (x0 >= fb.width || y0 >= fb.height) return;

	// Reuse the rand state slot at the block's top-left pixel.
	int rand_idx = y0 * fb.width + x0;
	RandState local_rand_state = rand_state[rand_idx];

	float u = (float(x0) + 0.5f * float(pixelate)) / float(fb.width);
	float v = (float(y0) + 0.5f * float(pixelate)) / float(fb.height);
	Ray r = ((Camera*)(*d_camera))->get_ray(u, v);
	glm::vec3 col = fb.color(r, *world, &local_rand_state);
	rand_state[rand_idx] = local_rand_state;

	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);
	glm::vec4 out(col, 1.0f);

	int x_end = x0 + pixelate;
	int y_end = y0 + pixelate;
	if (x_end > fb.width) x_end = fb.width;
	if (y_end > fb.height) y_end = fb.height;
	for (int py = y0; py < y_end; py++) {
		for (int px = x0; px < x_end; px++) {
			fb.writePixel(px, py, out);
		}
	}
}

__global__ void create_world(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera, CameraInfo camera_info) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new World();
		World* device_world = *d_world;

		device_world->add(new Sphere(glm::vec3(0, 0, -1), 0.5f, new Lambertian(glm::vec3(0.8f, 0.3f, 0.3f))));
		//device_world->add(new Sphere(glm::vec3(0, -100.5, -1), 100.0f, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f))));
		device_world->add(new Sphere(glm::vec3(-1.01, 0, -1), 0.5f, new Dielectric(1.5f)));
		device_world->add(new Sphere(glm::vec3(-1, 10, -1), 0.5f, new Dielectric(1.5f)));
		device_world->add(new Sphere(glm::vec3(1, 0, -1), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f)));

		device_world->add(new Sphere(glm::vec3(0, -1000.5, 0), 1000.0f,
			new Lambertian(
				new CheckerTexture(
					glm::vec3(0.2f, 0.3f, 0.1f),
					glm::vec3(0.9f)
				)
			)
		));

		// Emissive sphere light
		Sphere* light_sphere = new Sphere(glm::vec3(0, 3, -1), 1.0f, new Emissive(glm::vec3(4.0f, 4.0f, 4.0f)));
		device_world->add(light_sphere);
		device_world->add_light(light_sphere);

		// Emissive area light rectangle
		Rect* light_rect = new Rect(
			glm::vec3(-1, 3, -2), glm::vec3(2, 0, 0), glm::vec3(0, 0, 2),
			new Emissive(glm::vec3(4.0f, 4.0f, 4.0f)));
		device_world->add(light_rect);
		device_world->add_light(light_rect);

		// Blue triangle
		device_world->add(new Triangle(
			glm::vec3(-2, 0, -2), glm::vec3(-1, 2, -2), glm::vec3(0, 0, -2),
			new Lambertian(glm::vec3(0.1f, 0.2f, 0.8f))));

		curandState rand_state;
		curand_init(1984, 0, 0, &rand_state);
		device_world->build_bvh(&rand_state);

		*d_camera = camera_info.construct_camera();
	}
}

__global__ void render_init(int width, int height, thrust::device_ptr<RandState> rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state.get()[pixel_index]);
}


KernelInfo::KernelInfo(cudaGraphicsResource_t resources, int nx, int ny, int samples, int max_depth, float fov) {
	this->resources = resources;
	this->nx = nx;
	this->ny = ny;
	this->samples = samples;
	this->max_depth = max_depth;

	this->frame_buffer = new FrameBuffer(nx, ny, max_depth);

	camera_info = CameraInfo(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), fov, (float) nx, (float) ny);

	//checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<RandState>(nx * ny);

	d_world = thrust::device_new<World*>();

	// Increase device heap for device-side new in create_world/BVH construction
	size_t heap_size;
	check_cuda_errors(cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));
	if (heap_size < 64 * 1024 * 1024) {
		check_cuda_errors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024));
	}

	// Increase per-thread stack size for NEE (extra HitRecords + BVH recursion)
	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

	create_world<<<1, 1>>> (d_world, d_camera, camera_info);

	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

void KernelInfo::resize(int nx, int ny) {
	this->nx = nx;
	this->ny = ny;

	delete frame_buffer;
	this->frame_buffer = new FrameBuffer(nx, ny, max_depth);

	int tx = 8;
	int ty = 8;

	thrust::device_free(d_rand_state);
	d_rand_state = thrust::device_new<RandState>(nx * ny);


	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

__global__ void set_device_camera(thrust::device_ptr<Camera*> d_camera, glm::vec3 position, glm::vec3 forward, glm::vec3 up, float aspect_ratio) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		((Camera*) (*d_camera))->set_position(position);
		((Camera*) (*d_camera))->set_rotation(forward, up, aspect_ratio);
	}
}

void KernelInfo::set_camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up) {
	set_device_camera<<<1, 1>>> (d_camera, position, forward, up, (float) nx / (float) ny);
	check_cuda_errors(cudaGetLastError());
	// No cudaDeviceSynchronize needed — stream ordering guarantees this
	// completes before the next raytrace kernel on the same stream
}

void KernelInfo::render(bool camera_moving, int pixelate) {

	check_cuda_errors(cudaGraphicsMapResources(1, &resources));
	check_cuda_errors(cudaGraphicsResourceGetMappedPointer((void**)&(frame_buffer->device_ptr), &(frame_buffer->buffer_size), resources));

	int tx = 16;
	int ty = 16;

	if (pixelate > 1) {
		int eff_nx = (nx + pixelate - 1) / pixelate;
		int eff_ny = (ny + pixelate - 1) / pixelate;
		dim3 blocks((eff_nx + tx - 1) / tx, (eff_ny + ty - 1) / ty);
		dim3 threads(tx, ty);

		raytrace_pixelated<<<blocks, threads>>> (*frame_buffer, d_world, d_camera, d_rand_state, pixelate);
	} else {
		dim3 blocks(nx / tx + 1, ny / ty + 1);
		dim3 threads(tx, ty);

		int spp = camera_moving ? 1 : samples;

		// frame buffer is implicitly copied to the device each frame
		raytrace<<<blocks, threads>>> (*frame_buffer, d_world, d_camera, d_rand_state, spp, 0, 0);
	}

	check_cuda_errors(cudaGetLastError());
	// wait for the gpu to finish
	check_cuda_errors(cudaDeviceSynchronize());

	check_cuda_errors(cudaGraphicsUnmapResources(1, &resources));
}

void KernelInfo::map_pbo() {
	check_cuda_errors(cudaGraphicsMapResources(1, &resources));
	check_cuda_errors(cudaGraphicsResourceGetMappedPointer((void**)&(frame_buffer->device_ptr), &(frame_buffer->buffer_size), resources));
}

void KernelInfo::unmap_pbo() {
	check_cuda_errors(cudaGraphicsUnmapResources(1, &resources));
}

void KernelInfo::render_tile(int tile_offset_x, int tile_offset_y, int tile_w, int tile_h, int spp) {
	int tx = 16;
	int ty = 16;

	dim3 blocks((tile_w + tx - 1) / tx, (tile_h + ty - 1) / ty);
	dim3 threads(tx, ty);

	raytrace<<<blocks, threads>>> (*frame_buffer, d_world, d_camera, d_rand_state, spp, tile_offset_x, tile_offset_y);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

__global__ void free_scene(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera) {
	delete* d_world;
	delete* d_camera;
}

KernelInfo::~KernelInfo() {
	check_cuda_errors(cudaDeviceSynchronize());

	free_scene<<<1, 1>>> (d_world, d_camera);

	thrust::device_free(d_world);
	thrust::device_free(d_camera);
	thrust::device_free(d_rand_state);

	delete frame_buffer;
}