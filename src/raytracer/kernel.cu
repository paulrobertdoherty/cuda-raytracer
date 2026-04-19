
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include <glm/glm.hpp>
#include <curand_kernel.h>

#ifndef HEADLESS_BUILD
#include "Window.h"
#endif
#include "cuda_errors.h"
#include "FrameBuffer.h"
#include "Ray.h"
#include "Sphere.h"
#include "Rectangle.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "Disc.h"
#include "World.h"
#include "Camera.h"
#include "BVHNode.h"

#include <thrust/device_new.h>
#include <thrust/device_free.h>

#include "raytracer/kernel.h"
#include "Scene.h"
#include "Mesh.h"
#ifdef HEADLESS_BUILD
#include "HeadlessTexture.h"
#else
#include "GLTexture.h"
#endif
#include "MeshBVHBuilder.h"

#include <vector>

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

// Creates an empty world + the device-side camera. The actual scene contents
// are populated via build_world_from_desc, driven by the host-side Scene in
// KernelInfo::rebuild_world. Keeping this minimal makes the Scene the single
// source of truth for what the ray tracer renders.
__global__ void create_world(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera, CameraInfo camera_info) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new World();
		*d_camera = camera_info.construct_camera();
	}
}

// Rebuilds the device-side World from a POD descriptor array. Deletes the
// previous World (cascading through the BVH or the raw object array) and
// replaces it with a fresh one populated per-descriptor. Mesh vertex/index
// buffers are NOT freed here — they are owned by KernelInfo and outlive
// individual rebuilds.
__global__ void build_world_from_desc(thrust::device_ptr<World*> d_world,
                                      DeviceObjectDesc* descs, int count) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	// Tear down the previous world (BVH + hittables + their materials).
	if (*d_world) {
		delete *d_world;
	}

	// Over-allocate capacity to leave room for a few post-hoc additions.
	World* world = new World(count + 8);
	*d_world = world;

	for (int i = 0; i < count; i++) {
		const DeviceObjectDesc& d = descs[i];

		Material* mat = nullptr;
		switch (d.material) {
			case DeviceMaterialKind::Lambertian: {
				Texture* albedo = nullptr;
				if (d.use_checker) {
					albedo = new CheckerTexture(d.checker_color1, d.checker_color2);
				} else if (d.has_texture && d.d_texture_pixels) {
					albedo = new ImageTexture(
						d.d_texture_pixels, d.tex_width, d.tex_height, d.tex_channels);
				} else {
					albedo = new SolidColor(d.albedo);
				}
				
				Texture* normal = nullptr;
				if (d.has_normal && d.d_normal_pixels) {
					normal = new ImageTexture(d.d_normal_pixels, d.normal_width, d.normal_height, d.normal_channels);
				}

				Texture* specular = nullptr;
				if (d.has_specular && d.d_specular_pixels) {
					specular = new ImageTexture(d.d_specular_pixels, d.specular_width, d.specular_height, d.specular_channels);
				}

				mat = new Lambertian(albedo, normal, specular);
				break;
			}
			case DeviceMaterialKind::Metal:
				mat = new Metal(d.albedo, d.fuzz);
				break;
			case DeviceMaterialKind::Dielectric:
				mat = new Dielectric(d.ior);
				break;
			case DeviceMaterialKind::Emissive:
				mat = new Emissive(d.emission);
				break;
		}

		Hittable* h = nullptr;
		switch (d.kind) {
			case DeviceObjectKind::Sphere:
				h = new Sphere(d.center, d.radius, mat);
				break;
			case DeviceObjectKind::Triangle:
				h = new Triangle(d.v0, d.v1, d.v2, mat);
				break;
			case DeviceObjectKind::Rect:
				h = new Rect(d.rect_Q, d.rect_u, d.rect_v, mat);
				break;
			case DeviceObjectKind::Disc:
				h = new Disc(d.disc_center, d.disc_normal, d.disc_radius, mat);
				break;
			case DeviceObjectKind::Mesh:
				h = new TriangleMesh(d.d_mesh_vertices, d.d_mesh_normals, d.d_mesh_uvs, d.mesh_vcount,
				                     d.d_mesh_indices, d.mesh_icount,
				                     d.mesh_translate, d.mesh_scale,
				                     mat,
				                     d.mesh_aabb_min, d.mesh_aabb_max,
				                     d.d_mesh_bvh_nodes, d.mesh_bvh_node_count,
				                     d.d_mesh_reordered_tri_ids, d.mesh_tri_id_count);
				break;
		}

		world->add(h);
		if (d.is_light) world->add_light(h);
	}

	curandState rand_state;
	curand_init(1984, 0, 0, &rand_state);
	world->build_bvh(&rand_state);
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
	this->headless = false;
	this->d_headless_buffer = nullptr;

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

KernelInfo::KernelInfo(int nx, int ny, int samples, int max_depth, float fov) {
	this->resources = {};
	this->nx = nx;
	this->ny = ny;
	this->samples = samples;
	this->max_depth = max_depth;
	this->headless = true;
	this->d_headless_buffer = nullptr;

	this->frame_buffer = new FrameBuffer(nx, ny, max_depth);

	camera_info = CameraInfo(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), fov, (float) nx, (float) ny);

	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<RandState>(nx * ny);
	d_world = thrust::device_new<World*>();

	size_t heap_size;
	check_cuda_errors(cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));
	if (heap_size < 64 * 1024 * 1024) {
		check_cuda_errors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024));
	}
	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

	// Allocate device-side pixel buffer (no OpenGL PBO in headless mode)
	size_t buffer_bytes = (size_t)nx * ny * sizeof(uint32_t);
	check_cuda_errors(cudaMalloc(&d_headless_buffer, buffer_bytes));
	frame_buffer->device_ptr = d_headless_buffer;

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

void KernelInfo::render_to_buffer(std::vector<uint8_t>& output_rgba) {
	// Point frame_buffer at headless device buffer
	frame_buffer->device_ptr = d_headless_buffer;

	int tx = 16;
	int ty = 16;
	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);

	raytrace<<<blocks, threads>>> (*frame_buffer, d_world, d_camera, d_rand_state, samples, 0, 0);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	// Copy BGRA pixels from device to host
	size_t pixel_count = (size_t)nx * ny;
	std::vector<uint32_t> host_bgra(pixel_count);
	check_cuda_errors(cudaMemcpy(host_bgra.data(), d_headless_buffer,
	                             pixel_count * sizeof(uint32_t),
	                             cudaMemcpyDeviceToHost));

	// Convert BGRA → RGBA and flip vertically (OpenGL convention → image convention)
	output_rgba.resize(pixel_count * 4);
	for (int row = 0; row < ny; row++) {
		int src_row = ny - 1 - row; // flip Y
		for (int col = 0; col < nx; col++) {
			uint32_t packed = host_bgra[src_row * nx + col];
			uint8_t b = (packed >>  0) & 0xFF;
			uint8_t g = (packed >>  8) & 0xFF;
			uint8_t r = (packed >> 16) & 0xFF;
			uint8_t a = (packed >> 24) & 0xFF;
			size_t dst = ((size_t)row * nx + col) * 4;
			output_rgba[dst + 0] = r;
			output_rgba[dst + 1] = g;
			output_rgba[dst + 2] = b;
			output_rgba[dst + 3] = a;
		}
	}
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
	check_cuda_errors(cudaDeviceSynchronize());

	thrust::device_free(d_world);
	thrust::device_free(d_camera);
	thrust::device_free(d_rand_state);

	if (d_headless_buffer) {
		cudaFree(d_headless_buffer);
		d_headless_buffer = nullptr;
	}

	// Release any cached mesh vertex/index buffers. These are owned by
	// KernelInfo (not by the device-side TriangleMesh Hittables) so we
	// free them after the world has been torn down.
	for (glm::vec3* p : d_mesh_vertex_buffers) {
		if (p) cudaFree(p);
	}
	for (int* p : d_mesh_index_buffers) {
		if (p) cudaFree(p);
	}
	for (MeshBVHNode* p : d_mesh_bvh_buffers) {
		if (p) cudaFree(p);
	}
	for (int* p : d_mesh_tri_id_buffers) {
		if (p) cudaFree(p);
	}
	for (glm::vec2* p : d_mesh_uv_buffers) {
		if (p) cudaFree(p);
	}
	for (glm::vec3* p : d_mesh_normal_buffers) {
		if (p) cudaFree(p);
	}
	for (unsigned char* p : d_texture_pixel_buffers) {
		if (p) cudaFree(p);
	}
	d_mesh_vertex_buffers.clear();
	d_mesh_index_buffers.clear();
	d_mesh_vcounts.clear();
	d_mesh_icounts.clear();
	d_mesh_bvh_buffers.clear();
	d_mesh_bvh_node_counts.clear();
	d_mesh_tri_id_buffers.clear();
	d_mesh_tri_id_counts.clear();
	d_mesh_uv_buffers.clear();
	d_mesh_normal_buffers.clear();
	d_texture_pixel_buffers.clear();
	d_texture_widths.clear();
	d_texture_heights.clear();
	d_texture_channels.clear();

	delete frame_buffer;
}

void KernelInfo::rebuild_world(const Scene& scene) {
	// Any in-flight kernels must finish before we delete their world.
	check_cuda_errors(cudaDeviceSynchronize());

	// ---- Mesh buffer lifetime management ---------------------------------
	// KernelInfo owns a device-side vertex/index buffer per scene mesh,
	// indexed by mesh_index. A rebuild only re-uploads a slot when its
	// vertex/index count doesn't match the cached version (i.e., a new or
	// resized mesh). Dragging a mesh only mutates position/scale, which lives
	// in the descriptor, so the cached buffers are reused across drags.
	const auto& scene_meshes = scene.meshes();

	// Free any stale slots beyond the current mesh count (e.g., a mesh was
	// removed between rebuilds — not exercised yet but keeps the vectors
	// honest).
	for (size_t i = scene_meshes.size(); i < d_mesh_vertex_buffers.size(); i++) {
		if (d_mesh_vertex_buffers[i]) cudaFree(d_mesh_vertex_buffers[i]);
		if (d_mesh_index_buffers[i]) cudaFree(d_mesh_index_buffers[i]);
	}
	for (size_t i = scene_meshes.size(); i < d_mesh_bvh_buffers.size(); i++) {
		if (d_mesh_bvh_buffers[i]) cudaFree(d_mesh_bvh_buffers[i]);
		if (d_mesh_tri_id_buffers[i]) cudaFree(d_mesh_tri_id_buffers[i]);
	}
	for (size_t i = scene_meshes.size(); i < d_mesh_uv_buffers.size(); i++) {
		if (d_mesh_uv_buffers[i]) cudaFree(d_mesh_uv_buffers[i]);
	}
	for (size_t i = scene_meshes.size(); i < d_mesh_normal_buffers.size(); i++) {
		if (d_mesh_normal_buffers[i]) cudaFree(d_mesh_normal_buffers[i]);
	}
	d_mesh_vertex_buffers.resize(scene_meshes.size(), nullptr);
	d_mesh_index_buffers.resize(scene_meshes.size(), nullptr);
	d_mesh_vcounts.resize(scene_meshes.size(), 0);
	d_mesh_icounts.resize(scene_meshes.size(), 0);
	d_mesh_bvh_buffers.resize(scene_meshes.size(), nullptr);
	d_mesh_bvh_node_counts.resize(scene_meshes.size(), 0);
	d_mesh_tri_id_buffers.resize(scene_meshes.size(), nullptr);
	d_mesh_tri_id_counts.resize(scene_meshes.size(), 0);
	d_mesh_uv_buffers.resize(scene_meshes.size(), nullptr);
	d_mesh_normal_buffers.resize(scene_meshes.size(), nullptr);

	for (size_t i = 0; i < scene_meshes.size(); i++) {
		const Mesh* m = scene_meshes[i].get();
		int vcount = (int)m->vertices.size();
		int icount = (int)m->indices.size();

		// Skip re-upload when the cache already holds this mesh.
		if (d_mesh_vertex_buffers[i] && d_mesh_vcounts[i] == vcount &&
		    d_mesh_icounts[i] == icount) {
			continue;
		}

		if (d_mesh_vertex_buffers[i]) cudaFree(d_mesh_vertex_buffers[i]);
		if (d_mesh_index_buffers[i]) cudaFree(d_mesh_index_buffers[i]);
		if (d_mesh_uv_buffers[i])    cudaFree(d_mesh_uv_buffers[i]);
		if (d_mesh_normal_buffers[i]) cudaFree(d_mesh_normal_buffers[i]);
		d_mesh_vertex_buffers[i] = nullptr;
		d_mesh_index_buffers[i] = nullptr;
		d_mesh_uv_buffers[i] = nullptr;
		d_mesh_normal_buffers[i] = nullptr;

		if (vcount <= 0 || icount <= 0) {
			d_mesh_vcounts[i] = 0;
			d_mesh_icounts[i] = 0;
			continue;
		}

		// Pack positions, UVs, and indices into flat host buffers.
		std::vector<glm::vec3> positions;
		positions.reserve(vcount);
		for (const auto& v : m->vertices) positions.push_back(v.position);

		std::vector<glm::vec3> normals;
		normals.reserve(vcount);
		for (const auto& v : m->vertices) normals.push_back(v.normal);

		std::vector<glm::vec2> uvs;
		uvs.reserve(vcount);
		for (const auto& v : m->vertices) uvs.push_back(v.uv);

		std::vector<int> int_indices;
		int_indices.reserve(icount);
		for (unsigned int u : m->indices) int_indices.push_back((int)u);

		glm::vec3* d_verts = nullptr;
		glm::vec3* d_norm = nullptr;
		glm::vec2* d_uv = nullptr;
		int* d_idx = nullptr;
		check_cuda_errors(cudaMalloc(&d_verts, sizeof(glm::vec3) * vcount));
		check_cuda_errors(cudaMemcpy(d_verts, positions.data(),
		                             sizeof(glm::vec3) * vcount,
		                             cudaMemcpyHostToDevice));
		check_cuda_errors(cudaMalloc(&d_norm, sizeof(glm::vec3) * vcount));
		check_cuda_errors(cudaMemcpy(d_norm, normals.data(),
		                             sizeof(glm::vec3) * vcount,
		                             cudaMemcpyHostToDevice));
		check_cuda_errors(cudaMalloc(&d_uv, sizeof(glm::vec2) * vcount));
		check_cuda_errors(cudaMemcpy(d_uv, uvs.data(),
		                             sizeof(glm::vec2) * vcount,
		                             cudaMemcpyHostToDevice));
		check_cuda_errors(cudaMalloc(&d_idx, sizeof(int) * icount));
		check_cuda_errors(cudaMemcpy(d_idx, int_indices.data(),
		                             sizeof(int) * icount,
		                             cudaMemcpyHostToDevice));

		d_mesh_vertex_buffers[i] = d_verts;
		d_mesh_normal_buffers[i] = d_norm;
		d_mesh_uv_buffers[i] = d_uv;
		d_mesh_index_buffers[i] = d_idx;
		d_mesh_vcounts[i] = vcount;
		d_mesh_icounts[i] = icount;

		// Build per-mesh BVH on the host and upload the flat node array
		// plus the reordered triangle-ID array to the device.
		if (d_mesh_bvh_buffers[i]) cudaFree(d_mesh_bvh_buffers[i]);
		if (d_mesh_tri_id_buffers[i]) cudaFree(d_mesh_tri_id_buffers[i]);
		d_mesh_bvh_buffers[i] = nullptr;
		d_mesh_tri_id_buffers[i] = nullptr;

		int tri_count = icount / 3;
		MeshBVHBuildResult bvh = build_mesh_bvh(
			positions.data(), m->indices.data(), tri_count);

		int bvh_node_count = (int)bvh.nodes.size();
		int tri_id_count   = (int)bvh.reordered_tri_ids.size();

		if (bvh_node_count > 0) {
			MeshBVHNode* d_bvh = nullptr;
			check_cuda_errors(cudaMalloc(&d_bvh, sizeof(MeshBVHNode) * bvh_node_count));
			check_cuda_errors(cudaMemcpy(d_bvh, bvh.nodes.data(),
			                             sizeof(MeshBVHNode) * bvh_node_count,
			                             cudaMemcpyHostToDevice));
			d_mesh_bvh_buffers[i] = d_bvh;
		}
		d_mesh_bvh_node_counts[i] = bvh_node_count;

		if (tri_id_count > 0) {
			int* d_tri_ids = nullptr;
			check_cuda_errors(cudaMalloc(&d_tri_ids, sizeof(int) * tri_id_count));
			check_cuda_errors(cudaMemcpy(d_tri_ids, bvh.reordered_tri_ids.data(),
			                             sizeof(int) * tri_id_count,
			                             cudaMemcpyHostToDevice));
			d_mesh_tri_id_buffers[i] = d_tri_ids;
		}
		d_mesh_tri_id_counts[i] = tri_id_count;
	}

	// ---- Texture pixel buffer upload -------------------------------------
	const auto& scene_textures = scene.textures();

	for (size_t i = scene_textures.size(); i < d_texture_pixel_buffers.size(); i++) {
		if (d_texture_pixel_buffers[i]) cudaFree(d_texture_pixel_buffers[i]);
	}
	d_texture_pixel_buffers.resize(scene_textures.size(), nullptr);
	d_texture_widths.resize(scene_textures.size(), 0);
	d_texture_heights.resize(scene_textures.size(), 0);
	d_texture_channels.resize(scene_textures.size(), 0);

	for (size_t i = 0; i < scene_textures.size(); i++) {
		const SceneTexture* tex = scene_textures[i].get();
		if (!tex->raw_pixels()) continue;

		int w  = tex->width();
		int h  = tex->height();
		int ch = tex->channels();

		// Skip re-upload when size matches cached version.
		if (d_texture_pixel_buffers[i] &&
		    d_texture_widths[i] == w && d_texture_heights[i] == h &&
		    d_texture_channels[i] == ch) {
			continue;
		}
		if (d_texture_pixel_buffers[i]) cudaFree(d_texture_pixel_buffers[i]);

		size_t byte_count = (size_t)w * h * ch;
		unsigned char* d_pix = nullptr;
		check_cuda_errors(cudaMalloc(&d_pix, byte_count));
		check_cuda_errors(cudaMemcpy(d_pix, tex->raw_pixels(), byte_count,
		                             cudaMemcpyHostToDevice));
		d_texture_pixel_buffers[i] = d_pix;
		d_texture_widths[i]  = w;
		d_texture_heights[i] = h;
		d_texture_channels[i] = ch;
	}

	// ---- Build host-side descriptor vector -------------------------------
	const auto& scene_objects = scene.objects();
	std::vector<DeviceObjectDesc> descs;
	descs.reserve(scene_objects.size());

	for (const SceneObject& o : scene_objects) {
		DeviceObjectDesc d = {};

		switch (o.material) {
			case SceneMaterial::Lambertian: d.material = DeviceMaterialKind::Lambertian; break;
			case SceneMaterial::Metal:      d.material = DeviceMaterialKind::Metal; break;
			case SceneMaterial::Dielectric: d.material = DeviceMaterialKind::Dielectric; break;
			case SceneMaterial::Emissive:   d.material = DeviceMaterialKind::Emissive; break;
		}
		d.albedo = o.albedo;
		d.fuzz = o.fuzz;
		d.ior = o.ior;
		d.emission = o.emission;
		d.is_light = o.is_light ? 1 : 0;
		d.use_checker    = o.use_checker ? 1 : 0;
		d.checker_color1 = o.checker_color1;
		d.checker_color2 = o.checker_color2;

		switch (o.kind) {
			case ProxyKind::Sphere:
				d.kind = DeviceObjectKind::Sphere;
				d.center = o.center + o.position;
				d.radius = o.radius * o.scale;
				break;
			case ProxyKind::Triangle:
				d.kind = DeviceObjectKind::Triangle;
				d.v0 = o.position + o.scale * o.v0;
				d.v1 = o.position + o.scale * o.v1;
				d.v2 = o.position + o.scale * o.v2;
				break;
			case ProxyKind::Rect:
				d.kind = DeviceObjectKind::Rect;
				d.rect_Q = o.position + o.scale * o.Q;
				d.rect_u = o.scale * o.u;
				d.rect_v = o.scale * o.v;
				break;
			case ProxyKind::Disc:
				d.kind = DeviceObjectKind::Disc;
				d.disc_center = o.center + o.position;
				d.disc_normal = o.disc_normal;
				d.disc_radius = o.radius * o.scale;
				break;
			case ProxyKind::Mesh: {
				if (o.mesh_index < 0 || o.mesh_index >= (int)scene_meshes.size()) {
					continue;
				}
				if (!d_mesh_vertex_buffers[o.mesh_index] ||
				    !d_mesh_index_buffers[o.mesh_index]) {
					continue;
				}
				d.kind = DeviceObjectKind::Mesh;
				d.d_mesh_vertices = d_mesh_vertex_buffers[o.mesh_index];
				d.d_mesh_normals  = d_mesh_normal_buffers[o.mesh_index];
				d.d_mesh_uvs      = d_mesh_uv_buffers[o.mesh_index];
				d.d_mesh_indices  = d_mesh_index_buffers[o.mesh_index];
				d.mesh_vcount     = d_mesh_vcounts[o.mesh_index];
				d.mesh_icount     = d_mesh_icounts[o.mesh_index];
				d.mesh_translate  = o.position;
				d.mesh_scale      = o.scale;
				const Mesh* m = scene_meshes[o.mesh_index].get();
				d.mesh_aabb_min = o.position + o.scale * m->local_min;
				d.mesh_aabb_max = o.position + o.scale * m->local_max;
				d.d_mesh_bvh_nodes        = d_mesh_bvh_buffers[o.mesh_index];
				d.mesh_bvh_node_count     = d_mesh_bvh_node_counts[o.mesh_index];
				d.d_mesh_reordered_tri_ids = d_mesh_tri_id_buffers[o.mesh_index];
				d.mesh_tri_id_count        = d_mesh_tri_id_counts[o.mesh_index];
				// Image textures
				if (o.texture_index >= 0 &&
				    o.texture_index < (int)d_texture_pixel_buffers.size() &&
				    d_texture_pixel_buffers[o.texture_index]) {
					d.d_texture_pixels = d_texture_pixel_buffers[o.texture_index];
					d.tex_width        = d_texture_widths[o.texture_index];
					d.tex_height       = d_texture_heights[o.texture_index];
					d.tex_channels     = d_texture_channels[o.texture_index];
					d.has_texture      = 1;
				} else {
					d.d_texture_pixels = nullptr;
					d.has_texture      = 0;
				}

				if (o.normal_texture_index >= 0 &&
				    o.normal_texture_index < (int)d_texture_pixel_buffers.size() &&
				    d_texture_pixel_buffers[o.normal_texture_index]) {
					d.d_normal_pixels = d_texture_pixel_buffers[o.normal_texture_index];
					d.normal_width    = d_texture_widths[o.normal_texture_index];
					d.normal_height   = d_texture_heights[o.normal_texture_index];
					d.normal_channels = d_texture_channels[o.normal_texture_index];
					d.has_normal      = 1;
				} else {
					d.d_normal_pixels = nullptr;
					d.has_normal      = 0;
				}

				if (o.specular_texture_index >= 0 &&
				    o.specular_texture_index < (int)d_texture_pixel_buffers.size() &&
				    d_texture_pixel_buffers[o.specular_texture_index]) {
					d.d_specular_pixels = d_texture_pixel_buffers[o.specular_texture_index];
					d.specular_width    = d_texture_widths[o.specular_texture_index];
					d.specular_height   = d_texture_heights[o.specular_texture_index];
					d.specular_channels = d_texture_channels[o.specular_texture_index];
					d.has_specular      = 1;
				} else {
					d.d_specular_pixels = nullptr;
					d.has_specular      = 0;
				}
				break;
			}
		}

		descs.push_back(d);
	}

	// ---- Upload descriptors and launch the device-side rebuild -----------
	int count = (int)descs.size();
	DeviceObjectDesc* d_descs = nullptr;
	if (count > 0) {
		check_cuda_errors(cudaMalloc(&d_descs, sizeof(DeviceObjectDesc) * count));
		check_cuda_errors(cudaMemcpy(d_descs, descs.data(),
		                             sizeof(DeviceObjectDesc) * count,
		                             cudaMemcpyHostToDevice));
	}

	build_world_from_desc<<<1, 1>>>(d_world, d_descs, count);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	if (d_descs) {
		check_cuda_errors(cudaFree(d_descs));
	}
}