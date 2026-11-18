
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include <glm/glm.hpp>
#include <curand_kernel.h>

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
#include "MeshBVHBuilder.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <vector>
#include <memory>
#include <new>
#include <type_traits>

__global__ void raytrace(uint32_t* __restrict__ pixels, int width, int height, int max_depth,
                         thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera,
                         thrust::device_ptr<RandState> rand_state, int samples,
                         int tile_offset_x, int tile_offset_y) {

	// X AND Y coordinates with tile offset
	int i = threadIdx.x + blockIdx.x * blockDim.x + tile_offset_x;
	int j = threadIdx.y + blockIdx.y * blockDim.y + tile_offset_y;

	// return early if we're outside of the frame buffer
	if ((i >= width) || (j >= height)) return;

	int pixel_idx = j * width + i;

	RandState local_rand_state = rand_state[pixel_idx];

	glm::vec3 col = glm::vec3(0.0f, 0.0f, 0.0f);

	for (int s = 0; s < samples; s++) {
		// normalized screen coordinates
		float u = float(i + curand_uniform(&local_rand_state)) / float(width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(height);
		Ray r = ((Camera*)(*d_camera))->get_ray(u, v);
		col += trace_color(r, *world, &local_rand_state, max_depth);
	}
	rand_state[pixel_idx] = local_rand_state;
	col /= float(samples);
	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);

	write_pixel(pixels, pixel_idx, glm::vec4(col, 1.0f));
}

__global__ void raytrace_pixelated(uint32_t* __restrict__ pixels, int width, int height, int max_depth,
                                   thrust::device_ptr<World*> world, thrust::device_ptr<Camera*> d_camera,
                                   thrust::device_ptr<RandState> rand_state, int pixelate) {

	int ci = threadIdx.x + blockIdx.x * blockDim.x;
	int cj = threadIdx.y + blockIdx.y * blockDim.y;

	int x0 = ci * pixelate;
	int y0 = cj * pixelate;

	if (x0 >= width || y0 >= height) return;

	// Reuse the rand state slot at the block's top-left pixel.
	int rand_idx = y0 * width + x0;
	RandState local_rand_state = rand_state[rand_idx];

	float u = (float(x0) + 0.5f * float(pixelate)) / float(width);
	float v = (float(y0) + 0.5f * float(pixelate)) / float(height);
	Ray r = ((Camera*)(*d_camera))->get_ray(u, v);
	glm::vec3 col = trace_color(r, *world, &local_rand_state, max_depth);
	rand_state[rand_idx] = local_rand_state;

	col[0] = sqrtf(col[0]);
	col[1] = sqrtf(col[1]);
	col[2] = sqrtf(col[2]);
	glm::vec4 out(col, 1.0f);

	int x_end = x0 + pixelate;
	int y_end = y0 + pixelate;
	if (x_end > width) x_end = width;
	if (y_end > height) y_end = height;
	for (int py = y0; py < y_end; py++) {
		for (int px = x0; px < x_end; px++) {
			write_pixel(pixels, py * width + px, out);
		}
	}
}

// World and Camera persist for the renderer lifetime. The scene pool owns their scene data.
__global__ void create_world(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera, CameraInfo camera_info) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new World();
		*d_camera = camera_info.construct_camera();
	}
}

// ---------------------------------------------------------------------------
// POD descriptors used to rebuild the CUDA world from host state. These are
// uploaded as a contiguous device buffer; the construction kernel
// constructs each Hittable and Material in its own scene pool slot.
// ---------------------------------------------------------------------------

enum class DeviceObjectKind : std::uint8_t {
    Sphere = 0,
    Triangle = 1,
    Rect = 2,
    Mesh = 3,
    Disc = 4
};

enum class DeviceMaterialKind : std::uint8_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    Emissive = 3,
    SubsurfaceScattering = 4
};

struct DeviceObjectDesc {
    DeviceObjectKind kind;
    DeviceMaterialKind material;
    glm::vec3 albedo;
    float fuzz;
    float ior;
    glm::vec3 emission;
    int is_light; // 0/1

    // Subsurface scattering (when material == SubsurfaceScattering)
    float scattering_distance;
    glm::vec3 extinction_coeff;

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

    DeviceMeshView mesh;
    glm::vec3 mesh_translate;
    float mesh_scale;
    glm::vec3 mesh_aabb_min;
    glm::vec3 mesh_aabb_max;

    DeviceImageView diffuse;
    DeviceImageView normal;
    DeviceImageView specular;
};

static_assert(std::is_trivially_copyable_v<DeviceObjectDesc>);

// One slot per descriptor, shared by full builds and in-place updates.
// Placement construction sets device vtables. Every pointer in a slot borrows
// from this pool or the separate mesh/texture caches; no child owns memory.
struct SceneObjectSlot {
	std::aligned_union_t<0, Sphere, Triangle, Rect, Disc, TriangleMesh> geometry;
	std::aligned_union_t<0, Lambertian, Metal, Dielectric, Emissive, SubsurfaceScatter> material;
	std::aligned_union_t<0, SolidColor, CheckerTexture, ImageTexture> albedo;
	std::aligned_storage_t<sizeof(ImageTexture), alignof(ImageTexture)> normal;
	std::aligned_storage_t<sizeof(ImageTexture), alignof(ImageTexture)> specular;
};

__device__ static Hittable* construct_hittable_from_desc(const DeviceObjectDesc& d, SceneObjectSlot& slot) {
	Material* mat = nullptr;
	switch (d.material) {
		case DeviceMaterialKind::Lambertian: {
			Texture* albedo = nullptr;
			if (d.use_checker) {
				albedo = new (&slot.albedo) CheckerTexture(d.checker_color1, d.checker_color2);
			} else if (d.diffuse.pixels) {
				albedo = new (&slot.albedo) ImageTexture(d.diffuse);
			} else {
				albedo = new (&slot.albedo) SolidColor(d.albedo);
			}

			Texture* normal = nullptr;
			if (d.normal.pixels) {
				normal = new (&slot.normal) ImageTexture(d.normal);
			}

			Texture* specular = nullptr;
			if (d.specular.pixels) {
				specular = new (&slot.specular) ImageTexture(d.specular);
			}

			mat = new (&slot.material) Lambertian(albedo, normal, specular);
			break;
		}
		case DeviceMaterialKind::Metal:
			mat = new (&slot.material) Metal(d.albedo, d.fuzz);
			break;
		case DeviceMaterialKind::Dielectric:
			mat = new (&slot.material) Dielectric(d.ior);
			break;
		case DeviceMaterialKind::Emissive:
			mat = new (&slot.material) Emissive(new (&slot.albedo) SolidColor(d.emission));
			break;
		case DeviceMaterialKind::SubsurfaceScattering:
			mat = new (&slot.material) SubsurfaceScatter(d.albedo, d.scattering_distance, d.ior, d.extinction_coeff);
			break;
	}

	switch (d.kind) {
		case DeviceObjectKind::Sphere:
			return new (&slot.geometry) Sphere(d.center, d.radius, mat);
		case DeviceObjectKind::Triangle:
			return new (&slot.geometry) Triangle(d.v0, d.v1, d.v2, mat);
		case DeviceObjectKind::Rect:
			return new (&slot.geometry) Rect(d.rect_Q, d.rect_u, d.rect_v, mat);
		case DeviceObjectKind::Disc:
			return new (&slot.geometry) Disc(d.disc_center, d.disc_normal, d.disc_radius, mat);
		case DeviceObjectKind::Mesh:
			return new (&slot.geometry) TriangleMesh(d.mesh, d.mesh_translate, d.mesh_scale,
			                                         mat, d.mesh_aabb_min, d.mesh_aabb_max);
	}
	return nullptr;
}

__global__ void construct_objects_parallel(const DeviceObjectDesc* descs, int count,
                                           SceneObjectSlot* slots, Hittable** out) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < count) {
		out[i] = construct_hittable_from_desc(descs[i], slots[i]);
	}
}

__global__ void assemble_world(thrust::device_ptr<World*> d_world, Hittable** objects,
                               const DeviceObjectDesc* descs, int count, Hittable** sorted_objects,
                               Hittable** lights, BVHNode* nodes) {
	World* world = *d_world;
	world->objects = objects;
	world->lights = lights;
	world->number_of_objects = count;
	world->num_lights = 0;
	for (int i = 0; i < count; i++) {
		sorted_objects[i] = objects[i];
		if (descs[i].is_light) {
			objects[i]->light_index = world->num_lights;
			lights[world->num_lights++] = objects[i];
		}
	}
	world->build_bvh(sorted_objects, nodes);
}

// Publish the new primitive to every view of the edited slot. A singleton BVH
// range stores the same leaf in both child pointers.
__global__ void update_scene_object(thrust::device_ptr<World*> d_world,
                                    SceneObjectSlot* slots, Hittable** objects,
                                    DeviceObjectDesc* descs, int i, DeviceObjectDesc updated) {
	Hittable* old = objects[i];
	BVHNode* parent = old->bvh_parent;
	const bool was_left = parent->left == old;
	const bool was_right = parent->right == old;
	const int light_index = old->light_index;
	const bool lights_changed = descs[i].is_light != updated.is_light;
	descs[i] = updated;
	Hittable* leaf = construct_hittable_from_desc(updated, slots[i]);
	objects[i] = leaf;
	leaf->bvh_parent = parent;
	if (was_left) parent->left = leaf;
	if (was_right) parent->right = leaf;
	parent->refit_ancestors();

	World* world = *d_world;
	if (lights_changed) {
		world->num_lights = 0;
		for (int j = 0; j < world->number_of_objects; ++j) {
			if (descs[j].is_light) {
				objects[j]->light_index = world->num_lights;
				world->lights[world->num_lights++] = objects[j];
			}
		}
	} else if (light_index >= 0) {
		leaf->light_index = light_index;
		world->lights[light_index] = leaf;
	}
}

__global__ void render_init(int width, int height, thrust::device_ptr<RandState> rand_state, int seed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) return;
	int pixel_index = j * width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(seed, pixel_index, 0, &rand_state.get()[pixel_index]);
}


KernelInfo::KernelInfo(cudaGraphicsResource_t resources, int nx, int ny, int samples, int max_depth, float fov, int seed) : KernelInfo() {
	this->resources = resources;
	this->nx = nx;
	this->ny = ny;
	this->samples = samples;
	this->max_depth = max_depth;
	this->seed = seed;
	this->headless = false;
	this->d_headless_buffer = nullptr;

	camera_info = CameraInfo(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), fov, (float) nx, (float) ny);

	//checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<RandState>(nx * ny);
	d_rand_state_capacity = nx * ny;

	d_world = thrust::device_new<World*>();


	// Increase per-thread stack size for NEE (extra HitRecords + BVH recursion)
	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

	create_world<<<1, 1>>> (d_world, d_camera, camera_info);

	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;

	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state, seed);
	check_cuda_errors(cudaGetLastError());
	// No sync needed: stream ordering guarantees subsequent kernels on the
	// default stream see the initialized rand state.
}

KernelInfo::KernelInfo(int nx, int ny, int samples, int max_depth, float fov, int seed) : KernelInfo() {
	this->resources = {};
	this->nx = nx;
	this->ny = ny;
	this->samples = samples;
	this->max_depth = max_depth;
	this->seed = seed;
	this->headless = true;
	this->d_headless_buffer = nullptr;

	camera_info = CameraInfo(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), fov, (float) nx, (float) ny);

	d_camera = thrust::device_new<Camera*>();
	d_rand_state = thrust::device_new<RandState>(nx * ny);
	d_rand_state_capacity = nx * ny;
	d_world = thrust::device_new<World*>();

	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

	// Allocate device-side pixel buffer (no OpenGL PBO in headless mode)
	size_t buffer_bytes = (size_t)nx * ny * sizeof(uint32_t);
	check_cuda_errors(cudaMalloc(&d_headless_buffer, buffer_bytes));

	create_world<<<1, 1>>> (d_world, d_camera, camera_info);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;
	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>> (nx, ny, d_rand_state, seed);
	check_cuda_errors(cudaGetLastError());
	// No sync needed: stream ordering guarantees subsequent kernels on the
	// default stream see the initialized rand state.
}

void KernelInfo::render_to_buffer(std::vector<uint8_t>& output_rgba) {
	// 32x8 keeps each warp on a single row of pixels: framebuffer writes hit
	// one cache line per warp (vs two for 16x16) and the per-lane Philox
	// state load is row-aligned.
	constexpr int tx = 32;
	constexpr int ty = 8;
	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);

	raytrace<<<blocks, threads>>>(d_headless_buffer, nx, ny, max_depth, d_world, d_camera,
	                              d_rand_state, samples, 0, 0);
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

void KernelInfo::resize(int width, int height) {
	const int new_count = width * height;
	if (new_count > d_rand_state_capacity) {
		auto new_state = thrust::device_new<RandState>(new_count);
		uint32_t* new_pixels = nullptr;
		try {
			if (headless) {
				check_cuda_errors(cudaMalloc(&new_pixels, sizeof(uint32_t) * new_count));
			}
			const dim3 threads(8, 8);
			const dim3 blocks((width + 7) / 8, (height + 7) / 8);
			render_init<<<blocks, threads>>>(width, height, new_state, seed);
			check_cuda_errors(cudaGetLastError());
		} catch (...) {
			if (new_pixels) {
				cudaFree(new_pixels);
			}
			thrust::device_free(new_state);
			throw;
		}

		thrust::device_free(d_rand_state);
		d_rand_state = new_state;
		if (headless) {
			cudaFree(d_headless_buffer);
			d_headless_buffer = new_pixels;
		}
		d_rand_state_capacity = new_count;
	}

	// Smaller images reuse the random states at the new linear pixel indices.
	nx = width;
	ny = height;
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

	map_pbo();

	// 32x8: warp covers one row of pixels — coalesces framebuffer writes and
	// row-aligns the per-lane Philox state loads.
	constexpr int tx = 32;
	constexpr int ty = 8;

	if (pixelate > 1) {
		int eff_nx = (nx + pixelate - 1) / pixelate;
		int eff_ny = (ny + pixelate - 1) / pixelate;
		dim3 blocks((eff_nx + tx - 1) / tx, (eff_ny + ty - 1) / ty);
		dim3 threads(tx, ty);

		raytrace_pixelated<<<blocks, threads>>>(d_mapped_pixels, nx, ny, max_depth, d_world,
		                                        d_camera, d_rand_state, pixelate);
	} else {
		dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
		dim3 threads(tx, ty);

		int spp = camera_moving ? 1 : samples;

		raytrace<<<blocks, threads>>>(d_mapped_pixels, nx, ny, max_depth, d_world, d_camera,
		                              d_rand_state, spp, 0, 0);
	}

	check_cuda_errors(cudaGetLastError());
	// wait for the gpu to finish
	check_cuda_errors(cudaDeviceSynchronize());

	unmap_pbo();
}

void KernelInfo::map_pbo() {
	check_cuda_errors(cudaGraphicsMapResources(1, &resources));
	size_t mapped_size;
	const cudaError_t result = cudaGraphicsResourceGetMappedPointer(
		reinterpret_cast<void**>(&d_mapped_pixels), &mapped_size, resources);
	if (result != cudaSuccess) {
		cudaGraphicsUnmapResources(1, &resources);
		check_cuda_errors(result);
	}
}

void KernelInfo::unmap_pbo() {
	check_cuda_errors(cudaGraphicsUnmapResources(1, &resources));
	d_mapped_pixels = nullptr;
}

void KernelInfo::render_tile(int tile_offset_x, int tile_offset_y, int tile_w, int tile_h, int spp) {
	// 32x8 matches the row-aligned layout used by the main render path.
	constexpr int tx = 32;
	constexpr int ty = 8;

	dim3 blocks((tile_w + tx - 1) / tx, (tile_h + ty - 1) / ty);
	dim3 threads(tx, ty);

	raytrace<<<blocks, threads>>>(d_mapped_pixels, nx, ny, max_depth, d_world, d_camera,
	                              d_rand_state, spp, tile_offset_x, tile_offset_y);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
}

__global__ void free_scene(thrust::device_ptr<World*> d_world, thrust::device_ptr<Camera*> d_camera) {
	delete* d_world;
	delete* d_camera;
}

KernelInfo::~KernelInfo() {
	if (d_mapped_pixels) {
		cudaGraphicsUnmapResources(1, &resources);
	}

	// device_new value-initializes both pointer slots. Even if construction
	// failed before create_world ran, deleting their null contents is safe.
	if (d_world && d_camera) {
		free_scene<<<1, 1>>> (d_world, d_camera);
		const cudaError_t result = cudaDeviceSynchronize();
		if (result != cudaSuccess) {
			std::cerr << "CUDA scene cleanup failed: " << cudaGetErrorString(result) << '\n';
		}
	}

	for (thrust::device_ptr<void> ptr : {thrust::device_ptr<void>(d_world),
	                                    thrust::device_ptr<void>(d_camera),
	                                    thrust::device_ptr<void>(d_rand_state)}) {
		if (ptr) {
			try {
				thrust::device_free(ptr);
			} catch (const std::exception& error) {
				std::cerr << "CUDA allocation cleanup failed: " << error.what() << '\n';
			}
		}
	}

	if (d_headless_buffer) {
		cudaFree(d_headless_buffer);
		d_headless_buffer = nullptr;
	}

	cudaFree(d_slots);
}

namespace {
struct CudaFree {
	void operator()(void* buffer) const noexcept { cudaFree(buffer); }
};

template<typename T>
using DeviceBuffer = std::unique_ptr<T, CudaFree>;

template<typename T>
DeviceBuffer<T> upload_array(const T* data, size_t count) {
	if (count == 0) {
		return {};
	}
	T* allocation = nullptr;
	const size_t bytes = count * sizeof(T);
	check_cuda_errors(cudaMalloc(&allocation, bytes));
	DeviceBuffer<T> buffer(allocation);
	check_cuda_errors(cudaMemcpy(buffer.get(), data, bytes, cudaMemcpyHostToDevice));
	return buffer;
}
} // namespace

// Host-only owners. An upload that fails partway unwinds its completed
// allocations. Device-side code receives only the read-only view.
struct UploadedMesh {
	std::uint64_t revision;
	size_t vertex_count;
	size_t index_count;
	DeviceBuffer<glm::vec3> vertices;
	DeviceBuffer<glm::vec3> normals;
	DeviceBuffer<glm::vec2> uvs;
	DeviceBuffer<int> indices;
	DeviceBuffer<MeshBVHNode> bvh;
	DeviceBuffer<int> triangle_ids;

	explicit UploadedMesh(const Mesh& mesh)
		: revision(mesh.revision()), vertex_count(mesh.vertices.size()), index_count(mesh.indices.size()) {
		if (vertex_count == 0 || index_count < 3) {
			return;
		}
		if (vertex_count > INT_MAX || index_count > INT_MAX) {
			throw std::length_error("Mesh exceeds CUDA index range");
		}

		std::vector<glm::vec3> positions(vertex_count);
		std::vector<glm::vec3> vertex_normals(vertex_count);
		std::vector<glm::vec2> vertex_uvs(vertex_count);
		for (size_t i = 0; i < vertex_count; ++i) {
			positions[i] = mesh.vertices[i].position;
			vertex_normals[i] = mesh.vertices[i].normal;
			vertex_uvs[i] = mesh.vertices[i].uv;
		}
		const std::vector<int> triangle_indices(mesh.indices.begin(), mesh.indices.end());
		auto tree = build_mesh_bvh(positions.data(), mesh.indices.data(), static_cast<int>(index_count / 3));

		vertices = upload_array(positions.data(), vertex_count);
		normals = upload_array(vertex_normals.data(), vertex_count);
		uvs = upload_array(vertex_uvs.data(), vertex_count);
		indices = upload_array(triangle_indices.data(), index_count);
		bvh = upload_array(tree.nodes.data(), tree.nodes.size());
		triangle_ids = upload_array(tree.reordered_tri_ids.data(), tree.reordered_tri_ids.size());
	}

	bool matches(const Mesh& mesh) const {
		return revision == mesh.revision() && vertex_count == mesh.vertices.size() && index_count == mesh.indices.size();
	}

	DeviceMeshView view() const {
		return {vertices.get(), normals.get(), uvs.get(), indices.get(), bvh.get(), triangle_ids.get()};
	}
};

struct UploadedImage {
	std::uint64_t revision;
	int width;
	int height;
	int channels;
	DeviceBuffer<unsigned char> pixels;

	explicit UploadedImage(const SceneTexture& texture)
		: revision(texture.revision()), width(texture.width()), height(texture.height()), channels(texture.channels()) {
		if (texture.raw_pixels()) {
			pixels = upload_array(texture.raw_pixels(), static_cast<size_t>(width) * height * channels);
		}
	}

	DeviceImageView view() const {
		return {pixels.get(), width, height, channels};
	}
};

KernelInfo::AssetCache::AssetCache(const Scene& scene, const AssetCache& current) {
	meshes.reserve(scene.meshes().size());
	for (size_t i = 0; i < scene.meshes().size(); ++i) {
		const Mesh& mesh = *scene.meshes()[i];
		if (i < current.meshes.size() && current.meshes[i]->matches(mesh)) {
			meshes.push_back(current.meshes[i]);
		} else {
			meshes.push_back(std::make_shared<UploadedMesh>(mesh));
		}
	}

	images.reserve(scene.textures().size());
	for (size_t i = 0; i < scene.textures().size(); ++i) {
		const SceneTexture& texture = *scene.textures()[i];
		if (i < current.images.size() && current.images[i]->revision == texture.revision()) {
			images.push_back(current.images[i]);
		} else {
			images.push_back(std::make_shared<UploadedImage>(texture));
		}
	}
}

bool KernelInfo::AssetCache::matches_object_assets(const Scene& scene, const SceneObject& object) const {
	if (object.kind != ProxyKind::Mesh) {
		return true;
	}
	const int mesh = object.mesh_index;
	if (mesh >= 0 && static_cast<size_t>(mesh) < meshes.size() && !meshes[mesh]->matches(*scene.meshes()[mesh])) {
		return false;
	}
	for (int i : {object.texture_index, object.normal_texture_index, object.specular_texture_index}) {
		if (i >= 0 && static_cast<size_t>(i) < images.size() && images[i]->revision != scene.textures()[i]->revision()) {
			return false;
		}
	}
	return true;
}

bool KernelInfo::AssetCache::describe_object(const Scene& scene, const SceneObject& o, DeviceObjectDesc& d) const {
	switch (o.material) {
		case SceneMaterial::Lambertian:           d.material = DeviceMaterialKind::Lambertian; break;
		case SceneMaterial::Metal:                d.material = DeviceMaterialKind::Metal; break;
		case SceneMaterial::Dielectric:           d.material = DeviceMaterialKind::Dielectric; break;
		case SceneMaterial::Emissive:             d.material = DeviceMaterialKind::Emissive; break;
		case SceneMaterial::SubsurfaceScattering: d.material = DeviceMaterialKind::SubsurfaceScattering; break;
	}
	d.albedo = o.albedo;
	d.fuzz = o.fuzz;
	d.ior = o.ior;
	d.emission = o.emission;
	d.is_light = o.is_light ? 1 : 0;
	d.scattering_distance = o.scattering_distance;
	d.extinction_coeff = o.extinction_coeff;
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
			if (o.mesh_index < 0 || static_cast<size_t>(o.mesh_index) >= meshes.size()) {
				return false;
			}
			d.mesh = meshes[o.mesh_index]->view();
			if (!d.mesh.bvh) {
				return false;
			}
			d.kind = DeviceObjectKind::Mesh;
			d.mesh_translate = o.position;
			d.mesh_scale = o.scale;
			const Mesh& mesh = *scene.meshes()[o.mesh_index];
			d.mesh_aabb_min = o.position + o.scale * mesh.local_min;
			d.mesh_aabb_max = o.position + o.scale * mesh.local_max;
			const auto image_view = [this](int index) -> DeviceImageView {
				if (index < 0 || static_cast<size_t>(index) >= images.size()) {
					return {};
				}
				return images[index]->view();
			};
			d.diffuse = image_view(o.texture_index);
			d.normal = image_view(o.normal_texture_index);
			d.specular = image_view(o.specular_texture_index);
			break;
		}
		default:
			return false;
	}
	return true;
}

void KernelInfo::update_object(const Scene& scene, int scene_index) {
	const auto& objects = scene.objects();
	if (scene_index < 0 || static_cast<size_t>(scene_index) >= objects.size()) {
		return;
	}

	if (objects.size() != descriptor_indices.size() ||
	    scene.meshes().size() != assets.meshes.size() ||
	    scene.textures().size() != assets.images.size() ||
	    !assets.matches_object_assets(scene, objects[scene_index])) {
		rebuild_world(scene);
		return;
	}

	DeviceObjectDesc updated{};
	const bool renderable = assets.describe_object(scene, objects[scene_index], updated);
	const int i = descriptor_indices[scene_index];
	if (renderable != (i >= 0)) {
		rebuild_world(scene);
	} else if (renderable) {
		update_scene_object<<<1, 1>>>(d_world, d_slots, d_object_ptr_buffer, d_desc_buffer, i, updated);
		check_cuda_errors(cudaGetLastError());
	}
}

void KernelInfo::rebuild_world(const Scene& scene) {
	// Finish queued renders before reusing their scene or asset storage.
	check_cuda_errors(cudaDeviceSynchronize());

	const size_t required = scene.objects().size();
	if (required > static_cast<size_t>(std::numeric_limits<int>::max())) {
		throw std::length_error("Too many scene objects for CUDA descriptor indices");
	}
	AssetCache next_assets(scene, assets);
	const bool assets_changed = assets.meshes != next_assets.meshes || assets.images != next_assets.images;

	// Keep the old pool live until the new world has been assembled. This
	// also preserves its views if a larger allocation cannot be obtained.
	SceneObjectSlot* slots = d_slots;
	Hittable** object_ptrs = d_object_ptr_buffer;
	DeviceObjectDesc* device_descs = d_desc_buffer;
	Hittable** sorted_objects = d_sorted_objects;
	Hittable** lights = d_lights;
	BVHNode* nodes = d_bvh_nodes;
	std::unique_ptr<SceneObjectSlot, decltype(&cudaFree)> new_pool(nullptr, cudaFree);
	size_t capacity = d_scene_capacity;
	if (required > capacity || (assets_changed && required > 0)) {
		// A binary BVH over N leaves requires at most 2*N-1 nodes, including
		// the one-child wrappers used for singleton ranges.
		static_assert(alignof(SceneObjectSlot) >= alignof(DeviceObjectDesc));
		static_assert(alignof(DeviceObjectDesc) >= alignof(BVHNode));
		static_assert(alignof(BVHNode) >= alignof(Hittable*));
		constexpr size_t bytes_per_slot = sizeof(SceneObjectSlot) + sizeof(DeviceObjectDesc)
		                                + 2 * sizeof(BVHNode) + 3 * sizeof(Hittable*);
		if (required > capacity) {
			capacity = std::max(required, capacity + capacity / 2);
		}
		cudaError_t result = cudaMalloc(&slots, capacity * bytes_per_slot);
		if (result == cudaErrorMemoryAllocation && capacity > required) {
			// Retry without spare slots when memory is tight.
			cudaGetLastError();
			capacity = required;
			result = cudaMalloc(&slots, capacity * bytes_per_slot);
		}
		check_cuda_errors(result);
		new_pool.reset(slots);
		device_descs = reinterpret_cast<DeviceObjectDesc*>(slots + capacity);
		nodes = reinterpret_cast<BVHNode*>(device_descs + capacity);
		object_ptrs = reinterpret_cast<Hittable**>(nodes + 2 * capacity);
		sorted_objects = object_ptrs + capacity;
		lights = sorted_objects + capacity;
	}

	// Invalid mesh objects are skipped. Retain the scene-to-descriptor mapping
	// so editing a later object still addresses the correct device slot.
	const auto& objects = scene.objects();
	std::vector<DeviceObjectDesc> descs;
	descs.reserve(objects.size());
	std::vector<int> new_indices(objects.size(), -1);
	for (size_t i = 0; i < objects.size(); ++i) {
		DeviceObjectDesc desc{};
		if (next_assets.describe_object(scene, objects[i], desc)) {
			new_indices[i] = static_cast<int>(descs.size());
			descs.push_back(desc);
		}
	}

	// ---- Upload descriptors and launch the device-side rebuild -----------
	int count = (int)descs.size();
	if (count > 0) {
		check_cuda_errors(cudaMemcpy(device_descs, descs.data(), descs.size() * sizeof(DeviceObjectDesc),
		                             cudaMemcpyHostToDevice));
		constexpr int block = 64;
		const int grid = 1 + (count - 1) / block;
		construct_objects_parallel<<<grid, block>>>(device_descs, count, slots, object_ptrs);
		check_cuda_errors(cudaGetLastError());
	}
	assemble_world<<<1, 1>>>(d_world, object_ptrs, device_descs, count, sorted_objects, lights, nodes);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());
	if (new_pool) {
		cudaFree(d_slots);
		d_slots = new_pool.release();
		d_desc_buffer = device_descs;
		d_bvh_nodes = nodes;
		d_object_ptr_buffer = object_ptrs;
		d_sorted_objects = sorted_objects;
		d_lights = lights;
		d_scene_capacity = capacity;
	}
	assets = std::move(next_assets);
	descriptor_indices = std::move(new_indices);
}
