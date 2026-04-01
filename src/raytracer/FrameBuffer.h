#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include "device_launch_parameters.h"
#include "Hittable.h"
#include <curand_kernel.h>


#include "cuda_errors.h"

class World;

class FrameBuffer {
public:
	uint32_t* device_ptr; // RGBA8 internal format, but uses BGRA
	size_t buffer_size;
	unsigned int width;
	unsigned int height;
	int max_depth;


	__host__ FrameBuffer(unsigned int width, unsigned int height, int max_depth);

	__device__ void writePixel(int x, int y, glm::vec4 pixel);

	__device__ glm::vec3 color(const Ray& r, World* world, curandState* local_rand_state);
};

#ifdef __CUDACC__
#include "Material.h"
#include "World.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288f
#endif

__host__ FrameBuffer::FrameBuffer(unsigned int width, unsigned int height, int max_depth) : width{ width }, height{ height }, max_depth{ max_depth } {}

__device__ void FrameBuffer::writePixel(int x, int y, glm::vec4 pixel) {
	int idx = y * width + x;
	// convert RGBA to BGRA that buffer uses
	device_ptr[idx] = glm::packUnorm4x8(glm::vec4(pixel.b, pixel.g, pixel.r, pixel.a));
}

__device__ glm::vec3 FrameBuffer::color(const Ray& r, World* world, curandState* local_rand_state) {
	Ray cur_ray = r;
	glm::vec3 cur_attenuation = glm::vec3(1.0, 1.0, 1.0);
	glm::vec3 accumulated_color = glm::vec3(0.0f);
	bool count_emission = true;

	for (int i = 0; i < max_depth; i++) {
		HitRecord rec;
		if (world->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			// Only count emission if not already handled by NEE on previous bounce
			if (count_emission) {
				accumulated_color += cur_attenuation * rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			}

			Ray scattered;
			glm::vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				bool specular = rec.mat_ptr->is_specular();

				// NEE: direct light sampling for diffuse surfaces
				if (!specular && world->num_lights > 0) {
					int light_idx = (int)(curand_uniform(local_rand_state) * world->num_lights);
					if (light_idx >= world->num_lights) light_idx = world->num_lights - 1;
					Hittable* light = world->lights[light_idx];

					glm::vec3 light_point, light_normal;
					if (light->sample_point(local_rand_state, light_point, light_normal)) {
						glm::vec3 to_light = light_point - rec.p;
						float dist_sq = glm::dot(to_light, to_light);
						float dist = sqrtf(dist_sq);
						glm::vec3 to_light_dir = to_light / dist;

						float cos_theta_surface = glm::dot(rec.normal, to_light_dir);
						float cos_theta_light = fabsf(glm::dot(light_normal, -to_light_dir));

						if (cos_theta_surface > 0.0f && cos_theta_light > 0.0f) {
							Ray shadow_ray(rec.p, to_light_dir);
							HitRecord shadow_rec;
							bool occluded = world->hit(shadow_ray, 0.001f, dist - 0.001f, shadow_rec);

							if (!occluded) {
								HitRecord light_rec;
								if (light->hit(Ray(rec.p, to_light_dir), 0.001f, dist + 0.01f, light_rec)) {
									glm::vec3 Le = light_rec.mat_ptr->emitted(
										light_rec.u, light_rec.v, light_rec.p);

									float light_area = light->area();
									glm::vec3 nee = Le * (attenuation / (float)M_PI)
										* cos_theta_surface
										* (cos_theta_light / dist_sq)
										* light_area
										* (float)world->num_lights;

									accumulated_color += cur_attenuation * nee;
								}
							}
						}
					}
					count_emission = false;
				} else {
					count_emission = true;
				}

				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return accumulated_color;
			}
		}
		else {
			glm::vec3 unit_direction = glm::normalize(cur_ray.direction);
			float t = 0.5f * (unit_direction.y + 1.0f);
			glm::vec3 c = (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
			return accumulated_color + cur_attenuation * c;
		}
	}
	return accumulated_color;
}
#endif