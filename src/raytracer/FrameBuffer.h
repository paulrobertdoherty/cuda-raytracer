#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <glm/vec4.hpp>
#include <glm/packing.hpp>
#include "Material.h"
#include "World.h"
#include "HittableDispatch.h"
#include "Constants.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288f
#endif

// The OpenGL PBO and headless output buffer both store BGRA8.
__device__ inline void write_pixel(uint32_t* pixels, int index, glm::vec4 color) {
	pixels[index] = glm::packUnorm4x8(glm::vec4(color.b, color.g, color.r, color.a));
}

// Rejects non-finite samples that would poison the accumulation buffer for
// the rest of the run (NaNs are sticky under +=, *=).
__device__ inline bool is_finite_vec3(const glm::vec3& v) {
	return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ inline glm::vec3 trace_color(const Ray& r, World* world, RandState* local_rand_state, int max_depth) {
	Ray cur_ray = r;
	glm::vec3 cur_attenuation = glm::vec3(1.0, 1.0, 1.0);
	glm::vec3 accumulated_color = glm::vec3(0.0f);
	bool count_emission = true;

	constexpr int RR_MIN_BOUNCES = 3;
	constexpr float RR_MAX_SURVIVAL = 0.95f;

	for (int i = 0; i < max_depth; i++) {
		// Russian Roulette path termination after minimum bounces
		if (i >= RR_MIN_BOUNCES) {
			float max_component = fmaxf(cur_attenuation.x, fmaxf(cur_attenuation.y, cur_attenuation.z));
			float survival_prob = fminf(max_component, RR_MAX_SURVIVAL);
			if (curand_uniform(local_rand_state) > survival_prob) {
				return accumulated_color;
			}
			// Compensate for terminated paths to keep estimate unbiased
			cur_attenuation /= survival_prob;
		}

		HitRecord rec;
		if (dispatch_hit(world, cur_ray, T_SELF_INTERSECT, FLT_MAX, rec)) {
			// Only count emission if not already handled by NEE on previous bounce
			if (count_emission) {
				glm::vec3 emit = cur_attenuation * rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
				if (is_finite_vec3(emit)) accumulated_color += emit;
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
					if (dispatch_sample_point(light, local_rand_state, light_point, light_normal)) {
						glm::vec3 to_light = light_point - rec.p;
						float dist_sq = glm::dot(to_light, to_light);
						float dist = sqrtf(dist_sq);
						glm::vec3 to_light_dir = to_light / dist;

						float cos_theta_surface = glm::dot(rec.normal, to_light_dir);
						float cos_theta_light = fabsf(glm::dot(light_normal, -to_light_dir));

						if (cos_theta_surface > 0.0f && cos_theta_light > 0.0f) {
							Ray shadow_ray(rec.p, to_light_dir);
							HitRecord shadow_rec;
							bool occluded = dispatch_hit(world, shadow_ray, T_SELF_INTERSECT,
								dist - T_SELF_INTERSECT, shadow_rec);

							if (!occluded) {
								HitRecord light_rec;
								if (dispatch_hit(light, Ray(rec.p, to_light_dir), T_SELF_INTERSECT,
										dist + 0.01f, light_rec)) {
									glm::vec3 Le = light_rec.mat_ptr->emitted(
										light_rec.u, light_rec.v, light_rec.p);

									float light_area = dispatch_area(light);
									glm::vec3 nee = Le * (attenuation / (float)M_PI)
										* cos_theta_surface
										* (cos_theta_light / dist_sq)
										* light_area
										* (float)world->num_lights;

									glm::vec3 nee_contrib = cur_attenuation * nee;
									if (is_finite_vec3(nee_contrib))
										accumulated_color += nee_contrib;
								}
							}
						}
					}
					count_emission = false;
				} else {
					count_emission = true;
				}

				cur_attenuation *= attenuation;
				if (!is_finite_vec3(cur_attenuation)) return accumulated_color;
				cur_ray = scattered;
			}
			else {
				return accumulated_color;
			}
		}
		else {
			return accumulated_color;
		}
	}
	return accumulated_color;
}
