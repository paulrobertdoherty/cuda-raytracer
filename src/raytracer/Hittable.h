#pragma once

#include "Ray.h"
#include <glm/glm.hpp>
#include <curand_kernel.h>

// Use Philox counter-based RNG for per-pixel state: smaller working set and
// better throughput than the default XORWOW generator on modern GPUs.
using RandState = curandStatePhilox4_32_10_t;

#include "AABB.h"

class Material;

struct HitRecord {
	float t;
	float u, v;
	glm::vec3 p;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 bitangent;
	Material* mat_ptr;
	bool front_face;

	__device__ inline void set_face_normal(const Ray& r, const glm::vec3& outward_normal) {
		front_face = glm::dot(r.direction, outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hittable {
public:
	__device__ virtual ~Hittable() {}
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
	__device__ virtual bool bounding_box(float time0, float time1, AABB& output_box) const = 0;
	__device__ virtual bool sample_point(RandState* rng, glm::vec3& point, glm::vec3& normal_out) const {
		return false;
	}
	__device__ virtual float area() const { return 0.0f; }
};