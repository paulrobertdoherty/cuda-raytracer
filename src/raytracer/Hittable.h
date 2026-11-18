#pragma once

#include "Ray.h"
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <cstdint>

// Use Philox counter-based RNG for per-pixel state: smaller working set and
// better throughput than the default XORWOW generator on modern GPUs.
using RandState = curandStatePhilox4_32_10_t;

#include "AABB.h"

class Material;
class BVHNode;

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

// Tag used by HittableDispatch.h to devirtualize hit()/sample_point()/area()
// on the per-bounce hot path. Each subclass passes its kind to the base ctor
// below; the dispatcher switches on this tag and static_casts to the right
// subclass for an inline, qualified call (no vtable load).
enum class HittableKind : uint8_t {
	Sphere,
	Triangle,
	Rect,
	Disc,
	TriangleMesh,
	BVHNode,
	World,
};

class Hittable {
public:
	HittableKind kind;
	int light_index = -1;
	BVHNode* bvh_parent = nullptr;

	__device__ Hittable() : kind(HittableKind::Sphere) {} // sentinel, must be overwritten
	__device__ Hittable(HittableKind k) : kind(k) {}

	__device__ virtual ~Hittable() {}
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
	__device__ virtual bool bounding_box(float time0, float time1, AABB& output_box) const = 0;
	__device__ virtual bool sample_point(RandState* rng, glm::vec3& point, glm::vec3& normal_out) const {
		return false;
	}
	__device__ virtual float area() const { return 0.0f; }
};
