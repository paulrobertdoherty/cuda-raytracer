#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"
#include "Constants.h"

#define DISC_PI 3.14159265358979323846f

// A flat disc (filled circle) in a plane defined by a center point and normal.
// Useful as an emissive area light that looks more physically natural than a
// rectangle. Supports NEE via sample_point() + area().
class Disc : public Hittable {
public:
	glm::vec3 center;
	glm::vec3 normal;  // unit normal
	float     radius;
	Material* mat_ptr;

	// Precomputed
	float     D;         // plane constant: dot(normal, center)
	glm::vec3 tangent;   // in-plane axis 1
	glm::vec3 bitangent; // in-plane axis 2

	__device__ Disc() {}

	__device__ Disc(glm::vec3 center, glm::vec3 normal, float radius, Material* mat)
		: center(center), normal(glm::normalize(normal)), radius(radius), mat_ptr(mat) {
		D = glm::dot(this->normal, this->center);
		// Branchless ONB (Pixar / Frisvad-Duff) keeps construction divergence-free.
		float sign = copysignf(1.0f, this->normal.z);
		float a    = -1.0f / (sign + this->normal.z);
		float bxy  = this->normal.x * this->normal.y * a;
		tangent   = glm::vec3(1.0f + sign * this->normal.x * this->normal.x * a,
		                      sign * bxy,
		                      -sign * this->normal.x);
		bitangent = glm::vec3(bxy,
		                      sign + this->normal.y * this->normal.y * a,
		                      -this->normal.y);
	}

	__device__ ~Disc() {
		delete mat_ptr;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		float denom = glm::dot(normal, r.direction);
		if (fabsf(denom) < RAY_PARALLEL_EPS) return false;

		float t = (D - glm::dot(normal, r.origin)) / denom;
		if (t < t_min || t > t_max) return false;

		glm::vec3 p = r.at(t);
		glm::vec3 d = p - center;
		if (glm::dot(d, d) > radius * radius) return false;

		rec.t = t;
		rec.p = p;
		// UV: project onto tangent/bitangent, remap to [0,1]
		rec.u = glm::dot(d, tangent)   / radius * 0.5f + 0.5f;
		rec.v = glm::dot(d, bitangent) / radius * 0.5f + 0.5f;
		rec.mat_ptr = mat_ptr;
		rec.set_face_normal(r, normal);
		return true;
	}

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const override {
		// Extent along axis i = radius * sqrt(1 - normal[i]^2)
		glm::vec3 half(
			radius * sqrtf(fmaxf(0.0f, 1.0f - normal.x * normal.x)),
			radius * sqrtf(fmaxf(0.0f, 1.0f - normal.y * normal.y)),
			radius * sqrtf(fmaxf(0.0f, 1.0f - normal.z * normal.z))
		);
		glm::vec3 padding(0.0001f);
		output_box = AABB(center - half - padding, center + half + padding);
		return true;
	}

	__device__ bool sample_point(RandState* rng, glm::vec3& point, glm::vec3& normal_out) const override {
		// Uniform disc sampling: r = R*sqrt(xi), theta = 2pi*xi2
		float r   = radius * sqrtf(curand_uniform(rng));
		float phi = 2.0f * DISC_PI * curand_uniform(rng);
		float sp, cp;
		__sincosf(phi, &sp, &cp);
		point      = center + r * (cp * tangent + sp * bitangent);
		normal_out = normal;
		return true;
	}

	__device__ float area() const override {
		return DISC_PI * radius * radius;
	}
};
