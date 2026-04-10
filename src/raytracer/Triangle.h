#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"

class Triangle : public Hittable {
public:
	glm::vec3 v0, v1, v2;
	Material* mat_ptr;

	__device__ Triangle() {}

	__device__ Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, Material* mat)
		: v0(v0), v1(v1), v2(v2), mat_ptr(mat) {}

	__device__ ~Triangle() {
		delete mat_ptr;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		// Moller-Trumbore intersection
		glm::vec3 edge1 = v1 - v0;
		glm::vec3 edge2 = v2 - v0;
		glm::vec3 h = glm::cross(r.direction, edge2);
		float a = glm::dot(edge1, h);

		if (fabsf(a) < 1e-8f) return false;

		float f = 1.0f / a;
		glm::vec3 s = r.origin - v0;
		float u = f * glm::dot(s, h);
		if (u < 0.0f || u > 1.0f) return false;

		glm::vec3 q = glm::cross(s, edge1);
		float v = f * glm::dot(r.direction, q);
		if (v < 0.0f || u + v > 1.0f) return false;

		float t = f * glm::dot(edge2, q);
		if (t < t_min || t > t_max) return false;

		rec.t = t;
		rec.p = r.at(t);
		rec.u = u;
		rec.v = v;
		rec.mat_ptr = mat_ptr;
		glm::vec3 outward_normal = glm::normalize(glm::cross(edge1, edge2));
		rec.set_face_normal(r, outward_normal);
		return true;
	}

	__device__ bool sample_point(RandState* rng, glm::vec3& point, glm::vec3& normal_out) const override {
		float u_r = curand_uniform(rng);
		float v_r = curand_uniform(rng);
		if (u_r + v_r > 1.0f) { u_r = 1.0f - u_r; v_r = 1.0f - v_r; }
		point = v0 + u_r * (v1 - v0) + v_r * (v2 - v0);
		normal_out = glm::normalize(glm::cross(v1 - v0, v2 - v0));
		return true;
	}

	__device__ float area() const override { return 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0)); }

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const override {
		glm::vec3 mn = glm::min(glm::min(v0, v1), v2);
		glm::vec3 mx = glm::max(glm::max(v0, v1), v2);

		// pad thin axes
		glm::vec3 padding(0.0001f);
		output_box = AABB(mn - padding, mx + padding);
		return true;
	}
};
