#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"

class Rect : public Hittable {
public:
	glm::vec3 Q;       // corner point
	glm::vec3 u, v;    // edge vectors
	Material* mat_ptr;

	// precomputed
	glm::vec3 normal;
	float D;
	glm::vec3 w;

	__device__ Rect() {}

	__device__ Rect(glm::vec3 Q, glm::vec3 u, glm::vec3 v, Material* mat)
		: Q(Q), u(u), v(v), mat_ptr(mat) {
		glm::vec3 n = glm::cross(u, v);
		normal = glm::normalize(n);
		D = glm::dot(normal, Q);
		w = n / glm::dot(n, n);
	}

	__device__ ~Rect() {
		delete mat_ptr;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		float denom = glm::dot(normal, r.direction);
		// parallel to plane
		if (fabsf(denom) < 1e-8f) return false;

		float t = (D - glm::dot(normal, r.origin)) / denom;
		if (t < t_min || t > t_max) return false;

		glm::vec3 intersection = r.at(t);
		glm::vec3 planar = intersection - Q;

		float alpha = glm::dot(w, glm::cross(planar, v));
		float beta  = glm::dot(w, glm::cross(u, planar));

		// check if inside the quad [0,1] x [0,1]
		if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
			return false;

		rec.t = t;
		rec.p = intersection;
		rec.u = alpha;
		rec.v = beta;
		rec.mat_ptr = mat_ptr;
		rec.set_face_normal(r, normal);
		return true;
	}

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const override {
		glm::vec3 p0 = Q;
		glm::vec3 p1 = Q + u;
		glm::vec3 p2 = Q + v;
		glm::vec3 p3 = Q + u + v;

		glm::vec3 mn = glm::min(glm::min(p0, p1), glm::min(p2, p3));
		glm::vec3 mx = glm::max(glm::max(p0, p1), glm::max(p2, p3));

		// pad thin axes
		glm::vec3 padding(0.0001f);
		output_box = AABB(mn - padding, mx + padding);
		return true;
	}
};
