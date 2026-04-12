#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"

// A full indexed triangle mesh packed into one Hittable. Unlike individual
// Triangle objects, the mesh owns its vertex and index buffers and its own
// Material, so a single `delete` (in the destructor) frees everything
// exactly once — no double-free when thousands of triangles share a material.
//
// The buffers are supplied by the host via cudaMalloc; this class does NOT
// free them (KernelInfo keeps track of them separately and releases them in
// its destructor or on the next world rebuild). The material IS owned and
// deleted with the mesh.
class TriangleMesh : public Hittable {
public:
	glm::vec3* d_vertices; // device pointer, count = vertex_count
	int* d_indices;        // device pointer, count = index_count (multiple of 3)
	int vertex_count;
	int index_count;
	glm::vec3 translate;
	float scale;
	AABB world_aabb;
	Material* mat_ptr;

	__device__ TriangleMesh(glm::vec3* verts, int vcount,
	                         int* indices, int icount,
	                         glm::vec3 translate, float scale,
	                         Material* mat,
	                         glm::vec3 world_min, glm::vec3 world_max)
		: d_vertices(verts), d_indices(indices),
		  vertex_count(vcount), index_count(icount),
		  translate(translate), scale(scale),
		  world_aabb(world_min, world_max), mat_ptr(mat) {}

	__device__ ~TriangleMesh() {
		delete mat_ptr;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		// Early-out against the mesh bounding box.
		if (!world_aabb.hit(r, t_min, t_max)) return false;

		bool hit_any = false;
		float closest = t_max;
		int tri_count = index_count / 3;
		for (int i = 0; i < tri_count; i++) {
			int i0 = d_indices[3 * i + 0];
			int i1 = d_indices[3 * i + 1];
			int i2 = d_indices[3 * i + 2];
			glm::vec3 v0 = translate + scale * d_vertices[i0];
			glm::vec3 v1 = translate + scale * d_vertices[i1];
			glm::vec3 v2 = translate + scale * d_vertices[i2];

			// Moller-Trumbore (matches src/raytracer/Triangle.h)
			glm::vec3 edge1 = v1 - v0;
			glm::vec3 edge2 = v2 - v0;
			glm::vec3 h = glm::cross(r.direction, edge2);
			float a = glm::dot(edge1, h);
			if (fabsf(a) < 1e-8f) continue;
			float f = 1.0f / a;
			glm::vec3 s = r.origin - v0;
			float u = f * glm::dot(s, h);
			if (u < 0.0f || u > 1.0f) continue;
			glm::vec3 q = glm::cross(s, edge1);
			float v = f * glm::dot(r.direction, q);
			if (v < 0.0f || u + v > 1.0f) continue;
			float t = f * glm::dot(edge2, q);
			if (t < t_min || t > closest) continue;

			closest = t;
			rec.t = t;
			rec.p = r.at(t);
			rec.u = u;
			rec.v = v;
			rec.mat_ptr = mat_ptr;
			glm::vec3 outward_normal = glm::normalize(glm::cross(edge1, edge2));
			rec.set_face_normal(r, outward_normal);
			hit_any = true;
		}
		return hit_any;
	}

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const override {
		output_box = world_aabb;
		return true;
	}
};
