#pragma once

#include <glm/glm.hpp>
#include "Hittable.h"
#include "Material.h"
#include "AABB.h"
#include "MeshBVHNode.h"
#include "Constants.h"

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
	glm::vec3* d_normals;  // device pointer, count = vertex_count (may be nullptr)
	glm::vec2* d_uvs;      // device pointer, count = vertex_count (may be nullptr)
	int* d_indices;        // device pointer, count = index_count (multiple of 3)
	int vertex_count;
	int index_count;
	glm::vec3 translate;
	float scale;
	AABB world_aabb;
	Material* mat_ptr;

	// Per-mesh BVH (flat array in local space, built on host).
	MeshBVHNode* d_bvh_nodes;
	int bvh_node_count;
	int* d_reordered_tri_ids;
	int tri_id_count;

	__device__ TriangleMesh(glm::vec3* verts, glm::vec3* normals, glm::vec2* uvs, int vcount,
	                         int* indices, int icount,
	                         glm::vec3 translate, float scale,
	                         Material* mat,
	                         glm::vec3 world_min, glm::vec3 world_max,
	                         MeshBVHNode* bvh_nodes, int bvh_count,
	                         int* reordered_tri_ids, int tri_count)
		: d_vertices(verts), d_normals(normals), d_uvs(uvs), d_indices(indices),
		  vertex_count(vcount), index_count(icount),
		  translate(translate), scale(scale),
		  world_aabb(world_min, world_max), mat_ptr(mat),
		  d_bvh_nodes(bvh_nodes), bvh_node_count(bvh_count),
		  d_reordered_tri_ids(reordered_tri_ids), tri_id_count(tri_count) {}

	__device__ ~TriangleMesh() {
		delete mat_ptr;
	}

	// Inline AABB slab test operating on raw min/max + precomputed inv_dir.
	__device__ static bool slab_test(const glm::vec3& bmin, const glm::vec3& bmax,
	                                  const glm::vec3& origin, const glm::vec3& inv_dir,
	                                  float t_min, float t_max) {
		for (int a = 0; a < 3; a++) {
			float t0 = (bmin[a] - origin[a]) * inv_dir[a];
			float t1 = (bmax[a] - origin[a]) * inv_dir[a];
			if (inv_dir[a] < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;
			if (t_max <= t_min) return false;
		}
		return true;
	}

	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
		// Early-out against the mesh bounding box.
		if (!world_aabb.hit(r, t_min, t_max)) return false;

		// Inverse-transform ray from world space to local space.
		// World vertex = translate + scale * local_vertex, so:
		//   local_origin    = (ray.origin - translate) / scale
		//   local_direction = ray.direction / scale
		// The parametric t is preserved under uniform scaling.
		float inv_scale = 1.0f / scale;
		glm::vec3 local_origin = (r.origin - translate) * inv_scale;
		glm::vec3 local_dir    = r.direction * inv_scale;

		// Precompute inverse direction for AABB slab tests. If a component is
		// exactly zero, use a large finite value so (plane - origin) * inv_dir
		// cannot produce 0*inf = NaN. The sign is irrelevant because the
		// "parallel to slab" case is handled by the t_max/t_min comparisons:
		// with a huge magnitude, t0 and t1 blow up in opposite directions,
		// correctly reporting miss iff origin is outside the slab.
		constexpr float AXIS_PARALLEL_INV = 1e30f;
		glm::vec3 inv_dir = glm::vec3(
			local_dir.x != 0.0f ? 1.0f / local_dir.x : AXIS_PARALLEL_INV,
			local_dir.y != 0.0f ? 1.0f / local_dir.y : AXIS_PARALLEL_INV,
			local_dir.z != 0.0f ? 1.0f / local_dir.z : AXIS_PARALLEL_INV);

		// Stack-based iterative BVH traversal. 32 slots fits a balanced BVH
		// with ~4 billion triangles, so overflow here means the BVH is
		// pathologically unbalanced and something has gone wrong upstream.
		constexpr int MAX_STACK = 32;
		int stack[MAX_STACK];
		int stack_ptr = 0;
		stack[stack_ptr++] = 0; // push root

		bool hit_any = false;
		float closest = t_max;

		while (stack_ptr > 0) {
			int node_idx = stack[--stack_ptr];
			const MeshBVHNode& node = d_bvh_nodes[node_idx];

			// Test ray against this node's AABB.
			if (!slab_test(node.aabb_min, node.aabb_max,
			               local_origin, inv_dir, t_min, closest))
				continue;

			if (node.tri_count > 0) {
				// Leaf: test each triangle in the range.
				for (int i = 0; i < node.tri_count; i++) {
					int tri_id = d_reordered_tri_ids[node.tri_start + i];
					int i0 = d_indices[3 * tri_id + 0];
					int i1 = d_indices[3 * tri_id + 1];
					int i2 = d_indices[3 * tri_id + 2];

					glm::vec3 v0 = d_vertices[i0];
					glm::vec3 v1 = d_vertices[i1];
					glm::vec3 v2 = d_vertices[i2];

					// Moller-Trumbore in local space.
					glm::vec3 edge1 = v1 - v0;
					glm::vec3 edge2 = v2 - v0;
					glm::vec3 h = glm::cross(local_dir, edge2);
					float a = glm::dot(edge1, h);
					if (fabsf(a) < RAY_PARALLEL_EPS) continue;
					float f = 1.0f / a;
					glm::vec3 s = local_origin - v0;
					float u = f * glm::dot(s, h);
					if (u < 0.0f || u > 1.0f) continue;
					glm::vec3 q = glm::cross(s, edge1);
					float v = f * glm::dot(local_dir, q);
					if (v < 0.0f || u + v > 1.0f) continue;
					float t = f * glm::dot(edge2, q);
					if (t < t_min || t > closest) continue;

					closest = t;
					rec.t = t;
					rec.p = r.at(t); // world-space hit point
					// If UV buffer present, interpolate texture coordinates
					// from the triangle vertices using barycentric coords.
					glm::vec2 uv0(0), uv1(0), uv2(0);
					if (d_uvs) {
						uv0 = d_uvs[i0]; uv1 = d_uvs[i1]; uv2 = d_uvs[i2];
						glm::vec2 interp = (1.0f - u - v) * uv0 + u * uv1 + v * uv2;
						rec.u = interp.x;
						rec.v = interp.y;
					} else {
						rec.u = u;
						rec.v = v;
					}

					glm::vec3 outward_normal;
					if (d_normals) {
						glm::vec3 n0 = d_normals[i0];
						glm::vec3 n1 = d_normals[i1];
						glm::vec3 n2 = d_normals[i2];
						outward_normal = glm::normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
					} else {
						outward_normal = glm::normalize(glm::cross(edge1, edge2));
					}
					rec.set_face_normal(r, outward_normal);

					// Compute tangent and bitangent for normal mapping
					if (d_uvs) {
						glm::vec2 deltaUV1 = uv1 - uv0;
						glm::vec2 deltaUV2 = uv2 - uv0;

						float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

						if (isinf(f) || isnan(f)) {
							// Degenerate UVs, use arbitrary ONB
							glm::vec3 a = (fabsf(rec.normal.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
							rec.bitangent = glm::normalize(glm::cross(rec.normal, a));
							rec.tangent = glm::cross(rec.bitangent, rec.normal);
						} else {
							glm::vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
							rec.tangent = glm::normalize(tangent - rec.normal * glm::dot(rec.normal, tangent));
							rec.bitangent = glm::normalize(glm::cross(rec.normal, rec.tangent));
						}
					} else {
						// Build arbitrary ONB
						glm::vec3 a = (fabsf(rec.normal.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
						rec.bitangent = glm::normalize(glm::cross(rec.normal, a));
						rec.tangent = glm::cross(rec.bitangent, rec.normal);
					}

					rec.mat_ptr = mat_ptr;
					hit_any = true;
				}
			} else {
				// Internal node: push children, guarding the fixed-size stack.
				// Overflow here would silently drop subtrees and produce
				// missing-geometry artifacts; trap so the error is loud.
				if (stack_ptr + 2 > MAX_STACK) {
					__trap();
				}
				stack[stack_ptr++] = node.left_child;
				stack[stack_ptr++] = node.right_child;
			}
		}

		return hit_any;
	}

	__device__ bool bounding_box(float t0, float t1, AABB& output_box) const override {
		output_box = world_aabb;
		return true;
	}
};
