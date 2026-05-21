#pragma once

// Devirtualized dispatch for Hittable::hit on the per-bounce hot path.
//
// `obj->hit(...)` through a Hittable* normally goes through a vtable; warps
// that hit different geometry subclasses serialize on the indirect call.
// `dispatch_hit` switches on Hittable::kind and issues a qualified, non-
// virtual call to the concrete subclass — the compiler inlines every case
// body and lets PTXAS CSE shared loads across cases.
//
// This header is the single point that knows about every subclass, so it
// must be included after Hittable.h + all subclass headers. BVHNode::hit
// and World::hit are also defined here (rather than in their own headers)
// because they call into Hittable* children and therefore need the
// dispatcher in scope — putting the dispatcher in their headers would
// create a circular include.

#include "Hittable.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Rectangle.h"
#include "Disc.h"
#include "TriangleMesh.h"
#include "BVHNode.h"
#include "World.h"

__device__ inline bool dispatch_hit(const Hittable* obj, const Ray& r,
                                     float t_min, float t_max, HitRecord& rec) {
	switch (obj->kind) {
		case HittableKind::Sphere:
			return static_cast<const Sphere*>(obj)->Sphere::hit(r, t_min, t_max, rec);
		case HittableKind::Triangle:
			return static_cast<const Triangle*>(obj)->Triangle::hit(r, t_min, t_max, rec);
		case HittableKind::Rect:
			return static_cast<const Rect*>(obj)->Rect::hit(r, t_min, t_max, rec);
		case HittableKind::Disc:
			return static_cast<const Disc*>(obj)->Disc::hit(r, t_min, t_max, rec);
		case HittableKind::TriangleMesh:
			return static_cast<const TriangleMesh*>(obj)->TriangleMesh::hit(r, t_min, t_max, rec);
		case HittableKind::BVHNode:
			return static_cast<const BVHNode*>(obj)->BVHNode::hit(r, t_min, t_max, rec);
		case HittableKind::World:
			return static_cast<const World*>(obj)->World::hit(r, t_min, t_max, rec);
	}
	return false;
}

__device__ inline bool dispatch_sample_point(const Hittable* obj, RandState* rng,
                                              glm::vec3& point, glm::vec3& normal_out) {
	switch (obj->kind) {
		case HittableKind::Sphere:
			return static_cast<const Sphere*>(obj)->Sphere::sample_point(rng, point, normal_out);
		case HittableKind::Triangle:
			return static_cast<const Triangle*>(obj)->Triangle::sample_point(rng, point, normal_out);
		case HittableKind::Rect:
			return static_cast<const Rect*>(obj)->Rect::sample_point(rng, point, normal_out);
		case HittableKind::Disc:
			return static_cast<const Disc*>(obj)->Disc::sample_point(rng, point, normal_out);
		case HittableKind::TriangleMesh:
		case HittableKind::BVHNode:
		case HittableKind::World:
			return false;
	}
	return false;
}

__device__ inline float dispatch_area(const Hittable* obj) {
	switch (obj->kind) {
		case HittableKind::Sphere:   return static_cast<const Sphere*>(obj)->Sphere::area();
		case HittableKind::Triangle: return static_cast<const Triangle*>(obj)->Triangle::area();
		case HittableKind::Rect:     return static_cast<const Rect*>(obj)->Rect::area();
		case HittableKind::Disc:     return static_cast<const Disc*>(obj)->Disc::area();
		case HittableKind::TriangleMesh:
		case HittableKind::BVHNode:
		case HittableKind::World:
			return 0.0f;
	}
	return 0.0f;
}

// BVHNode::hit — descends into children via dispatch_hit (devirtualized).
__device__ inline bool BVHNode::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
	if (box.hit(r, t_min, t_max)) {
		HitRecord left_rec, right_rec;
		bool hit_left  = dispatch_hit(left,  r, t_min, t_max, left_rec);
		bool hit_right = dispatch_hit(right, r, t_min, t_max, right_rec);
		if (hit_left && hit_right) {
			rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
			return true;
		}
		if (hit_left)  { rec = left_rec;  return true; }
		if (hit_right) { rec = right_rec; return true; }
		return false;
	}
	return false;
}

// World::hit — routes through the BVH if built, else linear scan; both via
// dispatch_hit so lane divergence over Hittable subclasses doesn't go
// through the vtable.
__device__ inline bool World::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
	if (bvh_root) {
		return dispatch_hit(bvh_root, r, t_min, t_max, rec);
	}
	HitRecord temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;
	for (int i = 0; i < number_of_objects; i++) {
		if (dispatch_hit(objects[i], r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}
