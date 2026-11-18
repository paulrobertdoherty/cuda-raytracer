#pragma once

#include "Hittable.h"
#include "AABB.h"
#include <new>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Simple device-side sort by bounding box centroid along one axis. The
// top-level BVH has at most a few dozen entries so insertion sort is fine
// and avoids the CCCL device-side dispatch required for thrust::sort.
__device__ inline float bb_axis_centroid(Hittable* obj, int axis) {
    AABB box;
    obj->bounding_box(0, 0, box);
    if (axis == 0) return 0.5f * (box.min().x + box.max().x);
    if (axis == 1) return 0.5f * (box.min().y + box.max().y);
    return 0.5f * (box.min().z + box.max().z);
}

__device__ inline void sort_hittables(Hittable** objects, int n, int axis) {
    for (int i = 1; i < n; i++) {
        Hittable* key = objects[i];
        float key_val = bb_axis_centroid(key, axis);
        int j = i - 1;
        while (j >= 0 && bb_axis_centroid(objects[j], axis) > key_val) {
            objects[j + 1] = objects[j];
            j--;
        }
        objects[j + 1] = key;
    }
}

__device__ AABB surrounding_box(AABB box0, AABB box1);

__device__ inline float bvh_surface_area(const AABB& b) {
    glm::vec3 d = b.max() - b.min();
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Larger ranges use a median split to bound temporary stack usage.
constexpr int BVH_SAH_MAX_N = 64;

class BVHNode : public Hittable {
public:
	__device__ BVHNode();

    __device__ BVHNode(Hittable** objects, int n, float time0, float time1, BVHNode*& next_node, AABB* prefix)
        : Hittable(HittableKind::BVHNode) {
        if (n == 1) {
            left = right = objects[0];
        } else if (n == 2) {
            left = objects[0];
            right = objects[1];
        } else {
            int best_axis = 0;
            int best_split = n / 2;

            if (n <= BVH_SAH_MAX_N) {
                // SAH: for each axis, sort and evaluate every split position
                // using prefix+suffix bounding boxes. O(3*n^2) worst case, but
                // n is small so this is cheap compared with ray traversal.
                float best_cost = FLT_MAX;
                for (int axis = 0; axis < 3; axis++) {
                    sort_hittables(objects, n, axis);

                    AABB b;
                    objects[0]->bounding_box(time0, time1, b);
                    prefix[0] = b;
                    for (int i = 1; i < n; i++) {
                        objects[i]->bounding_box(time0, time1, b);
                        prefix[i] = surrounding_box(prefix[i - 1], b);
                    }

                    AABB suffix;
                    objects[n - 1]->bounding_box(time0, time1, suffix);
                    for (int s = n - 1; s >= 1; s--) {
                        float cost = s * bvh_surface_area(prefix[s - 1])
                                   + (n - s) * bvh_surface_area(suffix);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_axis = axis;
                            best_split = s;
                        }
                        objects[s - 1]->bounding_box(time0, time1, b);
                        suffix = surrounding_box(suffix, b);
                    }
                }
                // Re-sort by the winning axis so objects[0..best_split) holds the
                // left partition and objects[best_split..n) the right one.
                sort_hittables(objects, n, best_axis);
            } else {
                // Fallback for oversize input: pick the widest centroid axis
                // and split at the median. Still better than random axis.
                AABB cbox;
                objects[0]->bounding_box(time0, time1, cbox);
                glm::vec3 cmin = cbox.min();
                glm::vec3 cmax = cbox.max();
                for (int i = 1; i < n; i++) {
                    objects[i]->bounding_box(time0, time1, cbox);
                    cmin = glm::min(cmin, cbox.min());
                    cmax = glm::max(cmax, cbox.max());
                }
                glm::vec3 extent = cmax - cmin;
                best_axis = 0;
                if (extent.y > extent.x && extent.y >= extent.z) best_axis = 1;
                else if (extent.z > extent.x) best_axis = 2;
                sort_hittables(objects, n, best_axis);
                best_split = n / 2;
            }

            left  = new (next_node++) BVHNode(objects, best_split,
                                time0, time1, next_node, prefix);
            right = new (next_node++) BVHNode(objects + best_split, n - best_split,
                                time0, time1, next_node, prefix);
        }

        left->bvh_parent = this;
        right->bvh_parent = this;
        box = child_bounds(time0, time1);
    }

    __device__ AABB child_bounds(float time0, float time1) const {
        AABB box_left, box_right;
        left->bounding_box(time0, time1, box_left);
        right->bounding_box(time0, time1, box_right);
        return surrounding_box(box_left, box_right);
    }

    __device__ void refit_ancestors() {
        for (BVHNode* node = this; node; node = node->bvh_parent) {
            const AABB updated = node->child_bounds(0.0f, 0.0f);
            if (updated.min() == node->box.min() && updated.max() == node->box.max()) {
                break;
            }
            node->box = updated;
        }
    }

	// hit() body lives in HittableDispatch.h so it can call the tagged
	// dispatcher (which in turn must know about all subclasses). Declared
	// inline here, defined after the dispatcher is in scope.
	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = box;
        return true;
    }

	Hittable* left;
	Hittable* right;
	AABB box;

};