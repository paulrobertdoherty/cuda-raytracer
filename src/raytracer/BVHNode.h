#pragma once

#include "Hittable.h"
#include "AABB.h"

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

// Upper bound on top-level BVH size. The scene World caps capacity at a few
// dozen objects (see World.h), so 64 is comfortable headroom. Anything over
// this cap falls back to median split along the widest axis.
constexpr int BVH_SAH_MAX_N = 64;

class BVHNode : public Hittable {
public:
	__device__ BVHNode();

    __device__ BVHNode(Hittable** objects, int n, float time0, float time1, curandState* local_rand_state) {
        if (n == 1) {
            left = right = objects[0];
        } else if (n == 2) {
            left = objects[0];
            right = objects[1];
            AABB box_left, box_right;
            left->bounding_box(time0, time1, box_left);
            right->bounding_box(time0, time1, box_right);
            box = surrounding_box(box_left, box_right);
            return;
        } else {
            int best_axis = 0;
            int best_split = n / 2;

            if (n <= BVH_SAH_MAX_N) {
                // SAH: for each axis, sort and evaluate every split position
                // using prefix+suffix bounding boxes. O(3*n^2) worst case, but
                // n is small so this is cheap compared with ray traversal.
                float best_cost = FLT_MAX;
                AABB prefix[BVH_SAH_MAX_N];
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

            left  = new BVHNode(objects, best_split,
                                time0, time1, local_rand_state);
            right = new BVHNode(objects + best_split, n - best_split,
                                time0, time1, local_rand_state);
        }

        AABB box_left, box_right;

        if (!left->bounding_box(time0, time1, box_left) ||
            !right->bounding_box(time0, time1, box_right)) {
                return;
        }

        box = surrounding_box(box_left, box_right);
    }

	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (box.hit(r, t_min, t_max)) {
            HitRecord left_rec, right_rec;
            bool hit_left = left->hit(r, t_min, t_max, left_rec);
            bool hit_right = right->hit(r, t_min, t_max, right_rec);
            if (hit_left && hit_right) {
                if (left_rec.t < right_rec.t)
                    rec = left_rec;
                else
                    rec = right_rec;
                return true;
            }
            else if (hit_left) {
                rec = left_rec;
                return true;
            }
            else if (hit_right) {
                rec = right_rec;
                return true;
            }
            else
                return false;
        }
        else return false;
	}
    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = box;
        return true;
    }

	Hittable* left;
	Hittable* right;
	AABB box;

    __device__ ~BVHNode() {
        delete left;
        if (right != left) {
            delete right;
        }
    }
};