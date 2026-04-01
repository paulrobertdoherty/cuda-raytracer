#pragma once

#include "Hittable.h"
#include "AABB.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Simple device-side sort by bounding box axis (replaces thrust::sort to avoid
// CCCL device-side dispatch issues)
__device__ inline float bb_axis_min(Hittable* obj, int axis) {
    AABB box;
    obj->bounding_box(0, 0, box);
    if (axis == 0) return box.min().x;
    if (axis == 1) return box.min().y;
    return box.min().z;
}

__device__ inline void sort_hittables(Hittable** objects, int n, int axis) {
    // Insertion sort — fine for small n
    for (int i = 1; i < n; i++) {
        Hittable* key = objects[i];
        float key_val = bb_axis_min(key, axis);
        int j = i - 1;
        while (j >= 0 && bb_axis_min(objects[j], axis) > key_val) {
            objects[j + 1] = objects[j];
            j--;
        }
        objects[j + 1] = key;
    }
}

__device__ AABB surrounding_box(AABB box0, AABB box1);

class BVHNode : public Hittable {
public:
	__device__ BVHNode();

    __device__ BVHNode(Hittable** objects, int n, float time0, float time1, curandState* local_rand_state) {
        int axis = int(3 * curand_uniform(local_rand_state));
        sort_hittables(objects, n, axis);

        if (n == 1) {
            left = right = objects[0];
        } else if (n == 2) {
            left = objects[0];
            right = objects[1];
        } else {
            left = new BVHNode(objects, n / 2, time0, time1, local_rand_state);
            right = new BVHNode(objects + n / 2, n - n / 2, time0, time1, local_rand_state);
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