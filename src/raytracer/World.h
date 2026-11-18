#pragma once

#include "Hittable.h"
#include "AABB.h"

class BVHNode;

class World : public Hittable {
public:
    // All views refer to the renderer's reusable scene allocation.
    int number_of_objects = 0;
    Hittable** objects = nullptr;
    Hittable* bvh_root = nullptr;
    Hittable** lights = nullptr;
    int num_lights = 0;

    __device__ World() : Hittable(HittableKind::World) {}

    // hit() body lives in HittableDispatch.h so it can route child calls
    // (bvh_root->hit, objects[i]->hit) through the tagged dispatcher.
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;

    __device__ virtual bool bounding_box(float time0, float time1, AABB& output_box) const;

    __device__ void build_bvh(Hittable** sorted_objects, BVHNode* nodes);
};

#ifdef __CUDACC__

#include "BVHNode.h"

__device__ void World::build_bvh(Hittable** sorted_objects, BVHNode* nodes) {
    bvh_root = nullptr;
    if (number_of_objects > 0) {
        // A split finishes evaluating candidate bounds before recursing. One
        // scratch array serves the whole tree, independent of its depth.
        AABB prefix[BVH_SAH_MAX_N];
        BVHNode* next_node = nodes + 1;
        bvh_root = new (nodes) BVHNode(sorted_objects, number_of_objects, 0.0f, 0.0f, next_node, prefix);
    }
}

__device__ AABB surrounding_box(AABB box0, AABB box1) {
    glm::vec3 lo(fminf(box0.min().x, box1.min().x),
        fminf(box0.min().y, box1.min().y),
        fminf(box0.min().z, box1.min().z));

    glm::vec3 hi(fmaxf(box0.max().x, box1.max().x),
        fmaxf(box0.max().y, box1.max().y),
        fmaxf(box0.max().z, box1.max().z));

    return AABB(lo, hi);
}

__device__ bool World::bounding_box(float time0, float time1, AABB& output_box) const {
    if (number_of_objects < 1) {
        return false;
    }

    AABB temp_box;
    bool first_box = true;

    for (int i = 0; i < number_of_objects; i++) {
        if (!objects[i]->bounding_box(time0, time1, temp_box)) {
            return false;
        }
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

#endif