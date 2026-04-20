#pragma once

#include "Hittable.h"
#include "AABB.h"

#include "thrust/device_ptr.h"
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"

#include <curand_kernel.h>



class World : public Hittable {
public:
    int number_of_objects;
    int capacity;
    Hittable** objects;
    Hittable* bvh_root;

    Hittable** lights;
    int num_lights;
    int lights_capacity;

    __device__ World() : World(8) {}

    __device__ World(int initial_capacity) {
        int cap = initial_capacity > 0 ? initial_capacity : 8;
        objects = new Hittable*[cap];
        number_of_objects = 0;
        this->capacity = cap;
        bvh_root = nullptr;
        lights = new Hittable*[cap];
        num_lights = 0;
        lights_capacity = cap;
    }

    __device__ ~World();

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (bvh_root) {
            return bvh_root->hit(r, t_min, t_max, rec);
        }
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < number_of_objects; i++) {
            if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ virtual bool bounding_box(float time0, float time1, AABB& output_box) const;

    __device__ void build_bvh(curandState* rand_state);

    __device__ bool add(Hittable* object) {
        if (number_of_objects >= capacity) {
            int new_cap = capacity * 2;
            Hittable** bigger = new Hittable*[new_cap];
            for (int i = 0; i < number_of_objects; i++) bigger[i] = objects[i];
            delete[] objects;
            objects = bigger;
            capacity = new_cap;
        }

        objects[number_of_objects] = object;
        number_of_objects++;
        return true;
    }

    __device__ bool add_light(Hittable* object) {
        if (num_lights >= lights_capacity) {
            int new_cap = lights_capacity * 2;
            Hittable** bigger = new Hittable*[new_cap];
            for (int i = 0; i < num_lights; i++) bigger[i] = lights[i];
            delete[] lights;
            lights = bigger;
            lights_capacity = new_cap;
        }
        lights[num_lights++] = object;
        return true;
    }

};

/*
* This is a pretty bad way to do this, but I honestly don't know how to do it better
* During the compilation it is checked if the code is being compiled by nvcc
* If it is, then the destructor is defined
* If it is not, then the destructor is not defined to avoid multiple definitions
*/ 

#ifdef __CUDACC__

#include "BVHNode.h"

__device__ void World::build_bvh(curandState* rand_state) {
    if (number_of_objects > 0) {
        bvh_root = new BVHNode(objects, number_of_objects, 0.0f, 0.0f, rand_state);
    }
}

__device__ World::~World() {
    if (bvh_root) {
        delete bvh_root;
    } else {
        for (int i = 0; i < number_of_objects; i++) {
            delete objects[i];
        }
    }
    delete[] objects;
    delete[] lights;
}

__device__ AABB surrounding_box(AABB box0, AABB box1) {
    glm::vec3 lo(fmin(box0.min().x, box1.min().x),
        fmin(box0.min().y, box1.min().y),
        fmin(box0.min().z, box1.min().z));

    glm::vec3 hi(fmax(box0.max().x, box1.max().x),
        fmax(box0.max().y, box1.max().y),
        fmax(box0.max().z, box1.max().z));

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