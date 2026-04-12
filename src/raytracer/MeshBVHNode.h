#pragma once

#include <glm/glm.hpp>

// Flat BVH node for per-mesh acceleration. Built on the host, uploaded as a
// contiguous array to the device, and traversed iteratively with an explicit
// stack. All coordinates are in LOCAL mesh space (before translate/scale).
//
// Discrimination: tri_count > 0 means leaf, tri_count == 0 means internal.
struct MeshBVHNode {
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;

    // Internal nodes: indices into the flat node array.
    int left_child;
    int right_child;

    // Leaf nodes: range into the reordered triangle-ID array.
    int tri_start;
    int tri_count;   // >0 = leaf, 0 = internal node
};
