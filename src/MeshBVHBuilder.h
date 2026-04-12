#pragma once

#include <glm/glm.hpp>
#include "raytracer/MeshBVHNode.h"

#include <vector>

struct MeshBVHBuildResult {
    std::vector<MeshBVHNode> nodes;       // flat node array (root at index 0)
    std::vector<int> reordered_tri_ids;   // triangle IDs in traversal order
};

// Build a flat BVH over the triangles of a mesh. Coordinates are in local
// mesh space. The caller supplies vertex positions and a raw index buffer
// (3 consecutive indices per triangle).
//
// Returns the flat node array and a reordered triangle-ID array. Leaf nodes
// reference ranges in the reordered array; each entry is a triangle ID t
// meaning the triangle with vertex indices [indices[3*t], indices[3*t+1],
// indices[3*t+2]].
MeshBVHBuildResult build_mesh_bvh(
    const glm::vec3* positions,
    const unsigned int* indices,
    int tri_count,
    int max_leaf_size = 4
);
