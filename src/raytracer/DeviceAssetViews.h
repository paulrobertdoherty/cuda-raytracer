#pragma once

#include "MeshBVHNode.h"
#include <glm/glm.hpp>

// Trivially copyable, read-only views into the renderer's device asset cache.
// Descriptors, materials and geometry borrow these arrays for one world lifetime.
struct DeviceMeshView {
    const glm::vec3* vertices;
    const glm::vec3* normals;
    const glm::vec2* uvs;
    const int* indices;
    const MeshBVHNode* bvh;
    const int* triangle_ids;
};

struct DeviceImageView {
    const unsigned char* pixels;
    int width;
    int height;
    int channels;
};
