#pragma once

#include "raytracer/Camera.h"

#include <glm/glm.hpp>

// Host-side helpers for turning cursor coordinates into world-space rays and
// for reprojecting cursor motion onto a drag plane. All state lives in Input;
// these are pure functions.
namespace Picker {

// Compute the forward + up basis the rasterizer / camera uses. Matches the
// (slightly odd) Euler formula in Camera.h so picking aligns with display.
void compute_basis(const CameraInfo& cam, glm::vec3& forward, glm::vec3& up);

// Convert a cursor position (in window pixels, top-left origin) into a
// world-space ray. The ray is constructed from the same camera basis used
// by the raytracer so picked points match the rasterized preview.
void cursor_to_ray(const CameraInfo& cam,
                   double cursor_x, double cursor_y,
                   int window_w, int window_h,
                   glm::vec3& out_origin, glm::vec3& out_dir);

// Intersect a ray with a plane defined by a point + normal. Returns true on
// success and writes the intersection point into `out_point`.
bool intersect_plane(const glm::vec3& ray_origin,
                     const glm::vec3& ray_dir,
                     const glm::vec3& plane_point,
                     const glm::vec3& plane_normal,
                     glm::vec3& out_point);

} // namespace Picker
