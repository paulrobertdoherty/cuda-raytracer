#include "Picker.h"

#include <cmath>

namespace Picker {

void compute_basis(const CameraInfo& cam, glm::vec3& forward, glm::vec3& up) {
	float A = degrees_to_radians(cam.rotation.x);
	float B = degrees_to_radians(cam.rotation.y);
	float C = degrees_to_radians(cam.rotation.z);

	if (cam.rotation.x == 0 && cam.rotation.y == 0 && cam.rotation.z == 0) {
		forward = glm::vec3(0.0f, 0.0f, 1.0f);
		up = glm::vec3(0.0f, 1.0f, 0.0f);
	} else {
		forward = glm::vec3(
			-cosf(A) * sinf(B) * cosf(C) + sinf(A) * sinf(C),
			 cosf(A) * sinf(B) * sinf(C) + sinf(A) * cosf(C),
			 cosf(A) * cosf(B));
		up = glm::vec3(
			 sinf(A) * sinf(B) * cosf(C) + cosf(A) * sinf(C),
			-sinf(A) * sinf(B) * sinf(C) + cosf(A) * cosf(C),
			-sinf(A) * cosf(B));
	}
}

void cursor_to_ray(const CameraInfo& cam,
                   double cursor_x, double cursor_y,
                   int window_w, int window_h,
                   glm::vec3& out_origin, glm::vec3& out_dir) {
	glm::vec3 forward, up;
	compute_basis(cam, forward, up);

	// The Camera (src/raytracer/Camera.h) maps a normalized (u, v) in [0,1]
	// to a ray direction via:
	//     lower_left_corner = origin - horizontal/2 - vertical/2 - w
	//     dir = lower_left + u*horizontal + v*vertical - origin
	// where:
	//     w = forward, v = up, u (basis) = cross(v, w)
	// Reproduce the same math here so picked rays match the raytraced image
	// pixel-perfectly.

	float theta = degrees_to_radians(cam.fov);
	float half_h = tanf(theta / 2.0f);
	float viewport_h = 2.0f * half_h;
	float aspect = (float)window_w / (float)window_h;
	float viewport_w = aspect * viewport_h;

	glm::vec3 w = forward;
	glm::vec3 v_basis = up;
	glm::vec3 u_basis = glm::cross(v_basis, w);

	glm::vec3 horizontal = viewport_w * u_basis;
	glm::vec3 vertical = viewport_h * v_basis;
	glm::vec3 lower_left = cam.origin - horizontal / 2.0f - vertical / 2.0f - w;

	// cursor_y is top-left origin; the raytracer treats v=0 as the bottom
	// row, so flip vertically.
	float u_n = (float)cursor_x / (float)window_w;
	float v_n = 1.0f - (float)cursor_y / (float)window_h;

	glm::vec3 point_on_plane = lower_left + u_n * horizontal + v_n * vertical;
	out_origin = cam.origin;
	out_dir = glm::normalize(point_on_plane - cam.origin);
}

bool intersect_plane(const glm::vec3& ray_origin,
                     const glm::vec3& ray_dir,
                     const glm::vec3& plane_point,
                     const glm::vec3& plane_normal,
                     glm::vec3& out_point) {
	float denom = glm::dot(plane_normal, ray_dir);
	if (std::fabs(denom) < 1e-6f) return false;
	float t = glm::dot(plane_point - ray_origin, plane_normal) / denom;
	if (t <= 0.0f) return false;
	out_point = ray_origin + t * ray_dir;
	return true;
}

} // namespace Picker
