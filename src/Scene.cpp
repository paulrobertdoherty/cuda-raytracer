#include "Scene.h"

// Host-side mirror of create_world (src/raytracer/kernel.cu).
// Keep in sync manually.

static ProxyPrimitive make_sphere(glm::vec3 center, float radius, glm::vec3 color) {
	ProxyPrimitive p{};
	p.kind = ProxyKind::Sphere;
	p.color = color;
	p.center = center;
	p.radius = radius;
	return p;
}

static ProxyPrimitive make_triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 color) {
	ProxyPrimitive p{};
	p.kind = ProxyKind::Triangle;
	p.color = color;
	p.v0 = v0;
	p.v1 = v1;
	p.v2 = v2;
	return p;
}

static ProxyPrimitive make_rect(glm::vec3 Q, glm::vec3 u, glm::vec3 v, glm::vec3 color) {
	ProxyPrimitive p{};
	p.kind = ProxyKind::Rect;
	p.color = color;
	p.Q = Q;
	p.u = u;
	p.v = v;
	return p;
}

std::vector<ProxyPrimitive> build_host_scene() {
	std::vector<ProxyPrimitive> scene;

	scene.push_back(make_sphere(glm::vec3(0, 0, -1), 0.5f, glm::vec3(0.8f, 0.3f, 0.3f)));
	scene.push_back(make_sphere(glm::vec3(-1.01f, 0, -1), 0.5f, glm::vec3(0.85f, 0.9f, 0.95f)));
	scene.push_back(make_sphere(glm::vec3(-1, 10, -1), 0.5f, glm::vec3(0.85f, 0.9f, 0.95f)));
	scene.push_back(make_sphere(glm::vec3(1, 0, -1), 0.5f, glm::vec3(0.8f, 0.8f, 0.8f)));

	// Ground sphere — checker texture, use the average of the two checker colors.
	glm::vec3 ground_color = 0.5f * (glm::vec3(0.2f, 0.3f, 0.1f) + glm::vec3(0.9f));
	scene.push_back(make_sphere(glm::vec3(0, -1000.5f, 0), 1000.0f, ground_color));

	// Emissive sphere light — clamp emit color to [0,1] for the proxy.
	scene.push_back(make_sphere(glm::vec3(0, 3, -1), 1.0f, glm::vec3(1.0f)));

	// Emissive area light rectangle.
	scene.push_back(make_rect(
		glm::vec3(-1, 3, -2), glm::vec3(2, 0, 0), glm::vec3(0, 0, 2),
		glm::vec3(1.0f)));

	// Blue triangle.
	scene.push_back(make_triangle(
		glm::vec3(-2, 0, -2), glm::vec3(-1, 2, -2), glm::vec3(0, 0, -2),
		glm::vec3(0.1f, 0.2f, 0.8f)));

	return scene;
}
