#pragma once

#include <glm/glm.hpp>
#include <vector>

enum class ProxyKind {
	Sphere,
	Triangle,
	Rect
};

struct ProxyPrimitive {
	ProxyKind kind;
	glm::vec3 color;

	// Sphere: center + radius
	glm::vec3 center;
	float radius;

	// Triangle: v0, v1, v2
	glm::vec3 v0, v1, v2;

	// Rect: corner Q + edge vectors u, v
	glm::vec3 Q, u, v;
};

std::vector<ProxyPrimitive> build_host_scene();
