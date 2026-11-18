#pragma once

#include <glm/vec3.hpp>
#include <string>

struct BackendOptions {
	std::string output_path;
	int seed = 1984;
	glm::vec3 camera_pos{0.0f, 1.0f, 4.0f};
	glm::vec3 camera_target{0.0f, 0.0f, 0.0f};
};
