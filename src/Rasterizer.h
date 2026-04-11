#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "Shader.h"
#include "Scene.h"
#include "raytracer/Camera.h"

class Rasterizer {
public:
	Rasterizer();
	~Rasterizer();

	// Renders proxies into the currently bound framebuffer.
	// Caller must set the viewport before calling.
	void render(const CameraInfo& cam, float aspect);

private:
	std::unique_ptr<Shader> _shader;
	std::vector<ProxyPrimitive> _scene;

	// Unit sphere mesh (radius 1, centered at origin), drawn with model matrix.
	GLuint _sphere_vao = 0;
	GLuint _sphere_vbo = 0;
	GLuint _sphere_ibo = 0;
	GLsizei _sphere_index_count = 0;

	// Dynamic triangle list for triangle + rect proxies, rebuilt per draw call.
	GLuint _tri_vao = 0;
	GLuint _tri_vbo = 0;

	void build_sphere_mesh(int slices, int stacks);
};
