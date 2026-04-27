#pragma once

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "Shader.h"
#include "ShaderManager.h"
#include "Scene.h"
#include "raytracer/Camera.h"

class Rasterizer {
public:
	// The Rasterizer does not own its shaders or scene — it reads from the
	// ShaderManager and Scene owned by Window. The references must outlive
	// the Rasterizer.
	Rasterizer(ShaderManager& shaders, const Scene& scene);
	~Rasterizer();

	// Renders proxies into the currently bound framebuffer.
	// Caller must set the viewport before calling.
	void render(const CameraInfo& cam, float aspect);

	// Index of the currently highlighted (picked) object, or -1 for none.
	// The highlighted object is drawn with a brightened base color.
	void set_selected(int idx) { _selected_idx = idx; }
	int selected() const { return _selected_idx; }

private:
	ShaderManager& _shaders;
	const Scene& _scene;

	Shader* _flat_shader = nullptr;
	Shader* _mesh_shader = nullptr;

	int _selected_idx = -1;

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
