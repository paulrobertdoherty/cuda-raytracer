#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

// Interleaved CPU vertex used for both GL upload and CUDA triangle-mesh upload.
struct MeshVertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

// OpenGL mesh with interleaved VAO/VBO/EBO and CPU-side copies of the
// vertex and index data so the same mesh can seed the CUDA world.
class Mesh {
public:
	// Empty mesh; caller fills `vertices`/`indices` and then calls upload().
	Mesh() = default;
	~Mesh();

	Mesh(const Mesh&) = delete;
	Mesh& operator=(const Mesh&) = delete;

	// Push data into the GL buffers. Call after the CPU-side vectors are
	// populated.
	void upload();

	// Bind and draw as triangles. Assumes the caller has already activated a
	// shader program.
	void draw() const;

	// Axis-aligned bounding box of the mesh in local space.
	glm::vec3 local_min = glm::vec3(0.0f);
	glm::vec3 local_max = glm::vec3(0.0f);

	std::string default_diffuse_tex;
	std::string default_normal_tex;
	std::string default_specular_tex;

	// CPU-side copies retained for picking and CUDA upload.
	std::vector<MeshVertex> vertices;
	std::vector<unsigned int> indices;

private:
	GLuint _vao = 0;
	GLuint _vbo = 0;
	GLuint _ebo = 0;
	GLsizei _index_count = 0;
};
