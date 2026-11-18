#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <cstdint>
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

	std::uint64_t revision() const { return _revision; }

private:
	std::uint64_t _revision = 0;
	void upload_display();
	unsigned int _vao = 0;
	unsigned int _vbo = 0;
	unsigned int _ebo = 0;
	int _index_count = 0;
};
