#include "Mesh.h"

#include <algorithm>

Mesh::~Mesh() {
#ifndef HEADLESS_BUILD
	if (_ebo) glDeleteBuffers(1, &_ebo);
	if (_vbo) glDeleteBuffers(1, &_vbo);
	if (_vao) glDeleteVertexArrays(1, &_vao);
#endif
}

void Mesh::upload() {
	if (vertices.empty() || indices.empty()) {
#ifndef HEADLESS_BUILD
		_index_count = 0;
#endif
		return;
	}

	// Compute bounding box.
	local_min = vertices[0].position;
	local_max = vertices[0].position;
	for (const auto& v : vertices) {
		local_min = glm::min(local_min, v.position);
		local_max = glm::max(local_max, v.position);
	}

#ifndef HEADLESS_BUILD
	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);
	glGenBuffers(1, &_ebo);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(vertices.size() * sizeof(MeshVertex)),
		vertices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)),
		indices.data(), GL_STATIC_DRAW);

	// layout(location=0) vec3 aPos
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
		(void*)offsetof(MeshVertex, position));
	// layout(location=1) vec3 aNormal
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
		(void*)offsetof(MeshVertex, normal));
	// layout(location=2) vec2 aUV
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
		(void*)offsetof(MeshVertex, uv));

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	_index_count = (int)indices.size();
#endif
}

void Mesh::draw() const {
#ifndef HEADLESS_BUILD
	if (_index_count == 0) return;
	glBindVertexArray(_vao);
	glDrawElements(GL_TRIANGLES, _index_count, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
#endif
}
