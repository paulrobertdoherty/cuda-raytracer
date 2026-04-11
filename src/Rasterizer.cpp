#include "Rasterizer.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

Rasterizer::Rasterizer() {
	_shader = std::make_unique<Shader>("./shaders/rasterize.vert", "./shaders/rasterize.frag");
	_scene = build_host_scene();

	build_sphere_mesh(20, 14);

	glGenVertexArrays(1, &_tri_vao);
	glGenBuffers(1, &_tri_vbo);
	glBindVertexArray(_tri_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _tri_vbo);
	glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glBindVertexArray(0);
}

Rasterizer::~Rasterizer() {
	if (_sphere_ibo) glDeleteBuffers(1, &_sphere_ibo);
	if (_sphere_vbo) glDeleteBuffers(1, &_sphere_vbo);
	if (_sphere_vao) glDeleteVertexArrays(1, &_sphere_vao);
	if (_tri_vbo) glDeleteBuffers(1, &_tri_vbo);
	if (_tri_vao) glDeleteVertexArrays(1, &_tri_vao);
}

void Rasterizer::build_sphere_mesh(int slices, int stacks) {
	std::vector<float> verts;
	std::vector<unsigned int> indices;

	const float pi = 3.14159265358979323846f;
	for (int i = 0; i <= stacks; i++) {
		float phi = pi * float(i) / float(stacks);
		float sp = sinf(phi);
		float cp = cosf(phi);
		for (int j = 0; j <= slices; j++) {
			float theta = 2.0f * pi * float(j) / float(slices);
			float st = sinf(theta);
			float ct = cosf(theta);
			verts.push_back(sp * ct);
			verts.push_back(cp);
			verts.push_back(sp * st);
		}
	}

	for (int i = 0; i < stacks; i++) {
		for (int j = 0; j < slices; j++) {
			unsigned int a = i * (slices + 1) + j;
			unsigned int b = a + slices + 1;
			indices.push_back(a);
			indices.push_back(b);
			indices.push_back(a + 1);
			indices.push_back(b);
			indices.push_back(b + 1);
			indices.push_back(a + 1);
		}
	}

	_sphere_index_count = (GLsizei)indices.size();

	glGenVertexArrays(1, &_sphere_vao);
	glGenBuffers(1, &_sphere_vbo);
	glGenBuffers(1, &_sphere_ibo);

	glBindVertexArray(_sphere_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _sphere_vbo);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _sphere_ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glBindVertexArray(0);
}

static void compute_camera_basis(const CameraInfo& cam, glm::vec3& forward, glm::vec3& up) {
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

void Rasterizer::render(const CameraInfo& cam, float aspect) {
	glm::vec3 forward, up;
	compute_camera_basis(cam, forward, up);

	// Ray tracer's camera looks in -forward (see Camera.h get_ray + lower_left_corner).
	glm::mat4 view = glm::lookAt(cam.origin, cam.origin - forward, up);
	glm::mat4 proj = glm::perspective(glm::radians(cam.fov), aspect, 0.05f, 2000.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	_shader->use();
	GLint loc_model = glGetUniformLocation(_shader->ID, "uModel");
	GLint loc_view = glGetUniformLocation(_shader->ID, "uView");
	GLint loc_proj = glGetUniformLocation(_shader->ID, "uProj");
	GLint loc_color = glGetUniformLocation(_shader->ID, "uColor");

	glUniformMatrix4fv(loc_view, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(loc_proj, 1, GL_FALSE, glm::value_ptr(proj));

	// Draw spheres.
	glBindVertexArray(_sphere_vao);
	for (const auto& p : _scene) {
		if (p.kind != ProxyKind::Sphere) continue;
		glm::mat4 model(1.0f);
		model = glm::translate(model, p.center);
		model = glm::scale(model, glm::vec3(p.radius));
		glUniformMatrix4fv(loc_model, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(loc_color, 1, glm::value_ptr(p.color));
		glDrawElements(GL_TRIANGLES, _sphere_index_count, GL_UNSIGNED_INT, 0);
	}

	// Draw triangles + rects with identity model. Each primitive uses its own
	// uColor uniform, so we issue one draw call per primitive (cheap — there
	// are only a handful of these in the scene).
	glm::mat4 identity(1.0f);
	glUniformMatrix4fv(loc_model, 1, GL_FALSE, glm::value_ptr(identity));
	glBindVertexArray(_tri_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _tri_vbo);
	for (const auto& p : _scene) {
		if (p.kind == ProxyKind::Triangle) {
			float verts[9] = {
				p.v0.x, p.v0.y, p.v0.z,
				p.v1.x, p.v1.y, p.v1.z,
				p.v2.x, p.v2.y, p.v2.z,
			};
			glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
			glUniform3fv(loc_color, 1, glm::value_ptr(p.color));
			glDrawArrays(GL_TRIANGLES, 0, 3);
		} else if (p.kind == ProxyKind::Rect) {
			glm::vec3 a = p.Q;
			glm::vec3 b = p.Q + p.u;
			glm::vec3 c = p.Q + p.u + p.v;
			glm::vec3 d = p.Q + p.v;
			float verts[18] = {
				a.x, a.y, a.z,
				b.x, b.y, b.z,
				c.x, c.y, c.z,
				a.x, a.y, a.z,
				c.x, c.y, c.z,
				d.x, d.y, d.z,
			};
			glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
			glUniform3fv(loc_color, 1, glm::value_ptr(p.color));
			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisable(GL_DEPTH_TEST);
}
