#include "Rasterizer.h"

#include "Mesh.h"
#include "GLTexture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <vector>

Rasterizer::Rasterizer(ShaderManager& shaders, const Scene& scene)
	: _shaders(shaders), _scene(scene) {

	_flat_shader = _shaders.get_or_load("flat",
		"./shaders/rasterize.vert", "./shaders/rasterize.frag");
	_mesh_shader = _shaders.get_or_load("mesh_textured",
		"./shaders/mesh.vert", "./shaders/mesh.frag");

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

	// --- Flat-shaded pass: spheres, triangles, rects ---
	_flat_shader->use();
	GLint flat_model = glGetUniformLocation(_flat_shader->ID, "uModel");
	GLint flat_view  = glGetUniformLocation(_flat_shader->ID, "uView");
	GLint flat_proj  = glGetUniformLocation(_flat_shader->ID, "uProj");
	GLint flat_color = glGetUniformLocation(_flat_shader->ID, "uColor");
	glUniformMatrix4fv(flat_view, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(flat_proj, 1, GL_FALSE, glm::value_ptr(proj));

	// Draw spheres.
	glBindVertexArray(_sphere_vao);
	const auto& objs = _scene.objects();
	for (int i = 0; i < (int)objs.size(); i++) {
		const SceneObject& p = objs[i];
		if (p.kind != ProxyKind::Sphere) continue;
		glm::mat4 model(1.0f);
		model = glm::translate(model, p.center + p.position);
		model = glm::scale(model, glm::vec3(p.radius * p.scale));
		glUniformMatrix4fv(flat_model, 1, GL_FALSE, glm::value_ptr(model));
		glm::vec3 color = p.color * (i == _selected_idx ? 1.5f : 1.0f);
		glUniform3fv(flat_color, 1, glm::value_ptr(color));
		glDrawElements(GL_TRIANGLES, _sphere_index_count, GL_UNSIGNED_INT, 0);
	}

	// Draw triangles + rects. Each primitive uses its own uColor uniform so
	// we issue one draw call per primitive (cheap — there are only a handful
	// of these in the default scene).
	glBindVertexArray(_tri_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _tri_vbo);
	for (int i = 0; i < (int)objs.size(); i++) {
		const SceneObject& p = objs[i];
		glm::mat4 model(1.0f);
		model = glm::translate(model, p.position);
		model = glm::scale(model, glm::vec3(p.scale));
		glUniformMatrix4fv(flat_model, 1, GL_FALSE, glm::value_ptr(model));
		glm::vec3 color = p.color * (i == _selected_idx ? 1.5f : 1.0f);
		glUniform3fv(flat_color, 1, glm::value_ptr(color));

		if (p.kind == ProxyKind::Triangle) {
			float verts[9] = {
				p.v0.x, p.v0.y, p.v0.z,
				p.v1.x, p.v1.y, p.v1.z,
				p.v2.x, p.v2.y, p.v2.z,
			};
			glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
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
			glDrawArrays(GL_TRIANGLES, 0, 6);
		} else if (p.kind == ProxyKind::Disc) {
			// Triangle fan approximation of the disc (32 segments).
			const glm::vec3 n = glm::normalize(p.disc_normal);
			glm::vec3 a = (fabsf(n.x) > 0.9f) ? glm::vec3(0.0f, 1.0f, 0.0f)
			                                    : glm::vec3(1.0f, 0.0f, 0.0f);
			glm::vec3 bt = glm::normalize(glm::cross(n, a));
			glm::vec3 t  = glm::cross(bt, n);
			const int SEGS = 32;
			const float pi = 3.14159265358979323846f;
			std::vector<float> fan;
			fan.reserve((SEGS + 2) * 3);
			// center
			fan.push_back(p.center.x); fan.push_back(p.center.y); fan.push_back(p.center.z);
			for (int s = 0; s <= SEGS; s++) {
				float angle = 2.0f * pi * (float)s / (float)SEGS;
				glm::vec3 pt = p.center + p.radius * (cosf(angle) * t + sinf(angle) * bt);
				fan.push_back(pt.x); fan.push_back(pt.y); fan.push_back(pt.z);
			}
			glBufferData(GL_ARRAY_BUFFER, fan.size() * sizeof(float), fan.data(), GL_DYNAMIC_DRAW);
			glDrawArrays(GL_TRIANGLE_FAN, 0, SEGS + 2);
		}
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// --- Textured mesh pass ---
	const auto& meshes = _scene.meshes();
	const auto& textures = _scene.textures();
	bool have_mesh = false;
	for (const auto& p : objs) {
		if (p.kind == ProxyKind::Mesh) { have_mesh = true; break; }
	}
	if (have_mesh && _mesh_shader) {
		_mesh_shader->use();
		GLint mesh_model = glGetUniformLocation(_mesh_shader->ID, "uModel");
		GLint mesh_view  = glGetUniformLocation(_mesh_shader->ID, "uView");
		GLint mesh_proj  = glGetUniformLocation(_mesh_shader->ID, "uProj");
		GLint mesh_color = glGetUniformLocation(_mesh_shader->ID, "uColor");
		GLint mesh_hastex = glGetUniformLocation(_mesh_shader->ID, "uHasTexture");
		GLint mesh_diffuse = glGetUniformLocation(_mesh_shader->ID, "uDiffuse");
		GLint mesh_light_dir = glGetUniformLocation(_mesh_shader->ID, "uLightDir");
		glUniformMatrix4fv(mesh_view, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(mesh_proj, 1, GL_FALSE, glm::value_ptr(proj));
		glUniform1i(mesh_diffuse, 0);
		// Light follows the camera (headlight).
		glm::vec3 light_dir = glm::normalize(forward);
		glUniform3fv(mesh_light_dir, 1, glm::value_ptr(light_dir));

		for (int i = 0; i < (int)objs.size(); i++) {
			const SceneObject& p = objs[i];
			if (p.kind != ProxyKind::Mesh) continue;
			if (p.mesh_index < 0 || p.mesh_index >= (int)meshes.size()) continue;

			glm::mat4 model(1.0f);
			model = glm::translate(model, p.position);
			model = glm::scale(model, glm::vec3(p.scale));
			glUniformMatrix4fv(mesh_model, 1, GL_FALSE, glm::value_ptr(model));

			glm::vec3 color = p.color * (i == _selected_idx ? 1.5f : 1.0f);
			glUniform3fv(mesh_color, 1, glm::value_ptr(color));

			if (p.texture_index >= 0 && p.texture_index < (int)textures.size()) {
				glUniform1i(mesh_hastex, 1);
				textures[p.texture_index]->bind(0);
			} else {
				glUniform1i(mesh_hastex, 0);
			}

			meshes[p.mesh_index]->draw();
		}
	}

	glDisable(GL_DEPTH_TEST);
}
