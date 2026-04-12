#include "Scene.h"

#include "Mesh.h"
#include "GLTexture.h"
#include "ObjLoader.h"

#include <iostream>
#include <limits>

// ---------------------------------------------------------------------------
// Scene construction: defaults mirror kernel.cu's original create_world so
// preview + raytrace look the same on the default scene (the CUDA world is
// now rebuilt from this same source of truth).
// ---------------------------------------------------------------------------

namespace {

SceneObject make_sphere(const glm::vec3& center, float radius, const glm::vec3& color,
                        SceneMaterial mat, const glm::vec3& albedo_or_emit = glm::vec3(0.8f),
                        float fuzz = 0.0f, float ior = 1.5f, bool is_light = false,
                        const glm::vec3& emission = glm::vec3(0.0f)) {
	SceneObject o;
	o.kind = ProxyKind::Sphere;
	o.color = color;
	o.center = center;
	o.radius = radius;
	o.material = mat;
	o.albedo = albedo_or_emit;
	o.fuzz = fuzz;
	o.ior = ior;
	o.emission = emission;
	o.is_light = is_light;
	return o;
}

SceneObject make_triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                          const glm::vec3& color, const glm::vec3& albedo) {
	SceneObject o;
	o.kind = ProxyKind::Triangle;
	o.color = color;
	o.v0 = v0;
	o.v1 = v1;
	o.v2 = v2;
	o.albedo = albedo;
	o.material = SceneMaterial::Lambertian;
	return o;
}

SceneObject make_rect(const glm::vec3& Q, const glm::vec3& u, const glm::vec3& v,
                      const glm::vec3& color, const glm::vec3& emission) {
	SceneObject o;
	o.kind = ProxyKind::Rect;
	o.color = color;
	o.Q = Q;
	o.u = u;
	o.v = v;
	o.emission = emission;
	o.material = SceneMaterial::Emissive;
	o.is_light = true;
	return o;
}

SceneObject make_disc(const glm::vec3& center, const glm::vec3& normal, float radius,
                      const glm::vec3& color, const glm::vec3& emission) {
	SceneObject o;
	o.kind = ProxyKind::Disc;
	o.color = color;
	o.center = center;
	o.disc_normal = normal;
	o.radius = radius;
	o.emission = emission;
	o.material = SceneMaterial::Emissive;
	o.is_light = true;
	return o;
}

} // namespace

Scene::Scene() {
	// Ground sphere with checker texture.
	glm::vec3 ground_proxy = 0.5f * (glm::vec3(0.2f, 0.3f, 0.1f) + glm::vec3(0.9f));
	SceneObject ground = make_sphere(glm::vec3(0, -1000.5f, 0), 1000.0f,
		ground_proxy, SceneMaterial::Lambertian, ground_proxy);
	ground.use_checker    = true;
	ground.checker_color1 = glm::vec3(0.2f, 0.3f, 0.1f);
	ground.checker_color2 = glm::vec3(0.9f, 0.9f, 0.9f);
	_objects.push_back(ground);

	// Large metal sphere below the backpack (backpack loads at y≈0.5, scale=0.5)
	_objects.push_back(make_sphere(glm::vec3(0.0f, -0.2f, -1.0f), 0.3f,
		glm::vec3(0.8f, 0.8f, 0.8f),
		SceneMaterial::Metal, glm::vec3(0.8f, 0.8f, 0.8f), 0.1f));

	// Emissive disc light (replaces old sphere + rect lights)
	_objects.push_back(make_disc(
		glm::vec3(0.0f, 3.0f, -1.0f),   // center
		glm::vec3(0.0f, -1.0f, 0.0f),  // normal (pointing downward)
		1.0f,                             // radius
		glm::vec3(1.0f),                  // rasterizer color
		glm::vec3(4.0f, 4.0f, 4.0f)));  // emission
}

Scene::~Scene() = default;

int Scene::add_obj_from_file(const std::string& obj_path,
                              const std::string& texture_path,
                              const glm::vec3& position,
                              float scale) {
	auto mesh = ObjLoader::load(obj_path);
	if (!mesh) {
		std::cerr << "[Scene] Failed to load .obj file: " << obj_path << std::endl;
		return -1;
	}
	mesh->upload();
	std::string actual_tex_path = texture_path;
	if (actual_tex_path.empty() && !mesh->default_diffuse_tex.empty()) {
		actual_tex_path = mesh->default_diffuse_tex;
	}

	int mesh_idx = (int)_meshes.size();
	_meshes.push_back(std::move(mesh));

	int tex_idx = -1;
	if (!actual_tex_path.empty()) {
		auto tex = std::make_unique<GLTexture>();
		if (tex->load(actual_tex_path)) {
			tex_idx = (int)_textures.size();
			_textures.push_back(std::move(tex));
		} else {
			std::cerr << "[Scene] Failed to load texture: " << actual_tex_path << std::endl;
		}
	}

	SceneObject obj;
	obj.kind = ProxyKind::Mesh;
	obj.color = glm::vec3(1.0f);
	obj.position = position;
	obj.scale = scale;
	obj.mesh_index = mesh_idx;
	obj.texture_index = tex_idx;
	obj.material = SceneMaterial::Lambertian;
	obj.albedo = glm::vec3(0.8f);

	int idx = (int)_objects.size();
	_objects.push_back(obj);

	// Retrieve the newly added mesh's AABB
	const Mesh* loaded_mesh = _meshes.back().get();
	glm::vec3 world_min = obj.position + obj.scale * loaded_mesh->local_min;
	glm::vec3 world_max = obj.position + obj.scale * loaded_mesh->local_max;

	// Check for collisions with the metal sphere
	float delta_y = 0.0f;
	for (auto& scene_obj : _objects) {
		if (scene_obj.kind == ProxyKind::Sphere && scene_obj.material == SceneMaterial::Metal) {
			float r = scene_obj.radius * scene_obj.scale;
			glm::vec3 c = scene_obj.center + scene_obj.position;
			
			// AABB vs Sphere collision detection
			glm::vec3 closest_point = glm::clamp(c, world_min, world_max);
			glm::vec3 diff = closest_point - c;
			float dist2 = glm::dot(diff, diff);
			
			// If intersection occurs, move the sphere underneath the bounding box
			if (dist2 < r * r) {
				float new_y = world_min.y - r - scene_obj.center.y - 0.05f; // 0.05f padding
				delta_y = new_y - scene_obj.position.y;
				scene_obj.position.y = new_y;
			}
		}
	}

	// If the metal sphere was moved down, move the ground sphere down by the same amount
	if (delta_y != 0.0f) {
		for (auto& scene_obj : _objects) {
			if (scene_obj.kind == ProxyKind::Sphere && scene_obj.radius == 1000.0f) {
				scene_obj.position.y += delta_y;
			}
		}
	}

	return idx;
}

// ---------------------------------------------------------------------------
// Host-side ray-intersection helpers for picking. The geometry used for each
// primitive matches what the Rasterizer draws — so what you see is what you
// click on.
// ---------------------------------------------------------------------------

namespace {

bool hit_sphere(const glm::vec3& ray_o, const glm::vec3& ray_d,
                const glm::vec3& center, float radius, float& out_t) {
	glm::vec3 oc = ray_o - center;
	float a = glm::dot(ray_d, ray_d);
	float b = glm::dot(oc, ray_d);
	float c = glm::dot(oc, oc) - radius * radius;
	float disc = b * b - a * c;
	if (disc < 0.0f) return false;
	float sqd = std::sqrt(disc);
	float t = (-b - sqd) / a;
	if (t < 1e-3f) t = (-b + sqd) / a;
	if (t < 1e-3f) return false;
	out_t = t;
	return true;
}

bool hit_triangle(const glm::vec3& ray_o, const glm::vec3& ray_d,
                  const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                  float& out_t) {
	glm::vec3 edge1 = v1 - v0;
	glm::vec3 edge2 = v2 - v0;
	glm::vec3 h = glm::cross(ray_d, edge2);
	float a = glm::dot(edge1, h);
	if (std::fabs(a) < 1e-8f) return false;
	float f = 1.0f / a;
	glm::vec3 s = ray_o - v0;
	float u = f * glm::dot(s, h);
	if (u < 0.0f || u > 1.0f) return false;
	glm::vec3 q = glm::cross(s, edge1);
	float v = f * glm::dot(ray_d, q);
	if (v < 0.0f || u + v > 1.0f) return false;
	float t = f * glm::dot(edge2, q);
	if (t < 1e-3f) return false;
	out_t = t;
	return true;
}

bool hit_disc(const glm::vec3& ray_o, const glm::vec3& ray_d,
              const glm::vec3& center, const glm::vec3& normal,
              float radius, float& out_t) {
	float denom = glm::dot(normal, ray_d);
	if (std::fabs(denom) < 1e-8f) return false;
	float D = glm::dot(normal, center);
	float t = (D - glm::dot(normal, ray_o)) / denom;
	if (t < 1e-3f) return false;
	glm::vec3 p = ray_o + t * ray_d;
	glm::vec3 d = p - center;
	if (glm::dot(d, d) > radius * radius) return false;
	out_t = t;
	return true;
}

bool hit_aabb(const glm::vec3& ray_o, const glm::vec3& ray_d,
              const glm::vec3& mn, const glm::vec3& mx,
              float& out_t) {
	float tmin = 1e-3f;
	float tmax = 1e30f;
	for (int i = 0; i < 3; i++) {
		float inv_d = 1.0f / ray_d[i];
		float t0 = (mn[i] - ray_o[i]) * inv_d;
		float t1 = (mx[i] - ray_o[i]) * inv_d;
		if (inv_d < 0.0f) std::swap(t0, t1);
		tmin = t0 > tmin ? t0 : tmin;
		tmax = t1 < tmax ? t1 : tmax;
		if (tmax <= tmin) return false;
	}
	out_t = tmin;
	return true;
}

} // namespace

bool Scene::ray_intersect(const glm::vec3& ray_origin,
                          const glm::vec3& ray_dir,
                          float& out_t,
                          int& out_idx,
                          glm::vec3& out_point) const {
	float best_t = std::numeric_limits<float>::infinity();
	int best_idx = -1;

	for (int i = 0; i < (int)_objects.size(); i++) {
		const SceneObject& o = _objects[i];
		float t = 0.0f;
		bool hit = false;

		switch (o.kind) {
			case ProxyKind::Sphere: {
				glm::vec3 c = o.center + o.position;
				float r = o.radius * o.scale;
				hit = hit_sphere(ray_origin, ray_dir, c, r, t);
				break;
			}
			case ProxyKind::Triangle: {
				glm::vec3 v0w = o.position + o.scale * o.v0;
				glm::vec3 v1w = o.position + o.scale * o.v1;
				glm::vec3 v2w = o.position + o.scale * o.v2;
				hit = hit_triangle(ray_origin, ray_dir, v0w, v1w, v2w, t);
				break;
			}
			case ProxyKind::Rect: {
				glm::vec3 a = o.position + o.scale * o.Q;
				glm::vec3 b = o.position + o.scale * (o.Q + o.u);
				glm::vec3 c = o.position + o.scale * (o.Q + o.u + o.v);
				glm::vec3 d = o.position + o.scale * (o.Q + o.v);
				float t1, t2;
				bool h1 = hit_triangle(ray_origin, ray_dir, a, b, c, t1);
				bool h2 = hit_triangle(ray_origin, ray_dir, a, c, d, t2);
				if (h1 && h2) { t = std::min(t1, t2); hit = true; }
				else if (h1) { t = t1; hit = true; }
				else if (h2) { t = t2; hit = true; }
				break;
			}
			case ProxyKind::Disc: {
				glm::vec3 c = o.center + o.position;
				float r = o.radius * o.scale;
				glm::vec3 n = glm::normalize(o.disc_normal);
				hit = hit_disc(ray_origin, ray_dir, c, n, r, t);
				break;
			}
			case ProxyKind::Mesh: {
				if (o.mesh_index < 0 || o.mesh_index >= (int)_meshes.size()) break;
				const Mesh* m = _meshes[o.mesh_index].get();
				glm::vec3 mn = o.position + o.scale * m->local_min;
				glm::vec3 mx = o.position + o.scale * m->local_max;
				hit = hit_aabb(ray_origin, ray_dir, mn, mx, t);
				break;
			}
		}

		if (hit && t < best_t) {
			best_t = t;
			best_idx = i;
		}
	}

	if (best_idx < 0) return false;
	out_t = best_t;
	out_idx = best_idx;
	out_point = ray_origin + best_t * ray_dir;
	return true;
}
