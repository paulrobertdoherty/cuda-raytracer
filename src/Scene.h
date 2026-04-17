#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

class Mesh;
class GLTexture;

enum class ProxyKind {
	Sphere,
	Triangle,
	Rect,
	Mesh,
	Disc
};

// Material descriptors kept host-side so we can drive the CUDA world rebuild.
// Scope is intentionally narrow: the same four cases that create_world used.
enum class SceneMaterial {
	Lambertian,
	Metal,
	Dielectric,
	Emissive,
	SubsurfaceScattering
};

// Single scene object. The union-ish geometry fields are interpreted based on
// `kind`. Every object has a position offset and uniform scale on top of its
// local geometry so click-and-drag can translate anything.
struct SceneObject {
	ProxyKind kind;
	glm::vec3 color = glm::vec3(1.0f); // rasterizer uColor fallback
	glm::vec3 position = glm::vec3(0.0f); // world-space translation applied on top of local geometry
	float scale = 1.0f;

	// Material for the CUDA raytracer.
	SceneMaterial material = SceneMaterial::Lambertian;
	glm::vec3 albedo = glm::vec3(0.8f);
	float fuzz = 0.0f;
	float ior = 1.5f;
	glm::vec3 emission = glm::vec3(0.0f);
	bool is_light = false;

	// Subsurface scattering parameters (used when material == SubsurfaceScattering)
	float scattering_distance = 1.0f;
	glm::vec3 extinction_coeff = glm::vec3(1.0f, 0.2f, 0.1f);

	// Sphere: center + radius (center is the local center, position is added on top)
	glm::vec3 center = glm::vec3(0.0f);
	float radius = 1.0f;

	// Triangle: three local vertices
	glm::vec3 v0 = glm::vec3(0.0f), v1 = glm::vec3(0.0f), v2 = glm::vec3(0.0f);

	// Rect: corner Q + edge vectors u, v
	glm::vec3 Q = glm::vec3(0.0f), u = glm::vec3(0.0f), v = glm::vec3(0.0f);

	// Disc: uses center + radius fields above; disc_normal is the plane normal
	glm::vec3 disc_normal = glm::vec3(0.0f, -1.0f, 0.0f);

	// Mesh: index into Scene's mesh / texture lists. -1 if not applicable.
	int mesh_index = -1;
	int texture_index = -1;
	int normal_texture_index = -1;
	int specular_texture_index = -1;

	// Checker texture for Lambertian surfaces (e.g. ground)
	bool use_checker = false;
	glm::vec3 checker_color1 = glm::vec3(0.2f, 0.3f, 0.1f);
	glm::vec3 checker_color2 = glm::vec3(0.9f, 0.9f, 0.9f);
};

class Scene {
public:
	Scene();
	~Scene();

	// Scene owns meshes and textures — forward-declared to keep the header
	// lightweight. Non-copyable.
	Scene(const Scene&) = delete;
	Scene& operator=(const Scene&) = delete;

	// Ray / scene intersection used for mouse picking. Tests every object in
	// world space (honoring SceneObject::position and scale). Returns true if
	// any primitive was hit; fills `out_t`, `out_idx`, and `out_point`.
	bool ray_intersect(const glm::vec3& ray_origin,
	                   const glm::vec3& ray_dir,
	                   float& out_t,
	                   int& out_idx,
	                   glm::vec3& out_point) const;

	// Add a sphere to the scene. Returns the index of the new object.
	int add_sphere(const glm::vec3& center, float radius,
	               SceneMaterial material, const glm::vec3& albedo,
	               float fuzz, float ior,
	               const glm::vec3& emission, bool is_light);

	// Remove an object by index. Invalidates indices of objects after it.
	void remove_object(int index);

	// Load an .obj file (and optional textures), append a mesh SceneObject at
	// the given position and uniform scale. Texture paths may be empty to
	// auto-discover from the .mtl file. Returns index of new object, or -1.
	int add_obj_from_file(const std::string& obj_path,
	                       const std::string& diffuse_path,
	                       const glm::vec3& position,
	                       float scale,
	                       const std::string& normal_path = "",
	                       const std::string& specular_path = "");

	// Access to the owned resources.
	const std::vector<SceneObject>& objects() const { return _objects; }
	std::vector<SceneObject>& mutable_objects() { return _objects; }
	const std::vector<std::unique_ptr<Mesh>>& meshes() const { return _meshes; }
	const std::vector<std::unique_ptr<GLTexture>>& textures() const { return _textures; }

private:
	std::vector<SceneObject> _objects;
	std::vector<std::unique_ptr<Mesh>> _meshes;
	std::vector<std::unique_ptr<GLTexture>> _textures;
};
