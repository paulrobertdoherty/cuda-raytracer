#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

#include "Scene.h"
#include "raytracer/kernel.h"

class SceneController {
public:
	SceneController();

	SceneController(const SceneController&) = delete;
	SceneController& operator=(const SceneController&) = delete;

	Scene& scene() { return *_scene; }
	const Scene& scene() const { return *_scene; }
	const std::vector<SceneObject>& objects() const { return _scene->objects(); }
	std::vector<SceneObject>& mutable_objects() { return _scene->mutable_objects(); }

	int load_obj_from_file(const std::string& obj_path,
	                       const std::string& texture_path,
	                       const glm::vec3& position = glm::vec3(0.0f, 0.5f, -1.0f),
	                       float scale = 0.5f);

	int add_sphere(const glm::vec3& center, float radius,
	               SceneMaterial material, const glm::vec3& albedo,
	               float fuzz, float ior,
	               const glm::vec3& emission, bool is_light);

	void remove_object(int index);

	void rebuild_world(KernelInfo& renderer) const;

private:
	std::unique_ptr<Scene> _scene;
};