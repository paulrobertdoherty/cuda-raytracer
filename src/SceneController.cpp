#include "SceneController.h"

SceneController::SceneController()
	: _scene(std::make_unique<Scene>()) {}

int SceneController::load_obj_from_file(const std::string& obj_path,
                                        const std::string& texture_path,
                                        const glm::vec3& position,
                                        float scale) {
	if (obj_path.empty()) return -1;
	return _scene->add_obj_from_file(obj_path, texture_path, position, scale);
}

int SceneController::add_sphere(const glm::vec3& center, float radius,
                                SceneMaterial material, const glm::vec3& albedo,
                                float fuzz, float ior,
                                const glm::vec3& emission, bool is_light) {
	return _scene->add_sphere(center, radius, material, albedo, fuzz, ior, emission, is_light);
}

void SceneController::remove_object(int index) {
	_scene->remove_object(index);
}

void SceneController::rebuild_world(KernelInfo& renderer) const {
	renderer.rebuild_world(*_scene);
}