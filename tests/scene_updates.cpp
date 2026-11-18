#include "Scene.h"
#include "raytracer/kernel.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int width = 48;
constexpr int height = 32;
constexpr int samples = 8;
constexpr int depth = 6;
constexpr float fov = 50.0f;
constexpr int seed = 1357;

// Each renderer draws exactly one frame so their Philox states match.
template<typename Edit>
void compare_with_rebuild(const char* name, Edit edit) {
    Scene scene;
    KernelInfo edited(width, height, samples, depth, fov, seed);
    edit(scene, edited);
    KernelInfo fresh(width, height, samples, depth, fov, seed);
    fresh.rebuild_world(scene);

    std::vector<std::uint8_t> actual;
    std::vector<std::uint8_t> expected;
    for (auto* renderer : {&edited, &fresh}) {
        renderer->set_camera(glm::vec3(0, 0.45f, 2), glm::vec3(0, 0, 1), glm::vec3(0, 1, 0));
    }
    edited.render_to_buffer(actual);
    fresh.render_to_buffer(expected);
    if (actual != expected) {
        throw std::runtime_error(std::string(name) + " differs from a fresh world");
    }
}

int add_sphere(Scene& scene, float x, float y, float radius) {
    return scene.add_sphere(glm::vec3(x, y, -1), radius, SceneMaterial::Lambertian,
                            glm::vec3(0.8f, 0.1f, 0.05f), 0, 1.5f, glm::vec3(0), false);
}

void move_leaf(Scene& scene, KernelInfo& edited) {
    // This leaf and its former neighbors start outside the camera frustum.
    for (int i = 0; i < 32; ++i) {
        add_sphere(scene, 4.0f + i, 0.5f, 0.3f);
    }
    // A mesh without an asset has a scene index but no device descriptor.
    SceneObject missing_mesh{};
    missing_mesh.kind = ProxyKind::Mesh;
    scene.mutable_objects().push_back(missing_mesh);
    const int leaf = add_sphere(scene, 40.0f, 0.5f, 0.5f);
    edited.rebuild_world(scene);
    scene.mutable_objects()[leaf].position.x = -40.0f;
    edited.update_object(scene, leaf);
}

void replace_material(Scene& scene, KernelInfo& edited) {
    scene.mutable_objects().clear();
    const int leaf = add_sphere(scene, 0, 0.45f, 0.75f);
    auto& object = scene.mutable_objects()[leaf];
    object.use_checker = true;
    edited.rebuild_world(scene);
    object.material = SceneMaterial::Metal;
    object.albedo = glm::vec3(0.05f, 0.6f, 0.8f);
    edited.update_object(scene, leaf);
    // A single emissive leaf remains visible without any other lights.
    object.material = SceneMaterial::Emissive;
    object.emission = glm::vec3(0.2f, 1, 0.5f);
    edited.update_object(scene, leaf);
}

void toggle_light(Scene& scene, KernelInfo& edited, bool enable) {
    const int leaf = scene.add_sphere(glm::vec3(-0.7f, 1.1f, -0.5f), 0.3f,
        SceneMaterial::Emissive, glm::vec3(1), 0, 1.5f, glm::vec3(8, 1, 1), !enable);
    edited.rebuild_world(scene);
    auto& light = scene.mutable_objects()[leaf];
    if (!enable) {
        // Replace the primitive while it is still registered as a light.
        light.kind = ProxyKind::Disc;
        light.disc_normal = glm::normalize(glm::vec3(0, -1, 0.5f));
        edited.update_object(scene, leaf);
    }
    light.is_light = enable;
    edited.update_object(scene, leaf);
}

void empty_scene(Scene& scene, KernelInfo& edited) {
    edited.rebuild_world(scene);
    scene.mutable_objects().clear();
    edited.rebuild_world(scene);
}

void regrow_scene(Scene& scene, KernelInfo& edited) {
    empty_scene(scene, edited);
    // Grow beyond the first world's capacity, then edit a middle leaf.
    for (int i = 0; i < 25; ++i) {
        add_sphere(scene, (i % 5 - 2) * 0.35f, (i / 5 - 1) * 0.35f, 0.15f);
    }
    auto& center = scene.mutable_objects()[12];
    center.material = SceneMaterial::Emissive;
    center.emission = glm::vec3(1, 1, 0.5f);
    center.is_light = true;
    edited.rebuild_world(scene);
    center.radius *= 2.0f;
    edited.update_object(scene, 12);
}

} // namespace

int main() {
    try {
        compare_with_rebuild("moved BVH leaf", move_leaf);
        compare_with_rebuild("single leaf with a replacement material", replace_material);
        for (bool enable : {false, true}) {
            compare_with_rebuild(enable ? "enabled light" : "disabled light",
                [enable](Scene& scene, KernelInfo& renderer) { toggle_light(scene, renderer, enable); });
        }
        compare_with_rebuild("empty world", empty_scene);
        compare_with_rebuild("regrown world", regrow_scene);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
