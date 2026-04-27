#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <string>

class Window;

class Gui {
public:
    Gui(GLFWwindow* window);
    ~Gui();

    Gui(const Gui&) = delete;
    Gui& operator=(const Gui&) = delete;

    static void new_frame();
    void draw(Window& app);
    static void render();
    static bool wants_mouse() ;
    static bool wants_keyboard() ;
    bool visible() const { return _visible; }
    void toggle() { _visible = !_visible; }

    int panel_width() const { return _panel_width; }
    bool panel_on_right() const { return _panel_on_right; }

private:
    bool _visible = true;
    int _panel_width = 340;
    bool _panel_on_right = true;

    // State for "Add Object" panel
    int _new_obj_type = 0;
    int _new_obj_material = 0;
    glm::vec3 _new_sphere_center = glm::vec3(0.0f, 0.0f, -2.0f);
    float _new_sphere_radius = 0.3f;
    glm::vec3 _new_obj_albedo = glm::vec3(0.8f);
    float _new_obj_fuzz = 0.1f;
    float _new_obj_ior = 1.5f;
    glm::vec3 _new_obj_emission = glm::vec3(4.0f);
    float _new_obj_scatter_dist = 1.0f;
    glm::vec3 _new_obj_extinction = glm::vec3(1.0f, 0.2f, 0.1f);

    // Disc-specific state
    glm::vec3 _new_disc_center = glm::vec3(0.0f, 2.0f, -1.0f);
    glm::vec3 _new_disc_normal = glm::vec3(0.0f, -1.0f, 0.0f);
    float _new_disc_radius = 0.5f;

    // File dialog keys
    std::string _pending_obj_path;
    std::string _pending_diffuse_path;
    std::string _pending_normal_path;
    std::string _pending_specular_path;

    static void draw_mode_indicator(Window& app);
    static void draw_render_params(Window& app);
    static void draw_scene_objects(Window& app);
    void draw_add_object(Window& app);
    void draw_file_loader(Window& app);
    static void draw_camera_info(Window& app);
    void draw_file_dialogs(Window& app);
};
