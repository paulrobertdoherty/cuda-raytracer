#include "Gui.h"
#include "Window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

#include <cstring>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Gui::Gui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.Alpha = 0.95f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

Gui::~Gui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Gui::new_frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Gui::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool Gui::wants_mouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool Gui::wants_keyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

// ---------------------------------------------------------------------------
// Material helpers
// ---------------------------------------------------------------------------

static const char* material_names[] = { "Lambertian", "Metal", "Dielectric", "Emissive" };
static const int material_count = 4;

static SceneMaterial material_from_index(int idx) {
    switch (idx) {
        case 0: return SceneMaterial::Lambertian;
        case 1: return SceneMaterial::Metal;
        case 2: return SceneMaterial::Dielectric;
        case 3: return SceneMaterial::Emissive;
        default: return SceneMaterial::Lambertian;
    }
}

static int index_from_material(SceneMaterial mat) {
    switch (mat) {
        case SceneMaterial::Lambertian: return 0;
        case SceneMaterial::Metal:      return 1;
        case SceneMaterial::Dielectric: return 2;
        case SceneMaterial::Emissive:   return 3;
        default: return 0;
    }
}

static const char* kind_label(ProxyKind kind) {
    switch (kind) {
        case ProxyKind::Sphere:   return "Sphere";
        case ProxyKind::Triangle: return "Triangle";
        case ProxyKind::Rect:     return "Rect";
        case ProxyKind::Mesh:     return "Mesh";
        case ProxyKind::Disc:     return "Disc";
        default: return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// Main draw entry point
// ---------------------------------------------------------------------------

void Gui::draw(Window& app) {
    if (!_visible) return;

    draw_render_params(app);
    draw_camera_info(app);
    draw_scene_objects(app);
    draw_add_object(app);
    draw_file_loader(app);
}

// ---------------------------------------------------------------------------
// Render Parameters Panel
// ---------------------------------------------------------------------------

void Gui::draw_render_params(Window& app) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 220), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Render Parameters")) {
        ImGui::End();
        return;
    }

    bool changed = false;

    changed |= ImGui::SliderInt("Samples", &app.samples, 1, 128);
    changed |= ImGui::SliderInt("Max Depth", &app.max_depth, 1, 100);

    float prev_fov = app.fov;
    if (ImGui::SliderFloat("FOV", &app.fov, 10.0f, 170.0f)) {
        app.renderer().camera_info.fov = app.fov;
        app.renderer().set_camera(
            app.renderer().camera_info.origin,
            app.renderer().camera_info.forward(),
            app.renderer().camera_info.up());
        app.reset_accumulation();
    }

    ImGui::SliderInt("Tile Size", &app.tile_size, 16, 256);

    int ps = app.preview_scale;
    if (ImGui::SliderInt("Preview Scale", &ps, 1, 8)) {
        app.preview_scale = ps < 1 ? 1 : ps;
    }

    ImGui::Separator();
    if (ImGui::Button("Start Final Render")) {
        app.start_final_render();
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Camera Info Panel
// ---------------------------------------------------------------------------

void Gui::draw_camera_info(Window& app) {
    ImGui::SetNextWindowPos(ImVec2(10, 240), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 120), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Camera")) {
        ImGui::End();
        return;
    }

    CameraInfo& cam = app.renderer().camera_info;
    float pos[3] = { cam.origin.x, cam.origin.y, cam.origin.z };
    float rot[3] = { cam.rotation.x, cam.rotation.y, cam.rotation.z };

    if (ImGui::DragFloat3("Position", pos, 0.05f)) {
        cam.origin = glm::vec3(pos[0], pos[1], pos[2]);
        app.renderer().set_camera(cam.origin, cam.forward(), cam.up());
        app.reset_accumulation();
    }

    if (ImGui::DragFloat3("Rotation", rot, 0.5f)) {
        cam.rotation = glm::vec3(rot[0], rot[1], rot[2]);
        app.renderer().set_camera(cam.origin, cam.forward(), cam.up());
        app.reset_accumulation();
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Scene Objects Panel
// ---------------------------------------------------------------------------

void Gui::draw_scene_objects(Window& app) {
    ImGui::SetNextWindowPos(ImVec2(10, 370), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 350), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Scene Objects")) {
        ImGui::End();
        return;
    }

    auto& objects = app.scene().mutable_objects();
    int delete_idx = -1;
    bool any_changed = false;

    for (int i = 0; i < (int)objects.size(); i++) {
        SceneObject& o = objects[i];
        ImGui::PushID(i);

        char label[64];
        snprintf(label, sizeof(label), "[%d] %s##obj", i, kind_label(o.kind));

        if (ImGui::CollapsingHeader(label)) {
            bool changed = false;

            // Material
            int mat_idx = index_from_material(o.material);
            if (ImGui::Combo("Material", &mat_idx, material_names, material_count)) {
                o.material = material_from_index(mat_idx);
                // Set sensible defaults when switching materials
                if (o.material == SceneMaterial::Emissive) {
                    o.is_light = true;
                    if (o.emission == glm::vec3(0.0f)) o.emission = glm::vec3(4.0f);
                } else {
                    o.is_light = false;
                }
                changed = true;
            }

            // Material-specific properties
            if (o.material == SceneMaterial::Lambertian || o.material == SceneMaterial::Metal) {
                float col[3] = { o.albedo.r, o.albedo.g, o.albedo.b };
                if (ImGui::ColorEdit3("Albedo", col)) {
                    o.albedo = glm::vec3(col[0], col[1], col[2]);
                    o.color = o.albedo;
                    changed = true;
                }
            }
            if (o.material == SceneMaterial::Metal) {
                changed |= ImGui::SliderFloat("Fuzz", &o.fuzz, 0.0f, 1.0f);
            }
            if (o.material == SceneMaterial::Dielectric) {
                changed |= ImGui::SliderFloat("IOR", &o.ior, 1.0f, 3.0f);
            }
            if (o.material == SceneMaterial::Emissive) {
                float em[3] = { o.emission.r, o.emission.g, o.emission.b };
                if (ImGui::DragFloat3("Emission", em, 0.1f, 0.0f, 50.0f)) {
                    o.emission = glm::vec3(em[0], em[1], em[2]);
                    changed = true;
                }
            }

            changed |= ImGui::Checkbox("Is Light", &o.is_light);

            // Position & scale
            float pos[3] = { o.position.x, o.position.y, o.position.z };
            if (ImGui::DragFloat3("Position", pos, 0.05f)) {
                o.position = glm::vec3(pos[0], pos[1], pos[2]);
                changed = true;
            }
            changed |= ImGui::DragFloat("Scale", &o.scale, 0.01f, 0.01f, 100.0f);

            // Geometry-specific
            if (o.kind == ProxyKind::Sphere) {
                float c[3] = { o.center.x, o.center.y, o.center.z };
                if (ImGui::DragFloat3("Center", c, 0.05f)) {
                    o.center = glm::vec3(c[0], c[1], c[2]);
                    changed = true;
                }
                changed |= ImGui::DragFloat("Radius", &o.radius, 0.01f, 0.01f, 1000.0f);
            }
            if (o.kind == ProxyKind::Disc) {
                float c[3] = { o.center.x, o.center.y, o.center.z };
                if (ImGui::DragFloat3("Center", c, 0.05f)) {
                    o.center = glm::vec3(c[0], c[1], c[2]);
                    changed = true;
                }
                float n[3] = { o.disc_normal.x, o.disc_normal.y, o.disc_normal.z };
                if (ImGui::DragFloat3("Normal", n, 0.01f, -1.0f, 1.0f)) {
                    o.disc_normal = glm::vec3(n[0], n[1], n[2]);
                    changed = true;
                }
                changed |= ImGui::DragFloat("Radius", &o.radius, 0.01f, 0.01f, 100.0f);
            }

            // Checker texture
            if (o.material == SceneMaterial::Lambertian) {
                if (ImGui::Checkbox("Checker Texture", &o.use_checker)) changed = true;
                if (o.use_checker) {
                    float c1[3] = { o.checker_color1.r, o.checker_color1.g, o.checker_color1.b };
                    float c2[3] = { o.checker_color2.r, o.checker_color2.g, o.checker_color2.b };
                    if (ImGui::ColorEdit3("Checker Color 1", c1)) {
                        o.checker_color1 = glm::vec3(c1[0], c1[1], c1[2]);
                        changed = true;
                    }
                    if (ImGui::ColorEdit3("Checker Color 2", c2)) {
                        o.checker_color2 = glm::vec3(c2[0], c2[1], c2[2]);
                        changed = true;
                    }
                }
            }

            if (ImGui::Button("Delete")) {
                delete_idx = i;
            }

            if (changed) any_changed = true;
        }

        ImGui::PopID();
    }

    if (delete_idx >= 0) {
        app.scene().remove_object(delete_idx);
        any_changed = true;
    }

    if (any_changed) {
        app.scene_modified();
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Add Object Panel
// ---------------------------------------------------------------------------

void Gui::draw_add_object(Window& app) {
    ImGui::SetNextWindowPos(ImVec2(340, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 320), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Add Object")) {
        ImGui::End();
        return;
    }

    const char* obj_types[] = { "Sphere", "Disc" };
    ImGui::Combo("Type", &_new_obj_type, obj_types, 2);

    ImGui::Combo("Material", &_new_obj_material, material_names, material_count);

    // Material-specific inputs
    SceneMaterial mat = material_from_index(_new_obj_material);
    if (mat == SceneMaterial::Lambertian || mat == SceneMaterial::Metal) {
        float col[3] = { _new_obj_albedo.r, _new_obj_albedo.g, _new_obj_albedo.b };
        if (ImGui::ColorEdit3("Albedo", col)) {
            _new_obj_albedo = glm::vec3(col[0], col[1], col[2]);
        }
    }
    if (mat == SceneMaterial::Metal) {
        ImGui::SliderFloat("Fuzz", &_new_obj_fuzz, 0.0f, 1.0f);
    }
    if (mat == SceneMaterial::Dielectric) {
        ImGui::SliderFloat("IOR", &_new_obj_ior, 1.0f, 3.0f);
    }
    if (mat == SceneMaterial::Emissive) {
        float em[3] = { _new_obj_emission.r, _new_obj_emission.g, _new_obj_emission.b };
        if (ImGui::DragFloat3("Emission", em, 0.1f, 0.0f, 50.0f)) {
            _new_obj_emission = glm::vec3(em[0], em[1], em[2]);
        }
    }

    ImGui::Separator();

    if (_new_obj_type == 0) {
        // Sphere
        float c[3] = { _new_sphere_center.x, _new_sphere_center.y, _new_sphere_center.z };
        ImGui::DragFloat3("Center", c, 0.05f);
        _new_sphere_center = glm::vec3(c[0], c[1], c[2]);
        ImGui::DragFloat("Radius", &_new_sphere_radius, 0.01f, 0.01f, 100.0f);

        if (ImGui::Button("Add Sphere")) {
            bool is_light = (mat == SceneMaterial::Emissive);
            glm::vec3 emission = is_light ? _new_obj_emission : glm::vec3(0.0f);
            app.scene().add_sphere(_new_sphere_center, _new_sphere_radius,
                                   mat, _new_obj_albedo, _new_obj_fuzz, _new_obj_ior,
                                   emission, is_light);
            app.scene_modified();
        }
    } else {
        // Disc
        float c[3] = { _new_disc_center.x, _new_disc_center.y, _new_disc_center.z };
        ImGui::DragFloat3("Center", c, 0.05f);
        _new_disc_center = glm::vec3(c[0], c[1], c[2]);

        float n[3] = { _new_disc_normal.x, _new_disc_normal.y, _new_disc_normal.z };
        ImGui::DragFloat3("Normal", n, 0.01f, -1.0f, 1.0f);
        _new_disc_normal = glm::vec3(n[0], n[1], n[2]);

        ImGui::DragFloat("Radius", &_new_disc_radius, 0.01f, 0.01f, 100.0f);

        if (ImGui::Button("Add Disc")) {
            SceneObject disc;
            disc.kind = ProxyKind::Disc;
            disc.center = _new_disc_center;
            disc.disc_normal = _new_disc_normal;
            disc.radius = _new_disc_radius;
            disc.material = mat;
            disc.albedo = _new_obj_albedo;
            disc.fuzz = _new_obj_fuzz;
            disc.ior = _new_obj_ior;
            disc.is_light = (mat == SceneMaterial::Emissive);
            disc.emission = disc.is_light ? _new_obj_emission : glm::vec3(0.0f);
            disc.color = disc.is_light ? glm::vec3(1.0f) : disc.albedo;
            app.scene().mutable_objects().push_back(disc);
            app.scene_modified();
        }
    }

    ImGui::End();
}

// ---------------------------------------------------------------------------
// File Loader Panel (OBJ + textures)
// ---------------------------------------------------------------------------

void Gui::draw_file_loader(Window& app) {
    ImGui::SetNextWindowPos(ImVec2(340, 340), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Load Model")) {
        ImGui::End();
        return;
    }

    // OBJ file
    ImGui::Text("OBJ: %s", _pending_obj_path.empty() ? "(none)" : _pending_obj_path.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Browse##obj")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseOBJ", "Select OBJ File", ".obj", config);
    }

    // Diffuse texture
    ImGui::Text("Diffuse: %s", _pending_diffuse_path.empty() ? "(auto from .mtl)" : _pending_diffuse_path.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Browse##diffuse")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseDiffuse", "Select Diffuse Texture", ".png,.jpg,.jpeg,.tga,.bmp", config);
    }

    // Normal map
    ImGui::Text("Normal: %s", _pending_normal_path.empty() ? "(auto from .mtl)" : _pending_normal_path.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Browse##normal")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseNormal", "Select Normal Map", ".png,.jpg,.jpeg,.tga,.bmp", config);
    }

    // Specular map
    ImGui::Text("Specular: %s", _pending_specular_path.empty() ? "(auto from .mtl)" : _pending_specular_path.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Browse##specular")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseSpecular", "Select Specular Map", ".png,.jpg,.jpeg,.tga,.bmp", config);
    }

    ImGui::Separator();

    bool can_load = !_pending_obj_path.empty();
    if (!can_load) ImGui::BeginDisabled();
    if (ImGui::Button("Load Model Into Scene")) {
        int idx = app.scene().add_obj_from_file(
            _pending_obj_path, _pending_diffuse_path,
            glm::vec3(0.0f, 0.5f, -1.0f), 0.5f,
            _pending_normal_path, _pending_specular_path);
        if (idx >= 0) {
            app.scene_modified();
            _pending_obj_path.clear();
            _pending_diffuse_path.clear();
            _pending_normal_path.clear();
            _pending_specular_path.clear();
        }
    }
    if (!can_load) ImGui::EndDisabled();

    if (ImGui::Button("Clear Selections")) {
        _pending_obj_path.clear();
        _pending_diffuse_path.clear();
        _pending_normal_path.clear();
        _pending_specular_path.clear();
    }

    // Handle file dialog results
    if (ImGuiFileDialog::Instance()->Display("ChooseOBJ")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            _pending_obj_path = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseDiffuse")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            _pending_diffuse_path = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseNormal")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            _pending_normal_path = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseSpecular")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            _pending_specular_path = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::End();
}
