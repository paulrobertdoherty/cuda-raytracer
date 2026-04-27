#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <memory>
#include <string>

#include "RenderParams.h"
#include "RenderMode.h"
#include "SceneController.h"
#include "RenderLoop.h"
#include "InputHandler.h"
#include "Gui.h"
#include "raytracer/kernel.h"

class Window {
public:
	Window(int width, int height, const RenderParams& params, std::string obj_path = "", std::string texture_path = "");

	RenderParams& render_params() { return _params; }
	const RenderParams& render_params() const { return _params; }

	Scene& scene();
	const Scene& scene() const;
	KernelInfo& renderer();
	const KernelInfo& renderer() const;
	bool input_edit_mode() const;

	void scene_modified();
	void reset_accumulation();
	void start_final_render();

	int window_width() const { return _window_width; }
	int window_height() const { return _window_height; }

	void on_mouse_button(int button, int action, int mods);
	void on_framebuffer_resize(int w, int h);

	int init();
	void destroy();
	void update_viewport();

private:
	GLFWwindow* _window = nullptr;
	int _window_width = 0;
	int _window_height = 0;

	std::unique_ptr<SceneController> _scene_controller;
	std::unique_ptr<RenderLoop> _render_loop;
	std::unique_ptr<InputHandler> _input_handler;
	std::unique_ptr<Gui> _gui;

	RenderParams _params;
	RenderMode _render_mode = RenderMode::PREVIEW;
	int _frame_count = 1;
	bool _camera_moving = false;
	bool _rasterization_enabled = false;

	std::string _obj_path;
	std::string _texture_path;

	std::chrono::steady_clock::time_point _last_frame;

	int init_glfw();
	static int init_glad();
	void tick();
};