#include "Window.h"

#include <iostream>
#include <iomanip>
#include <memory>

#include "SceneController.h"
#include "RenderLoop.h"
#include "InputHandler.h"
#include "Gui.h"

Window::Window(int width, int height, const RenderParams& params, std::string obj_path, std::string texture_path) {
	_window_width = width;
	_window_height = height;
	_params = params;
	if (_params.preview_scale < 1) _params.preview_scale = 1;
	_frame_count = 1;
	_camera_moving = false;
	_render_mode = RenderMode::PREVIEW;
	_rasterization_enabled = false;
	_obj_path = std::move(obj_path);
	_texture_path = std::move(texture_path);

	_scene_controller = std::make_unique<SceneController>();
	_render_loop = std::make_unique<RenderLoop>();
	_input_handler = std::make_unique<InputHandler>();
}

static void glfw_error_callback(int code, const char* description) {
	std::cerr << "[GLFW error " << code << "] " << description << "\n";
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	auto* w = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	if (w) w->on_mouse_button(button, action, mods);
}

int Window::init_glfw() {
	glfwSetErrorCallback(glfw_error_callback);
#if defined(__linux__) && defined(GLFW_PLATFORM)
	glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif

	if (!glfwInit()) {
		std::cout << "Failed to initialize GLFW" << "\n";
		return -1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

	_window = glfwCreateWindow(_window_width, _window_height, "A CUDA ray tracer", nullptr, nullptr);

	glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	glfwSetWindowUserPointer(_window, reinterpret_cast<void*>(this));
	glfwSetMouseButtonCallback(_window, mouse_button_callback);

	if (_window == nullptr) {
		std::cout << "Failed to create GLFW window" << "\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(_window);

	return 0;
}

void Window::on_mouse_button(int button, int action, int mods) {
	_input_handler->on_mouse_button(button, action, mods);
}

int Window::init_glad() {
	if (!gladLoadGL(glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << "\n";
		return -1;
	}
	return 0;
}

void Window::on_framebuffer_resize(int w, int h) {
	_window_width = w;
	_window_height = h;
	update_viewport();
}

void Window::update_viewport() {
	int render_w = _window_width;
	int vp_x = 0;

	if (_gui && _gui->visible()) {
		int pw = _gui->panel_width();
		if (pw >= _window_width) pw = _window_width / 2;
		render_w = _window_width - pw;
		vp_x = _gui->panel_on_right() ? 0 : pw;
	}

	_render_loop->set_viewport(vp_x, render_w, _window_height);
}

static void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
	auto* myWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	myWindow->on_framebuffer_resize(w, h);
}

int Window::init() {
	if (init_glfw() != 0) return -1;
	if (init_glad() != 0) return -1;

	glViewport(0, 0, _window_width, _window_height);
	glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);

	_render_loop->init_gl_resources(_window, _render_loop->shaders(), _scene_controller->scene(), _params);

	if (!_obj_path.empty()) {
		_scene_controller->load_obj_from_file(_obj_path, _texture_path);
	}
	_scene_controller->rebuild_world(_render_loop->renderer());

	_gui = std::make_unique<Gui>(_window);
	update_viewport();

	_last_frame = std::chrono::steady_clock::now();

	while (!glfwWindowShouldClose(_window)) {
		tick();
	}

	destroy();
	return 0;
}

Scene& Window::scene() { return _scene_controller->scene(); }
const Scene& Window::scene() const { return _scene_controller->scene(); }
KernelInfo& Window::renderer() { return _render_loop->renderer(); }
const KernelInfo& Window::renderer() const { return _render_loop->renderer(); }
bool Window::input_edit_mode() const { return _input_handler->edit_mode(); }

void Window::scene_modified() {
	_scene_controller->rebuild_world(_render_loop->renderer());
	_frame_count = 1;
}

void Window::reset_accumulation() {
	_frame_count = 1;
}

void Window::start_final_render() {
	if (_render_mode == RenderMode::PREVIEW) {
		_render_mode = RenderMode::RENDER_FINAL;
		_frame_count = 1;
		_rasterization_enabled = false;
	}
}

void Window::destroy() {
	_gui.reset();
	_render_loop->destroy();
	glfwDestroyWindow(_window);
	glfwTerminate();
}

void Window::tick() {
	std::chrono::steady_clock::time_point this_frame = std::chrono::steady_clock::now();
	float t_diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_frame - _last_frame).count();
	_last_frame = this_frame;

	_gui->new_frame();

	bool gui_was_visible = _gui->visible();
	bool was_raster_enabled = _rasterization_enabled;

	_input_handler->tick(_window, *_gui, *_scene_controller,
	                     _render_loop->renderer(),
	                     _render_loop->rasterizer(),
	                     _window_width, _window_height,
	                     _render_mode, _frame_count, _camera_moving,
	                     _rasterization_enabled, t_diff);

	if (_gui->visible() != gui_was_visible) {
		update_viewport();
	}

	if (was_raster_enabled && !_rasterization_enabled) {
		glClearTexImage(_render_loop->accum_texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	}

	_gui->draw(*this);

	_render_loop->render_frame(_window, _render_mode, _params,
	                           _frame_count, _camera_moving,
	                           _rasterization_enabled);

	_gui->render();

	glfwSwapBuffers(_window);
	glfwPollEvents();

	if (_render_mode != RenderMode::IDLE)
		_frame_count++;
}