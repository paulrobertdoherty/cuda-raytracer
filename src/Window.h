#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>

#include <algorithm>

#include "Shader.h"
#include "ShaderManager.h"
#include "Quad.h"
#include "Input.h"
#include "Rasterizer.h"
#include "RenderParams.h"
#include "Scene.h"
#include "Gui.h"
#include "raytracer/kernel.h"

#include <string>

enum class RenderMode { PREVIEW, RENDER_FINAL, IDLE };

class Window {
public:
	// Render viewport size (the area the ray tracer renders into).
	unsigned int width;
	unsigned int height;

	Window(unsigned int width, unsigned int height, const RenderParams& params, std::string obj_path = "", std::string texture_path = "");

	// Interactive tuning knobs. Returned by reference so ImGui widgets can
	// bind to the fields directly; Window picks up changes on the next tick.
	RenderParams& render_params() { return _params; }
	const RenderParams& render_params() const { return _params; }

	int init();
	void destroy();
	void resize(unsigned int width, unsigned int height);

	// Called from GLFW callbacks (which only have a GLFWwindow*, so we look
	// the Window up via glfwGetWindowUserPointer).
	void on_mouse_button(int button, int action, int mods);

	// Public accessors for the GUI to interact with
	Scene& scene() { return *_scene; }
	KernelInfo& renderer() { return *_current_frame->_renderer; }
	bool input_edit_mode() const { return _input.edit_mode; }

	// Rebuild the CUDA world from the scene and reset accumulation.
	void scene_modified();
	// Reset frame accumulation without rebuilding the world.
	void reset_accumulation();
	// Switch to RENDER_FINAL mode.
	void start_final_render();

	// Full GLFW framebuffer dimensions.
	unsigned int window_width() const { return _window_width; }
	unsigned int window_height() const { return _window_height; }

	// Called by the GLFW framebuffer-size callback when the window is resized.
	void on_framebuffer_resize(unsigned int w, unsigned int h);

	// Recompute the render viewport from the full window size and panel state.
	// Called after window resize, GUI toggle, or panel-side change.
	void update_viewport();

	// The pixel offset where the render viewport begins (accounts for sidebar).
	int viewport_x() const { return _viewport_x; }

private:
	GLFWwindow* _window;

	unsigned int _window_width;
	unsigned int _window_height;
	int _viewport_x = 0;

	ShaderManager _shaders;
	Shader* _screen_shader = nullptr;
	Shader* _accum_shader = nullptr;
	std::unique_ptr<Quad> _blit_quad;
	std::unique_ptr<Quad> _accum_frame;
	std::unique_ptr<Quad> _current_frame;
	std::unique_ptr<Quad> _raster_frame;
	GLuint _raster_depth_rb;
	std::unique_ptr<Rasterizer> _rasterizer;
	std::unique_ptr<Scene> _scene;
	std::unique_ptr<Gui> _gui;
	std::string _obj_path;
	std::string _texture_path;

	int init_glad();
	int init_glfw();
	int init_framebuffer();
	int init_quad();

	RenderParams _params;
	Input _input;
	int _frame_count;
	bool _camera_moving;
	RenderMode _render_mode;
	bool _enter_was_pressed;
	bool _r_was_pressed;
	bool _tab_was_pressed;
	bool _rasterization_enabled;
	bool _g_was_pressed;
	std::chrono::steady_clock::time_point _last_frame;

	void tick_input(float t_diff);
	void tick_render();
	void tick();
};
