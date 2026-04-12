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
#include "Scene.h"
#include "raytracer/kernel.h"

#include <string>

enum class RenderMode { PREVIEW, RENDER_FINAL, IDLE };

class Window {
public:
	unsigned int width;
	unsigned int height;
	int samples;
	int max_depth;
	float fov;
	int tile_size;
	int preview_scale;

	Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov, int tile_size = DEFAULT_TILE_SIZE, int preview_scale = 1, std::string obj_path = "", std::string texture_path = "");

	int init();
	void destroy();
	void resize(unsigned int width, unsigned int height);

	// Called from GLFW callbacks (which only have a GLFWwindow*, so we look
	// the Window up via glfwGetWindowUserPointer).
	void on_mouse_button(int button, int action, int mods);
private:
	GLFWwindow* _window;

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
	std::string _obj_path;
	std::string _texture_path;

	int init_glad();
	int init_glfw();
	int init_framebuffer();
	int init_quad();

	Input _input;
	int _frame_count;
	bool _camera_moving;
	RenderMode _render_mode;
	bool _enter_was_pressed;
	bool _r_was_pressed;
	bool _tab_was_pressed;
	bool _rasterization_enabled;
	std::chrono::steady_clock::time_point _last_frame;


	void tick_input(float t_diff);
	void tick_render();
	void tick();
};