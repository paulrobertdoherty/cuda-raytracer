#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>

#include <algorithm>

#include "Shader.h"
#include "Quad.h"
#include "Input.h"
#include "Rasterizer.h"
#include "raytracer/kernel.h"

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

	Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov, int tile_size = DEFAULT_TILE_SIZE, int preview_scale = 1);

	int init();
	void destroy();
	void resize(unsigned int width, unsigned int height);
private:
	GLFWwindow* _window;

	std::unique_ptr<Shader> _shader;
	std::unique_ptr<Shader> _accum_shader;
	std::unique_ptr<Quad> _blit_quad;
	std::unique_ptr<Quad> _accum_frame;
	std::unique_ptr<Quad> _current_frame;
	std::unique_ptr<Quad> _raster_frame;
	GLuint _raster_depth_rb;
	std::unique_ptr<Rasterizer> _rasterizer;

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
	bool _rasterization_enabled;
	std::chrono::steady_clock::time_point _last_frame;


	void tick_input(float t_diff);
	void tick_render();
	void tick();
};