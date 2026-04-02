#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>

#include <algorithm>

#include "Shader.h"
#include "Quad.h"
#include "Input.h"

enum class RenderMode { PREVIEW, RENDER_FINAL, IDLE };

class Window {
public:
	unsigned int width;
	unsigned int height;
	int samples;
	int max_depth;
	float fov;

	Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov);

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

	int init_glad();
	int init_glfw();
	int init_framebuffer();
	int init_quad();

	Input _input;
	int _frame_count;
	bool _camera_moving;
	RenderMode _render_mode;
	bool _enter_was_pressed;
	std::chrono::steady_clock::time_point _last_frame;

	// Convergence test: set to true after the one-time comparison fires.
	// Guarded by CONVERGENCE_TEST so it compiles away when disabled.
#ifdef CONVERGENCE_TEST
	bool _convergence_test_done = false;
#endif


	void tick_input(float t_diff);
	void tick_render();
	void tick();
};