#pragma once

#include "Input.h"
#include "raytracer/kernel.h"
#include "RenderState.h"

class Gui;
class Rasterizer;
class SceneController;

class InputHandler {
public:
	InputHandler();

	InputHandler(const InputHandler&) = delete;
	InputHandler& operator=(const InputHandler&) = delete;

	Input& input() { return _input; }
	const Input& input() const { return _input; }

	void on_mouse_button(int button, int action, int mods);
	bool edit_mode() const { return _input.edit_mode; }

	void tick(GLFWwindow* window,
	          Gui& gui,
	          SceneController& scene_controller,
	          KernelInfo& renderer,
	          Rasterizer& rasterizer,
	          RenderState& state,
	          float t_diff);

private:
	Input _input;
	bool _enter_was_pressed = false;
	bool _r_was_pressed = false;
	bool _tab_was_pressed = false;
	bool _g_was_pressed = false;
};
