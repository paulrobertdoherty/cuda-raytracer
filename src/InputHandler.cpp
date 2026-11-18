#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "InputHandler.h"
#include "Gui.h"
#include "SceneController.h"
#include "Rasterizer.h"

InputHandler::InputHandler() = default;

void InputHandler::on_mouse_button(int button, int action, int mods) {
	_input.on_mouse_button(button, action, mods);
}

void InputHandler::tick(GLFWwindow* window,
                        Gui& gui,
                        SceneController& scene_controller,
                        KernelInfo& renderer,
                        Rasterizer& rasterizer,
                        RenderState& state,
                        float t_diff) {
	_input.process_quit(window);

	bool g_down = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
	if (g_down && !_g_was_pressed) {
		gui.toggle();
		if (gui.visible() || _input.edit_mode) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			double xpos;
			double ypos;
			glfwGetCursorPos(window, &xpos, &ypos);
			_input.last_xpos = xpos;
			_input.last_ypos = ypos;
		}
	}
	_g_was_pressed = g_down;

	bool imgui_kb = gui.visible() && gui.wants_keyboard();
	bool imgui_mouse = gui.visible() && gui.wants_mouse();

	bool enter_down = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if (enter_down && !_enter_was_pressed && !imgui_kb) {
		state.toggle_final();
	}
	_enter_was_pressed = enter_down;

	// Track key edges in every mode so held keys do not fire on resuming preview.
	bool r_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
	bool toggle_raster = r_down && !_r_was_pressed;
	_r_was_pressed = r_down;
	bool tab_down = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
	bool toggle_edit = tab_down && !_tab_was_pressed && !imgui_kb;
	_tab_was_pressed = tab_down;

	auto* preview = state.active_preview();
	if (!preview) {
		return;
	}

	if (toggle_raster) {
		preview->raster_enabled = !preview->raster_enabled;
		state.reset_accumulation();
	}
	if (toggle_edit) {
		_input.edit_mode = !_input.edit_mode;
		if (_input.edit_mode) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			preview->raster_enabled = true;
			state.reset_accumulation();
			_input.lmb_was_down = false;
			_input.selected_idx = -1;
		} else {
			if (!gui.visible()) {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				double xpos;
				double ypos;
				glfwGetCursorPos(window, &xpos, &ypos);
				_input.last_xpos = xpos;
				_input.last_ypos = ypos;
			}
			rasterizer.set_selected(-1);
		}
	}

	if (_input.edit_mode) {
		if (!imgui_mouse) {
			int w = 0;
			int h = 0;
			glfwGetWindowSize(window, &w, &h);
			int moved_index = _input.process_edit_mouse(window,
				scene_controller.scene(), renderer.camera_info, w, h);
			if (moved_index >= 0) {
				renderer.update_object(scene_controller.scene(), moved_index);
			}
		}

		rasterizer.set_selected(_input.selected_idx);

		if (_input.lmb_down) {
			state.reset_accumulation();
		}
		preview->camera_moving = false;
	} else if (!imgui_kb) {
		_input.process_camera_movement(window, renderer, t_diff);
		preview->camera_moving = _input.has_camera_moved();
		if (preview->camera_moving) {
			state.reset_accumulation();
		}
	} else {
		preview->camera_moving = false;
	}
}
