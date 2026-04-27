#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "InputHandler.h"
#include "Gui.h"
#include "SceneController.h"
#include "Rasterizer.h"

InputHandler::InputHandler() {}

void InputHandler::on_mouse_button(int button, int action, int mods) {
	_input.on_mouse_button(button, action, mods);
}

void InputHandler::tick(GLFWwindow* window,
                        Gui& gui,
                        SceneController& scene_controller,
                        KernelInfo& renderer,
                        Rasterizer& rasterizer,
                        int /*window_w*/, int /*window_h*/,
                        RenderMode& render_mode,
                        int& frame_count,
                        bool& camera_moving,
                        bool& raster_enabled,
                        float t_diff) {
	_input.process_quit(window);

	bool g_down = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
	if (g_down && !_g_was_pressed) {
		gui.toggle();
		if (gui.visible() || _input.edit_mode) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			double xpos, ypos;
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
		if (render_mode == RenderMode::PREVIEW) {
			render_mode = RenderMode::RENDER_FINAL;
			frame_count = 1;
			raster_enabled = false;
		} else if (render_mode == RenderMode::IDLE) {
			render_mode = RenderMode::PREVIEW;
		}
	}
	_enter_was_pressed = enter_down;

	bool r_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
	if (r_down && !_r_was_pressed && render_mode == RenderMode::PREVIEW) {
		raster_enabled = !raster_enabled;
		frame_count = 1;
	}
	_r_was_pressed = r_down;

	bool tab_down = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
	if (tab_down && !_tab_was_pressed && render_mode == RenderMode::PREVIEW && !imgui_kb) {
		_input.edit_mode = !_input.edit_mode;
		if (_input.edit_mode) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			raster_enabled = true;
			frame_count = 1;
			_input.lmb_was_down = false;
			_input.selected_idx = -1;
		} else {
			if (!gui.visible()) {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				double xpos, ypos;
				glfwGetCursorPos(window, &xpos, &ypos);
				_input.last_xpos = xpos;
				_input.last_ypos = ypos;
			}
			rasterizer.set_selected(-1);
		}
	}
	_tab_was_pressed = tab_down;

	if (render_mode == RenderMode::PREVIEW) {
		if (_input.edit_mode) {
			if (!imgui_mouse) {
				int w = 0, h = 0;
				glfwGetWindowSize(window, &w, &h);
				_input.process_edit_mouse(window,
					scene_controller.scene(),
					renderer.camera_info, w, h);
			}

			rasterizer.set_selected(_input.selected_idx);

			if (_input.scene_dirty && !_input.lmb_down) {
				renderer.rebuild_world(scene_controller.scene());
				_input.scene_dirty = false;
				frame_count = 1;
			}
			if (_input.lmb_down) frame_count = 1;
			camera_moving = false;
		} else if (!imgui_kb) {
			_input.process_camera_movement(window, renderer, t_diff);
			camera_moving = _input.has_camera_moved();
			if (camera_moving) frame_count = 1;
		} else {
			camera_moving = false;
		}
	}
}