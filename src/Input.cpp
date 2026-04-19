#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include <glm/glm.hpp>

#include "Input.h"
#include "Scene.h"
#include "Picker.h"


#define M_PI 3.14159265358979323846264338327950288f

#define isPressed(x) glfwGetKey(window,x)==GLFW_PRESS

void Input::process_quit(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

bool Input::has_camera_moved() {
	return has_moved;
}

void Input::process_camera_movement(GLFWwindow* window, KernelInfo& kernelInfo, float t_diff) {
	has_moved = false;
	glm::vec3 position = kernelInfo.camera_info.origin;
	glm::vec3 rotation = kernelInfo.camera_info.rotation;

	// ** rotation **
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	double x_diff = 0.0, y_diff = 0.0;

	int width, height;

	glfwGetWindowSize(window, &width, &height);

	// Mouse-look only makes sense when the cursor is captured (sidebar hidden,
	// not in edit mode). When the cursor is visible, the user is interacting
	// with the GUI, and applying a yaw/pitch from every cursor move would drag
	// the camera around with the mouse. Still update last_xpos/last_ypos so we
	// don't apply a huge accumulated delta on the frame the cursor re-captures.
	bool cursor_captured = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
	if (cursor_captured) {
		x_diff = (xpos - last_xpos) / width;
		y_diff = (ypos - last_ypos) / height;
		if (x_diff != 0 || y_diff != 0) has_moved = true;
	}

	// pitch
	rotation.x += y_diff * 30.0f;
	// yaw
	rotation.y += x_diff * 30.0f;

	// roll
	if (isPressed(GLFW_KEY_E)) {
		rotation.z += 1.0f;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_Q)) {
		rotation.z -= 1.0f;
		has_moved = true;
	}

	kernelInfo.camera_info.rotation = rotation;

	float A = degrees_to_radians(rotation.x);
	float B = degrees_to_radians(rotation.y);
	float C = degrees_to_radians(rotation.z);

	glm::vec3 forward;
	glm::vec3 up;

	if (rotation.x == 0 && rotation.y == 0 && rotation.z == 0) {
		forward = glm::vec3(0.0f, 0.0f, 1.0f);
		up = glm::vec3(0.0f, 1.0f, 0.0f);
	}
	else {
		forward = glm::vec3(-cos(A) * sin(B) * cos(C) + sin(A) * sin(C), cos(A) * sin(B) * sin(C) + sin(A) * cos(C), cos(A) * cos(B));
		up = glm::vec3(sin(A) * sin(B) * cos(C) + cos(A) * sin(C), -sin(A) * sin(B) * sin(C) + cos(A) * cos(C), -sin(A) * cos(B));
	}

	last_xpos = xpos;
	last_ypos = ypos;
	// **

	float SPEED_ = 0.125f * (t_diff / 20.0f);

	if (isPressed(GLFW_KEY_W)) {
		speed.z += SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_A)) {
		speed.x -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_S)) {
		speed.z -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_D)) {
		speed.x += SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_LEFT_CONTROL)) {
		speed.y -= SPEED_;
		has_moved = true;
	}
	if (isPressed(GLFW_KEY_SPACE)) {
		speed.y += SPEED_;
		has_moved = true;
	}

	position += glm::cross(up, forward) * speed.x * 0.1f;
	position.y += speed.y * 0.1f;
	position += forward * -speed.z * 0.1f;

	for (int i = 0; i < 3; i++) {
		if (abs(speed[i]) < 0.05) {
			speed[i] = 0.0;
		}
		if (speed[i] > 0.0) {
			speed[i] -= 0.075 * (t_diff / 20.0f);
		}
		else if (speed[i] < 0.0) {
			speed[i] += 0.075 * (t_diff / 20.0f);
		}
	}

	kernelInfo.camera_info.origin = position;

	kernelInfo.set_camera(position, forward, up);
}

void Input::on_mouse_button(int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		lmb_down = (action == GLFW_PRESS);
	}
}

void Input::process_edit_mouse(GLFWwindow* window,
                               Scene& scene,
                               const CameraInfo& cam,
                               int window_w, int window_h) {
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	bool pressed_this_frame = lmb_down && !lmb_was_down;
	bool released_this_frame = !lmb_down && lmb_was_down;

	if (pressed_this_frame) {
		glm::vec3 ro, rd;
		Picker::cursor_to_ray(cam, xpos, ypos, window_w, window_h, ro, rd);

		float t;
		int idx;
		glm::vec3 hit_point;
		if (scene.ray_intersect(ro, rd, t, idx, hit_point)) {
			selected_idx = idx;
			drag_anchor = hit_point;

			// Drag plane: parallel to the screen, passing through the hit
			// point. Camera looks in -forward (see Camera.h), so pick the
			// forward vector from Picker and use it directly as the plane
			// normal (points toward the camera when negated).
			glm::vec3 forward, up;
			Picker::compute_basis(cam, forward, up);
			drag_plane_normal = forward;
		} else {
			selected_idx = -1;
		}
	}

	if (lmb_down && selected_idx >= 0) {
		glm::vec3 ro, rd;
		Picker::cursor_to_ray(cam, xpos, ypos, window_w, window_h, ro, rd);
		glm::vec3 new_point;
		if (Picker::intersect_plane(ro, rd, drag_anchor, drag_plane_normal, new_point)) {
			glm::vec3 delta = new_point - drag_anchor;
			if (glm::dot(delta, delta) > 0.0f) {
				scene.mutable_objects()[selected_idx].position += delta;
				drag_anchor = new_point;
				scene_dirty = true;
			}
		}
	}

	if (released_this_frame) {
		// Keep the selected_idx until the next press so the outline persists
		// briefly; the caller uses scene_dirty to decide when to rebuild the
		// CUDA world.
		selected_idx = -1;
	}

	lmb_was_down = lmb_down;
	last_xpos = xpos;
	last_ypos = ypos;
}
