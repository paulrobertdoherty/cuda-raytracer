#pragma once

#include "raytracer/Camera.h"
#include "raytracer/kernel.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <GLFW/glfw3.h>

class Scene;

struct Input {
	double last_xpos;
	double last_ypos;
	glm::vec3 speed;
	bool has_moved;

	// --- Edit mode state ---
	bool edit_mode = false;
	bool lmb_down = false;
	bool lmb_was_down = false;
	int selected_idx = -1;
	glm::vec3 drag_anchor = glm::vec3(0.0f);
	glm::vec3 drag_plane_normal = glm::vec3(0.0f, 0.0f, 1.0f);
	// True when the scene has been modified since the last drag release — set
	// on drag, cleared by the caller after it rebuilds the CUDA world.
	bool scene_dirty = false;

	Input() {
		speed = glm::vec3(0.0f, 0.0f, 0.0f);
		last_xpos = 0;
		last_ypos = 0;
		has_moved = false;
	}
	void process_quit(GLFWwindow* window);
	void process_camera_movement(GLFWwindow* window, KernelInfo& kernelInfo, float t_diff);

	// Mouse button events arriving from the GLFW callback installed on the
	// Window. Updates lmb_down; actual picking/drag logic lives in
	// process_edit_mouse.
	void on_mouse_button(int button, int action, int mods);

	// Process edit-mode mouse interaction: picking on LMB press, drag plane
	// reprojection while held, and selection teardown on release.
	void process_edit_mouse(GLFWwindow* window,
	                        Scene& scene,
	                        const CameraInfo& cam,
	                        int window_w, int window_h);

	bool has_camera_moved();
};
