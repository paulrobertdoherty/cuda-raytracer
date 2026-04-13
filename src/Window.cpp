#include "Window.h"

#include <iostream>
#include <iomanip>
#include <memory>

#include "Rasterizer.h"

Window::Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov, int tile_size, int preview_scale, std::string obj_path, std::string texture_path) {
	Window::width = width;
	Window::height = height;
	Window::samples = samples;
	Window::max_depth = max_depth;
	Window::fov = fov;
	Window::tile_size = tile_size;
	Window::preview_scale = preview_scale < 1 ? 1 : preview_scale;
	Window::_frame_count = 0;
	Window::_camera_moving = false;
	Window::_render_mode = RenderMode::PREVIEW;
	Window::_enter_was_pressed = false;
	Window::_r_was_pressed = false;
	Window::_rasterization_enabled = false;
	Window::_raster_depth_rb = 0;
	Window::_obj_path = std::move(obj_path);
	Window::_texture_path = std::move(texture_path);
	Window::_tab_was_pressed = false;
	Window::_g_was_pressed = false;
}

static void glfw_error_callback(int code, const char* description) {
	std::cerr << "[GLFW error " << code << "] " << description << std::endl;
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	Window* w = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	if (w) w->on_mouse_button(button, action, mods);
}

int Window::init_glfw() {
	glfwSetErrorCallback(glfw_error_callback);
#if defined(__linux__)
	// On Wayland sessions, GLFW's Wayland/EGL backend binds to the Mesa EGL
	// implementation on the integrated GPU and ignores __NV_PRIME_RENDER_OFFLOAD,
	// so cudaGraphicsGLRegisterBuffer fails (or no window appears at all).
	// Force the X11 backend so we go through XWayland + GLX, which honors PRIME
	// offload and lets the GL context live on the NVIDIA dGPU alongside CUDA.
	glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif

	// Initialize and configure GLFW
	if (!glfwInit()) {
		std::cout << "Failed to initialize GLFW" << std::endl;
		return -1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Initialize and create window for GLFW
	_window = glfwCreateWindow(Window::width, Window::height, "A CUDA ray tracer", NULL, NULL);

	// Hides the cursor and captures it (default fly mode).
	glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetWindowUserPointer(_window, reinterpret_cast<void*>(this));

	// Install mouse button callback so edit-mode click-and-drag can pick
	// objects. The callback forwards to Window::on_mouse_button which
	// delegates to Input.
	glfwSetMouseButtonCallback(_window, mouse_button_callback);

	// Check if window was created
	if (_window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(_window);

	return 0;
}

void Window::on_mouse_button(int button, int action, int mods) {
	_input.on_mouse_button(button, action, mods);
}

int Window::init_glad() {
	// Initialize GLAD before calling any OpenGL function
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	return 0;
}

void Window::resize(unsigned int w, unsigned int h) {
	this->width = w;
	this->height = h;
	this->_current_frame->resize(w, h);
	this->_accum_frame->resize(w, h);
	this->_blit_quad->resize(w, h);
	this->_raster_frame->resize(w, h);

	// Resize the depth renderbuffer attached to _raster_frame and re-attach.
	if (_raster_depth_rb != 0) {
		glBindRenderbuffer(GL_RENDERBUFFER, _raster_depth_rb);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
		glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _raster_depth_rb);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// resize the frame buffer
	glViewport(0, 0, width, height);
	Window* myWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

	myWindow->resize(width, height);
}

int Window::init_framebuffer() {
	// Set dimensions of frame buffer and map coordinates
	glViewport(0, 0, Window::width, Window::height);

	// Set callback function for when window gets resized
	glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);

	return 0;
}

int Window::init_quad() {

	// Pre-load the shaders needed for screen blit and motion-blur accumulation.
	// The rasterization shaders ("flat" and "mesh_textured") are loaded on
	// demand by the Rasterizer from the same ShaderManager.
	_screen_shader = _shaders.load("screen",
		"./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_accum_shader = _shaders.load("accumulate",
		"./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");

	_blit_quad = std::make_unique<Quad>(Window::width, Window::height);
	_blit_quad->make_FBO();

	_current_frame = std::make_unique<Quad>(Window::width, Window::height);
	_current_frame->cuda_init(samples, max_depth, fov);
	_current_frame->make_FBO();

	_accum_frame = std::make_unique<Quad>(Window::width, Window::height);
	_accum_frame->make_FBO();

	_accum_shader->use();
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "currentFrameTex"), 0);
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "lastFrameTex"), 1);

	// Rasterization-mode framebuffer (color + depth) and the rasterizer itself.
	_raster_frame = std::make_unique<Quad>(Window::width, Window::height);
	_raster_frame->make_FBO();

	glGenRenderbuffers(1, &_raster_depth_rb);
	glBindRenderbuffer(GL_RENDERBUFFER, _raster_depth_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, Window::width, Window::height);
	glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _raster_depth_rb);

	// Initialize raster_frame texture to opaque black so unrasterized sessions
	// degrade gracefully when used as a RENDER_FINAL underlay.
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Scene owns meshes/textures. Must be created after a GL context exists so
	// GLTexture uploads work, and before the Rasterizer so the Rasterizer can
	// bind it by reference.
	_scene = std::make_unique<Scene>();
	if (!_obj_path.empty()) {
		_scene->add_obj_from_file(_obj_path, _texture_path, glm::vec3(0.0f, 0.5f, -1.0f), 0.5f);
	}

	_rasterizer = std::make_unique<Rasterizer>(_shaders, *_scene);

	// Replace the hardcoded CUDA world with the Scene-derived one so the ray
	// tracer renders exactly what the rasterizer previews.
	_current_frame->_renderer->rebuild_world(*_scene);

	// Initialize the ImGui-based GUI overlay.
	_gui = std::make_unique<Gui>(_window);

	return 0;
}

int Window::init() {

	if (init_glfw() != 0) return -1;
	if (init_glad() != 0) return -1;
	if (init_framebuffer() != 0) return -1;
	if (init_quad() != 0) return -1;

	_last_frame = std::chrono::steady_clock::now();

	while (!glfwWindowShouldClose(_window)) {
		Window::tick();
	}

	Window::destroy();
	return 0;
}

void Window::scene_modified() {
	_current_frame->_renderer->rebuild_world(*_scene);
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

	// Shut down ImGui before destroying the GL context.
	_gui.reset();

	// Terminate CUDA allocated buffer
	_current_frame->cuda_destroy();

	// Terminate GLFW
	glfwDestroyWindow(_window);
	glfwTerminate();
}

void Window::tick_input(float t_diff) {

	_input.process_quit(_window);

	// G key: toggle GUI visibility (always processed, even when ImGui has focus)
	bool g_down = glfwGetKey(_window, GLFW_KEY_G) == GLFW_PRESS;
	if (g_down && !_g_was_pressed) {
		_gui->toggle();
		// When GUI becomes visible, show cursor so user can interact with it.
		// When hidden, restore the previous cursor mode.
		if (_gui->visible() || _input.edit_mode) {
			glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		} else {
			glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			double xpos, ypos;
			glfwGetCursorPos(_window, &xpos, &ypos);
			_input.last_xpos = xpos;
			_input.last_ypos = ypos;
		}
	}
	_g_was_pressed = g_down;

	// When ImGui wants keyboard, skip game input handling (except quit and G toggle above)
	bool imgui_kb = _gui->visible() && _gui->wants_keyboard();
	bool imgui_mouse = _gui->visible() && _gui->wants_mouse();

	// Detect Enter key press (rising edge) — skip when ImGui has keyboard
	bool enter_down = glfwGetKey(_window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if (enter_down && !_enter_was_pressed && !imgui_kb) {
		if (_render_mode == RenderMode::PREVIEW) {
			_render_mode = RenderMode::RENDER_FINAL;
			_frame_count = 1;
			// Rasterization is a preview-only feature; turn it off when
			// transitioning to the final render. The last rasterized frame
			// remains in _raster_frame and is used as an underlay below.
			_rasterization_enabled = false;
		} else if (_render_mode == RenderMode::IDLE) {
			_render_mode = RenderMode::PREVIEW;
		}
	}
	_enter_was_pressed = enter_down;

	// Detect R key press (rising edge) — toggle rasterization preview.
	bool r_down = glfwGetKey(_window, GLFW_KEY_R) == GLFW_PRESS;
	if (r_down && !_r_was_pressed && _render_mode == RenderMode::PREVIEW && !imgui_kb) {
		_rasterization_enabled = !_rasterization_enabled;
		// Restart accumulation when toggling so the first ray-traced frame
		// after switching back is treated as frame 1.
		_frame_count = 1;
	}
	_r_was_pressed = r_down;

	// Detect Tab key press (rising edge) — toggle edit mode.
	bool tab_down = glfwGetKey(_window, GLFW_KEY_TAB) == GLFW_PRESS;
	if (tab_down && !_tab_was_pressed && _render_mode == RenderMode::PREVIEW && !imgui_kb) {
		_input.edit_mode = !_input.edit_mode;
		if (_input.edit_mode) {
			// Show cursor, freeze camera movement. Force rasterization on so
			// the user sees object outlines while editing.
			glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			_rasterization_enabled = true;
			_frame_count = 1;
			// Clear any drag state from a previous edit session.
			_input.lmb_was_down = false;
			_input.selected_idx = -1;
		} else {
			// Back to fly mode. Only hide cursor if the GUI overlay is also
			// hidden — if the GUI is visible the cursor must remain so the
			// user can interact with the panels.
			if (!_gui->visible()) {
				glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				double xpos, ypos;
				glfwGetCursorPos(_window, &xpos, &ypos);
				_input.last_xpos = xpos;
				_input.last_ypos = ypos;
			}
			_rasterizer->set_selected(-1);
		}
	}
	_tab_was_pressed = tab_down;

	if (_render_mode == RenderMode::PREVIEW) {
		if (_input.edit_mode) {
			// Only forward mouse events to edit-mode picking when ImGui
			// doesn't want the mouse (so clicking on a panel doesn't drag
			// a scene object).
			if (!imgui_mouse) {
				int w = 0, h = 0;
				glfwGetWindowSize(_window, &w, &h);
				_input.process_edit_mouse(_window, *_scene,
					_current_frame->_renderer->camera_info, w, h);
			}

			// Keep the rasterizer's highlight in sync with the active selection.
			_rasterizer->set_selected(_input.selected_idx);

			// While dragging, restart accumulation so the ray-traced preview
			// stays sharp; on release, rebuild the CUDA world from the
			// modified scene so the next ray-traced frame reflects the new
			// layout.
			if (_input.scene_dirty && !_input.lmb_down) {
				_current_frame->_renderer->rebuild_world(*_scene);
				_input.scene_dirty = false;
				_frame_count = 1;
			}
			if (_input.lmb_down) _frame_count = 1;
			_camera_moving = false;
		} else if (!_gui->visible()) {
			// Camera fly mode — only when the GUI overlay is hidden so
			// the cursor is captured and mouse-look makes sense.
			_input.process_camera_movement(_window, *(_current_frame->_renderer), t_diff);
			_camera_moving = _input.has_camera_moved();
			if (_camera_moving) _frame_count = 1;
		} else {
			// GUI is visible but not in edit mode — show cursor for GUI
			// interaction but don't process camera movement.
			_camera_moving = false;
		}
	}
}

void copyFrameBufferTexture(int width, int height, int fboIn, int textureIn, int fboOut, int textureOut) {
	// Bind input FBO + texture to a color attachment
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fboIn);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIn, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	// Bind destination FBO + texture to another color attachment
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboOut);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureOut, 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT1);

	// Specify source, destination drawing (sub)rectangles.
	glBlitFramebuffer(0, 0, width, height,
		0, 0, width, height,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	// Unbind the color attachments
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
}

void Window::tick_render() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if (_render_mode == RenderMode::RENDER_FINAL) {
		// Seed _accum_frame with the rasterized underlay (or solid black if
		// rasterization was never used). Completed ray-traced tiles are blitted
		// directly into _accum_frame as they finish — no alpha channel needed.
		copyFrameBufferTexture(Window::width, Window::height,
			_raster_frame->framebuffer, _raster_frame->texture,
			_accum_frame->framebuffer, _accum_frame->texture);

		// Tiled progressive rendering: render one tile at a time, displaying
		// each tile as it completes so the image "paints in" progressively.
		bool aborted = false;
		int tiles_x = (width + tile_size - 1) / tile_size;
		int tiles_y = (height + tile_size - 1) / tile_size;

		for (int ty = 0; ty < tiles_y && !aborted; ty++) {
			for (int tx = 0; tx < tiles_x && !aborted; tx++) {
				int tile_ox = tx * tile_size;
				int tile_oy = ty * tile_size;
				int tile_w = (tile_ox + tile_size > (int)width) ? (int)width - tile_ox : tile_size;
				int tile_h = (tile_oy + tile_size > (int)height) ? (int)height - tile_oy : tile_size;

				// Map PBO so CUDA can write to it
				_current_frame->_renderer->map_pbo();

				// Launch kernel for just this tile
				_current_frame->_renderer->render_tile(tile_ox, tile_oy, tile_w, tile_h, samples);

				// Unmap PBO so OpenGL can read from it
				_current_frame->_renderer->unmap_pbo();

				// Upload this tile's pixels from PBO to texture
				_current_frame->upload_tile(tile_ox, tile_oy, tile_w, tile_h);

				// Blit just the completed tile from _current_frame into
				// _accum_frame, overwriting the rasterized underlay for that
				// region. Untouched regions keep the rasterized image.
				glBindFramebuffer(GL_READ_FRAMEBUFFER, _current_frame->framebuffer);
				glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _current_frame->texture, 0);
				glReadBuffer(GL_COLOR_ATTACHMENT0);
				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _accum_frame->framebuffer);
				glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, _accum_frame->texture, 0);
				glDrawBuffer(GL_COLOR_ATTACHMENT1);
				glBlitFramebuffer(tile_ox, tile_oy, tile_ox + tile_w, tile_oy + tile_h,
				                  tile_ox, tile_oy, tile_ox + tile_w, tile_oy + tile_h,
				                  GL_COLOR_BUFFER_BIT, GL_NEAREST);
				glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
				glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);

				// Display the progressively-updated _accum_frame to screen.
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glDisable(GL_DEPTH_TEST);
				glClear(GL_COLOR_BUFFER_BIT);
				_screen_shader->use();
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
				glBindVertexArray(_current_frame->VAO);
				glDrawArrays(GL_TRIANGLES, 0, 6);

				glfwSwapBuffers(_window);
				glfwPollEvents();

				// Check for abort
				if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
					aborted = true;
				}
			}
		}

		// _accum_frame now holds the complete composited image (raster underlay
		// with all ray-traced tiles painted on top). IDLE will redisplay it.
		_render_mode = RenderMode::IDLE;
	} else if (_render_mode == RenderMode::PREVIEW) {
		if (_rasterization_enabled) {
			// Rasterization preview: draw flat-shaded proxies into _raster_frame.
			glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
			glViewport(0, 0, width, height);
			glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			_rasterizer->render(_current_frame->_renderer->camera_info, (float)width / (float)height);

			// Copy _raster_frame into _accum_frame so the trailing display
			// pass picks it up unchanged.
			copyFrameBufferTexture(Window::width, Window::height,
				_raster_frame->framebuffer, _raster_frame->texture,
				_accum_frame->framebuffer, _accum_frame->texture);
		} else {
			int active_scale = (_camera_moving && preview_scale > 1) ? preview_scale : 1;

			glBindFramebuffer(GL_FRAMEBUFFER, _current_frame->framebuffer);
			_current_frame->render_kernel(true, active_scale);
			_screen_shader->use();
			glBindVertexArray(_current_frame->VAO);
			glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
			glDrawArrays(GL_TRIANGLES, 0, 6);

			if (active_scale > 1) {
				// Pixelated path: copy current frame straight into accum frame.
				// Skipping the accumulate shader keeps the pixels sharp instead
				// of smearing the coarse blocks against the previous high-res
				// accumulated image.
				copyFrameBufferTexture(Window::width, Window::height,
					_current_frame->framebuffer, _current_frame->texture,
					_accum_frame->framebuffer, _accum_frame->texture);
			} else {
				// Copy accumulated frames to another texture so that we can sample it
				copyFrameBufferTexture(Window::width, Window::height, _accum_frame->framebuffer, _accum_frame->texture, _blit_quad->framebuffer, _blit_quad->texture);

				// Composite the accumulated frames with the current one (motion blur)
				glBindFramebuffer(GL_FRAMEBUFFER, _accum_frame->framebuffer);
				glActiveTexture(GL_TEXTURE0 + 0);
				glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
				glActiveTexture(GL_TEXTURE0 + 1);
				glBindTexture(GL_TEXTURE_2D, _blit_quad->texture);
				_accum_shader->use();
				glUniform1i(glGetUniformLocation(_accum_shader->ID, "frameCount"), _frame_count);
				glUniform1i(glGetUniformLocation(_accum_shader->ID, "cameraMoving"), _camera_moving ? 1 : 0);
				glDrawArrays(GL_TRIANGLES, 0, 6);
			}
		}
	}
	// IDLE: no GPU work, just redisplay the last frame in _accum_frame

	// Render result to screen (always — in IDLE this redisplays the last frame)
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, width, height);
	_screen_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glBindVertexArray(_current_frame->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Note: glfwSwapBuffers / glfwPollEvents are called by tick() after
	// the ImGui overlay is rendered on top.
}

void Window::tick() {

	std::chrono::steady_clock::time_point this_frame = std::chrono::steady_clock::now();
	float t_diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_frame - _last_frame).count();
	_last_frame = this_frame;

	// Start a new ImGui frame (must come before tick_input so ImGui can
	// report WantCaptureMouse / WantCaptureKeyboard).
	_gui->new_frame();

	// Input
	tick_input(t_diff);

	// Build the ImGui draw list (the actual GL draw happens after tick_render
	// so the GUI overlays the rendered image).
	_gui->draw(*this);

	 // Render
	tick_render();

	// Draw the ImGui overlay on top of the final composited image (after
	// the screen-space blit but before glfwSwapBuffers).
	_gui->render();

	// Check and call events and swap buffers between frames
	glfwSwapBuffers(_window);
	glfwPollEvents();

	if (_render_mode != RenderMode::IDLE)
		_frame_count++;
}