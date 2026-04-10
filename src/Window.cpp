#include "Window.h"

#include <iostream>
#include <iomanip>
#include <memory>

Window::Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov, int tile_size) {
	Window::width = width;
	Window::height = height;
	Window::samples = samples;
	Window::max_depth = max_depth;
	Window::fov = fov;
	Window::tile_size = tile_size;
	Window::_frame_count = 0;
	Window::_camera_moving = false;
	Window::_render_mode = RenderMode::PREVIEW;
	Window::_enter_was_pressed = false;
}

static void glfw_error_callback(int code, const char* description) {
	std::cerr << "[GLFW error " << code << "] " << description << std::endl;
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

	// Hides the cursor and captures it
	glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetWindowUserPointer(_window, reinterpret_cast<void*>(this));

	// Check if window was created
	if (_window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(_window);

	return 0;
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
	
	_blit_quad = std::make_unique<Quad>(Window::width, Window::height);
	_blit_quad->make_FBO();

	_shader = std::make_unique<Shader>("./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_current_frame = std::make_unique<Quad>(Window::width, Window::height);
	_current_frame->cuda_init(samples, max_depth, fov);
	_current_frame->make_FBO();

	_accum_shader = std::make_unique<Shader>("./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");
	_accum_frame = std::make_unique<Quad>(Window::width, Window::height);
	_accum_frame->make_FBO();

	_accum_shader->use();
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "currentFrameTex"), 0);
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "lastFrameTex"), 1);

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

void Window::destroy() {

	// Terminate CUDA allocated buffer
	_current_frame->cuda_destroy();

	// Terminate GLFW
	glfwDestroyWindow(_window);
	glfwTerminate();
}

void Window::tick_input(float t_diff) {

	_input.process_quit(_window);

	// Detect Enter key press (rising edge)
	bool enter_down = glfwGetKey(_window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if (enter_down && !_enter_was_pressed) {
		if (_render_mode == RenderMode::PREVIEW) {
			_render_mode = RenderMode::RENDER_FINAL;
			_frame_count = 1;
		} else if (_render_mode == RenderMode::IDLE) {
			_render_mode = RenderMode::PREVIEW;
		}
	}
	_enter_was_pressed = enter_down;

	// Only process camera movement in preview mode
	if (_render_mode == RenderMode::PREVIEW) {
		_input.process_camera_movement(_window, *(_current_frame->_renderer), t_diff);
		_camera_moving = _input.has_camera_moved();
		if (_camera_moving) _frame_count = 1;
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

				// Draw the full-screen quad to show the partially-updated image
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glDisable(GL_DEPTH_TEST);
				_shader->use();
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
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

		// Copy final (or partial) result to accum frame for display
		// First upload the full texture from PBO (already done tile by tile above)
		glBindFramebuffer(GL_FRAMEBUFFER, _current_frame->framebuffer);
		_shader->use();
		glBindVertexArray(_current_frame->VAO);
		glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		copyFrameBufferTexture(Window::width, Window::height, _current_frame->framebuffer, _current_frame->texture, _accum_frame->framebuffer, _accum_frame->texture);

		_render_mode = RenderMode::IDLE;
	} else if (_render_mode == RenderMode::PREVIEW) {
		// 1 SPP per frame for responsive preview
		glBindFramebuffer(GL_FRAMEBUFFER, _current_frame->framebuffer);
		_current_frame->render_kernel(true);
		_shader->use();
		glBindVertexArray(_current_frame->VAO);
		glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
		glDrawArrays(GL_TRIANGLES, 0, 6);

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
	// IDLE: no GPU work, just redisplay the last frame in _accum_frame

	// Render result to screen (always — in IDLE this redisplays the last frame)
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glBindVertexArray(_current_frame->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Check and call events and swap buffers between frames
	glfwSwapBuffers(_window);
	glfwPollEvents();
}

void Window::tick() {

	std::chrono::steady_clock::time_point this_frame = std::chrono::steady_clock::now();
	float t_diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_frame - _last_frame).count();
	_last_frame = this_frame;

	// Print FPS
	//std::cout << "\r" << std::fixed << std::setprecision(2) << 1000.0 / t_diff << " fps";

	// Input
	tick_input(t_diff);

	 // Render
	tick_render();

	if (_render_mode != RenderMode::IDLE)
		_frame_count++;
}