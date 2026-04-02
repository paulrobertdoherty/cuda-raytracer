// ============================================================
// Debug flags — uncomment to enable the corresponding feature.
// IMPORTANT: these MUST be defined before any #include so that
// Window.h picks up the guarded member variables.
//
//   DEBUG_ACCUMULATION : frame-by-frame GL diagnostics + pixel
//                        readback from both kernel output and
//                        the accumulated buffer.
//                        Also uncomment DEBUG_ACCUMULATION in
//                        kernel.cu to get matching CUDA logs.
//
//   DISABLE_PROGRESS   : suppresses the "\rRendering: XX%"
//                        stdout output entirely so you can rule
//                        out stdout flushing as a cause.
//
//   CONVERGENCE_TEST   : after accumulation completes, does one
//                        full-SPP reference kernel call and
//                        compares its center pixel against the
//                        accumulated center pixel, printing
//                        PASS / FAIL with the numeric diff.
// ============================================================
#define DEBUG_ACCUMULATION
#define DISABLE_PROGRESS
#define CONVERGENCE_TEST

#include "Window.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

Window::Window(unsigned int width, unsigned int height, int samples, int max_depth, float fov) {
	Window::width = width;
	Window::height = height;
	Window::samples = samples;
	Window::max_depth = max_depth;
	Window::fov = fov;
	Window::_frame_count = 0;
	Window::_camera_moving = false;
	Window::_render_mode = RenderMode::PREVIEW;
	Window::_enter_was_pressed = false;
}

int Window::init_glfw() {
	// Initialize and configure GLFW
	glfwInit();
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
	
	_blit_quad = std::make_unique<Quad>(Window::width, Window::height, GL_RGBA16F);
	_blit_quad->make_FBO();

	_shader = std::make_unique<Shader>("./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_current_frame = std::make_unique<Quad>(Window::width, Window::height);
	_current_frame->cuda_init(samples, max_depth, fov);
	_current_frame->make_FBO();

	_accum_shader = std::make_unique<Shader>("./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");
	_accum_frame = std::make_unique<Quad>(Window::width, Window::height, GL_RGBA16F);
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
#ifdef CONVERGENCE_TEST
			_convergence_test_done = false;
#endif
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

// Helper: read back one RGBA float pixel from an FBO at (x, y).
// Returns false and leaves out[] unchanged if the FBO is incomplete.
static bool readback_pixel(unsigned int fbo, int x, int y, float out[4]) {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	GLenum status = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "[GL] FBO " << fbo << " incomplete, status=0x"
		          << std::hex << status << std::dec << "\n";
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		return false;
	}
	glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, out);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	return true;
}

void Window::tick_render() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if (_render_mode != RenderMode::IDLE) {
		// Always render 1 SPP per frame (fast, no freeze)
		_current_frame->render_kernel(true);

#ifdef DEBUG_ACCUMULATION
		// ----------------------------------------------------------
		// Log raw kernel output for the center pixel (1-SPP frame).
		// Note: _current_frame is GL_RGBA8, BGRA-uploaded → read as
		// RGBA gives correct channel order after GL swizzle.
		// ----------------------------------------------------------
		if (_render_mode == RenderMode::RENDER_FINAL) {
			float cur_px[4] = {0};
			if (readback_pixel(_current_frame->framebuffer, width / 2, height / 2, cur_px)) {
				std::cout << "[GL] kernel_out=(" << cur_px[0] << ", " << cur_px[1]
				          << ", " << cur_px[2] << ")\n";
			}
			GLenum gl_err = glGetError();
			if (gl_err != GL_NO_ERROR)
				std::cout << "[GL] glGetError after kernel readback: 0x"
				          << std::hex << gl_err << std::dec << "\n";
		}
#endif

		// Ping-pong: swap so _blit_quad holds the previous accumulated frame
		// and _accum_frame becomes the new render target
#ifdef DEBUG_ACCUMULATION
		if (_render_mode == RenderMode::RENDER_FINAL) {
			std::cout << "[GL] frame=" << _frame_count
			          << "  before swap: accum_tex=" << _accum_frame->texture
			          << "  blit_tex=" << _blit_quad->texture << "\n";
		}
#endif

		std::swap(_accum_frame, _blit_quad);

#ifdef DEBUG_ACCUMULATION
		if (_render_mode == RenderMode::RENDER_FINAL) {
			std::cout << "[GL]           after  swap: accum_tex=" << _accum_frame->texture
			          << "  blit_tex=" << _blit_quad->texture
			          << "  blend_weight=" << (1.0f / float(_frame_count)) << "\n";
		}
#endif

		// Composite the accumulated frames with the current one
		// PREVIEW: motion blend (no accumulation)
		// RENDER_FINAL: proper accumulation via frameCount
		bool shader_camera_moving = (_render_mode == RenderMode::PREVIEW);
		glBindFramebuffer(GL_FRAMEBUFFER, _accum_frame->framebuffer);

#ifdef DEBUG_ACCUMULATION
		if (_render_mode == RenderMode::RENDER_FINAL) {
			GLenum fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (fbo_status != GL_FRAMEBUFFER_COMPLETE)
				std::cout << "[GL] accum_frame FBO incomplete! status=0x"
				          << std::hex << fbo_status << std::dec << "\n";
		}
#endif

		glBindVertexArray(_current_frame->VAO);
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, _blit_quad->texture);
		_accum_shader->use();
		glUniform1i(glGetUniformLocation(_accum_shader->ID, "frameCount"), _frame_count);
		glUniform1i(glGetUniformLocation(_accum_shader->ID, "cameraMoving"), shader_camera_moving ? 1 : 0);
		glDrawArrays(GL_TRIANGLES, 0, 6);

#ifdef DEBUG_ACCUMULATION
		// ----------------------------------------------------------
		// Read back the accumulated center pixel after blending.
		// _accum_frame is GL_RGBA16F so values can exceed 1.0.
		// ----------------------------------------------------------
		if (_render_mode == RenderMode::RENDER_FINAL) {
			float acc_px[4] = {0};
			if (readback_pixel(_accum_frame->framebuffer, width / 2, height / 2, acc_px)) {
				std::cout << "[GL] accum_out=(" << acc_px[0] << ", " << acc_px[1]
				          << ", " << acc_px[2] << ")\n";
			}
			GLenum gl_err = glGetError();
			if (gl_err != GL_NO_ERROR)
				std::cout << "[GL] glGetError after accum readback: 0x"
				          << std::hex << gl_err << std::dec << "\n";
		}
#endif

		// Check if RENDER_FINAL accumulation is complete
		if (_render_mode == RenderMode::RENDER_FINAL) {
#ifndef DISABLE_PROGRESS
			int progress = (_frame_count * 100) / samples;
			std::cout << "\rRendering: " << progress << "%" << std::flush;
#endif
			if (_frame_count >= samples) {
#ifndef DISABLE_PROGRESS
				std::cout << "\rRendering: 100%" << std::endl;
#endif
				_render_mode = RenderMode::IDLE;

#ifdef CONVERGENCE_TEST
				// --------------------------------------------------
				// Convergence test: compare the shader-accumulated
				// center pixel against a single full-SPP reference
				// kernel call.  Both are pre-gamma linear values.
				// --------------------------------------------------
				if (!_convergence_test_done) {
					_convergence_test_done = true;

					// Read accumulated center pixel (GL_RGBA16F)
					float accum_px[4] = {0};
					readback_pixel(_accum_frame->framebuffer, width / 2, height / 2, accum_px);

					// Single full-SPP reference render into _current_frame
					_current_frame->render_kernel(false); // false → spp = samples

					float ref_px[4] = {0};
					readback_pixel(_current_frame->framebuffer, width / 2, height / 2, ref_px);

					float diff = std::sqrt(
						std::pow(accum_px[0] - ref_px[0], 2.0f) +
						std::pow(accum_px[1] - ref_px[1], 2.0f) +
						std::pow(accum_px[2] - ref_px[2], 2.0f));

					std::cout << "\n[CONVERGENCE TEST]\n";
					std::cout << "  Accumulated result : ("
					          << accum_px[0] << ", " << accum_px[1] << ", " << accum_px[2] << ")\n";
					std::cout << "  Single-call ref    : ("
					          << ref_px[0] << ", " << ref_px[1] << ", " << ref_px[2] << ")\n";
					std::cout << "  RGB distance       : " << diff << "\n";
					std::cout << "  " << (diff < 0.05f ? "PASS (within 0.05 tolerance)"
					                                   : "FAIL (> 0.05 tolerance)") << "\n\n";
				}
#endif // CONVERGENCE_TEST
			}
		}
	}

	// Render result to screen (always — in IDLE this redisplays the last frame)
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glBindVertexArray(_current_frame->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

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