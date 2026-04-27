#include "RenderLoop.h"

RenderLoop::RenderLoop() {}

int RenderLoop::init_gl_resources(GLFWwindow* window,
                                  ShaderManager& shaders,
                                  const Scene& scene,
                                  const RenderParams& params) {
	_screen_shader = shaders.load("screen",
		"./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_accum_shader = shaders.load("accumulate",
		"./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");

	{
		int fb_w = 0, fb_h = 0;
		glfwGetFramebufferSize(window, &fb_w, &fb_h);
		_render_width = fb_w;
		_render_height = fb_h;
	}

	_blit_quad = std::make_unique<Quad>(_render_width, _render_height);
	_blit_quad->make_FBO();

	_current_frame = std::make_unique<Quad>(_render_width, _render_height);
	_current_frame->cuda_init(params.samples, params.max_depth, params.fov);
	_current_frame->make_FBO();

	_accum_frame = std::make_unique<Quad>(_render_width, _render_height);
	_accum_frame->make_FBO();

	_accum_shader->use();
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "currentFrameTex"), 0);
	glUniform1i(glGetUniformLocation(_accum_shader->ID, "lastFrameTex"), 1);

	_raster_frame = std::make_unique<Quad>(_render_width, _render_height);
	_raster_frame->make_FBO();

	glGenRenderbuffers(1, &_raster_depth_rb);
	glBindRenderbuffer(GL_RENDERBUFFER, _raster_depth_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, _render_width, _render_height);
	glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _raster_depth_rb);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	_rasterizer = std::make_unique<Rasterizer>(shaders, scene);

	_current_frame->_renderer->rebuild_world(scene);

	return 0;
}

void RenderLoop::destroy() {
	_current_frame->cuda_destroy();
}

void RenderLoop::resize(int width, int height) {
	if (width == _render_width && height == _render_height) return;
	if (width == 0 || height == 0) return;

	_render_width = width;
	_render_height = height;

	_current_frame->resize(width, height);
	_accum_frame->resize(width, height);
	_blit_quad->resize(width, height);
	_raster_frame->resize(width, height);

	if (_raster_depth_rb != 0) {
		glBindRenderbuffer(GL_RENDERBUFFER, _raster_depth_rb);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
		glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _raster_depth_rb);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void RenderLoop::set_viewport(int x, int render_w, int render_h) {
	_viewport_x = x;
	resize(render_w, render_h);
}

void RenderLoop::copyFrameBufferTexture(int width, int height,
                                        int fboIn, int textureIn,
                                        int fboOut, int textureOut) {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fboIn);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIn, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboOut);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureOut, 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT1);

	glBlitFramebuffer(0, 0, width, height,
		0, 0, width, height,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
}

void RenderLoop::render_frame(GLFWwindow* window,
                              RenderMode& render_mode,
                              const RenderParams& params,
                              int& frame_count,
                              bool camera_moving,
                              bool raster_enabled) {
	int window_w = 0, window_h = 0;
	glfwGetFramebufferSize(window, &window_w, &window_h);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, window_w, window_h);
	glDisable(GL_DEPTH_TEST);
	glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if (render_mode == RenderMode::RENDER_FINAL) {
		copyFrameBufferTexture(_render_width, _render_height,
			_raster_frame->framebuffer, _raster_frame->texture,
			_accum_frame->framebuffer, _accum_frame->texture);

		bool aborted = false;
		const int tile_size = params.tile_size;
		int tiles_x = (_render_width + tile_size - 1) / tile_size;
		int tiles_y = (_render_height + tile_size - 1) / tile_size;

		for (int ty = 0; ty < tiles_y && !aborted; ty++) {
			for (int tx = 0; tx < tiles_x && !aborted; tx++) {
				int tile_ox = tx * tile_size;
				int tile_oy = ty * tile_size;
				int tile_w = (tile_ox + tile_size > _render_width) ? _render_width - tile_ox : tile_size;
				int tile_h = (tile_oy + tile_size > _render_height) ? _render_height - tile_oy : tile_size;

				_current_frame->_renderer->map_pbo();
				_current_frame->_renderer->render_tile(tile_ox, tile_oy, tile_w, tile_h, params.samples);
				_current_frame->_renderer->unmap_pbo();

				_current_frame->upload_tile(tile_ox, tile_oy, tile_w, tile_h);

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

				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glViewport(_viewport_x, 0, _render_width, _render_height);
				glDisable(GL_DEPTH_TEST);
				glClear(GL_COLOR_BUFFER_BIT);
				_screen_shader->use();
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
				glBindVertexArray(_current_frame->VAO);
				glDrawArrays(GL_TRIANGLES, 0, 6);

				glfwSwapBuffers(window);
				glfwPollEvents();

				if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
					aborted = true;
				}
			}
		}

		render_mode = RenderMode::IDLE;
	} else if (render_mode == RenderMode::PREVIEW) {
		if (raster_enabled) {
			glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
			glViewport(0, 0, _render_width, _render_height);
			glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			_rasterizer->render(_current_frame->_renderer->camera_info, (float)_render_width / (float)_render_height);

			copyFrameBufferTexture(_render_width, _render_height,
				_raster_frame->framebuffer, _raster_frame->texture,
				_accum_frame->framebuffer, _accum_frame->texture);
		} else {
			bool fresh = camera_moving || frame_count <= 1;
			int active_scale = (fresh && params.preview_scale > 1) ? params.preview_scale : 1;

			glViewport(0, 0, _render_width, _render_height);

			glBindFramebuffer(GL_FRAMEBUFFER, _current_frame->framebuffer);
			_current_frame->render_kernel(true, active_scale);
			_screen_shader->use();
			glBindVertexArray(_current_frame->VAO);
			glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
			glDrawArrays(GL_TRIANGLES, 0, 6);

			if (active_scale > 1) {
				copyFrameBufferTexture(_render_width, _render_height,
					_current_frame->framebuffer, _current_frame->texture,
					_accum_frame->framebuffer, _accum_frame->texture);
			} else {
				copyFrameBufferTexture(_render_width, _render_height,
					_accum_frame->framebuffer, _accum_frame->texture,
					_blit_quad->framebuffer, _blit_quad->texture);

				glBindFramebuffer(GL_FRAMEBUFFER, _accum_frame->framebuffer);
				glActiveTexture(GL_TEXTURE0 + 0);
				glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
				glActiveTexture(GL_TEXTURE0 + 1);
				glBindTexture(GL_TEXTURE_2D, _blit_quad->texture);
				_accum_shader->use();
				glUniform1i(glGetUniformLocation(_accum_shader->ID, "frameCount"), frame_count);
				glUniform1i(glGetUniformLocation(_accum_shader->ID, "cameraMoving"), camera_moving ? 1 : 0);
				glDrawArrays(GL_TRIANGLES, 0, 6);
			}
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(_viewport_x, 0, _render_width, _render_height);
	_screen_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glBindVertexArray(_current_frame->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glViewport(0, 0, window_w, window_h);
}