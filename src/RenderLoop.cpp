#include "RenderLoop.h"

#include <iostream>

int RenderLoop::init_gl_resources(GLFWwindow* window,
                                  ShaderManager& shaders,
                                  const Scene& scene,
                                  const RenderParams& params) {
	_screen_shader = shaders.load("screen",
		"./shaders/rendertype_screen.vert", "./shaders/rendertype_screen.frag");
	_accum_shader = shaders.load("accumulate",
		"./shaders/rendertype_accumulate.vert", "./shaders/rendertype_accumulate.frag");

	if (!_screen_shader || !_screen_shader->ID || !_accum_shader || !_accum_shader->ID) {
		std::cerr << "[RenderLoop] shader load failed: screen="
		          << static_cast<const void*>(_screen_shader)
		          << " accumulate=" << static_cast<const void*>(_accum_shader) << "\n";
		return 1;
	}

	{
		int fb_w = 0;
		int fb_h = 0;
		glfwGetFramebufferSize(window, &fb_w, &fb_h);
		_render_width = fb_w;
		_render_height = fb_h;
	}

	_blit_quad = std::make_unique<Quad>(_render_width, _render_height);
	_blit_quad->make_FBO();

	_current_frame = std::make_unique<Quad>(_render_width, _render_height, params);
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

	return 0;
}

RenderLoop::~RenderLoop() {
	if (_raster_depth_rb != 0) {
		glDeleteRenderbuffers(1, &_raster_depth_rb);
	}
}

void RenderLoop::resize(int width, int height) {
	if (width == _render_width && height == _render_height) {
		return;
	}
	if (width == 0 || height == 0) {
		return;
	}

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

namespace {

// Each Quad's framebuffer keeps its texture on COLOR_ATTACHMENT0. A blit
// must leave those attachments available for the next preview draw.
void blit(const Quad& from, const Quad& to, int x, int y, int width, int height) {
	glBindFramebuffer(GL_READ_FRAMEBUFFER, from.framebuffer);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, to.framebuffer);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glBlitFramebuffer(x, y, x + width, y + height,
	                  x, y, x + width, y + height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

} // namespace

void RenderLoop::render_frame(GLFWwindow* window, RenderState& state, const RenderParams& params) {
	int window_w = 0;
	int window_h = 0;
	glfwGetFramebufferSize(window, &window_w, &window_h);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, window_w, window_h);
	glDisable(GL_DEPTH_TEST);
	glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	state.draw_frame(
		[&](const PreviewState& preview) { draw_preview(preview, params); },
		[&] { draw_final(window, params); });

	present();
	glViewport(0, 0, window_w, window_h);
}

void RenderLoop::draw_final(GLFWwindow* window, const RenderParams& params) {
	blit(*_raster_frame, *_accum_frame, 0, 0, _render_width, _render_height);

	const int tile_size = params.tile_size;
	int tiles_x = (_render_width + tile_size - 1) / tile_size;
	int tiles_y = (_render_height + tile_size - 1) / tile_size;

	for (int ty = 0; ty < tiles_y; ty++) {
		for (int tx = 0; tx < tiles_x; tx++) {
			int tile_ox = tx * tile_size;
			int tile_oy = ty * tile_size;
			int tile_w = (tile_ox + tile_size > _render_width) ? _render_width - tile_ox : tile_size;
			int tile_h = (tile_oy + tile_size > _render_height) ? _render_height - tile_oy : tile_size;

			_current_frame->render_tile(tile_ox, tile_oy, tile_w, tile_h, params.samples);

			blit(*_current_frame, *_accum_frame, tile_ox, tile_oy, tile_w, tile_h);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glClear(GL_COLOR_BUFFER_BIT);
			present();

			glfwSwapBuffers(window);
			glfwPollEvents();

			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				return;
			}
		}
	}
}

void RenderLoop::draw_preview(const PreviewState& preview, const RenderParams& params) {
	if (preview.raster_enabled) {
		glBindFramebuffer(GL_FRAMEBUFFER, _raster_frame->framebuffer);
		glViewport(0, 0, _render_width, _render_height);
		glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		_rasterizer->render(renderer().camera_info, (float)_render_width / (float)_render_height);

		blit(*_raster_frame, *_accum_frame, 0, 0, _render_width, _render_height);
	} else {
		bool fresh = preview.camera_moving || preview.frame_count <= 1;
		int active_scale = (fresh && params.preview_scale > 1) ? params.preview_scale : 1;

		glViewport(0, 0, _render_width, _render_height);

		// CUDA uploads directly into the current frame texture.
		_current_frame->render_kernel(true, active_scale);

		if (active_scale > 1) {
			blit(*_current_frame, *_accum_frame, 0, 0, _render_width, _render_height);
		} else {
			blit(*_accum_frame, *_blit_quad, 0, 0, _render_width, _render_height);

			glBindFramebuffer(GL_FRAMEBUFFER, _accum_frame->framebuffer);
			glActiveTexture(GL_TEXTURE0 + 0);
			glBindTexture(GL_TEXTURE_2D, _current_frame->texture);
			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_2D, _blit_quad->texture);
			_accum_shader->use();
			glUniform1i(glGetUniformLocation(_accum_shader->ID, "frameCount"), preview.frame_count);
			glUniform1i(glGetUniformLocation(_accum_shader->ID, "cameraMoving"), preview.camera_moving ? 1 : 0);
			glBindVertexArray(_current_frame->VAO);
			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
	}
}

void RenderLoop::present() const {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(_viewport_x, 0, _render_width, _render_height);
	_screen_shader->use();
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, _accum_frame->texture);
	glBindVertexArray(_current_frame->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}
