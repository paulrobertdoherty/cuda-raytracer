#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <memory>

#include "Shader.h"
#include "ShaderManager.h"
#include "Quad.h"
#include "Rasterizer.h"
#include "RenderParams.h"
#include "RenderMode.h"
#include "raytracer/kernel.h"

class RenderLoop {
public:
	RenderLoop();

	RenderLoop(const RenderLoop&) = delete;
	RenderLoop& operator=(const RenderLoop&) = delete;

	int init_gl_resources(GLFWwindow* window,
	                      ShaderManager& shaders,
	                      const Scene& scene,
	                      const RenderParams& params);

	void destroy();

	void resize(int width, int height);
	void set_viewport(int x, int render_w, int render_h);

	void render_frame(GLFWwindow* window,
	                  RenderMode& render_mode,
	                  const RenderParams& params,
	                  int& frame_count,
	                  bool camera_moving,
	                  bool raster_enabled);

	Quad& current_frame() { return *_current_frame; }
	const Quad& current_frame() const { return *_current_frame; }
	GLuint accum_texture() const { return _accum_frame->texture; }
	KernelInfo& renderer() { return *_current_frame->_renderer; }
	const KernelInfo& renderer() const { return *_current_frame->_renderer; }
	Rasterizer& rasterizer() { return *_rasterizer; }

	int render_width() const { return _render_width; }
	int render_height() const { return _render_height; }
	int viewport_x() const { return _viewport_x; }

	ShaderManager& shaders() { return _shaders; }

private:
	int _render_width = 0;
	int _render_height = 0;
	int _viewport_x = 0;

	ShaderManager _shaders;
	Shader* _screen_shader = nullptr;
	Shader* _accum_shader = nullptr;
	std::unique_ptr<Quad> _blit_quad;
	std::unique_ptr<Quad> _accum_frame;
	std::unique_ptr<Quad> _current_frame;
	std::unique_ptr<Quad> _raster_frame;
	GLuint _raster_depth_rb = 0;
	std::unique_ptr<Rasterizer> _rasterizer;

	static void copyFrameBufferTexture(int width, int height,
	                                   int fboIn, int textureIn,
	                                   int fboOut, int textureOut);
};