#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "raytracer/kernel.h"

#include <vector>
#include <memory>

struct RenderParams;

class Quad {
public:
	unsigned int VAO;
	unsigned int VBO;

	unsigned int texture;
	unsigned int framebuffer = 0;

	int width, height;

	std::vector<float> vertices;
	Quad(int width, int height);
	Quad(int width, int height, const RenderParams& params);
	~Quad();

	void render_kernel(bool camera_moving, int pixelate = 1) const;
	void render_tile(int x, int y, int w, int h, int samples) const;
	KernelInfo& renderer() { return require_renderer(); }
	const KernelInfo& renderer() const { return require_renderer(); }
	void resize(int width, int height);
	void make_FBO();

private:
	KernelInfo& require_renderer() const;
	unsigned int PBO;
	cudaGraphicsResource_t _resource = nullptr;
	std::unique_ptr<KernelInfo> _renderer;
};
