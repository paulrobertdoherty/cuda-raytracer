#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "raytracer/kernel.h"

#include <vector>
#include <memory>

class Quad {
public:
	unsigned int VAO;
	unsigned int VBO;

	unsigned int texture;
	unsigned int PBO; // pixel buffer object
	std::unique_ptr<KernelInfo> _renderer;
	unsigned int framebuffer;

	cudaGraphicsResource_t CGR;

	unsigned int width, height;
	GLenum internal_format;

	std::vector<float> vertices;
	Quad(unsigned int width, unsigned int height, GLenum internal_format = GL_RGBA8);

	void cuda_init(int samples, int max_depth, float fov);
	void cuda_destroy();
	void render_kernel(bool camera_moving);
	void resize(unsigned int width, unsigned int height);
	void make_FBO();
};