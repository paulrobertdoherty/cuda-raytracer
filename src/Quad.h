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

	int width, height;

	std::vector<float> vertices;
	Quad(int width, int height);

	void cuda_init(int samples, int max_depth, float fov);
	void cuda_destroy();
	void render_kernel(bool camera_moving, int pixelate = 1);
	void upload_tile(int x, int y, int w, int h);
	void resize(int width, int height);
	void make_FBO();
};