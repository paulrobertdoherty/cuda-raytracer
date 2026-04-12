
#include "device_launch_parameters.h"
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>


class Texture {
public:
	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const = 0;
};

class SolidColor : public Texture {
public:
	__device__ SolidColor() {}
	__device__ SolidColor(glm::vec3 c) : color_value(c) {}
	__device__ SolidColor(float red, float green, float blue) : SolidColor(glm::vec3(red, green, blue)) {}


	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const override {
		return color_value;
	}

private:
	glm::vec3 color_value;
};

class CheckerTexture : public Texture {
public:
	__device__ CheckerTexture() {}
	__device__ CheckerTexture(Texture* even_tex, Texture* odd_tex) : even(even_tex), odd(odd_tex) {}
	__device__ CheckerTexture(glm::vec3 even_color, glm::vec3 odd_color) : even(new SolidColor(even_color)), odd(new SolidColor(odd_color)) {}

	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const override {
		double sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

	Texture* even;
	Texture* odd;

	__device__ ~CheckerTexture() {
		delete even;
		delete odd;
	}
};

// Device-side image texture. The pixel buffer is owned by KernelInfo and must
// NOT be freed in this destructor — only the ImageTexture object itself is
// deleted when the owning Material is destroyed.
class ImageTexture : public Texture {
public:
	const unsigned char* d_pixels; // raw device pointer, NOT owned
	int width;
	int height;
	int channels;

	__device__ ImageTexture(const unsigned char* pixels, int w, int h, int ch)
		: d_pixels(pixels), width(w), height(h), channels(ch) {}

	__device__ virtual glm::vec3 value(double u, double v, const glm::vec3& p) const override {
		float uf = fminf(fmaxf((float)u, 0.0f), 1.0f);
		float vf = fminf(fmaxf((float)v, 0.0f), 1.0f);
		int i = (int)(uf * (width  - 1));
		int j = (int)(vf * (height - 1));
		int idx = (j * width + i) * channels;
		float r = d_pixels[idx + 0] / 255.0f;
		float g = d_pixels[idx + 1] / 255.0f;
		float b = (channels >= 3) ? d_pixels[idx + 2] / 255.0f : g;
		return glm::vec3(r, g, b);
	}
	// Does NOT free d_pixels — KernelInfo owns the device buffer.
};