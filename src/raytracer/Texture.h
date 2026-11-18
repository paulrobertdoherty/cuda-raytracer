
#include "device_launch_parameters.h"
#include "DeviceAssetViews.h"
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>


class Texture {
public:
	// NVCC infers a host execution space for an explicitly defaulted destructor.
	__device__ virtual ~Texture() {}
	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const = 0;
};

class SolidColor : public Texture {
public:
	__device__ SolidColor() {}
	__device__ SolidColor(glm::vec3 c) : color_value(c) {}
	__device__ SolidColor(float red, float green, float blue) : SolidColor(glm::vec3(red, green, blue)) {}


	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override {
		return color_value;
	}

private:
	glm::vec3 color_value;
};

class CheckerTexture : public Texture {
public:
	__device__ CheckerTexture(glm::vec3 even_color, glm::vec3 odd_color)
		: even(even_color), odd(odd_color) {}

	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override {
		float sines = sinf(10.0f * p.x) * sinf(10.0f * p.y) * sinf(10.0f * p.z);
		if (sines < 0.0f) {
			return odd;
		}
		else {
			return even;
		}
	}

private:
	glm::vec3 even;
	glm::vec3 odd;
};

// The image pixels belong to KernelInfo's texture cache.
class ImageTexture : public Texture {
public:
	__device__ explicit ImageTexture(DeviceImageView image) : image(image) {}

	__device__ virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override {
		// __restrict__ alias lets PTXAS route the byte loads through the
		// read-only cache; the buffer is never aliased by other state on
		// this code path.
		const unsigned char* __restrict__ pix = image.pixels;
		float uf = __saturatef(u);
		float vf = __saturatef(v);
		int i = (int)(uf * (image.width - 1));
		int j = (int)(vf * (image.height - 1));
		int idx = (j * image.width + i) * image.channels;
		float r = pix[idx + 0] / 255.0f;
		float g = (image.channels >= 2) ? pix[idx + 1] / 255.0f : r;
		float b = (image.channels >= 3) ? pix[idx + 2] / 255.0f : g;
		return glm::vec3(r, g, b);
	}

private:
	DeviceImageView image;
};
