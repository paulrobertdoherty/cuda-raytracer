#include "HeadlessTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

HeadlessTexture::~HeadlessTexture() {
	if (_pixels) {
		stbi_image_free(_pixels);
		_pixels = nullptr;
	}
}

bool HeadlessTexture::load(const std::string& path) {
	stbi_set_flip_vertically_on_load(1);

	unsigned char* data = stbi_load(path.c_str(), &_width, &_height, &_channels, 0);
	if (!data) {
		std::cerr << "[HeadlessTexture] stbi_load failed for " << path
		          << ": " << stbi_failure_reason() << "\n";
		return false;
	}

	_pixels = data;

	std::cout << "[HeadlessTexture] Loaded " << path << " — "
	          << _width << "x" << _height << " (" << _channels << " channels)" << "\n";
	return true;
}
