#include "HeadlessTexture.h"
#include "AssetRevision.h"

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

	int width;
	int height;
	int channels;
	unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
	if (!data) {
		std::cerr << "[HeadlessTexture] stbi_load failed for " << path
		          << ": " << stbi_failure_reason() << "\n";
		return false;
	}
	_width = width;
	_height = height;
	_channels = channels;

	stbi_image_free(_pixels);
	_pixels = data;
	_revision = next_asset_revision();

	std::cout << "[HeadlessTexture] Loaded " << path << " — "
	          << _width << "x" << _height << " (" << _channels << " channels)" << "\n";
	return true;
}
