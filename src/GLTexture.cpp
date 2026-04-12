#include "GLTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

GLTexture::~GLTexture() {
	if (_id) {
		glDeleteTextures(1, &_id);
		_id = 0;
	}
	if (_pixels) {
		stbi_image_free(_pixels);
		_pixels = nullptr;
	}
}

bool GLTexture::load(const std::string& path) {
	stbi_set_flip_vertically_on_load(1);

	unsigned char* data = stbi_load(path.c_str(), &_width, &_height, &_channels, 0);
	if (!data) {
		std::cerr << "[GLTexture] stbi_load failed for " << path
		          << ": " << stbi_failure_reason() << std::endl;
		return false;
	}

	GLenum fmt = GL_RGB;
	GLenum internal_fmt = GL_RGB8;
	if (_channels == 1) { fmt = GL_RED; internal_fmt = GL_R8; }
	else if (_channels == 3) { fmt = GL_RGB; internal_fmt = GL_RGB8; }
	else if (_channels == 4) { fmt = GL_RGBA; internal_fmt = GL_RGBA8; }

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	// Reasonable defaults.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Ensure tight packing for odd widths (RGB, etc.)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, _width, _height, 0, fmt, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	// Retain raw pixels for CUDA upload (freed in destructor).
	_pixels = data;

	std::cout << "[GLTexture] Loaded " << path << " — "
	          << _width << "x" << _height << " (" << _channels << " channels)" << std::endl;
	return true;
}

void GLTexture::bind(unsigned int unit) const {
	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, _id);
}
