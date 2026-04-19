#pragma once

#include <string>

// Minimal texture loader for headless mode — loads image pixels via stb_image
// without requiring an OpenGL context. Provides the same interface that
// kernel.cu's rebuild_world needs (raw_pixels, width, height, channels).
class HeadlessTexture {
public:
	HeadlessTexture() = default;
	~HeadlessTexture();

	HeadlessTexture(const HeadlessTexture&) = delete;
	HeadlessTexture& operator=(const HeadlessTexture&) = delete;

	bool load(const std::string& path);

	int width()  const { return _width; }
	int height() const { return _height; }
	int channels() const { return _channels; }
	const unsigned char* raw_pixels() const { return _pixels; }

private:
	int            _width    = 0;
	int            _height   = 0;
	int            _channels = 0;
	unsigned char* _pixels   = nullptr;
};
