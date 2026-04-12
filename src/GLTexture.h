#pragma once

#include <glad/glad.h>

#include <memory>
#include <string>

// Minimal 2D GL texture loaded from an image file via stb_image. RAII; the
// underlying GL texture is released on destruction.
class GLTexture {
public:
	GLTexture() = default;
	~GLTexture();

	GLTexture(const GLTexture&) = delete;
	GLTexture& operator=(const GLTexture&) = delete;

	// Load an image from disk. Returns true on success.
	bool load(const std::string& path);

	// Bind to the given texture unit (e.g. 0 for GL_TEXTURE0).
	void bind(unsigned int unit) const;

	GLuint id() const { return _id; }
	int width() const { return _width; }
	int height() const { return _height; }

private:
	GLuint _id = 0;
	int _width = 0;
	int _height = 0;
};
