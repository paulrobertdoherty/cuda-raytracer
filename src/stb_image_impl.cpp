// Single TU that defines STB_IMAGE_IMPLEMENTATION. Texture sources
// (GLTexture.cpp / HeadlessTexture.cpp) #include "stb_image.h" without
// the macro so the implementation is emitted exactly once per build.
//
// Project-wide convention: stbi_set_flip_vertically_on_load(1) is used
// by every loader, so loaded textures match GL's bottom-left origin.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
