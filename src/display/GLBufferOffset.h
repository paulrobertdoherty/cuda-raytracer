#pragma once

#include <cstddef>

// OpenGL accepts a byte offset encoded in a pointer when a buffer is bound.
inline const void* gl_buffer_offset(std::size_t bytes) {
	return reinterpret_cast<const void*>(bytes); // NOLINT(performance-no-int-to-ptr)
}
