#include "Application.h"
#include "Window.h"

#include <cstdlib>

void print_backend_usage() {}

OptionResult parse_backend_option(const std::string&, const char*, ApplicationOptions&) {
	return OptionResult::Unknown;
}

int run_application(const ApplicationOptions& options, const char*) {
#if defined(__linux__)
	// Select the NVIDIA GPU before GLFW creates a context for CUDA interop.
	setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 0);
	setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 0);
#endif
	Window window(options.width, options.height, options.params, options.obj_path, options.texture_path);
	return window.init();
}
