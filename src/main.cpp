
#include "Window.h"

#include <iostream>
#include <string>
#include <cstdlib>

void print_usage(const char* program_name) {
	std::cout << "Usage: " << program_name << " [options]\n"
		<< "Options:\n"
		<< "  --width <int>     Window width (default: 800)\n"
		<< "  --height <int>    Window height (default: 600)\n"
		<< "  --samples <int>   Samples per pixel (default: 3)\n"
		<< "  --depth <int>     Max ray bounce depth (default: 50)\n"
		<< "  --fov <float>     Camera field of view in degrees (default: 90.0)\n"
		<< "  --help            Show this help message\n";
}

int main(int argc, char* argv[])
{
	int width = 800;
	int height = 600;
	int samples = 3;
	int max_depth = 50;
	float fov = 90.0f;

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];

		if (arg == "--help") {
			print_usage(argv[0]);
			return 0;
		} else if (arg == "--width" && i + 1 < argc) {
			width = std::atoi(argv[++i]);
		} else if (arg == "--height" && i + 1 < argc) {
			height = std::atoi(argv[++i]);
		} else if (arg == "--samples" && i + 1 < argc) {
			samples = std::atoi(argv[++i]);
		} else if (arg == "--depth" && i + 1 < argc) {
			max_depth = std::atoi(argv[++i]);
		} else if (arg == "--fov" && i + 1 < argc) {
			fov = std::atof(argv[++i]);
		} else {
			std::cerr << "Unknown option: " << arg << "\n";
			print_usage(argv[0]);
			return 1;
		}
	}

	Window window(width, height, samples, max_depth, fov);

	return window.init();
}
