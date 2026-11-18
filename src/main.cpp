#include "Application.h"
#include "cuda_errors.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>

void print_usage(const char* program_name) {
	std::cout << "Usage: " << program_name << " [options]\n"
		<< "Options:\n"
		<< "  --width <int>      Window width (default: 800)\n"
		<< "  --height <int>     Window height (default: 600)\n"
		<< "  --samples <int>    Samples per pixel (default: 3)\n"
		<< "  --depth <int>      Max ray bounce depth (default: 50)\n"
		<< "  --fov <float>      Camera field of view in degrees (default: 90.0)\n"
		<< "  --tile-size <int>  Tile size for progressive rendering (default: 64)\n"
		<< "  --preview-scale <int>  Pixelation factor used while moving the camera in preview (default: 1)\n"
		<< "  --obj <path>       Load a wavefront .obj mesh into the scene\n"
		<< "  --texture <path>   Diffuse texture (PNG/JPG/TGA) for the loaded mesh\n";
	print_backend_usage();
	std::cout << "  --help             Show this help message\n";
}

bool parse_int_arg(const char* flag, const char* str, int& out) {
	char* end = nullptr;
	errno = 0;
	long v = std::strtol(str, &end, 10);
	if (errno != 0 || end == str || *end != '\0' || v < INT_MIN || v > INT_MAX) {
		std::cerr << "Error: " << flag << " expected an integer, got '" << str << "'\n";
		return false;
	}
	out = static_cast<int>(v);
	return true;
}

namespace {
bool parse_float_arg(const char* flag, const char* str, float& out) {
	char* end = nullptr;
	errno = 0;
	double v = std::strtod(str, &end);
	if (errno != 0 || end == str || *end != '\0') {
		std::cerr << "Error: " << flag << " expected a number, got '" << str << "'\n";
		return false;
	}
	out = static_cast<float>(v);
	return true;
}

int run(int argc, char* argv[]) {
	ApplicationOptions options;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];

		if (arg == "--help") {
			print_usage(argv[0]);
			return 0;
		} else if (arg == "--width" && i + 1 < argc) {
			if (!parse_int_arg("--width", argv[++i], options.width)) {
				return 1;
			}
		} else if (arg == "--height" && i + 1 < argc) {
			if (!parse_int_arg("--height", argv[++i], options.height)) {
				return 1;
			}
		} else if (arg == "--samples" && i + 1 < argc) {
			if (!parse_int_arg("--samples", argv[++i], options.params.samples)) {
				return 1;
			}
		} else if (arg == "--depth" && i + 1 < argc) {
			if (!parse_int_arg("--depth", argv[++i], options.params.max_depth)) {
				return 1;
			}
		} else if (arg == "--fov" && i + 1 < argc) {
			if (!parse_float_arg("--fov", argv[++i], options.params.fov)) {
				return 1;
			}
		} else if (arg == "--tile-size" && i + 1 < argc) {
			if (!parse_int_arg("--tile-size", argv[++i], options.params.tile_size)) {
				return 1;
			}
		} else if (arg == "--preview-scale" && i + 1 < argc) {
			if (!parse_int_arg("--preview-scale", argv[++i], options.params.preview_scale)) {
				return 1;
			}
			if (options.params.preview_scale < 1) {
				options.params.preview_scale = 1;
			}
		} else if (arg == "--obj" && i + 1 < argc) {
			options.obj_path = argv[++i];
		} else if (arg == "--texture" && i + 1 < argc) {
			options.texture_path = argv[++i];
		} else {
			auto result = parse_backend_option(arg, i + 1 < argc ? argv[i + 1] : nullptr, options);
			if (result == OptionResult::Unknown) {
				std::cerr << "Unknown option: " << arg << "\n";
				print_usage(argv[0]);
				return 1;
			}
			if (result == OptionResult::Invalid) {
				return 1;
			}
			++i;
		}
	}
	return run_application(options, argv[0]);
}
} // namespace

int main(int argc, char* argv[]) {
	try {
		return run(argc, argv);
	} catch (const CudaError& e) {
		std::cerr << "Fatal: " << e.what() << "\n";
		return 2;
	} catch (const std::exception& e) {
		std::cerr << "Fatal: " << e.what() << "\n";
		return 1;
	}
}
