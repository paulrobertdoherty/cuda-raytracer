#include "Application.h"
#include "Scene.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

namespace {
bool parse_vec3(const char* flag, const char* str, glm::vec3& out) {
	float x = 0;
	float y = 0;
	float z = 0;
	if (sscanf(str, "%f,%f,%f", &x, &y, &z) != 3) {
		std::cerr << "Error: " << flag
		          << " expected x,y,z (three comma-separated floats), got '"
		          << str << "'\n";
		return false;
	}
	out = glm::vec3(x, y, z);
	return true;
}
} // namespace

void print_backend_usage() {
	std::cout << "  --output <path>    Output file (required). Writes PNG if path ends in .png, else raw RGBA.\n"
		<< "  --seed <int>       Philox RNG seed (default: 1984). Same seed + same scene = same image.\n"
		<< "  --camera-pos <x,y,z>   Camera position (default: 0,1,4)\n"
		<< "  --camera-target <x,y,z> Camera look-at target (default: 0,0,0)\n";
}

OptionResult parse_backend_option(const std::string& arg, const char* value, ApplicationOptions& options) {
	if (!value) {
		return OptionResult::Unknown;
	}
	if (arg == "--output") {
		options.output_path = value;
	} else if (arg == "--seed") {
		if (!parse_int_arg("--seed", value, options.seed)) {
			return OptionResult::Invalid;
		}
	} else if (arg == "--camera-pos" || arg == "--camera-target") {
		auto& position = arg == "--camera-pos" ? options.camera_pos : options.camera_target;
		if (!parse_vec3(arg.c_str(), value, position)) {
			return OptionResult::Invalid;
		}
	} else {
		return OptionResult::Unknown;
	}
	return OptionResult::Parsed;
}

int run_application(const ApplicationOptions& options, const char* program_name) {
	if (options.output_path.empty()) {
		std::cerr << "Error: --output <path> is required in headless mode\n";
		print_usage(program_name);
		return 1;
	}

	std::cout << "Headless render: " << options.width << "x" << options.height
	          << ", " << options.params.samples << " spp, depth " << options.params.max_depth
	          << ", seed " << options.seed << "\n";

	// Create the scene
	Scene scene;
	if (!options.obj_path.empty()) {
		int idx = scene.add_obj_from_file(options.obj_path, options.texture_path,
		                                   glm::vec3(0.0f, 0.5f, -1.0f), 0.5f);
		if (idx < 0) {
			std::cerr << "Failed to load OBJ: " << options.obj_path << "\n";
			return 1;
		}
	}

	// Create headless renderer
	KernelInfo renderer(options.width, options.height, options.params.samples, options.params.max_depth, options.params.fov, options.seed);

	// Set camera. Reject coincident pos/target — normalize(0) is NaN and would
	// poison every camera ray.
	glm::vec3 view = options.camera_target - options.camera_pos;
	float view_len = glm::length(view);
	if (view_len < 1e-6f) {
		std::cerr << "Error: --camera-pos and --camera-target are coincident; "
		          << "give the camera a direction to look in.\n";
		return 1;
	}
	// Camera places the image plane along its negative Z axis.
	const glm::vec3 camera_z = -view / view_len;
	renderer.set_camera(options.camera_pos, camera_z, glm::vec3(0.0f, 1.0f, 0.0f));

	// Build the CUDA world from the scene
	renderer.rebuild_world(scene);

	// Render
	std::cout << "Rendering..." << "\n";
	std::vector<uint8_t> pixels;
	renderer.render_to_buffer(pixels);

	// Write PNG if path ends in ".png" (case-insensitive), else raw RGBA bytes.
	std::string lowered = options.output_path;
	std::transform(lowered.begin(), lowered.end(), lowered.begin(),
	               [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
	bool want_png = lowered.size() >= 4 &&
	                lowered.compare(lowered.size() - 4, 4, ".png") == 0;

	if (want_png) {
		// stride = width * 4 bytes (RGBA8). render_to_buffer already produced
		// top-row-first pixels (Y-flipped from GL convention), so no further
		// flipping needed.
		int rc = stbi_write_png(options.output_path.c_str(), options.width, options.height, 4,
		                        pixels.data(), options.width * 4);
		if (rc == 0) {
			std::cerr << "Failed to write PNG: " << options.output_path << "\n";
			return 1;
		}
		std::cout << "Saved " << options.width << "x" << options.height
		          << " PNG to: " << options.output_path << "\n";
	} else {
		std::ofstream out(options.output_path, std::ios::binary);
		if (!out) {
			std::cerr << "Failed to open: " << options.output_path << "\n";
			return 1;
		}
		out.write(reinterpret_cast<const char*>(pixels.data()),
		          static_cast<std::streamsize>(pixels.size()));
		if (!out) {
			std::cerr << "Failed to write: " << options.output_path << "\n";
			return 1;
		}
		std::cout << "Saved " << options.width << "x" << options.height
		          << " raw RGBA to: " << options.output_path << "\n";
	}

	return 0;
}
