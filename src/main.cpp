
#ifndef HEADLESS_BUILD
#include "Window.h"
#endif
#include "raytracer/kernel.h"
#include "Scene.h"

#include <iostream>
#include <string>
#include <cstdlib>
#ifdef HEADLESS_BUILD
#include <fstream>
#endif

#if defined(__linux__) && !defined(HEADLESS_BUILD)
// On hybrid-GPU laptops, force GLX/EGL to use the NVIDIA dGPU so that the
// OpenGL context lives on the same device as CUDA. Without this, GLFW picks
// the integrated GPU via Mesa and cudaGraphicsGLRegisterBuffer fails with
// cudaErrorInvalidGraphicsContext (219). Must be set before glfwInit().
__attribute__((constructor))
static void force_nvidia_prime_offload() {
	setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 0);
	setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 0);
}
#endif

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
		<< "  --texture <path>   Diffuse texture (PNG/JPG/TGA) for the loaded mesh\n"
#ifdef HEADLESS_BUILD
		<< "  --output <path>    Output raw RGBA file path (required in headless mode)\n"
		<< "  --camera-pos <x,y,z>   Camera position (default: 0,1,4)\n"
		<< "  --camera-target <x,y,z> Camera look-at target (default: 0,0,0)\n"
#endif
		<< "  --help             Show this help message\n";
}

#ifdef HEADLESS_BUILD
static glm::vec3 parse_vec3(const char* str) {
	float x = 0, y = 0, z = 0;
	sscanf(str, "%f,%f,%f", &x, &y, &z);
	return glm::vec3(x, y, z);
}
#endif

int main(int argc, char* argv[])
{
	int width = 800;
	int height = 600;
	int samples = 3;
	int max_depth = 50;
	float fov = 90.0f;
	int tile_size = DEFAULT_TILE_SIZE;
	int preview_scale = 1;
	std::string obj_path;
	std::string texture_path;
#ifdef HEADLESS_BUILD
	std::string output_path;
	glm::vec3 camera_pos(0.0f, 1.0f, 4.0f);
	glm::vec3 camera_target(0.0f, 0.0f, 0.0f);
#endif

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
		} else if (arg == "--tile-size" && i + 1 < argc) {
			tile_size = std::atoi(argv[++i]);
		} else if (arg == "--preview-scale" && i + 1 < argc) {
			preview_scale = std::atoi(argv[++i]);
			if (preview_scale < 1) preview_scale = 1;
		} else if (arg == "--obj" && i + 1 < argc) {
			obj_path = argv[++i];
		} else if (arg == "--texture" && i + 1 < argc) {
			texture_path = argv[++i];
#ifdef HEADLESS_BUILD
		} else if (arg == "--output" && i + 1 < argc) {
			output_path = argv[++i];
		} else if (arg == "--camera-pos" && i + 1 < argc) {
			camera_pos = parse_vec3(argv[++i]);
		} else if (arg == "--camera-target" && i + 1 < argc) {
			camera_target = parse_vec3(argv[++i]);
#endif
		} else {
			std::cerr << "Unknown option: " << arg << "\n";
			print_usage(argv[0]);
			return 1;
		}
	}

#ifdef HEADLESS_BUILD
	if (output_path.empty()) {
		std::cerr << "Error: --output <path> is required in headless mode\n";
		print_usage(argv[0]);
		return 1;
	}

	std::cout << "Headless render: " << width << "x" << height
	          << ", " << samples << " spp, depth " << max_depth << std::endl;

	// Create the scene
	Scene scene;
	if (!obj_path.empty()) {
		int idx = scene.add_obj_from_file(obj_path, texture_path,
		                                   glm::vec3(0.0f, 0.5f, -1.0f), 0.5f);
		if (idx < 0) {
			std::cerr << "Failed to load OBJ: " << obj_path << std::endl;
			return 1;
		}
	}

	// Create headless renderer
	KernelInfo renderer(width, height, samples, max_depth, fov);

	// Set camera
	glm::vec3 forward = glm::normalize(camera_target - camera_pos);
	renderer.set_camera(camera_pos, forward, glm::vec3(0.0f, 1.0f, 0.0f));

	// Build the CUDA world from the scene
	renderer.rebuild_world(scene);

	// Render
	std::cout << "Rendering..." << std::endl;
	std::vector<uint8_t> pixels;
	renderer.render_to_buffer(pixels);

	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		std::cerr << "Failed to open: " << output_path << std::endl;
		return 1;
	}
	out.write(reinterpret_cast<const char*>(pixels.data()),
	          static_cast<std::streamsize>(pixels.size()));
	if (!out) {
		std::cerr << "Failed to write: " << output_path << std::endl;
		return 1;
	}
	std::cout << "Saved " << width << "x" << height
	          << " RGBA to: " << output_path << std::endl;

	return 0;
#else
	Window window(width, height, samples, max_depth, fov, tile_size, preview_scale,
		obj_path, texture_path);

	return window.init();
#endif
}
