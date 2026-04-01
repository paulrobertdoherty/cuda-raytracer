# CUDA Ray Tracer

A real-time GPU-accelerated path tracer written in C++ and CUDA, with OpenGL for display.

This is a fork of [Ancientkingg/cuda-raytracer](https://github.com/Ancientkingg/cuda-raytracer) and an experiment in using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to extend a ray tracer. The original project implemented a real-time CUDA path tracer with movable camera, temporal accumulation, BVH acceleration, and textures. This fork uses Claude Code to add new rendering features, primitives, and performance improvements on top of that foundation.

## What Claude Code added

- **Emissive materials** for area lights (previously only sky gradient illumination)
- **Rectangle and triangle primitives** (enabling flat area lights and mesh building blocks)
- **Adaptive samples-per-pixel** that scales SPP dynamically to maintain a target frame rate
- **Temporal reprojection** so camera movement blends smoothly instead of resetting to full noise
- **Command-line parameters** for width, height, SPP, max depth, and FOV
- **CUDA 13+ compatibility** fixes (BVH sort, MSVC conforming preprocessor, `CUDA_ARCHITECTURES native`)

## Building

Requires:
- CMake 3.24+
- CUDA Toolkit (12+ recommended, 13+ tested)
- A C++17 compiler (MSVC on Windows, GCC/Clang on Linux)
- An NVIDIA GPU

```bash
cmake --preset x64-release
cmake --build out/build/x64-release
```

## Running

```bash
cd out/build/x64-release
./cuda-raytracer --width 1280 --height 720 --samples 8 --depth 50 --fov 75
```

All parameters are optional (defaults: 800x600, 3 SPP, depth 50, FOV 90). The `--samples` value is the maximum SPP; the adaptive controller scales it down automatically if the GPU can't sustain 30 fps.

### Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move horizontally |
| Space / Ctrl | Move up / down |
| Mouse | Look around |
| Q / E | Roll camera |
| Esc | Quit |

## Architecture

```
                         +-----------+
                         |  main.cpp |  Parse CLI args, create Window
                         +-----+-----+
                               |
                         +-----v-----+
                         | Window    |  GLFW window, main loop, adaptive SPP
                         +-----+-----+
                               |
                 +-------------+-------------+
                 |                           |
           +-----v-----+             +------v------+
           |   Input    |             | tick_render |  4-stage OpenGL pipeline
           +-----+------+             +------+------+
                 |                           |
     camera movement                +--------+--------+--------+
     & velocity decay               |        |        |        |
                 |               Render   Copy to  Accumulate  Display
           +-----v------+       current   blit     blend to    to
           | KernelInfo  |       frame    quad     accum FBO   screen
           | .set_camera |         |
           +-------------+   +----v----+
                             | raytrace |  CUDA kernel (per-pixel)
                             +----+-----+
                                  |
                    +-------------+-------------+
                    |             |              |
              +-----v---+  +-----v------+  +----v-------+
              | Camera  |  | FrameBuffer|  | World      |
              | get_ray |  | .color()   |  | .hit()     |
              +---------+  +-----+------+  +-----+------+
                                 |               |
                           recursive         +---v---+
                           bounce loop       |  BVH  |  AABB tree traversal
                                 |           +---+---+
                    +------------+---+           |
                    |            |   |      +----+----+----+----+
               on hit:      on miss:  max  |    |    |    |    |
               emit +       sky      depth | Sphere Rect Triangle ...
               scatter      gradient       +----+----+----+----+
                    |                            |
              +-----v--------+             +-----v--------+
              |  Materials   |             |  Hittable    |
              | Lambertian   |             | .hit()       |
              | Metal        |             | .bounding_box|
              | Dielectric   |             +--------------+
              | Emissive     |
              +--------------+

Accumulation shader (GLSL):
  Static camera  -> progressive blend toward converged image (up to 500 frames)
  Moving camera  -> exponential blend (20% new / 80% old) to reduce ghosting
```

### Source layout

```
src/
  main.cpp                  Entry point, CLI argument parsing
  Window.cpp/.h             GLFW window, main loop, adaptive SPP, temporal blend
  Input.cpp/.h              Keyboard/mouse input, camera velocity
  Quad.cpp/.h               OpenGL quad with PBO for CUDA-GL interop
  Shader.h                  OpenGL shader loader
  cuda_errors.cpp/.h        CUDA error checking helpers

  raytracer/
    kernel.cu/.h            CUDA kernels: raytrace, create_world, set_device_camera
    Camera.h                Camera with position, rotation, FOV
    Ray.h                   Ray struct (origin + direction)
    FrameBuffer.h           Per-pixel color computation (recursive bounce loop)
    World.h                 Scene container with optional BVH root
    BVHNode.h               Bounding volume hierarchy (AABB tree)
    AABB.h                  Axis-aligned bounding box (slab intersection)
    Hittable.h              Base class + HitRecord
    Sphere.h                Sphere primitive
    Rectangle.h             Quad primitive (corner + two edge vectors)
    Triangle.h              Triangle primitive (Moller-Trumbore intersection)
    Material.h              Lambertian, Metal, Dielectric, Emissive
    Texture.h               SolidColor, CheckerTexture

  shaders/
    rendertype_screen.*     Pass-through vertex/fragment shader
    rendertype_accumulate.* Temporal accumulation with motion-aware blending
```

## Original project

The original ray tracer by [Ancientkingg](https://github.com/Ancientkingg) implements the core rendering pipeline following the [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) series adapted to CUDA, with real-time display via OpenGL interop, a movable camera, BVH acceleration, and temporal frame accumulation. See the [original repository](https://github.com/Ancientkingg/cuda-raytracer) for the full development walkthrough.
