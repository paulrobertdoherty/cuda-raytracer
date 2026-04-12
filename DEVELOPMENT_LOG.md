# Development Log: cuda-raytracer

This document chronicles the collaborative development of this CUDA path tracer by Paul Doherty and Claude (Anthropic), from March 29 to April 12, 2026. The project is a fork of [Ancientkingg's cuda-raytracer](https://github.com/Ancientkingg/cuda-raytracer), originally a real-time CUDA ray tracer with BVH acceleration, interactive camera, and basic materials. Over roughly two weeks, it was transformed into a full-featured path tracer with OBJ loading, PBR textures, tiled rendering, and a hybrid rasterization/ray-tracing preview system.

## Phase 1: Initial Improvements (March 29 -- April 1)

### Enable BVH and fix CUDA 13+/MSVC build (Paul, March 29)

The fork began with enabling the existing BVH acceleration structure and fixing compilation issues with CUDA 13+ and MSVC. This was the foundation commit that got the upstream code running cleanly on Windows.

### Command-line parameters (Claude, March 30)

The first Claude contribution replaced hard-coded magic constants with CLI arguments: `--width`, `--height`, `--samples`, `--depth`, and `--fov`. These were threaded through `Window -> Quad -> KernelInfo -> kernel/FrameBuffer`, establishing the pattern of runtime configurability that later features would build on. Merged as PR #1.

### Rendering and performance improvements (Claude, March 31)

A large feature branch added several capabilities in one commit:

- **Emissive materials** -- new `Emissive` material class with a virtual `emitted()` method on the `Material` base
- **New primitives** -- `Rectangle` (quad) and `Triangle` (Moller-Trumbore intersection)
- **Adaptive SPP** -- dynamic samples-per-pixel scaling targeting 30 FPS frame budget
- **Temporal reprojection** -- blending during camera motion via an accumulation shader
- **Camera update optimization** -- replaced a `set_device_camera` kernel launch with `cudaMemcpyAsync`
- **Build modernization** -- bumped CMake minimum to 3.24, switched to `CUDA_ARCHITECTURES native`

### README rewrite (Paul, March 31)

Rewrote the README to reflect the fork's new architecture with a diagram of the rendering pipeline.

### Next Event Estimation (Paul, April 1)

Implemented direct light sampling (Next Event Estimation / NEE), removed the adaptive SPP system in favor of a fixed accumulation approach, and increased the sample accumulation count. This was a significant rendering quality improvement -- instead of only finding lights by random chance, rays now explicitly sample light sources at each bounce.

## Phase 2: The Accumulation Bug Saga (April 2 -- 3)

This was the most protracted debugging effort of the project, spanning multiple failed attempts across both Paul and Claude before a working solution was found.

### Render-on-enter mode (Paul + Claude Haiku, April 2)

The three-state render mode was introduced: `PREVIEW` (camera unlocked, 1 SPP, responsive), `RENDER_FINAL` (camera locked, accumulating samples), and `IDLE` (displaying the finished frame). Enter toggled between these states.

The implementation changed from a single large kernel launch (all samples at once, which froze the GPU) to per-frame 1 SPP rendering with shader-based accumulation. **This broke sample accumulation** -- the final image showed only 1 sample per pixel regardless of the configured count. The commit message itself documents this as a known issue.

### Fix attempt 1: UB self-draw fix (Claude, April 2)

Claude identified that `tick_render` was drawing `_current_frame`'s texture onto its own framebuffer -- simultaneously reading and writing the same OpenGL texture, which is undefined behavior. The fix removed the redundant self-draw and upgraded accumulation textures from `GL_RGBA8` to `GL_RGBA16F` for proper precision during multi-frame blending. **This did not fully resolve the accumulation problem.**

### Debug instrumentation (Paul + Claude Opus, April 2)

With the bug persisting, comprehensive debug instrumentation was added: per-frame GL diagnostics with pixel readbacks, CUDA-side center-pixel logging, blend weight tracking, FBO completeness checks, and a convergence test comparing shader-accumulated output against a single full-SPP reference kernel call. Three compile-time `#ifdef` flags (`DEBUG_ACCUMULATION`, `DISABLE_PROGRESS`, `CONVERGENCE_TEST`) guarded all diagnostics for zero overhead when disabled. A `logs.txt` file was created to capture output.

### Fix attempt 2: Three more bugs found (Claude, April 2)

Claude identified three bugs working together to break accumulation:

1. `render_kernel(true)` hardcoded `camera_moving=true`, forcing `spp=1` even in `RENDER_FINAL` mode (originating from the render-on-enter commit)
2. `std::swap` had replaced `copyFrameBufferTexture`, causing the accumulation shader to read stale data and decay to zero (introduced by the debug instrumentation commit)
3. `sqrtf` gamma correction had been removed from the kernel (also from the debug commit)

The fix restored correct behavior and removed all debug instrumentation. **However, the per-frame accumulation approach still had fundamental issues.**

### The working solution: single-launch RENDER_FINAL (Claude, April 3)

The final resolution abandoned the per-frame 1 SPP accumulation strategy entirely. Instead, `RENDER_FINAL` now launches a single kernel call with the full `--samples` count, producing correct results at the cost of a GPU stall during rendering. The accumulation shader was also fixed to use a proper running average (`1/frameCount`), and Lambertian scattering was switched to cosine-weighted sampling.

This branch (`claude/fix-sample-accumulation-VEHzz`) was merged into master, replacing the broken accumulation code from the multiple failed fix attempts on master.

## Phase 3: Platform Port (April 6)

### Windows to Linux port (Paul + Claude Opus)

A major infrastructure change ported the entire build system to Linux. This session (partially preserved in exported logs) involved:

- **CMake modernization** -- replaced deprecated `FindCUDA` with `CUDAToolkit` targets (`CUDA::cudart`, `CUDA::curand`); gated Windows-only defines behind `if(WIN32)`
- **CMake presets** -- added `linux-base`/`linux-debug`/`linux-release`/`linux-cuda-debug` presets using Ninja, gcc/g++, with `CMAKE_CUDA_HOST_COMPILER` pinned to g++-13 (CUDA 12.4 doesn't support gcc-14 on Debian 13)
- **FindGLFW3 fix** -- added `/usr/lib/x86_64-linux-gnu` search path; gated Windows-only paths
- **Deleted Windows artifacts** -- removed bundled `include/GLFW/` (3.3.8) and `lib/glfw3.lib` that shadowed system GLFW 3.4 headers
- **CUDA 12.4 virtual destructor fix** -- replaced `__device__ virtual ~Hittable() = default` with an explicit empty body (CUDA 12.4 enforces execution-space matching)
- **Hybrid GPU support** -- on Linux, set `__NV_PRIME_RENDER_OFFLOAD=1` and `__GLX_VENDOR_LIBRARY_NAME=nvidia` so hybrid-GPU laptops route GL to the NVIDIA dGPU
- **Wayland workaround** -- requested the X11 backend via `glfwInitHint` so PRIME offload is honored on Wayland sessions

## Phase 4: Rendering Features (April 10 -- 11)

### Tiled progressive rendering and path tracing optimizations (Claude, April 10)

A substantial rendering upgrade delivered via a remote Claude Code session:

- **Tiled rendering** -- framebuffer divided into configurable tiles (default 64x64, `--tile-size` CLI) rendered left-to-right, top-to-bottom with per-tile display updates and `glTexSubImage2D` partial uploads. Escape aborts mid-render.
- **Cosine-weighted hemisphere sampling** -- replaced `random_in_unit_sphere` with proper cosine-weighted sampling in `Lambertian::scatter()`, reducing variance for diffuse surfaces
- **Russian Roulette path termination** -- after 3 bounces, low-contribution paths are terminated probabilistically with survivor compensation to stay unbiased
- **Philox RNG** -- switched per-pixel RNG from `curandState` (XORWOW) to `curandStatePhilox4_32_10_t` for better GPU throughput

### Pixelated preview and rasterization mode (Paul + Claude Opus, April 11)

Two preview features for interactive use:

- **`--preview-scale <N>`** -- in PREVIEW mode, a `raytrace_pixelated` kernel renders one ray per NxN block, giving a fast pixelated preview. Accumulation blend is bypassed to keep blocks sharp.
- **R key rasterization toggle** -- a new `Rasterizer` class generates meshes for scene primitives and draws them with flat color using dedicated shaders. When Enter starts `RENDER_FINAL`, the last rasterized frame is used as an underlay -- tiles are blitted over it as they complete, so unrendered regions show the rasterized preview.

## Phase 5: OBJ Loading and Textures (April 11 -- 12)

### OBJ/texture loading, shader manager, and click-and-drag editing (Claude, April 12)

The single largest commit (~18,900 lines added), establishing the mesh loading pipeline:

- **tinyobjloader + stb_image** vendored under `thirdparty/`
- **ObjLoader** -- flattens all OBJ shapes into one indexed `MeshVertex` buffer, synthesizes flat normals when missing
- **New RAII helpers** -- `Mesh`, `GLTexture`, `ShaderManager`, `Picker`
- **Scene as single source of truth** -- `Scene` owns all proxies, meshes, and textures; both the rasterizer and CUDA kernel read from it via `DeviceObjectDesc` POD descriptors
- **Click-and-drag editing** -- `Picker` + `Scene::ray_intersect` for object selection; mouse drag reprojects onto a screen-parallel plane; Tab toggles edit mode
- **`TriangleMesh` hittable** -- device-side mesh with shared vertex/index buffers
- **CLI flags** -- `--obj <path>` and `--texture <path>` for asset loading

### MTL-relative path fix (Paul + Claude Sonnet, April 11)

Fixed tinyobjloader not finding `.mtl` files: passed the OBJ file's parent directory as `mtl_basedir` so materials and textures resolve relative to the model, not the working directory.

### GPU-side per-mesh BVH (Claude, April 12)

Replaced the O(N) brute-force triangle loop in `TriangleMesh::hit()` with an iterative stack-based BVH traversal:

- **SAH-binned build** -- 16 bins, max 4 triangles per leaf, built on host
- **Flat node array** -- uploaded as index-based `MeshBVHNode` array alongside reordered triangle IDs
- **Device traversal** -- 32-entry fixed stack, operates in local mesh space with inverse-transformed ray

New files: `MeshBVHBuilder.h/.cpp`, `MeshBVHNode.h`.

### Scene rework and disc light (Paul + Claude Sonnet, April 12)

Replaced the test scene primitives with a disc light and metal sphere. Introduced the `Disc` hittable with ray-plane + radius intersection, correct AABB, and uniform disc sampling for NEE. Fixed two CUDA texture pipeline bugs: ground checker texture not propagating through descriptors, and mesh image textures not uploading UV buffers or interpolating texture coordinates.

### Floor intersection and MTL texture auto-loading (Paul, April 12)

Fixed floor geometry intersection and added automatic diffuse texture loading from MTL material definitions.

### Normal and specular mapping (Paul, April 12)

The final feature commit added PBR texture support:

- Normal mapping with tangent-space-to-world-space transformation in `TriangleMesh::hit()`
- Specular mapping modulating material reflectance
- Device-side normal/specular texture sampling in the CUDA kernel
- Host-side upload pipeline through `DeviceObjectDesc` and `KernelInfo`

---

## Summary of Contributions

| Contributor | Commits | Key areas |
|---|---|---|
| **Paul** | 16 | Platform port, NEE, render modes, preview features, scene design, PBR textures, bug fixes |
| **Claude** | 9 | CLI params, rendering improvements, accumulation fixes, tiled rendering, OBJ loading, mesh BVH |
| **Paul + Claude** | (overlapping) | Debug instrumentation, Linux port, various bug fixes |

## Models Used

Several Claude models were used throughout the project:

- **Claude Opus 4.6** -- Linux port, debug instrumentation, general development sessions
- **Claude Sonnet 4.6** -- MTL path fix, disc/texture work, normal/specular mapping sessions
- **Claude Haiku 4.5** -- Initial render-on-enter mode
- **Claude (remote sessions)** -- Tiled rendering, OBJ loading, mesh BVH (via Claude Code remote agents)
