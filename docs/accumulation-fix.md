# Accumulation Failure: Root Cause Analysis & Fix

## Summary

After pressing Enter to start RENDER_FINAL mode, the accumulated image showed
only 1 sample per pixel instead of converging over `--samples` frames. This
was caused by two bugs: one in `8a0995b` and one in `602b8aa`.

## Root Causes

### Bug 1: Kernel always received `spp=1` (introduced in `8a0995b`)

In `Window::tick_render()`, the CUDA kernel was called with a hardcoded `true`:

```cpp
_current_frame->render_kernel(true);   // always camera_moving=true
```

Inside `KernelInfo::render(bool camera_moving)`:

```cpp
int spp = camera_moving ? 1 : samples;   // always took the 1-SPP branch
```

Even in RENDER_FINAL mode (camera locked), the kernel only computed 1 sample
per pixel per frame. Debug logs confirmed every frame printed `spp=1`.

**Fix**: Pass `_camera_moving` instead of `true`. In RENDER_FINAL mode the
camera is locked so `_camera_moving` is `false`, giving the kernel the full
`samples` count.

### Bug 2: `std::swap` broke accumulation (introduced in `602b8aa`)

The debug commit replaced `copyFrameBufferTexture()` (a proper FBO blit) with
`std::swap(_accum_frame, _blit_quad)`. This caused the accumulation shader's
`lastFrameTex` to read stale data instead of the previous accumulated result.

Evidence from debug logs — accumulated values decayed exponentially:
- Frame 5: `accum_out = 0.446` (one good sample hit the light)
- Frame 6: `accum_out = 0.005` (expected ~0.372 from running average)

**Fix**: Restored `copyFrameBufferTexture` blit.

### Bug 3: Missing gamma correction (introduced in `602b8aa`)

The `sqrtf()` gamma pass was removed from the CUDA kernel, producing
too-dark linear values written to 8-bit textures.

**Fix**: Restored `sqrtf` gamma correction.

## What Was NOT Changed

Everything from `8a0995b` and `3483edd` that works correctly is preserved:

- Render mode system (PREVIEW / RENDER_FINAL / IDLE) and Enter key handling
- Sky background removal and Lambertian scattering change (from `8a0995b`)
- `GL_RGBA16F` textures for accum/blit quads (from `3483edd`)
- PBO/texture unbind cleanup in `render_kernel` (from `3483edd`)
- VAO bind before accumulation shader draw (from `3483edd`)
- Accumulation shader formula: `mix(accumulated, current, 1.0/frameCount)`

## Files Changed

- `src/Window.cpp` — Removed debug code, fixed `render_kernel(true)` → `render_kernel(_camera_moving)`, restored `copyFrameBufferTexture`
- `src/Window.h` — Removed convergence test member
- `src/raytracer/kernel.cu` — Removed debug instrumentation, restored `sqrtf` gamma correction
