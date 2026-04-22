#pragma once

#include "raytracer/kernel.h"

// Cohesive value object for interactive render tuning knobs. Owned by Window
// and exposed to the GUI via Window::render_params(); grouping them here keeps
// the GUI from reaching into Window's own fields to mutate state.
struct RenderParams {
    int samples = 3;
    int max_depth = 50;
    float fov = 90.0f;
    int tile_size = DEFAULT_TILE_SIZE;
    int preview_scale = 1;
};
