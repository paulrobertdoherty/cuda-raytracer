#include "RenderState.h"

#include <iostream>
#include <vector>

int main() {
    RenderState state;
    std::vector<int> preview_frames;
    int final_draws = 0;
    const auto draw = [&] {
        state.draw_frame(
            [&](const PreviewState& preview) { preview_frames.push_back(preview.frame_count); },
            [&] { ++final_draws; });
    };

    draw();
    draw();
    state.reset_accumulation();
    draw();
    state.active_preview()->raster_enabled = true;
    state.active_preview()->camera_moving = true;

    state.start_final();
    state.toggle_final();  // A second request before the first tile cannot cancel it.
    draw();
    state.start_final();   // The sidebar button only starts a final render from preview.
    draw();
    if (final_draws != 1 || state.active_preview() || state.raster_preview_active()) {
        std::cerr << "The completed final render should remain on screen\n";
        return 1;
    }

    state.toggle_final();
    const auto* preview = state.active_preview();
    if (!preview || preview->raster_enabled || preview->camera_moving) {
        std::cerr << "Enter should resume ray-traced preview\n";
        return 1;
    }
    draw();
    state.toggle_final();  // Enter can also start a final render.
    draw();
    draw();
    if (final_draws != 2 || preview_frames != std::vector<int>{1, 2, 1, 1}) {
        std::cerr << "Unexpected render count or preview accumulation sequence\n";
        return 1;
    }
    return 0;
}
