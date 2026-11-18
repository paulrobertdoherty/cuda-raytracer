#pragma once

#include <cstdint>

struct PreviewState {
	int frame_count = 1;
	bool raster_enabled = false;
	bool camera_moving = false;
};

// Input and the sidebar request transitions. When final drawing returns, idle
// keeps the last image on screen, including a partial render stopped by Escape.
class RenderState {
public:
	PreviewState* active_preview() {
		return _phase == Phase::Preview ? &_preview : nullptr;
	}

	void reset_accumulation() { _preview.frame_count = 1; }

	bool raster_preview_active() const {
		return _phase == Phase::Preview && _preview.raster_enabled;
	}

	void start_final() {
		if (_phase == Phase::Preview) {
			_phase = Phase::Final;
		}
	}

	void toggle_final() {
		if (_phase == Phase::Idle) {
			_preview = {};
			_phase = Phase::Preview;
		} else {
			start_final();
		}
	}

	template <class DrawPreview, class DrawFinal>
	void draw_frame(DrawPreview draw_preview, DrawFinal draw_final) {
		switch (_phase) {
		case Phase::Preview:
			draw_preview(_preview);
			++_preview.frame_count;
			break;
		case Phase::Final:
			draw_final();
			_phase = Phase::Idle;
			break;
		case Phase::Idle:
			break;
		}
	}

private:
	enum class Phase : std::uint8_t { Preview, Final, Idle };
	Phase _phase = Phase::Preview;
	PreviewState _preview;
};
