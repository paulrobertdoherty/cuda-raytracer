#pragma once

#include <atomic>
#include <cstdint>

// A fresh revision distinguishes both reloads and different assets that occupy
// the same scene slot. Zero denotes an asset with no uploaded content.
inline std::uint64_t next_asset_revision() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}
