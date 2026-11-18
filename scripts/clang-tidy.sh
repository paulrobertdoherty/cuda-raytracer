#!/usr/bin/env bash
# Run clang-tidy over host C++ sources using compile_commands.json from build/.
#
# Usage:
#   scripts/clang-tidy.sh                      # check the configured host sources and their headers
#   scripts/clang-tidy.sh src/Window.cpp       # check a single file
#   scripts/clang-tidy.sh --fix                # apply auto-fixes where available
#
# Output is tee'd to build/tidy.log for later summarization.
#
# Skipped from the full sweep:
#   - .cu/.cuh files (CUDA tidy needs special flags; out of scope)
#   - src/raytracer/*.h (CUDA __device__/__global__ keywords trip the host
#     parser). Pass them explicitly if you want to try anyway.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
log_file="$build_dir/tidy.log"

if ! command -v clang-tidy >/dev/null 2>&1; then
  echo "error: clang-tidy not found on PATH" >&2
  exit 127
fi

if [[ ! -f "$build_dir/compile_commands.json" ]]; then
  echo "Configuring CMake build at $build_dir ..."
  cmake -S "$repo_root" -B "$build_dir" >/dev/null
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  files=("$@")
  extra_args=()
else
  # Use the configured translation units so mutually exclusive backends
  # are checked with their own include paths.
  mapfile -t files < <(python3 - "$build_dir/compile_commands.json" "$repo_root/src" <<'PYTHON'
import json
from pathlib import Path
import sys

root = Path(sys.argv[2])
entries = json.loads(Path(sys.argv[1]).read_text())
paths = {Path(entry["directory"]) / entry["file"] for entry in entries}
for path in sorted(paths):
    if root in path.parents and path.suffix == ".cpp":
        print(path)
PYTHON
  )
  extra_args=("$@")
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no source files found under src/"
  exit 1
fi

echo "Running clang-tidy on ${#files[@]} file(s); logging to $log_file ..."
clang-tidy -p "$build_dir" "${extra_args[@]}" "${files[@]}" 2>&1 | tee "$log_file"
