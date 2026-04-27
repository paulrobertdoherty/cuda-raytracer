#!/usr/bin/env bash
# Run clang-tidy over host C++ sources using compile_commands.json from build/.
#
# Usage:
#   scripts/clang-tidy.sh                      # check all host .cpp/.h under src/
#   scripts/clang-tidy.sh src/Window.cpp       # check a single file
#   scripts/clang-tidy.sh --fix                # apply auto-fixes where available
#
# .cu/.cuh files are skipped (CUDA tidy needs special flags; out of scope).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"

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
  mapfile -d '' files < <(find "$repo_root/src" \( -name '*.cpp' -o -name '*.h' \) -print0)
  extra_args=("$@")
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no source files found under src/"
  exit 1
fi

echo "Running clang-tidy on ${#files[@]} file(s) ..."
clang-tidy -p "$build_dir" "${extra_args[@]}" "${files[@]}"
