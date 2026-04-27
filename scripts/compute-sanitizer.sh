#!/usr/bin/env bash
# Build the headless raytracer with CUDA debug info and run it under
# NVIDIA compute-sanitizer to catch GPU-side bugs.
#
# Usage:
#   scripts/compute-sanitizer.sh              # default: memcheck
#   scripts/compute-sanitizer.sh racecheck    # shared-memory data races
#   scripts/compute-sanitizer.sh initcheck    # uninitialized device reads
#   scripts/compute-sanitizer.sh synccheck    # __syncthreads / warp-sync misuse
#
# A 64x64x1spp render is used to keep sanitizer overhead bearable
# (compute-sanitizer slows kernels by 10-100x).

set -euo pipefail

tool="${1:-memcheck}"
case "$tool" in
  memcheck|racecheck|initcheck|synccheck) ;;
  *) echo "error: unknown tool '$tool' (expected memcheck|racecheck|initcheck|synccheck)" >&2; exit 2 ;;
esac

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "error: compute-sanitizer not found on PATH (ships with the CUDA toolkit)" >&2
  exit 127
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build-headless"
binary="$build_dir/cuda-raytracer"
out_png="${COMPUTE_SANITIZER_OUTPUT:-/tmp/sanitize.png}"

if [[ ! -x "$binary" ]]; then
  echo "Configuring headless debug build at $build_dir ..."
  cmake -S "$repo_root" -B "$build_dir" \
    -DHEADLESS=ON -DIS_CUDA_DEBUG=TRUE -DCMAKE_BUILD_TYPE=Debug >/dev/null
  cmake --build "$build_dir" --parallel
fi

echo "Running compute-sanitizer --tool=$tool on a 64x64x1spp render ..."
set +e
compute-sanitizer --tool="$tool" --launch-timeout=600 \
  "$binary" \
    --width 64 --height 64 --samples 1 --depth 2 \
    --output "$out_png"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "compute-sanitizer ($tool): PASS — wrote $out_png"
else
  echo "compute-sanitizer ($tool): FAIL (exit $status)" >&2
fi
exit $status
