#!/usr/bin/env bash
# Build the headless raytracer with CUDA debug info and run it under
# NVIDIA compute-sanitizer across multiple tools and scenarios.
#
# Usage:
#   scripts/compute-sanitizer.sh                       # all tools x all scenarios
#   scripts/compute-sanitizer.sh memcheck              # one tool, all scenarios
#   scripts/compute-sanitizer.sh --tool=initcheck      # explicit tool flag
#   scripts/compute-sanitizer.sh --obj=path/to.obj     # force a specific mesh
#   scripts/compute-sanitizer.sh --quick               # legacy: memcheck, 64x64x1spp
#   scripts/compute-sanitizer.sh --width=192 --depth=8 # override render dims
#
# Tools (default: all four):
#   memcheck   — OOB device memory access, leaks
#   initcheck  — reads of uninitialized device memory  (catches the
#                Camera self-assign / construct_camera class of bugs)
#   racecheck  — shared-memory data races
#   synccheck  — __syncthreads / __syncwarp misuse
#
# Scenarios:
#   default — spheres-only scene (no mesh, no texture). Exercises the
#             core path-tracing kernel and the sphere/disc primitives.
#   obj+bvh — adds --obj <auto-discovered mesh>, which exercises the
#             mesh BVH traversal, texture sampling, and per-vertex
#             attribute paths. Skipped if no .obj is found.
#
# Compute-sanitizer slows kernels 10-100x, so render dims are small.
# Output PNGs land under $COMPUTE_SANITIZER_OUTDIR (default /tmp/sanitize).

set -euo pipefail

WIDTH=128
HEIGHT=128
SAMPLES=1
DEPTH=4
TOOLS=()
OBJ_PATH=""
DEFAULT_TOOLS=(memcheck initcheck racecheck synccheck)

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --quick) WIDTH=64; HEIGHT=64; SAMPLES=1; DEPTH=2; TOOLS=(memcheck) ;;
    --tool=*) TOOLS+=("${1#--tool=}") ;;
    --obj=*) OBJ_PATH="${1#--obj=}" ;;
    --width=*) WIDTH="${1#--width=}" ;;
    --height=*) HEIGHT="${1#--height=}" ;;
    --samples=*) SAMPLES="${1#--samples=}" ;;
    --depth=*) DEPTH="${1#--depth=}" ;;
    memcheck|racecheck|initcheck|synccheck) TOOLS+=("$1") ;;
    *) echo "error: unknown arg '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

[[ ${#TOOLS[@]} -eq 0 ]] && TOOLS=("${DEFAULT_TOOLS[@]}")

for t in "${TOOLS[@]}"; do
  case "$t" in
    memcheck|racecheck|initcheck|synccheck) ;;
    *) echo "error: unknown tool '$t' (expected memcheck|racecheck|initcheck|synccheck)" >&2; exit 2 ;;
  esac
done

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "error: compute-sanitizer not found on PATH (ships with the CUDA toolkit)" >&2
  exit 127
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build-headless"
binary="$build_dir/cuda-raytracer"

if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  echo "Configuring headless debug build at $build_dir ..."
  cmake -S "$repo_root" -B "$build_dir" \
    -DHEADLESS=ON -DIS_CUDA_DEBUG=TRUE -DCMAKE_BUILD_TYPE=Debug >/dev/null
fi
echo "Building $build_dir ..."
cmake --build "$build_dir" --parallel >/dev/null

# Auto-discover an .obj so the BVH + texture-sampling paths get exercised.
if [[ -z "$OBJ_PATH" ]]; then
  for cand in \
    "$repo_root/bunny.obj" \
    "$repo_root/backpack/backpack.obj" \
    "$repo_root/build/bunny.obj" \
    "$build_dir/bunny.obj"; do
    if [[ -f "$cand" ]]; then OBJ_PATH="$cand"; break; fi
  done
fi

SCENARIO_NAMES=(default)
SCENARIO_ARGS=("")
if [[ -n "$OBJ_PATH" && -f "$OBJ_PATH" ]]; then
  SCENARIO_NAMES+=(obj+bvh)
  SCENARIO_ARGS+=("--obj $OBJ_PATH")
  echo "Mesh scenario will use: $OBJ_PATH"
else
  echo "No .obj auto-discovered; skipping BVH/texture scenario." >&2
  echo "  (pass --obj=path/to.obj to force one)" >&2
fi

out_dir="${COMPUTE_SANITIZER_OUTDIR:-/tmp/sanitize}"
mkdir -p "$out_dir"

results=()
failures=0

for tool in "${TOOLS[@]}"; do
  for i in "${!SCENARIO_NAMES[@]}"; do
    name="${SCENARIO_NAMES[$i]}"
    extra="${SCENARIO_ARGS[$i]}"
    out_png="$out_dir/${tool}_${name}.png"
    echo
    echo "==> $tool / $name  (${WIDTH}x${HEIGHT}, ${SAMPLES}spp, depth ${DEPTH})"
    set +e
    # shellcheck disable=SC2086  -- intentional word-split of $extra into argv
    compute-sanitizer --tool="$tool" --launch-timeout=600 \
      "$binary" \
        --width "$WIDTH" --height "$HEIGHT" \
        --samples "$SAMPLES" --depth "$DEPTH" \
        $extra \
        --output "$out_png"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      results+=("PASS  $tool / $name  -> $out_png")
    else
      results+=("FAIL  $tool / $name  (exit $status)")
      failures=$((failures + 1))
    fi
  done
done

echo
echo "===== compute-sanitizer summary ====="
for r in "${results[@]}"; do echo "  $r"; done
echo "====================================="

if [[ $failures -gt 0 ]]; then
  echo "$failures of ${#results[@]} runs FAILED" >&2
  exit 1
fi
echo "All ${#results[@]} runs PASS"
exit 0
