# Repo Issue Audit — CUDA Ray Tracer

Catalog of issues identified across GPU kernels, host architecture, and code quality. Each item has a beads ticket for later triage. This is documentation only — no code was modified.

**Severity rubric**

- **P0** — correctness bug that can corrupt output or crash (none found that aren't already tracked)
- **P1** — major performance problem or significant design flaw
- **P2** — moderate refactor / perf win / safety gap
- **P3** — nit / hygiene / polish

---

## GPU / CUDA Kernel Issues (15)

### P1 — major

| Beads | Title |
|---|---|
| [`cuda-raytracer-0x5`](#0x5) | Virtual `Hittable::hit` / `Material::scatter` dispatch causes warp divergence |
| [`cuda-raytracer-ftm`](#ftm) | BVH uses random-axis median split, not SAH |
| [`cuda-raytracer-x4p`](#x4p) | Single-thread `<<<1,1>>>` scene-construction kernels |
| [`cuda-raytracer-8e6`](#8e6) | Inconsistent self-intersection epsilons (1e-8 vs 0.001f) |
| [`cuda-raytracer-dv9`](#dv9) | No NaN/Inf guards in main path-tracing loop |
| [`cuda-raytracer-823`](#823) | AABB slab test divides by ray direction without guarding zero components |
| [`cuda-raytracer-5yi`](#5yi) | Divergent NEE material-type branch |

<a id="0x5"></a>**`cuda-raytracer-0x5` — Virtual dispatch in hot loop** (`FrameBuffer.h:78,105`, `Material.h`)
Vtable dispatch inside the per-bounce loop causes warp serialization when threads hit different subclasses. *Why it matters:* hottest kernel in the project; divergence directly costs throughput.

<a id="ftm"></a>**`cuda-raytracer-ftm` — Random-axis median BVH split** (`src/raytracer/BVHNode.h:39-51`)
Build picks axis randomly and splits at median. *Why it matters:* produces deep, unbalanced trees; traversal cost scales with tree quality.

<a id="x4p"></a>**`cuda-raytracer-x4p` — `<<<1,1>>>` scene kernels** (`kernel.cu:112-117, 124-212, ~437`)
`create_world`, `build_world_from_desc`, `free_scene` all launched with one thread. *Why it matters:* BVH built serially on device; every scene rebuild stalls startup.

<a id="8e6"></a>**`cuda-raytracer-8e6` — Inconsistent self-intersection epsilons** (`FrameBuffer.h:70,100,104`, `Triangle.h:29`, `TriangleMesh.h:123`, `Disc.h:44`, `Rectangle.h:36`)
Geometry uses `1e-8`, shadow rays use `0.001f`. *Why it matters:* mismatch produces shadow acne or false occlusion.

<a id="dv9"></a>**`cuda-raytracer-dv9` — No NaN/Inf guards** (`FrameBuffer.h:48-137`)
A single poisoned sample contaminates the accumulation buffer forever. *Why it matters:* hard-to-diagnose persistent artifacts across frames.

<a id="823"></a>**`cuda-raytracer-823` — AABB div-by-zero** (`AABB.h:26-31`)
Divides by ray direction without handling zero components. *Why it matters:* undefined behavior for axis-aligned rays, which occur naturally at scene axes.

<a id="5yi"></a>**`cuda-raytracer-5yi` — NEE material branch divergence** (`FrameBuffer.h:82-122`)
NEE path branches on material type per bounce. *Why it matters:* warp efficiency suffers on scenes with mixed materials.

### P2 — moderate

| Beads | Title |
|---|---|
| [`cuda-raytracer-ydu`](#ydu) | `MAX_STACK=32` fixed, no overflow check |
| [`cuda-raytracer-vwx`](#vwx) | Missing `__restrict__` on device pointers |
| [`cuda-raytracer-p8s`](#p8s) | `curand_init` over full grid on every resize |
| [`cuda-raytracer-85s`](#85s) | `FrameBuffer` passed by value |
| [`cuda-raytracer-ppx`](#ppx) | Per-frame `cudaMalloc`/`cudaFree` for descriptor array |
| [`cuda-raytracer-45u`](#45u) | Block size hardcoded 16×16 |
| [`cuda-raytracer-bl7`](#bl7) | Device-side `new`/`delete` with no pooling |
| [`cuda-raytracer-8u7`](#8u7) | `World` capacity hardcoded to 20 |

<a id="ydu"></a>**`cuda-raytracer-ydu` — MAX_STACK overflow silent** (`TriangleMesh.h:89-91`)
Mesh BVH traversal stack is fixed 32 entries with no guard. *Why it matters:* deep meshes silently overflow on GPU.

<a id="vwx"></a>**`cuda-raytracer-vwx` — Missing `__restrict__`** (`kernel.h`, `TriangleMesh.h`, `Material.h`)
Device pointers lack the non-aliasing qualifier. *Why it matters:* inhibits loop optimization and cache hints.

<a id="p8s"></a>**`cuda-raytracer-p8s` — curand full-grid reinit on resize** (`kernel.cu:214-221`)
Re-initializes RNG state for every pixel on every resize — ~33M `curand_init` calls at 4K. *Why it matters:* noticeable hitch on window resize.

<a id="85s"></a>**`cuda-raytracer-85s` — FrameBuffer by value** (`kernel.cu:41,72`)
Struct copied through kernel parameter space each launch. *Why it matters:* avoidable per-launch overhead.

<a id="ppx"></a>**`cuda-raytracer-ppx` — Per-frame descriptor allocs** (`kernel.cu:~805-810`)
`cudaMalloc`/`cudaFree` for the descriptor array every frame. *Why it matters:* device allocator is not free; should be persistent.

<a id="45u"></a>**`cuda-raytracer-45u` — Fixed 16×16 block size** (`kernel.cu:258,301,314,389,427`)
No occupancy tuning. *Why it matters:* likely suboptimal on different GPUs / register pressures.

<a id="bl7"></a>**`cuda-raytracer-bl7` — Device new/delete** (`kernel.cu:~134-200`)
In-kernel `new`/`delete` for Materials, Textures, BVH nodes. *Why it matters:* device heap fragments across rebuilds; no pooling.

<a id="8u7"></a>**`cuda-raytracer-8u7` — World capacity 20** (`World.h:26,28,30,40,63-71`)
Fixed array; `add()` / `add_light()` silently return false on overflow. *Why it matters:* host/device state diverge when scenes grow. Carry-over from `TODO.md`.

---

## Host Architecture Issues (9)

### P1 — major

| Beads | Title |
|---|---|
| [`cuda-raytracer-ek6`](#ek6) | `Window` is a 625-line god object |
| [`cuda-raytracer-sk7`](#sk7) | Circular `Gui` ↔ `Window` coupling |

<a id="ek6"></a>**`cuda-raytracer-ek6` — Window god object** (`src/Window.cpp`)
GL setup, input, render loop, Scene, GUI, and CUDA orchestration live in one class. *Why it matters:* change amplification; untestable.

<a id="sk7"></a>**`cuda-raytracer-sk7` — Gui ↔ Window coupling** (`Gui.cpp:99,179-187`)
`Gui` directly mutates `Window` public members (`app.samples`, `app.fov`). *Why it matters:* circular dependency, no encapsulation.

### P2 — moderate

| Beads | Title |
|---|---|
| [`cuda-raytracer-dar`](#dar) | Full scene rebuild on every modification |
| [`cuda-raytracer-dpe`](#dpe) | `Quad` exposes raw CUDA interop members as public |
| [`cuda-raytracer-bci`](#bci) | `HEADLESS_BUILD` macro scattered across files |
| [`cuda-raytracer-hbk`](#hbk) | `RenderMode` conditionals spread across tick methods |
| [`cuda-raytracer-pjz`](#pjz) | Silent CUDA errors in headless `render_to_buffer` |
| [`cuda-raytracer-5mt`](#5mt) | No tests; `test_compile.sh` is a build smoke-check |
| [`cuda-raytracer-0ab`](#0ab) | Manual init/cleanup ordering in `Window` |

<a id="dar"></a>**`cuda-raytracer-dar` — Full rebuild on any change** (`Window.cpp:229,259,400`, `kernel.cu`)
Moving one object rebuilds the whole scene and BVH. *Why it matters:* makes even trivial edits laggy.

<a id="dpe"></a>**`cuda-raytracer-dpe` — Quad public interop members** (`Quad.h:12-35`)
Raw `cudaGraphicsResource_t` and `unique_ptr<KernelInfo>` are public fields. *Why it matters:* any caller can corrupt ownership.

<a id="bci"></a>**`cuda-raytracer-bci` — HEADLESS_BUILD scattered** (`main.cpp`, `Quad.cpp`, `Scene.h`)
`#ifdef` branches in multiple files. *Why it matters:* hard to reason about which code depends on GL.

<a id="hbk"></a>**`cuda-raytracer-hbk` — RenderMode conditionals** (`Window.cpp:268-273,318-330,451-505`)
`PREVIEW` vs `RENDER_FINAL` branches duplicated across tick methods. *Why it matters:* implicit state machine is error-prone.

<a id="pjz"></a>**`cuda-raytracer-pjz` — Silent CUDA errors in headless path** (`kernel.cu:286-307`)
Headless `render_to_buffer` path doesn't check CUDA errors. *Why it matters:* failed renders produce blank/garbage PNGs with no log.

<a id="5mt"></a>**`cuda-raytracer-5mt` — No regression harness** (`test_compile.sh`)
Compile-only; no rendering or math tests. *Why it matters:* silent regressions in output quality or material math. Filed as `--type=feature`.

<a id="0ab"></a>**`cuda-raytracer-0ab` — Brittle init/cleanup ordering** (`Window.cpp:241-286`)
Manual per-subsystem init/teardown. *Why it matters:* reorder mistakes cause double-free or leaks.

### P3 — polish

| Beads | Title |
|---|---|
| [`cuda-raytracer-bqz`](#bqz) | `SceneTexture` GL/Headless alias with no common base |

<a id="bqz"></a>**`cuda-raytracer-bqz` — SceneTexture alias** (`Scene.h:11-17`)
Compile-time swap with no shared interface. *Why it matters:* each new operation must be duplicated in both types.

---

## Code Quality & Hygiene (9)

### P2 — moderate

| Beads | Title |
|---|---|
| [`cuda-raytracer-10q`](#10q) | Vendored `glm` and `glad` source trees |
| [`cuda-raytracer-fp2`](#fp2) | `DeviceObjectDesc` POD with split ownership |

<a id="10q"></a>**`cuda-raytracer-10q` — Vendored glm/glad** (`include/glm/` 3.1M, `include/glad/` 212K)
Source-copy vendoring. *Why it matters:* inflates clone, manual upstream tracking. Suggest FetchContent or submodule.

<a id="fp2"></a>**`cuda-raytracer-fp2` — DeviceObjectDesc ownership split** (`kernel.h:80-114`)
POD with raw device pointers; ownership split between `KernelInfo` (host) and `World` (device). *Why it matters:* no clear contract on who frees what.

### P3 — polish

| Beads | Title |
|---|---|
| [`cuda-raytracer-exu`](#exu) | `.gitignore` gaps |
| [`cuda-raytracer-yrt`](#yrt) | `std::exit(99)` on CUDA error |
| [`cuda-raytracer-y67`](#y67) | Ad-hoc `std::cout`/`cerr` logging |
| [`cuda-raytracer-z5z`](#z5z) | No `-Wall -Wextra` for CUDA host code |
| [`cuda-raytracer-3zx`](#3zx) | Hardcoded shader paths |
| [`cuda-raytracer-pm5`](#pm5) | README 500-frame docs mismatch |

<a id="exu"></a>**`cuda-raytracer-exu` — .gitignore gaps**
Currently untracked: `build-headless/`, `obj-render.png`, root `2026-04-12-*.txt`, `scripts/`, `sfcompute-integration-plan.md`. *Why it matters:* repeated ambiguity over what to commit.

<a id="yrt"></a>**`cuda-raytracer-yrt` — std::exit(99)** (`src/cuda_errors.cpp:12`)
Skips destructors on CUDA failure. *Why it matters:* GL/ImGui/CUDA state unclean on exit; should throw.

<a id="y67"></a>**`cuda-raytracer-y67` — Ad-hoc logging** (`~12 files`)
No levels, no redirect, no unified facade. *Why it matters:* noisy and inflexible.

<a id="z5z"></a>**`cuda-raytracer-z5z` — No CUDA -Wall** (`CMakeLists.txt:167-177`)
`-Wall -Wextra` not propagated via `-Xcompiler`. *Why it matters:* device builds silently accept warnings host would reject.

<a id="3zx"></a>**`cuda-raytracer-3zx` — Hardcoded shader paths** (`src/Rasterizer.cpp:15-18`)
Relative `./shaders/...` paths break if run from wrong cwd. *Why it matters:* installed binaries can't find shaders.

<a id="pm5"></a>**`cuda-raytracer-pm5` — README 500-frame mismatch** (`README.md:128-130`)
Docs reference a rate the kernel doesn't expose. *Why it matters:* docs/code drift.

---

## Summary Table

| ID | Severity | Category | Type | Title |
|---|---|---|---|---|
| `cuda-raytracer-0x5` | P1 | GPU | task | Virtual Hittable/Material dispatch divergence |
| `cuda-raytracer-ftm` | P1 | GPU | task | BVH random-axis median split (not SAH) |
| `cuda-raytracer-x4p` | P1 | GPU | task | Single-thread scene-construction kernels |
| `cuda-raytracer-8e6` | P1 | GPU | bug | Inconsistent self-intersection epsilons |
| `cuda-raytracer-dv9` | P1 | GPU | bug | No NaN/Inf guards in path-tracing loop |
| `cuda-raytracer-823` | P1 | GPU | bug | AABB slab test div-by-zero |
| `cuda-raytracer-5yi` | P1 | GPU | task | Divergent NEE material branch |
| `cuda-raytracer-ek6` | P1 | Arch | task | Window is a 625-line god object |
| `cuda-raytracer-sk7` | P1 | Arch | task | Gui ↔ Window circular coupling |
| `cuda-raytracer-ydu` | P2 | GPU | bug | MAX_STACK=32 overflow silent |
| `cuda-raytracer-vwx` | P2 | GPU | task | Missing `__restrict__` |
| `cuda-raytracer-p8s` | P2 | GPU | task | curand_init full-grid on resize |
| `cuda-raytracer-85s` | P2 | GPU | task | FrameBuffer passed by value |
| `cuda-raytracer-ppx` | P2 | GPU | task | Per-frame descriptor malloc/free |
| `cuda-raytracer-45u` | P2 | GPU | task | Block size hardcoded 16×16 |
| `cuda-raytracer-bl7` | P2 | GPU | task | Device `new`/`delete` with no pooling |
| `cuda-raytracer-8u7` | P2 | GPU | bug | World capacity hardcoded to 20 |
| `cuda-raytracer-dar` | P2 | Arch | task | Full scene rebuild on any change |
| `cuda-raytracer-dpe` | P2 | Arch | task | Quad public CUDA interop members |
| `cuda-raytracer-bci` | P2 | Arch | task | HEADLESS_BUILD macro scattered |
| `cuda-raytracer-hbk` | P2 | Arch | task | RenderMode conditionals spread |
| `cuda-raytracer-pjz` | P2 | Arch | bug | Silent CUDA errors in headless path |
| `cuda-raytracer-5mt` | P2 | Arch | feature | No tests / no regression harness |
| `cuda-raytracer-0ab` | P2 | Arch | task | Brittle Window init/cleanup ordering |
| `cuda-raytracer-10q` | P2 | CQ | task | Vendored glm/glad source trees |
| `cuda-raytracer-fp2` | P2 | CQ | task | DeviceObjectDesc POD ownership split |
| `cuda-raytracer-bqz` | P3 | Arch | task | SceneTexture GL/Headless alias |
| `cuda-raytracer-exu` | P3 | CQ | task | `.gitignore` gaps |
| `cuda-raytracer-yrt` | P3 | CQ | bug | `std::exit(99)` on CUDA error |
| `cuda-raytracer-y67` | P3 | CQ | task | Ad-hoc logging |
| `cuda-raytracer-z5z` | P3 | CQ | task | No `-Wall -Wextra` for CUDA |
| `cuda-raytracer-3zx` | P3 | CQ | bug | Hardcoded shader paths |
| `cuda-raytracer-pm5` | P3 | CQ | bug | README 500-frame mismatch |

**Counts:** 9 × P1 · 17 × P2 · 7 × P3 · 33 total · 11 bugs · 21 tasks · 1 feature

---

## Categories key

- **GPU** — device kernels, CUDA memory, BVH, intersection math
- **Arch** — host-side architecture (Window/Gui/Scene/headless split)
- **CQ** — code quality, hygiene, build, docs

## How to pick up this work

```bash
bd show cuda-raytracer-<id>   # read the full description
bd update cuda-raytracer-<id> --claim
# ... fix ...
bd close cuda-raytracer-<id>
```

Pre-existing UI bugs (not from this audit) that also remain open: `cuda-raytracer-20x`, `cuda-raytracer-hpy`, `cuda-raytracer-1bo`.
