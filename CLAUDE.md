# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / run

Toolchain prerequisites: NVIDIA OptiX SDK 9.0.0 (`OPTIX_ROOT`), CUDA Toolkit (NVRTC), LLVM/Clang (`LIBCLANG_PATH` — already set in `.cargo/config.toml` to `C:\Program Files\LLVM\bin`), Python 3.11+ (PyO3 abi3-py311), and an OptiX-capable NVIDIA GPU.

There is **no standalone CLI binary**. The renderer ships only as a PyO3 extension (`vibrt_native.pyd`) that the Blender addon loads in-process. The disk-loading path was removed alongside the addon's subprocess fallback.

```bash
cargo build --release -p vibrt --features python    # produces target/release/vibrt_native.dll (the cdylib)
cargo build --release -p vibrt                      # rlib only — useful when iterating on Rust without rebuilding pyo3 deps
make dev-install                                    # build .pyd, stage next to addon source, junction into Blender's user addons
```

Pure-Python unit tests live alongside the Blender addon and run under plain CPython (no Blender required):

```bash
make test            # blender/vibrt_blender/test_*.py — currently color_fold
```

There is no `cargo test` suite. End-to-end render verification is by rendering test scenes:

```bash
make junk_shop-preview      # vibrt render of test_scenes/junk_shop/junk_shop/junk_shop.blend
make junk_shop-cycles       # Cycles reference render of the same .blend
make previews               # all three .blend scenes (junk_shop / classroom / bmw27)
make cycles-previews        # all three Cycles refs
```

`make` overrides: `SPP=`, `PCT=` (render resolution percentage), `DENOISE=1`. The .blend test scenes (junk_shop / classroom / bmw27) live under gitignored `test_scenes/<name>/<name>/<name>.blend` paths — they're large public-domain assets and not committed.

The preview pipeline spawns Blender headless (`scripts/render_blend.py` → `scripts/_blender_render.py`), force-loads the working-tree addon, and triggers a single F12 render via `vibrt_native.render(...)`. There is no `vibrt.exe` and no scene.json/scene.bin on disk.

## Architecture

This is a workspace with three Rust crates and a Python Blender addon. The data flow is **Blender → (json_str, bin_bytes, texture_arrays[]) → vibrt_native.render → RGBA float buffer → Combined pass**. Everything happens in-process inside Blender; PyO3 hands Python-owned buffers to Rust, which borrows from them for the duration of the render.

### Rust crates

- **`optix-sys/`** — bindgen FFI to OptiX 9.0.0. Pure unsafe.
- **`optix/`** — safe wrapper: RAII handles, builders (`ProgramGroup::hitgroup().closest_hit(...).build()`), typed enums, error/log surfacing. Re-exports `DeviceContext`, `Module`, `Pipeline`, `ProgramGroup`, SBT helpers, denoiser. The renderer uses this layer, never `optix-sys` directly.
- **`vibrt/`** — the renderer.
  - `[lib] name = "vibrt_native"`; `crate-type = ["rlib", "cdylib"]`. The cdylib lands at `target/release/vibrt_native.dll` and is staged as `blender/vibrt_blender/vibrt_native.pyd` so Blender imports it as `vibrt_blender.vibrt_native`.
  - Feature-gated `python` flag pulls in `pyo3` + `numpy`. Plain `cargo build --release` produces only the rlib (useful for type-checking iteration).
  - There is no `[[bin]]` section. The PyO3 module (`#[pymodule] fn vibrt_native`) is the only public entry point.

### `vibrt/src/` map

- **`scene_format.rs`** — `serde::Deserialize` schema for `scene.json`. The bin holds mesh / index / vertex-color / colour-graph LUT blobs referenced via `BlobRef { offset, len }`. Texture pixels do **not** live in the bin; `TextureDesc.array_index` indexes into a caller-supplied `Vec<PyBuffer<f32>>` (the `texture_arrays` argument to `render`). Schema versioned (`version: 1`); bumping requires updating both the exporter and `load_scene_from_bytes`.
- **`scene_loader.rs`** — `LoadedScene<'bin>` with `Cow<'bin, [f32]>` / `Cow<'bin, [u32]>` fields. Linear textures borrow zero-copy from the per-texture array; sRGB textures are owned + linearised via rayon (`par_chunks_mut`). Mesh attributes borrow from the bin via `bytemuck::try_cast_slice`. Displacement-perturbed vertices end up owned. Keep this borrowing structure intact when modifying — copying out of `Cow::Borrowed` defeats the FFI's whole point.
  - Single entry point: `load_scene_from_bytes(json_text: &str, bin: &[u8], texture_arrays: &[&[f32]]) -> LoadedScene<'a>`.
- **`render.rs`** — the GPU pipeline body. CUDA + OptiX context creation, NVRTC PTX compile, texture upload, accel-structure build, SBT, kernel launch, optional denoise, readback. Returns `RenderOutput { pixels, width, height }`.
  - Progress / cancellation goes through the `Progress` trait. `StdoutProgress` for ad-hoc CLI use cases (debug builds, future tooling); `PyProgress` (in `python.rs`) routes log lines to Blender's Info panel and polls Blender's `test_break` for Esc-cancel.
- **`pipeline.rs`** — small but load-bearing: `compile_ptx()` (NVRTC entry point with header-inlining hack so the `#include "devicecode.h"` doesn't need a real include search path at runtime), `generate_ggx_energy_lut()` for Kulla-Conty, `build_envmap_cdf()` for HDRI importance sampling.
- **Volumes** are homogeneous-only: `VolumeParams` lives on each `PrincipledMaterial` (mesh-bounded — junk_shop's `Smoke`, glass-with-fog hybrids) and on `SceneFile::world_volume` (atmospheric haze). Coefficients are precomputed (σ_t, σ_s, σ_e RGB plus HG anisotropy g) into a `VolumeGpu` block per-material; `PrincipledGpu.volume` is the device pointer (null = no volume). `volume_only=true` on a material marks the boundary mesh as invisible to surface shading. The path tracer keeps a fixed-depth `VolumeStack` (max 4) per ray; the world volume sits implicitly at index −1. Heterogeneous volumes (OpenVDB, 3D textures, density attributes) are out of scope until 3D-texture infrastructure exists; procedural drivers fall back to socket defaults with a one-time warning.
- **`devicecode.cu` / `devicecode.h`** — the OptiX device program: ray gen, miss, closest-hit, any-hit (transparent shadows). Compiled at runtime via NVRTC. `include_str!`-baked source — if a `.cu` edit doesn't seem to take effect, force a rebuild with `cargo clean -p vibrt`.
- **`principled.rs`** — host-side material upload + colour-graph (RGBCurve / HueSat / Mix etc.) compilation to the small VM the device code interprets. Pairs with `color_fold.rs` (Blender-side analogue lives in `material_export.py`).
- **`gpu_types.rs`** — POD structs that cross the host↔device boundary. Layout must match `devicecode.h`.
- **`python.rs`** — the only PyO3 surface: `render(scene_json, scene_bin, opts, log_cb=None, cancel_cb=None, texture_arrays=None)` returning a `(h, w, 4)` float32 ndarray. Releases the GIL via `py.allow_threads` for the entire render so CUDA driver threads don't deadlock against the interpreter; reacquires it briefly inside `PyProgress` callbacks. `scene_bin` is taken as `PyBuffer<u8>`, `texture_arrays` as `Vec<PyBuffer<f32>>` — both pin their Python source for the duration of the call.

### Python addon (`blender/vibrt_blender/`)

- **`engine.py`** — `VibrtRenderEngine(bpy.types.RenderEngine)`. F12 entry point. The addon renders **only** through the bundled `vibrt_native.pyd`; if the extension isn't importable the engine reports an error and stops. There is no subprocess fallback.
- **`exporter.py`** — the bulk of the export logic. Single entry point `export_scene_to_memory(depsgraph, texture_pct=None) -> (json_str, bytearray, list[ndarray])`. The `BinWriter` writes mesh / index / vertex-color / colour-graph LUT blobs into a `bytearray` via `memoryview` slice assignment (with auto-grow on overflow); textures go through `write_texture_pixels` and are parked into a separate per-texture array list. The bin runs at tens of MB even on heavy scenes; texture data (~12 GB on junk_shop) lives entirely in the per-texture array list and travels across PyO3 as `Vec<PyBuffer<f32>>`.
- **`material_export.py`** — Principled BSDF + node-graph compilation. Bakes RGBCurve/HueSat/Gamma/Invert/Clamp/ColorRamp/BrightContrast/Mix(MIX/MULTIPLY/ADD/SUBTRACT, constant side) into texture pixels; emits residual sequences as a small "colour graph" the device code interprets. Detects pure-emissive single-quad meshes and promotes them to area_rect lights so NEE can importance-sample them.
- **`color_fold.py`** — export-time constant folder for the JSON color graph. A node whose every input resolves to `{"type": "const"}` is evaluated host-side and rewritten in-place; the GPU interpreter sees a shorter chain. Op evaluators (mix/math/invert/hue_sat/rgb_curve/bright_contrast) mirror `devicecode.cu` branch-for-branch and are unit-tested in `test_color_fold.py` (no bpy dependency, runs under plain CPython via `make test`).
- **`hair_export.py`** — particle-system hair → ribbon mesh tessellation. (`rendered_child_count` is per-parent, not total — easy off-by-N if you forget.)
- **`runner.py`** — `find_native_module()` and `run_render_inproc()`. The latter is a thin wrapper over `vibrt_native.render(...)` that forwards `texture_arrays`.
- **`build_addon.py`** — packages `vibrt_blender.zip`. `--with-native` builds the cdylib via cargo and stages it as `vibrt_native.pyd`/`.so`/`.dylib`. `--stage-only` does the build + copy without zipping (used by `make dev-install` for the junctioned-addon workflow). A no-`--with-native` zip won't actually render — the addon errors out without the extension.

## Scene-format invariants worth knowing

- Right-handed, Z-up, metres throughout. Blender and vibrt agree, so there's **no axis conversion** — only matrix transposition (Blender 4×4 is column-major via mathutils; the JSON expects row-major).
- Camera looks down local −Z (Blender convention).
- Texture `colorspace` is either `"srgb"` or `"linear"`. Linearisation happens once at scene-load time on the host (rayon-parallel); the GPU sees only linear data.
- Mesh material indices are per-triangle (`MeshDesc::material_indices`), indexing into `ObjectDesc::materials` (the per-object slot list), not the global `materials` array directly.
- Area lights: Blender area lights emit along local −Z; vibrt's `area_rect` expects local +Z. The exporter flips the third column of the transform — don't introduce a second flip on either side.

## Memory / cost intuition

- `.blend` export → in-memory buffer is the dominant cost on heavy scenes. The big steady-state user is texture pixels (`image.pixels.foreach_get` → 16 bytes per pixel), not mesh data. Junk_shop's 142 textures sum to ~12 GB of f32 RGBA.
- Texture quantisation (f32 → fp16/u8) is the natural next optimisation; the current code path is correct but bandwidth-bound on large scenes.
- NVRTC compile is uninterruptible and adds ~1–2 s to the first render after Blender starts; subsequent renders re-pay it because there's no on-disk PTX cache. Don't add cancellation polls inside `pipeline::compile_ptx` — they won't help.

## Helper scripts (`scripts/`)

- **`render_blend.py`** — top-level entry point used by `make <scene>-preview`. Spawns Blender headless and runs `_blender_render.py` inside it.
- **`_blender_render.py`** — runs inside `blender -b … --python …`. Force-imports the working-tree addon, sets `render.engine = "VIBRT"`, calls `bpy.ops.render.render(write_still=True)`.
- **`_inproc_export_bench.py`** — micro-benchmark for `export_scene_to_memory`. Run via `blender -b <scene>.blend --python scripts/_inproc_export_bench.py` to get the per-bucket export breakdown plus an end-to-end render.
- **`render_cycles.py` / `_blender_cycles.py`** — Cycles reference renderer (no vibrt involved). Used for side-by-side comparison.
