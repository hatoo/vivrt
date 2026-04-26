# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / run

Toolchain prerequisites: NVIDIA OptiX SDK 9.0.0 (`OPTIX_ROOT`), CUDA Toolkit (NVRTC), LLVM/Clang (`LIBCLANG_PATH` — already set in `.cargo/config.toml` to `C:\Program Files\LLVM\bin`), and an OptiX-capable NVIDIA GPU.

```bash
cargo build --release -p vibrt                               # CLI binary only
cargo build --release -p vibrt --features python             # also build target/release/vibrt_native.dll (PyO3 ext)
cargo run   --release -p vibrt -- scene.json --spp 128 --output out.exr
cargo run   --release -p vibrt -- --compile-only             # NVRTC sanity-check device code without rendering
```

There is no `cargo test` suite in this repo. Verification is done by re-running scenes and comparing output: regenerate scene JSON with `make`, then `make <scene>-preview`. For diffs against Cycles, use `make <scene>-cycles`. Render-correctness changes should be verified on `cornell` (small, fast), `veach_mis` (light-sampling stress), and one of the heavier `.blend` scenes (`classroom`, `bmw27`, `junk_shop`).

The Makefile is the source of truth for common workflows — see its header for full usage. Useful targets:

```bash
make                        # regenerate every test_scenes/*/scene.json from make_scene.py
make <scene>                # regenerate one
make <scene>-preview        # render with vibrt
make <scene>-cycles         # render the same .blend with Cycles for reference
make addon                  # blender/vibrt_blender.zip (Python sources only)
make addon-with-native      # also `cargo build --features python` and bundle the .pyd
make dev-install            # build .pyd, stage next to addon source, junction into Blender's user-addons dir
make clean                  # scene.json, scene.bin, preview.png, addon zip, staged .pyd
```

Override `SPP=`, `PCT=` (resolution percentage for `.blend` scenes), `TEXTURE_PCT=`, `DENOISE=1` on the make command line. `PYTHON` defaults to `py`; `VIBRT` defaults to `./target/release/vibrt.exe`.

## Architecture

This is a workspace with three Rust crates and a Python addon. The data flow is **Blender → JSON+binary scene format → vibrt renderer → RGBA float buffer → image / Blender pixel pass**. The same renderer code services both the standalone CLI and the in-Blender F12 path; the only difference is whether the scene buffer arrives via `fs::read` or via PyO3 from a `bytearray`.

### Rust crates

- **`optix-sys/`** — bindgen FFI to OptiX 9.0.0. Pure unsafe.
- **`optix/`** — safe wrapper: RAII handles, builders (`ProgramGroup::hitgroup().closest_hit(...).build()`), typed enums, error/log surfacing. Re-exports `DeviceContext`, `Module`, `Pipeline`, `ProgramGroup`, SBT helpers, denoiser. The renderer uses this layer, never `optix-sys` directly.
- **`vibrt/`** — the renderer. Hybrid `bin + cdylib + rlib`:
  - `[lib] name = "vibrt_native"` so the cdylib filename matches the Python extension's import name (`target/release/vibrt_native.dll` → renamed to `vibrt_native.pyd` and dropped next to the addon's `__init__.py`). Cargo `.pdb` collision (#6313) was the secondary motivation.
  - Feature-gated `python` flag pulls in `pyo3` + `numpy`. Plain `cargo build --release` produces only the CLI; the Blender addon falls back to the subprocess path when the `.pyd` isn't bundled.
  - `src/main.rs` is a ~80-line CLI shim; the real work is `src/lib.rs` → `render::render_to_pixels`.

### `vibrt/src/` map

- **`scene_format.rs`** — `serde::Deserialize` schema for `scene.json`. The bin file is opaque; everything in `scene.json` references it via `BlobRef { offset, len }`. Schema versioned (`version: 1`); bumping requires updating both the exporter and `load_scene_from_bytes`.
- **`scene_loader.rs`** — `LoadedScene<'bin>` with `Cow<'bin, [f32]>` / `Cow<'bin, [u32]>` fields. Linear textures and unmodified mesh attributes **borrow zero-copy** from the input bin via `bytemuck::try_cast_slice`. sRGB textures are owned + linearised via rayon (`par_chunks_mut`). Displacement-perturbed mesh vertices are owned. Keep this borrowing structure intact when modifying — copying out of `Cow::Borrowed` defeats the in-process FFI's whole point.
  - Two entry points: `load_scene_from_path(json_path, &mut json_text, &mut bin)` for the CLI, `load_scene_from_bytes(json_str, bin)` for the Python path. The CLI variant exists only because `LoadedScene` borrows from caller-owned storage.
- **`render.rs`** — the GPU pipeline body. CUDA + OptiX context creation, NVRTC PTX compile (cached at the cudarc layer? no — it's recompiled every render), texture upload, accel-structure build, SBT, kernel launch, optional denoise, readback. Returns `RenderOutput { pixels, width, height }`. The CLI calls `image_io::save_image`; the Python path returns the buffer to numpy.
  - Progress / cancellation goes through the `Progress` trait. `StdoutProgress` for the CLI; `PyProgress` (in `python.rs`) routes log lines to Blender's Info panel and polls Blender's `test_break` for Esc-cancel.
- **`pipeline.rs`** — small but load-bearing: `compile_ptx()` (NVRTC entry point with header-inlining hack so the `#include "devicecode.h"` doesn't need a real include search path at runtime), `generate_ggx_energy_lut()` for Kulla-Conty, `build_envmap_cdf()` for HDRI importance sampling.
- **`devicecode.cu` / `devicecode.h`** — the OptiX device program: ray gen, miss, closest-hit, any-hit (transparent shadows). Compiled at runtime via NVRTC. **Modifying these requires a rebuild of vibrt only after a `cargo clean -p vibrt`** if the `include_str!`-baked source changes don't propagate (cargo doesn't track `include_str!` deps reliably across all workflows — if a `.cu` edit doesn't seem to take effect, force a rebuild).
- **`principled.rs`** — host-side material upload + colour-graph (RGBCurve / HueSat / Mix etc.) compilation to the small VM the device code interprets. Pairs with `color_fold.rs` (Blender-side analogue lives in `material_export.py`).
- **`gpu_types.rs`** — POD structs that cross the host↔device boundary. Layout must match `devicecode.h`.
- **`python.rs`** — the only PyO3 surface: a single `render(scene_json, scene_bin, opts, log_cb=None, cancel_cb=None)` function returning a `(h, w, 4)` float32 ndarray. Releases the GIL via `py.allow_threads` for the entire render so CUDA driver threads don't deadlock against the interpreter; reacquires it briefly inside `PyProgress` callbacks. `scene_bin` is taken as `PyBuffer<u8>` so `bytes`/`bytearray`/`memoryview` all work without a finalisation copy.

### Python addon (`blender/vibrt_blender/`)

- **`engine.py`** — `VibrtRenderEngine(bpy.types.RenderEngine)`. F12 entry point. Tries `runner.find_native_module()` first; on success calls the in-process path (no temp files, no subprocess). On `ImportError` or any in-process exception (except `KeyboardInterrupt`), falls back to writing `scene.json`/`scene.bin` to `bpy.app.tempdir` and spawning `vibrt.exe`. Pixels go into the Combined pass via `combined.rect.foreach_set`.
- **`exporter.py`** — the bulk of the export logic. `_export_into(depsgraph, writer, ...)` is the shared core; `export_scene(...)` writes to disk for the CLI tooling, `export_scene_to_memory(...)` returns `(json_str, bytearray)` for the in-process path. `BinWriter` accepts three sink types (file via `tofile`, `bytearray` via memoryview slice — pre-sized for the in-memory case, `BytesIO` as fallback). All blob writes return `{offset, len}` BlobRefs.
- **`material_export.py`** — Principled BSDF + node-graph compilation. Bakes RGBCurve/HueSat/Gamma/Invert/Clamp/ColorRamp/BrightContrast/Mix(MIX/MULTIPLY/ADD/SUBTRACT, constant side) into texture pixels; emits residual sequences as a small "colour graph" the device code interprets. Detects pure-emissive single-quad meshes and promotes them to area_rect lights so NEE can importance-sample them.
- **`hair_export.py`** — particle-system hair → ribbon mesh tessellation. (`rendered_child_count` is per-parent, not total — easy off-by-N if you forget.)
- **`runner.py`** — both render paths: `run_render_inproc` calls `vibrt_native.render`; `run_render` spawns `vibrt.exe` and pipes stdout into the Info panel.
- **`build_addon.py`** — packages `vibrt_blender.zip`. `--with-native` builds the cdylib via cargo and stages it as `vibrt_native.pyd`/`.so`/`.dylib`. `--stage-only` does the build + copy without zipping (used by `make dev-install` for the junctioned-addon workflow).

### CLI vs in-process: why both exist

The disk roundtrip is the bottleneck on heavy scenes: `junk_shop`'s `scene.bin` is ~12 GB of float32 RGBA texture pixels. The CLI path pays write+read+decode (~15 s on `junk_shop`); the in-process path keeps the buffer in Python's heap, hands a `&[u8]` pointer to Rust via PyBuffer, and `LoadedScene` borrows from it. Linear textures and mesh attributes are zero-copy from that point on; only sRGB textures and displacement-perturbed vertices ever allocate. **Both paths must remain functional** — the CLI is used by `scripts/render_blend.py` and the Makefile, and the in-process path falls through to subprocess if the `.pyd` isn't bundled.

## Scene-format invariants worth knowing

- Right-handed, Z-up, metres throughout. Blender and vibrt agree, so there's **no axis conversion** — only matrix transposition (Blender 4×4 is column-major via mathutils; the JSON expects row-major).
- Camera looks down local −Z (Blender convention).
- Texture `colorspace` is either `"srgb"` or `"linear"`. Linearisation happens once at scene-load time on the host; the GPU sees only linear data.
- Mesh material indices are per-triangle (`MeshDesc::material_indices`), indexing into `ObjectDesc::materials` (the per-object slot list), not the global `materials` array directly.
- Area lights: Blender area lights emit along local −Z; vibrt's `area_rect` expects local +Z. The exporter flips the third column of the transform — don't introduce a second flip on either side.

## Memory / cost intuition

- `.blend` export → in-memory buffer is the dominant cost on heavy scenes. The big steady-state user is texture pixels (`image.pixels.foreach_get` → 16 bytes per pixel), not mesh data.
- Texture quantisation (f32 → fp16/u8) is the natural next optimisation; the current code path is correct but bandwidth-bound on large scenes.
- NVRTC compile is uninterruptible and adds ~1–2 s to the first render after Blender starts; subsequent renders re-pay it because there's no on-disk PTX cache. Don't add cancellation polls inside `pipeline::compile_ptx` — they won't help.
