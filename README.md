# vibrt

A GPU path-tracing renderer for Blender, powered by NVIDIA OptiX 9.

## Crates

| Crate | Description |
|-------|-------------|
| [`optix-sys`](optix-sys/) | Raw FFI bindings to OptiX 9.0.0 (bindgen) |
| [`optix`](optix/) | Safe Rust wrapper with RAII, builders, and type-safe enums |
| [`vibrt`](vibrt/) | OptiX path-tracing renderer with a Blender-native JSON+binary input format |

The [`blender/vibrt_blender/`](blender/) directory contains the Blender addon that registers `vibrt` as a render engine, exports the scene, spawns the renderer, and loads the result back into Blender.

## Features

- Path tracing with NEE + MIS for point / sun / spot / area-rect lights and HDRI envmap (importance sampling)
- Transparent shadow rays through transmissive / alpha-cutout surfaces
- Principled BSDF: base color, metallic, roughness, IOR, transmission, emission, anisotropy, coat, sheen, subsurface, alpha cutout, displacement, bump, with image textures for base color / normal / roughness / metallic
- GGX multi-scattering energy compensation (Kulla-Conty)
- Indirect-contribution luminance clamp to suppress fireflies
- Triangle mesh geometry with multi-material slots (per-triangle material index); multi-object scenes via single-level OptiX instancing
- Right-handed Z-up (no axis conversion from Blender)
- Output: EXR (linear RGBA float), PNG (tonemapped sRGB), or raw float binary (addon ingestion)

## Requirements

- **NVIDIA OptiX SDK 9.0.0** — set `OPTIX_ROOT` or install at default location
- **CUDA Toolkit** — for NVRTC runtime compilation
- **LLVM/Clang** — for bindgen (set `LIBCLANG_PATH` if not auto-detected)
- **NVIDIA GPU** with driver supporting OptiX 9
- **Blender 4.0+** (tested on 5.1) — for the addon

## Building

```bash
cargo build --release -p vibrt
```

## Usage

### CLI (standalone)

```bash
cargo run --release -p vibrt -- scene.json --spp 128 --output render.exr
```

Common flags: `--spp`, `--depth`, `--clamp-indirect`, `--width`, `--height`. See [`vibrt/src/main.rs`](vibrt/src/main.rs) for the full list.

### Rendering `.blend` files from the CLI

[`scripts/render_blend.py`](scripts/render_blend.py) spawns Blender headlessly to export a `.blend` via the addon's exporter, then runs `vibrt` on the result:

```bash
py scripts/render_blend.py scene.blend --output render.png --spp 128
```

### Blender addon

See [`blender/README.md`](blender/README.md) for installation and usage.

### Test scenes & Makefile

Hand-built scenes live in [`test_scenes/`](test_scenes/) (`cornell`, `furnace`, `veach_mis`, `classroom`). The top-level [`Makefile`](Makefile) regenerates scenes and previews:

```bash
make                      # regenerate every test_scenes/*/scene.json
make veach_mis-preview    # render a preview for one scene
make previews             # render previews for all scenes (incl. .blend-based)
make addon                # build blender/vibrt_blender.zip
make dev-install          # junction the addon into Blender's user addons dir
```
