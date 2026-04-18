# vibrt-blender

A GPU path-tracing renderer for Blender, powered by NVIDIA OptiX 9.

## Crates

| Crate | Description |
|-------|-------------|
| [`optix-sys`](optix-sys/) | Raw FFI bindings to OptiX 9.0.0 (bindgen) |
| [`optix`](optix/) | Safe Rust wrapper with RAII, builders, and type-safe enums |
| [`vibrt-blender`](vibrt-blender/) | OptiX path-tracing renderer with a Blender-native JSON+binary input format |

The [`blender/vibrt_blender/`](blender/) directory contains the Blender addon that registers `vibrt-blender` as a render engine, exports the scene, spawns the renderer, and loads the result back into Blender.

## Features

- Path tracing with NEE for point / sun / spot / area-rect lights and HDRI envmap (importance sampling)
- Principled BSDF (base color, metallic, roughness, IOR, transmission, emission) with image textures for base color / normal / roughness / metallic
- Triangle mesh geometry, multi-object scenes via single-level OptiX instancing
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
cargo build --release -p vibrt-blender
```

## Usage

### CLI (standalone)

```bash
cargo run --release -p vibrt-blender -- scene.json --spp 128 --output render.exr
```

See [`test_scenes/cornell/`](test_scenes/cornell/) for a hand-built Cornell-box example.

### Blender addon

See [`blender/README.md`](blender/README.md) for installation and usage.
