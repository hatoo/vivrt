# vibrt

A Rust workspace for GPU ray tracing with NVIDIA OptiX 9.

## Crates

| Crate | Description |
|-------|-------------|
| [`optix-sys`](optix-sys/) | Raw FFI bindings to OptiX 9.0.0 (bindgen) |
| [`optix`](optix/) | Safe Rust wrapper with RAII, builders, and type-safe enums |
| [`pbrt-parser`](pbrt-parser/) | Zero-dependency parser for PBRTv4 scene files |
| [`vibrt`](vibrt/) | OptiX path tracing renderer for PBRTv4 scenes |

## vibrt renderer

A GPU path tracer that reads [PBRTv4](https://pbrt.org/fileformat-v4) scene files and renders them using OptiX hardware ray tracing.

### Features

- Path tracing with configurable depth and samples per pixel
- Materials: diffuse, coated diffuse, conductor, coated conductor, dielectric, mix (stochastic, nested)
- GGX microfacet BRDF with anisotropic roughness, VNDF importance sampling, Kulla-Conty energy compensation
- Exact conductor Fresnel (complex IOR eta+ik), coated conductor with multi-layer scattering
- Geometry: triangle meshes, PLY meshes (binary, gzip), spheres (built-in intersection), bilinear patches, Loop subdivision
- Lighting: distant, infinite (envmap with importance sampling), sphere area lights, triangle area lights with power-weighted two-level NEE
- Textures: imagemap (bilinear), bump map, normal map, roughness map, alpha cutout, checkerboard, planar UV mapping
- Mix and directionmix texture blending
- sRGB-aware texture loading
- CUDA device code compiled at runtime via NVRTC

### Usage

```bash
cargo run --release -p vibrt -- scene.pbrt
cargo run --release -p vibrt -- scene.pbrt --spp 64 --width 800 --height 600
```

### Example renders

![crown](crown.png)

```bash
# Simple glass sphere on checkerboard
cargo run --release -p vibrt -- test.pbrt

# Killeroo scene (coated diffuse + area lights)
cargo run --release -p vibrt -- path/to/killeroos/killeroo-simple.pbrt --spp 64

# Crown scene (conductors, dielectrics, textures, 793 objects)
cargo run --release -p vibrt -- path/to/crown/crown.pbrt --spp 32 --width 500 --height 700
```

## Requirements

- **NVIDIA OptiX SDK 9.0.0** -- set `OPTIX_ROOT` or install at default location
- **CUDA Toolkit** -- for NVRTC runtime compilation
- **LLVM/Clang** -- for bindgen (set `LIBCLANG_PATH` if not auto-detected)
- **NVIDIA GPU** with driver supporting OptiX 9

## Building

```bash
cargo build --release -p vibrt
```
