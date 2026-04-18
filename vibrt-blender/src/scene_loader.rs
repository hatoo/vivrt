//! Load scene.json + scene.bin into host-side structures ready for GPU upload.

use crate::gpu_types::{AreaRectLight, PointLight, SpotLight, SunLight};
use crate::scene_format::*;
use crate::transform;
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

pub struct LoadedMesh {
    pub vertices: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub indices: Vec<u32>,
}

pub struct LoadedTexture {
    /// RGBA f32, width*height*4 floats, already linearised if colorspace == "srgb".
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

pub struct LoadedObject {
    pub mesh: u32,
    pub material: u32,
    pub transform: [f32; 12],
}

pub struct LoadedScene {
    pub file: SceneFile,
    pub meshes: Vec<LoadedMesh>,
    pub textures: Vec<LoadedTexture>,
    pub objects: Vec<LoadedObject>,
    pub point_lights: Vec<PointLight>,
    pub sun_lights: Vec<SunLight>,
    pub spot_lights: Vec<SpotLight>,
    pub rect_lights: Vec<AreaRectLight>,
    /// Envmap texture fully materialised (linear RGB only, width*height*3 floats).
    /// `None` if world is Constant.
    pub envmap_rgb: Option<(Vec<f32>, u32, u32)>,
}

pub fn load_scene(json_path: &Path) -> Result<LoadedScene> {
    let json_text = std::fs::read_to_string(json_path)
        .with_context(|| format!("reading {}", json_path.display()))?;
    let file: SceneFile = serde_json::from_str(&json_text).context("parsing scene.json")?;

    if file.version != 1 {
        return Err(anyhow!("unsupported scene.json version: {}", file.version));
    }

    let scene_dir: PathBuf = json_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let binary_path = scene_dir.join(&file.binary);
    let binary_file = std::fs::File::open(&binary_path)
        .with_context(|| format!("opening {}", binary_path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&binary_file) }.context("mmap of scene.bin failed")?;
    let bin: &[u8] = &mmap;

    let meshes = file
        .meshes
        .iter()
        .map(|m| load_mesh(m, bin))
        .collect::<Result<Vec<_>>>()?;
    let textures = file
        .textures
        .iter()
        .map(|t| load_texture(t, bin))
        .collect::<Result<Vec<_>>>()?;

    let objects = file
        .objects
        .iter()
        .map(|o| LoadedObject {
            mesh: o.mesh,
            material: o.material,
            transform: transform::from_4x4_row_major(&o.transform),
        })
        .collect();

    let mut point_lights = Vec::new();
    let mut sun_lights = Vec::new();
    let mut spot_lights = Vec::new();
    let mut rect_lights = Vec::new();
    for light in &file.lights {
        match *light {
            LightDesc::Point {
                position,
                color,
                power,
                radius,
            } => {
                // Blender point light: total emitted power W. Convert to radiance on sphere surface.
                // L = power / (4π² * r²) (uniform radiance on sphere, integrates to 4π * πr² * L = 4π²r²L = power)
                let r = radius.max(1e-3);
                let coeff = power / (4.0 * std::f32::consts::PI * std::f32::consts::PI * r * r);
                let emission = [color[0] * coeff, color[1] * coeff, color[2] * coeff];
                point_lights.push(PointLight {
                    position,
                    radius: r,
                    emission,
                    _pad: 0.0,
                });
            }
            LightDesc::Sun {
                direction,
                color,
                strength,
                angle_rad,
            } => {
                let dir = normalize3(direction);
                // Point TO light (same convention as existing renderer's DistantLight).
                let towards = [-dir[0], -dir[1], -dir[2]];
                let emission = [
                    color[0] * strength,
                    color[1] * strength,
                    color[2] * strength,
                ];
                sun_lights.push(SunLight {
                    direction: towards,
                    cos_angle: angle_rad.cos().max(0.0),
                    emission,
                    _pad: 0.0,
                });
            }
            LightDesc::Spot {
                transform: t4,
                color,
                power,
                cone_rad,
                blend,
            } => {
                let t = transform::from_4x4_row_major(&t4);
                let position = [t[3], t[7], t[11]];
                // Blender spot: local -Z is emission direction.
                let dir = transform::transform_dir(&t, [0.0, 0.0, -1.0]);
                let dir = normalize3(dir);
                let cos_outer = cone_rad.cos();
                let cos_inner = (cone_rad * (1.0 - blend)).cos();
                // Approx radiance from power: point-like with directional falloff.
                let solid_angle = 2.0 * std::f32::consts::PI * (1.0 - cos_outer).max(1e-4);
                let coeff = power / solid_angle;
                let emission = [color[0] * coeff, color[1] * coeff, color[2] * coeff];
                spot_lights.push(SpotLight {
                    position,
                    _pad0: 0.0,
                    direction: dir,
                    cos_outer,
                    emission,
                    cos_inner,
                });
            }
            LightDesc::AreaRect {
                transform: t4,
                size,
                color,
                power,
            } => {
                let t = transform::from_4x4_row_major(&t4);
                // Local rectangle lies in XY with +Z normal, centred at origin.
                let center = [t[3], t[7], t[11]];
                let u_axis = normalize3(transform::transform_dir(&t, [1.0, 0.0, 0.0]));
                let v_axis = normalize3(transform::transform_dir(&t, [0.0, 1.0, 0.0]));
                let normal = normalize3(transform::transform_dir(&t, [0.0, 0.0, 1.0]));
                let corner = [
                    center[0] - 0.5 * size[0] * u_axis[0] - 0.5 * size[1] * v_axis[0],
                    center[1] - 0.5 * size[0] * u_axis[1] - 0.5 * size[1] * v_axis[1],
                    center[2] - 0.5 * size[0] * u_axis[2] - 0.5 * size[1] * v_axis[2],
                ];
                let area = (size[0] * size[1]).max(1e-6);
                let coeff = power / (area * std::f32::consts::PI);
                let emission = [color[0] * coeff, color[1] * coeff, color[2] * coeff];
                rect_lights.push(AreaRectLight {
                    corner,
                    size_u: size[0],
                    u_axis,
                    size_v: size[1],
                    v_axis,
                    _pad0: 0.0,
                    normal,
                    _pad1: 0.0,
                    emission,
                    power,
                });
            }
        }
    }

    // Envmap: extract RGB (3 channels) from textures if world is an envmap.
    let envmap_rgb = if let Some(WorldDesc::Envmap { texture, .. }) = &file.world {
        let idx = *texture as usize;
        let tex = textures
            .get(idx)
            .ok_or_else(|| anyhow!("envmap texture index out of range: {}", idx))?;
        let mut rgb = Vec::with_capacity((tex.width * tex.height * 3) as usize);
        for chunk in tex.data.chunks_exact(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }
        Some((rgb, tex.width, tex.height))
    } else {
        None
    };

    Ok(LoadedScene {
        file,
        meshes,
        textures,
        objects,
        point_lights,
        sun_lights,
        spot_lights,
        rect_lights,
        envmap_rgb,
    })
}

fn slice_bin(bin: &[u8], blob: BlobRef) -> Result<&[u8]> {
    let start = blob.offset as usize;
    let end = start
        .checked_add(blob.len as usize)
        .ok_or_else(|| anyhow!("blob offset+len overflow"))?;
    if end > bin.len() {
        return Err(anyhow!(
            "blob out of range: offset={} len={} bin_len={}",
            blob.offset,
            blob.len,
            bin.len()
        ));
    }
    Ok(&bin[start..end])
}

fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]));
    }
    out
}

fn read_u32_vec(bytes: &[u8]) -> Vec<u32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]));
    }
    out
}

fn load_mesh(desc: &MeshDesc, bin: &[u8]) -> Result<LoadedMesh> {
    let vertices = read_f32_vec(slice_bin(bin, desc.vertices)?);
    let normals = match desc.normals {
        Some(b) => read_f32_vec(slice_bin(bin, b)?),
        None => Vec::new(),
    };
    let uvs = match desc.uvs {
        Some(b) => read_f32_vec(slice_bin(bin, b)?),
        None => Vec::new(),
    };
    let indices = read_u32_vec(slice_bin(bin, desc.indices)?);

    if indices.len() % 3 != 0 {
        return Err(anyhow!("mesh index count is not a multiple of 3"));
    }
    if vertices.len() % 3 != 0 {
        return Err(anyhow!("mesh vertex count is not a multiple of 3"));
    }

    Ok(LoadedMesh {
        vertices,
        normals,
        uvs,
        indices,
    })
}

fn load_texture(desc: &TextureDesc, bin: &[u8]) -> Result<LoadedTexture> {
    let bytes = slice_bin(bin, desc.pixels)?;
    let raw = read_f32_vec(bytes);
    let expected = (desc.width * desc.height * desc.channels) as usize;
    if raw.len() != expected {
        return Err(anyhow!(
            "texture pixel count mismatch: got {}, expected {} (w={} h={} c={})",
            raw.len(),
            expected,
            desc.width,
            desc.height,
            desc.channels
        ));
    }

    let srgb = desc.colorspace.eq_ignore_ascii_case("srgb");
    let mut out = Vec::with_capacity((desc.width * desc.height * 4) as usize);
    match desc.channels {
        3 => {
            for c in raw.chunks_exact(3) {
                let r = if srgb { srgb_to_linear(c[0]) } else { c[0] };
                let g = if srgb { srgb_to_linear(c[1]) } else { c[1] };
                let b = if srgb { srgb_to_linear(c[2]) } else { c[2] };
                out.extend_from_slice(&[r, g, b, 1.0]);
            }
        }
        4 => {
            for c in raw.chunks_exact(4) {
                let r = if srgb { srgb_to_linear(c[0]) } else { c[0] };
                let g = if srgb { srgb_to_linear(c[1]) } else { c[1] };
                let b = if srgb { srgb_to_linear(c[2]) } else { c[2] };
                out.extend_from_slice(&[r, g, b, c[3]]);
            }
        }
        _ => {
            return Err(anyhow!(
                "unsupported texture channel count: {}",
                desc.channels
            ))
        }
    }

    Ok(LoadedTexture {
        data: out,
        width: desc.width,
        height: desc.height,
    })
}

fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-20 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}
