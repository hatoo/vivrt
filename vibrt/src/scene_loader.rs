//! Load scene.json + scene.bin into host-side structures ready for GPU upload.

use crate::gpu_types::{AreaRectLight, PointLight, SpotLight, SunLight};
use crate::scene_format::*;
use crate::transform;
use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use std::borrow::Cow;

pub struct LoadedMesh<'bin> {
    /// f32 x 3 per vertex. Borrowed directly from the source bin when the
    /// mesh has no displacement; an owned `Vec<f32>` when displacement
    /// perturbation has to mutate the positions.
    pub vertices: Cow<'bin, [f32]>,
    pub normals: Cow<'bin, [f32]>,
    pub uvs: Cow<'bin, [f32]>,
    pub indices: Cow<'bin, [u32]>,
    /// u32 per triangle; empty when the mesh is single-material.
    pub material_indices: Cow<'bin, [u32]>,
    /// f32 x 3 per vertex; empty when the mesh has no vertex colour attribute
    /// referenced by any material. See `MeshDesc::vertex_colors`.
    pub vertex_colors: Cow<'bin, [f32]>,
    /// f32 x 3 per vertex; empty when the mesh has no authored tangent (the
    /// common case — only hair ribbon meshes ship one today). See
    /// `MeshDesc::tangents`.
    pub tangents: Cow<'bin, [f32]>,
}

pub struct LoadedTexture<'bin> {
    /// RGBA f32, width*height*4 floats, already linearised if colorspace ==
    /// "srgb". Linear textures (`colorspace == "linear"`, channels == 4)
    /// borrow directly from the bin — no host-side memcpy. sRGB textures get
    /// an owned `Vec<f32>` because the linearisation has to write into a
    /// fresh buffer; that pass folds the copy and the conversion together.
    pub data: Cow<'bin, [f32]>,
    pub width: u32,
    pub height: u32,
}

pub struct LoadedObject {
    pub mesh: u32,
    pub material: u32,
    pub transform: [f32; 12],
}

pub struct LoadedScene<'bin> {
    pub file: SceneFile,
    pub meshes: Vec<LoadedMesh<'bin>>,
    pub textures: Vec<LoadedTexture<'bin>>,
    pub objects: Vec<LoadedObject>,
    pub point_lights: Vec<PointLight>,
    pub sun_lights: Vec<SunLight>,
    pub spot_lights: Vec<SpotLight>,
    pub rect_lights: Vec<AreaRectLight>,
    /// Envmap texture fully materialised (linear RGB only, width*height*3 floats).
    /// `None` if world is Constant.
    pub envmap_rgb: Option<(Vec<f32>, u32, u32)>,
}

/// Parse an in-memory scene.json + scene.bin pair into a [`LoadedScene`].
///
/// Mesh and texture payloads are borrowed from `bin` wherever possible;
/// only sRGB textures and displacement-modified mesh vertices end up with
/// owned `Vec`s. The returned scene's lifetime is tied to `bin`'s.
///
/// `texture_arrays` is the optional list of caller-owned f32 slices that
/// `TextureDesc::array_index` references. The in-process FFI exporter
/// uses this to skip the bin's texture-concatenation memcpy: each
/// texture's pixel buffer is passed directly across PyO3 instead of
/// being re-stitched into a giant bytearray. The CLI / disk path passes
/// an empty slice — every texture there ships with a `pixels` BlobRef
/// pointing into the bin.
pub fn load_scene_from_bytes<'a>(
    json_text: &str,
    bin: &'a [u8],
    texture_arrays: &[&'a [f32]],
) -> Result<LoadedScene<'a>> {
    let file: SceneFile = serde_json::from_str(json_text).context("parsing scene.json")?;

    if file.version != 1 {
        return Err(anyhow!("unsupported scene.json version: {}", file.version));
    }

    // ~half of junk_shop's 142 textures are sRGB and pay a host-side
    // pow-2.4 conversion in `load_texture` (linear ones are zero-copy).
    // The conversion is the dominant scene-load cost on big scenes (~6 s
    // single-threaded). Each texture loads independently, so par_iter
    // straightforwardly buys an n-core speedup; rayon's work-stealing
    // handles the size imbalance between huge JPGs and tiny lookup maps.
    let textures = file
        .textures
        .par_iter()
        .map(|t| load_texture(t, texture_arrays))
        .collect::<Result<Vec<_>>>()?;
    let mut meshes = file
        .meshes
        .iter()
        .map(|m| load_mesh(m, bin))
        .collect::<Result<Vec<_>>>()?;
    for (mi, mdesc) in file.meshes.iter().enumerate() {
        if let (Some(tex_id), s) = (mdesc.displacement_tex, mdesc.displacement_strength) {
            if s != 0.0 {
                let tex = textures.get(tex_id as usize).ok_or_else(|| {
                    anyhow!("mesh {} displacement_tex index out of range", mi)
                })?;
                let m = &mut meshes[mi];
                if m.uvs.is_empty() {
                    eprintln!(
                        "[vibrt] warn: mesh {} has displacement_tex but no UVs \
                         — displacement skipped",
                        mi
                    );
                    continue;
                }
                if m.normals.is_empty() {
                    eprintln!(
                        "[vibrt] warn: mesh {} has displacement_tex but no \
                         normals — displacement skipped",
                        mi
                    );
                    continue;
                }
                apply_displacement(m, tex, s);
            }
        }
    }

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
                // Blender point light: total emitted power Φ W. The device samples
                // this as a delta point, so `emission` must be intensity I = Φ/(4π)
                // (W/sr) — the shader then computes irradiance as I/d². The old
                // formula Φ/(4π²r²) was the outgoing radiance on the sphere
                // surface, which only applies if we sample the surface (we don't);
                // using it as I made pendants ~1/(πr²) times too bright.
                let r = radius.max(1e-3);
                let coeff = power / (4.0 * std::f32::consts::PI);
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
                camera_visible,
                two_sided,
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
                    two_sided,
                    normal,
                    camera_visible,
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

/// Reinterpret the bin slice as `&[f32]` without copying. Falls back to a
/// fresh `Vec<f32>` only when the slice is misaligned (bytemuck panics
/// otherwise) — which never happens for blobs the exporter writes (BlobRefs
/// are always written at 16-byte-aligned offsets), but the fallback keeps
/// us safe against hand-authored test bins.
fn cast_f32_slice(bytes: &[u8]) -> Cow<'_, [f32]> {
    match bytemuck::try_cast_slice::<u8, f32>(bytes) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(read_f32_vec_owned(bytes)),
    }
}

fn cast_u32_slice(bytes: &[u8]) -> Cow<'_, [u32]> {
    match bytemuck::try_cast_slice::<u8, u32>(bytes) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(read_u32_vec_owned(bytes)),
    }
}

fn read_f32_vec_owned(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32_vec_owned(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn sample_heightmap(tex: &LoadedTexture, u: f32, v: f32) -> f32 {
    if tex.width == 0 || tex.height == 0 {
        return 0.0;
    }
    let w = tex.width as i32;
    let h = tex.height as i32;
    let uu = u - u.floor();
    let vv = v - v.floor();
    let fx = uu * w as f32 - 0.5;
    let fy = (1.0 - vv) * h as f32 - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let dx = fx - x0 as f32;
    let dy = fy - y0 as f32;
    let wrap = |v: i32, m: i32| -> i32 {
        let r = v.rem_euclid(m);
        r
    };
    let x0w = wrap(x0, w);
    let y0w = wrap(y0, h);
    let x1w = wrap(x0 + 1, w);
    let y1w = wrap(y0 + 1, h);
    let at = |x: i32, y: i32| -> f32 {
        let base = ((y as u32 * tex.width + x as u32) * 4) as usize;
        tex.data[base] // R channel
    };
    let c00 = at(x0w, y0w);
    let c10 = at(x1w, y0w);
    let c01 = at(x0w, y1w);
    let c11 = at(x1w, y1w);
    let c0 = c00 * (1.0 - dx) + c10 * dx;
    let c1 = c01 * (1.0 - dx) + c11 * dx;
    c0 * (1.0 - dy) + c1 * dy
}

fn apply_displacement(mesh: &mut LoadedMesh, tex: &LoadedTexture, strength: f32) {
    // Missing UVs/normals are caught by the caller with a warning; keep the
    // guard as a defensive no-op in case the invariant ever changes.
    if mesh.uvs.is_empty() || mesh.normals.is_empty() {
        return;
    }
    // Displacement perturbs vertex positions in place — promote the
    // (potentially borrowed) Cow to an owned Vec so we can mutate it. Other
    // attributes stay borrowed.
    let nv = mesh.vertices.len() / 3;
    let verts = mesh.vertices.to_mut();
    for i in 0..nv {
        let u = mesh.uvs[i * 2 + 0];
        let v = mesh.uvs[i * 2 + 1];
        let h = sample_heightmap(tex, u, v) * strength;
        verts[i * 3 + 0] += mesh.normals[i * 3 + 0] * h;
        verts[i * 3 + 1] += mesh.normals[i * 3 + 1] * h;
        verts[i * 3 + 2] += mesh.normals[i * 3 + 2] * h;
    }
}

fn load_mesh<'a>(desc: &MeshDesc, bin: &'a [u8]) -> Result<LoadedMesh<'a>> {
    let vertices = cast_f32_slice(slice_bin(bin, desc.vertices)?);
    let normals = match desc.normals {
        Some(b) => cast_f32_slice(slice_bin(bin, b)?),
        None => Cow::Owned(Vec::new()),
    };
    let uvs = match desc.uvs {
        Some(b) => cast_f32_slice(slice_bin(bin, b)?),
        None => Cow::Owned(Vec::new()),
    };
    let indices = cast_u32_slice(slice_bin(bin, desc.indices)?);
    let material_indices = match desc.material_indices {
        Some(b) => cast_u32_slice(slice_bin(bin, b)?),
        None => Cow::Owned(Vec::new()),
    };
    let vertex_colors = match desc.vertex_colors {
        Some(b) => cast_f32_slice(slice_bin(bin, b)?),
        None => Cow::Owned(Vec::new()),
    };
    let tangents = match desc.tangents {
        Some(b) => cast_f32_slice(slice_bin(bin, b)?),
        None => Cow::Owned(Vec::new()),
    };

    if indices.len() % 3 != 0 {
        return Err(anyhow!("mesh index count is not a multiple of 3"));
    }
    if vertices.len() % 3 != 0 {
        return Err(anyhow!("mesh vertex count is not a multiple of 3"));
    }
    let num_tris = indices.len() / 3;
    if !material_indices.is_empty() && material_indices.len() != num_tris {
        return Err(anyhow!(
            "mesh material_indices count {} != triangle count {}",
            material_indices.len(),
            num_tris
        ));
    }
    if !vertex_colors.is_empty() && vertex_colors.len() != vertices.len() {
        return Err(anyhow!(
            "mesh vertex_colors len {} != vertices len {} (expected f32x3 per vertex)",
            vertex_colors.len(),
            vertices.len()
        ));
    }
    if !tangents.is_empty() && tangents.len() != vertices.len() {
        return Err(anyhow!(
            "mesh tangents len {} != vertices len {} (expected f32x3 per vertex)",
            tangents.len(),
            vertices.len()
        ));
    }

    Ok(LoadedMesh {
        vertices,
        normals,
        uvs,
        indices,
        material_indices,
        vertex_colors,
        tangents,
    })
}

fn load_texture<'a>(
    desc: &TextureDesc,
    texture_arrays: &[&'a [f32]],
) -> Result<LoadedTexture<'a>> {
    // Texture pixels always travel as a caller-owned f32 slice (parked in
    // a `Vec<PyBuffer<f32>>` on the Python side). `array_index` picks the
    // right one — pixel data was removed from the bin.
    let arr = texture_arrays
        .get(desc.array_index as usize)
        .ok_or_else(|| {
            anyhow!(
                "texture.array_index = {} out of range (have {} arrays)",
                desc.array_index,
                texture_arrays.len()
            )
        })?;
    let src: Cow<'a, [f32]> = Cow::Borrowed(*arr);
    let expected = (desc.width * desc.height * desc.channels) as usize;
    if src.len() != expected {
        return Err(anyhow!(
            "texture pixel count mismatch: got {}, expected {} (w={} h={} c={})",
            src.len(),
            expected,
            desc.width,
            desc.height,
            desc.channels
        ));
    }

    let srgb = desc.colorspace.eq_ignore_ascii_case("srgb");
    let pixel_count = (desc.width * desc.height) as usize;

    let data: Cow<'a, [f32]> = match desc.channels {
        4 if !srgb => {
            // Linear 4-channel: borrow straight from the source (bin or
            // caller-supplied array). No host-side memcpy, no allocation.
            // This is the dominant cost saving on junk_shop-class scenes.
            src
        }
        4 => {
            // sRGB 4-channel: the linearisation has to write into a fresh
            // buffer regardless. Parallelise the per-pixel conversion across
            // a worker pool — a single 4K base-color JPG is 16 M pixels of
            // pow-2.4 and dominates load_scene on its own. par_chunks_mut
            // gives each worker a contiguous range so there's no false
            // sharing on the output buffer.
            let mut out: Vec<f32> = vec![0.0; src.len()];
            const CHUNK: usize = 1 << 16; // 64K floats = 16K pixels per task
            out.par_chunks_mut(CHUNK)
                .zip(src.par_chunks(CHUNK))
                .for_each(|(dst, s)| {
                    for (d, c) in dst.chunks_exact_mut(4).zip(s.chunks_exact(4)) {
                        d[0] = srgb_to_linear(c[0]);
                        d[1] = srgb_to_linear(c[1]);
                        d[2] = srgb_to_linear(c[2]);
                        d[3] = c[3];
                    }
                });
            Cow::Owned(out)
        }
        3 => {
            // 3-channel sources still require RGBA expansion. Blender
            // images from the addon are always 4-channel, so this path is
            // only hit by hand-written test scenes.
            let mut out = Vec::with_capacity(pixel_count * 4);
            for c in src.chunks_exact(3) {
                let r = if srgb { srgb_to_linear(c[0]) } else { c[0] };
                let g = if srgb { srgb_to_linear(c[1]) } else { c[1] };
                let b = if srgb { srgb_to_linear(c[2]) } else { c[2] };
                out.extend_from_slice(&[r, g, b, 1.0]);
            }
            Cow::Owned(out)
        }
        _ => {
            return Err(anyhow!(
                "unsupported texture channel count: {}",
                desc.channels
            ))
        }
    };

    Ok(LoadedTexture {
        data,
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
