//! Host-side material upload for Principled BSDF.

use crate::gpu_types::{
    ColorGraphNode, PrincipledGpu, VolumeGpu, COLOR_NODE_BRIGHT_CONTRAST, COLOR_NODE_CONST,
    COLOR_NODE_HUE_SAT, COLOR_NODE_IMAGE_TEX, COLOR_NODE_INVERT, COLOR_NODE_MATH, COLOR_NODE_MIX,
    COLOR_NODE_RGB_CURVE, COLOR_NODE_VERTEX_COLOR,
};
use crate::scene_format::{ColorFactor, ColorGraph, ColorNode, PrincipledMaterial, VolumeParams};
use crate::scene_loader::LoadedTexture;
use crate::CudaResultExt;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

fn dptr(slice: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

fn dptr_u32(slice: &CudaSlice<u32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

/// Upload all textures once; returns parallel vector of device slices + dims.
pub fn upload_textures(
    textures: &[LoadedTexture],
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<CudaSlice<f32>>,
) -> Result<Vec<(optix_sys::CUdeviceptr, i32, i32)>> {
    let mut out = Vec::with_capacity(textures.len());
    for tex in textures {
        let slice = stream.clone_htod(&tex.data[..]).cuda()?;
        let ptr = dptr(&slice, stream);
        out.push((ptr, tex.width as i32, tex.height as i32));
        bufs.push(slice);
    }
    Ok(out)
}

/// Info returned by `upload_color_graph` so the caller can stitch it into
/// `PrincipledGpu` (nodes pointer, length, output index).
#[derive(Copy, Clone, Default)]
pub struct ColorGraphGpu {
    pub nodes: optix_sys::CUdeviceptr,
    pub len: i32,
    pub output: i32,
}

/// Flatten a `ColorGraph` into the on-device `ColorGraphNode` layout and
/// upload it. Returns a descriptor the material upload can splice in.
pub fn upload_color_graph(
    graph: &ColorGraph,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    stream: &Arc<CudaStream>,
    bufs: &mut Vec<CudaSlice<u32>>,
    f_bufs: &mut Vec<CudaSlice<f32>>,
) -> Result<ColorGraphGpu> {
    // Constant subgraphs were folded to `const` nodes by the Python exporter
    // (`vibrt_blender/color_fold.py`) before serialisation. The graph
    // arriving here already has them collapsed.
    if graph.nodes.len() > 255 {
        return Err(anyhow!(
            "color_graph has {} nodes (max 255)",
            graph.nodes.len()
        ));
    }
    // RGBCurve nodes carry their LUT inline in JSON; allocate them per-node
    // on the GPU so `flatten_color_graph` can splice the device pointer
    // into the payload.
    let mut lut_ptrs: Vec<optix_sys::CUdeviceptr> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        if let ColorNode::RgbCurve { lut, .. } = node {
            if lut.len() != 768 {
                return Err(anyhow!(
                    "color_graph RgbCurve.lut has {} entries (want 768)",
                    lut.len()
                ));
            }
            let slice = stream.clone_htod(lut).cuda()?;
            let ptr = {
                let (p, _sync) = slice.device_ptr(stream);
                p as optix_sys::CUdeviceptr
            };
            f_bufs.push(slice);
            lut_ptrs.push(ptr);
        } else {
            lut_ptrs.push(0);
        }
    }
    let flat = flatten_color_graph(graph, textures, &lut_ptrs)?;
    // ColorGraphNode is repr(C) + 16 × u32 with no padding (see the
    // compile-time size/align asserts in `gpu_types.rs`), so reinterpreting
    // the buffer as `&[u32]` is sound and lets us reuse the existing
    // CudaSlice<u32> storage in `bufs`.
    let flat_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(flat.as_ptr() as *const u32, flat.len() * 16)
    };
    let slice = stream.clone_htod(flat_u32).cuda()?;
    let ptr = dptr_u32(&slice, stream);
    bufs.push(slice);
    let len = graph.nodes.len() as i32;
    let output = match graph.output {
        Some(o) => o as i32,
        None => len - 1,
    };
    if output < 0 || output >= len {
        return Err(anyhow!(
            "color_graph.output = {} out of range [0, {})",
            output,
            len
        ));
    }
    Ok(ColorGraphGpu {
        nodes: ptr,
        len,
        output,
    })
}

fn flatten_color_graph(
    graph: &ColorGraph,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    lut_ptrs: &[optix_sys::CUdeviceptr],
) -> Result<Vec<ColorGraphNode>> {
    let mut out: Vec<ColorGraphNode> = Vec::with_capacity(graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        let mut record = ColorGraphNode {
            tag: 0,
            payload: [0u32; 15],
        };
        flatten_one(node, i, textures, lut_ptrs, &mut record)?;
        out.push(record);
    }
    Ok(out)
}

fn flatten_one(
    node: &ColorNode,
    self_idx: usize,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    lut_ptrs: &[optix_sys::CUdeviceptr],
    out: &mut ColorGraphNode,
) -> Result<()> {
    let check_ref = |r: u32, field: &str| -> Result<()> {
        if (r as usize) >= self_idx {
            Err(anyhow!(
                "color_graph node {} {}: references node {} >= self",
                self_idx,
                field,
                r
            ))
        } else {
            Ok(())
        }
    };
    let p = &mut out.payload;
    match node {
        ColorNode::Const { rgb } => {
            out.tag = COLOR_NODE_CONST;
            p[0] = rgb[0].to_bits();
            p[1] = rgb[1].to_bits();
            p[2] = rgb[2].to_bits();
        }
        ColorNode::ImageTex { tex, uv } => {
            let (ptr, w, h) = *textures.get(*tex as usize).ok_or_else(|| {
                anyhow!("color_graph ImageTex.tex = {} out of range", tex)
            })?;
            out.tag = COLOR_NODE_IMAGE_TEX;
            p[0] = ptr as u32;
            p[1] = (ptr >> 32) as u32;
            p[2] = w as u32;
            p[3] = h as u32;
            p[4] = 4; // RGBA f32 — every loaded texture is padded to 4 channels
            for k in 0..6 {
                p[5 + k] = uv[k].to_bits();
            }
        }
        ColorNode::Mix {
            a,
            b,
            fac,
            blend,
            clamp,
        } => {
            check_ref(*a, "a")?;
            check_ref(*b, "b")?;
            let blend_id = parse_blend(blend)?;
            out.tag = COLOR_NODE_MIX;
            p[0] = *a;
            p[1] = *b;
            p[2] = blend_id;
            match fac {
                ColorFactor::Const(v) => {
                    p[3] = 0; // fac_src = const
                    p[4] = 0;
                    p[6] = v.to_bits();
                }
                ColorFactor::Node { node } => {
                    check_ref(*node, "fac.node")?;
                    p[3] = 1; // fac_src = node
                    p[4] = *node;
                    p[6] = 0;
                }
            }
            p[5] = if *clamp { 1 } else { 0 };
        }
        ColorNode::Invert { input, fac } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_INVERT;
            p[0] = *input;
            p[1] = fac.to_bits();
        }
        ColorNode::Math {
            input,
            op,
            b,
            c,
            clamp,
            swap,
        } => {
            check_ref(*input, "input")?;
            let op_id = parse_math_op(op)?;
            out.tag = COLOR_NODE_MATH;
            p[0] = *input;
            p[1] = op_id;
            p[2] = if *clamp { 1 } else { 0 };
            p[3] = b.to_bits();
            p[4] = c.to_bits();
            p[5] = if *swap { 1 } else { 0 };
        }
        ColorNode::HueSat {
            input,
            hue,
            saturation,
            value,
            fac,
        } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_HUE_SAT;
            p[0] = *input;
            p[1] = hue.to_bits();
            p[2] = saturation.to_bits();
            p[3] = value.to_bits();
            p[4] = fac.to_bits();
        }
        ColorNode::RgbCurve { input, .. } => {
            check_ref(*input, "input")?;
            let lut_ptr = lut_ptrs[self_idx];
            out.tag = COLOR_NODE_RGB_CURVE;
            p[0] = *input;
            p[1] = lut_ptr as u32;
            p[2] = (lut_ptr >> 32) as u32;
        }
        ColorNode::BrightContrast {
            input,
            bright,
            contrast,
        } => {
            check_ref(*input, "input")?;
            out.tag = COLOR_NODE_BRIGHT_CONTRAST;
            p[0] = *input;
            p[1] = bright.to_bits();
            p[2] = contrast.to_bits();
        }
        ColorNode::VertexColor {} => {
            out.tag = COLOR_NODE_VERTEX_COLOR;
        }
    }
    Ok(())
}

pub(crate) fn parse_blend(s: &str) -> Result<u32> {
    Ok(match s {
        "mix" => 0,
        "multiply" => 1,
        "add" => 2,
        "subtract" => 3,
        "screen" => 4,
        "divide" => 5,
        "difference" => 6,
        "darken" => 7,
        "lighten" => 8,
        "overlay" => 9,
        "soft_light" => 10,
        "linear_light" => 11,
        other => return Err(anyhow!("unknown Mix blend {:?}", other)),
    })
}

pub(crate) fn parse_math_op(s: &str) -> Result<u32> {
    Ok(match s {
        "add" => 0,
        "subtract" => 1,
        "multiply" => 2,
        "divide" => 3,
        "power" => 4,
        "multiply_add" => 5,
        "minimum" => 6,
        "maximum" => 7,
        other => return Err(anyhow!("unknown Math op {:?}", other)),
    })
}

/// Precompute the GPU-side volume coefficients from authored params.
/// Cycles' principled volume convention:
///   σ_s = color × density   (RGB scattering coefficient)
///   σ_a = absorption_color × density   (RGB absorption)
///   σ_t = σ_s + σ_a
///   σ_e = emission_color × emission_strength × density
/// Negative or NaN inputs collapse to 0; the device code expects
/// non-negative coefficients.
pub fn make_volume_gpu(v: &VolumeParams) -> VolumeGpu {
    let d = v.density.max(0.0);
    let mut sigma_s = [0.0f32; 3];
    let mut sigma_a = [0.0f32; 3];
    let mut emission = [0.0f32; 3];
    for i in 0..3 {
        sigma_s[i] = (v.color[i].max(0.0)) * d;
        sigma_a[i] = (v.absorption_color[i].max(0.0)) * d;
        emission[i] = (v.emission_color[i].max(0.0)) * v.emission_strength.max(0.0) * d;
    }
    let sigma_t = [
        sigma_s[0] + sigma_a[0],
        sigma_s[1] + sigma_a[1],
        sigma_s[2] + sigma_a[2],
    ];
    VolumeGpu {
        sigma_t,
        _pad0: 0.0,
        sigma_s,
        _pad1: 0.0,
        emission,
        anisotropy: v.anisotropy.clamp(-0.999, 0.999),
    }
}

/// Build a GPU material struct from a host Principled material.
pub fn make_material_data(
    mat: &PrincipledMaterial,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    graph: ColorGraphGpu,
    volume_ptr: optix_sys::CUdeviceptr,
) -> PrincipledGpu {
    let lookup = |id: Option<u32>| -> (optix_sys::CUdeviceptr, i32, i32, i32) {
        match id {
            Some(i) => match textures.get(i as usize).copied() {
                Some(t) => (t.0, t.1, t.2, 4),
                None => {
                    eprintln!(
                        "[vibrt] warn: material texture index {} out of range \
                         (have {} textures) — using null texture",
                        i,
                        textures.len()
                    );
                    (0, 0, 0, 0)
                }
            },
            None => (0, 0, 0, 0),
        }
    };
    let (bc_ptr, bc_w, bc_h, bc_c) = lookup(mat.base_color_tex);
    let (n_ptr, n_w, n_h, n_c) = lookup(mat.normal_tex);
    let (r_ptr, r_w, r_h, r_c) = lookup(mat.roughness_tex);
    let (m_ptr, m_w, m_h, m_c) = lookup(mat.metallic_tex);
    let (t_ptr, t_w, t_h, t_c) = lookup(mat.transmission_tex);
    let (e_ptr, e_w, e_h, e_c) = lookup(mat.emission_tex);
    let (b_ptr, b_w, b_h, b_c) = lookup(mat.bump_tex);

    PrincipledGpu {
        base_color: mat.base_color,
        metallic: mat.metallic.clamp(0.0, 1.0),
        roughness: mat.roughness.clamp(0.0, 1.0),
        ior: mat.ior.max(1.0),
        transmission: mat.transmission.clamp(0.0, 1.0),
        emission: mat.emission,
        base_color_tex: bc_ptr,
        base_color_tex_w: bc_w,
        base_color_tex_h: bc_h,
        base_color_tex_channels: bc_c,
        normal_tex: n_ptr,
        normal_tex_w: n_w,
        normal_tex_h: n_h,
        normal_tex_channels: n_c,
        roughness_tex: r_ptr,
        roughness_tex_w: r_w,
        roughness_tex_h: r_h,
        roughness_tex_channels: r_c,
        metallic_tex: m_ptr,
        metallic_tex_w: m_w,
        metallic_tex_h: m_h,
        metallic_tex_channels: m_c,
        transmission_tex: t_ptr,
        transmission_tex_w: t_w,
        transmission_tex_h: t_h,
        transmission_tex_channels: t_c,
        emission_tex: e_ptr,
        emission_tex_w: e_w,
        emission_tex_h: e_h,
        emission_tex_channels: e_c,
        uv_transform: mat.uv_transform,
        normal_strength: mat.normal_strength,
        bump_strength: mat.bump_strength,
        bump_tex: b_ptr,
        bump_tex_w: b_w,
        bump_tex_h: b_h,
        bump_tex_channels: b_c,
        alpha_threshold: mat.alpha_threshold.clamp(0.0, 1.0),
        anisotropy: mat.anisotropy.clamp(-1.0, 1.0),
        tangent_rotation: mat.tangent_rotation,
        coat_weight: mat.coat_weight.clamp(0.0, 1.0),
        coat_roughness: mat.coat_roughness.clamp(0.0, 1.0),
        coat_ior: mat.coat_ior.max(1.0),
        sheen_weight: mat.sheen_weight.clamp(0.0, 1.0),
        sheen_roughness: mat.sheen_roughness.clamp(0.0, 1.0),
        sheen_tint: mat.sheen_tint,
        sss_weight: mat.sss_weight.clamp(0.0, 1.0),
        sss_radius: mat.sss_radius,
        hair_weight: mat.hair_weight.clamp(0.0, 1.0),
        hair_offset: mat.hair_offset,
        hair_roughness_u: mat.hair_roughness_u.clamp(1e-3, 1.0),
        hair_roughness_v: mat.hair_roughness_v.clamp(1e-3, 1.0),
        use_vertex_color: if mat.use_vertex_color { 1 } else { 0 },
        color_graph_nodes: graph.nodes,
        color_graph_len: graph.len,
        color_graph_output: graph.output,
        volume: volume_ptr,
        volume_only: if mat.volume_only { 1 } else { 0 },
    }
}
