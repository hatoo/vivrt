//! Host-side material upload for Principled BSDF.

use crate::gpu_types::{
    ColorGraphNode, PrincipledGpu, COLOR_NODE_BRIGHT_CONTRAST, COLOR_NODE_CONST,
    COLOR_NODE_HUE_SAT, COLOR_NODE_IMAGE_TEX, COLOR_NODE_INVERT, COLOR_NODE_MATH, COLOR_NODE_MIX,
    COLOR_NODE_RGB_CURVE,
};
use crate::scene_format::{ColorFactor, ColorGraph, ColorNode, PrincipledMaterial};
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
        let slice = stream.clone_htod(&tex.data).cuda()?;
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
    // Each ColorGraphNode is 16 u32 (64 bytes). `flat` is that already, but
    // packed as a Vec<u32> for cheap htod.
    let slice = stream.clone_htod(&flat).cuda()?;
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
) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::with_capacity(graph.nodes.len() * 16);
    for (i, node) in graph.nodes.iter().enumerate() {
        let mut record = [0u32; 16];
        flatten_one(node, i, textures, lut_ptrs, &mut record)?;
        out.extend_from_slice(&record);
    }
    Ok(out)
}

fn flatten_one(
    node: &ColorNode,
    self_idx: usize,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    lut_ptrs: &[optix_sys::CUdeviceptr],
    out: &mut [u32; 16],
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
    match node {
        ColorNode::Const { rgb } => {
            out[0] = COLOR_NODE_CONST;
            out[1] = rgb[0].to_bits();
            out[2] = rgb[1].to_bits();
            out[3] = rgb[2].to_bits();
        }
        ColorNode::ImageTex { tex, uv } => {
            let (ptr, w, h) = *textures.get(*tex as usize).ok_or_else(|| {
                anyhow!("color_graph ImageTex.tex = {} out of range", tex)
            })?;
            out[0] = COLOR_NODE_IMAGE_TEX;
            out[1] = ptr as u32;
            out[2] = (ptr >> 32) as u32;
            out[3] = w as u32;
            out[4] = h as u32;
            out[5] = 4; // RGBA f32 — every loaded texture is padded to 4 channels
            for k in 0..6 {
                out[6 + k] = uv[k].to_bits();
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
            out[0] = COLOR_NODE_MIX;
            out[1] = *a;
            out[2] = *b;
            out[3] = blend_id;
            match fac {
                ColorFactor::Const(v) => {
                    out[4] = 0; // fac_src = const
                    out[5] = 0;
                    out[7] = v.to_bits();
                }
                ColorFactor::Node { node } => {
                    check_ref(*node, "fac.node")?;
                    out[4] = 1; // fac_src = node
                    out[5] = *node;
                    out[7] = 0;
                }
            }
            out[6] = if *clamp { 1 } else { 0 };
        }
        ColorNode::Invert { input, fac } => {
            check_ref(*input, "input")?;
            out[0] = COLOR_NODE_INVERT;
            out[1] = *input;
            out[2] = fac.to_bits();
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
            out[0] = COLOR_NODE_MATH;
            out[1] = *input;
            out[2] = op_id;
            out[3] = if *clamp { 1 } else { 0 };
            out[4] = b.to_bits();
            out[5] = c.to_bits();
            out[6] = if *swap { 1 } else { 0 };
        }
        ColorNode::HueSat {
            input,
            hue,
            saturation,
            value,
            fac,
        } => {
            check_ref(*input, "input")?;
            out[0] = COLOR_NODE_HUE_SAT;
            out[1] = *input;
            out[2] = hue.to_bits();
            out[3] = saturation.to_bits();
            out[4] = value.to_bits();
            out[5] = fac.to_bits();
        }
        ColorNode::RgbCurve { input, .. } => {
            check_ref(*input, "input")?;
            let lut_ptr = lut_ptrs[self_idx];
            out[0] = COLOR_NODE_RGB_CURVE;
            out[1] = *input;
            out[2] = lut_ptr as u32;
            out[3] = (lut_ptr >> 32) as u32;
        }
        ColorNode::BrightContrast {
            input,
            bright,
            contrast,
        } => {
            check_ref(*input, "input")?;
            out[0] = COLOR_NODE_BRIGHT_CONTRAST;
            out[1] = *input;
            out[2] = bright.to_bits();
            out[3] = contrast.to_bits();
        }
    }
    Ok(())
}

fn parse_blend(s: &str) -> Result<u32> {
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

fn parse_math_op(s: &str) -> Result<u32> {
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

/// Build a GPU material struct from a host Principled material.
pub fn make_material_data(
    mat: &PrincipledMaterial,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
    graph: ColorGraphGpu,
) -> PrincipledGpu {
    let lookup = |id: Option<u32>| -> (optix_sys::CUdeviceptr, i32, i32, i32) {
        match id {
            Some(i) => {
                let t = textures.get(i as usize).copied().unwrap_or((0, 0, 0));
                (t.0, t.1, t.2, 4)
            }
            None => (0, 0, 0, 0),
        }
    };
    let (bc_ptr, bc_w, bc_h, bc_c) = lookup(mat.base_color_tex);
    let (n_ptr, n_w, n_h, n_c) = lookup(mat.normal_tex);
    let (r_ptr, r_w, r_h, r_c) = lookup(mat.roughness_tex);
    let (m_ptr, m_w, m_h, m_c) = lookup(mat.metallic_tex);
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
        sss_anisotropy: mat.sss_anisotropy.clamp(-0.99, 0.99),
        use_vertex_color: if mat.use_vertex_color { 1 } else { 0 },
        color_graph_nodes: graph.nodes,
        color_graph_len: graph.len,
        color_graph_output: graph.output,
    }
}
