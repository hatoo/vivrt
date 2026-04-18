//! Host-side material upload for Principled BSDF.

use crate::gpu_types::PrincipledGpu;
use crate::scene_format::PrincipledMaterial;
use crate::scene_loader::LoadedTexture;
use crate::CudaResultExt;
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

fn dptr(slice: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
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

/// Build a GPU material struct from a host Principled material.
pub fn make_material_data(
    mat: &PrincipledMaterial,
    textures: &[(optix_sys::CUdeviceptr, i32, i32)],
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
    }
}
