//! Device-code compilation + precomputed lookup tables.

use anyhow::Result;

pub fn find_optix_include() -> String {
    if let Ok(root) = std::env::var("OPTIX_ROOT") {
        return format!("{root}/include");
    }
    #[cfg(target_os = "windows")]
    {
        let default = r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include";
        if std::path::Path::new(default).exists() {
            return default.to_string();
        }
    }
    #[cfg(target_os = "linux")]
    {
        let default = "/usr/local/NVIDIA-OptiX-SDK-9.0.0/include";
        if std::path::Path::new(default).exists() {
            return default.to_string();
        }
    }
    panic!("OptiX SDK not found. Set OPTIX_ROOT.");
}

pub fn compile_ptx() -> Result<String> {
    let cu_src = include_str!("devicecode.cu");
    let header_src = include_str!("devicecode.h");
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![find_optix_include()],
        use_fast_math: Some(true),
        ..Default::default()
    };
    let full_src = format!(
        "// -- inlined devicecode.h --\n{}\n// -- devicecode.cu --\n{}",
        header_src,
        cu_src.replace("#include \"devicecode.h\"", "// (inlined above)")
    );
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&full_src, opts)
        .map_err(|e| anyhow::anyhow!("NVRTC compile failed: {e:?}"))?;
    Ok(ptx.to_src().to_string())
}

pub const GGX_LUT_SIZE: usize = 32;

/// Monte-Carlo precompute of GGX directional albedo E(cosθ, α) and E_avg(α).
pub fn generate_ggx_energy_lut() -> (Vec<f32>, Vec<f32>) {
    let n = GGX_LUT_SIZE;
    let num_samples = 16384u32;
    let mut e_lut = vec![0.0f32; n * n];
    let mut e_avg = vec![0.0f32; n];
    for ai in 0..n {
        let alpha = (ai as f32 + 0.5) / n as f32;
        let a2 = alpha * alpha;
        for ci in 0..n {
            let cos_theta = (ci as f32 + 0.5) / n as f32;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let mut sum = 0.0f64;
            for s in 0..num_samples {
                let u1 = s as f64 / num_samples as f64;
                let mut bits = s;
                bits = (bits << 16) | (bits >> 16);
                bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
                bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
                bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
                bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
                let u2 = bits as f64 / 0x1_0000_0000u64 as f64;
                let a2_d = a2 as f64;
                let cos_h = ((1.0 - u2) / (1.0 + (a2_d - 1.0) * u2)).sqrt();
                let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
                let phi = 2.0 * std::f64::consts::PI * u1;
                let hx = sin_h * phi.cos();
                let hy = sin_h * phi.sin();
                let hz = cos_h;
                let vx = sin_theta as f64;
                let vy = 0.0;
                let vz = cos_theta as f64;
                let v_dot_h = vx * hx + vy * hy + vz * hz;
                if v_dot_h <= 0.0 {
                    continue;
                }
                let lz = 2.0 * v_dot_h * hz - vz;
                if lz <= 0.0 {
                    continue;
                }
                let n_dot_v = cos_theta as f64;
                let n_dot_l = lz;
                let n_dot_h = cos_h;
                let g1_v =
                    2.0 * n_dot_v / (n_dot_v + (a2_d + (1.0 - a2_d) * n_dot_v * n_dot_v).sqrt());
                let g1_l =
                    2.0 * n_dot_l / (n_dot_l + (a2_d + (1.0 - a2_d) * n_dot_l * n_dot_l).sqrt());
                let weight = g1_v * g1_l * v_dot_h / (n_dot_v * n_dot_h);
                sum += weight;
            }
            e_lut[ci * n + ai] = (sum / num_samples as f64) as f32;
        }
        let mut avg = 0.0f64;
        for ci in 0..n {
            let mu = (ci as f64 + 0.5) / n as f64;
            avg += e_lut[ci * n + ai] as f64 * 2.0 * mu / n as f64;
        }
        e_avg[ai] = avg as f32;
    }
    (e_lut, e_avg)
}

/// Build marginal + conditional CDFs for envmap importance sampling (RGB data).
pub fn build_envmap_cdf(rgb: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>, f32) {
    let w = width as usize;
    let h = height as usize;
    let mut cond = vec![0.0f32; h * (w + 1)];
    let mut row_int = vec![0.0f32; h];
    for y in 0..h {
        let sin_t = (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin();
        let ro = y * (w + 1);
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let lum = 0.2126 * rgb[idx] + 0.7152 * rgb[idx + 1] + 0.0722 * rgb[idx + 2];
            cond[ro + x + 1] = cond[ro + x] + lum * sin_t;
        }
        row_int[y] = cond[ro + w];
        if row_int[y] > 0.0 {
            let inv = 1.0 / row_int[y];
            for x in 1..=w {
                cond[ro + x] *= inv;
            }
        }
    }
    let mut marg = vec![0.0f32; h + 1];
    for y in 0..h {
        marg[y + 1] = marg[y] + row_int[y];
    }
    let total = marg[h];
    if total > 0.0 {
        let inv = 1.0 / total;
        for y in 1..=h {
            marg[y] *= inv;
        }
    }
    (marg, cond, total)
}
