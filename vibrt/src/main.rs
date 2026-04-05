#![allow(clippy::missing_transmute_annotations)]

mod bsdf;
mod gpu_types;
mod ply;
mod scene;
mod subdivision;
mod transform;

use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use gpu_types::*;
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix::*;
use scene::{parse_scene, SceneShape};
use std::mem;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "vibrt",
    about = "OptiX path tracing renderer for PBRTv4 scenes"
)]
struct Args {
    /// Input .pbrt scene file
    input: Option<String>,

    /// Output image file (.png)
    #[arg(short, long)]
    output: Option<String>,

    /// Override samples per pixel
    #[arg(short, long)]
    spp: Option<u32>,

    /// Override max ray depth
    #[arg(short, long)]
    depth: Option<u32>,

    /// Override image width
    #[arg(long)]
    width: Option<u32>,

    /// Override image height
    #[arg(long)]
    height: Option<u32>,

    /// Only compile the CUDA kernel (no scene loading or rendering)
    #[arg(long)]
    compile_only: bool,
}

/// Extension trait to convert cudarc's DriverError to anyhow::Error.
trait CudaResultExt<T> {
    fn cuda(self) -> Result<T>;
}

impl<T> CudaResultExt<T> for std::result::Result<T, cudarc::driver::DriverError> {
    fn cuda(self) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("CUDA error: {e:?}"))
    }
}

fn dptr<T>(slice: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

fn alloc_and_copy<T>(
    stream: &Arc<cudarc::driver::CudaStream>,
    val: &T,
) -> Result<optix_sys::CUdeviceptr> {
    let size = mem::size_of_val(val);
    let cu_stream = stream.cu_stream();
    unsafe {
        let dptr = cudarc::driver::result::malloc_async(cu_stream, size).cuda()?;
        cudarc::driver::result::memcpy_htod_async(
            dptr,
            std::slice::from_raw_parts(val as *const T as *const u8, size),
            cu_stream,
        )
        .cuda()?;
        Ok(dptr as optix_sys::CUdeviceptr)
    }
}

fn alloc_and_copy_slice<T>(
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[T],
) -> Result<optix_sys::CUdeviceptr> {
    let size = mem::size_of_val(data);
    if size == 0 {
        return Ok(0);
    }
    let cu_stream = stream.cu_stream();
    unsafe {
        let dptr = cudarc::driver::result::malloc_async(cu_stream, size).cuda()?;
        cudarc::driver::result::memcpy_htod_async(
            dptr,
            std::slice::from_raw_parts(data.as_ptr() as *const u8, size),
            cu_stream,
        )
        .cuda()?;
        Ok(dptr as optix_sys::CUdeviceptr)
    }
}

const GGX_LUT_SIZE: usize = 32;

/// Precompute GGX directional albedo E(cosθ, α) via Monte Carlo integration.
/// E(μ, α) = integral of (D*G*F)/(4*μ) * cos(θi) dωi with F=1.
/// Also computes E_avg(α) = 2 * integral of E(μ, α) * μ dμ.
fn generate_ggx_energy_lut() -> (Vec<f32>, Vec<f32>) {
    let n = GGX_LUT_SIZE;
    let num_samples = 1024u32;
    let mut e_lut = vec![0.0f32; n * n]; // [cos_theta][alpha]
    let mut e_avg = vec![0.0f32; n];

    for ai in 0..n {
        let alpha = (ai as f32 + 0.5) / n as f32;
        let a2 = alpha * alpha;

        for ci in 0..n {
            let cos_theta = (ci as f32 + 0.5) / n as f32;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

            let mut sum = 0.0f64;
            for s in 0..num_samples {
                // Quasi-random sampling using Hammersley sequence
                let u1 = s as f64 / num_samples as f64;
                let mut u2_bits = s;
                u2_bits = (u2_bits << 16) | (u2_bits >> 16);
                u2_bits = ((u2_bits & 0x55555555) << 1) | ((u2_bits & 0xAAAAAAAA) >> 1);
                u2_bits = ((u2_bits & 0x33333333) << 2) | ((u2_bits & 0xCCCCCCCC) >> 2);
                u2_bits = ((u2_bits & 0x0F0F0F0F) << 4) | ((u2_bits & 0xF0F0F0F0) >> 4);
                u2_bits = ((u2_bits & 0x00FF00FF) << 8) | ((u2_bits & 0xFF00FF00) >> 8);
                let u2 = u2_bits as f64 / 0x100000000u64 as f64;

                // Sample GGX half-vector
                let a2_d = a2 as f64;
                let cos_h = ((1.0 - u2) / (1.0 + (a2_d - 1.0) * u2)).sqrt();
                let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
                let phi = 2.0 * std::f64::consts::PI * u1;

                // Half-vector in local space (V is along z)
                let hx = sin_h * phi.cos();
                let hy = sin_h * phi.sin();
                let hz = cos_h;

                // View vector in local space
                let vx = sin_theta as f64;
                let vy = 0.0f64;
                let vz = cos_theta as f64;

                // Reflect V around H to get L
                let v_dot_h = vx * hx + vy * hy + vz * hz;
                if v_dot_h <= 0.0 {
                    continue;
                }
                let _lx = 2.0 * v_dot_h * hx - vx;
                let _ly = 2.0 * v_dot_h * hy - vy;
                let lz = 2.0 * v_dot_h * hz - vz;

                if lz <= 0.0 {
                    continue;
                }

                // Smith G2 / (4 * NdotV * NdotH)
                let n_dot_v = cos_theta as f64;
                let n_dot_l = lz;
                let n_dot_h = cos_h;

                // G1 for view
                let g1_v =
                    2.0 * n_dot_v / (n_dot_v + (a2_d + (1.0 - a2_d) * n_dot_v * n_dot_v).sqrt());
                let g1_l =
                    2.0 * n_dot_l / (n_dot_l + (a2_d + (1.0 - a2_d) * n_dot_l * n_dot_l).sqrt());

                // Weight for importance-sampled GGX: G2 * VdotH / (NdotV * NdotH)
                let weight = g1_v * g1_l * v_dot_h / (n_dot_v * n_dot_h);
                sum += weight;
            }
            e_lut[ci * n + ai] = (sum / num_samples as f64) as f32;
        }

        // Compute E_avg by integrating E(μ, α) * 2μ dμ
        let mut avg = 0.0f64;
        for ci in 0..n {
            let mu = (ci as f64 + 0.5) / n as f64;
            avg += e_lut[ci * n + ai] as f64 * 2.0 * mu / n as f64;
        }
        e_avg[ai] = avg as f32;
    }

    (e_lut, e_avg)
}

#[allow(clippy::too_many_arguments)]
fn make_material_data(
    mat: &scene::SceneMaterial,
    texture_data: optix_sys::CUdeviceptr,
    texture_width: i32,
    texture_height: i32,
    bump_data: optix_sys::CUdeviceptr,
    bump_width: i32,
    bump_height: i32,
    alpha_data: optix_sys::CUdeviceptr,
    alpha_width: i32,
    alpha_height: i32,
    roughness_data: optix_sys::CUdeviceptr,
    roughness_width: i32,
    roughness_height: i32,
    normalmap_data: optix_sys::CUdeviceptr,
    normalmap_width: i32,
    normalmap_height: i32,
) -> MaterialData {
    let params = match mat.material_type {
        MAT_DIELECTRIC => MaterialParams {
            dielectric: DielectricParams {
                eta: mat.eta,
                tint: mat.tint,
            },
        },
        MAT_CONDUCTOR | MAT_COATED_CONDUCTOR => MaterialParams {
            conductor: ConductorParams {
                eta: mat.conductor_eta,
                k: mat.conductor_k,
            },
        },
        _ => MaterialParams {
            diffuse: DiffuseParams {
                has_checkerboard: if mat.has_checkerboard { 1 } else { 0 },
                checker_scale_u: mat.checker_scale_u,
                checker_scale_v: mat.checker_scale_v,
                checker_color1: mat.checker_color1,
                checker_color2: mat.checker_color2,
            },
        },
    };
    MaterialData {
        material_type: mat.material_type,
        albedo: mat.albedo,
        emission: mat.emission,
        roughness: mat.roughness,
        roughness_v: mat.roughness_v,
        coat_roughness: mat.coat_roughness,
        coat_eta: mat.coat_eta,
        coat_thickness: mat.coat_thickness,
        coat_albedo: mat.coat_albedo,
        params,
        texture_mapping: 0,
        texture_inv_transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        texture_data,
        texture_width,
        texture_height,
        bump_data,
        bump_width,
        bump_height,
        alpha_data,
        alpha_width,
        alpha_height,
        roughness_data,
        roughness_width,
        roughness_height,
        normalmap_data,
        normalmap_width,
        normalmap_height,
        mix_mat1: 0,
        mix_mat2: 0,
        mix_amount_data: 0,
        mix_amount_width: 0,
        mix_amount_height: 0,
        mix_amount_value: 0.5,
    }
}

fn upload_material(
    mat_data: &MaterialData,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<optix_sys::CUdeviceptr> {
    alloc_and_copy(stream, mat_data)
}

/// Recursively upload a SceneMaterial (including mix sub-materials) to GPU.
fn upload_scene_material(
    mat: &scene::SceneMaterial,
    stream: &Arc<cudarc::driver::CudaStream>,
    bufs: &mut Vec<CudaSlice<u8>>,
    tex_cache: &mut std::collections::HashMap<
        *const scene::ImageTexture,
        (optix_sys::CUdeviceptr, i32, i32),
    >,
    bump_cache: &mut std::collections::HashMap<
        *const scene::ImageTexture,
        (optix_sys::CUdeviceptr, i32, i32),
    >,
) -> Result<optix_sys::CUdeviceptr> {
    let upload_tex = |tex: &std::sync::Arc<scene::ImageTexture>,
                      stream: &Arc<cudarc::driver::CudaStream>,
                      bufs: &mut Vec<CudaSlice<u8>>,
                      cache: &mut std::collections::HashMap<
        *const scene::ImageTexture,
        (optix_sys::CUdeviceptr, i32, i32),
    >|
     -> Result<(optix_sys::CUdeviceptr, i32, i32)> {
        let key = std::sync::Arc::as_ptr(tex);
        if let Some(&cached) = cache.get(&key) {
            return Ok(cached);
        }
        let s = stream.clone_htod(&tex.data).cuda()?;
        let ptr = dptr(&s, stream);
        bufs.push(unsafe { std::mem::transmute(s) });
        let result = (ptr, tex.width as i32, tex.height as i32);
        cache.insert(key, result);
        Ok(result)
    };
    let upload_gray = |tex: &std::sync::Arc<scene::ImageTexture>,
                       stream: &Arc<cudarc::driver::CudaStream>,
                       bufs: &mut Vec<CudaSlice<u8>>,
                       cache: &mut std::collections::HashMap<
        *const scene::ImageTexture,
        (optix_sys::CUdeviceptr, i32, i32),
    >|
     -> Result<(optix_sys::CUdeviceptr, i32, i32)> {
        let key = std::sync::Arc::as_ptr(tex);
        if let Some(&cached) = cache.get(&key) {
            return Ok(cached);
        }
        let gray: Vec<f32> = tex
            .data
            .chunks(3)
            .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
            .collect();
        let s = stream.clone_htod(&gray).cuda()?;
        let ptr = dptr(&s, stream);
        bufs.push(unsafe { std::mem::transmute(s) });
        let result = (ptr, tex.width as i32, tex.height as i32);
        cache.insert(key, result);
        Ok(result)
    };

    let (d_tex, tw, th) = if let Some(ref t) = mat.texture {
        upload_tex(t, stream, bufs, tex_cache)?
    } else {
        (0, 0, 0)
    };
    let (d_bump, bw, bh) = if let Some(ref b) = mat.bump_map {
        upload_gray(b, stream, bufs, bump_cache)?
    } else {
        (0, 0, 0)
    };
    let (d_alpha, aw, ah) = if let Some(ref a) = mat.alpha_map {
        upload_gray(a, stream, bufs, bump_cache)?
    } else {
        (0, 0, 0)
    };
    let (d_rough, rw, rh) = if let Some(ref r) = mat.roughness_map {
        upload_gray(r, stream, bufs, bump_cache)?
    } else {
        (0, 0, 0)
    };
    let (d_nmap, nw, nh) = if let Some(ref n) = mat.normal_map {
        upload_tex(n, stream, bufs, tex_cache)?
    } else {
        (0, 0, 0)
    };

    let mut mat_data = make_material_data(
        mat, d_tex, tw, th, d_bump, bw, bh, d_alpha, aw, ah, d_rough, rw, rh, d_nmap, nw, nh,
    );

    // Set texture mapping type
    if let Some(ref tex) = mat.texture {
        if let Some(ref sph) = tex.spherical {
            mat_data.texture_mapping = 1;
            mat_data.texture_inv_transform = sph.inv_transform;
        } else if let Some(ref cyl) = tex.cylindrical {
            mat_data.texture_mapping = 2;
            mat_data.texture_inv_transform = cyl.inv_transform;
        }
    }

    // Recursively upload mix sub-materials
    if let Some(ref m1) = mat.mix_mat1 {
        mat_data.mix_mat1 = upload_scene_material(m1, stream, bufs, tex_cache, bump_cache)?;
    }
    if let Some(ref m2) = mat.mix_mat2 {
        mat_data.mix_mat2 = upload_scene_material(m2, stream, bufs, tex_cache, bump_cache)?;
    }
    if let Some(ref amt) = mat.mix_amount {
        let (d, w, h) = upload_gray(amt, stream, bufs, bump_cache)?;
        mat_data.mix_amount_data = d;
        mat_data.mix_amount_width = w;
        mat_data.mix_amount_height = h;
    }
    mat_data.mix_amount_value = mat.mix_amount_value;

    upload_material(&mat_data, stream)
}

fn compute_camera(
    eye: &[f32; 3],
    look: &[f32; 3],
    up: &[f32; 3],
    fov: f32,
    aspect: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let w = [look[0] - eye[0], look[1] - eye[1], look[2] - eye[2]];
    let wlen = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let w = [w[0] / wlen, w[1] / wlen, w[2] / wlen];

    let u = [
        up[1] * w[2] - up[2] * w[1],
        up[2] * w[0] - up[0] * w[2],
        up[0] * w[1] - up[1] * w[0],
    ];
    let ulen = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let u = [u[0] / ulen, u[1] / ulen, u[2] / ulen];

    let v = [
        w[1] * u[2] - w[2] * u[1],
        w[2] * u[0] - w[0] * u[2],
        w[0] * u[1] - w[1] * u[0],
    ];

    // PBRT applies FOV to the narrower image dimension
    let (half_w, half_h) = if aspect >= 1.0 {
        // Landscape: FOV is vertical
        let hh = (fov.to_radians() * 0.5).tan();
        (aspect * hh, hh)
    } else {
        // Portrait: FOV is horizontal
        let hw = (fov.to_radians() * 0.5).tan();
        (hw, hw / aspect)
    };

    (
        [u[0] * half_w, u[1] * half_w, u[2] * half_w],
        [v[0] * half_h, v[1] * half_h, v[2] * half_h],
        w,
    )
}

struct GasEntry {
    handle: optix_sys::OptixTraversableHandle,
    sbt_offset: u32,
    transform: [f32; 12],
    is_sphere: bool,
}

fn compile_ptx() -> Result<String> {
    let cu_src = include_str!("devicecode.cu");
    let header_src = include_str!("devicecode.h");

    let optix_include = find_optix_include();
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![optix_include],
        use_fast_math: Some(true),
        options: vec![format!(
            "--include-path={}",
            std::env::current_dir()?.join("renderer/src").display()
        )],
        ..Default::default()
    };

    let full_src = format!(
        "// Inlined devicecode.h\n{}\n// devicecode.cu\n{}",
        header_src,
        cu_src.replace("#include \"devicecode.h\"", "// (inlined above)")
    );

    println!("Compiling device code with NVRTC...");
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&full_src, opts)
        .map_err(|e| anyhow::anyhow!("NVRTC compilation failed: {e:?}"))?;
    Ok(ptx.to_src().to_string())
}

/// Build marginal + conditional CDFs for envmap importance sampling.
fn build_envmap_cdf(env: &scene::ImageTexture) -> (Vec<f32>, Vec<f32>, f32) {
    let w = env.width as usize;
    let h = env.height as usize;

    let mut conditional_cdf = vec![0.0f32; h * (w + 1)];
    let mut row_integrals = vec![0.0f32; h];

    for y in 0..h {
        let sin_theta = (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin();
        let row_offset = y * (w + 1);
        conditional_cdf[row_offset] = 0.0;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let lum =
                0.2126 * env.data[idx] + 0.7152 * env.data[idx + 1] + 0.0722 * env.data[idx + 2];
            conditional_cdf[row_offset + x + 1] = conditional_cdf[row_offset + x] + lum * sin_theta;
        }
        row_integrals[y] = conditional_cdf[row_offset + w];
        if row_integrals[y] > 0.0 {
            let inv = 1.0 / row_integrals[y];
            for x in 1..=w {
                conditional_cdf[row_offset + x] *= inv;
            }
        }
    }

    let mut marginal_cdf = vec![0.0f32; h + 1];
    for y in 0..h {
        marginal_cdf[y + 1] = marginal_cdf[y] + row_integrals[y];
    }
    let total = marginal_cdf[h];
    if total > 0.0 {
        let inv = 1.0 / total;
        for y in 1..=h {
            marginal_cdf[y] *= inv;
        }
    }

    (marginal_cdf, conditional_cdf, total)
}

fn find_optix_include() -> String {
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

fn save_image(path: &str, width: u32, height: u32, pixels: &[u32]) -> Result<()> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in (0..height).rev() {
        for x in 0..width {
            let pixel = pixels[(y * width + x) as usize];
            rgb.push((pixel & 0xFF) as u8);
            rgb.push(((pixel >> 8) & 0xFF) as u8);
            rgb.push(((pixel >> 16) & 0xFF) as u8);
        }
    }

    if path.ends_with(".ppm") {
        use std::io::Write;
        let mut file = std::fs::File::create(path).context("Failed to create output file")?;
        write!(file, "P6\n{width} {height}\n255\n")?;
        file.write_all(&rgb)?;
    } else {
        let file = std::fs::File::create(path).context("Failed to create output file")?;
        let w = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .context("Failed to write PNG header")?;
        writer
            .write_image_data(&rgb)
            .context("Failed to write PNG data")?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Args::parse();

    if cli.compile_only {
        compile_ptx()?;
        println!("Kernel compilation successful.");
        return Ok(());
    }

    let input_path = cli
        .input
        .as_ref()
        .context("Input file required (use --compile-only to skip)")?;
    let input =
        std::fs::read_to_string(input_path).context(format!("Failed to read {}", input_path))?;

    let scene_dir = std::path::Path::new(input_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let mut scene = parse_scene(&input, scene_dir);

    if let Some(spp) = cli.spp {
        scene.spp = spp;
    }
    if let Some(depth) = cli.depth {
        scene.max_depth = depth;
    }
    // Clamp max depth to OptiX limit
    scene.max_depth = scene.max_depth.min(31);
    if let Some(w) = cli.width {
        scene.width = w;
    }
    if let Some(h) = cli.height {
        scene.height = h;
    }
    if let Some(ref o) = cli.output {
        scene.filename = o.clone();
    }

    println!(
        "Scene: {}x{}, {} spp, {} objects, {} distant lights, {} sphere lights, {} triangle lights",
        scene.width,
        scene.height,
        scene.spp,
        scene.objects.len(),
        scene.distant_lights.len(),
        scene.sphere_lights.len(),
        scene.triangle_lights.len()
    );

    // --- CUDA / OptiX init ---
    let cuda_ctx = CudaContext::new(0).cuda().context("CUDA context")?;
    let stream = cuda_ctx.default_stream();
    let cu_stream = stream.cu_stream() as optix_sys::CUstream;

    let optix_handle = optix::init().context("OptiX init")?;
    let ctx = DeviceContext::new(
        &optix_handle,
        cuda_ctx.cu_ctx() as optix_sys::CUcontext,
        &DeviceContextOptions::default(),
    )
    .context("OptiX context")?;

    // --- Compile PTX ---
    let ptx_src = compile_ptx()?;

    // --- OptiX pipeline ---
    let has_spheres = scene
        .objects
        .iter()
        .any(|o| matches!(o.shape, SceneShape::Sphere { .. }));

    let mut prim_flags = PrimitiveTypeFlags::TRIANGLE;
    if has_spheres {
        prim_flags |= PrimitiveTypeFlags::CUSTOM;
    }

    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(if scene.objects.len() > 1 || has_spheres {
            TraversableGraphFlags::ALLOW_SINGLE_LEVEL_INSTANCING
        } else {
            TraversableGraphFlags::ALLOW_SINGLE_GAS
        })
        .num_payload_values(30)
        .num_attribute_values(2)
        .uses_primitive_type_flags(prim_flags);

    let module_opts = ModuleCompileOptions::default();
    let module = Module::new(&ctx, &module_opts, &pipeline_options, ptx_src.as_bytes())
        .context("module")?
        .value;

    let raygen_pg = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")
        .context("raygen")?
        .value;
    let miss_pg = ProgramGroup::miss(&ctx, &module, "__miss__ms")
        .context("miss")?
        .value;

    let hitgroup_tri_pg = ProgramGroup::hitgroup(&ctx)
        .closest_hit(&module, "__closesthit__ch")
        .any_hit(&module, "__anyhit__alpha")
        .build()
        .context("hitgroup_tri")?
        .value;

    let hitgroup_sphere_pg = if has_spheres {
        Some(
            ProgramGroup::hitgroup(&ctx)
                .closest_hit(&module, "__closesthit__sphere")
                .intersection(&module, "__intersection__sphere")
                .build()
                .context("hitgroup_sphere")?
                .value,
        )
    } else {
        None
    };

    let mut all_pgs: Vec<&ProgramGroup> = vec![&raygen_pg, &miss_pg, &hitgroup_tri_pg];
    if let Some(ref pg) = hitgroup_sphere_pg {
        all_pgs.push(pg);
    }

    let pipeline = Pipeline::new(
        &ctx,
        &pipeline_options,
        &PipelineLinkOptions {
            max_trace_depth: 2, // iterative path tracing: primary + shadow ray
        },
        &all_pgs,
    )
    .context("pipeline")?
    .value;
    let max_graph_depth = if scene.objects.len() > 1 || has_spheres {
        2
    } else {
        1
    };
    pipeline.set_stack_size(2048, 2048, 2048, max_graph_depth)?;

    // --- Build geometry ---
    let mut _device_buffers: Vec<CudaSlice<u8>> = Vec::new();
    let mut gas_entries: Vec<GasEntry> = Vec::new();
    let mut tri_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();
    let mut sphere_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();

    let build_options = AccelBuildOptions {
        build_flags: BuildFlags::PREFER_FAST_TRACE,
        operation: BuildOperation::Build,
    };

    // Cache GPU pointers for shared textures to avoid duplicate uploads
    use scene::ImageTexture;
    use std::collections::HashMap;
    let mut tex_cache: HashMap<*const ImageTexture, (optix_sys::CUdeviceptr, i32, i32)> =
        HashMap::new();
    // For bump maps stored as grayscale, cache by source pointer
    let mut bump_cache: HashMap<*const ImageTexture, (optix_sys::CUdeviceptr, i32, i32)> =
        HashMap::new();

    for obj in &scene.objects {
        match &obj.shape {
            SceneShape::Sphere { radius } => {
                let aabb = [-*radius, -*radius, -*radius, *radius, *radius, *radius];
                let d_aabb = stream.clone_htod(&aabb).cuda()?;
                let aabb_ptrs = [dptr(&d_aabb, &stream)];
                let flags = [GeometryFlags::NONE];

                let custom_input = accel::CustomPrimitiveInput {
                    aabb_buffers: &aabb_ptrs,
                    num_primitives: 1,
                    stride: 0,
                    flags: &flags,
                    num_sbt_records: 1,
                    primitive_index_offset: 0,
                };

                let bi = [BuildInput::CustomPrimitives(custom_input)];
                let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &bi)
                    .context("sphere accel memory")?;
                let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
                let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.cuda()?;

                let handle = accel::accel_build(
                    &ctx,
                    cu_stream,
                    &build_options,
                    &bi,
                    dptr(&d_temp, &stream),
                    sizes.temp_size,
                    dptr(&d_output, &stream),
                    sizes.output_size,
                )
                .context("sphere accel build")?;

                // Store radius in num_vertices field (reinterpreted as float in shader)
                let d_mat_ptr = upload_scene_material(
                    &obj.material,
                    &stream,
                    &mut _device_buffers,
                    &mut tex_cache,
                    &mut bump_cache,
                )?;
                let mut hg_data = HitGroupData {
                    mat: d_mat_ptr,
                    vertices: 0,
                    normals: 0,
                    indices: 0,
                    texcoords: 0,
                    num_vertices: 0,
                };
                hg_data.num_vertices = radius.to_bits() as i32;

                let sbt_offset = sphere_hg_records.len() as u32;
                sphere_hg_records.push(SbtRecord::new(
                    hitgroup_sphere_pg.as_ref().context("no sphere hitgroup")?,
                    hg_data,
                )?);

                gas_entries.push(GasEntry {
                    handle,
                    sbt_offset,
                    transform: obj.transform,
                    is_sphere: true,
                });

                _device_buffers.push(unsafe { std::mem::transmute(d_aabb) });
                _device_buffers.push(d_temp);
                _device_buffers.push(d_output);
            }
            SceneShape::TriangleMesh {
                vertices,
                indices,
                texcoords,
                normals,
            } => {
                let transformed = transform::transform_vertices(vertices, &obj.transform);
                let d_verts = stream.clone_htod(&transformed).cuda()?;
                let d_indices: CudaSlice<i32> = stream.clone_htod(indices).cuda()?;
                let d_tc = if !texcoords.is_empty() {
                    let s = stream.clone_htod(texcoords).cuda()?;
                    let ptr = dptr(&s, &stream);
                    _device_buffers.push(unsafe { std::mem::transmute(s) });
                    ptr
                } else {
                    0
                };
                let d_normals = if !normals.is_empty() {
                    let transformed_normals = transform::transform_normals(normals, &obj.transform);
                    let s = stream.clone_htod(&transformed_normals).cuda()?;
                    let ptr = dptr(&s, &stream);
                    _device_buffers.push(unsafe { std::mem::transmute(s) });
                    ptr
                } else {
                    0
                };

                let vert_ptrs = [dptr(&d_verts, &stream)];
                let flags = [GeometryFlags::NONE];
                let num_verts = transformed.len() as u32 / 3;
                let num_tris = indices.len() as u32 / 3;

                let tri_input = TriangleArrayInput::new(
                    &vert_ptrs,
                    num_verts,
                    VertexFormat::Float3,
                    3 * mem::size_of::<f32>() as u32,
                    &flags,
                )
                .with_indices(
                    dptr(&d_indices, &stream),
                    num_tris,
                    IndicesFormat::UnsignedInt3,
                    3 * mem::size_of::<i32>() as u32,
                );

                let bi = [BuildInput::Triangles(tri_input)];
                let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &bi)
                    .context("tri accel memory")?;
                let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
                let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.cuda()?;

                let handle = accel::accel_build(
                    &ctx,
                    cu_stream,
                    &build_options,
                    &bi,
                    dptr(&d_temp, &stream),
                    sizes.temp_size,
                    dptr(&d_output, &stream),
                    sizes.output_size,
                )
                .context("tri accel build")?;

                let d_mat_ptr = upload_scene_material(
                    &obj.material,
                    &stream,
                    &mut _device_buffers,
                    &mut tex_cache,
                    &mut bump_cache,
                )?;

                let hg_data = HitGroupData {
                    mat: d_mat_ptr,
                    vertices: dptr(&d_verts, &stream),
                    normals: d_normals,
                    indices: dptr(&d_indices, &stream),
                    texcoords: d_tc,
                    num_vertices: num_verts as i32,
                };

                let sbt_offset = tri_hg_records.len() as u32;
                tri_hg_records.push(SbtRecord::new(&hitgroup_tri_pg, hg_data)?);

                gas_entries.push(GasEntry {
                    handle,
                    sbt_offset,
                    transform: transform::identity(),
                    is_sphere: false,
                });

                _device_buffers.push(unsafe { std::mem::transmute(d_verts) });
                _device_buffers.push(unsafe { std::mem::transmute(d_indices) });
                _device_buffers.push(d_temp);
                _device_buffers.push(d_output);
            }
        }
    }

    stream.synchronize().cuda()?;

    // --- Build IAS or use single GAS ---
    let needs_ias = has_spheres || gas_entries.len() > 1;
    let traversable = if needs_ias {
        let sphere_sbt_base = tri_hg_records.len() as u32;
        let instances: Vec<optix_sys::OptixInstance> = gas_entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let sbt_offset = if entry.is_sphere {
                    sphere_sbt_base + entry.sbt_offset
                } else {
                    entry.sbt_offset
                };
                optix_sys::OptixInstance {
                    transform: entry.transform,
                    instanceId: i as u32,
                    sbtOffset: sbt_offset,
                    visibilityMask: 255,
                    flags: optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE.0 as u32,
                    traversableHandle: entry.handle,
                    pad: [0; 2],
                }
            })
            .collect();

        let d_instances = alloc_and_copy_slice(&stream, &instances)?;
        let ias_input = [BuildInput::Instances(accel::InstanceArrayInput {
            instances: d_instances,
            num_instances: instances.len() as u32,
        })];

        let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &ias_input)
            .context("IAS memory")?;
        let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
        let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.cuda()?;

        let ias_handle = accel::accel_build(
            &ctx,
            cu_stream,
            &build_options,
            &ias_input,
            dptr(&d_temp, &stream),
            sizes.temp_size,
            dptr(&d_output, &stream),
            sizes.output_size,
        )
        .context("IAS build")?;
        stream.synchronize().cuda()?;
        _device_buffers.push(d_temp);
        _device_buffers.push(d_output);
        ias_handle
    } else if !gas_entries.is_empty() {
        gas_entries[0].handle
    } else {
        anyhow::bail!("No geometry in scene");
    };

    // --- SBT ---
    let raygen_record = SbtRecord::new(&raygen_pg, RayGenData {})?;
    let miss_record = SbtRecord::new(
        &miss_pg,
        MissData {
            bg_color: [0.0, 0.0, 0.0],
        },
    )?;

    let d_rg = alloc_and_copy(&stream, &raygen_record)?;
    let d_ms = alloc_and_copy(&stream, &miss_record)?;

    let hg_stride = mem::size_of::<SbtRecord<HitGroupData>>();
    let mut all_hg_records: Vec<u8> = Vec::new();
    for rec in &tri_hg_records {
        let bytes = unsafe { std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride) };
        all_hg_records.extend_from_slice(bytes);
    }
    for rec in &sphere_hg_records {
        let bytes = unsafe { std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride) };
        all_hg_records.extend_from_slice(bytes);
    }
    let total_hg_count = tri_hg_records.len() + sphere_hg_records.len();
    let d_hg = alloc_and_copy_slice(&stream, &all_hg_records)?;

    let sbt = ShaderBindingTableBuilder::new(d_rg)
        .miss_records(d_ms, mem::size_of_val(&miss_record) as u32, 1)
        .hitgroup_records(d_hg, hg_stride as u32, total_hg_count as u32)
        .build()
        .context("SBT")?;

    // --- Camera & lights ---
    let (mut cam_u, cam_v, cam_w) = compute_camera(
        &scene.cam_eye,
        &scene.cam_look,
        &scene.cam_up,
        scene.fov,
        scene.width as f32 / scene.height as f32,
    );
    if scene.cam_flip_x {
        cam_u = [-cam_u[0], -cam_u[1], -cam_u[2]];
    }

    let d_distant_lights = if scene.distant_lights.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &scene.distant_lights)?
    };
    let d_sphere_lights = if scene.sphere_lights.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &scene.sphere_lights)?
    };
    let d_triangle_lights = if scene.triangle_lights.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &scene.triangle_lights)?
    };
    let d_triangle_light_vertices = if scene.triangle_light_vertices.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &scene.triangle_light_vertices)?
    };

    // Build power-weighted CDF over triangle light groups (object-based)
    let (d_tri_light_groups, d_tri_light_group_cdf) = if scene.triangle_light_groups.is_empty() {
        (0, 0)
    } else {
        let groups = &scene.triangle_light_groups;
        let n = groups.len();
        let mut cdf = vec![0.0f32; n + 1];
        for (i, g) in groups.iter().enumerate() {
            cdf[i + 1] = cdf[i] + g.total_power;
        }
        let total = cdf[n];
        if total > 0.0 {
            let inv = 1.0 / total;
            for v in cdf.iter_mut().skip(1) {
                *v *= inv;
            }
        }
        (
            alloc_and_copy_slice(&stream, groups)?,
            alloc_and_copy_slice(&stream, &cdf)?,
        )
    };

    // --- GGX energy compensation LUT ---
    let (ggx_e_lut_data, ggx_e_avg_data) = generate_ggx_energy_lut();
    let d_ggx_e_lut = alloc_and_copy_slice(&stream, &ggx_e_lut_data)?;
    let d_ggx_e_avg = alloc_and_copy_slice(&stream, &ggx_e_avg_data)?;

    // --- Environment map + importance sampling CDF ---
    let (d_envmap, envmap_w, envmap_h, d_marginal_cdf, d_conditional_cdf, envmap_integral) =
        if let Some(ref env) = scene.envmap {
            let d = alloc_and_copy_slice(&stream, &env.data)?;
            let (marginal, conditional, total) = build_envmap_cdf(env);
            let d_m = alloc_and_copy_slice(&stream, &marginal)?;
            let d_c = alloc_and_copy_slice(&stream, &conditional)?;
            (d, env.width as i32, env.height as i32, d_m, d_c, total)
        } else {
            (0, 0, 0, 0, 0, 0.0)
        };

    // --- Launch ---
    let pixel_count = (scene.width * scene.height) as usize;
    let d_image: CudaSlice<u32> = stream.alloc_zeros(pixel_count).cuda()?;

    let launch_params = LaunchParams {
        image: dptr(&d_image, &stream),
        width: scene.width,
        height: scene.height,
        samples_per_pixel: scene.spp,
        max_depth: scene.max_depth,
        cam_eye: scene.cam_eye,
        cam_u,
        cam_v,
        cam_w,
        traversable,
        ambient_light: scene.ambient_light,
        num_distant_lights: scene.distant_lights.len() as i32,
        distant_lights: d_distant_lights,
        num_sphere_lights: scene.sphere_lights.len() as i32,
        sphere_lights: d_sphere_lights,
        num_triangle_lights: scene.triangle_lights.len() as i32,
        triangle_lights: d_triangle_lights,
        triangle_light_vertices: d_triangle_light_vertices,
        triangle_light_groups: d_tri_light_groups,
        num_triangle_light_groups: scene.triangle_light_groups.len() as i32,
        triangle_light_group_cdf: d_tri_light_group_cdf,
        envmap_data: d_envmap,
        envmap_width: envmap_w,
        envmap_height: envmap_h,
        envmap_marginal_cdf: d_marginal_cdf,
        envmap_conditional_cdf: d_conditional_cdf,
        envmap_integral: envmap_integral,
        envmap_inv_rotation: {
            let inv = transform::invert(&scene.envmap_transform);
            // Extract 3x3 rotation from 3x4 inverse transform
            [
                inv[0], inv[1], inv[2], inv[4], inv[5], inv[6], inv[8], inv[9], inv[10],
            ]
        },
        has_portal: if scene.portal.is_some() { 1 } else { 0 },
        portal: if let Some(p) = &scene.portal {
            [
                p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2],
                p[3][0], p[3][1], p[3][2],
            ]
        } else {
            [0.0; 12]
        },
        ggx_e_lut: d_ggx_e_lut,
        ggx_e_avg_lut: d_ggx_e_avg,
    };
    let d_params = alloc_and_copy(&stream, &launch_params)?;

    println!(
        "Rendering {}x{} @ {} spp...",
        scene.width, scene.height, scene.spp
    );
    let render_start = std::time::Instant::now();
    pipeline
        .launch(
            cu_stream,
            d_params,
            mem::size_of::<LaunchParams>(),
            &sbt,
            scene.width,
            scene.height,
            1,
        )
        .context("launch")?;
    stream.synchronize().cuda()?;
    let render_elapsed = render_start.elapsed();
    println!("Rendering took {render_elapsed:.2?}");

    let pixels = stream.clone_dtoh(&d_image).cuda()?;

    let mut output_file = scene.filename.replace(".exr", ".png");
    if !output_file.ends_with(".png") && !output_file.ends_with(".ppm") {
        output_file = format!("{output_file}.png");
    }
    save_image(&output_file, scene.width, scene.height, &pixels)?;
    println!("Saved {output_file}");

    Ok(())
}
