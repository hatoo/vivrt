#![allow(clippy::missing_transmute_annotations)]

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
    input: String,

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

#[allow(clippy::too_many_arguments)]
fn make_hitgroup_data(
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
    texcoords: optix_sys::CUdeviceptr,
    normals: optix_sys::CUdeviceptr,
    indices: optix_sys::CUdeviceptr,
    vertices: optix_sys::CUdeviceptr,
    num_vertices: i32,
) -> HitGroupData {
    let params = match mat.material_type {
        MAT_DIELECTRIC => MaterialParams {
            dielectric: DielectricParams {
                eta: mat.eta,
                tint: mat.tint,
            },
        },
        MAT_CONDUCTOR => MaterialParams {
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
    HitGroupData {
        material_type: mat.material_type,
        albedo: mat.albedo,
        emission: mat.emission,
        roughness: mat.roughness,
        roughness_v: mat.roughness_v,
        coat_roughness: mat.coat_roughness,
        coat_eta: mat.coat_eta,
        params,
        vertices,
        normals,
        indices,
        texcoords,
        num_vertices,
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
    }
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

    let input =
        std::fs::read_to_string(&cli.input).context(format!("Failed to read {}", cli.input))?;

    let scene_dir = std::path::Path::new(&cli.input)
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
    let ptx_src = ptx.to_src();

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
        .num_payload_values(23)
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
            max_trace_depth: scene.max_depth,
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

    struct GasEntry {
        handle: optix_sys::OptixTraversableHandle,
        sbt_offset: u32,
        transform: [f32; 12],
        is_sphere: bool,
    }
    let mut gas_entries: Vec<GasEntry> = Vec::new();
    let mut tri_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();
    let mut sphere_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();

    let build_options = AccelBuildOptions {
        build_flags: BuildFlags::PREFER_FAST_TRACE,
        operation: BuildOperation::Build,
    };

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
                let mut hg_data = make_hitgroup_data(
                    &obj.material,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
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
                    let s = stream.clone_htod(normals).cuda()?;
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

                // Upload image texture if present
                let (d_tex, tex_w, tex_h) = if let Some(ref tex) = obj.material.texture {
                    let s = stream.clone_htod(&tex.data).cuda()?;
                    let ptr = dptr(&s, &stream);
                    _device_buffers.push(unsafe { std::mem::transmute(s) });
                    (ptr, tex.width as i32, tex.height as i32)
                } else {
                    (0, 0, 0)
                };

                // Upload bump map if present (convert RGB to grayscale)
                let (d_bump, bump_w, bump_h) = if let Some(ref bmp) = obj.material.bump_map {
                    let gray: Vec<f32> = bmp
                        .data
                        .chunks(3)
                        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
                        .collect();
                    let s = stream.clone_htod(&gray).cuda()?;
                    let ptr = dptr(&s, &stream);
                    _device_buffers.push(unsafe { std::mem::transmute(s) });
                    (ptr, bmp.width as i32, bmp.height as i32)
                } else {
                    (0, 0, 0)
                };

                // Upload alpha map if present
                let mut upload_grayscale = |img: &scene::ImageTexture| -> anyhow::Result<(
                    optix_sys::CUdeviceptr,
                    i32,
                    i32,
                )> {
                    let gray: Vec<f32> = img
                        .data
                        .chunks(3)
                        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
                        .collect();
                    let s = stream.clone_htod(&gray).cuda()?;
                    let ptr = dptr(&s, &stream);
                    _device_buffers.push(unsafe { std::mem::transmute(s) });
                    Ok((ptr, img.width as i32, img.height as i32))
                };

                let (d_alpha, alpha_w, alpha_h) = if let Some(ref a) = obj.material.alpha_map {
                    upload_grayscale(a)?
                } else {
                    (0, 0, 0)
                };

                let (d_rough, rough_w, rough_h) = if let Some(ref r) = obj.material.roughness_map {
                    upload_grayscale(r)?
                } else {
                    (0, 0, 0)
                };

                let hg_data = make_hitgroup_data(
                    &obj.material,
                    d_tex,
                    tex_w,
                    tex_h,
                    d_bump,
                    bump_w,
                    bump_h,
                    d_alpha,
                    alpha_w,
                    alpha_h,
                    d_rough,
                    rough_w,
                    rough_h,
                    d_tc,
                    d_normals,
                    dptr(&d_indices, &stream),
                    dptr(&d_verts, &stream),
                    num_verts as i32,
                );

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
