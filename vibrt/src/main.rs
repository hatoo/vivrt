#![allow(clippy::missing_transmute_annotations)]

mod camera;
mod gpu_types;
mod image_io;
mod pipeline;
mod principled;
mod scene_format;
mod scene_loader;
mod transform;

use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use gpu_types::*;
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix::*;
use scene_loader::LoadedScene;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "vibrt", about = "Blender-native OptiX path-tracing renderer")]
struct Args {
    /// Input scene.json (scene.bin must be alongside).
    input: Option<PathBuf>,

    /// Output image (.exr or .png). Default: output.exr
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override samples per pixel.
    #[arg(short, long)]
    spp: Option<u32>,

    /// Override max ray depth.
    #[arg(short, long)]
    depth: Option<u32>,

    /// Override indirect-contribution luminance clamp. 0 disables.
    #[arg(long = "clamp-indirect")]
    clamp_indirect: Option<f32>,

    /// Override image width.
    #[arg(long)]
    width: Option<u32>,

    /// Override image height.
    #[arg(long)]
    height: Option<u32>,

    /// Only compile the device code.
    #[arg(long)]
    compile_only: bool,
}

pub trait CudaResultExt<T> {
    fn cuda(self) -> Result<T>;
}

impl<T> CudaResultExt<T> for std::result::Result<T, cudarc::driver::DriverError> {
    fn cuda(self) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("CUDA error: {e:?}"))
    }
}

fn dptr_u8(s: &CudaSlice<u8>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}
fn dptr_u32(s: &CudaSlice<u32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}
fn dptr_f32(s: &CudaSlice<f32>, stream: &CudaStream) -> optix_sys::CUdeviceptr {
    let (p, _) = s.device_ptr(stream);
    p as optix_sys::CUdeviceptr
}

fn alloc_and_copy<T>(stream: &Arc<CudaStream>, v: &T) -> Result<optix_sys::CUdeviceptr> {
    let size = mem::size_of_val(v);
    let cu = stream.cu_stream();
    unsafe {
        let d = cudarc::driver::result::malloc_async(cu, size).cuda()?;
        cudarc::driver::result::memcpy_htod_async(
            d,
            std::slice::from_raw_parts(v as *const T as *const u8, size),
            cu,
        )
        .cuda()?;
        Ok(d as optix_sys::CUdeviceptr)
    }
}
fn alloc_and_copy_slice<T>(stream: &Arc<CudaStream>, data: &[T]) -> Result<optix_sys::CUdeviceptr> {
    let size = mem::size_of_val(data);
    if size == 0 {
        return Ok(0);
    }
    let cu = stream.cu_stream();
    unsafe {
        let d = cudarc::driver::result::malloc_async(cu, size).cuda()?;
        cudarc::driver::result::memcpy_htod_async(
            d,
            std::slice::from_raw_parts(data.as_ptr() as *const u8, size),
            cu,
        )
        .cuda()?;
        Ok(d as optix_sys::CUdeviceptr)
    }
}

struct GasEntry {
    handle: optix_sys::OptixTraversableHandle,
    sbt_offset: u32,
    transform: [f32; 12],
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.compile_only {
        pipeline::compile_ptx()?;
        println!("Compilation OK.");
        return Ok(());
    }

    let input = args
        .input
        .as_ref()
        .context("input scene.json required (or use --compile-only)")?;
    let mut scene = scene_loader::load_scene(input)?;

    if let Some(v) = args.spp {
        scene.file.render.spp = v;
    }
    if let Some(v) = args.depth {
        scene.file.render.max_depth = v;
    }
    if let Some(v) = args.clamp_indirect {
        scene.file.render.clamp_indirect = v;
    }
    if let Some(v) = args.width {
        scene.file.render.width = v;
    }
    if let Some(v) = args.height {
        scene.file.render.height = v;
    }
    scene.file.render.max_depth = scene.file.render.max_depth.min(31);

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("output.exr"));

    println!(
        "Scene: {}x{} @ {} spp, {} objects, {} meshes, {} mats, {} textures, lights(p/s/spot/rect): {}/{}/{}/{}",
        scene.file.render.width,
        scene.file.render.height,
        scene.file.render.spp,
        scene.objects.len(),
        scene.meshes.len(),
        scene.file.materials.len(),
        scene.textures.len(),
        scene.point_lights.len(),
        scene.sun_lights.len(),
        scene.spot_lights.len(),
        scene.rect_lights.len(),
    );

    render(&scene, &output)
}

fn render(scene: &LoadedScene, output: &std::path::Path) -> Result<()> {
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

    let ptx_src = pipeline::compile_ptx()?;

    // --- Pipeline ---
    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_LEVEL_INSTANCING)
        .num_payload_values(2)
        .num_attribute_values(2)
        .uses_primitive_type_flags(PrimitiveTypeFlags::TRIANGLE);

    let module_opts = ModuleCompileOptions::default();
    let module = Module::new(&ctx, &module_opts, &pipeline_options, ptx_src.as_bytes())
        .context("module")?
        .value;

    let raygen_pg = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")
        .context("raygen")?
        .value;
    let miss_rg_pg = ProgramGroup::miss(&ctx, &module, "__miss__ms")
        .context("miss radiance")?
        .value;
    let miss_shadow_pg = ProgramGroup::miss(&ctx, &module, "__miss__shadow")
        .context("miss shadow")?
        .value;
    let hit_rg_pg = ProgramGroup::hitgroup(&ctx)
        .closest_hit(&module, "__closesthit__ch")
        .any_hit(&module, "__anyhit__ah")
        .build()
        .context("hitgroup radiance")?
        .value;
    let hit_shadow_pg = ProgramGroup::hitgroup(&ctx)
        .closest_hit(&module, "__closesthit__shadow")
        .any_hit(&module, "__anyhit__ah")
        .build()
        .context("hitgroup shadow")?
        .value;

    let all_pgs: Vec<&ProgramGroup> = vec![
        &raygen_pg,
        &miss_rg_pg,
        &miss_shadow_pg,
        &hit_rg_pg,
        &hit_shadow_pg,
    ];
    let pipeline = Pipeline::new(
        &ctx,
        &pipeline_options,
        &PipelineLinkOptions { max_trace_depth: 2 },
        &all_pgs,
    )
    .context("pipeline")?
    .value;
    pipeline.set_stack_size(2048, 2048, 2048, 2)?;

    // --- Upload textures (shared across materials) ---
    let mut _tex_buffers: Vec<CudaSlice<f32>> = Vec::new();
    let tex_slots = principled::upload_textures(&scene.textures, &stream, &mut _tex_buffers)?;

    // --- Upload materials ---
    // Colour-graph buffers (node records + per-node LUT data) live for the
    // lifetime of the render; stash them so their CudaSlice drops are
    // deferred. Separate u32 / f32 vectors because RGBCurve LUTs are float.
    let mut _color_graph_buffers: Vec<CudaSlice<u32>> = Vec::new();
    let mut _color_graph_lut_buffers: Vec<CudaSlice<f32>> = Vec::new();
    let mut mat_device_ptrs: Vec<optix_sys::CUdeviceptr> = Vec::new();
    for mat in &scene.file.materials {
        let graph = match &mat.color_graph {
            Some(g) => principled::upload_color_graph(
                g,
                &tex_slots,
                &stream,
                &mut _color_graph_buffers,
                &mut _color_graph_lut_buffers,
            )?,
            None => principled::ColorGraphGpu::default(),
        };
        let gpu = principled::make_material_data(mat, &tex_slots, graph);
        mat_device_ptrs.push(alloc_and_copy(&stream, &gpu)?);
    }

    // --- Build per-mesh GAS ---
    let build_options = AccelBuildOptions {
        build_flags: BuildFlags::PREFER_FAST_TRACE,
        operation: BuildOperation::Build,
    };

    struct MeshGpu {
        handle: optix_sys::OptixTraversableHandle,
        d_verts: optix_sys::CUdeviceptr,
        d_normals: optix_sys::CUdeviceptr,
        d_indices: optix_sys::CUdeviceptr,
        d_uvs: optix_sys::CUdeviceptr,
        d_mat_indices: optix_sys::CUdeviceptr,
        d_vertex_colors: optix_sys::CUdeviceptr,
        num_verts: i32,
    }
    let mut _buffers: Vec<CudaSlice<u8>> = Vec::new();
    let mut _f32_buffers: Vec<CudaSlice<f32>> = Vec::new();
    let mut _u32_buffers: Vec<CudaSlice<u32>> = Vec::new();
    let mut meshes_gpu: Vec<MeshGpu> = Vec::new();

    for m in &scene.meshes {
        let d_verts = stream.clone_htod(&m.vertices).cuda()?;
        let d_indices = stream.clone_htod(&m.indices).cuda()?;
        let verts_ptr = dptr_f32(&d_verts, &stream);
        let idx_ptr = dptr_u32(&d_indices, &stream);

        let vert_ptrs = [verts_ptr];
        let flags = [GeometryFlags::NONE];
        let num_verts = (m.vertices.len() / 3) as u32;
        let num_tris = (m.indices.len() / 3) as u32;

        let tri_input = TriangleArrayInput::new(
            &vert_ptrs,
            num_verts,
            VertexFormat::Float3,
            3 * mem::size_of::<f32>() as u32,
            &flags,
        )
        .with_indices(
            idx_ptr,
            num_tris,
            IndicesFormat::UnsignedInt3,
            3 * mem::size_of::<u32>() as u32,
        );

        let bi = [BuildInput::Triangles(tri_input)];
        let sizes =
            accel::accel_compute_memory_usage(&ctx, &build_options, &bi).context("accel sizes")?;
        let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
        let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.cuda()?;

        let handle = accel::accel_build(
            &ctx,
            cu_stream,
            &build_options,
            &bi,
            dptr_u8(&d_temp, &stream),
            sizes.temp_size,
            dptr_u8(&d_output, &stream),
            sizes.output_size,
        )
        .context("accel build")?;

        let d_normals = if !m.normals.is_empty() {
            let s = stream.clone_htod(&m.normals).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };
        let d_uvs = if !m.uvs.is_empty() {
            let s = stream.clone_htod(&m.uvs).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };
        let d_mat_indices = if !m.material_indices.is_empty() {
            let s = stream.clone_htod(&m.material_indices).cuda()?;
            let p = dptr_u32(&s, &stream);
            _u32_buffers.push(s);
            p
        } else {
            0
        };
        let d_vertex_colors = if !m.vertex_colors.is_empty() {
            let s = stream.clone_htod(&m.vertex_colors).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };

        meshes_gpu.push(MeshGpu {
            handle,
            d_verts: verts_ptr,
            d_normals,
            d_indices: idx_ptr,
            d_uvs,
            d_mat_indices,
            d_vertex_colors,
            num_verts: num_verts as i32,
        });

        _f32_buffers.push(d_verts);
        _u32_buffers.push(d_indices);
        _buffers.push(d_temp);
        _buffers.push(d_output);
    }
    stream.synchronize().cuda()?;

    // --- Build per-object SBT records (2 per object: radiance + shadow) ---
    let mut tri_hg_records: Vec<SbtRecord<HitGroupData>> = Vec::new();
    let mut gas_entries: Vec<GasEntry> = Vec::new();
    let mut _mat_table_bufs: Vec<CudaSlice<u64>> = Vec::new();
    for (obj_idx, obj) in scene.objects.iter().enumerate() {
        let m = &meshes_gpu[obj.mesh as usize];
        let mat_ptr = *mat_device_ptrs
            .get(obj.material as usize)
            .ok_or_else(|| anyhow::anyhow!("object {} material index out of range", obj_idx))?;

        // Per-object material table (used when the mesh has per-triangle IDs).
        let obj_materials = &scene.file.objects[obj_idx].materials;
        let (materials_ptr, num_materials) = if !obj_materials.is_empty() {
            let ptrs: Vec<u64> = obj_materials
                .iter()
                .map(|mi| {
                    mat_device_ptrs.get(*mi as usize).copied().unwrap_or(mat_ptr)
                        as u64
                })
                .collect();
            let slice = stream.clone_htod(&ptrs).cuda()?;
            let (p, _) = slice.device_ptr(&stream);
            _mat_table_bufs.push(slice);
            (p as optix_sys::CUdeviceptr, ptrs.len() as i32)
        } else {
            (0, 0)
        };

        let hg = HitGroupData {
            mat: mat_ptr,
            vertices: m.d_verts,
            normals: m.d_normals,
            indices: m.d_indices,
            uvs: m.d_uvs,
            num_vertices: m.num_verts,
            area_light_group: -1,
            material_indices: m.d_mat_indices,
            materials: materials_ptr,
            num_materials,
            vertex_colors: m.d_vertex_colors,
        };
        tri_hg_records.push(SbtRecord::new(&hit_rg_pg, hg)?);
        tri_hg_records.push(SbtRecord::new(&hit_shadow_pg, hg)?);
        gas_entries.push(GasEntry {
            handle: m.handle,
            sbt_offset: (obj_idx as u32) * 2,
            transform: obj.transform,
        });
    }

    // --- Build IAS ---
    if gas_entries.is_empty() {
        anyhow::bail!("no geometry in scene");
    }
    let instances: Vec<optix_sys::OptixInstance> = gas_entries
        .iter()
        .enumerate()
        .map(|(i, e)| optix_sys::OptixInstance {
            transform: e.transform,
            instanceId: i as u32,
            sbtOffset: e.sbt_offset,
            visibilityMask: 255,
            flags: optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE.0 as u32,
            traversableHandle: e.handle,
            pad: [0; 2],
        })
        .collect();
    let d_instances = alloc_and_copy_slice(&stream, &instances)?;
    let ias_input = [BuildInput::Instances(accel::InstanceArrayInput {
        instances: d_instances,
        num_instances: instances.len() as u32,
    })];
    let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &ias_input)?;
    let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.cuda()?;
    let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.cuda()?;
    let traversable = accel::accel_build(
        &ctx,
        cu_stream,
        &build_options,
        &ias_input,
        dptr_u8(&d_temp, &stream),
        sizes.temp_size,
        dptr_u8(&d_output, &stream),
        sizes.output_size,
    )?;
    stream.synchronize().cuda()?;
    _buffers.push(d_temp);
    _buffers.push(d_output);

    // --- SBT ---
    let raygen_record = SbtRecord::new(&raygen_pg, RayGenData {})?;
    let miss_rg_record = SbtRecord::new(&miss_rg_pg, MissData { _unused: 0 })?;
    let miss_shadow_record = SbtRecord::new(&miss_shadow_pg, MissData { _unused: 0 })?;

    let d_rg = alloc_and_copy(&stream, &raygen_record)?;
    let mut miss_bytes: Vec<u8> = Vec::new();
    let miss_stride = mem::size_of_val(&miss_rg_record);
    miss_bytes.extend_from_slice(unsafe {
        std::slice::from_raw_parts(&miss_rg_record as *const _ as *const u8, miss_stride)
    });
    miss_bytes.extend_from_slice(unsafe {
        std::slice::from_raw_parts(&miss_shadow_record as *const _ as *const u8, miss_stride)
    });
    let d_miss = alloc_and_copy_slice(&stream, &miss_bytes)?;

    let hg_stride = mem::size_of::<SbtRecord<HitGroupData>>();
    let mut hg_bytes: Vec<u8> = Vec::new();
    for rec in &tri_hg_records {
        hg_bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride)
        });
    }
    let d_hg = alloc_and_copy_slice(&stream, &hg_bytes)?;

    let sbt = ShaderBindingTableBuilder::new(d_rg)
        .miss_records(d_miss, miss_stride as u32, 2)
        .hitgroup_records(d_hg, hg_stride as u32, tri_hg_records.len() as u32)
        .build()?;

    // --- Camera ---
    let aspect = scene.file.render.width as f32 / scene.file.render.height as f32;
    let (eye, cam_u, cam_v, cam_w) = camera::compute_camera(
        &scene.file.camera.transform,
        scene.file.camera.fov_y_rad,
        aspect,
    );

    // --- Lights ---
    let d_points = alloc_and_copy_slice(&stream, &scene.point_lights)?;
    let d_suns = alloc_and_copy_slice(&stream, &scene.sun_lights)?;
    let d_spots = alloc_and_copy_slice(&stream, &scene.spot_lights)?;
    let d_rects = alloc_and_copy_slice(&stream, &scene.rect_lights)?;
    let rect_cdf = build_rect_light_cdf(&scene.rect_lights);
    let d_rect_cdf = alloc_and_copy_slice(&stream, &rect_cdf)?;

    // --- World / envmap ---
    let (world_type, world_color, world_strength) = match &scene.file.world {
        Some(scene_format::WorldDesc::Constant { color, strength }) => (0, *color, *strength),
        Some(scene_format::WorldDesc::Envmap { strength, .. }) => (1, [0.0; 3], *strength),
        None => (0, [0.0; 3], 0.0),
    };
    let envmap_rot = match &scene.file.world {
        Some(scene_format::WorldDesc::Envmap { rotation_z_rad, .. }) => *rotation_z_rad,
        _ => 0.0,
    };
    let (d_envmap, env_w, env_h, d_mcdf, d_ccdf, env_integral) =
        if let Some((rgb, w, h)) = &scene.envmap_rgb {
            let d_env = alloc_and_copy_slice(&stream, rgb)?;
            let (m, c, total) = pipeline::build_envmap_cdf(rgb, *w, *h);
            let d_m = alloc_and_copy_slice(&stream, &m)?;
            let d_c = alloc_and_copy_slice(&stream, &c)?;
            (d_env, *w as i32, *h as i32, d_m, d_c, total)
        } else {
            (0, 0, 0, 0, 0, 0.0)
        };

    // --- GGX LUTs ---
    let (lut_e, lut_e_avg) = pipeline::generate_ggx_energy_lut();
    let d_ggx_e = alloc_and_copy_slice(&stream, &lut_e)?;
    let d_ggx_e_avg = alloc_and_copy_slice(&stream, &lut_e_avg)?;

    // --- Image buffer (float4 per pixel) ---
    let pixel_count = (scene.file.render.width * scene.file.render.height) as usize;
    let d_image: CudaSlice<f32> = stream.alloc_zeros(pixel_count * 4).cuda()?;

    let lp = LaunchParams {
        image: dptr_f32(&d_image, &stream),
        width: scene.file.render.width,
        height: scene.file.render.height,
        samples_per_pixel: scene.file.render.spp,
        max_depth: scene.file.render.max_depth,
        cam_eye: eye,
        cam_u,
        cam_v,
        cam_w,
        cam_lens_radius: scene.file.camera.lens_radius,
        cam_focal_distance: scene.file.camera.focal_distance,
        traversable,
        num_point_lights: scene.point_lights.len() as i32,
        point_lights: d_points,
        num_sun_lights: scene.sun_lights.len() as i32,
        sun_lights: d_suns,
        num_spot_lights: scene.spot_lights.len() as i32,
        spot_lights: d_spots,
        num_rect_lights: scene.rect_lights.len() as i32,
        rect_lights: d_rects,
        rect_light_cdf: d_rect_cdf,
        world_type,
        world_color,
        world_strength,
        envmap_data: d_envmap,
        envmap_width: env_w,
        envmap_height: env_h,
        envmap_marginal_cdf: d_mcdf,
        envmap_conditional_cdf: d_ccdf,
        envmap_integral: env_integral,
        envmap_rotation_z_rad: envmap_rot,
        ggx_e_lut: d_ggx_e,
        ggx_e_avg_lut: d_ggx_e_avg,
        clamp_indirect: scene.file.render.clamp_indirect,
    };
    let d_params = alloc_and_copy(&stream, &lp)?;

    println!("Rendering...");
    let t0 = std::time::Instant::now();
    pipeline
        .launch(
            cu_stream,
            d_params,
            mem::size_of::<LaunchParams>(),
            &sbt,
            scene.file.render.width,
            scene.file.render.height,
            1,
        )
        .context("launch")?;
    stream.synchronize().cuda()?;
    println!("Render: {:.2?}", t0.elapsed());

    // --- Readback + save ---
    let rgba: Vec<f32> = stream.clone_dtoh(&d_image).cuda()?;
    image_io::save_image(
        output,
        scene.file.render.width,
        scene.file.render.height,
        &rgba,
    )?;
    println!("Saved {}", output.display());
    Ok(())
}

fn build_rect_light_cdf(rects: &[AreaRectLight]) -> Vec<f32> {
    let n = rects.len();
    let mut cdf = vec![0.0f32; n + 1];
    for (i, r) in rects.iter().enumerate() {
        cdf[i + 1] = cdf[i] + r.power.max(0.0);
    }
    let total = cdf[n];
    if total > 0.0 {
        let inv = 1.0 / total;
        for v in cdf.iter_mut().skip(1) {
            *v *= inv;
        }
    }
    cdf
}
