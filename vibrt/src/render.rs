//! GPU render pipeline: CUDA + OptiX setup, per-scene uploads, kernel launch,
//! optional denoise, readback. Returns pixels rather than writing to disk so
//! both the CLI binary and the in-process Python path can share it.
//!
//! The body here is the verbatim split-out of what used to be `render()` in
//! `main.rs`; the only behavioural changes are (1) progress text routes
//! through the `Progress` trait so callers can capture it (Blender's status
//! bar, CLI stdout, ...), and (2) the function returns `RenderOutput` instead
//! of saving an image — `image_io::save_image` is now the CLI's job.

use crate::gpu_types::*;
use crate::scene_format;
use crate::scene_loader;
use crate::scene_loader::LoadedScene;
use crate::{camera, pipeline, principled, CudaResultExt};

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix::*;
use std::mem;
use std::sync::Arc;

/// Per-render knobs supplied by the CLI / Python wrapper. `None` fields use
/// whatever the scene file specified; `denoise` is a flag (always known).
#[derive(Default, Clone, Copy)]
pub struct RenderOptions {
    pub spp: Option<u32>,
    pub max_depth: Option<u32>,
    pub clamp_indirect: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub denoise: bool,
}

/// Final rendered RGBA buffer plus the dimensions the caller can save / hand
/// to numpy.
pub struct RenderOutput {
    /// width × height × 4 floats, top-left origin (Blender convention; vibrt
    /// has always written its PNGs / EXRs assuming this layout).
    pub pixels: Vec<f32>,
    /// width × height single-channel primary-ray hit distance from the
    /// camera, in metres. Misses store `cam_clip_end`. Drives the addon's
    /// Mist / Z passes so Cycles-authored compositors (mist haze,
    /// depth-driven masks) work on top of vibrt's output.
    pub depth: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Lets the caller redirect progress messages and (later) cancel mid-render.
/// `log` is called for the same lines that today print to stdout from the CLI
/// (`Scene: ...`, `Rendering...`, `Render: ...`, etc.). `cancelled` is
/// reserved for Step 3 when the Python wrapper polls Blender's break flag —
/// the CLI's `StdoutProgress` always returns false.
pub trait Progress {
    fn log(&mut self, msg: &str);
    fn cancelled(&mut self) -> bool {
        false
    }
}

/// Default `Progress` impl for the CLI: lines go to stdout, no cancellation.
pub struct StdoutProgress;
impl Progress for StdoutProgress {
    fn log(&mut self, msg: &str) {
        println!("{msg}");
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

/// Light types that carry an optional IES profile. The trait gives
/// `upload_ies_buffers` a uniform way to patch each light's
/// `ies_data` / `ies_n_v` / `ies_n_h` fields without three near-
/// identical helpers.
trait HasIes {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32);
}
impl HasIes for crate::gpu_types::PointLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}
impl HasIes for crate::gpu_types::SpotLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}
impl HasIes for crate::gpu_types::AreaRectLight {
    fn set_ies(&mut self, data: optix_sys::CUdeviceptr, n_v: u32, n_h: u32) {
        self.ies_data = data;
        self.ies_n_v = n_v;
        self.ies_n_h = n_h;
    }
}

/// Pack each `Some(IesProfile)` into a flat float buffer
/// `[thetas (n_v) | phis (n_h) | normalised candelas (n_v * n_h)]`,
/// upload to the device, and patch the matching light's IES fields
/// to point at it. Returns the device pointers so the caller can keep
/// the allocations alive for the lifetime of the render.
///
/// The packing is what `ies_lookup` in devicecode.cu reads; keep the
/// two in sync.
fn upload_ies_buffers<L: HasIes>(
    stream: &Arc<CudaStream>,
    lights: &mut [L],
    profiles: &[Option<scene_format::IesProfile>],
) -> Result<Vec<optix_sys::CUdeviceptr>> {
    let mut out = Vec::new();
    for (i, profile) in profiles.iter().enumerate() {
        let Some(p) = profile else { continue; };
        let n_v = p.thetas_deg.len();
        let n_h = p.phis_deg.len().max(1);
        if n_v == 0 || p.candelas.len() != n_v * p.phis_deg.len() {
            // Malformed profile — silently skip rather than crashing
            // mid-render. The exporter is supposed to validate; this
            // is just defence in depth.
            continue;
        }
        let peak = p.peak_candela.max(1e-6);
        let mut buf: Vec<f32> = Vec::with_capacity(n_v + n_h + n_v * n_h);
        buf.extend_from_slice(&p.thetas_deg);
        // Synthesise a single phi=0 entry for radially-symmetric
        // profiles so the GPU lookup can read the phi axis without
        // a special case for n_h==1.
        if p.phis_deg.is_empty() {
            buf.push(0.0);
        } else {
            buf.extend_from_slice(&p.phis_deg);
        }
        for c in &p.candelas {
            buf.push(c / peak);
        }
        let d = alloc_and_copy_slice(stream, &buf)?;
        lights[i].set_ies(d, n_v as u32, n_h as u32);
        out.push(d);
    }
    Ok(out)
}

struct GasEntry {
    handle: optix_sys::OptixTraversableHandle,
    sbt_offset: u32,
    transform: [f32; 12],
    cast_shadow: bool,
    camera_visible: bool,
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

/// Render `scene` to an in-memory RGBA buffer.
///
/// Applies any non-`None` overrides from `opts` to the scene's render block,
/// then runs the same pipeline today's `render()` does — minus the final
/// `image_io::save_image` call, which is the CLI's responsibility now.
pub fn render_to_pixels(
    scene: &LoadedScene,
    opts: &RenderOptions,
    progress: &mut dyn Progress,
) -> Result<RenderOutput> {
    // Apply overrides into a local copy of the render settings. The caller's
    // LoadedScene is borrowed `&` so we can't mutate `scene.file.render`; the
    // copy keeps the override semantics that the old `main` had.
    let mut render_settings = scene.file.render;
    if let Some(v) = opts.spp {
        render_settings.spp = v;
    }
    if let Some(v) = opts.max_depth {
        render_settings.max_depth = v;
    }
    if let Some(v) = opts.clamp_indirect {
        render_settings.clamp_indirect = v;
    }
    if let Some(v) = opts.width {
        render_settings.width = v;
    }
    if let Some(v) = opts.height {
        render_settings.height = v;
    }
    render_settings.max_depth = render_settings.max_depth.min(31);

    progress.log(&format!(
        "Scene: {}x{} @ {} spp, {} objects, {} meshes, {} mats, {} textures, lights(p/s/spot/rect/mesh): {}/{}/{}/{}/{}",
        render_settings.width,
        render_settings.height,
        render_settings.spp,
        scene.objects.len(),
        scene.meshes.len(),
        scene.file.materials.len(),
        scene.textures.len(),
        scene.point_lights.len(),
        scene.sun_lights.len(),
        scene.spot_lights.len(),
        scene.rect_lights.len(),
        scene.mesh_lights.len(),
    ));

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
    // 4096 bytes for the continuation stack (was 2048): trace_path
    // stack-allocates a small array of MaterialEval slots (one per
    // secondary closure level) so nested MixShader chains can be
    // evaluated; with MAX_SECONDARY=3 in devicecode.cu, four
    // ~144-byte MaterialEvals plus the surrounding locals push past
    // 2048 on classroom and trip CUDA_ERROR_INVALID_PC. Doubling
    // restores headroom; OptiX recomputes per-thread stacks lazily so
    // there's no static cost when shallow scenes are rendered.
    pipeline.set_stack_size(2048, 2048, 4096, 2)?;

    // --- Upload textures (shared across materials) ---
    let mut _tex_buffers: Vec<CudaSlice<f32>> = Vec::new();
    let tex_slots = principled::upload_textures(&scene.textures, &stream, &mut _tex_buffers)?;

    // --- Upload materials ---
    // Colour-graph buffers (node records + per-node LUT data) live for the
    // lifetime of the render; stash them so their CudaSlice drops are
    // deferred. Separate u32 / f32 vectors because RGBCurve LUTs are float.
    let mut _color_graph_buffers: Vec<CudaSlice<u32>> = Vec::new();
    let mut _color_graph_lut_buffers: Vec<CudaSlice<f32>> = Vec::new();
    // Volume parameter blocks live alongside materials. Each material that
    // has a Volume slot gets one allocation; the device-side `PrincipledGpu`
    // holds a pointer back to it (or null when absent).
    let mut _volume_storage: Vec<optix_sys::CUdeviceptr> = Vec::new();
    let mut mat_device_ptrs: Vec<optix_sys::CUdeviceptr> = Vec::new();
    // Recursively upload one material node (and its left/right binary-tree
    // children, when present) to the device, returning the device pointer
    // to its PrincipledGpu. Each call may allocate its own color graph
    // buffer + volume buffer + child sub-trees. The depth cap is a sanity
    // guard against producer bugs — even a fully balanced 4-leaf tree only
    // recurses 3 levels deep, so >8 is well past any artist intent.
    fn upload_one_mat(
        mat: &scene_format::PrincipledMaterial,
        tex_slots: &[(optix_sys::CUdeviceptr, i32, i32)],
        stream: &Arc<CudaStream>,
        color_graph_buffers: &mut Vec<CudaSlice<u32>>,
        color_graph_lut_buffers: &mut Vec<CudaSlice<f32>>,
        volume_storage: &mut Vec<optix_sys::CUdeviceptr>,
        depth: u32,
    ) -> Result<optix_sys::CUdeviceptr> {
        if depth > 8 {
            anyhow::bail!(
                "PrincipledMaterial Mix tree deeper than 8 levels — likely a \
                 producer-side bug; refusing to recurse further"
            );
        }
        let graph = match &mat.color_graph {
            Some(g) => principled::upload_color_graph(
                g,
                tex_slots,
                stream,
                color_graph_buffers,
                color_graph_lut_buffers,
            )?,
            None => principled::ColorGraphGpu::default(),
        };
        let volume_ptr = match &mat.volume {
            Some(v) => {
                let vg = principled::make_volume_gpu(v);
                let p = alloc_and_copy(stream, &vg)?;
                volume_storage.push(p);
                p
            }
            None => 0,
        };
        let secondary_ptr = match &mat.secondary {
            Some(sec) => upload_one_mat(
                sec,
                tex_slots,
                stream,
                color_graph_buffers,
                color_graph_lut_buffers,
                volume_storage,
                depth + 1,
            )?,
            None => 0,
        };
        let left_subtree_ptr = match &mat.left_subtree {
            Some(lhs) => upload_one_mat(
                lhs,
                tex_slots,
                stream,
                color_graph_buffers,
                color_graph_lut_buffers,
                volume_storage,
                depth + 1,
            )?,
            None => 0,
        };
        let gpu = principled::make_material_data(
            mat, tex_slots, graph, volume_ptr, secondary_ptr, left_subtree_ptr,
        );
        Ok(alloc_and_copy(stream, &gpu)?)
    }
    for mat in &scene.file.materials {
        let p = upload_one_mat(
            mat,
            &tex_slots,
            &stream,
            &mut _color_graph_buffers,
            &mut _color_graph_lut_buffers,
            &mut _volume_storage,
            0,
        )?;
        mat_device_ptrs.push(p);
    }
    // World volume sits in LaunchParams and is shared by every ray that
    // doesn't enter a tighter mesh-bounded volume. Allocated once.
    let world_volume_ptr: optix_sys::CUdeviceptr = match &scene.file.world_volume {
        Some(v) => alloc_and_copy(&stream, &principled::make_volume_gpu(v))?,
        None => 0,
    };

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
        d_tangents: optix_sys::CUdeviceptr,
        num_verts: i32,
    }
    let mut _buffers: Vec<CudaSlice<u8>> = Vec::new();
    let mut _f32_buffers: Vec<CudaSlice<f32>> = Vec::new();
    let mut _u32_buffers: Vec<CudaSlice<u32>> = Vec::new();
    let mut meshes_gpu: Vec<MeshGpu> = Vec::new();

    for m in &scene.meshes {
        let d_verts = stream.clone_htod(&m.vertices[..]).cuda()?;
        let d_indices = stream.clone_htod(&m.indices[..]).cuda()?;
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
            let s = stream.clone_htod(&m.normals[..]).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };
        let d_uvs = if !m.uvs.is_empty() {
            let s = stream.clone_htod(&m.uvs[..]).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };
        let d_mat_indices = if !m.material_indices.is_empty() {
            let s = stream.clone_htod(&m.material_indices[..]).cuda()?;
            let p = dptr_u32(&s, &stream);
            _u32_buffers.push(s);
            p
        } else {
            0
        };
        let d_vertex_colors = if !m.vertex_colors.is_empty() {
            let s = stream.clone_htod(&m.vertex_colors[..]).cuda()?;
            let p = dptr_f32(&s, &stream);
            _f32_buffers.push(s);
            p
        } else {
            0
        };
        let d_tangents = if !m.tangents.is_empty() {
            let s = stream.clone_htod(&m.tangents[..]).cuda()?;
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
            d_tangents,
            num_verts: num_verts as i32,
        });

        _f32_buffers.push(d_verts);
        _u32_buffers.push(d_indices);
        _buffers.push(d_temp);
        _buffers.push(d_output);
    }
    stream.synchronize().cuda()?;

    // Reverse mapping object_idx → mesh_light_idx (or -1 when the object
    // isn't registered as a mesh-light). Built before the SBT loop so each
    // HitGroupData can carry the right `area_light_group`. The mesh-light
    // device upload itself happens later (after light gathering) — it
    // only consumes the table; doesn't write to it.
    let object_to_mesh_light: Vec<i32> = {
        let mut m = vec![-1i32; scene.objects.len()];
        for (i, ml) in scene.mesh_lights.iter().enumerate() {
            if let Some(slot) = m.get_mut(ml.object_idx as usize) {
                *slot = i as i32;
            }
        }
        m
    };

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
                .map(|mi| match mat_device_ptrs.get(*mi as usize).copied() {
                    Some(p) => p as u64,
                    None => {
                        eprintln!(
                            "[vibrt] warn: object {} per-tri material index {} out of range \
                             (have {} materials) — falling back to object's primary material",
                            obj_idx,
                            mi,
                            mat_device_ptrs.len()
                        );
                        mat_ptr as u64
                    }
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
            // Mesh-light index for this instance, or -1 when the object
            // isn't registered for mesh-light NEE. The closest-hit shader
            // checks `>= 0` to enable MIS-weighting of BSDF-sampled
            // emission against the NEE sampler.
            area_light_group: object_to_mesh_light
                .get(obj_idx)
                .copied()
                .unwrap_or(-1),
            material_indices: m.d_mat_indices,
            materials: materials_ptr,
            num_materials,
            vertex_colors: m.d_vertex_colors,
            tangents: m.d_tangents,
        };
        tri_hg_records.push(SbtRecord::new(&hit_rg_pg, hg)?);
        tri_hg_records.push(SbtRecord::new(&hit_shadow_pg, hg)?);
        gas_entries.push(GasEntry {
            handle: m.handle,
            sbt_offset: (obj_idx as u32) * 2,
            transform: obj.transform,
            cast_shadow: scene.file.objects[obj_idx].cast_shadow,
            camera_visible: scene.file.objects[obj_idx].camera_visible,
        });
    }

    // --- Build IAS ---
    if gas_entries.is_empty() {
        anyhow::bail!("no geometry in scene");
    }
    // Visibility mask bits (mirrored in `scene_loader::VIS_MASK_*`):
    //   0x01 = primary radiance (camera ray + specular bounce chain)
    //   0x02 = shadow ray (= "blocks NEE shadow rays from receivers"
    //          when AND-masked into the ray's mask)
    //   0x04 = secondary radiance (diffuse bounce continuation)
    // Camera-hidden emissive meshes drop the 0x01 bit so camera/specular
    // rays pass straight through, while still appearing on diffuse-bounce
    // rays + appearing as a NEE source via mesh-lights.
    let instances: Vec<optix_sys::OptixInstance> = gas_entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut mask: u32 = scene_loader::VIS_MASK_SECONDARY;
            if e.camera_visible {
                mask |= scene_loader::VIS_MASK_PRIMARY;
            }
            if e.cast_shadow {
                mask |= scene_loader::VIS_MASK_SHADOW;
            }
            optix_sys::OptixInstance {
                transform: e.transform,
                instanceId: i as u32,
                sbtOffset: e.sbt_offset,
                visibilityMask: mask,
                flags: optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE.0 as u32,
                traversableHandle: e.handle,
                pad: [0; 2],
            }
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
    let aspect = render_settings.width as f32 / render_settings.height as f32;
    let (eye, cam_u, cam_v, cam_w) = camera::compute_camera(
        &scene.file.camera.transform,
        scene.file.camera.fov_y_rad,
        aspect,
    );

    // --- Lights ---
    // IES profiles must be uploaded BEFORE the light vectors are
    // copied to the device, so each light struct carries its IES
    // device pointer when it lands on the GPU. Patch the host-side
    // copies in place; the upstream (LoadedScene) is consumed below
    // anyway. `_d_ies_*` keep the device buffers alive for the
    // lifetime of the render — drop on render exit.
    let mut point_lights = scene.point_lights.clone();
    let mut spot_lights = scene.spot_lights.clone();
    let mut rect_lights = scene.rect_lights.clone();
    let _d_ies_point = upload_ies_buffers(&stream, &mut point_lights, &scene.point_ies)?;
    let _d_ies_spot = upload_ies_buffers(&stream, &mut spot_lights, &scene.spot_ies)?;
    let _d_ies_rect = upload_ies_buffers(&stream, &mut rect_lights, &scene.rect_ies)?;
    let d_points = alloc_and_copy_slice(&stream, &point_lights)?;
    let d_suns = alloc_and_copy_slice(&stream, &scene.sun_lights)?;
    let d_spots = alloc_and_copy_slice(&stream, &spot_lights)?;
    let d_rects = alloc_and_copy_slice(&stream, &rect_lights)?;
    let rect_cdf = build_rect_light_cdf(&rect_lights);
    let d_rect_cdf = alloc_and_copy_slice(&stream, &rect_cdf)?;

    // Mesh lights: convert each host triangle into the GPU `MeshLightTri`
    // POD, resolving `emission_tex_idx` to the device-side texture
    // pointer/dimensions from `tex_slots`. Then upload per-triangle
    // table + inner CDF, copy the outer `MeshLight` array, and build
    // the outer CDF over per-light total power.
    let mut mesh_lights_gpu: Vec<MeshLight> = Vec::with_capacity(scene.mesh_lights.len());
    let mut mesh_light_outer_cdf = vec![0.0f32; scene.mesh_lights.len() + 1];
    for (i, ml) in scene.mesh_lights.iter().enumerate() {
        let gpu_tris: Vec<MeshLightTri> = ml
            .triangles
            .iter()
            .map(|t| {
                let (tex_ptr, tex_w, tex_h, tex_c) = match t.emission_tex_idx {
                    Some(idx) => match tex_slots.get(idx as usize).copied() {
                        // tex_slots returns (CUdeviceptr, w, h); channels
                        // are always 4 in vibrt's texture pipeline (RGBA
                        // f32). Mirrors how `make_material_data` resolves
                        // texture slots in principled.rs.
                        Some((p, w, h)) => (p, w, h, 4),
                        None => (0, 0, 0, 0),
                    },
                    None => (0, 0, 0, 0),
                };
                MeshLightTri {
                    v0: t.v0,
                    _pad0: 0.0,
                    v1: t.v1,
                    _pad1: 0.0,
                    v2: t.v2,
                    area: t.area,
                    emission: t.emission,
                    _pad2: 0.0,
                    uv0: t.uv0,
                    uv1: t.uv1,
                    uv2: t.uv2,
                    _pad3: [0.0, 0.0],
                    emission_tex: tex_ptr,
                    emission_tex_w: tex_w,
                    emission_tex_h: tex_h,
                    emission_tex_channels: tex_c,
                    _pad4: [0, 0, 0],
                }
            })
            .collect();
        let d_tris = alloc_and_copy_slice(&stream, &gpu_tris)?;
        let d_cdf = alloc_and_copy_slice(&stream, &ml.cdf)?;
        mesh_lights_gpu.push(MeshLight {
            triangles: d_tris,
            cdf: d_cdf,
            num_triangles: ml.triangles.len() as u32,
            two_sided: if ml.two_sided { 1 } else { 0 },
            power: ml.power,
            _pad: 0,
        });
        mesh_light_outer_cdf[i + 1] = mesh_light_outer_cdf[i] + ml.power.max(0.0);
    }
    let mesh_light_total_power = *mesh_light_outer_cdf.last().unwrap_or(&0.0);
    if mesh_light_total_power > 0.0 {
        let inv = 1.0 / mesh_light_total_power;
        for v in mesh_light_outer_cdf.iter_mut().skip(1) {
            *v *= inv;
        }
    }
    let d_mesh_lights = if mesh_lights_gpu.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &mesh_lights_gpu)?
    };
    let d_mesh_light_cdf = if mesh_lights_gpu.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &mesh_light_outer_cdf)?
    };

    // --- World / envmap ---
    let (world_type, world_color, world_strength) = match &scene.file.world {
        Some(scene_format::WorldDesc::Constant { color, strength }) => (0, *color, *strength),
        Some(scene_format::WorldDesc::Envmap { strength, .. }) => (1, [0.0; 3], *strength),
        // For mixed worlds the per-layer strengths live on each
        // EnvmapLayer; world_strength is unused on the kernel side
        // (world_background takes the layer mix).
        Some(scene_format::WorldDesc::Mixed { .. }) => (2, [0.0; 3], 1.0),
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
    // Mixed-envmap layers (uploaded only when WorldDesc::Mixed).
    let (d_envmap_a, env_w_a, env_h_a) = if let Some((rgb, w, h)) =
        &scene.envmap_layer_a_rgb
    {
        let d = alloc_and_copy_slice(&stream, rgb)?;
        (d, *w as i32, *h as i32)
    } else {
        (0, 0, 0)
    };
    let (d_envmap_b, env_w_b, env_h_b) = if let Some((rgb, w, h)) =
        &scene.envmap_layer_b_rgb
    {
        let d = alloc_and_copy_slice(&stream, rgb)?;
        (d, *w as i32, *h as i32)
    } else {
        (0, 0, 0)
    };
    let (env_strength_a, env_strength_b, env_rot_a, env_rot_b, env_mix_fac, env_split_camera) =
        match &scene.file.world {
            Some(scene_format::WorldDesc::Mixed {
                a,
                b,
                fac,
                split_by_camera_ray,
            }) => (
                a.strength,
                b.strength,
                a.rotation,
                b.rotation,
                *fac,
                if *split_by_camera_ray { 1 } else { 0 },
            ),
            _ => (
                0.0,
                0.0,
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                0.0,
                0,
            ),
        };

    // --- GGX LUTs ---
    let (lut_e, lut_e_avg) = pipeline::generate_ggx_energy_lut();
    let d_ggx_e = alloc_and_copy_slice(&stream, &lut_e)?;
    let d_ggx_e_avg = alloc_and_copy_slice(&stream, &lut_e_avg)?;

    // --- Image buffer (float4 per pixel) ---
    let pixel_count = (render_settings.width * render_settings.height) as usize;
    let d_image: CudaSlice<f32> = stream.alloc_zeros(pixel_count * 4).cuda()?;

    // Denoiser guide AOVs: allocated (float3 per pixel) only when --denoise.
    let d_albedo: Option<CudaSlice<f32>> = if opts.denoise {
        Some(stream.alloc_zeros(pixel_count * 3).cuda()?)
    } else {
        None
    };
    let d_normal: Option<CudaSlice<f32>> = if opts.denoise {
        Some(stream.alloc_zeros(pixel_count * 3).cuda()?)
    } else {
        None
    };
    let albedo_aov = d_albedo.as_ref().map_or(0, |s| dptr_f32(s, &stream));
    let normal_aov = d_normal.as_ref().map_or(0, |s| dptr_f32(s, &stream));

    // Per-pixel primary-ray hit distance. Always allocated; consumed by
    // the addon to populate Mist / Z passes for the compositor.
    let d_depth: CudaSlice<f32> = stream.alloc_zeros(pixel_count).cuda()?;
    let depth_aov = dptr_f32(&d_depth, &stream);

    // Single u32 counter for the camera-ray back-face skip's exhausted-cap
    // event (see trace_path in devicecode.cu). Always allocated so the
    // kernel never has to null-check the pointer; the host warns after
    // the render if it came back non-zero.
    let d_back_face_warn: CudaSlice<u32> = stream.alloc_zeros(1).cuda()?;
    let primary_back_face_skip_exhausted = dptr_u32(&d_back_face_warn, &stream);

    let lp = LaunchParams {
        image: dptr_f32(&d_image, &stream),
        width: render_settings.width,
        height: render_settings.height,
        samples_per_pixel: render_settings.spp,
        max_depth: render_settings.max_depth,
        max_diffuse_bounces: render_settings.max_diffuse_bounces,
        max_glossy_bounces: render_settings.max_glossy_bounces,
        max_transmission_bounces: render_settings.max_transmission_bounces,
        cam_eye: eye,
        cam_u,
        cam_v,
        cam_w,
        cam_lens_radius: scene.file.camera.lens_radius,
        cam_focal_distance: scene.file.camera.focal_distance,
        cam_clip_start: scene.file.camera.clip_start,
        cam_clip_end: scene.file.camera.clip_end,
        cam_shift_x: scene.file.camera.shift_x,
        cam_shift_y: scene.file.camera.shift_y,
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
        num_mesh_lights: scene.mesh_lights.len() as i32,
        mesh_lights: d_mesh_lights,
        mesh_light_cdf: d_mesh_light_cdf,
        mesh_light_total_power,
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
        clamp_indirect: render_settings.clamp_indirect,
        clamp_direct: render_settings.clamp_direct,
        filter_glossy: render_settings.filter_glossy,
        envmap_data_a: d_envmap_a,
        envmap_width_a: env_w_a,
        envmap_height_a: env_h_a,
        envmap_data_b: d_envmap_b,
        envmap_width_b: env_w_b,
        envmap_height_b: env_h_b,
        envmap_strength_a: env_strength_a,
        envmap_strength_b: env_strength_b,
        envmap_rotation_a: env_rot_a,
        envmap_rotation_b: env_rot_b,
        envmap_mix_fac: env_mix_fac,
        envmap_split_by_camera_ray: env_split_camera,
        albedo_aov,
        normal_aov,
        depth_aov,
        world_volume: world_volume_ptr,
        primary_back_face_skip_exhausted,
    };
    let d_params = alloc_and_copy(&stream, &lp)?;

    progress.log("Rendering...");
    let t0 = std::time::Instant::now();
    pipeline
        .launch(
            cu_stream,
            d_params,
            mem::size_of::<LaunchParams>(),
            &sbt,
            render_settings.width,
            render_settings.height,
            1,
        )
        .context("launch")?;
    stream.synchronize().cuda()?;
    progress.log(&format!("Render: {:.2?}", t0.elapsed()));

    // Read back the camera-ray back-face skip counter. Non-zero means the
    // camera ray hit > 4 back-facing surfaces in a row on at least one
    // sample — clip_start was set inside an unusually deep stack of
    // hidden geometry, and the affected pixels fell back to shading the
    // last back-face we hit. Emit one warn line so this isn't silent.
    let back_face_warn: Vec<u32> =
        stream.clone_dtoh(&d_back_face_warn).cuda()?;
    if let Some(&n) = back_face_warn.first() {
        if n > 0 {
            let spp = render_settings.spp.max(1) as f64;
            let est_pixels = (n as f64 / spp).round() as u64;
            progress.log(&format!(
                "[vibrt] warn: camera-ray back-face skip exhausted its retry \
                 cap on {n} samples (~{est_pixels} pixels at {spp} spp); \
                 those pixels fall back to shading the last back-face. \
                 Likely cause: clip_start lands the camera inside thick / \
                 deeply-nested geometry."
            ));
        }
    }

    // --- Optional OptiX denoiser (AOV model, HDR float4, no guides) ---
    let d_final = if opts.denoise {
        let t_dn = std::time::Instant::now();
        let w = render_settings.width;
        let h = render_settings.height;

        let denoiser = denoiser::Denoiser::new(
            &ctx,
            DenoiserModelKind::Aov,
            &denoiser::DenoiserOptions {
                guide_albedo: true,
                guide_normal: true,
                ..Default::default()
            },
        )
        .context("denoiser create")?;

        let sizes = denoiser
            .compute_memory_resources(w, h)
            .context("denoiser sizes")?;
        let d_state: CudaSlice<u8> = unsafe { stream.alloc(sizes.state_size) }.cuda()?;
        let d_scratch: CudaSlice<u8> =
            unsafe { stream.alloc(sizes.without_overlap_scratch_size) }.cuda()?;
        let d_intensity_scratch: CudaSlice<u8> =
            unsafe { stream.alloc(sizes.compute_intensity_size) }.cuda()?;
        let d_intensity: CudaSlice<f32> = stream.alloc_zeros(1).cuda()?;
        let d_denoised: CudaSlice<f32> = stream.alloc_zeros(pixel_count * 4).cuda()?;

        denoiser
            .setup(
                cu_stream,
                w,
                h,
                dptr_u8(&d_state, &stream),
                sizes.state_size,
                dptr_u8(&d_scratch, &stream),
                sizes.without_overlap_scratch_size,
            )
            .context("denoiser setup")?;

        let pixel_stride = 4 * mem::size_of::<f32>() as u32;
        let input_image = denoiser::Image2D {
            data: dptr_f32(&d_image, &stream),
            width: w,
            height: h,
            row_stride: w * pixel_stride,
            pixel_stride,
            format: PixelFormat::Float4,
        };
        let output_image = denoiser::Image2D {
            data: dptr_f32(&d_denoised, &stream),
            ..input_image
        };

        denoiser
            .compute_intensity(
                cu_stream,
                &input_image,
                dptr_f32(&d_intensity, &stream),
                dptr_u8(&d_intensity_scratch, &stream),
                sizes.compute_intensity_size,
            )
            .context("denoiser compute_intensity")?;

        let params = denoiser::DenoiserParams {
            hdr_intensity: dptr_f32(&d_intensity, &stream),
            blend_factor: 0.0,
            hdr_average_color: 0,
            temporal_mode_use_previous_layers: false,
        };
        let layer = denoiser::DenoiserLayer {
            input: input_image,
            output: output_image,
            previous_output: None,
        };

        let f3_stride = 3 * mem::size_of::<f32>() as u32;
        let guide_albedo_img = d_albedo.as_ref().map(|s| denoiser::Image2D {
            data: dptr_f32(s, &stream),
            width: w,
            height: h,
            row_stride: w * f3_stride,
            pixel_stride: f3_stride,
            format: PixelFormat::Float3,
        });
        let guide_normal_img = d_normal.as_ref().map(|s| denoiser::Image2D {
            data: dptr_f32(s, &stream),
            width: w,
            height: h,
            row_stride: w * f3_stride,
            pixel_stride: f3_stride,
            format: PixelFormat::Float3,
        });
        let guide = denoiser::DenoiserGuideLayer {
            albedo: guide_albedo_img,
            normal: guide_normal_img,
            flow: None,
        };

        denoiser
            .invoke(
                cu_stream,
                &params,
                dptr_u8(&d_state, &stream),
                sizes.state_size,
                &guide,
                &[layer],
                0,
                0,
                dptr_u8(&d_scratch, &stream),
                sizes.without_overlap_scratch_size,
            )
            .context("denoiser invoke")?;
        stream.synchronize().cuda()?;
        progress.log(&format!("Denoise: {:.2?}", t_dn.elapsed()));

        // Keep scratch/state buffers alive until the stream has caught up.
        drop((d_state, d_scratch, d_intensity_scratch, d_intensity));
        d_denoised
    } else {
        d_image
    };

    // --- Readback ---
    let pixels: Vec<f32> = stream.clone_dtoh(&d_final).cuda()?;
    let depth: Vec<f32> = stream.clone_dtoh(&d_depth).cuda()?;
    Ok(RenderOutput {
        pixels,
        depth,
        width: render_settings.width,
        height: render_settings.height,
    })
}
