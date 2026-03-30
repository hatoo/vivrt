use clap::Parser;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix::*;
use pbrt_parser::{self, Directive, ParamType, ParamValue};
use std::mem;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "renderer",
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

// Must match devicecode.h
const MAT_DIFFUSE: i32 = 0;
const MAT_DIELECTRIC: i32 = 1;

#[repr(C)]
#[derive(Copy, Clone)]
struct DistantLight {
    direction: [f32; 3],
    emission: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LaunchParams {
    image: optix_sys::CUdeviceptr,
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    cam_eye: [f32; 3],
    cam_u: [f32; 3],
    cam_v: [f32; 3],
    cam_w: [f32; 3],
    traversable: optix_sys::OptixTraversableHandle,
    ambient_light: [f32; 3],
    num_distant_lights: i32,
    distant_lights: optix_sys::CUdeviceptr,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
struct MissData {
    bg_color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct HitGroupData {
    material_type: i32,
    albedo: [f32; 3],
    eta: f32,
    has_checkerboard: i32,
    checker_scale_u: f32,
    checker_scale_v: f32,
    checker_color1: [f32; 3],
    checker_color2: [f32; 3],
    texcoords: optix_sys::CUdeviceptr,
    normals: optix_sys::CUdeviceptr,
    indices: optix_sys::CUdeviceptr,
    vertices: optix_sys::CUdeviceptr,
    num_vertices: i32,
}

fn dptr<T>(slice: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> optix_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as optix_sys::CUdeviceptr
}

fn alloc_and_copy<T>(stream: &Arc<cudarc::driver::CudaStream>, val: &T) -> optix_sys::CUdeviceptr {
    let size = mem::size_of_val(val);
    let cu_stream = stream.cu_stream();
    unsafe {
        let dptr = cudarc::driver::result::malloc_async(cu_stream, size).unwrap();
        cudarc::driver::result::memcpy_htod_async(
            dptr,
            std::slice::from_raw_parts(val as *const T as *const u8, size),
            cu_stream,
        )
        .unwrap();
        dptr as optix_sys::CUdeviceptr
    }
}

fn alloc_and_copy_slice<T>(
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[T],
) -> optix_sys::CUdeviceptr {
    let size = mem::size_of_val(data);
    if size == 0 {
        return 0;
    }
    let cu_stream = stream.cu_stream();
    unsafe {
        let dptr = cudarc::driver::result::malloc_async(cu_stream, size).unwrap();
        cudarc::driver::result::memcpy_htod_async(
            dptr,
            std::slice::from_raw_parts(data.as_ptr() as *const u8, size),
            cu_stream,
        )
        .unwrap();
        dptr as optix_sys::CUdeviceptr
    }
}

// ---- Scene representation ----

struct SceneMaterial {
    material_type: i32,
    albedo: [f32; 3],
    eta: f32,
    has_checkerboard: bool,
    checker_scale_u: f32,
    checker_scale_v: f32,
    checker_color1: [f32; 3],
    checker_color2: [f32; 3],
}

impl Default for SceneMaterial {
    fn default() -> Self {
        Self {
            material_type: MAT_DIFFUSE,
            albedo: [0.5, 0.5, 0.5],
            eta: 1.5,
            has_checkerboard: false,
            checker_scale_u: 1.0,
            checker_scale_v: 1.0,
            checker_color1: [1.0, 1.0, 1.0],
            checker_color2: [0.0, 0.0, 0.0],
        }
    }
}

enum SceneShape {
    Sphere {
        radius: f32,
    },
    TriangleMesh {
        vertices: Vec<f32>,  // 3 per vertex
        indices: Vec<i32>,   // 3 per triangle
        texcoords: Vec<f32>, // 2 per vertex, may be empty
    },
}

struct SceneObject {
    shape: SceneShape,
    material: SceneMaterial,
    transform: [f32; 12], // 3x4 row-major (identity by default)
}

struct ParsedScene {
    width: u32,
    height: u32,
    fov: f32,
    cam_eye: [f32; 3],
    cam_look: [f32; 3],
    cam_up: [f32; 3],
    spp: u32,
    max_depth: u32,
    ambient_light: [f32; 3],
    distant_lights: Vec<DistantLight>,
    objects: Vec<SceneObject>,
    filename: String,
}

fn get_param_floats<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [f64]> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Floats(v) => Some(v.as_slice()),
            _ => None,
        })
}

fn get_param_float(params: &[pbrt_parser::Param], name: &str) -> Option<f32> {
    get_param_floats(params, name).and_then(|v| v.first().map(|x| *x as f32))
}

fn get_param_string<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
            _ => None,
        })
}

fn get_param_ints<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [i64]> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Ints(v) => Some(v.as_slice()),
            _ => None,
        })
}

fn get_param_rgb(params: &[pbrt_parser::Param], name: &str) -> Option<[f32; 3]> {
    // Check for "rgb" typed param or "spectrum" with floats
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Floats(v) if v.len() >= 3 => Some([v[0] as f32, v[1] as f32, v[2] as f32]),
            _ => None,
        })
}

fn get_param_texture_ref<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|p| p.name == name && p.ty == ParamType::Texture)
        .and_then(|p| match &p.value {
            ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
            _ => None,
        })
}

fn blackbody_to_rgb(kelvin: f32) -> [f32; 3] {
    // Approximate blackbody color (Tanner Helland approximation)
    let temp = kelvin / 100.0;
    let r = if temp <= 66.0 {
        1.0
    } else {
        let x = temp - 60.0;
        (329.699_f32 * x.powf(-0.13320_f32) / 255.0).clamp(0.0, 1.0)
    };
    let g = if temp <= 66.0 {
        let x = temp;
        (99.4708_f32 * x.ln() - 161.1196_f32) / 255.0
    } else {
        let x = temp - 60.0;
        288.1222_f32 * x.powf(-0.07551_f32) / 255.0
    }
    .clamp(0.0, 1.0);
    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let x = temp - 10.0;
        (138.5177_f32 * x.ln() - 305.0448_f32) / 255.0
    }
    .clamp(0.0, 1.0);
    [r, g, b]
}

fn identity_transform() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

fn parse_scene(input: &str) -> ParsedScene {
    let scene = pbrt_parser::parse(input).expect("Failed to parse PBRT scene");

    let mut parsed = ParsedScene {
        width: 400,
        height: 400,
        fov: 90.0,
        cam_eye: [0.0, 0.0, 0.0],
        cam_look: [0.0, 0.0, -1.0],
        cam_up: [0.0, 1.0, 0.0],
        spp: 16,
        max_depth: 5,
        ambient_light: [0.0; 3],
        distant_lights: Vec::new(),
        objects: Vec::new(),
        filename: "output.png".to_string(),
    };

    // Texture storage (name -> checkerboard params)
    struct CheckerTex {
        scale_u: f32,
        scale_v: f32,
        color1: [f32; 3],
        color2: [f32; 3],
    }
    let mut textures = std::collections::HashMap::<String, CheckerTex>::new();

    let mut current_material = SceneMaterial::default();
    let mut current_transform = identity_transform();
    let mut transform_stack: Vec<([f32; 12], SceneMaterial)> = Vec::new();

    for directive in &scene.directives {
        match directive {
            Directive::Film { params, .. } => {
                if let Some(v) = get_param_ints(params, "xresolution") {
                    parsed.width = v[0] as u32;
                }
                if let Some(v) = get_param_ints(params, "yresolution") {
                    parsed.height = v[0] as u32;
                }
                if let Some(s) = get_param_string(params, "filename") {
                    parsed.filename = s.to_string();
                }
            }
            Directive::Camera { params, .. } => {
                if let Some(f) = get_param_float(params, "fov") {
                    parsed.fov = f;
                }
            }
            Directive::LookAt { eye, look, up } => {
                parsed.cam_eye = [eye[0] as f32, eye[1] as f32, eye[2] as f32];
                parsed.cam_look = [look[0] as f32, look[1] as f32, look[2] as f32];
                parsed.cam_up = [up[0] as f32, up[1] as f32, up[2] as f32];
            }
            Directive::Sampler { params, .. } => {
                if let Some(v) = get_param_ints(params, "pixelsamples") {
                    parsed.spp = v[0] as u32;
                }
            }
            Directive::Integrator { params, .. } => {
                if let Some(v) = get_param_ints(params, "maxdepth") {
                    parsed.max_depth = v[0] as u32;
                }
            }
            Directive::WorldBegin => {
                current_transform = identity_transform();
            }
            Directive::AttributeBegin => {
                transform_stack.push((current_transform, current_material.clone()));
            }
            Directive::AttributeEnd => {
                if let Some((t, m)) = transform_stack.pop() {
                    current_transform = t;
                    current_material = m;
                }
            }
            Directive::Translate { v } => {
                current_transform[3] += v[0] as f32;
                current_transform[7] += v[1] as f32;
                current_transform[11] += v[2] as f32;
            }
            Directive::Identity => {
                current_transform = identity_transform();
            }
            Directive::LightSource { ty, params } => {
                match ty.as_str() {
                    "infinite" => {
                        if let Some(c) = get_param_rgb(params, "L") {
                            parsed.ambient_light = c;
                        }
                    }
                    "distant" => {
                        let from = get_param_floats(params, "from")
                            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                            .unwrap_or([0.0, 0.0, 1.0]);
                        let to = get_param_floats(params, "to")
                            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                            .unwrap_or([0.0, 0.0, 0.0]);
                        let dir = {
                            let dx = from[0] - to[0];
                            let dy = from[1] - to[1];
                            let dz = from[2] - to[2];
                            let len = (dx * dx + dy * dy + dz * dz).sqrt();
                            [dx / len, dy / len, dz / len]
                        };
                        // Check for blackbody or RGB emission
                        let mut emission = get_param_rgb(params, "L").unwrap_or([1.0, 1.0, 1.0]);
                        // Check for blackbody L
                        if let Some(p) = params
                            .iter()
                            .find(|p| p.name == "L" && p.ty == ParamType::Blackbody)
                        {
                            if let ParamValue::Floats(v) = &p.value {
                                if let Some(&k) = v.first() {
                                    emission = blackbody_to_rgb(k as f32);
                                }
                            }
                        }
                        let scale = get_param_float(params, "scale").unwrap_or(1.0);
                        emission[0] *= scale;
                        emission[1] *= scale;
                        emission[2] *= scale;
                        parsed.distant_lights.push(DistantLight {
                            direction: dir,
                            emission,
                        });
                    }
                    _ => eprintln!("Unsupported light type: {ty}"),
                }
            }
            Directive::Material { ty, params } => {
                current_material = SceneMaterial::default();
                match ty.as_str() {
                    "diffuse" => {
                        current_material.material_type = MAT_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") {
                            current_material.albedo = c;
                        }
                        // Check for texture reference
                        if let Some(tex_name) = get_param_texture_ref(params, "reflectance") {
                            if let Some(tex) = textures.get(tex_name) {
                                current_material.has_checkerboard = true;
                                current_material.checker_scale_u = tex.scale_u;
                                current_material.checker_scale_v = tex.scale_v;
                                current_material.checker_color1 = tex.color1;
                                current_material.checker_color2 = tex.color2;
                            }
                        }
                    }
                    "dielectric" => {
                        current_material.material_type = MAT_DIELECTRIC;
                        current_material.eta = get_param_float(params, "eta").unwrap_or(1.5);
                    }
                    _ => eprintln!("Unsupported material type: {ty}"),
                }
            }
            Directive::Texture {
                name,
                class,
                params,
                ..
            } => {
                if class == "checkerboard" {
                    let scale_u = get_param_float(params, "uscale").unwrap_or(1.0);
                    let scale_v = get_param_float(params, "vscale").unwrap_or(1.0);
                    let color1 = get_param_rgb(params, "tex1").unwrap_or([1.0, 1.0, 1.0]);
                    let color2 = get_param_rgb(params, "tex2").unwrap_or([0.0, 0.0, 0.0]);
                    textures.insert(
                        name.clone(),
                        CheckerTex {
                            scale_u,
                            scale_v,
                            color1,
                            color2,
                        },
                    );
                }
            }
            Directive::Shape { ty, params } => {
                let shape = match ty.as_str() {
                    "sphere" => {
                        let radius = get_param_float(params, "radius").unwrap_or(1.0);
                        SceneShape::Sphere { radius }
                    }
                    "trianglemesh" => {
                        let verts: Vec<f32> = get_param_floats(params, "P")
                            .map(|v| v.iter().map(|x| *x as f32).collect())
                            .unwrap_or_default();
                        let indices: Vec<i32> = get_param_ints(params, "indices")
                            .map(|v| v.iter().map(|x| *x as i32).collect())
                            .unwrap_or_default();
                        let texcoords: Vec<f32> = get_param_floats(params, "uv")
                            .map(|v| v.iter().map(|x| *x as f32).collect())
                            .unwrap_or_default();
                        SceneShape::TriangleMesh {
                            vertices: verts,
                            indices,
                            texcoords,
                        }
                    }
                    "bilinearmesh" => {
                        // Convert bilinear patch (4 vertices) to 2 triangles
                        let verts: Vec<f32> = get_param_floats(params, "P")
                            .map(|v| v.iter().map(|x| *x as f32).collect())
                            .unwrap_or_default();
                        let texcoords: Vec<f32> = get_param_floats(params, "uv")
                            .map(|v| v.iter().map(|x| *x as f32).collect())
                            .unwrap_or_default();
                        let indices = vec![0, 1, 2, 1, 3, 2];
                        SceneShape::TriangleMesh {
                            vertices: verts,
                            indices,
                            texcoords,
                        }
                    }
                    _ => {
                        eprintln!("Unsupported shape type: {ty}");
                        continue;
                    }
                };
                parsed.objects.push(SceneObject {
                    shape,
                    material: current_material.clone(),
                    transform: current_transform,
                });
            }
            _ => {}
        }
    }

    parsed
}

impl Clone for SceneMaterial {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

fn main() {
    let cli = Args::parse();

    let input = std::fs::read_to_string(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", cli.input);
        std::process::exit(1);
    });

    let mut scene = parse_scene(&input);

    // Apply CLI overrides
    if let Some(spp) = cli.spp {
        scene.spp = spp;
    }
    if let Some(depth) = cli.depth {
        scene.max_depth = depth;
    }
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
        "Scene: {}x{}, {} spp, {} objects, {} distant lights",
        scene.width,
        scene.height,
        scene.spp,
        scene.objects.len(),
        scene.distant_lights.len()
    );

    // --- CUDA / OptiX init ---
    let cuda_ctx = CudaContext::new(0).expect("CUDA context");
    let stream = cuda_ctx.default_stream();
    let cu_stream = stream.cu_stream() as optix_sys::CUstream;

    let optix_handle = optix::init().expect("OptiX init");
    let ctx = DeviceContext::new(
        &optix_handle,
        cuda_ctx.cu_ctx() as optix_sys::CUcontext,
        &DeviceContextOptions::default(),
    )
    .expect("OptiX context");

    // --- Compile PTX ---
    let cu_src = include_str!("devicecode.cu");
    let header_src = include_str!("devicecode.h");

    let optix_include = find_optix_include();
    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![optix_include],
        use_fast_math: Some(true),
        options: vec![format!(
            "--include-path={}",
            std::env::current_dir()
                .unwrap()
                .join("renderer/src")
                .display()
        )],
        ..Default::default()
    };

    // NVRTC doesn't support #include for local files easily, so inline the header
    let full_src = format!(
        "// Inlined devicecode.h\n{}\n// devicecode.cu\n{}",
        header_src,
        cu_src.replace("#include \"devicecode.h\"", "// (inlined above)")
    );

    println!("Compiling device code with NVRTC...");
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&full_src, opts).unwrap_or_else(|e| {
        eprintln!("NVRTC failed: {e:?}");
        std::process::exit(1);
    });
    let ptx_src = ptx.to_src();

    // --- OptiX pipeline ---
    let has_spheres = scene
        .objects
        .iter()
        .any(|o| matches!(o.shape, SceneShape::Sphere { .. }));

    let mut prim_flags = PrimitiveTypeFlags::TRIANGLE;
    if has_spheres {
        prim_flags |= PrimitiveTypeFlags::SPHERE;
    }

    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(if has_spheres {
            TraversableGraphFlags::ALLOW_SINGLE_LEVEL_INSTANCING
        } else {
            TraversableGraphFlags::ALLOW_SINGLE_GAS
        })
        .num_payload_values(10)
        .num_attribute_values(2)
        .uses_primitive_type_flags(prim_flags);

    let module_opts = ModuleCompileOptions::default();
    let module = Module::new(&ctx, &module_opts, &pipeline_options, ptx_src.as_bytes())
        .expect("module")
        .value;

    // Get built-in sphere intersection module if needed
    let sphere_is_module = if has_spheres {
        Some(
            Module::builtin_is(&ctx, &module_opts, &pipeline_options, PrimitiveType::Sphere)
                .expect("sphere IS module"),
        )
    } else {
        None
    };

    let raygen_pg = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")
        .expect("raygen")
        .value;
    let miss_pg = ProgramGroup::miss(&ctx, &module, "__miss__ms")
        .expect("miss")
        .value;

    let hitgroup_tri_pg = ProgramGroup::hitgroup(&ctx)
        .closest_hit(&module, "__closesthit__ch")
        .build()
        .expect("hitgroup_tri")
        .value;

    let hitgroup_sphere_pg = if let Some(ref is_mod) = sphere_is_module {
        Some(
            ProgramGroup::hitgroup(&ctx)
                .closest_hit(&module, "__closesthit__sphere")
                .intersection(is_mod, "")
                .build()
                .expect("hitgroup_sphere")
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
    .expect("pipeline")
    .value;
    pipeline.set_stack_size(2048, 2048, 2048, 2).unwrap();

    // --- Build geometry ---
    // Separate GASes for triangles and spheres (OptiX requires same prim type per GAS)
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
                // Single sphere: center at origin (transform applied via IAS instance)
                let center = [0.0f32, 0.0, 0.0];
                let radius_val = [*radius];

                let d_center = stream.clone_htod(&center).unwrap();
                let d_radius = stream.clone_htod(&radius_val).unwrap();
                let center_ptrs = [dptr(&d_center, &stream)];
                let radius_ptrs = [dptr(&d_radius, &stream)];
                let flags = [GeometryFlags::NONE];

                let sphere_input = accel::SphereArrayInput {
                    vertex_buffers: &center_ptrs,
                    vertex_stride: 0,
                    num_vertices: 1,
                    radius_buffers: &radius_ptrs,
                    radius_stride: 0,
                    single_radius: true,
                    flags: &flags,
                    num_sbt_records: 1,
                };

                let bi = [BuildInput::Spheres(sphere_input)];
                let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &bi)
                    .expect("sphere accel memory");
                let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.unwrap();
                let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.unwrap();

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
                .expect("sphere accel build");

                let hg_data = HitGroupData {
                    material_type: obj.material.material_type,
                    albedo: obj.material.albedo,
                    eta: obj.material.eta,
                    has_checkerboard: 0,
                    checker_scale_u: 0.0,
                    checker_scale_v: 0.0,
                    checker_color1: [0.0; 3],
                    checker_color2: [0.0; 3],
                    texcoords: 0,
                    normals: 0,
                    indices: 0,
                    vertices: 0,
                    num_vertices: 0,
                };

                let sbt_offset = sphere_hg_records.len() as u32;
                sphere_hg_records
                    .push(SbtRecord::new(hitgroup_sphere_pg.as_ref().unwrap(), hg_data).unwrap());

                gas_entries.push(GasEntry {
                    handle,
                    sbt_offset,
                    transform: obj.transform,
                    is_sphere: true,
                });

                _device_buffers.push(unsafe { std::mem::transmute(d_center) });
                _device_buffers.push(unsafe { std::mem::transmute(d_radius) });
                _device_buffers.push(d_temp);
                _device_buffers.push(d_output);
            }
            SceneShape::TriangleMesh {
                vertices,
                indices,
                texcoords,
            } => {
                let transformed = transform_vertices(vertices, &obj.transform);
                let d_verts = stream.clone_htod(&transformed).unwrap();
                let d_indices: CudaSlice<i32> = stream.clone_htod(indices).unwrap();
                let d_tc = if !texcoords.is_empty() {
                    let s = stream.clone_htod(texcoords).unwrap();
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
                    .expect("tri accel memory");
                let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.unwrap();
                let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.unwrap();

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
                .expect("tri accel build");

                let hg_data = HitGroupData {
                    material_type: obj.material.material_type,
                    albedo: obj.material.albedo,
                    eta: obj.material.eta,
                    has_checkerboard: if obj.material.has_checkerboard { 1 } else { 0 },
                    checker_scale_u: obj.material.checker_scale_u,
                    checker_scale_v: obj.material.checker_scale_v,
                    checker_color1: obj.material.checker_color1,
                    checker_color2: obj.material.checker_color2,
                    texcoords: d_tc,
                    normals: 0,
                    indices: dptr(&d_indices, &stream),
                    vertices: dptr(&d_verts, &stream),
                    num_vertices: num_verts as i32,
                };

                let sbt_offset = tri_hg_records.len() as u32;
                tri_hg_records.push(SbtRecord::new(&hitgroup_tri_pg, hg_data).unwrap());

                gas_entries.push(GasEntry {
                    handle,
                    sbt_offset,
                    transform: identity_transform(), // vertices already pre-transformed
                    is_sphere: false,
                });

                _device_buffers.push(unsafe { std::mem::transmute(d_verts) });
                _device_buffers.push(unsafe { std::mem::transmute(d_indices) });
                _device_buffers.push(d_temp);
                _device_buffers.push(d_output);
            }
        }
    }

    stream.synchronize().unwrap();

    // --- Build IAS (or use single GAS if no spheres) ---
    let traversable = if has_spheres {
        // SBT layout: [tri_hg_records..., sphere_hg_records...]
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
                let inst = optix_sys::OptixInstance {
                    transform: entry.transform,
                    instanceId: i as u32,
                    sbtOffset: sbt_offset,
                    visibilityMask: 255,
                    flags: optix_sys::OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE.0 as u32,
                    traversableHandle: entry.handle,
                    pad: [0; 2],
                };
                inst
            })
            .collect();

        let d_instances = alloc_and_copy_slice(&stream, &instances);

        let ias_input = [BuildInput::Instances(accel::InstanceArrayInput {
            instances: d_instances,
            num_instances: instances.len() as u32,
        })];

        let sizes = accel::accel_compute_memory_usage(&ctx, &build_options, &ias_input)
            .expect("IAS memory");
        let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.unwrap();
        let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.unwrap();

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
        .expect("IAS build");
        stream.synchronize().unwrap();

        _device_buffers.push(d_temp);
        _device_buffers.push(d_output);

        ias_handle
    } else {
        gas_entries[0].handle
    };

    // --- SBT ---
    let raygen_record = SbtRecord::new(&raygen_pg, RayGenData {}).unwrap();
    let miss_record = SbtRecord::new(
        &miss_pg,
        MissData {
            bg_color: [0.0, 0.0, 0.0],
        },
    )
    .unwrap();

    let d_rg = alloc_and_copy(&stream, &raygen_record);
    let d_ms = alloc_and_copy(&stream, &miss_record);

    // Hitgroup records: triangles first, then spheres
    let mut all_hg_records: Vec<u8> = Vec::new();
    let hg_stride = mem::size_of::<SbtRecord<HitGroupData>>();
    for rec in &tri_hg_records {
        let bytes = unsafe { std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride) };
        all_hg_records.extend_from_slice(bytes);
    }
    for rec in &sphere_hg_records {
        let bytes = unsafe { std::slice::from_raw_parts(rec as *const _ as *const u8, hg_stride) };
        all_hg_records.extend_from_slice(bytes);
    }
    let total_hg_count = tri_hg_records.len() + sphere_hg_records.len();
    let d_hg = alloc_and_copy_slice(&stream, &all_hg_records);

    let sbt = ShaderBindingTableBuilder::new(d_rg)
        .miss_records(d_ms, mem::size_of_val(&miss_record) as u32, 1)
        .hitgroup_records(d_hg, hg_stride as u32, total_hg_count as u32)
        .build()
        .expect("SBT");

    // --- Camera ---
    let (cam_u, cam_v, cam_w) = compute_camera(
        &scene.cam_eye,
        &scene.cam_look,
        &scene.cam_up,
        scene.fov,
        scene.width as f32 / scene.height as f32,
    );

    // --- Distant lights on device ---
    let d_lights = if scene.distant_lights.is_empty() {
        0
    } else {
        alloc_and_copy_slice(&stream, &scene.distant_lights)
    };

    // --- Output image ---
    let pixel_count = (scene.width * scene.height) as usize;
    let d_image: CudaSlice<u32> = stream.alloc_zeros(pixel_count).unwrap();

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
        distant_lights: d_lights,
    };
    let d_params = alloc_and_copy(&stream, &launch_params);

    // --- Launch ---
    println!(
        "Rendering {}x{} @ {} spp...",
        scene.width, scene.height, scene.spp
    );
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
        .expect("launch");
    stream.synchronize().unwrap();

    // --- Download and save ---
    let pixels = stream.clone_dtoh(&d_image).unwrap();

    let mut output_file = scene.filename.replace(".exr", ".png");
    if !output_file.ends_with(".png") && !output_file.ends_with(".ppm") {
        output_file = format!("{output_file}.png");
    }
    save_image(&output_file, scene.width, scene.height, &pixels);
    println!("Saved {output_file}");
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
        w[1] * up[2] - w[2] * up[1],
        w[2] * up[0] - w[0] * up[2],
        w[0] * up[1] - w[1] * up[0],
    ];
    let ulen = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let u = [u[0] / ulen, u[1] / ulen, u[2] / ulen];

    let v = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];

    let half_h = (fov.to_radians() * 0.5).tan();
    let half_w = aspect * half_h;

    let cam_u = [u[0] * half_w, u[1] * half_w, u[2] * half_w];
    let cam_v = [v[0] * half_h, v[1] * half_h, v[2] * half_h];
    let cam_w = w;

    (cam_u, cam_v, cam_w)
}

fn transform_vertices(verts: &[f32], t: &[f32; 12]) -> Vec<f32> {
    let mut result = Vec::with_capacity(verts.len());
    for i in (0..verts.len()).step_by(3) {
        let x = verts[i];
        let y = verts[i + 1];
        let z = verts[i + 2];
        result.push(t[0] * x + t[1] * y + t[2] * z + t[3]);
        result.push(t[4] * x + t[5] * y + t[6] * z + t[7]);
        result.push(t[8] * x + t[9] * y + t[10] * z + t[11]);
    }
    result
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

fn save_image(path: &str, width: u32, height: u32, pixels: &[u32]) {
    // Convert ABGR packed pixels to RGB bytes (flipped vertically)
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
        let mut file = std::fs::File::create(path).expect("Failed to create output file");
        write!(file, "P6\n{width} {height}\n255\n").unwrap();
        file.write_all(&rgb).unwrap();
    } else {
        let file = std::fs::File::create(path).expect("Failed to create output file");
        let w = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("Failed to write PNG header");
        writer
            .write_image_data(&rgb)
            .expect("Failed to write PNG data");
    }
}
