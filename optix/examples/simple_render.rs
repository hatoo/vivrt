use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use optix::accel::{self, AccelBuildOptions, BuildInput, TriangleArrayInput};
use optix::*;
use std::mem;
use std::sync::Arc;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

// Must match devicecode.h layout exactly
#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    image: CUdeviceptr,
    image_width: u32,
    image_height: u32,
    cam_eye: [f32; 3],
    cam_u: [f32; 3],
    cam_v: [f32; 3],
    cam_w: [f32; 3],
    handle: OptixTraversableHandle,
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
#[derive(Copy, Clone)]
struct HitGroupData {}

/// Allocate device memory and copy a value into it. Returns raw CUdeviceptr.
fn alloc_and_copy<T>(stream: &Arc<cudarc::driver::CudaStream>, val: &T) -> CUdeviceptr {
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
        dptr as CUdeviceptr
    }
}

/// Get the raw CUdeviceptr from a CudaSlice.
fn dptr<T>(slice: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as CUdeviceptr
}

fn main() {
    // --- CUDA init (via cudarc) ---
    let cuda_ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = cuda_ctx.default_stream();
    let cu_stream = stream.cu_stream() as optix_sys::CUstream;

    // --- OptiX init ---
    let optix_handle = optix::init().expect("Failed to initialize OptiX");
    let ctx = DeviceContext::new(
        &optix_handle,
        cuda_ctx.cu_ctx() as optix_sys::CUcontext,
        &DeviceContextOptions::default(),
    )
    .expect("Failed to create OptiX context");

    // --- Module ---
    let pipeline_options = PipelineCompileOptions::new("params")
        .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
        .num_payload_values(3)
        .num_attribute_values(3);

    let ptx = compile_cu(
        std::path::Path::new(file!()).parent().unwrap(),
        "devicecode.cu",
    );
    let module = Module::new(
        &ctx,
        &ModuleCompileOptions::default(),
        &pipeline_options,
        ptx.as_bytes(),
    )
    .expect("Failed to create module")
    .value;

    // --- Program groups ---
    let raygen_pg = ProgramGroup::raygen(&ctx, &module, "__raygen__rg")
        .expect("raygen")
        .value;
    let miss_pg = ProgramGroup::miss(&ctx, &module, "__miss__ms")
        .expect("miss")
        .value;
    let hitgroup_pg = ProgramGroup::hitgroup(&ctx)
        .closest_hit(&module, "__closesthit__ch")
        .build()
        .expect("hitgroup")
        .value;

    // --- Pipeline ---
    let pipeline = Pipeline::new(
        &ctx,
        &pipeline_options,
        &PipelineLinkOptions { max_trace_depth: 1 },
        &[&raygen_pg, &miss_pg, &hitgroup_pg],
    )
    .expect("pipeline")
    .value;
    pipeline.set_stack_size(2048, 2048, 2048, 1).unwrap();

    // --- Acceleration structure ---
    let vertices: [f32; 9] = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0];
    let d_vertices = stream.clone_htod(&vertices).unwrap();
    let d_vertices_ptr = dptr(&d_vertices, &stream);
    let vertex_buffers = [d_vertices_ptr];
    let geo_flags = [GeometryFlags::NONE];

    let build_options = AccelBuildOptions {
        build_flags: BuildFlags::ALLOW_COMPACTION,
        operation: BuildOperation::Build,
    };

    let make_tri_input = || {
        TriangleArrayInput::new(
            &vertex_buffers,
            3,
            VertexFormat::Float3,
            3 * mem::size_of::<f32>() as u32,
            &geo_flags,
        )
    };

    let sizes = accel::accel_compute_memory_usage(
        &ctx,
        &build_options,
        &[BuildInput::Triangles(make_tri_input())],
    )
    .expect("accel memory");

    let d_temp: CudaSlice<u8> = unsafe { stream.alloc(sizes.temp_size) }.unwrap();
    let d_output: CudaSlice<u8> = unsafe { stream.alloc(sizes.output_size) }.unwrap();

    let gas_handle = accel::accel_build(
        &ctx,
        cu_stream,
        &build_options,
        &[BuildInput::Triangles(make_tri_input())],
        dptr(&d_temp, &stream),
        sizes.temp_size,
        dptr(&d_output, &stream),
        sizes.output_size,
    )
    .expect("accel build");

    stream.synchronize().unwrap();
    drop(d_temp);

    // --- SBT ---
    let raygen_record = SbtRecord::new(&raygen_pg, RayGenData {}).unwrap();
    let miss_record = SbtRecord::new(
        &miss_pg,
        MissData {
            bg_color: [0.1, 0.1, 0.3],
        },
    )
    .unwrap();
    let hitgroup_record = SbtRecord::new(&hitgroup_pg, HitGroupData {}).unwrap();

    let d_rg = alloc_and_copy(&stream, &raygen_record);
    let d_ms = alloc_and_copy(&stream, &miss_record);
    let d_hg = alloc_and_copy(&stream, &hitgroup_record);

    let sbt = ShaderBindingTableBuilder::new(d_rg)
        .miss_records(d_ms, mem::size_of_val(&miss_record) as u32, 1)
        .hitgroup_records(d_hg, mem::size_of_val(&hitgroup_record) as u32, 1)
        .build()
        .expect("SBT build");

    // --- Output image ---
    let pixel_count = (WIDTH * HEIGHT) as usize;
    let d_image: CudaSlice<u32> = stream.alloc_zeros(pixel_count).unwrap();

    let params = Params {
        image: dptr(&d_image, &stream),
        image_width: WIDTH,
        image_height: HEIGHT,
        cam_eye: [0.0, 0.0, 2.0],
        cam_u: [1.2, 0.0, 0.0],
        cam_v: [0.0, 0.9, 0.0],
        cam_w: [0.0, 0.0, -1.0],
        handle: gas_handle,
    };
    let d_params = alloc_and_copy(&stream, &params);

    // --- Launch ---
    println!("Launching OptiX render ({WIDTH} x {HEIGHT})...");
    pipeline
        .launch(
            cu_stream,
            d_params,
            mem::size_of::<Params>(),
            &sbt,
            WIDTH,
            HEIGHT,
            1,
        )
        .expect("launch");
    stream.synchronize().unwrap();

    // --- Download and save ---
    let pixels = stream.clone_dtoh(&d_image).unwrap();
    save_ppm("output.ppm", WIDTH, HEIGHT, &pixels);
    println!("Saved output.ppm ({WIDTH} x {HEIGHT})");
}

/// Compile a .cu file to PTX at runtime using NVRTC.
fn compile_cu(dir: &std::path::Path, filename: &str) -> String {
    let cu_path = dir.join(filename);
    let src = std::fs::read_to_string(&cu_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", cu_path.display()));

    let optix_include = find_optix_include();

    let opts = cudarc::nvrtc::CompileOptions {
        include_paths: vec![optix_include, dir.to_string_lossy().into_owned()],
        use_fast_math: Some(true),
        ..Default::default()
    };

    println!("Compiling {} with NVRTC...", cu_path.display());
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&src, opts)
        .unwrap_or_else(|e| panic!("NVRTC compilation failed:\n{e:?}"));
    ptx.to_src()
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
    panic!("OptiX SDK not found. Set OPTIX_ROOT environment variable.");
}

fn save_ppm(path: &str, width: u32, height: u32, pixels: &[u32]) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("Failed to create output file");
    write!(file, "P6\n{width} {height}\n255\n").unwrap();
    for y in (0..height).rev() {
        for x in 0..width {
            let pixel = pixels[(y * width + x) as usize];
            let r = (pixel & 0xFF) as u8;
            let g = ((pixel >> 8) & 0xFF) as u8;
            let b = ((pixel >> 16) & 0xFF) as u8;
            file.write_all(&[r, g, b]).unwrap();
        }
    }
}
