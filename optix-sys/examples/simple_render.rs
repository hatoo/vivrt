use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use optix_sys::*;
use std::ffi::c_void;
use std::mem;
use std::ptr;
use std::sync::Arc;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

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

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct SbtRecord<T: Copy> {
    header: [u8; OPTIX_SBT_RECORD_HEADER_SIZE],
    data: T,
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

macro_rules! optix_check {
    ($call:expr) => {{
        let result = $call;
        if result != OptixResult::OPTIX_SUCCESS {
            panic!("OptiX error: {} returned {:?}", stringify!($call), result.0);
        }
    }};
}

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

fn dptr<T>(slice: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr as CUdeviceptr
}

fn main() {
    let cuda_ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = cuda_ctx.default_stream();
    let cu_stream = stream.cu_stream() as CUstream;

    let optix = optix_init().expect("Failed to initialize OptiX");

    unsafe {
        // --- OptiX device context ---
        let mut ctx: OptixDeviceContext = ptr::null_mut();
        let ctx_options = OptixDeviceContextOptions::default();
        optix_check!((optix.optixDeviceContextCreate.unwrap())(
            cuda_ctx.cu_ctx() as CUcontext,
            &ctx_options,
            &mut ctx,
        ));

        // --- Module ---
        let ptx = compile_cu(
            std::path::Path::new(file!()).parent().unwrap(),
            "devicecode.cu",
        );
        let ptx_cstr = std::ffi::CString::new(ptx.as_str()).unwrap();

        let module_options = OptixModuleCompileOptions {
            maxRegisterCount: OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT as i32,
            optLevel: OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel: OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            ..Default::default()
        };

        let pipeline_options = OptixPipelineCompileOptions {
            usesMotionBlur: 0,
            traversableGraphFlags:
                OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS.0 as u32,
            numPayloadValues: 3,
            numAttributeValues: 3,
            exceptionFlags: OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE.0 as u32,
            pipelineLaunchParamsVariableName: b"params\0".as_ptr() as *const i8,
            ..Default::default()
        };

        let mut module: OptixModule = ptr::null_mut();
        let mut log = [0u8; 2048];
        let mut log_size = log.len();
        optix_check!((optix.optixModuleCreate.unwrap())(
            ctx,
            &module_options,
            &pipeline_options,
            ptx_cstr.as_ptr(),
            ptx.len(),
            log.as_mut_ptr() as *mut i8,
            &mut log_size,
            &mut module,
        ));

        // --- Program groups ---
        let pg_options = OptixProgramGroupOptions::default();
        let mut raygen_pg: OptixProgramGroup = ptr::null_mut();
        let mut miss_pg: OptixProgramGroup = ptr::null_mut();
        let mut hitgroup_pg: OptixProgramGroup = ptr::null_mut();

        {
            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    raygen: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: b"__raygen__rg\0".as_ptr() as *const i8,
                    },
                },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut raygen_pg,
            ));
        }

        {
            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: b"__miss__ms\0".as_ptr() as *const i8,
                    },
                },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut miss_pg,
            ));
        }

        {
            let mut hitgroup = OptixProgramGroupHitgroup::default();
            hitgroup.moduleCH = module;
            hitgroup.entryFunctionNameCH = b"__closesthit__ch\0".as_ptr() as *const i8;
            let desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 { hitgroup },
                flags: 0,
            };
            log_size = log.len();
            optix_check!((optix.optixProgramGroupCreate.unwrap())(
                ctx,
                &desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut hitgroup_pg,
            ));
        }

        // --- Pipeline ---
        let link_options = OptixPipelineLinkOptions { maxTraceDepth: 1 };
        let program_groups = [raygen_pg, miss_pg, hitgroup_pg];
        let mut pipeline: OptixPipeline = ptr::null_mut();
        log_size = log.len();
        optix_check!((optix.optixPipelineCreate.unwrap())(
            ctx,
            &pipeline_options,
            &link_options,
            program_groups.as_ptr(),
            program_groups.len() as u32,
            log.as_mut_ptr() as *mut i8,
            &mut log_size,
            &mut pipeline,
        ));
        optix_check!((optix.optixPipelineSetStackSize.unwrap())(
            pipeline, 2048, 2048, 2048, 1
        ));

        // --- Acceleration structure ---
        let vertices: [f32; 9] = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0];
        let d_vertices = stream.clone_htod(&vertices).unwrap();
        let d_vertices_ptr = dptr(&d_vertices, &stream);

        let flags = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE.0 as u32;
        let triangle_input = OptixBuildInputTriangleArray {
            vertexBuffers: &d_vertices_ptr,
            numVertices: 3,
            vertexFormat: OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes: 3 * mem::size_of::<f32>() as u32,
            flags: &flags,
            numSbtRecords: 1,
            ..Default::default()
        };

        let build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1 {
                triangleArray: triangle_input,
            },
        };

        let accel_options = OptixAccelBuildOptions {
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION.0 as u32,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            ..Default::default()
        };

        let mut buffer_sizes = OptixAccelBufferSizes::default();
        optix_check!((optix.optixAccelComputeMemoryUsage.unwrap())(
            ctx,
            &accel_options,
            &build_input,
            1,
            &mut buffer_sizes,
        ));

        let d_temp: CudaSlice<u8> = stream.alloc(buffer_sizes.tempSizeInBytes).unwrap();
        let d_output: CudaSlice<u8> = stream.alloc(buffer_sizes.outputSizeInBytes).unwrap();

        let mut gas_handle: OptixTraversableHandle = 0;
        optix_check!((optix.optixAccelBuild.unwrap())(
            ctx,
            cu_stream,
            &accel_options,
            &build_input,
            1,
            dptr(&d_temp, &stream),
            buffer_sizes.tempSizeInBytes,
            dptr(&d_output, &stream),
            buffer_sizes.outputSizeInBytes,
            &mut gas_handle,
            ptr::null(),
            0,
        ));
        stream.synchronize().unwrap();
        drop(d_temp);

        // --- SBT ---
        let mut raygen_record = SbtRecord {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: RayGenData {},
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            raygen_pg,
            raygen_record.header.as_mut_ptr() as *mut c_void
        ));

        let mut miss_record = SbtRecord {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: MissData {
                bg_color: [0.1, 0.1, 0.3],
            },
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            miss_pg,
            miss_record.header.as_mut_ptr() as *mut c_void
        ));

        let mut hitgroup_record = SbtRecord {
            header: [0u8; OPTIX_SBT_RECORD_HEADER_SIZE],
            data: HitGroupData {},
        };
        optix_check!((optix.optixSbtRecordPackHeader.unwrap())(
            hitgroup_pg,
            hitgroup_record.header.as_mut_ptr() as *mut c_void
        ));

        let d_rg = alloc_and_copy(&stream, &raygen_record);
        let d_ms = alloc_and_copy(&stream, &miss_record);
        let d_hg = alloc_and_copy(&stream, &hitgroup_record);

        let sbt = OptixShaderBindingTable {
            raygenRecord: d_rg,
            exceptionRecord: 0,
            missRecordBase: d_ms,
            missRecordStrideInBytes: mem::size_of_val(&miss_record) as u32,
            missRecordCount: 1,
            hitgroupRecordBase: d_hg,
            hitgroupRecordStrideInBytes: mem::size_of_val(&hitgroup_record) as u32,
            hitgroupRecordCount: 1,
            callablesRecordBase: 0,
            callablesRecordStrideInBytes: 0,
            callablesRecordCount: 0,
        };

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
        optix_check!((optix.optixLaunch.unwrap())(
            pipeline,
            cu_stream,
            d_params,
            mem::size_of::<Params>(),
            &sbt,
            WIDTH,
            HEIGHT,
            1,
        ));
        stream.synchronize().unwrap();

        // --- Download and save ---
        let pixels = stream.clone_dtoh(&d_image).unwrap();
        save_ppm("output.ppm", WIDTH, HEIGHT, &pixels);
        println!("Saved output.ppm ({WIDTH} x {HEIGHT})");

        // --- Cleanup ---
        (optix.optixPipelineDestroy.unwrap())(pipeline);
        (optix.optixProgramGroupDestroy.unwrap())(raygen_pg);
        (optix.optixProgramGroupDestroy.unwrap())(miss_pg);
        (optix.optixProgramGroupDestroy.unwrap())(hitgroup_pg);
        (optix.optixModuleDestroy.unwrap())(module);
        (optix.optixDeviceContextDestroy.unwrap())(ctx);
    }
}

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
