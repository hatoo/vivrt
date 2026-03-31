//! GPU struct types shared between host and device code.
//! Must match `devicecode.h` layout exactly.

pub const MAT_DIFFUSE: i32 = 0;
pub const MAT_DIELECTRIC: i32 = 1;
pub const MAT_COATED_DIFFUSE: i32 = 2;
pub const MAT_CONDUCTOR: i32 = 3;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DistantLight {
    pub direction: [f32; 3],
    pub emission: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SphereLight {
    pub center: [f32; 3],
    pub radius: f32,
    pub emission: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TriangleLight {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub emission: [f32; 3],
    pub normal: [f32; 3],
    pub area: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DiffuseParams {
    pub has_checkerboard: i32,
    pub checker_scale_u: f32,
    pub checker_scale_v: f32,
    pub checker_color1: [f32; 3],
    pub checker_color2: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DielectricParams {
    pub eta: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CoatedDiffuseParams {
    pub roughness: f32,
}

/// Union of material-specific parameters.
/// Size = max(DiffuseParams, DielectricParams, CoatedDiffuseParams).
#[repr(C)]
#[derive(Copy, Clone)]
pub union MaterialParams {
    pub diffuse: DiffuseParams,
    pub dielectric: DielectricParams,
    pub coated: CoatedDiffuseParams,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HitGroupData {
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    pub params: MaterialParams,
    pub texture_data: optix_sys::CUdeviceptr,
    pub texture_width: i32,
    pub texture_height: i32,
    pub texcoords: optix_sys::CUdeviceptr,
    pub normals: optix_sys::CUdeviceptr,
    pub indices: optix_sys::CUdeviceptr,
    pub vertices: optix_sys::CUdeviceptr,
    pub num_vertices: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LaunchParams {
    pub image: optix_sys::CUdeviceptr,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,
    pub cam_eye: [f32; 3],
    pub cam_u: [f32; 3],
    pub cam_v: [f32; 3],
    pub cam_w: [f32; 3],
    pub traversable: optix_sys::OptixTraversableHandle,
    pub ambient_light: [f32; 3],
    pub num_distant_lights: i32,
    pub distant_lights: optix_sys::CUdeviceptr,
    pub num_sphere_lights: i32,
    pub sphere_lights: optix_sys::CUdeviceptr,
    pub num_triangle_lights: i32,
    pub triangle_lights: optix_sys::CUdeviceptr,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MissData {
    pub bg_color: [f32; 3],
}
