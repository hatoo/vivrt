//! GPU struct types shared between host and device code.
//! Must match `devicecode.h` layout exactly.

pub const MAT_DIFFUSE: i32 = 0;
pub const MAT_DIELECTRIC: i32 = 1;
pub const MAT_COATED_DIFFUSE: i32 = 2;
pub const MAT_CONDUCTOR: i32 = 3;
pub const MAT_COATED_CONDUCTOR: i32 = 4;

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
    pub tint: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ConductorParams {
    pub eta: [f32; 3],
    pub k: [f32; 3],
}

/// Union of material-specific parameters.
#[repr(C)]
#[derive(Copy, Clone)]
pub union MaterialParams {
    pub diffuse: DiffuseParams,
    pub dielectric: DielectricParams,
    pub conductor: ConductorParams,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HitGroupData {
    // Material
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    pub roughness: f32,
    pub roughness_v: f32,
    pub coat_roughness: f32,
    pub coat_eta: f32,
    pub params: MaterialParams,
    // Geometry
    pub vertices: optix_sys::CUdeviceptr,
    pub normals: optix_sys::CUdeviceptr,
    pub indices: optix_sys::CUdeviceptr,
    pub texcoords: optix_sys::CUdeviceptr,
    pub num_vertices: i32,
    // Texture maps
    pub texture_data: optix_sys::CUdeviceptr,
    pub texture_width: i32,
    pub texture_height: i32,
    pub bump_data: optix_sys::CUdeviceptr,
    pub bump_width: i32,
    pub bump_height: i32,
    pub alpha_data: optix_sys::CUdeviceptr,
    pub alpha_width: i32,
    pub alpha_height: i32,
    pub roughness_data: optix_sys::CUdeviceptr,
    pub roughness_width: i32,
    pub roughness_height: i32,
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
    // GGX energy compensation LUT (Kulla-Conty)
    pub ggx_e_lut: optix_sys::CUdeviceptr, // E(cosθ, α), 32x32 float
    pub ggx_e_avg_lut: optix_sys::CUdeviceptr, // E_avg(α), 32 float
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MissData {
    pub bg_color: [f32; 3],
}
