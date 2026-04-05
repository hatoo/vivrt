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
    pub i0: i32,
    pub i1: i32,
    pub i2: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TriangleLightGroup {
    pub start: u32,         // index into triangle_lights array
    pub count: u32,         // number of triangles in this group
    pub total_power: f32,   // sum of area * luminance
    pub vertex_offset: u32, // offset into triangle_light_vertices (in vertex count)
    pub emission: [f32; 3], // shared emission for all triangles in group
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
pub struct MaterialData {
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    pub roughness: f32,
    pub roughness_v: f32,
    pub coat_roughness: f32,
    pub coat_eta: f32,
    pub coat_thickness: f32,
    pub coat_albedo: [f32; 3],
    pub params: MaterialParams,
    // Texture maps
    pub texture_mapping: i32, // 0=uv, 1=spherical, 2=cylindrical
    pub texture_inv_transform: [f32; 12],
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
    pub normalmap_data: optix_sys::CUdeviceptr,
    pub normalmap_width: i32,
    pub normalmap_height: i32,
    // Mix material (recursive tree)
    pub mix_mat1: optix_sys::CUdeviceptr, // first sub-material (0 if leaf)
    pub mix_mat2: optix_sys::CUdeviceptr, // second sub-material (0 if leaf)
    pub mix_amount_data: optix_sys::CUdeviceptr,
    pub mix_amount_width: i32,
    pub mix_amount_height: i32,
    pub mix_amount_value: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HitGroupData {
    pub mat: optix_sys::CUdeviceptr, // pointer to MaterialData on device
    // Geometry
    pub vertices: optix_sys::CUdeviceptr,
    pub normals: optix_sys::CUdeviceptr,
    pub indices: optix_sys::CUdeviceptr,
    pub texcoords: optix_sys::CUdeviceptr,
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
    pub triangle_light_vertices: optix_sys::CUdeviceptr, // float[n*3], world-space
    pub triangle_light_groups: optix_sys::CUdeviceptr,   // TriangleLightGroup[]
    pub num_triangle_light_groups: i32,
    pub triangle_light_group_cdf: optix_sys::CUdeviceptr, // float[num_groups+1], power-weighted
    // Environment map (IBL)
    pub envmap_data: optix_sys::CUdeviceptr, // RGB float, width*height*3 (0 = no envmap)
    pub envmap_width: i32,
    pub envmap_height: i32,
    pub envmap_marginal_cdf: optix_sys::CUdeviceptr, // float[height+1], marginal CDF over rows
    pub envmap_conditional_cdf: optix_sys::CUdeviceptr, // float[height*(width+1)], conditional CDFs
    pub envmap_integral: f32,                        // total luminance integral
    // Environment map rotation (inverse of light transform, 3x3 row-major)
    pub envmap_inv_rotation: [f32; 9],
    // Environment map portal
    pub has_portal: i32,
    pub portal: [f32; 12], // 4 vertices * 3 floats
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
