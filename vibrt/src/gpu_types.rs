//! GPU struct types shared between host and device.
//! Layout must match `devicecode.h` exactly.

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PrincipledGpu {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ior: f32,
    pub transmission: f32,
    pub emission: [f32; 3],
    pub base_color_tex: optix_sys::CUdeviceptr,
    pub base_color_tex_w: i32,
    pub base_color_tex_h: i32,
    pub base_color_tex_channels: i32,
    pub normal_tex: optix_sys::CUdeviceptr,
    pub normal_tex_w: i32,
    pub normal_tex_h: i32,
    pub normal_tex_channels: i32,
    pub roughness_tex: optix_sys::CUdeviceptr,
    pub roughness_tex_w: i32,
    pub roughness_tex_h: i32,
    pub roughness_tex_channels: i32,
    pub metallic_tex: optix_sys::CUdeviceptr,
    pub metallic_tex_w: i32,
    pub metallic_tex_h: i32,
    pub metallic_tex_channels: i32,
    /// 2x3 row-major affine UV transform: uv' = [M[0..3]·(u,v,1), M[3..6]·(u,v,1)].
    pub uv_transform: [f32; 6],
    pub normal_strength: f32,
    pub bump_strength: f32,
    pub bump_tex: optix_sys::CUdeviceptr,
    pub bump_tex_w: i32,
    pub bump_tex_h: i32,
    pub bump_tex_channels: i32,
    pub alpha_threshold: f32,
    pub anisotropy: f32,
    pub tangent_rotation: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    pub coat_ior: f32,
    pub sheen_weight: f32,
    pub sheen_roughness: f32,
    pub sheen_tint: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HitGroupData {
    pub mat: optix_sys::CUdeviceptr,
    pub vertices: optix_sys::CUdeviceptr,
    pub normals: optix_sys::CUdeviceptr,
    pub indices: optix_sys::CUdeviceptr,
    pub uvs: optix_sys::CUdeviceptr,
    pub num_vertices: i32,
    pub area_light_group: i32,
    /// u32 per triangle — index into `materials[]`. 0 if null.
    pub material_indices: optix_sys::CUdeviceptr,
    /// PrincipledGpu* per slot. Used when `material_indices` != 0.
    pub materials: optix_sys::CUdeviceptr,
    pub num_materials: i32,
    pub _pad_hg: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub emission: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SunLight {
    pub direction: [f32; 3],
    pub cos_angle: f32,
    pub emission: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SpotLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub direction: [f32; 3],
    pub cos_outer: f32,
    pub emission: [f32; 3],
    pub cos_inner: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AreaRectLight {
    /// Corner in world space (center - 0.5*size_u*u - 0.5*size_v*v)
    pub corner: [f32; 3],
    pub size_u: f32,
    pub u_axis: [f32; 3],
    pub size_v: f32,
    pub v_axis: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub emission: [f32; 3],
    pub power: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LaunchParams {
    /// float4 image buffer (linear RGB + weight)
    pub image: optix_sys::CUdeviceptr,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub max_depth: u32,

    pub cam_eye: [f32; 3],
    pub cam_u: [f32; 3],
    pub cam_v: [f32; 3],
    pub cam_w: [f32; 3],
    pub cam_lens_radius: f32,
    pub cam_focal_distance: f32,

    pub traversable: optix_sys::OptixTraversableHandle,

    pub num_point_lights: i32,
    pub point_lights: optix_sys::CUdeviceptr,
    pub num_sun_lights: i32,
    pub sun_lights: optix_sys::CUdeviceptr,
    pub num_spot_lights: i32,
    pub spot_lights: optix_sys::CUdeviceptr,
    pub num_rect_lights: i32,
    pub rect_lights: optix_sys::CUdeviceptr,
    pub rect_light_cdf: optix_sys::CUdeviceptr,

    /// World background: 0 = constant, 1 = envmap
    pub world_type: i32,
    pub world_color: [f32; 3],
    pub world_strength: f32,
    pub envmap_data: optix_sys::CUdeviceptr,
    pub envmap_width: i32,
    pub envmap_height: i32,
    pub envmap_marginal_cdf: optix_sys::CUdeviceptr,
    pub envmap_conditional_cdf: optix_sys::CUdeviceptr,
    pub envmap_integral: f32,
    pub envmap_rotation_z_rad: f32,

    pub ggx_e_lut: optix_sys::CUdeviceptr,
    pub ggx_e_avg_lut: optix_sys::CUdeviceptr,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MissData {
    pub _unused: i32,
}
