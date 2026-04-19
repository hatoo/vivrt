//! GPU struct types shared between host and device.
//! Layout must match `devicecode.h` exactly.

/// Single node in a material's colour graph. The graph is a sequential
/// list; each node reads from earlier nodes by index and writes one RGB
/// value. The device-side evaluator allocates a small stack of `float3`
/// slots (one per node) and runs through them in order.
///
/// Fields are type-punned into `payload[]` (u32 for indices / tags,
/// bit-cast float for constants). Host side fills the packing; device
/// side mirrors the layout via the same struct in `devicecode.h`.
///
/// Payload layout per tag (`payload[i]` as u32 unless noted):
///   Const       [0..3] = rgb (float bitcast)
///   ImageTex    [0] = tex_ptr_lo, [1] = tex_ptr_hi, [2] = w, [3] = h,
///               [4] = channels, [5..11] = uv_transform (6 floats)
///   Mix         [0] = in_a, [1] = in_b, [2] = blend_type, [3] = fac_src
///               (0=const, 1=node), [4] = fac_node, [5] = clamp (0/1),
///               [6] = fac_const (float)
///   Invert      [0] = in, [1] = fac (float)
///   Math        [0] = in, [1] = op, [2] = clamp (0/1),
///               [3] = b_const (float), [4] = c_const (float),
///               [5] = swap (0/1) — flips operand order for non-commutative
///                     ops (subtract/divide/power); ignored for others.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColorGraphNode {
    pub tag: u32,
    pub payload: [u32; 15],
}

// Must stay in sync with `devicecode.h` and `scene_loader.rs`.
pub const COLOR_NODE_CONST: u32 = 0;
pub const COLOR_NODE_IMAGE_TEX: u32 = 1;
pub const COLOR_NODE_MIX: u32 = 2;
pub const COLOR_NODE_INVERT: u32 = 3;
pub const COLOR_NODE_MATH: u32 = 4;
pub const COLOR_NODE_HUE_SAT: u32 = 5;
pub const COLOR_NODE_RGB_CURVE: u32 = 6;
pub const COLOR_NODE_BRIGHT_CONTRAST: u32 = 7;

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
    pub sss_weight: f32,
    pub sss_radius: [f32; 3],
    pub sss_anisotropy: f32,
    /// i32 used as bool — multiply base_color by interpolated vertex color.
    pub use_vertex_color: i32,
    /// Optional colour graph replacing `base_color_tex`. Non-null pointer +
    /// positive `color_graph_len` means "evaluate this graph to derive the
    /// base colour and ignore `base_color_tex`".
    pub color_graph_nodes: optix_sys::CUdeviceptr,
    pub color_graph_len: i32,
    /// Index into `color_graph_nodes` whose output is the final colour.
    /// Defaults to `color_graph_len - 1`.
    pub color_graph_output: i32,
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
    /// f32 x 3 per vertex. 0 when the mesh has no vertex color attribute.
    pub vertex_colors: optix_sys::CUdeviceptr,
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

    pub clamp_indirect: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RayGenData {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MissData {
    pub _unused: i32,
}
