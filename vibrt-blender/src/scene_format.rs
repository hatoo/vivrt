//! JSON schema for Blender → vibrt-blender scene files.
//!
//! The scene is described by two files in the same directory:
//! - `scene.json` (this schema)
//! - `scene.bin` (opaque binary blobs; referenced by `BlobRef`)

use serde::Deserialize;

#[derive(Deserialize)]
pub struct SceneFile {
    pub version: u32,
    pub binary: String,
    pub render: RenderSettings,
    pub camera: CameraDesc,
    #[serde(default)]
    pub meshes: Vec<MeshDesc>,
    #[serde(default)]
    pub materials: Vec<PrincipledMaterial>,
    #[serde(default)]
    pub textures: Vec<TextureDesc>,
    #[serde(default)]
    pub objects: Vec<ObjectDesc>,
    #[serde(default)]
    pub lights: Vec<LightDesc>,
    pub world: Option<WorldDesc>,
}

#[derive(Deserialize)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub max_depth: u32,
}

#[derive(Deserialize)]
pub struct CameraDesc {
    /// 4x4 row-major matrix_world from Blender. Camera looks down local -Z.
    pub transform: [f32; 16],
    pub fov_y_rad: f32,
    #[serde(default)]
    pub lens_radius: f32,
    #[serde(default = "one_f32")]
    pub focal_distance: f32,
}

fn one_f32() -> f32 {
    1.0
}

#[derive(Deserialize, Copy, Clone)]
pub struct BlobRef {
    pub offset: u64,
    pub len: u64,
}

#[derive(Deserialize)]
pub struct MeshDesc {
    /// f32 x 3 per vertex
    pub vertices: BlobRef,
    /// f32 x 3 per vertex; optional
    #[serde(default)]
    pub normals: Option<BlobRef>,
    /// f32 x 2 per vertex; optional
    #[serde(default)]
    pub uvs: Option<BlobRef>,
    /// u32 x 3 per triangle
    pub indices: BlobRef,
}

#[derive(Deserialize)]
pub struct PrincipledMaterial {
    pub base_color: [f32; 3],
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "half_f32")]
    pub roughness: f32,
    #[serde(default = "default_ior")]
    pub ior: f32,
    #[serde(default)]
    pub transmission: f32,
    #[serde(default)]
    pub emission: [f32; 3],
    #[serde(default)]
    pub base_color_tex: Option<u32>,
    #[serde(default)]
    pub normal_tex: Option<u32>,
    #[serde(default)]
    pub roughness_tex: Option<u32>,
    #[serde(default)]
    pub metallic_tex: Option<u32>,
}

fn half_f32() -> f32 {
    0.5
}

fn default_ior() -> f32 {
    1.45
}

#[derive(Deserialize)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// 3 or 4
    pub channels: u32,
    /// "srgb" | "linear"
    pub colorspace: String,
    /// f32 little-endian pixels, width*height*channels floats
    pub pixels: BlobRef,
}

#[derive(Deserialize)]
pub struct ObjectDesc {
    pub mesh: u32,
    pub material: u32,
    /// 4x4 row-major matrix_world
    pub transform: [f32; 16],
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LightDesc {
    Point {
        position: [f32; 3],
        color: [f32; 3],
        power: f32,
        #[serde(default = "default_point_radius")]
        radius: f32,
    },
    Sun {
        direction: [f32; 3],
        color: [f32; 3],
        strength: f32,
        #[serde(default)]
        angle_rad: f32,
    },
    Spot {
        /// 4x4 row-major. Light shines along local -Z.
        transform: [f32; 16],
        color: [f32; 3],
        power: f32,
        cone_rad: f32,
        #[serde(default)]
        blend: f32,
    },
    AreaRect {
        /// 4x4 row-major. Area plane is local XY, emission along local +Z.
        transform: [f32; 16],
        size: [f32; 2],
        color: [f32; 3],
        power: f32,
    },
}

fn default_point_radius() -> f32 {
    0.05
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorldDesc {
    Constant {
        color: [f32; 3],
        #[serde(default = "one_f32")]
        strength: f32,
    },
    Envmap {
        texture: u32,
        #[serde(default)]
        rotation_z_rad: f32,
        #[serde(default = "one_f32")]
        strength: f32,
    },
}
