//! JSON schema for Blender → vibrt-blender scene files.
//!
//! The scene is described by two files in the same directory:
//! - `scene.json` (this schema)
//! - `scene.bin` (opaque binary blobs; referenced by `BlobRef`)

use serde::Deserialize;

#[derive(Deserialize)]
pub struct SceneFile {
    pub version: u32,
    /// Filename of the sibling .bin blob. Resolved relative to the JSON path
    /// by `load_scene_from_path`; ignored by `load_scene_from_bytes` (the
    /// in-process renderer hands the buffer directly).
    #[allow(dead_code)]
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
    /// Optional homogeneous volume that fills the entire scene (atmospheric
    /// haze / fog). Cycles' "World Output → Volume" socket lands here.
    /// `None` => clear atmosphere (current behaviour).
    #[serde(default)]
    pub world_volume: Option<VolumeParams>,
}

/// Homogeneous volume parameters shared by per-material volumes (scattering
/// boundaries on closed meshes, e.g. junk_shop's `Smoke`) and the world
/// volume (everywhere). All quantities are in metres⁻¹ before being scaled by
/// `density`.
///
/// Cycles' Principled Volume conventions:
/// - `color * density` is the per-channel scattering coefficient σ_s.
/// - `absorption_color * density` is σ_a. (When σ_a = 0 the volume is purely
///   scattering — junk_shop's smoke is configured this way.)
/// - σ_t = σ_s + σ_a. The renderer samples a single scalar majorant (the
///   per-channel max of σ_t) for distance sampling and tracks per-channel
///   beam transmittance separately.
/// - Emission = `emission_color * emission_strength + blackbody_rgb *
///   blackbody_intensity`, all scaled by `density`. The exporter folds the
///   blackbody term in before serialising; the renderer just sees one
///   pre-multiplied RGB.
#[derive(Deserialize, Default, Copy, Clone)]
pub struct VolumeParams {
    /// Scattering tint. Multiplied by `density` to give σ_s.
    #[serde(default = "default_volume_color")]
    pub color: [f32; 3],
    #[serde(default = "one_f32")]
    pub density: f32,
    /// Henyey-Greenstein g, in [-1, 1]. 0 = isotropic.
    #[serde(default)]
    pub anisotropy: f32,
    /// Multiplied by `density` to give σ_a.
    #[serde(default)]
    pub absorption_color: [f32; 3],
    /// Pre-multiplied with `emission_strength` and folded with any blackbody
    /// term by the exporter; the renderer scales by density on the GPU side.
    #[serde(default)]
    pub emission_color: [f32; 3],
    #[serde(default)]
    pub emission_strength: f32,
}

fn default_volume_color() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(Deserialize, Copy, Clone)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub max_depth: u32,
    /// Clamp for indirect (bounce>=1) contribution luminance. 0 disables.
    #[serde(default)]
    pub clamp_indirect: f32,
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
    /// u32 per triangle — index into ObjectDesc::materials.
    /// When absent, the whole mesh uses ObjectDesc::material.
    #[serde(default)]
    pub material_indices: Option<BlobRef>,
    /// Grayscale heightmap (texture index). Sampled at vertex UVs; offsets
    /// each vertex along its normal by `sampled * displacement_strength`.
    /// Applied once at scene-load before BLAS construction.
    #[serde(default)]
    pub displacement_tex: Option<u32>,
    #[serde(default)]
    pub displacement_strength: f32,
    /// f32 x 3 per vertex; optional. Written when a material references a
    /// vertex color Attribute for base_color. Interpolated barycentrically at
    /// the hit and multiplied into base_color when the material has
    /// `use_vertex_color` set.
    #[serde(default)]
    pub vertex_colors: Option<BlobRef>,
    /// f32 x 3 per vertex; optional. Object-space tangent direction authored
    /// by the exporter — used by the hair Kajiya-Kay lobe to read the strand
    /// axis at the hit point. When present, the device interpolates it
    /// barycentrically and overrides the default build_frame() tangent. The
    /// exporter only emits this for hair ribbon meshes today.
    #[serde(default)]
    pub tangents: Option<BlobRef>,
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
    #[serde(default)]
    pub transmission_tex: Option<u32>,
    /// Multiplied into the constant `emission` so a single image (e.g. an
    /// emissive billboard or display panel) drives per-pixel emission. Used
    /// when the Cycles graph wires a TexImage directly into Emission.Color.
    #[serde(default)]
    pub emission_tex: Option<u32>,
    /// 2x3 row-major affine UV transform [a,b,tu, c,d,tv]: uv' = M · (uv, 1).
    /// Default identity.
    #[serde(default = "identity_uv_transform")]
    pub uv_transform: [f32; 6],
    #[serde(default = "one_f32")]
    pub normal_strength: f32,
    #[serde(default)]
    pub bump_tex: Option<u32>,
    #[serde(default = "one_f32")]
    pub bump_strength: f32,
    /// If > 0, any-hit discards the hit when base_color_tex.a < threshold.
    #[serde(default)]
    pub alpha_threshold: f32,
    /// [0,1]; stretches GGX along tangent vs. bitangent.
    #[serde(default)]
    pub anisotropy: f32,
    /// Radians; rotates the shading tangent before evaluating anisotropy.
    #[serde(default)]
    pub tangent_rotation: f32,
    #[serde(default)]
    pub coat_weight: f32,
    #[serde(default = "default_coat_roughness")]
    pub coat_roughness: f32,
    #[serde(default = "default_coat_ior")]
    pub coat_ior: f32,
    #[serde(default)]
    pub sheen_weight: f32,
    #[serde(default = "half_f32")]
    pub sheen_roughness: f32,
    #[serde(default = "default_sheen_tint")]
    pub sheen_tint: [f32; 3],
    #[serde(default)]
    pub sss_weight: f32,
    #[serde(default = "default_sss_radius")]
    pub sss_radius: [f32; 3],
    /// When > 0, the surface is shaded with a Kajiya-Kay hair lobe instead
    /// of the standard diffuse+specular path. Triggered by Cycles'
    /// `ShaderNodeBsdfHair` and MixShader'd fractions of it.
    #[serde(default)]
    pub hair_weight: f32,
    /// Radians. Tilt of the specular cone away from perpendicular to the
    /// tangent — Cycles' "Offset" slider on Hair BSDF (angular shift of the
    /// highlight along the strand).
    #[serde(default)]
    pub hair_offset: f32,
    /// [0,1]. Along-strand spread of the specular highlight.
    #[serde(default = "default_hair_roughness")]
    pub hair_roughness_u: f32,
    /// [0,1]. Across-strand spread. Currently folded into the same cone
    /// width; kept separate so the exporter can faithfully record Cycles'
    /// RoughnessV and we can split the lobe later without a schema break.
    #[serde(default = "default_hair_roughness")]
    pub hair_roughness_v: f32,
    /// When true, multiply base_color by the mesh's interpolated vertex
    /// color (MeshDesc::vertex_colors). Set by the Blender exporter when a
    /// material drives its base colour via a ShaderNodeAttribute.
    #[serde(default)]
    pub use_vertex_color: bool,
    /// Optional per-pixel colour graph that replaces `base_color_tex` when
    /// present. Lets complex Cycles shader setups (classroom's paintedCeiling
    /// is the canonical case — Mix of two textures under different UV
    /// mappings, then multiplied by an AO map through an Invert / Math
    /// chain) survive the exporter without getting collapsed into a single
    /// baked texture.
    #[serde(default)]
    pub color_graph: Option<ColorGraph>,
    /// Cycles' material output → Volume socket. When present the bounded
    /// mesh acts as a volume container: rays entering a front face start
    /// integrating volume scattering until they exit a back face. Combine
    /// with `volume_only=true` to make the boundary surface invisible
    /// (junk_shop's `Smoke` material — Surface socket is unlinked).
    #[serde(default)]
    pub volume: Option<VolumeParams>,
    /// True when the material's Surface socket is unlinked. The boundary
    /// mesh is then a pure volume container — primary rays pass through it
    /// (no surface shading), but the volume inside still scatters / absorbs.
    /// Set by the exporter alongside `volume`.
    #[serde(default)]
    pub volume_only: bool,
}

/// Sequential list of colour-producing nodes. Each node reads either from
/// earlier nodes (by index) or from constants/textures; the last node's
/// value is the material's base colour. Input indices must be strictly
/// less than the owning node's index so the graph stays a DAG in
/// topological order.
#[derive(Deserialize, Clone)]
pub struct ColorGraph {
    pub nodes: Vec<ColorNode>,
    /// Index of the output node. Default: last node.
    #[serde(default)]
    pub output: Option<u32>,
}

/// Factor input for a Mix node: either a constant scalar or a reference to
/// another node's value (converted to luminance at evaluation time).
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum ColorFactor {
    Const(f32),
    Node { node: u32 },
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ColorNode {
    /// Literal RGB.
    Const { rgb: [f32; 3] },
    /// Sample a texture (by index into `SceneFile::textures`) with its own
    /// UV transform applied on top of the mesh UV.
    ImageTex {
        tex: u32,
        #[serde(default = "identity_uv_transform")]
        uv: [f32; 6],
    },
    /// Mix two inputs with the given blend and factor. `blend` follows
    /// Blender's naming: "mix" / "multiply" / "add" / "subtract" / "screen"
    /// / "divide" / "difference" / "darken" / "lighten" / "overlay" /
    /// "soft_light" / "linear_light".
    Mix {
        a: u32,
        b: u32,
        #[serde(default = "default_mix_fac")]
        fac: ColorFactor,
        #[serde(default = "default_blend")]
        blend: String,
        #[serde(default)]
        clamp: bool,
    },
    /// Per-channel RGB invert with mix factor (`out = lerp(rgb, 1-rgb, fac)`).
    Invert {
        input: u32,
        #[serde(default = "one_f32")]
        fac: f32,
    },
    /// Scalar math on the input treated per-channel. Ops mirror
    /// ShaderNodeMath: "add" / "subtract" / "multiply" / "divide" /
    /// "power" / "multiply_add" / "minimum" / "maximum".
    ///
    /// `swap` flips operand order for non-commutative ops (subtract /
    /// divide / power): when true the op evaluates as `b OP input` instead
    /// of `input OP b`. Emitted when Cycles drives the second Math input
    /// (rather than the first) with the texture chain. Ignored for
    /// commutative ops.
    Math {
        input: u32,
        #[serde(default = "default_math_op")]
        op: String,
        #[serde(default)]
        b: f32,
        #[serde(default)]
        c: f32,
        #[serde(default)]
        clamp: bool,
        #[serde(default)]
        swap: bool,
    },
    /// Blender HueSaturation: shift hue, scale saturation/value, then blend
    /// with the original by `fac`. All scalars are constants (Blender
    /// interprets linked ones here too but very rare).
    HueSat {
        input: u32,
        #[serde(default = "half_f32")]
        hue: f32,
        #[serde(default = "one_f32")]
        saturation: f32,
        #[serde(default = "one_f32")]
        value: f32,
        #[serde(default = "one_f32")]
        fac: f32,
    },
    /// Per-channel RGB curve, pre-baked by the exporter into a `768`-entry
    /// LUT (256 samples × 3 channels). Each channel's entry already folds
    /// the per-channel curve through the combined curve, matching how
    /// Blender stacks them in ShaderNodeRGBCurve.
    RgbCurve {
        input: u32,
        /// 768 floats: R[0..256], G[0..256], B[0..256]. Each subarray is
        /// the curve sampled at `i/255` and clamped to [0, 1].
        lut: Vec<f32>,
    },
    /// Blender BrightContrast: `out = a * rgb + b` with Cycles' parameter
    /// convention `a = 1 + contrast`, `b = bright - contrast/2`.
    BrightContrast {
        input: u32,
        #[serde(default)]
        bright: f32,
        #[serde(default)]
        contrast: f32,
    },
    /// Blender ShaderNodeAttribute reading a per-vertex colour. The mesh must
    /// ship the matching colour attribute (exporter plumbs this via
    /// `_vertex_color_attr`); when absent the device falls back to white.
    VertexColor {},
}

fn default_mix_fac() -> ColorFactor {
    ColorFactor::Const(0.5)
}

fn default_blend() -> String {
    "mix".to_string()
}

fn default_math_op() -> String {
    "multiply".to_string()
}

fn default_sss_radius() -> [f32; 3] {
    [1.0, 0.2, 0.1]
}

fn default_coat_roughness() -> f32 {
    0.03
}
fn default_hair_roughness() -> f32 {
    // Matches Cycles Hair BSDF's default 0.3 on Roughness U/V.
    0.3
}
fn default_coat_ior() -> f32 {
    1.5
}
fn default_camera_visible() -> u32 {
    1
}
fn default_sheen_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn identity_uv_transform() -> [f32; 6] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
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
    /// f32 little-endian pixels, width*height*channels floats. Either this
    /// or `array_index` must be present. The disk path always emits
    /// `pixels`; the in-process FFI exporter emits `array_index` so the
    /// pixel data can be passed across PyO3 as a separate `PyBuffer<f32>`
    /// list, skipping the bin's concatenation memcpy entirely.
    #[serde(default)]
    pub pixels: Option<BlobRef>,
    /// Index into the caller-supplied `texture_arrays` slice. Mutually
    /// exclusive with `pixels`.
    #[serde(default)]
    pub array_index: Option<u32>,
}

#[derive(Deserialize)]
pub struct ObjectDesc {
    pub mesh: u32,
    pub material: u32,
    /// Per-slot material IDs; used together with `MeshDesc::material_indices`.
    #[serde(default)]
    pub materials: Vec<u32>,
    /// 4x4 row-major matrix_world
    pub transform: [f32; 16],
    /// Cycles' object "Ray Visibility → Shadow". When false, shadow rays skip
    /// this instance — matches Cycles' use of paper-lantern shades that are
    /// camera-visible but transparent to NEE.
    #[serde(default = "default_true")]
    pub cast_shadow: bool,
}

fn default_true() -> bool { true }

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
        /// 0 = hidden from primary/specular camera rays (NEE still samples
        /// it), 1 = visible. Defaults to visible to preserve the behaviour
        /// of existing hand-authored scenes.
        #[serde(default = "default_camera_visible")]
        camera_visible: u32,
        /// 1 = emissive-mesh style (radiates from both faces, matches
        /// Blender's default for emissive meshes). 0 = Blender-Area style
        /// (single-sided). Default stays 0 so existing scenes are
        /// unchanged.
        #[serde(default)]
        two_sided: u32,
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
