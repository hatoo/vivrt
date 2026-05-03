//! JSON schema for the vibrt scene format.
//!
//! Produced by `blender/vibrt_blender/exporter.py::export_scene_to_memory`
//! and consumed by `scene_loader::load_scene_from_bytes`. The Python side
//! drives `vibrt_native.render(scene_json, scene_bin, ...)` directly — the
//! schema is not designed for on-disk archival; texture pixels travel
//! across PyO3 as a separate `Vec<PyBuffer<f32>>` rather than living in
//! the bin (see `TextureDesc::array_index`).

use serde::Deserialize;

#[derive(Deserialize)]
pub struct SceneFile {
    pub version: u32,
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
    /// Per-lobe bounce caps mirroring Cycles' `diffuse_bounces` /
    /// `glossy_bounces` / `transmission_bounces`. Default to `max_depth`
    /// (= no per-lobe restriction) so older scene JSONs still render.
    #[serde(default = "default_lobe_bounces")]
    pub max_diffuse_bounces: u32,
    #[serde(default = "default_lobe_bounces")]
    pub max_glossy_bounces: u32,
    #[serde(default = "default_lobe_bounces")]
    pub max_transmission_bounces: u32,
    /// Clamp for indirect (bounce>=1) contribution luminance. 0 disables.
    #[serde(default)]
    pub clamp_indirect: f32,
    /// Clamp for direct (NEE on the primary surface) contribution
    /// luminance. Mirrors Cycles' `sample_clamp_direct`. 0 disables.
    #[serde(default)]
    pub clamp_direct: f32,
    /// Cycles' Light Paths > Filter Glossy (`cycles.blur_glossy`).
    /// Inflates BSDF roughness on subsequent bounces proportional to
    /// `sqrt(filter_glossy / min_ray_pdf)` so a near-mirror lobe seen
    /// after a previous glossy bounce can't refocus radiance into a
    /// solid angle smaller than the path has already established. 0
    /// disables. Cycles defaults to 1.0; pabellon ships at 5.0.
    #[serde(default)]
    pub filter_glossy: f32,
}

fn default_lobe_bounces() -> u32 {
    32
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
    /// Cycles' "Clip Start". Primary rays skip geometry closer than this
    /// — the lone_monk camera sits just behind an alcove wall that the
    /// 1.12m clip_start hides; without it the wall fills the frame and
    /// no scene structure shows through.
    #[serde(default)]
    pub clip_start: f32,
    /// Cycles' "Clip End". Caps the primary ray's travel; defaults large
    /// when missing so heritage scenes still render to infinity.
    #[serde(default = "default_clip_end")]
    pub clip_end: f32,
    /// Lens shift in NDC fractions of the half-extent on each axis (i.e.
    /// `cx / (2 * half_w)` and `cy / (2 * half_h)` of Blender's view
    /// frame at z = -1). The device adds `2 * shift` to the NDC pixel
    /// coordinate before projecting. The exporter computes this from
    /// `cam.view_frame(scene=scene)` so it already accounts for
    /// sensor_fit and the "shift is a fraction of the larger sensor
    /// dimension" Blender convention — values here are NOT the raw
    /// `Camera.shift_x/y` from Blender.
    #[serde(default)]
    pub shift_x: f32,
    #[serde(default)]
    pub shift_y: f32,
}

fn default_clip_end() -> f32 {
    1.0e20
}

fn one_f32() -> f32 {
    1.0
}

/// Index into the caller-supplied `mesh_blobs: &[&[u8]]` slice. Each blob
/// is a contiguous byte buffer that the loader reinterprets as f32 / u32
/// according to the field's declared type. Replaces the old
/// `BlobRef { offset, len }` — the bin used to be one giant bytearray and
/// every field stored an offset into it; now each blob is its own numpy
/// array on the Python side, parked into a list, and addressed by index.
pub type BlobIdx = u32;

#[derive(Deserialize)]
pub struct MeshDesc {
    /// f32 x 3 per vertex
    pub vertices: BlobIdx,
    /// f32 x 3 per vertex; optional
    #[serde(default)]
    pub normals: Option<BlobIdx>,
    /// f32 x 2 per vertex; optional
    #[serde(default)]
    pub uvs: Option<BlobIdx>,
    /// u32 x 3 per triangle
    pub indices: BlobIdx,
    /// u32 per triangle — index into ObjectDesc::materials.
    /// When absent, the whole mesh uses ObjectDesc::material.
    #[serde(default)]
    pub material_indices: Option<BlobIdx>,
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
    pub vertex_colors: Option<BlobIdx>,
    /// f32 x 3 per vertex; optional. Object-space tangent direction authored
    /// by the exporter — used by the hair Kajiya-Kay lobe to read the strand
    /// axis at the hit point. When present, the device interpolates it
    /// barycentrically and overrides the default build_frame() tangent. The
    /// exporter only emits this for hair ribbon meshes today.
    #[serde(default)]
    pub tangents: Option<BlobIdx>,
}

#[derive(Deserialize)]
pub struct PrincipledMaterial {
    pub base_color: [f32; 3],
    #[serde(default)]
    pub metallic: f32,
    /// Cycles' `ShaderNodeBsdfAnisotropic` / `ShaderNodeBsdfGlossy` is
    /// a pure GGX reflector with NO Fresnel — `bsdf_microfacet_ggx_setup`
    /// in Cycles sets `MicrofacetFresnel::NONE`, and the per-direction
    /// reflectance is just `Color × D × G / (4·NoV·NoL)`. vibrt's metal
    /// lobe applies Schlick Fresnel with F0 = base_color, which lifts
    /// reflectance toward white at grazing — over-bright on dark
    /// materials (BMWBlack on the bmw27 test scene rendered visibly
    /// brighter on the rounded car body). When `pure_glossy` is true,
    /// the device code skips the Schlick lift on the metallic lobe and
    /// uses `base_color` as a uniform tint, matching Cycles' Glossy
    /// semantics. Implies `metallic = 1.0` (the exporter sets both
    /// when mapping `ShaderNodeBsdfAnisotropic` / Glossy).
    #[serde(default)]
    pub pure_glossy: bool,
    /// Cycles' `ShaderNodeBsdfDiffuse` is a Lambertian / Oren-Nayar
    /// reflector with NO specular component. vibrt's Principled at
    /// `metallic=0, transmission=0` still emits a small dielectric
    /// Schlick term `(1-VoH)^5 × GGX` (because `F0_d` is positive even
    /// at ior=1.45 and grows toward 1 at grazing) plus the Kulla-Conty
    /// dielectric MS compensation — both lift the rounded surfaces of
    /// BMW27's diffuse car body (BMWWhite, BMWBlue, Interior, TireRubber
    /// account for ~30k polys) toward white at glancing angles. When
    /// `pure_diffuse` is true, the device code drops the specular and
    /// coat lobes entirely and returns just `base_color × Lambert × NoL`.
    /// Set by the exporter's `_from_diffuse` alongside `metallic=0`.
    #[serde(default)]
    pub pure_diffuse: bool,
    /// Cycles' Principled `Specular IOR Level` input (default 0.5).
    /// Scales the dielectric Schlick F0:
    ///   F0_d = ((ior-1)/(ior+1))² × 2 × specular_ior_level
    /// At the default 0.5 the multiplier is 1.0 (no change). flat_archiviz
    /// has 48 of 97 Principled materials with non-default values (most
    /// in 0.30-0.40 range), which lower F0 and reduce the dielectric
    /// spec lift at grazing — without honouring this, the surfaces
    /// over-bright. See `c:/tmp/cycles-src/src/kernel/svm/closure.h:418`.
    #[serde(default = "half_f32")]
    pub specular_ior_level: f32,
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
    /// Constant Alpha < 1 mix weight from Principled's `Alpha` socket
    /// (no texture). Cycles' Principled blends `alpha × principled_lobes
    /// + (1-alpha) × straight-through transparent`. Defaults to 1.0
    /// (fully opaque); the kernel adds a transparent-passthrough lobe
    /// with weight `1 - alpha_blend` when this is < 1. Distinct from
    /// `alpha_threshold` (texture-driven binary cutout for leaves/grass).
    #[serde(default = "one_f32")]
    pub alpha_blend: f32,
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
    /// Cycles `ShaderNodeBsdfTranslucent` weight in [0,1]. A real Translucent
    /// lobe — Lambertian on the *back* hemisphere — that lights paper /
    /// lampshade / leaf surfaces from the side opposite the viewer. Replaces
    /// the previous wrap-Lambert SSS hack which under-illuminated thin
    /// sheets (archiviz lamp shade was visibly darker than Cycles). When
    /// > 0 the same fraction of `w_diffuse` is redirected to the back
    /// hemisphere, so `translucent_weight=1` yields pure Translucent and
    /// intermediate values mix with forward Lambert (Mix Shader case).
    #[serde(default)]
    pub translucent_weight: f32,
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
    /// Per-instance uniform random in [0..1) — derived from the OptiX
    /// instance id at hit time. Mirrors `ShaderNodeObjectInfo.Random`. The
    /// scalar is broadcast to all three RGB channels so downstream Mix /
    /// ColorRamp consumers can use it both as a colour and as a Fac.
    ObjectRandom {},
    /// Look up a 1D RGB ramp at a Fac driven by another graph slot. Used
    /// when ColorRamp's Fac chain is non-foldable (typically
    /// `ObjectInfo.Random` feeding a per-instance palette). The exporter
    /// bakes the ramp to `lut.len() / 3` evenly-spaced linear-RGB stops.
    ColorRamp {
        input: u32,
        /// `n_stops × 3` floats: R[0..n], G[0..n], B[0..n] flattened by
        /// stop, so payload[i*3..i*3+3] is the i-th stop's linear RGB.
        lut: Vec<f32>,
    },
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
    /// Index into the caller-supplied `texture_arrays` slice. Pixel data
    /// always travels alongside the scene as a separate list of f32
    /// buffers — the bin no longer carries texture pixels.
    pub array_index: u32,
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

/// IES (Illuminating Engineering Society) photometric profile.
///
/// Sampled in the host-side `ies::lookup_normalised` and the GPU's
/// `ies_lookup` from `(theta, phi)` in the light's local frame, where
/// `theta` is the angle from the light's local +Z (Blender's IES
/// convention is angle from -Z, but the exporter inverts so that
/// "directly down" matches +Z when sampling). The table stores raw
/// candela values; `peak_candela` is precomputed so the runtime can
/// normalise to a [0, 1] directional shape multiplier and let the
/// light's own `power`/`color` carry absolute brightness — same
/// convention Cycles uses (the IES Texture Node's `Factor` output is
/// `candela / max(candela)`).
#[derive(Deserialize, Clone, Debug)]
pub struct IesProfile {
    /// Vertical angles in degrees. Strictly increasing, in [0, 180].
    pub thetas_deg: Vec<f32>,
    /// Horizontal angles in degrees. Strictly increasing. Length 1
    /// means the profile is radially symmetric.
    pub phis_deg: Vec<f32>,
    /// Candela values, row-major over `phis_deg × thetas_deg` so that
    /// `candelas[h * thetas_deg.len() + v]` is the intensity at
    /// `(phis_deg[h], thetas_deg[v])`. Length must equal
    /// `thetas_deg.len() * phis_deg.len()`.
    pub candelas: Vec<f32>,
    /// Peak (maximum) candela across the table. Precomputed so the GPU
    /// doesn't have to scan the table at every sample. Zero/negative is
    /// treated as "no IES" (returns multiplier 1.0).
    pub peak_candela: f32,
    /// Solid-angle integral of `candela / peak_candela` over the
    /// sphere (steradians). Legacy "flux preservation" support — kept
    /// so older scene.json blobs without `peak_absolute_candela` still
    /// load. The renderer prefers `peak_absolute_candela` when present.
    #[serde(default)]
    pub integral_norm: f32,
    /// Absolute peak intensity in W/sr — `peak_candela × multiplier ×
    /// ballast × ballast_lamp_photometric × 4π/177.83`. Mirrors Cycles'
    /// `util/ies.cpp:151,164-180` candela→W/sr conversion (177.83 lm/W
    /// is D65's luminous efficacy). The renderer uses
    /// `coeff = power × peak_absolute_candela / π` so `Le_at_angle =
    /// color × power × candela_absolute(angle) / π` exactly matches the
    /// `kernel/svm/ies.h:31` × point-light `eval_fac=1/π` chain Cycles
    /// uses. Zero (or missing in old json) falls back to the legacy
    /// `power / integral_norm` flux-preservation path.
    #[serde(default)]
    pub peak_absolute_candela: f32,
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
        /// Object→world rotation 3×3 (row-major). Identity when no IES
        /// is attached. With IES, `wi_local = transpose(rotation) * wi_world`
        /// gives the direction in light-local frame for table lookup.
        /// Blender Point lights have a `matrix_world` rotation that is
        /// invisible without IES (truly isotropic) but matters here.
        #[serde(default = "identity_rotation_3x3")]
        light_rotation: [f32; 9],
        #[serde(default)]
        ies: Option<IesProfile>,
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
        /// IES profile attached to this spot. Sampled in the same
        /// local frame the cone uses (axis = local -Z = "down");
        /// see `IesProfile`.
        #[serde(default)]
        ies: Option<IesProfile>,
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
        /// IES profile attached to this area light. Sampled in the
        /// rect's local frame (emission direction = local +Z).
        #[serde(default)]
        ies: Option<IesProfile>,
    },
}

fn default_point_radius() -> f32 {
    0.05
}

/// One layer of a (possibly mixed) world envmap. Mirrors a single
/// `ShaderNodeBackground` driving its `Color` from a `ShaderNodeTexEnvironment`
/// or `ShaderNodeTexSky`. Used as a building block for the `Mixed` world
/// variant; single-layer worlds still go through the simpler `Envmap`
/// variant for backwards compatibility.
#[derive(Deserialize, Clone)]
pub struct EnvmapLayer {
    pub texture: u32,
    /// 3×3 row-major rotation applied to the world-space sample direction
    /// before the projection-specific lookup. `[1,0,0, 0,1,0, 0,0,1]` is
    /// identity. Mapping nodes with arbitrary Euler XYZ rotation are
    /// pre-composed into this matrix on the host.
    #[serde(default = "identity_rotation_3x3")]
    pub rotation: [f32; 9],
    #[serde(default = "one_f32")]
    pub strength: f32,
    /// Projection mode used when sampling the layer's texture from a
    /// world-space ray direction. `"equirect"` — the usual HDRI mapping
    /// `(u, v) = (φ/2π, θ/π)`. `"flat"` — Cycles' ShaderNodeTexImage
    /// projection=FLAT, which uses the rotated direction's `(x, y)`
    /// components directly as UV (regardless of `z`). Pabellon's sunset
    /// world drives a 4928×3264 sRGB JPG photo this way; sampling it as
    /// equirect is wrong everywhere on the hemisphere.
    #[serde(default = "default_projection")]
    pub projection: String,
    /// Texture-coordinate extension applied to UV outside `[0, 1]`.
    /// `"repeat"` — wrap (default). `"extend"` — clamp to edge.
    /// `"clip"` — return zero (treat sample as black, matching Cycles).
    /// For `projection="equirect"` we always wrap u and clamp v
    /// (longitude is periodic, latitude is bounded), so this field is
    /// only consulted by FLAT-projection layers.
    #[serde(default = "default_extension")]
    pub extension: String,
}

fn default_projection() -> String {
    "equirect".into()
}

fn default_extension() -> String {
    "repeat".into()
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
    /// Two-layer envmap mixed by `fac`. Used when the world is a
    /// `ShaderNodeMixShader(Background_a, Background_b)` — pabellon's
    /// sunset world is the canonical case (Sky Texture + HDRI). Each
    /// layer keeps its own rotation + strength so HDRIs (sampled at
    /// native resolution) and Sky Texture bakes can coexist without
    /// losing peak luminance to the single combined bake the
    /// `_world_needs_full_bake` path currently produces.
    ///
    /// `fac=0` selects `a` only; `fac=1` selects `b` only; intermediate
    /// values do a linear blend per Cycles' MixShader convention.
    ///
    /// Importance sampling: the scene loader currently builds a CDF
    /// from a host-side rasterised mix; future work could fold that
    /// into the GPU side so updates to either layer don't trigger a
    /// rebuild.
    Mixed {
        a: EnvmapLayer,
        b: EnvmapLayer,
        #[serde(default = "half_f32")]
        fac: f32,
        /// When true, `fac` is ignored and the world is split by ray type:
        /// `a` is what *non-camera* (lighting / NEE / indirect) rays see,
        /// `b` is what *camera* rays (and rays after a chain of specular
        /// bounces) see. Set by the exporter when the MixShader's factor
        /// is driven by `ShaderNodeLightPath.is_camera_ray` — the canonical
        /// archiviz idiom for separating a high-strength ambient lighting
        /// envmap from a low-strength backplate visible to the camera.
        ///
        /// The CDF is built from `a` only (NEE samples lighting); `b` is
        /// looked up directly on camera-ray misses. With this flag set the
        /// scene loader skips `build_mixed_envmap_grid` and uses `a`'s
        /// pixels verbatim as `envmap_data`.
        #[serde(default)]
        split_by_camera_ray: bool,
    },
}

fn identity_rotation_3x3() -> [f32; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}
