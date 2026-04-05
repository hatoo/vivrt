//! PBRT scene parsing and representation.

use crate::gpu_types::*;
use crate::{ply, subdivision, transform};
use pbrt_parser::{self, Directive, ParamType, ParamValue};
use std::path::Path;

pub struct ImageTexture {
    pub data: Vec<f32>, // RGB float, width*height*3
    pub width: u32,
    pub height: u32,
}

pub struct SceneMaterial {
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub eta: f32,
    pub roughness: f32,
    pub tint: [f32; 3], // absorption tint for dielectrics (from MediumInterface)
    pub emission: [f32; 3],
    pub has_checkerboard: bool,
    pub checker_scale_u: f32,
    pub checker_scale_v: f32,
    pub checker_color1: [f32; 3],
    pub checker_color2: [f32; 3],
    pub roughness_v: f32,
    pub conductor_eta: [f32; 3],
    pub conductor_k: [f32; 3],
    pub coat_roughness: f32,
    pub coat_eta: f32,
    pub texture: Option<std::sync::Arc<ImageTexture>>,
    pub bump_map: Option<std::sync::Arc<ImageTexture>>,
    pub alpha_map: Option<std::sync::Arc<ImageTexture>>,
    pub roughness_map: Option<std::sync::Arc<ImageTexture>>,
}

impl Default for SceneMaterial {
    fn default() -> Self {
        Self {
            material_type: MAT_DIFFUSE,
            albedo: [0.5, 0.5, 0.5],
            eta: 1.5,
            roughness: 1.0,
            tint: [1.0, 1.0, 1.0],
            emission: [0.0, 0.0, 0.0],
            has_checkerboard: false,
            checker_scale_u: 1.0,
            checker_scale_v: 1.0,
            checker_color1: [1.0, 1.0, 1.0],
            checker_color2: [0.0, 0.0, 0.0],
            roughness_v: 0.0,
            conductor_eta: [0.0, 0.0, 0.0],
            conductor_k: [0.0, 0.0, 0.0],
            coat_roughness: 0.0,
            coat_eta: 1.5,
            texture: None,
            bump_map: None,
            alpha_map: None,
            roughness_map: None,
        }
    }
}

impl Clone for SceneMaterial {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            bump_map: self.bump_map.clone(),
            alpha_map: self.alpha_map.clone(),
            roughness_map: self.roughness_map.clone(),
            ..*self
        }
    }
}

#[derive(Clone)]
pub enum SceneShape {
    Sphere {
        radius: f32,
    },
    TriangleMesh {
        vertices: Vec<f32>,
        indices: Vec<i32>,
        texcoords: Vec<f32>,
        normals: Vec<f32>, // per-vertex normals (3 per vertex), empty = flat shading
    },
}

#[derive(Clone)]
pub struct SceneObject {
    pub shape: SceneShape,
    pub material: SceneMaterial,
    pub transform: [f32; 12],
}

pub struct ParsedScene {
    pub width: u32,
    pub height: u32,
    pub fov: f32,
    pub cam_eye: [f32; 3],
    pub cam_look: [f32; 3],
    pub cam_up: [f32; 3],
    pub spp: u32,
    pub max_depth: u32,
    pub ambient_light: [f32; 3],
    pub distant_lights: Vec<DistantLight>,
    pub sphere_lights: Vec<SphereLight>,
    pub triangle_lights: Vec<TriangleLight>,
    pub triangle_light_groups: Vec<TriangleLightGroup>,
    pub objects: Vec<SceneObject>,
    pub filename: String,
    pub cam_flip_x: bool,
    pub envmap: Option<ImageTexture>,
}

// ---- Parameter access tracking ----

/// Wraps a param list and tracks which params were accessed.
/// Warns about unaccessed params on drop.
struct ParamSet<'a> {
    params: &'a [pbrt_parser::Param],
    accessed: std::cell::RefCell<std::collections::HashSet<usize>>,
    context: String,
}

impl<'a> ParamSet<'a> {
    fn new(params: &'a [pbrt_parser::Param], context: impl Into<String>) -> Self {
        Self {
            params,
            accessed: std::cell::RefCell::new(std::collections::HashSet::new()),
            context: context.into(),
        }
    }

    fn mark(&self, name: &str) {
        for (i, p) in self.params.iter().enumerate() {
            if p.name == name {
                self.accessed.borrow_mut().insert(i);
            }
        }
    }

    fn floats(&self, name: &str) -> Option<&'a [f64]> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| match &p.value {
                ParamValue::Floats(v) => Some(v.as_slice()),
                _ => None,
            })
    }

    fn float(&self, name: &str) -> Option<f32> {
        self.floats(name).and_then(|v| v.first().map(|x| *x as f32))
    }

    fn string(&self, name: &str) -> Option<&'a str> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| match &p.value {
                ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
                _ => None,
            })
    }

    fn ints(&self, name: &str) -> Option<&'a [i64]> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| match &p.value {
                ParamValue::Ints(v) => Some(v.as_slice()),
                _ => None,
            })
    }

    fn bool(&self, name: &str) -> Option<bool> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Bool)
            .and_then(|p| match &p.value {
                ParamValue::Bools(v) => v.first().copied(),
                _ => None,
            })
    }

    fn rgb(&self, name: &str) -> Option<[f32; 3]> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| match &p.value {
                ParamValue::Floats(v) if v.len() >= 3 => {
                    Some([v[0] as f32, v[1] as f32, v[2] as f32])
                }
                _ => None,
            })
    }

    fn texture_ref(&self, name: &str) -> Option<&'a str> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Texture)
            .and_then(|p| match &p.value {
                ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
                _ => None,
            })
    }

    fn spectrum_avg(&self, name: &str) -> Option<f32> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Spectrum)
            .and_then(|p| match &p.value {
                ParamValue::Floats(v) if v.len() >= 2 => {
                    let values: Vec<f64> = v.iter().skip(1).step_by(2).copied().collect();
                    if values.is_empty() {
                        None
                    } else {
                        Some((values.iter().sum::<f64>() / values.len() as f64) as f32)
                    }
                }
                _ => None,
            })
    }

    fn spectrum_string(&self, name: &str) -> Option<&'a str> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Spectrum)
            .and_then(|p| match &p.value {
                ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
                _ => None,
            })
    }

    /// Sample interleaved spectrum [wavelength, value, ...] at R/G/B wavelengths.
    fn spectrum_rgb(&self, name: &str) -> Option<[f32; 3]> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Spectrum)
            .and_then(|p| match &p.value {
                ParamValue::Floats(v) if v.len() >= 4 => Some(sample_spectrum_rgb(v)),
                _ => None,
            })
    }

    fn blackbody(&self, name: &str) -> Option<f32> {
        self.mark(name);
        self.params
            .iter()
            .find(|p| p.name == name && p.ty == ParamType::Blackbody)
            .and_then(|p| match &p.value {
                ParamValue::Floats(v) => v.first().map(|x| *x as f32),
                _ => None,
            })
    }
}

impl Drop for ParamSet<'_> {
    fn drop(&mut self) {
        let accessed = self.accessed.borrow();
        for (i, p) in self.params.iter().enumerate() {
            if !accessed.contains(&i) {
                eprintln!(
                    "  warning: unhandled param \"{} {}\" in {}",
                    match p.ty {
                        ParamType::Integer => "integer",
                        ParamType::Float => "float",
                        ParamType::Point2 => "point2",
                        ParamType::Vector2 => "vector2",
                        ParamType::Point3 => "point3",
                        ParamType::Vector3 => "vector3",
                        ParamType::Normal3 => "normal3",
                        ParamType::Bool => "bool",
                        ParamType::String => "string",
                        ParamType::Rgb => "rgb",
                        ParamType::Spectrum => "spectrum",
                        ParamType::Blackbody => "blackbody",
                        ParamType::Texture => "texture",
                    },
                    p.name,
                    self.context
                );
            }
        }
    }
}

/// Linearly interpolate interleaved [wavelength, value, ...] spectrum at a target wavelength.
fn sample_spectrum_at(data: &[f64], wavelength: f64) -> f64 {
    let n = data.len() / 2;
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return data[1];
    }
    // Clamp to range
    if wavelength <= data[0] {
        return data[1];
    }
    if wavelength >= data[(n - 1) * 2] {
        return data[(n - 1) * 2 + 1];
    }
    // Find interval and lerp
    for i in 0..n - 1 {
        let w0 = data[i * 2];
        let v0 = data[i * 2 + 1];
        let w1 = data[(i + 1) * 2];
        let v1 = data[(i + 1) * 2 + 1];
        if wavelength >= w0 && wavelength <= w1 {
            let t = (wavelength - w0) / (w1 - w0);
            return v0 + t * (v1 - v0);
        }
    }
    data[(n - 1) * 2 + 1]
}

/// Sample interleaved spectrum data at representative R/G/B wavelengths.
fn sample_spectrum_rgb(data: &[f64]) -> [f32; 3] {
    // Representative wavelengths: R=630nm, G=532nm, B=467nm
    [
        sample_spectrum_at(data, 630.0) as f32,
        sample_spectrum_at(data, 532.0) as f32,
        sample_spectrum_at(data, 467.0) as f32,
    ]
}

/// RGB eta/k for named metal spectra (pre-computed from measured spectral data).
fn named_metal_eta(name: &str) -> Option<[f32; 3]> {
    if name.contains("Au") {
        Some([0.143, 0.374, 1.442])
    } else if name.contains("Ag") {
        Some([0.050, 0.056, 0.047])
    } else if name.contains("Cu") && !name.contains("CuZn") {
        Some([0.271, 0.677, 1.316])
    } else if name.contains("Al") {
        Some([1.654, 0.880, 0.521])
    } else if name.contains("CuZn") {
        Some([0.445, 0.582, 1.100])
    } else {
        None
    }
}

fn named_metal_k(name: &str) -> Option<[f32; 3]> {
    if name.contains("Au") {
        Some([3.983, 2.380, 1.603])
    } else if name.contains("Ag") {
        Some([4.484, 3.390, 2.440])
    } else if name.contains("Cu") && !name.contains("CuZn") {
        Some([3.610, 2.625, 2.292])
    } else if name.contains("Al") {
        Some([9.224, 6.270, 4.837])
    } else if name.contains("CuZn") {
        Some([3.600, 2.600, 1.900])
    } else {
        None
    }
}

/// Parse conductor eta from ParamSet: supports "spectrum eta" (named or inline), "rgb eta", "float eta".
fn parse_conductor_eta(p: &ParamSet) -> Option<[f32; 3]> {
    // Named spectrum string
    if let Some(s) = p.spectrum_string("eta") {
        if let Some(v) = named_metal_eta(s) {
            return Some(v);
        }
        eprintln!("  warning: unknown named spectrum for eta: {s}");
    }
    // Inline spectrum data
    if let Some(v) = p.spectrum_rgb("eta") {
        return Some(v);
    }
    // RGB
    if let Some(v) = p.rgb("eta") {
        return Some(v);
    }
    // Single float → uniform across channels
    if let Some(v) = p.float("eta") {
        return Some([v, v, v]);
    }
    None
}

/// Parse conductor k from ParamSet: supports "spectrum k" (named or inline), "rgb k", "float k".
fn parse_conductor_k(p: &ParamSet) -> Option<[f32; 3]> {
    if let Some(s) = p.spectrum_string("k") {
        if let Some(v) = named_metal_k(s) {
            return Some(v);
        }
        eprintln!("  warning: unknown named spectrum for k: {s}");
    }
    if let Some(v) = p.spectrum_rgb("k") {
        return Some(v);
    }
    if let Some(v) = p.rgb("k") {
        return Some(v);
    }
    if let Some(v) = p.float("k") {
        return Some([v, v, v]);
    }
    None
}

/// Compute normal-incidence reflectance from conductor eta/k.
fn conductor_f0(eta: &[f32; 3], k: &[f32; 3]) -> [f32; 3] {
    let mut f0 = [0.0f32; 3];
    for i in 0..3 {
        let e = eta[i];
        let kk = k[i];
        f0[i] = ((e - 1.0) * (e - 1.0) + kk * kk) / ((e + 1.0) * (e + 1.0) + kk * kk);
    }
    f0
}

/// Parse uroughness/vroughness into (roughness_u, roughness_v).
/// If only "roughness" is given, both are the same (isotropic).
/// Parse roughness and apply PBRT's RoughnessToAlpha mapping.
/// When remap=true (PBRT default): alpha = sqrt(roughness)
/// When remap=false: alpha = roughness (used directly as GGX alpha)
fn parse_roughness(p: &ParamSet, prefix: &str, remap: bool) -> (f32, f32) {
    let u_key = if prefix.is_empty() {
        "uroughness".to_string()
    } else {
        format!("{prefix}.uroughness")
    };
    let v_key = if prefix.is_empty() {
        "vroughness".to_string()
    } else {
        format!("{prefix}.vroughness")
    };
    let r_key = if prefix.is_empty() {
        "roughness".to_string()
    } else {
        format!("{prefix}.roughness")
    };

    let u = p.float(&u_key);
    let v = p.float(&v_key);
    let r = p.float(&r_key);

    let (ru, rv) = match (u, v, r) {
        (Some(u), Some(v), _) => (u, v),
        (Some(u), None, _) => (u, u),
        (None, Some(v), _) => (v, v),
        (None, None, Some(r)) => (r, r),
        _ => (0.0, 0.0),
    };

    if remap {
        (ru.sqrt(), rv.sqrt())
    } else {
        (ru, rv)
    }
}

fn blackbody_to_rgb(kelvin: f32) -> [f32; 3] {
    let temp = kelvin / 100.0;
    let r = if temp <= 66.0 {
        1.0
    } else {
        let x = temp - 60.0;
        (329.699_f32 * x.powf(-0.13320_f32) / 255.0).clamp(0.0, 1.0)
    };
    let g = if temp <= 66.0 {
        let x = temp;
        (99.4708_f32 * x.ln() - 161.1196_f32) / 255.0
    } else {
        let x = temp - 60.0;
        288.1222_f32 * x.powf(-0.07551_f32) / 255.0
    }
    .clamp(0.0, 1.0);
    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let x = temp - 10.0;
        (138.5177_f32 * x.ln() - 305.0448_f32) / 255.0
    }
    .clamp(0.0, 1.0);
    [r, g, b]
}

// ---- Shape parsing helper ----

fn parse_shape(ty: &str, params: &[pbrt_parser::Param], scene_dir: &Path) -> Option<SceneShape> {
    let p = ParamSet::new(params, format!("Shape \"{ty}\""));
    // Known but not fully used
    match ty {
        "sphere" => {
            let radius = p.float("radius").unwrap_or(1.0);
            Some(SceneShape::Sphere { radius })
        }
        "loopsubdiv" => {
            let verts: Vec<f32> = p
                .floats("P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let indices: Vec<i32> = p
                .ints("indices")
                .map(|v| v.iter().map(|x| *x as i32).collect())
                .unwrap_or_default();
            let levels = p
                .ints("levels")
                .and_then(|v| v.first().map(|x| *x as u32))
                .unwrap_or(3);
            let (subdivided_verts, subdivided_indices) =
                subdivision::loop_subdivide(&verts, &indices, levels);
            let normals =
                subdivision::compute_smooth_normals(&subdivided_verts, &subdivided_indices);
            Some(SceneShape::TriangleMesh {
                vertices: subdivided_verts,
                indices: subdivided_indices,
                texcoords: Vec::new(),
                normals,
            })
        }
        "trianglemesh" => {
            let verts: Vec<f32> = p
                .floats("P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let indices: Vec<i32> = p
                .ints("indices")
                .map(|v| v.iter().map(|x| *x as i32).collect())
                .unwrap_or_default();
            let texcoords: Vec<f32> = p
                .floats("uv")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let normals: Vec<f32> = p
                .floats("N")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            Some(SceneShape::TriangleMesh {
                vertices: verts,
                indices,
                texcoords,
                normals,
            })
        }
        "bilinearmesh" => {
            let verts: Vec<f32> = p
                .floats("P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let texcoords: Vec<f32> = if p.floats("uv").is_some() {
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            } else {
                Vec::new()
            };
            let indices = vec![0, 1, 3, 0, 3, 2];
            Some(SceneShape::TriangleMesh {
                vertices: verts,
                indices,
                texcoords,
                normals: Vec::new(),
            })
        }
        "disk" => {
            let radius = p.float("radius").unwrap_or(1.0);
            let inner_radius = p.float("innerradius").unwrap_or(0.0);
            let height = p.float("height").unwrap_or(0.0);
            let n_segments = 64;
            let mut verts = Vec::new();
            let mut indices = Vec::new();

            if inner_radius > 0.0 {
                for i in 0..n_segments {
                    let t0 = 2.0 * std::f32::consts::PI * i as f32 / n_segments as f32;
                    let t1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / n_segments as f32;
                    let base = verts.len() as i32 / 3;
                    verts.extend_from_slice(&[
                        inner_radius * t0.cos(),
                        inner_radius * t0.sin(),
                        height,
                        radius * t0.cos(),
                        radius * t0.sin(),
                        height,
                        radius * t1.cos(),
                        radius * t1.sin(),
                        height,
                        inner_radius * t1.cos(),
                        inner_radius * t1.sin(),
                        height,
                    ]);
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            } else {
                verts.extend_from_slice(&[0.0, 0.0, height]);
                for i in 0..=n_segments {
                    let theta = 2.0 * std::f32::consts::PI * i as f32 / n_segments as f32;
                    verts.extend_from_slice(&[radius * theta.cos(), radius * theta.sin(), height]);
                }
                for i in 0..n_segments {
                    indices.extend_from_slice(&[0, (i + 1) as i32, (i + 2) as i32]);
                }
            }
            let n_verts = verts.len() / 3;
            let normals = vec![0.0f32, 0.0, 1.0].repeat(n_verts);

            Some(SceneShape::TriangleMesh {
                vertices: verts,
                indices,
                texcoords: Vec::new(),
                normals,
            })
        }
        "plymesh" => {
            let filename = p.string("filename")?;
            let path = scene_dir.join(filename);
            let mesh = match ply::load(&path) {
                Some(m) => m,
                None => {
                    eprintln!("  warning: failed to load PLY: {}", path.display());
                    return None;
                }
            };
            Some(SceneShape::TriangleMesh {
                vertices: mesh.vertices,
                indices: mesh.indices,
                texcoords: mesh.texcoords,
                normals: mesh.normals,
            })
        }
        _ => {
            eprintln!("  warning: unsupported shape type: {ty}");
            None
        }
    }
}

// ---- Main scene parser ----

pub fn parse_scene(input: &str, scene_dir: &Path) -> ParsedScene {
    let scene = pbrt_parser::parse(input).expect("Failed to parse PBRT scene");

    let mut parsed = ParsedScene {
        width: 400,
        height: 400,
        fov: 90.0,
        cam_eye: [0.0, 0.0, 0.0],
        cam_look: [0.0, 0.0, -1.0],
        cam_up: [0.0, 1.0, 0.0],
        spp: 16,
        max_depth: 5,
        ambient_light: [0.0; 3],
        distant_lights: Vec::new(),
        sphere_lights: Vec::new(),
        triangle_lights: Vec::new(),
        triangle_light_groups: Vec::new(),
        objects: Vec::new(),
        filename: "output.png".to_string(),
        cam_flip_x: false,
        envmap: None,
    };

    #[derive(Clone)]
    struct CheckerTex {
        scale_u: f32,
        scale_v: f32,
        color1: [f32; 3],
        color2: [f32; 3],
    }
    #[derive(Clone)]
    enum SceneTexture {
        Checker(CheckerTex),
        Image(std::sync::Arc<ImageTexture>),
    }
    let mut textures = std::collections::HashMap::<String, SceneTexture>::new();
    let mut named_materials = std::collections::HashMap::<String, SceneMaterial>::new();
    // Named media: name -> tint color (exp(-sigma_a * typical_distance))
    let mut named_media = std::collections::HashMap::<String, [f32; 3]>::new();
    let mut current_material = SceneMaterial::default();
    let mut current_transform = transform::identity();
    let mut reverse_orientation = false;
    let mut transform_stack: Vec<([f32; 12], SceneMaterial, bool)> = Vec::new();
    let mut named_coord_systems: std::collections::HashMap<String, [f32; 12]> =
        std::collections::HashMap::new();
    let mut _in_world = false;

    // Object instancing: ObjectBegin/ObjectEnd/ObjectInstance
    let mut named_objects: std::collections::HashMap<String, Vec<SceneObject>> =
        std::collections::HashMap::new();
    let mut current_object_name: Option<String> = None;

    let mut directive_queue: std::collections::VecDeque<Directive> =
        scene.directives.into_iter().collect();

    while let Some(ref directive) = directive_queue.pop_front() {
        match directive {
            Directive::Film { params, .. } => {
                let p = ParamSet::new(params, "Film");
                if let Some(v) = p.ints("xresolution") {
                    parsed.width = v[0] as u32;
                }
                if let Some(v) = p.ints("yresolution") {
                    parsed.height = v[0] as u32;
                }
                if let Some(s) = p.string("filename") {
                    parsed.filename = s.to_string();
                }
            }
            Directive::Camera { params, .. } => {
                let p = ParamSet::new(params, "Camera");
                if let Some(f) = p.float("fov") {
                    parsed.fov = f;
                }
            }
            Directive::LookAt { eye, look, up } => {
                let e = [eye[0] as f32, eye[1] as f32, eye[2] as f32];
                let l = [look[0] as f32, look[1] as f32, look[2] as f32];
                let u = [up[0] as f32, up[1] as f32, up[2] as f32];

                // Detect handedness flip from pre-LookAt Scale (e.g. Scale -1 1 1)
                if current_transform != transform::identity() {
                    let t = &current_transform;
                    let det = t[0] * (t[5] * t[10] - t[6] * t[9])
                        - t[1] * (t[4] * t[10] - t[6] * t[8])
                        + t[2] * (t[4] * t[9] - t[5] * t[8]);
                    if det < 0.0 {
                        parsed.cam_flip_x = true;
                    }
                    current_transform = transform::identity();
                }

                parsed.cam_eye = e;
                parsed.cam_look = l;
                parsed.cam_up = u;
            }
            Directive::Sampler { params, .. } => {
                let p = ParamSet::new(params, "Sampler");
                if let Some(v) = p.ints("pixelsamples") {
                    parsed.spp = v[0] as u32;
                }
            }
            Directive::Integrator { params, .. } => {
                let p = ParamSet::new(params, "Integrator");
                if let Some(v) = p.ints("maxdepth") {
                    parsed.max_depth = v[0] as u32;
                }
            }
            Directive::WorldBegin => {
                // In PBRT, CTM = LookAt * R (post-multiplied). This means
                // the world is further transformed by R before the camera
                // sees it. Equivalently, eye/look/up are transformed by
                // the inverse of R. For a rotation, inverse = transpose.
                if current_transform != transform::identity() {
                    let t = &current_transform;
                    // CTM = LookAt * R is camera-from-world. Camera vectors
                    // need world-from-camera = R^-1 * LookAt^-1.
                    // Apply R^-1 (transpose for rotations) to eye, look, up.
                    let transform_point = |p: [f32; 3]| -> [f32; 3] {
                        [
                            t[0] * p[0] + t[4] * p[1] + t[8] * p[2] + t[3],
                            t[1] * p[0] + t[5] * p[1] + t[9] * p[2] + t[7],
                            t[2] * p[0] + t[6] * p[1] + t[10] * p[2] + t[11],
                        ]
                    };
                    let transform_vec = |v: [f32; 3]| -> [f32; 3] {
                        [
                            t[0] * v[0] + t[4] * v[1] + t[8] * v[2],
                            t[1] * v[0] + t[5] * v[1] + t[9] * v[2],
                            t[2] * v[0] + t[6] * v[1] + t[10] * v[2],
                        ]
                    };
                    parsed.cam_eye = transform_point(parsed.cam_eye);
                    parsed.cam_look = transform_point(parsed.cam_look);
                    parsed.cam_up = transform_vec(parsed.cam_up);
                }
                // Save built-in coordinate systems
                named_coord_systems.insert("camera".to_string(), current_transform);
                named_coord_systems.insert("world".to_string(), transform::identity());
                current_transform = transform::identity();
                _in_world = true;
            }
            Directive::AttributeBegin => {
                transform_stack.push((
                    current_transform,
                    current_material.clone(),
                    reverse_orientation,
                ));
            }
            Directive::AttributeEnd => {
                if let Some((t, m, ro)) = transform_stack.pop() {
                    current_transform = t;
                    current_material = m;
                    reverse_orientation = ro;
                }
            }
            Directive::ReverseOrientation => {
                reverse_orientation = !reverse_orientation;
            }
            Directive::CoordinateSystem(name) => {
                named_coord_systems.insert(name.clone(), current_transform);
            }
            Directive::CoordSysTransform(name) => {
                if let Some(t) = named_coord_systems.get(name) {
                    current_transform = *t;
                } else {
                    eprintln!("Warning: unknown coordinate system: {name}");
                }
            }
            Directive::ObjectBegin(name) => {
                current_object_name = Some(name.clone());
                named_objects.entry(name.clone()).or_default();
            }
            Directive::ObjectEnd => {
                current_object_name = None;
            }
            Directive::ObjectInstance(name) => {
                if let Some(obj_shapes) = named_objects.get(name) {
                    for obj in obj_shapes.clone() {
                        let combined_transform = transform::mul(&current_transform, &obj.transform);
                        register_area_light(
                            &obj.shape,
                            &obj.material,
                            &combined_transform,
                            &mut parsed,
                        );
                        parsed.objects.push(SceneObject {
                            shape: obj.shape,
                            material: obj.material,
                            transform: combined_transform,
                        });
                    }
                } else {
                    eprintln!("Warning: unknown object instance: {name}");
                }
            }
            Directive::Translate { v } => {
                let t = transform::translate(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = transform::mul(&current_transform, &t);
            }
            Directive::Scale { v } => {
                let s = transform::scale(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = transform::mul(&current_transform, &s);
            }
            Directive::Rotate { angle, axis } => {
                let r = transform::rotate(
                    *angle as f32,
                    axis[0] as f32,
                    axis[1] as f32,
                    axis[2] as f32,
                );
                current_transform = transform::mul(&current_transform, &r);
            }
            Directive::Identity => {
                current_transform = transform::identity();
            }
            Directive::Transform { m } => {
                // PBRT Transform uses column-major 4x4, convert to row-major 3x4
                current_transform = [
                    m[0] as f32,
                    m[4] as f32,
                    m[8] as f32,
                    m[12] as f32,
                    m[1] as f32,
                    m[5] as f32,
                    m[9] as f32,
                    m[13] as f32,
                    m[2] as f32,
                    m[6] as f32,
                    m[10] as f32,
                    m[14] as f32,
                ];
            }
            Directive::ConcatTransform { m } => {
                let t = [
                    m[0] as f32,
                    m[4] as f32,
                    m[8] as f32,
                    m[12] as f32,
                    m[1] as f32,
                    m[5] as f32,
                    m[9] as f32,
                    m[13] as f32,
                    m[2] as f32,
                    m[6] as f32,
                    m[10] as f32,
                    m[14] as f32,
                ];
                current_transform = transform::mul(&current_transform, &t);
            }
            Directive::LightSource { ty, params } => match ty.as_str() {
                "infinite" => {
                    let p = ParamSet::new(params, "LightSource \"infinite\"");
                    let float_scale = p.float("scale").unwrap_or(1.0);
                    let scale = p.rgb("L").unwrap_or([1.0, 1.0, 1.0]);
                    let scale = [
                        scale[0] * float_scale,
                        scale[1] * float_scale,
                        scale[2] * float_scale,
                    ];
                    if let Some(filename) = p.string("filename") {
                        let path = scene_dir.join(filename);
                        match image::open(&path) {
                            Ok(img) => {
                                let rgb = img.to_rgb32f();
                                let (w, h) = rgb.dimensions();
                                let mut data: Vec<f32> = rgb.into_raw();
                                // Apply L scale to envmap pixels
                                for pixel in data.chunks_exact_mut(3) {
                                    pixel[0] *= scale[0];
                                    pixel[1] *= scale[1];
                                    pixel[2] *= scale[2];
                                }
                                println!("Loaded envmap: {}x{} from {}", w, h, path.display());
                                parsed.envmap = Some(ImageTexture {
                                    data,
                                    width: w,
                                    height: h,
                                });
                            }
                            Err(e) => {
                                eprintln!("Failed to load envmap {}: {e}", path.display());
                                parsed.ambient_light = scale;
                            }
                        }
                    } else {
                        parsed.ambient_light = scale;
                    }
                }
                "distant" => {
                    let p = ParamSet::new(params, "LightSource \"distant\"");
                    let from = p
                        .floats("from")
                        .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                        .unwrap_or([0.0, 0.0, 1.0]);
                    let to = p
                        .floats("to")
                        .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                        .unwrap_or([0.0, 0.0, 0.0]);
                    let dir = {
                        let dx = from[0] - to[0];
                        let dy = from[1] - to[1];
                        let dz = from[2] - to[2];
                        let len = (dx * dx + dy * dy + dz * dz).sqrt();
                        [dx / len, dy / len, dz / len]
                    };
                    let mut emission = p.rgb("L").unwrap_or([1.0, 1.0, 1.0]);
                    if let Some(k) = p.blackbody("L") {
                        emission = blackbody_to_rgb(k);
                    }
                    let scale = p.float("scale").unwrap_or(1.0);
                    emission[0] *= scale;
                    emission[1] *= scale;
                    emission[2] *= scale;
                    parsed.distant_lights.push(DistantLight {
                        direction: dir,
                        emission,
                    });
                }
                _ => eprintln!("Unsupported light type: {ty}"),
            },
            Directive::Material { ty, params } => {
                let p = ParamSet::new(params, format!("Material \"{ty}\""));
                current_material = SceneMaterial::default();
                match ty.as_str() {
                    "diffuse" => {
                        current_material.material_type = MAT_DIFFUSE;
                        if let Some(c) = p.rgb("reflectance") {
                            current_material.albedo = c;
                        }
                        if let Some(tex_name) = p.texture_ref("reflectance") {
                            match textures.get(tex_name) {
                                Some(SceneTexture::Checker(tex)) => {
                                    current_material.has_checkerboard = true;
                                    current_material.checker_scale_u = tex.scale_u;
                                    current_material.checker_scale_v = tex.scale_v;
                                    current_material.checker_color1 = tex.color1;
                                    current_material.checker_color2 = tex.color2;
                                }
                                Some(SceneTexture::Image(img)) => {
                                    current_material.texture = Some(img.clone());
                                }
                                None => {
                                    eprintln!("  warning: unknown texture reference: {}", tex_name);
                                }
                            }
                        }
                    }
                    "coateddiffuse" => {
                        current_material.material_type = MAT_COATED_DIFFUSE;
                        if let Some(c) = p.rgb("reflectance") {
                            current_material.albedo = c;
                        }
                        if let Some(tex_name) = p.texture_ref("reflectance") {
                            match textures.get(tex_name) {
                                Some(SceneTexture::Checker(tex)) => {
                                    current_material.has_checkerboard = true;
                                    current_material.checker_scale_u = tex.scale_u;
                                    current_material.checker_scale_v = tex.scale_v;
                                    current_material.checker_color1 = tex.color1;
                                    current_material.checker_color2 = tex.color2;
                                }
                                Some(SceneTexture::Image(img)) => {
                                    current_material.texture = Some(img.clone());
                                }
                                None => {
                                    eprintln!("  warning: unknown texture reference: {}", tex_name);
                                }
                            }
                        }
                        let remap = p.bool("remaproughness").unwrap_or(true);
                        let (ru, rv) = parse_roughness(&p, "", remap);
                        current_material.roughness = ru;
                        current_material.roughness_v = rv;
                        current_material.coat_eta = p.float("eta").unwrap_or(1.5);
                    }
                    "conductor" | "coatedconductor" => {
                        let is_coated = ty == "coatedconductor";
                        current_material.material_type = if is_coated {
                            MAT_COATED_CONDUCTOR
                        } else {
                            MAT_CONDUCTOR
                        };
                        if let Some(c) = p.rgb("reflectance") {
                            current_material.albedo = c;
                        } else if let Some(eta) = parse_conductor_eta(&p) {
                            current_material.conductor_eta = eta;
                            if let Some(k) = parse_conductor_k(&p) {
                                current_material.conductor_k = k;
                            }
                            current_material.albedo = conductor_f0(
                                &current_material.conductor_eta,
                                &current_material.conductor_k,
                            );
                        } else {
                            // Default: gold-like (PBRT defaults to Cu for coatedconductor)
                            current_material.conductor_eta = [0.143, 0.374, 1.442];
                            current_material.conductor_k = [3.983, 2.380, 1.603];
                            current_material.albedo = conductor_f0(
                                &current_material.conductor_eta,
                                &current_material.conductor_k,
                            );
                        }
                        let remap = p.bool("remaproughness").unwrap_or(true);
                        if is_coated {
                            let (ru, rv) = parse_roughness(&p, "conductor", remap);
                            current_material.roughness = ru;
                            current_material.roughness_v = rv;
                            current_material.coat_roughness =
                                parse_roughness(&p, "interface", remap).0;
                            current_material.coat_eta = p.float("interface.eta").unwrap_or(1.5);
                        } else {
                            let (ru, rv) = parse_roughness(&p, "", remap);
                            current_material.roughness = ru;
                            current_material.roughness_v = rv;
                        }
                    }
                    "dielectric" | "thindielectric" => {
                        current_material.material_type = MAT_DIELECTRIC;
                        current_material.eta = p
                            .float("eta")
                            .or_else(|| p.spectrum_avg("eta"))
                            .unwrap_or(1.5);
                    }
                    _ => eprintln!("  warning: unsupported material type: {ty}"),
                }
                // Mark known-but-unhandled params so they don't trigger warnings
                if let Some(tex_name) = p.texture_ref("displacement") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => {
                            current_material.bump_map = Some(img.clone())
                        }
                        Some(_) => {}
                        None => eprintln!("  warning: displacement texture not found: {tex_name}"),
                    }
                }
                if let Some(tex_name) = p.texture_ref("alpha") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => {
                            current_material.alpha_map = Some(img.clone())
                        }
                        Some(_) => {}
                        None => eprintln!("  warning: alpha texture not found: {tex_name}"),
                    }
                }
                if let Some(tex_name) = p.texture_ref("roughness") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => {
                            current_material.roughness_map = Some(img.clone())
                        }
                        Some(_) => {}
                        None => eprintln!("  warning: roughness texture not found: {tex_name}"),
                    }
                }
            }
            Directive::MakeNamedMaterial { name, params } => {
                let p = ParamSet::new(params, format!("MakeNamedMaterial \"{name}\""));
                let ty = p.string("type").unwrap_or("diffuse");
                let mut mat = SceneMaterial::default();
                match ty {
                    "diffuse" => {
                        mat.material_type = MAT_DIFFUSE;
                        if let Some(c) = p.rgb("reflectance") {
                            mat.albedo = c;
                        }
                        if let Some(tex_name) = p.texture_ref("reflectance") {
                            match textures.get(tex_name) {
                                Some(SceneTexture::Checker(tex)) => {
                                    mat.has_checkerboard = true;
                                    mat.checker_scale_u = tex.scale_u;
                                    mat.checker_scale_v = tex.scale_v;
                                    mat.checker_color1 = tex.color1;
                                    mat.checker_color2 = tex.color2;
                                }
                                Some(SceneTexture::Image(img)) => {
                                    mat.texture = Some(img.clone());
                                }
                                None => {
                                    eprintln!("  warning: unknown texture reference: {}", tex_name);
                                }
                            }
                        }
                    }
                    "coateddiffuse" => {
                        mat.material_type = MAT_COATED_DIFFUSE;
                        if let Some(c) = p.rgb("reflectance") {
                            mat.albedo = c;
                        }
                        let remap = p.bool("remaproughness").unwrap_or(true);
                        let (ru, rv) = parse_roughness(&p, "", remap);
                        mat.roughness = ru;
                        mat.roughness_v = rv;
                        mat.coat_eta = p.float("eta").unwrap_or(1.5);
                        if let Some(tex_name) = p.texture_ref("reflectance") {
                            if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                                mat.texture = Some(img.clone());
                            }
                        }
                    }
                    "conductor" | "coatedconductor" => {
                        let is_coated = ty == "coatedconductor";
                        mat.material_type = if is_coated {
                            MAT_COATED_CONDUCTOR
                        } else {
                            MAT_CONDUCTOR
                        };
                        if let Some(c) = p.rgb("reflectance") {
                            mat.albedo = c;
                        } else if let Some(eta) = parse_conductor_eta(&p) {
                            mat.conductor_eta = eta;
                            if let Some(k) = parse_conductor_k(&p) {
                                mat.conductor_k = k;
                            }
                            mat.albedo = conductor_f0(&mat.conductor_eta, &mat.conductor_k);
                        } else {
                            mat.conductor_eta = [0.143, 0.374, 1.442];
                            mat.conductor_k = [3.983, 2.380, 1.603];
                            mat.albedo = conductor_f0(&mat.conductor_eta, &mat.conductor_k);
                        }
                        let remap = p.bool("remaproughness").unwrap_or(true);
                        if is_coated {
                            let (ru, rv) = parse_roughness(&p, "conductor", remap);
                            mat.roughness = ru;
                            mat.roughness_v = rv;
                            mat.coat_roughness = parse_roughness(&p, "interface", remap).0;
                            mat.coat_eta = p.float("interface.eta").unwrap_or(1.5);
                        } else {
                            let (ru, rv) = parse_roughness(&p, "", remap);
                            mat.roughness = ru;
                            mat.roughness_v = rv;
                        }
                    }
                    "dielectric" | "thindielectric" => {
                        mat.material_type = MAT_DIELECTRIC;
                        mat.eta = p
                            .float("eta")
                            .or_else(|| p.spectrum_avg("eta"))
                            .unwrap_or(1.5);
                    }
                    "mix" => {
                        // Use the first referenced material as approximation
                        if let Some(param) = params.iter().find(|param| param.name == "materials") {
                            if let ParamValue::Strings(names) = &param.value {
                                if let Some(first) = names.first() {
                                    if let Some(base) = named_materials.get(first.as_str()) {
                                        mat = base.clone();
                                    }
                                }
                            }
                        }
                    }
                    "measured" => {
                        let filename = p.string("filename").unwrap_or("");
                        let bsdf_path = scene_dir.join(filename);
                        if let Some(approx) = crate::bsdf::load_and_approximate(&bsdf_path) {
                            mat.albedo = approx.albedo;
                            mat.roughness = approx.roughness;
                            if approx.is_metallic {
                                mat.material_type = MAT_CONDUCTOR;
                                mat.conductor_eta = approx.eta;
                                mat.conductor_k = approx.k;
                            } else {
                                mat.material_type = MAT_COATED_DIFFUSE;
                                mat.coat_eta = 1.5;
                            }
                            eprintln!(
                                "  Loaded measured BSDF: {} → {} (albedo=[{:.2},{:.2},{:.2}], roughness={:.2})",
                                filename,
                                if approx.is_metallic { "conductor" } else { "coateddiffuse" },
                                approx.albedo[0], approx.albedo[1], approx.albedo[2],
                                approx.roughness,
                            );
                        } else {
                            // Fallback: coated diffuse
                            mat.material_type = MAT_COATED_DIFFUSE;
                            mat.coat_eta = 1.5;
                            mat.roughness = 0.1;
                            eprintln!(
                                "  warning: failed to load measured BSDF: {}, using fallback",
                                filename,
                            );
                        }
                    }
                    _ => {
                        eprintln!("  warning: unsupported MakeNamedMaterial type: {ty}");
                    }
                }
                if let Some(tex_name) = p.texture_ref("displacement") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => mat.bump_map = Some(img.clone()),
                        Some(_) => {}
                        None => eprintln!("  warning: displacement texture not found: {tex_name}"),
                    }
                }
                if let Some(tex_name) = p.texture_ref("alpha") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => mat.alpha_map = Some(img.clone()),
                        Some(_) => {}
                        None => eprintln!("  warning: alpha texture not found: {tex_name}"),
                    }
                }
                if let Some(tex_name) = p.texture_ref("roughness") {
                    match textures.get(tex_name) {
                        Some(SceneTexture::Image(img)) => mat.roughness_map = Some(img.clone()),
                        Some(_) => {}
                        None => eprintln!("  warning: roughness texture not found: {tex_name}"),
                    }
                }
                named_materials.insert(name.clone(), mat);
            }
            Directive::NamedMaterial(name) => {
                if let Some(mat) = named_materials.get(name.as_str()) {
                    current_material = mat.clone();
                } else {
                    eprintln!("Unknown named material: {name}");
                }
            }
            Directive::Texture {
                name,
                class,
                params,
                ..
            } => {
                let p = ParamSet::new(params, format!("Texture \"{name}\" \"{class}\""));
                if class == "checkerboard" {
                    textures.insert(
                        name.clone(),
                        SceneTexture::Checker(CheckerTex {
                            scale_u: p.float("uscale").unwrap_or(1.0),
                            scale_v: p.float("vscale").unwrap_or(1.0),
                            color1: p.rgb("tex1").unwrap_or([1.0, 1.0, 1.0]),
                            color2: p.rgb("tex2").unwrap_or([0.0, 0.0, 0.0]),
                        }),
                    );
                } else if class == "imagemap" {
                    if let Some(filename) = p.string("filename") {
                        let path = scene_dir.join(filename);
                        match image::open(&path) {
                            Ok(img) => {
                                let rgb = img.to_rgb32f();
                                let (w, h) = rgb.dimensions();
                                let data: Vec<f32> = rgb.into_raw();
                                println!("Loaded texture: {}x{} from {}", w, h, path.display());
                                textures.insert(
                                    name.clone(),
                                    SceneTexture::Image(std::sync::Arc::new(ImageTexture {
                                        data,
                                        width: w,
                                        height: h,
                                    })),
                                );
                            }
                            Err(e) => eprintln!("Failed to load texture {}: {e}", path.display()),
                        }
                    }
                } else if class == "constant" {
                    // Constant texture: single RGB or float value
                    // Store as a 1x1 imagemap so it works in texture chains
                    let color = p.rgb("value").unwrap_or_else(|| {
                        let v = p.float("value").unwrap_or(1.0);
                        [v, v, v]
                    });
                    textures.insert(
                        name.clone(),
                        SceneTexture::Image(std::sync::Arc::new(ImageTexture {
                            data: vec![color[0], color[1], color[2]],
                            width: 1,
                            height: 1,
                        })),
                    );
                } else if class == "scale" {
                    let scale_val = p.float("scale").unwrap_or(1.0);
                    if let Some(tex_ref) = p.texture_ref("tex") {
                        if let Some(SceneTexture::Image(base)) = textures.get(tex_ref) {
                            let scaled_data: Vec<f32> =
                                base.data.iter().map(|v| v * scale_val).collect();
                            textures.insert(
                                name.clone(),
                                SceneTexture::Image(std::sync::Arc::new(ImageTexture {
                                    data: scaled_data,
                                    width: base.width,
                                    height: base.height,
                                })),
                            );
                        } else if let Some(base) = textures.get(tex_ref) {
                            textures.insert(name.clone(), base.clone());
                        }
                    }
                } else if class == "mix" {
                    if let Some(tex_ref) = p.texture_ref("tex") {
                        if let Some(base) = textures.get(tex_ref) {
                            textures.insert(name.clone(), base.clone());
                        }
                    }
                    if let Some(tex_ref) = p.texture_ref("tex1") {
                        if let Some(base) = textures.get(tex_ref) {
                            textures.insert(name.clone(), base.clone());
                        }
                    }
                } else {
                    eprintln!("  warning: unsupported texture class: {class} (texture \"{name}\")");
                }
            }
            Directive::Shape { ty, params } => {
                if let Some(mut shape) = parse_shape(ty, params, scene_dir) {
                    if reverse_orientation {
                        if let SceneShape::TriangleMesh {
                            ref mut indices,
                            ref mut normals,
                            ..
                        } = shape
                        {
                            for tri in indices.chunks_exact_mut(3) {
                                tri.swap(1, 2);
                            }
                            for n in normals.iter_mut() {
                                *n = -*n;
                            }
                        }
                    }
                    let obj = SceneObject {
                        shape,
                        material: current_material.clone(),
                        transform: current_transform,
                    };
                    if let Some(ref name) = current_object_name {
                        // Inside ObjectBegin: collect shapes into named group
                        named_objects.get_mut(name).unwrap().push(obj);
                    } else {
                        register_area_light(&obj.shape, &obj.material, &obj.transform, &mut parsed);
                        parsed.objects.push(obj);
                    }
                }
            }
            Directive::AreaLightSource { ty, params } => {
                if ty == "diffuse" {
                    let p = ParamSet::new(params, "AreaLightSource \"diffuse\"");
                    let mut emission = p.rgb("L").unwrap_or([1.0, 1.0, 1.0]);
                    if let Some(k) = p.blackbody("L") {
                        emission = blackbody_to_rgb(k);
                    }
                    let scale = p.float("scale").unwrap_or(1.0);
                    emission[0] *= scale;
                    emission[1] *= scale;
                    emission[2] *= scale;
                    current_material.emission = emission;
                }
            }
            Directive::MakeNamedMedium { name, params } => {
                let p = ParamSet::new(params, format!("MakeNamedMedium \"{name}\""));
                if let Some(sigma_a) = p.rgb("sigma_a") {
                    let d = 3.0; // approximate gem thickness
                    let tint = [
                        (-sigma_a[0] * d).exp(),
                        (-sigma_a[1] * d).exp(),
                        (-sigma_a[2] * d).exp(),
                    ];
                    named_media.insert(name.clone(), tint);
                }
            }
            Directive::MediumInterface { interior, .. } => {
                if let Some(tint) = named_media.get(interior.as_str()) {
                    current_material.tint = *tint;
                }
            }
            Directive::Include(path) => {
                let include_path = scene_dir.join(path);
                match std::fs::read_to_string(&include_path) {
                    Ok(content) => {
                        let included = pbrt_parser::parse(&content).unwrap_or_else(|e| {
                            panic!("Failed to parse {}: {e}", include_path.display())
                        });
                        for (i, d) in included.directives.into_iter().enumerate() {
                            directive_queue.insert(i, d);
                        }
                    }
                    Err(e) => eprintln!("Failed to include {}: {e}", include_path.display()),
                }
            }
            other => {
                eprintln!("Warning: unhandled directive: {}", directive_name(other));
            }
        }
    }
    parsed
}

fn directive_name(d: &Directive) -> &'static str {
    match d {
        Directive::Camera { .. } => "Camera",
        Directive::Film { .. } => "Film",
        Directive::Sampler { .. } => "Sampler",
        Directive::Integrator { .. } => "Integrator",
        Directive::PixelFilter { .. } => "PixelFilter",
        Directive::Accelerator { .. } => "Accelerator",
        Directive::ColorSpace(_) => "ColorSpace",
        Directive::Option { .. } => "Option",
        Directive::Identity => "Identity",
        Directive::Translate { .. } => "Translate",
        Directive::Scale { .. } => "Scale",
        Directive::Rotate { .. } => "Rotate",
        Directive::LookAt { .. } => "LookAt",
        Directive::Transform { .. } => "Transform",
        Directive::ConcatTransform { .. } => "ConcatTransform",
        Directive::CoordinateSystem(_) => "CoordinateSystem",
        Directive::CoordSysTransform(_) => "CoordSysTransform",
        Directive::TransformTimes { .. } => "TransformTimes",
        Directive::ActiveTransform(_) => "ActiveTransform",
        Directive::ReverseOrientation => "ReverseOrientation",
        Directive::WorldBegin => "WorldBegin",
        Directive::AttributeBegin => "AttributeBegin",
        Directive::AttributeEnd => "AttributeEnd",
        Directive::Attribute { .. } => "Attribute",
        Directive::Shape { .. } => "Shape",
        Directive::Material { .. } => "Material",
        Directive::MakeNamedMaterial { .. } => "MakeNamedMaterial",
        Directive::NamedMaterial(_) => "NamedMaterial",
        Directive::Texture { .. } => "Texture",
        Directive::AreaLightSource { .. } => "AreaLightSource",
        Directive::LightSource { .. } => "LightSource",
        Directive::MakeNamedMedium { .. } => "MakeNamedMedium",
        Directive::MediumInterface { .. } => "MediumInterface",
        Directive::ObjectBegin(_) => "ObjectBegin",
        Directive::ObjectEnd => "ObjectEnd",
        Directive::ObjectInstance(_) => "ObjectInstance",
        Directive::Include(_) => "Include",
        Directive::Import(_) => "Import",
    }
}

fn register_area_light(
    shape: &SceneShape,
    mat: &SceneMaterial,
    xform: &[f32; 12],
    scene: &mut ParsedScene,
) {
    let em = mat.emission;
    if em[0] > 0.0 || em[1] > 0.0 || em[2] > 0.0 {
        if let SceneShape::Sphere { radius } = shape {
            let center = [xform[3], xform[7], xform[11]];
            scene.sphere_lights.push(SphereLight {
                center,
                radius: *radius,
                emission: em,
                _pad: 0.0,
            });
        }
        if let SceneShape::TriangleMesh {
            vertices, indices, ..
        } = shape
        {
            let start = scene.triangle_lights.len() as u32;
            let lum = 0.2126 * em[0] + 0.7152 * em[1] + 0.0722 * em[2];
            let mut group_power = 0.0f32;
            let transformed = transform::transform_vertices(vertices, xform);
            for tri in indices.chunks(3) {
                if tri.len() < 3 {
                    continue;
                }
                let i0 = tri[0] as usize * 3;
                let i1 = tri[1] as usize * 3;
                let i2 = tri[2] as usize * 3;
                let v0 = [transformed[i0], transformed[i0 + 1], transformed[i0 + 2]];
                let v1 = [transformed[i1], transformed[i1 + 1], transformed[i1 + 2]];
                let v2 = [transformed[i2], transformed[i2 + 1], transformed[i2 + 2]];
                let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let area = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() * 0.5;
                let len = area * 2.0;
                let normal = if len > 0.0 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 1.0, 0.0]
                };
                group_power += area * lum;
                scene.triangle_lights.push(TriangleLight {
                    v0,
                    v1,
                    v2,
                    emission: em,
                    normal,
                    area,
                    _pad: 0.0,
                });
            }
            let count = scene.triangle_lights.len() as u32 - start;
            if count > 0 {
                scene.triangle_light_groups.push(TriangleLightGroup {
                    start,
                    count,
                    total_power: group_power,
                    _pad: 0.0,
                });
            }
        }
    }
}
