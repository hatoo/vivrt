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
    pub objects: Vec<SceneObject>,
    pub filename: String,
    pub cam_flip_x: bool,
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

/// Approximate reflectance color for named metal spectra.
fn metal_color_from_params(p: &ParamSet) -> Option<[f32; 3]> {
    let eta_str = p.spectrum_string("eta");
    match eta_str {
        Some(s) if s.contains("Au") => Some([1.0, 0.78, 0.34]), // gold
        Some(s) if s.contains("Ag") => Some([0.97, 0.96, 0.91]), // silver
        Some(s) if s.contains("Cu") && !s.contains("CuZn") => Some([0.96, 0.64, 0.54]), // copper
        Some(s) if s.contains("Al") => Some([0.91, 0.92, 0.93]), // aluminum
        Some(s) if s.contains("CuZn") => Some([0.94, 0.83, 0.49]), // brass
        _ => None,
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
        "plymesh" => {
            let filename = p.string("filename")?;
            let mesh = ply::load(&scene_dir.join(filename))?;
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
        objects: Vec::new(),
        filename: "output.png".to_string(),
        cam_flip_x: false,
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
    let mut transform_stack: Vec<([f32; 12], SceneMaterial)> = Vec::new();
    let mut _in_world = false;

    for directive in &scene.directives {
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
                // Known but not implemented
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
                current_transform = transform::identity();
                _in_world = true;
            }
            Directive::AttributeBegin => {
                transform_stack.push((current_transform, current_material.clone()));
            }
            Directive::AttributeEnd => {
                if let Some((t, m)) = transform_stack.pop() {
                    current_transform = t;
                    current_material = m;
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
                    if let Some(c) = p.rgb("L") {
                        parsed.ambient_light = c;
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
                                None => {}
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
                                None => {}
                            }
                        }
                        current_material.roughness = p.float("roughness").unwrap_or(0.0);
                    }
                    "conductor" | "coatedconductor" => {
                        current_material.material_type = MAT_CONDUCTOR;
                        if let Some(c) = p.rgb("reflectance") {
                            current_material.albedo = c;
                        } else if let Some(c) = metal_color_from_params(&p) {
                            current_material.albedo = c;
                        } else {
                            current_material.albedo = [0.8, 0.7, 0.3];
                        }
                        current_material.roughness = p
                            .float("roughness")
                            .or(p.float("uroughness"))
                            .unwrap_or(0.01);
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
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        current_material.bump_map = Some(img.clone());
                    }
                }
                if let Some(tex_name) = p.texture_ref("alpha") {
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        current_material.alpha_map = Some(img.clone());
                    }
                }
                if let Some(tex_name) = p.texture_ref("roughness") {
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        current_material.roughness_map = Some(img.clone());
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
                                None => {}
                            }
                        }
                    }
                    "coateddiffuse" => {
                        mat.material_type = MAT_COATED_DIFFUSE;
                        if let Some(c) = p.rgb("reflectance") {
                            mat.albedo = c;
                        }
                        mat.roughness = p
                            .float("roughness")
                            .or(p.float("uroughness"))
                            .unwrap_or(0.0);
                        if let Some(tex_name) = p.texture_ref("reflectance") {
                            if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                                mat.texture = Some(img.clone());
                            }
                        }
                    }
                    "conductor" | "coatedconductor" => {
                        mat.material_type = MAT_CONDUCTOR;
                        if let Some(c) = p.rgb("reflectance") {
                            mat.albedo = c;
                        } else if let Some(c) = metal_color_from_params(&p) {
                            mat.albedo = c;
                        } else {
                            mat.albedo = [0.8, 0.7, 0.3];
                        }
                        mat.roughness = p
                            .float("roughness")
                            .or(p.float("uroughness"))
                            .unwrap_or(0.01);
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
                    _ => {
                        eprintln!("  warning: unsupported MakeNamedMaterial type: {ty}");
                    }
                }
                // Mark known-but-unhandled params so they don't trigger warnings
                // Bind displacement/bump map for any material type
                if let Some(tex_name) = p.texture_ref("displacement") {
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        mat.bump_map = Some(img.clone());
                    }
                }
                if let Some(tex_name) = p.texture_ref("alpha") {
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        mat.alpha_map = Some(img.clone());
                    }
                }
                if let Some(tex_name) = p.texture_ref("roughness") {
                    if let Some(SceneTexture::Image(img)) = textures.get(tex_name) {
                        mat.roughness_map = Some(img.clone());
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
                } else if class == "scale" || class == "mix" {
                    if let Some(tex_ref) = p.texture_ref("tex") {
                        if let Some(base) = textures.get(tex_ref) {
                            textures.insert(name.clone(), base.clone());
                        }
                    }
                    // Also check tex1/tex2 for mix
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
                if let Some(shape) = parse_shape(ty, params, scene_dir) {
                    register_area_light(&shape, &current_material, &current_transform, &mut parsed);
                    parsed.objects.push(SceneObject {
                        shape,
                        material: current_material.clone(),
                        transform: current_transform,
                    });
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
                        for inc_directive in &included.directives {
                            if let Directive::Shape { ty, params } = inc_directive {
                                if let Some(shape) = parse_shape(ty, params, scene_dir) {
                                    parsed.objects.push(SceneObject {
                                        shape,
                                        material: current_material.clone(),
                                        transform: current_transform,
                                    });
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("Failed to include {}: {e}", include_path.display()),
                }
            }
            Directive::ReverseOrientation
            | Directive::ObjectBegin(_)
            | Directive::ObjectEnd
            | Directive::ObjectInstance(_)
            | Directive::Attribute { .. }
            | Directive::Option { .. }
            | Directive::ColorSpace(_) => {
                // Known but not implemented
            }
            _ => {}
        }
    }
    parsed
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
            // Register each triangle as an area light
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
        }
    }
}
