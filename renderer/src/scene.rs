//! PBRT scene parsing and representation.

use crate::gpu_types::*;
use pbrt_parser::{self, Directive, ParamType, ParamValue};
use std::path::Path;

pub struct SceneMaterial {
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub eta: f32,
    pub emission: [f32; 3],
    pub has_checkerboard: bool,
    pub checker_scale_u: f32,
    pub checker_scale_v: f32,
    pub checker_color1: [f32; 3],
    pub checker_color2: [f32; 3],
}

impl Default for SceneMaterial {
    fn default() -> Self {
        Self {
            material_type: MAT_DIFFUSE,
            albedo: [0.5, 0.5, 0.5],
            eta: 1.5,
            emission: [0.0, 0.0, 0.0],
            has_checkerboard: false,
            checker_scale_u: 1.0,
            checker_scale_v: 1.0,
            checker_color1: [1.0, 1.0, 1.0],
            checker_color2: [0.0, 0.0, 0.0],
        }
    }
}

impl Clone for SceneMaterial {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

pub enum SceneShape {
    Sphere { radius: f32 },
    TriangleMesh {
        vertices: Vec<f32>,
        indices: Vec<i32>,
        texcoords: Vec<f32>,
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
    pub objects: Vec<SceneObject>,
    pub filename: String,
}

// ---- Parameter helpers ----

fn get_param_floats<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [f64]> {
    params.iter().find(|p| p.name == name).and_then(|p| match &p.value {
        ParamValue::Floats(v) => Some(v.as_slice()),
        _ => None,
    })
}

fn get_param_float(params: &[pbrt_parser::Param], name: &str) -> Option<f32> {
    get_param_floats(params, name).and_then(|v| v.first().map(|x| *x as f32))
}

fn get_param_string<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a str> {
    params.iter().find(|p| p.name == name).and_then(|p| match &p.value {
        ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
        _ => None,
    })
}

fn get_param_ints<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [i64]> {
    params.iter().find(|p| p.name == name).and_then(|p| match &p.value {
        ParamValue::Ints(v) => Some(v.as_slice()),
        _ => None,
    })
}

fn get_param_rgb(params: &[pbrt_parser::Param], name: &str) -> Option<[f32; 3]> {
    params.iter().find(|p| p.name == name).and_then(|p| match &p.value {
        ParamValue::Floats(v) if v.len() >= 3 => Some([v[0] as f32, v[1] as f32, v[2] as f32]),
        _ => None,
    })
}

fn get_param_texture_ref<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|p| p.name == name && p.ty == ParamType::Texture)
        .and_then(|p| match &p.value {
            ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
            _ => None,
        })
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

// ---- Transform helpers ----

pub fn identity_transform() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

fn mul_transform(a: &[f32; 12], b: &[f32; 12]) -> [f32; 12] {
    let mut r = [0.0f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += a[i * 4 + k] * b[k * 4 + j];
            }
            if j == 3 {
                sum += a[i * 4 + 3];
            }
            r[i * 4 + j] = sum;
        }
    }
    r
}

fn translate_matrix(tx: f32, ty: f32, tz: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz]
}

fn scale_matrix(sx: f32, sy: f32, sz: f32) -> [f32; 12] {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0]
}

fn rotate_matrix(angle_deg: f32, ax: f32, ay: f32, az: f32) -> [f32; 12] {
    let angle = angle_deg.to_radians();
    let c = angle.cos();
    let s = angle.sin();
    let len = (ax * ax + ay * ay + az * az).sqrt();
    let (x, y, z) = (ax / len, ay / len, az / len);
    let t = 1.0 - c;
    [
        t * x * x + c,     t * x * y - s * z, t * x * z + s * y, 0.0,
        t * x * y + s * z, t * y * y + c,     t * y * z - s * x, 0.0,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c,     0.0,
    ]
}

pub fn transform_vertices(verts: &[f32], t: &[f32; 12]) -> Vec<f32> {
    let mut result = Vec::with_capacity(verts.len());
    for i in (0..verts.len()).step_by(3) {
        let x = verts[i];
        let y = verts[i + 1];
        let z = verts[i + 2];
        result.push(t[0] * x + t[1] * y + t[2] * z + t[3]);
        result.push(t[4] * x + t[5] * y + t[6] * z + t[7]);
        result.push(t[8] * x + t[9] * y + t[10] * z + t[11]);
    }
    result
}

// ---- Shape parsing helper ----

fn parse_shape(ty: &str, params: &[pbrt_parser::Param]) -> Option<SceneShape> {
    match ty {
        "sphere" => {
            let radius = get_param_float(params, "radius").unwrap_or(1.0);
            Some(SceneShape::Sphere { radius })
        }
        "trianglemesh" | "loopsubdiv" => {
            let verts: Vec<f32> = get_param_floats(params, "P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let indices: Vec<i32> = get_param_ints(params, "indices")
                .map(|v| v.iter().map(|x| *x as i32).collect())
                .unwrap_or_default();
            let texcoords: Vec<f32> = get_param_floats(params, "uv")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            Some(SceneShape::TriangleMesh { vertices: verts, indices, texcoords })
        }
        "bilinearmesh" => {
            let verts: Vec<f32> = get_param_floats(params, "P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let texcoords: Vec<f32> = if get_param_floats(params, "uv").is_some() {
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            } else {
                Vec::new()
            };
            let indices = vec![0, 1, 3, 0, 3, 2];
            Some(SceneShape::TriangleMesh { vertices: verts, indices, texcoords })
        }
        _ => {
            eprintln!("Unsupported shape type: {ty}");
            None
        }
    }
}

// ---- Main scene parser ----

pub fn parse_scene(input: &str, scene_dir: &Path) -> ParsedScene {
    let scene = pbrt_parser::parse(input).expect("Failed to parse PBRT scene");

    let mut parsed = ParsedScene {
        width: 400, height: 400, fov: 90.0,
        cam_eye: [0.0, 0.0, 0.0],
        cam_look: [0.0, 0.0, -1.0],
        cam_up: [0.0, 1.0, 0.0],
        spp: 16, max_depth: 5,
        ambient_light: [0.0; 3],
        distant_lights: Vec::new(),
        sphere_lights: Vec::new(),
        objects: Vec::new(),
        filename: "output.png".to_string(),
    };

    struct CheckerTex { scale_u: f32, scale_v: f32, color1: [f32; 3], color2: [f32; 3] }
    let mut textures = std::collections::HashMap::<String, CheckerTex>::new();
    let mut current_material = SceneMaterial::default();
    let mut current_transform = identity_transform();
    let mut transform_stack: Vec<([f32; 12], SceneMaterial)> = Vec::new();

    for directive in &scene.directives {
        match directive {
            Directive::Film { params, .. } => {
                if let Some(v) = get_param_ints(params, "xresolution") { parsed.width = v[0] as u32; }
                if let Some(v) = get_param_ints(params, "yresolution") { parsed.height = v[0] as u32; }
                if let Some(s) = get_param_string(params, "filename") { parsed.filename = s.to_string(); }
            }
            Directive::Camera { params, .. } => {
                if let Some(f) = get_param_float(params, "fov") { parsed.fov = f; }
            }
            Directive::LookAt { eye, look, up } => {
                parsed.cam_eye = [eye[0] as f32, eye[1] as f32, eye[2] as f32];
                parsed.cam_look = [look[0] as f32, look[1] as f32, look[2] as f32];
                parsed.cam_up = [up[0] as f32, up[1] as f32, up[2] as f32];
            }
            Directive::Sampler { params, .. } => {
                if let Some(v) = get_param_ints(params, "pixelsamples") { parsed.spp = v[0] as u32; }
            }
            Directive::Integrator { params, .. } => {
                if let Some(v) = get_param_ints(params, "maxdepth") { parsed.max_depth = v[0] as u32; }
            }
            Directive::WorldBegin => { current_transform = identity_transform(); }
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
                let t = translate_matrix(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = mul_transform(&current_transform, &t);
            }
            Directive::Scale { v } => {
                let s = scale_matrix(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = mul_transform(&current_transform, &s);
            }
            Directive::Rotate { angle, axis } => {
                let r = rotate_matrix(*angle as f32, axis[0] as f32, axis[1] as f32, axis[2] as f32);
                current_transform = mul_transform(&current_transform, &r);
            }
            Directive::Identity => { current_transform = identity_transform(); }
            Directive::LightSource { ty, params } => {
                match ty.as_str() {
                    "infinite" => {
                        if let Some(c) = get_param_rgb(params, "L") { parsed.ambient_light = c; }
                    }
                    "distant" => {
                        let from = get_param_floats(params, "from")
                            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                            .unwrap_or([0.0, 0.0, 1.0]);
                        let to = get_param_floats(params, "to")
                            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                            .unwrap_or([0.0, 0.0, 0.0]);
                        let dir = {
                            let dx = from[0] - to[0]; let dy = from[1] - to[1]; let dz = from[2] - to[2];
                            let len = (dx * dx + dy * dy + dz * dz).sqrt();
                            [dx / len, dy / len, dz / len]
                        };
                        let mut emission = get_param_rgb(params, "L").unwrap_or([1.0, 1.0, 1.0]);
                        if let Some(p) = params.iter().find(|p| p.name == "L" && p.ty == ParamType::Blackbody) {
                            if let ParamValue::Floats(v) = &p.value {
                                if let Some(&k) = v.first() { emission = blackbody_to_rgb(k as f32); }
                            }
                        }
                        let scale = get_param_float(params, "scale").unwrap_or(1.0);
                        emission[0] *= scale; emission[1] *= scale; emission[2] *= scale;
                        parsed.distant_lights.push(DistantLight { direction: dir, emission });
                    }
                    _ => eprintln!("Unsupported light type: {ty}"),
                }
            }
            Directive::Material { ty, params } => {
                current_material = SceneMaterial::default();
                match ty.as_str() {
                    "diffuse" => {
                        current_material.material_type = MAT_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") { current_material.albedo = c; }
                        if let Some(tex_name) = get_param_texture_ref(params, "reflectance") {
                            if let Some(tex) = textures.get(tex_name) {
                                current_material.has_checkerboard = true;
                                current_material.checker_scale_u = tex.scale_u;
                                current_material.checker_scale_v = tex.scale_v;
                                current_material.checker_color1 = tex.color1;
                                current_material.checker_color2 = tex.color2;
                            }
                        }
                    }
                    "coateddiffuse" | "coatedconductor" | "conductor" => {
                        current_material.material_type = MAT_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") { current_material.albedo = c; }
                    }
                    "dielectric" | "thindielectric" => {
                        current_material.material_type = MAT_DIELECTRIC;
                        current_material.eta = get_param_float(params, "eta").unwrap_or(1.5);
                    }
                    _ => eprintln!("Unsupported material type: {ty}"),
                }
            }
            Directive::Texture { name, class, params, .. } => {
                if class == "checkerboard" {
                    textures.insert(name.clone(), CheckerTex {
                        scale_u: get_param_float(params, "uscale").unwrap_or(1.0),
                        scale_v: get_param_float(params, "vscale").unwrap_or(1.0),
                        color1: get_param_rgb(params, "tex1").unwrap_or([1.0, 1.0, 1.0]),
                        color2: get_param_rgb(params, "tex2").unwrap_or([0.0, 0.0, 0.0]),
                    });
                }
            }
            Directive::Shape { ty, params } => {
                if let Some(shape) = parse_shape(ty, params) {
                    register_area_light(&shape, &current_material, &current_transform, &mut parsed);
                    parsed.objects.push(SceneObject {
                        shape, material: current_material.clone(), transform: current_transform,
                    });
                }
            }
            Directive::AreaLightSource { ty, params } => {
                if ty == "diffuse" {
                    if let Some(c) = get_param_rgb(params, "L") { current_material.emission = c; }
                }
            }
            Directive::Include(path) => {
                let include_path = scene_dir.join(path);
                match std::fs::read_to_string(&include_path) {
                    Ok(content) => {
                        let included = pbrt_parser::parse(&content)
                            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", include_path.display()));
                        for inc_directive in &included.directives {
                            if let Directive::Shape { ty, params } = inc_directive {
                                if let Some(shape) = parse_shape(ty, params) {
                                    parsed.objects.push(SceneObject {
                                        shape, material: current_material.clone(), transform: current_transform,
                                    });
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("Failed to include {}: {e}", include_path.display()),
                }
            }
            _ => {}
        }
    }
    parsed
}

fn register_area_light(
    shape: &SceneShape, mat: &SceneMaterial, transform: &[f32; 12], scene: &mut ParsedScene,
) {
    let em = mat.emission;
    if em[0] > 0.0 || em[1] > 0.0 || em[2] > 0.0 {
        if let SceneShape::Sphere { radius } = shape {
            let center = [transform[3], transform[7], transform[11]];
            scene.sphere_lights.push(SphereLight {
                center, radius: *radius, emission: em, _pad: 0.0,
            });
        }
    }
}
