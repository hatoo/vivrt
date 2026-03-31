//! PBRT scene parsing and representation.

use crate::gpu_types::*;
use pbrt_parser::{self, Directive, ParamType, ParamValue};
use std::io::Read;
use std::path::Path;

pub struct SceneMaterial {
    pub material_type: i32,
    pub albedo: [f32; 3],
    pub eta: f32,
    pub roughness: f32,
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
            roughness: 1.0,
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

// ---- Parameter helpers ----

fn get_param_floats<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [f64]> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Floats(v) => Some(v.as_slice()),
            _ => None,
        })
}

fn get_param_float(params: &[pbrt_parser::Param], name: &str) -> Option<f32> {
    get_param_floats(params, name).and_then(|v| v.first().map(|x| *x as f32))
}

fn get_param_string<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Strings(v) => v.first().map(|s| s.as_str()),
            _ => None,
        })
}

fn get_param_ints<'a>(params: &'a [pbrt_parser::Param], name: &str) -> Option<&'a [i64]> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
            ParamValue::Ints(v) => Some(v.as_slice()),
            _ => None,
        })
}

fn get_param_rgb(params: &[pbrt_parser::Param], name: &str) -> Option<[f32; 3]> {
    params
        .iter()
        .find(|p| p.name == name)
        .and_then(|p| match &p.value {
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
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        0.0,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        0.0,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
        0.0,
    ]
}

fn cross3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3f(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
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

// ---- Loop subdivision ----

fn loop_subdivide(verts: &[f32], indices: &[i32], levels: u32) -> (Vec<f32>, Vec<i32>) {
    use std::collections::HashMap;

    let mut positions: Vec<[f32; 3]> = verts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let mut tris: Vec<[i32; 3]> = indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    for _ in 0..levels {
        let num_verts = positions.len();

        // Build adjacency: for each edge, find the two opposite vertices
        // edge_key = (min_vertex, max_vertex) -> (opposite_vertex_1, opposite_vertex_2)
        let mut edge_faces: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (fi, tri) in tris.iter().enumerate() {
            for e in 0..3 {
                let a = tri[e];
                let b = tri[(e + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edge_faces.entry(key).or_default().push(fi);
            }
        }

        // Build vertex neighbors
        let mut neighbors: Vec<Vec<i32>> = vec![Vec::new(); num_verts];
        for &(a, b) in edge_faces.keys() {
            if !neighbors[a as usize].contains(&b) {
                neighbors[a as usize].push(b);
            }
            if !neighbors[b as usize].contains(&a) {
                neighbors[b as usize].push(a);
            }
        }

        // Create edge vertices
        let mut edge_vertex_map: HashMap<(i32, i32), i32> = HashMap::new();
        let mut new_positions = positions.clone();

        for (&(a, b), faces) in &edge_faces {
            let pa = positions[a as usize];
            let pb = positions[b as usize];

            let new_pos = if faces.len() == 2 {
                // Interior edge: find opposite vertices
                let tri0 = &tris[faces[0]];
                let tri1 = &tris[faces[1]];
                let c = tri0.iter().find(|&&v| v != a && v != b).unwrap();
                let d = tri1.iter().find(|&&v| v != a && v != b).unwrap();
                let pc = positions[*c as usize];
                let pd = positions[*d as usize];
                // Loop rule: 3/8 * (A + B) + 1/8 * (C + D)
                [
                    3.0 / 8.0 * (pa[0] + pb[0]) + 1.0 / 8.0 * (pc[0] + pd[0]),
                    3.0 / 8.0 * (pa[1] + pb[1]) + 1.0 / 8.0 * (pc[1] + pd[1]),
                    3.0 / 8.0 * (pa[2] + pb[2]) + 1.0 / 8.0 * (pc[2] + pd[2]),
                ]
            } else {
                // Boundary edge: midpoint
                [
                    0.5 * (pa[0] + pb[0]),
                    0.5 * (pa[1] + pb[1]),
                    0.5 * (pa[2] + pb[2]),
                ]
            };

            let idx = new_positions.len() as i32;
            new_positions.push(new_pos);
            edge_vertex_map.insert((a, b), idx);
        }

        // Update existing vertex positions
        for i in 0..num_verts {
            let n = neighbors[i].len();
            if n < 2 {
                continue;
            }

            let beta = if n == 3 {
                3.0 / 16.0
            } else {
                3.0 / (8.0 * n as f32)
            };

            let p = positions[i];
            let mut sum = [0.0f32; 3];
            for &nb in &neighbors[i] {
                let pn = positions[nb as usize];
                sum[0] += pn[0];
                sum[1] += pn[1];
                sum[2] += pn[2];
            }

            new_positions[i] = [
                (1.0 - n as f32 * beta) * p[0] + beta * sum[0],
                (1.0 - n as f32 * beta) * p[1] + beta * sum[1],
                (1.0 - n as f32 * beta) * p[2] + beta * sum[2],
            ];
        }

        // Create new triangles: each old triangle splits into 4
        let mut new_tris = Vec::with_capacity(tris.len() * 4);
        for tri in &tris {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];

            let edge_vert = |a: i32, b: i32| -> i32 {
                let key = if a < b { (a, b) } else { (b, a) };
                edge_vertex_map[&key]
            };

            let m01 = edge_vert(v0, v1);
            let m12 = edge_vert(v1, v2);
            let m20 = edge_vert(v2, v0);

            new_tris.push([v0, m01, m20]);
            new_tris.push([v1, m12, m01]);
            new_tris.push([v2, m20, m12]);
            new_tris.push([m01, m12, m20]);
        }

        positions = new_positions;
        tris = new_tris;
    }

    let flat_verts: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    let flat_indices: Vec<i32> = tris.iter().flat_map(|t| t.iter().copied()).collect();
    (flat_verts, flat_indices)
}

/// Compute area-weighted per-vertex normals from triangle mesh.
fn compute_smooth_normals(verts: &[f32], indices: &[i32]) -> Vec<f32> {
    let num_verts = verts.len() / 3;
    let mut normals = vec![0.0f32; num_verts * 3];

    for tri in indices.chunks(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = [verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]];
        let v1 = [verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]];
        let v2 = [verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]];

        // Cross product (unnormalized = area-weighted)
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        for &vi in &[i0, i1, i2] {
            normals[vi * 3] += n[0];
            normals[vi * 3 + 1] += n[1];
            normals[vi * 3 + 2] += n[2];
        }
    }

    // Normalize
    for i in 0..num_verts {
        let nx = normals[i * 3];
        let ny = normals[i * 3 + 1];
        let nz = normals[i * 3 + 2];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 0.0 {
            normals[i * 3] /= len;
            normals[i * 3 + 1] /= len;
            normals[i * 3 + 2] /= len;
        }
    }

    normals
}

// ---- PLY mesh loading ----

fn load_ply_mesh(path: &Path) -> Option<SceneShape> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open PLY file {}: {e}", path.display()));

    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz")
        || path.to_string_lossy().contains(".ply.gz")
    {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let mut buf_reader = std::io::BufReader::new(reader);
    let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
    let ply = parser
        .read_ply(&mut buf_reader)
        .unwrap_or_else(|e| panic!("Failed to parse PLY file {}: {e}", path.display()));

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    if let Some(verts) = ply.payload.get("vertex") {
        for v in verts {
            let x = prop_float(v, "x");
            let y = prop_float(v, "y");
            let z = prop_float(v, "z");
            vertices.push(x);
            vertices.push(y);
            vertices.push(z);

            // Optional normals
            if v.contains_key("nx") {
                let nx = prop_float(v, "nx");
                let ny = prop_float(v, "ny");
                let nz = prop_float(v, "nz");
                normals.push(nx);
                normals.push(ny);
                normals.push(nz);
            }
        }
    }

    if let Some(faces) = ply.payload.get("face") {
        for f in faces {
            if let Some(ply_rs::ply::Property::ListInt(ref idx)) = f.get("vertex_indices") {
                // Triangulate: fan from first vertex
                for i in 1..idx.len() - 1 {
                    indices.push(idx[0]);
                    indices.push(idx[i as usize] as i32);
                    indices.push(idx[i as usize + 1] as i32);
                }
            } else if let Some(ply_rs::ply::Property::ListUInt(ref idx)) = f.get("vertex_indices") {
                for i in 1..idx.len() - 1 {
                    indices.push(idx[0] as i32);
                    indices.push(idx[i as usize] as i32);
                    indices.push(idx[i as usize + 1] as i32);
                }
            }
        }
    }

    println!(
        "Loaded PLY: {} vertices, {} triangles from {}",
        vertices.len() / 3,
        indices.len() / 3,
        path.display()
    );

    Some(SceneShape::TriangleMesh {
        vertices,
        indices,
        texcoords: Vec::new(),
        normals,
    })
}

fn prop_float(element: &ply_rs::ply::DefaultElement, key: &str) -> f32 {
    match element.get(key) {
        Some(ply_rs::ply::Property::Float(v)) => *v,
        Some(ply_rs::ply::Property::Double(v)) => *v as f32,
        _ => 0.0,
    }
}

// ---- Shape parsing helper ----

fn parse_shape(ty: &str, params: &[pbrt_parser::Param], scene_dir: &Path) -> Option<SceneShape> {
    match ty {
        "sphere" => {
            let radius = get_param_float(params, "radius").unwrap_or(1.0);
            Some(SceneShape::Sphere { radius })
        }
        "loopsubdiv" => {
            let verts: Vec<f32> = get_param_floats(params, "P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let indices: Vec<i32> = get_param_ints(params, "indices")
                .map(|v| v.iter().map(|x| *x as i32).collect())
                .unwrap_or_default();
            let levels = get_param_ints(params, "levels")
                .and_then(|v| v.first().map(|x| *x as u32))
                .unwrap_or(3);
            let (subdivided_verts, subdivided_indices) = loop_subdivide(&verts, &indices, levels);
            let normals = compute_smooth_normals(&subdivided_verts, &subdivided_indices);
            Some(SceneShape::TriangleMesh {
                vertices: subdivided_verts,
                indices: subdivided_indices,
                texcoords: Vec::new(),
                normals,
            })
        }
        "trianglemesh" => {
            let verts: Vec<f32> = get_param_floats(params, "P")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            let indices: Vec<i32> = get_param_ints(params, "indices")
                .map(|v| v.iter().map(|x| *x as i32).collect())
                .unwrap_or_default();
            let texcoords: Vec<f32> = get_param_floats(params, "uv")
                .map(|v| v.iter().map(|x| *x as f32).collect())
                .unwrap_or_default();
            Some(SceneShape::TriangleMesh {
                vertices: verts,
                indices,
                texcoords,
                normals: Vec::new(),
            })
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
            Some(SceneShape::TriangleMesh {
                vertices: verts,
                indices,
                texcoords,
                normals: Vec::new(),
            })
        }
        "plymesh" => {
            let filename = get_param_string(params, "filename")?;
            load_ply_mesh(&scene_dir.join(filename))
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

    struct CheckerTex {
        scale_u: f32,
        scale_v: f32,
        color1: [f32; 3],
        color2: [f32; 3],
    }
    let mut textures = std::collections::HashMap::<String, CheckerTex>::new();
    let mut current_material = SceneMaterial::default();
    let mut current_transform = identity_transform();
    let mut transform_stack: Vec<([f32; 12], SceneMaterial)> = Vec::new();
    let mut _in_world = false;

    for directive in &scene.directives {
        match directive {
            Directive::Film { params, .. } => {
                if let Some(v) = get_param_ints(params, "xresolution") {
                    parsed.width = v[0] as u32;
                }
                if let Some(v) = get_param_ints(params, "yresolution") {
                    parsed.height = v[0] as u32;
                }
                if let Some(s) = get_param_string(params, "filename") {
                    parsed.filename = s.to_string();
                }
            }
            Directive::Camera { params, .. } => {
                if let Some(f) = get_param_float(params, "fov") {
                    parsed.fov = f;
                }
            }
            Directive::LookAt { eye, look, up } => {
                let mut e = [eye[0] as f32, eye[1] as f32, eye[2] as f32];
                let mut l = [look[0] as f32, look[1] as f32, look[2] as f32];
                let mut u = [up[0] as f32, up[1] as f32, up[2] as f32];

                // Detect handedness flip from pre-LookAt Scale (e.g. Scale -1 1 1)
                if current_transform != identity_transform() {
                    let t = &current_transform;
                    let det = t[0] * (t[5] * t[10] - t[6] * t[9])
                        - t[1] * (t[4] * t[10] - t[6] * t[8])
                        + t[2] * (t[4] * t[9] - t[5] * t[8]);
                    if det < 0.0 {
                        parsed.cam_flip_x = true;
                    }
                    current_transform = identity_transform();
                }

                parsed.cam_eye = e;
                parsed.cam_look = l;
                parsed.cam_up = u;
            }
            Directive::Sampler { params, .. } => {
                if let Some(v) = get_param_ints(params, "pixelsamples") {
                    parsed.spp = v[0] as u32;
                }
            }
            Directive::Integrator { params, .. } => {
                if let Some(v) = get_param_ints(params, "maxdepth") {
                    parsed.max_depth = v[0] as u32;
                }
            }
            Directive::WorldBegin => {
                // In PBRT, CTM = LookAt * R (post-multiplied). This means
                // the world is further transformed by R before the camera
                // sees it. Equivalently, eye/look/up are transformed by
                // the inverse of R. For a rotation, inverse = transpose.
                if current_transform != identity_transform() {
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
                current_transform = identity_transform();
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
                let t = translate_matrix(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = mul_transform(&current_transform, &t);
            }
            Directive::Scale { v } => {
                let s = scale_matrix(v[0] as f32, v[1] as f32, v[2] as f32);
                current_transform = mul_transform(&current_transform, &s);
            }
            Directive::Rotate { angle, axis } => {
                let r = rotate_matrix(
                    *angle as f32,
                    axis[0] as f32,
                    axis[1] as f32,
                    axis[2] as f32,
                );
                current_transform = mul_transform(&current_transform, &r);
            }
            Directive::Identity => {
                current_transform = identity_transform();
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
                current_transform = mul_transform(&current_transform, &t);
            }
            Directive::LightSource { ty, params } => match ty.as_str() {
                "infinite" => {
                    if let Some(c) = get_param_rgb(params, "L") {
                        parsed.ambient_light = c;
                    }
                }
                "distant" => {
                    let from = get_param_floats(params, "from")
                        .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                        .unwrap_or([0.0, 0.0, 1.0]);
                    let to = get_param_floats(params, "to")
                        .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                        .unwrap_or([0.0, 0.0, 0.0]);
                    let dir = {
                        let dx = from[0] - to[0];
                        let dy = from[1] - to[1];
                        let dz = from[2] - to[2];
                        let len = (dx * dx + dy * dy + dz * dz).sqrt();
                        [dx / len, dy / len, dz / len]
                    };
                    let mut emission = get_param_rgb(params, "L").unwrap_or([1.0, 1.0, 1.0]);
                    if let Some(p) = params
                        .iter()
                        .find(|p| p.name == "L" && p.ty == ParamType::Blackbody)
                    {
                        if let ParamValue::Floats(v) = &p.value {
                            if let Some(&k) = v.first() {
                                emission = blackbody_to_rgb(k as f32);
                            }
                        }
                    }
                    let scale = get_param_float(params, "scale").unwrap_or(1.0);
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
                current_material = SceneMaterial::default();
                match ty.as_str() {
                    "diffuse" => {
                        current_material.material_type = MAT_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") {
                            current_material.albedo = c;
                        }
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
                    "coateddiffuse" => {
                        current_material.material_type = MAT_COATED_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") {
                            current_material.albedo = c;
                        }
                        current_material.roughness =
                            get_param_float(params, "roughness").unwrap_or(0.0);
                    }
                    "coatedconductor" | "conductor" => {
                        current_material.material_type = MAT_COATED_DIFFUSE;
                        if let Some(c) = get_param_rgb(params, "reflectance") {
                            current_material.albedo = c;
                        }
                        current_material.roughness =
                            get_param_float(params, "roughness").unwrap_or(0.0);
                    }
                    "dielectric" | "thindielectric" => {
                        current_material.material_type = MAT_DIELECTRIC;
                        current_material.eta = get_param_float(params, "eta").unwrap_or(1.5);
                    }
                    _ => eprintln!("Unsupported material type: {ty}"),
                }
            }
            Directive::Texture {
                name,
                class,
                params,
                ..
            } => {
                if class == "checkerboard" {
                    textures.insert(
                        name.clone(),
                        CheckerTex {
                            scale_u: get_param_float(params, "uscale").unwrap_or(1.0),
                            scale_v: get_param_float(params, "vscale").unwrap_or(1.0),
                            color1: get_param_rgb(params, "tex1").unwrap_or([1.0, 1.0, 1.0]),
                            color2: get_param_rgb(params, "tex2").unwrap_or([0.0, 0.0, 0.0]),
                        },
                    );
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
                    if let Some(c) = get_param_rgb(params, "L") {
                        current_material.emission = c;
                    }
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
            _ => {}
        }
    }
    parsed
}

fn register_area_light(
    shape: &SceneShape,
    mat: &SceneMaterial,
    transform: &[f32; 12],
    scene: &mut ParsedScene,
) {
    let em = mat.emission;
    if em[0] > 0.0 || em[1] > 0.0 || em[2] > 0.0 {
        if let SceneShape::Sphere { radius } = shape {
            let center = [transform[3], transform[7], transform[11]];
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
            let transformed = transform_vertices(vertices, transform);
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
