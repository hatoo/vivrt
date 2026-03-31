//! Fast binary PLY mesh loader.

use std::io::Read;
use std::path::Path;

pub struct PlyMesh {
    pub vertices: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<i32>,
}

pub fn load(path: &Path) -> Option<PlyMesh> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open PLY file {}: {e}", path.display()));

    let buf_file = std::io::BufReader::new(file);
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz")
        || path.to_string_lossy().contains(".ply.gz")
    {
        Box::new(flate2::read::GzDecoder::new(buf_file))
    } else {
        Box::new(buf_file)
    };
    let mut reader = std::io::BufReader::new(reader);

    // Parse header
    let mut num_vertices = 0usize;
    let mut num_faces = 0usize;
    let mut is_binary_le = false;

    #[derive(Clone, Copy, PartialEq)]
    enum PropRole {
        X,
        Y,
        Z,
        Nx,
        Ny,
        Nz,
        U,
        V,
        Other,
    }

    let mut vertex_props: Vec<(PropType, PropRole)> = Vec::new();
    let mut face_index_type = PropType::Int;
    let mut face_count_type = PropType::Uchar;
    let mut in_vertex_element = false;
    let mut in_face_element = false;

    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .expect("Failed to read PLY header");
        let line = line.trim();
        if line == "end_header" {
            break;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                is_binary_le = parts.get(1) == Some(&"binary_little_endian");
            }
            "element" => {
                in_vertex_element = false;
                in_face_element = false;
                if parts.get(1) == Some(&"vertex") {
                    num_vertices = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                    in_vertex_element = true;
                } else if parts.get(1) == Some(&"face") {
                    num_faces = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                    in_face_element = true;
                }
            }
            "property" if in_vertex_element && parts.len() >= 3 => {
                let ptype = parse_prop_type(parts[1]);
                let role = match parts[2] {
                    "x" => PropRole::X,
                    "y" => PropRole::Y,
                    "z" => PropRole::Z,
                    "nx" => PropRole::Nx,
                    "ny" => PropRole::Ny,
                    "nz" => PropRole::Nz,
                    "u" | "s" | "texture_u" => PropRole::U,
                    "v" | "t" | "texture_v" => PropRole::V,
                    _ => PropRole::Other,
                };
                vertex_props.push((ptype, role));
            }
            "property" if in_face_element && parts.get(1) == Some(&"list") && parts.len() >= 5 => {
                face_count_type = parse_prop_type(parts[2]);
                face_index_type = parse_prop_type(parts[3]);
            }
            _ => {}
        }
    }

    if !is_binary_le {
        eprintln!(
            "Only binary_little_endian PLY is supported: {}",
            path.display()
        );
        return None;
    }

    let has_normals = vertex_props.iter().any(|(_, r)| *r == PropRole::Nx);
    let has_uvs = vertex_props.iter().any(|(_, r)| *r == PropRole::U);

    let mut vertices = Vec::with_capacity(num_vertices * 3);
    let mut normals = if has_normals {
        Vec::with_capacity(num_vertices * 3)
    } else {
        Vec::new()
    };
    let mut texcoords = if has_uvs {
        Vec::with_capacity(num_vertices * 2)
    } else {
        Vec::new()
    };

    for _ in 0..num_vertices {
        for &(ptype, role) in &vertex_props {
            let val = read_prop_f32(&mut reader, ptype);
            match role {
                PropRole::X | PropRole::Y | PropRole::Z => vertices.push(val),
                PropRole::Nx | PropRole::Ny | PropRole::Nz => normals.push(val),
                PropRole::U | PropRole::V => texcoords.push(val),
                PropRole::Other => {}
            }
        }
    }

    let mut indices = Vec::with_capacity(num_faces * 3);
    for _ in 0..num_faces {
        let count = read_prop_f32(&mut reader, face_count_type) as usize;
        let mut face_indices = Vec::with_capacity(count);
        for _ in 0..count {
            face_indices.push(read_prop_f32(&mut reader, face_index_type) as i32);
        }
        for i in 1..count - 1 {
            indices.push(face_indices[0]);
            indices.push(face_indices[i]);
            indices.push(face_indices[i + 1]);
        }
    }

    println!(
        "Loaded PLY: {} vertices, {} triangles from {}",
        vertices.len() / 3,
        indices.len() / 3,
        path.display()
    );

    Some(PlyMesh {
        vertices,
        normals,
        texcoords,
        indices,
    })
}

fn parse_prop_type(s: &str) -> PropType {
    match s {
        "float" | "float32" => PropType::Float,
        "double" | "float64" => PropType::Double,
        "uchar" | "uint8" => PropType::Uchar,
        "int" | "int32" => PropType::Int,
        "uint" | "uint32" => PropType::UInt,
        "short" | "int16" => PropType::Short,
        "ushort" | "uint16" => PropType::UShort,
        _ => PropType::Float,
    }
}

#[derive(Clone, Copy)]
enum PropType {
    Float,
    Double,
    Uchar,
    Int,
    UInt,
    Short,
    UShort,
}

fn read_prop_f32(r: &mut impl Read, ptype: PropType) -> f32 {
    match ptype {
        PropType::Float => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            f32::from_le_bytes(b)
        }
        PropType::Double => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b).unwrap();
            f64::from_le_bytes(b) as f32
        }
        PropType::Uchar => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).unwrap();
            b[0] as f32
        }
        PropType::Int => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            i32::from_le_bytes(b) as f32
        }
        PropType::UInt => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            u32::from_le_bytes(b) as f32
        }
        PropType::Short => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).unwrap();
            i16::from_le_bytes(b) as f32
        }
        PropType::UShort => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).unwrap();
            u16::from_le_bytes(b) as f32
        }
    }
}
