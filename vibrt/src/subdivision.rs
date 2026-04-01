//! Loop subdivision and smooth normal computation.

use std::collections::HashMap;

/// Perform Loop subdivision on a triangle mesh.
pub fn loop_subdivide(verts: &[f32], indices: &[i32], levels: u32) -> (Vec<f32>, Vec<i32>) {
    let mut positions: Vec<[f32; 3]> = verts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let mut tris: Vec<[i32; 3]> = indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    for _ in 0..levels {
        let num_verts = positions.len();

        let mut edge_faces: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (fi, tri) in tris.iter().enumerate() {
            for e in 0..3 {
                let a = tri[e];
                let b = tri[(e + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edge_faces.entry(key).or_default().push(fi);
            }
        }

        let mut neighbors: Vec<Vec<i32>> = vec![Vec::new(); num_verts];
        for &(a, b) in edge_faces.keys() {
            if !neighbors[a as usize].contains(&b) {
                neighbors[a as usize].push(b);
            }
            if !neighbors[b as usize].contains(&a) {
                neighbors[b as usize].push(a);
            }
        }

        let mut edge_vertex_map: HashMap<(i32, i32), i32> = HashMap::new();
        let mut new_positions = positions.clone();

        for (&(a, b), faces) in &edge_faces {
            let pa = positions[a as usize];
            let pb = positions[b as usize];

            let new_pos = if faces.len() == 2 {
                let tri0 = &tris[faces[0]];
                let tri1 = &tris[faces[1]];
                let c = tri0.iter().find(|&&v| v != a && v != b).unwrap();
                let d = tri1.iter().find(|&&v| v != a && v != b).unwrap();
                let pc = positions[*c as usize];
                let pd = positions[*d as usize];
                [
                    3.0 / 8.0 * (pa[0] + pb[0]) + 1.0 / 8.0 * (pc[0] + pd[0]),
                    3.0 / 8.0 * (pa[1] + pb[1]) + 1.0 / 8.0 * (pc[1] + pd[1]),
                    3.0 / 8.0 * (pa[2] + pb[2]) + 1.0 / 8.0 * (pc[2] + pd[2]),
                ]
            } else {
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
pub fn compute_smooth_normals(verts: &[f32], indices: &[i32]) -> Vec<f32> {
    let num_verts = verts.len() / 3;
    let mut normals = vec![0.0f32; num_verts * 3];

    for tri in indices.chunks(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = [verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]];
        let v1 = [verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]];
        let v2 = [verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]];

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

/// Generate an icosphere with the given radius and subdivision level.
/// Returns (vertices, normals, indices).
pub fn make_icosphere(radius: f32, subdivisions: u32) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

    let mut verts = vec![
        -1.0, t, 0.0, 1.0, t, 0.0, -1.0, -t, 0.0, 1.0, -t, 0.0, 0.0, -1.0, t, 0.0, 1.0, t, 0.0,
        -1.0, -t, 0.0, 1.0, -t, t, 0.0, -1.0, t, 0.0, 1.0, -t, 0.0, -1.0, -t, 0.0, 1.0,
    ];

    let mut indices: Vec<i32> = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];

    // Normalize initial vertices to unit sphere
    for i in (0..verts.len()).step_by(3) {
        let len = (verts[i] * verts[i] + verts[i + 1] * verts[i + 1] + verts[i + 2] * verts[i + 2])
            .sqrt();
        verts[i] /= len;
        verts[i + 1] /= len;
        verts[i + 2] /= len;
    }

    // Subdivide
    for _ in 0..subdivisions {
        let mut new_indices = Vec::new();
        let mut midpoint_cache = HashMap::new();

        let mut get_midpoint = |a: i32, b: i32, verts: &mut Vec<f32>| -> i32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = midpoint_cache.get(&key) {
                return idx;
            }
            let ai = a as usize * 3;
            let bi = b as usize * 3;
            let mx = (verts[ai] + verts[bi]) * 0.5;
            let my = (verts[ai + 1] + verts[bi + 1]) * 0.5;
            let mz = (verts[ai + 2] + verts[bi + 2]) * 0.5;
            let len = (mx * mx + my * my + mz * mz).sqrt();
            let idx = (verts.len() / 3) as i32;
            verts.push(mx / len);
            verts.push(my / len);
            verts.push(mz / len);
            midpoint_cache.insert(key, idx);
            idx
        };

        for tri in indices.chunks(3) {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = get_midpoint(a, b, &mut verts);
            let bc = get_midpoint(b, c, &mut verts);
            let ca = get_midpoint(c, a, &mut verts);
            new_indices.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
        }
        indices = new_indices;
    }

    // Normals = normalized positions (unit sphere)
    let normals = verts.clone();

    // Scale by radius
    for v in &mut verts {
        *v *= radius;
    }

    (verts, normals, indices)
}
