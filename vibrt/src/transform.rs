//! 3x4 row-major affine transform helpers.

pub fn identity() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Multiply two 3x4 row-major transforms: result = a * b
pub fn mul(a: &[f32; 12], b: &[f32; 12]) -> [f32; 12] {
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

pub fn translate(tx: f32, ty: f32, tz: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz]
}

pub fn scale(sx: f32, sy: f32, sz: f32) -> [f32; 12] {
    [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0]
}

pub fn rotate(angle_deg: f32, ax: f32, ay: f32, az: f32) -> [f32; 12] {
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

/// Invert a 3x4 affine transform. Returns identity if singular.
pub fn invert(t: &[f32; 12]) -> [f32; 12] {
    // Cofactor matrix of 3x3 part
    let c00 = t[5] * t[10] - t[6] * t[9];
    let c01 = t[6] * t[8] - t[4] * t[10];
    let c02 = t[4] * t[9] - t[5] * t[8];
    let det = t[0] * c00 + t[1] * c01 + t[2] * c02;
    if det.abs() < 1e-20 {
        return identity();
    }
    let inv_det = 1.0 / det;
    let c10 = t[2] * t[9] - t[1] * t[10];
    let c11 = t[0] * t[10] - t[2] * t[8];
    let c12 = t[1] * t[8] - t[0] * t[9];
    let c20 = t[1] * t[6] - t[2] * t[5];
    let c21 = t[2] * t[4] - t[0] * t[6];
    let c22 = t[0] * t[5] - t[1] * t[4];
    // Inverse 3x3
    let r00 = c00 * inv_det;
    let r01 = c10 * inv_det;
    let r02 = c20 * inv_det;
    let r10 = c01 * inv_det;
    let r11 = c11 * inv_det;
    let r12 = c21 * inv_det;
    let r20 = c02 * inv_det;
    let r21 = c12 * inv_det;
    let r22 = c22 * inv_det;
    // Inverse translation
    let tx = -(r00 * t[3] + r01 * t[7] + r02 * t[11]);
    let ty = -(r10 * t[3] + r11 * t[7] + r12 * t[11]);
    let tz = -(r20 * t[3] + r21 * t[7] + r22 * t[11]);
    [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
}

/// Transform normals by the inverse-transpose of the 3x3 upper-left.
/// Uses cofactor matrix (equivalent to inverse-transpose up to scale).
pub fn transform_normals(normals: &[f32], t: &[f32; 12]) -> Vec<f32> {
    // Cofactor matrix of the 3x3 part (= inverse-transpose * det)
    let c00 = t[5] * t[10] - t[6] * t[9];
    let c01 = t[6] * t[8] - t[4] * t[10];
    let c02 = t[4] * t[9] - t[5] * t[8];
    let c10 = t[2] * t[9] - t[1] * t[10];
    let c11 = t[0] * t[10] - t[2] * t[8];
    let c12 = t[1] * t[8] - t[0] * t[9];
    let c20 = t[1] * t[6] - t[2] * t[5];
    let c21 = t[2] * t[4] - t[0] * t[6];
    let c22 = t[0] * t[5] - t[1] * t[4];

    let mut result = Vec::with_capacity(normals.len());
    for i in (0..normals.len()).step_by(3) {
        let x = normals[i];
        let y = normals[i + 1];
        let z = normals[i + 2];
        let nx = c00 * x + c10 * y + c20 * z;
        let ny = c01 * x + c11 * y + c21 * z;
        let nz = c02 * x + c12 * y + c22 * z;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-20 {
            result.push(nx / len);
            result.push(ny / len);
            result.push(nz / len);
        } else {
            result.push(0.0);
            result.push(0.0);
            result.push(1.0);
        }
    }
    result
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
