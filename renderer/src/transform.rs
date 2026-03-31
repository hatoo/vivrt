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

/// Determinant of the 3x3 rotation part.
pub fn det3x3(t: &[f32; 12]) -> f32 {
    t[0] * (t[5] * t[10] - t[6] * t[9]) - t[1] * (t[4] * t[10] - t[6] * t[8])
        + t[2] * (t[4] * t[9] - t[5] * t[8])
}
