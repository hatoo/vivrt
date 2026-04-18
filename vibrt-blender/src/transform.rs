//! 3x4 row-major affine transform helpers.

#![allow(dead_code)]

pub fn identity() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Convert a 4x4 row-major matrix (16 floats) to a 3x4 row-major affine transform.
pub fn from_4x4_row_major(m: &[f32; 16]) -> [f32; 12] {
    [
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
    ]
}

pub fn invert(t: &[f32; 12]) -> [f32; 12] {
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
    let r00 = c00 * inv_det;
    let r01 = c10 * inv_det;
    let r02 = c20 * inv_det;
    let r10 = c01 * inv_det;
    let r11 = c11 * inv_det;
    let r12 = c21 * inv_det;
    let r20 = c02 * inv_det;
    let r21 = c12 * inv_det;
    let r22 = c22 * inv_det;
    let tx = -(r00 * t[3] + r01 * t[7] + r02 * t[11]);
    let ty = -(r10 * t[3] + r11 * t[7] + r12 * t[11]);
    let tz = -(r20 * t[3] + r21 * t[7] + r22 * t[11]);
    [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
}

/// Extract the 3x3 rotation (row-major 9 floats) from a 3x4 transform.
pub fn rotation_3x3(t: &[f32; 12]) -> [f32; 9] {
    [t[0], t[1], t[2], t[4], t[5], t[6], t[8], t[9], t[10]]
}

/// Transform a point by a 3x4 affine.
pub fn transform_point(t: &[f32; 12], p: [f32; 3]) -> [f32; 3] {
    [
        t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[3],
        t[4] * p[0] + t[5] * p[1] + t[6] * p[2] + t[7],
        t[8] * p[0] + t[9] * p[1] + t[10] * p[2] + t[11],
    ]
}

/// Transform a direction (no translation) by a 3x4 affine.
pub fn transform_dir(t: &[f32; 12], v: [f32; 3]) -> [f32; 3] {
    [
        t[0] * v[0] + t[1] * v[1] + t[2] * v[2],
        t[4] * v[0] + t[5] * v[1] + t[6] * v[2],
        t[8] * v[0] + t[9] * v[1] + t[10] * v[2],
    ]
}
