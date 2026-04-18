//! Camera extraction from Blender `matrix_world`.
//!
//! Blender camera convention: local `-Z` is forward, local `+Y` is up, local `+X` is right.
//! The world-space transform is a 4x4 row-major matrix.

/// Compute the OptiX pinhole camera basis from a Blender `matrix_world` (4x4 row-major),
/// vertical FOV (radians), and image aspect (width/height).
///
/// Returns `(eye, cam_u, cam_v, cam_w)` where:
/// - `eye`: world-space camera origin
/// - `cam_u`: right vector scaled by `tan(fov_x/2)`
/// - `cam_v`: up vector scaled by `tan(fov_y/2)`
/// - `cam_w`: unit forward vector
pub fn compute_camera(
    transform: &[f32; 16],
    fov_y_rad: f32,
    aspect: f32,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    // row-major: element (i,j) = transform[i*4 + j]
    // Column 0 = right (+X), column 1 = up (+Y), column 2 = back (+Z), column 3 = translation.
    // Forward = -Z, up = +Y, right = +X.
    let right = [transform[0], transform[4], transform[8]];
    let up = [transform[1], transform[5], transform[9]];
    let forward = [-transform[2], -transform[6], -transform[10]];
    let eye = [transform[3], transform[7], transform[11]];

    let right = normalize(right);
    let up = normalize(up);
    let forward = normalize(forward);

    let half_h = (fov_y_rad * 0.5).tan();
    let half_w = half_h * aspect;

    let cam_u = [right[0] * half_w, right[1] * half_w, right[2] * half_w];
    let cam_v = [up[0] * half_h, up[1] * half_h, up[2] * half_h];
    (eye, cam_u, cam_v, forward)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-20 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}
