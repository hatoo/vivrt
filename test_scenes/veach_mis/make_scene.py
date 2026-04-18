"""Generate a Veach-style MIS demo scene for vibrt.

The scene contrasts four metallic plates of varying roughness (0.005 -> 0.35)
lit purely by an HDR environment map containing a small, intense sun disc on a
dim sky. This is the setup where envmap MIS earns its keep:

  * Smooth plates: BSDF sampling efficiently finds the sun via the narrow
    specular lobe, while envmap importance sampling rarely lands inside the
    BSDF cone. Envmap-only NEE is very noisy here.
  * Rough plates / diffuse floor: envmap importance sampling is efficient
    (the sun is the envmap's luminance peak), while BSDF sampling rarely
    hits the tiny sun. BSDF-only sampling is very noisy here.

With MIS (power heuristic) both regimes denoise at the same sample count.

Run: python make_scene.py
Produces: scene.json and scene.bin in the same directory.
"""

import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).parent


def write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def pack_f32(xs):
    return struct.pack(f"<{len(xs)}f", *xs)


def pack_u32(xs):
    return struct.pack(f"<{len(xs)}I", *xs)


def identity_mat4():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def quad(p0, p1, p2, p3, n):
    verts = [*p0, *p1, *p2, *p3]
    normals = [*n, *n, *n, *n]
    indices = [0, 1, 2, 0, 2, 3]
    return verts, normals, indices


def tilted_plate(cx, cy, cz, width_x, depth_y, alpha_rad):
    """A flat quad in local XY, rotated by alpha around X (back edge up), then translated.

    Returns (verts[12], normals[12], indices[6]).
    """
    c, s = math.cos(alpha_rad), math.sin(alpha_rad)

    def tf(lx, ly):
        return [cx + lx, cy + ly * c, cz + ly * s]

    hw, hd = width_x / 2.0, depth_y / 2.0
    p0 = tf(-hw, -hd)
    p1 = tf(hw, -hd)
    p2 = tf(hw, hd)
    p3 = tf(-hw, hd)
    # Rx(alpha) applied to (0,0,1) -> (0, -sin a, cos a)
    n = [0.0, -s, c]
    return quad(p0, p1, p2, p3, n)


def camera_matrix_look_at(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """Blender-style matrix_world for a camera looking from eye to target.

    Blender camera convention: local -Z = forward, local +Y = up, local +X = right.
    Returns a 4x4 row-major matrix as a flat list of 16 floats.
    """
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def norm(v):
        m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return (v[0] / m, v[1] / m, v[2] / m)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    forward = norm(sub(target, eye))
    right = norm(cross(forward, up_hint))
    up = cross(right, forward)

    # Columns of matrix_world: [right, up, -forward, eye]
    col0 = right
    col1 = up
    col2 = (-forward[0], -forward[1], -forward[2])
    col3 = eye
    return [
        col0[0], col1[0], col2[0], col3[0],
        col0[1], col1[1], col2[1], col3[1],
        col0[2], col1[2], col2[2], col3[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def make_envmap(width=512, height=256):
    """Procedural HDR envmap: dim blue sky + one small intense sun disc.

    Equirectangular layout matching devicecode.cu:
      theta = pi * (y + 0.5) / height  (0 at +Z, pi at -Z)
      phi   = 2*pi * (x + 0.5) / width (phi=0 is +X, phi=pi/2 is +Y)

    A single small sun is placed at (theta, phi) = (50 deg, 90 deg). For
    plates tilted 20 deg about X in front of a camera at ~(0,-4,1.5) this
    puts the sun right in the specular reflection direction, producing a
    sharp highlight on smooth plates and a broad glow on rough ones.
    """
    sky_zenith = (0.08, 0.14, 0.26)
    sky_horizon = (0.28, 0.35, 0.45)
    sky_strength = 0.25

    # (theta_c, phi_c, radius_rad, color, intensity)
    suns = [
        (math.radians(50.0), math.radians(90.0), math.radians(1.8),
         (1.0, 0.92, 0.78), 1500.0),
    ]

    sun_cos = []
    sun_dir = []
    for (theta_c, phi_c, rad, color, intensity) in suns:
        sun_cos.append(math.cos(rad))
        sun_dir.append((
            math.sin(theta_c) * math.cos(phi_c),
            math.sin(theta_c) * math.sin(phi_c),
            math.cos(theta_c),
            color,
            intensity,
        ))

    pixels = []
    for y in range(height):
        theta = math.pi * (y + 0.5) / height
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for x in range(width):
            phi = 2.0 * math.pi * (x + 0.5) / width
            dx = sin_t * math.cos(phi)
            dy = sin_t * math.sin(phi)
            dz = cos_t

            # Sky gradient along zenith angle: zenith at theta=0, horizon at theta=pi/2.
            t = abs(cos_t)  # 1 at zenith/nadir, 0 at horizon
            r = ((1.0 - t) * sky_horizon[0] + t * sky_zenith[0]) * sky_strength
            g = ((1.0 - t) * sky_horizon[1] + t * sky_zenith[1]) * sky_strength
            b = ((1.0 - t) * sky_horizon[2] + t * sky_zenith[2]) * sky_strength

            for cos_r, (sx, sy, sz, color, intensity) in zip(sun_cos, sun_dir):
                cdot = dx * sx + dy * sy + dz * sz
                if cdot >= cos_r:
                    r += color[0] * intensity
                    g += color[1] * intensity
                    b += color[2] * intensity

            pixels.append(r)
            pixels.append(g)
            pixels.append(b)

    return width, height, pixels


def main():
    buf = bytearray()
    meshes = []
    objects = []
    materials = []
    textures = []

    def add_mesh_object(v_list, n_list, i_list, material_id):
        mesh = {
            "vertices": write_blob(buf, pack_f32(v_list)),
            "normals": write_blob(buf, pack_f32(n_list)),
            "indices": write_blob(buf, pack_u32(i_list)),
        }
        mesh_id = len(meshes)
        meshes.append(mesh)
        objects.append({
            "mesh": mesh_id,
            "material": material_id,
            "transform": identity_mat4(),
        })

    # --- Materials ---
    M_FLOOR = len(materials)
    materials.append({
        "base_color": [0.58, 0.55, 0.50],
        "metallic": 0.0,
        "roughness": 0.9,
    })

    # Four metallic plates: smooth (front) -> rough (back).
    plate_roughnesses = [0.005, 0.03, 0.10, 0.35]
    M_PLATE = []
    for r in plate_roughnesses:
        M_PLATE.append(len(materials))
        materials.append({
            "base_color": [0.95, 0.93, 0.88],
            "metallic": 1.0,
            "roughness": r,
        })

    # Camera (declared early so plate tilts can aim reflections at the sun).
    eye = (0.0, -4.5, 1.8)
    target = (0.0, 0.7, 1.0)

    # --- Geometry ---
    # Floor: large diffuse quad at z=0.
    v, n, i = quad(
        [-4.0, -2.0, 0.0],
        [4.0, -2.0, 0.0],
        [4.0, 6.0, 0.0],
        [-4.0, 6.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    add_mesh_object(v, n, i, M_FLOOR)

    # Four metal plates stacked going away from the camera. Each plate's X-axis
    # tilt is chosen so the mirror direction from camera-to-plate-center lands
    # exactly on the sun, so every plate shows a highlight whose sharpness
    # reveals its roughness.
    sun_theta = math.radians(50.0)
    sun_phi = math.radians(90.0)
    sun_dir = (
        math.sin(sun_theta) * math.cos(sun_phi),
        math.sin(sun_theta) * math.sin(sun_phi),
        math.cos(sun_theta),
    )
    plate_size = (3.2, 0.55)
    plate_positions = [
        (0.0, -0.30, 0.55),
        (0.0,  0.30, 1.05),
        (0.0,  0.95, 1.55),
        (0.0,  1.70, 2.10),
    ]
    for pos, mat_id in zip(plate_positions, M_PLATE):
        dx, dy, dz = (pos[0] - eye[0], pos[1] - eye[1], pos[2] - eye[2])
        dm = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / dm, dy / dm, dz / dm
        # Plate normal = half vector between sun direction and view direction.
        hx = sun_dir[0] - dx
        hy = sun_dir[1] - dy
        hz = sun_dir[2] - dz
        hm = math.sqrt(hx * hx + hy * hy + hz * hz)
        nx, ny, nz = hx / hm, hy / hm, hz / hm
        # Our plate orientation is a rotation about X: n = (0, -sin a, cos a).
        alpha = math.atan2(-ny, nz)
        v, n, i = tilted_plate(pos[0], pos[1], pos[2],
                               plate_size[0], plate_size[1], alpha)
        add_mesh_object(v, n, i, mat_id)

    # --- Envmap texture ---
    ew, eh, env_pixels = make_envmap(width=512, height=256)
    tex_blob = write_blob(buf, pack_f32(env_pixels))
    textures.append({
        "width": ew,
        "height": eh,
        "channels": 3,
        "colorspace": "linear",
        "pixels": tex_blob,
    })

    # --- Camera ---
    cam_mat = camera_matrix_look_at(eye, target)

    scene = {
        "version": 1,
        "binary": "scene.bin",
        "render": {
            "width": 800,
            "height": 500,
            "spp": 128,
            "max_depth": 6,
        },
        "camera": {
            "transform": cam_mat,
            "fov_y_rad": math.radians(38.0),
        },
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "objects": objects,
        "lights": [],
        "world": {
            "type": "envmap",
            "texture": 0,
            "strength": 1.0,
            "rotation_z_rad": 0.0,
        },
    }

    (HERE / "scene.bin").write_bytes(bytes(buf))
    (HERE / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"Wrote scene.json ({(HERE / 'scene.json').stat().st_size} bytes)")
    print(f"Wrote scene.bin ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
