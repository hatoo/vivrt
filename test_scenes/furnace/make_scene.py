"""White-furnace test for GGX energy compensation.

Five metallic spheres, base_color = (1,1,1), metallic = 1.0, with roughness
sweeping from near-mirror to fully rough, sit inside a uniform white envmap
(L_env = 1 in every direction). The envmap path enables NEE + MIS, so the
integrand converges cleanly even on rough surfaces. With a perfectly energy-
conserving microfacet BRDF every sphere should integrate to ~1.0 outgoing
radiance and look indistinguishable from the background. Without multi-
scattering compensation, the rough spheres visibly darken (energy lost to
inter-microfacet bounces is dropped on the floor by single-scattering
Cook-Torrance + Smith).

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


def uv_sphere(cx, cy, cz, radius, u_segments=48, v_segments=24):
    """UV sphere centered at (cx,cy,cz). Returns (verts, normals, indices)."""
    verts = []
    normals = []
    for j in range(v_segments + 1):
        v = j / v_segments
        theta = math.pi * v  # 0 (north pole) -> pi (south pole)
        st, ct = math.sin(theta), math.cos(theta)
        for i in range(u_segments + 1):
            u = i / u_segments
            phi = 2.0 * math.pi * u
            sp, cp = math.sin(phi), math.cos(phi)
            nx, ny, nz = st * cp, st * sp, ct
            verts.extend([cx + radius * nx, cy + radius * ny, cz + radius * nz])
            normals.extend([nx, ny, nz])
    indices = []
    row = u_segments + 1
    for j in range(v_segments):
        for i in range(u_segments):
            a = j * row + i
            b = a + 1
            c = a + row
            d = c + 1
            indices.extend([a, c, b, b, c, d])
    return verts, normals, indices


def camera_matrix_look_at(eye, target, up_hint=(0.0, 0.0, 1.0)):
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


def main():
    buf = bytearray()
    meshes = []
    objects = []
    materials = []

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

    # (metallic, roughness) tuples. First is a white diffuse reference (should
    # converge to exactly 1.0 in a white furnace); the rest sweep roughness on
    # pure white metal to exercise GGX energy compensation.
    spheres = [
        (0.0, 0.9),   # diffuse reference
        (1.0, 0.05),
        (1.0, 0.25),
        (1.0, 0.50),
        (1.0, 0.75),
        (1.0, 1.00),
    ]
    radius = 0.45
    spacing = 2.5  # far apart so specular reflections can't see neighbors
    n = len(spheres)
    x0 = -spacing * (n - 1) / 2.0
    for k, (met, rough) in enumerate(spheres):
        mat_id = len(materials)
        materials.append({
            "base_color": [1.0, 1.0, 1.0],
            "metallic": met,
            "roughness": rough,
        })
        cx = x0 + k * spacing
        v, nrm, idx = uv_sphere(cx, 0.0, 0.0, radius)
        add_mesh_object(v, nrm, idx, mat_id)

    # Uniform white envmap (constant radiance 1 in every direction). Using an
    # envmap (rather than world_type=constant) enables envmap NEE + MIS, which
    # drastically reduces variance for rough reflections — critical for a
    # furnace test where the only light is the background.
    env_w, env_h = 8, 4
    env_pixels = [1.0] * (env_w * env_h * 3)
    env_blob = write_blob(buf, pack_f32(env_pixels))
    textures = [{
        "width": env_w,
        "height": env_h,
        "channels": 3,
        "colorspace": "linear",
        "pixels": env_blob,
    }]

    eye = (0.0, -12.0, 0.6)
    target = (0.0, 0.0, 0.0)
    cam_mat = camera_matrix_look_at(eye, target)

    scene = {
        "version": 1,
        "binary": "scene.bin",
        "render": {
            "width": 900,
            "height": 200,
            "spp": 256,
            "max_depth": 8,
        },
        "camera": {
            "transform": cam_mat,
            "fov_y_rad": math.radians(22.0),
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
