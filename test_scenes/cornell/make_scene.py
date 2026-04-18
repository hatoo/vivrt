"""Generate a Cornell-box-ish scene.json + scene.bin for vibrt-blender.

Run: python make_scene.py
Produces: scene.json and scene.bin in the same directory.
"""

import json
import struct
from pathlib import Path

HERE = Path(__file__).parent


def write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    # align to 16 bytes
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def quad(p0, p1, p2, p3, n):
    # Returns (vertices[12], normals[12], indices[6]) for two triangles.
    verts = [*p0, *p1, *p2, *p3]
    normals = [*n, *n, *n, *n]
    indices = [0, 1, 2, 0, 2, 3]
    return verts, normals, indices


def cube(center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    V = []
    N = []
    I = []

    def add_face(p0, p1, p2, p3, n):
        base = len(V) // 3
        V.extend([*p0, *p1, *p2, *p3])
        N.extend([*n, *n, *n, *n])
        I.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    # +X
    add_face(
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx + sx, cy + sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [1, 0, 0],
    )
    # -X
    add_face(
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
        [-1, 0, 0],
    )
    # +Y
    add_face(
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [0, 1, 0],
    )
    # -Y
    add_face(
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz + sz],
        [cx - sx, cy - sy, cz + sz],
        [0, -1, 0],
    )
    # +Z (top)
    add_face(
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
        [0, 0, 1],
    )
    # -Z (bottom)
    add_face(
        [cx - sx, cy + sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx - sx, cy - sy, cz - sz],
        [0, 0, -1],
    )
    return V, N, I


def pack_f32(xs):
    return struct.pack(f"<{len(xs)}f", *xs)


def pack_u32(xs):
    return struct.pack(f"<{len(xs)}I", *xs)


def identity_mat4_rowmajor():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def translate_mat4_rowmajor(tx, ty, tz):
    return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1]


def main():
    buf = bytearray()
    meshes = []
    objects = []
    materials = []

    # Wall meshes as individual quads.
    # Cornell box axis: Z is up, camera looks along -Y.
    # Room: x in [-1,1], y in [-1,1], z in [0,2].
    def add_wall(v_list, n_list, i_list, material_id):
        mesh = {
            "vertices": write_blob(buf, pack_f32(v_list)),
            "normals": write_blob(buf, pack_f32(n_list)),
            "indices": write_blob(buf, pack_u32(i_list)),
        }
        mesh_id = len(meshes)
        meshes.append(mesh)
        objects.append(
            {
                "mesh": mesh_id,
                "material": material_id,
                "transform": identity_mat4_rowmajor(),
            }
        )

    # Materials (index matters)
    M_WHITE = 0
    M_RED = 1
    M_GREEN = 2
    M_METAL = 3
    M_GLASS = 4

    materials.append(
        {"base_color": [0.73, 0.73, 0.73], "metallic": 0.0, "roughness": 0.9}
    )
    materials.append(
        {"base_color": [0.65, 0.05, 0.05], "metallic": 0.0, "roughness": 0.9}
    )
    materials.append(
        {"base_color": [0.12, 0.45, 0.15], "metallic": 0.0, "roughness": 0.9}
    )
    materials.append(
        {"base_color": [0.95, 0.93, 0.88], "metallic": 1.0, "roughness": 0.15}
    )
    materials.append(
        {
            "base_color": [0.95, 0.95, 0.95],
            "metallic": 0.0,
            "roughness": 0.02,
            "ior": 1.5,
            "transmission": 1.0,
        }
    )

    # Floor (Z=0) white
    v, n, i = quad(
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]
    )
    add_wall(v, n, i, M_WHITE)
    # Ceiling (Z=2) white
    v, n, i = quad(
        [-1, -1, 2], [-1, 1, 2], [1, 1, 2], [1, -1, 2], [0, 0, -1]
    )
    add_wall(v, n, i, M_WHITE)
    # Back wall (Y=1) white
    v, n, i = quad(
        [-1, 1, 0], [1, 1, 0], [1, 1, 2], [-1, 1, 2], [0, -1, 0]
    )
    add_wall(v, n, i, M_WHITE)
    # Left wall (X=-1) red
    v, n, i = quad(
        [-1, -1, 0], [-1, 1, 0], [-1, 1, 2], [-1, -1, 2], [1, 0, 0]
    )
    add_wall(v, n, i, M_RED)
    # Right wall (X=1) green
    v, n, i = quad(
        [1, 1, 0], [1, -1, 0], [1, -1, 2], [1, 1, 2], [-1, 0, 0]
    )
    add_wall(v, n, i, M_GREEN)

    # Metal cube (left)
    v, n, i = cube([-0.35, 0.2, 0.3], [0.6, 0.6, 0.6])
    add_wall(v, n, i, M_METAL)
    # Glass cube (right)
    v, n, i = cube([0.35, -0.1, 0.3], [0.5, 0.5, 0.5])
    add_wall(v, n, i, M_GLASS)

    # Area light on ceiling (size 0.6 x 0.6, centered)
    # Blender-style area transform: local XY plane, emission along local +Z.
    # Here we want light facing downward, so rotate local Z to -Z world.
    area_transform = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 1.99,
        0, 0, 0, 1,
    ]
    lights = [
        {
            "type": "area_rect",
            "transform": area_transform,
            "size": [0.6, 0.6],
            "color": [1.0, 0.96, 0.85],
            "power": 45.0,
        }
    ]

    # Camera: eye at (0, -3, 1), looking at (0, 0, 1), up=+Z.
    # Blender camera convention: local -Z = forward, local +Y = up, local +X = right.
    # world forward = (0,1,0); world up = (0,0,1); world right = forward x up = (1,0,0).
    # Columns of matrix_world: col0=local-X-in-world=(1,0,0), col1=local-Y-in-world=(0,0,1),
    #   col2=local-Z-in-world=(0,-1,0), col3=translation=(0,-3,1).
    # Row-major layout (row i, col j = m[i*4+j]):
    camera_mat = [
        1, 0, 0, 0.0,    # row 0
        0, 0, -1, -3.0,  # row 1
        0, 1, 0, 1.0,    # row 2
        0, 0, 0, 1,      # row 3
    ]

    scene = {
        "version": 1,
        "binary": "scene.bin",
        "render": {
            "width": 600,
            "height": 600,
            "spp": 64,
            "max_depth": 8,
        },
        "camera": {
            "transform": camera_mat,
            "fov_y_rad": 0.6911112,  # ~39.6 deg (Blender default)
        },
        "meshes": meshes,
        "materials": materials,
        "textures": [],
        "objects": objects,
        "lights": lights,
        "world": {"type": "constant", "color": [0.0, 0.0, 0.0], "strength": 0.0},
    }

    (HERE / "scene.bin").write_bytes(bytes(buf))
    (HERE / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"Wrote scene.json ({(HERE/'scene.json').stat().st_size} bytes)")
    print(f"Wrote scene.bin ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
