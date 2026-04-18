"""Export the evaluated Blender scene to scene.json + scene.bin.

Blender uses right-handed, Z-up, metres. vibrt uses the same, so no
axis conversion is required — only matrix transposition (Blender matrices are
column-major via mathutils, our JSON schema expects row-major 16-float arrays).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import bpy

from . import material_export


def _matrix_to_row_major(m) -> list[float]:
    return [m[i][j] for i in range(4) for j in range(4)]


def _pack_f32(xs) -> bytes:
    return struct.pack(f"<{len(xs)}f", *xs)


def _pack_u32(xs) -> bytes:
    return struct.pack(f"<{len(xs)}I", *xs)


def _write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def _export_mesh(obj_eval, buf: bytearray) -> dict | None:
    """Export a mesh + per-triangle material indices.

    Returns a dict suitable for a `MeshDesc` — always includes `material_indices`
    when there's more than one slot, otherwise the entry is omitted.
    """
    mesh = obj_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            return None
        verts: list[float] = []
        normals: list[float] = []
        uvs: list[float] = []
        indices: list[int] = []
        mat_indices: list[int] = []
        uv_layer = mesh.uv_layers.active
        uv_data = uv_layer.data if uv_layer is not None else None
        if hasattr(mesh, "calc_normals_split"):
            mesh.calc_normals_split()
        num_slots = len(obj_eval.material_slots)
        for tri in mesh.loop_triangles:
            for k in range(3):
                li = tri.loops[k]
                vi = mesh.loops[li].vertex_index
                v = mesh.vertices[vi].co
                n = tri.split_normals[k]
                verts.extend((v.x, v.y, v.z))
                normals.extend((n[0], n[1], n[2]))
                if uv_data is not None:
                    uv = uv_data[li].uv
                    uvs.extend((uv.x, uv.y))
                else:
                    uvs.extend((0.0, 0.0))
                indices.append(len(indices))
            mat_indices.append(int(tri.material_index) if num_slots > 1 else 0)
        desc = {
            "vertices": _write_blob(buf, _pack_f32(verts)),
            "normals": _write_blob(buf, _pack_f32(normals)),
            "uvs": _write_blob(buf, _pack_f32(uvs)) if uv_data else None,
            "indices": _write_blob(buf, _pack_u32(indices)),
        }
        if num_slots > 1:
            desc["material_indices"] = _write_blob(buf, _pack_u32(mat_indices))
        return desc
    finally:
        obj_eval.to_mesh_clear()


def _export_camera(scene, cam_obj, aspect: float) -> dict:
    cam = cam_obj.data
    if cam.type != "PERSP":
        raise ValueError(
            f"vibrt only supports perspective cameras (got {cam.type})"
        )
    # Blender's sensor_fit + sensor_width/height govern how angle maps to vertical
    # FOV. Use the "vertical FOV" approximation from `angle_y`.
    fov_y = cam.angle_y
    matrix = _matrix_to_row_major(cam_obj.matrix_world)
    return {
        "transform": matrix,
        "fov_y_rad": fov_y,
        "lens_radius": 0.0,
        "focal_distance": max(cam.dof.focus_distance, 0.001)
        if cam.dof.use_dof
        else 1.0,
    }


def _export_light(obj, buf: bytearray, textures: list) -> dict | None:
    light = obj.data
    col = list(light.color)
    mw = _matrix_to_row_major(obj.matrix_world)
    if light.type == "POINT":
        position = [obj.matrix_world[i][3] for i in range(3)]
        return {
            "type": "point",
            "position": position,
            "color": col,
            "power": float(light.energy),
            "radius": max(light.shadow_soft_size, 0.005),
        }
    if light.type == "SUN":
        # Blender sun: local -Z is light direction in world
        direction = [-obj.matrix_world[i][2] for i in range(3)]
        return {
            "type": "sun",
            "direction": direction,
            "color": col,
            "strength": float(light.energy),
            "angle_rad": float(light.angle),
        }
    if light.type == "SPOT":
        return {
            "type": "spot",
            "transform": mw,
            "color": col,
            "power": float(light.energy),
            "cone_rad": float(light.spot_size) * 0.5,
            "blend": float(light.spot_blend),
        }
    if light.type == "AREA":
        if light.shape == "SQUARE":
            size = [float(light.size), float(light.size)]
        elif light.shape == "RECTANGLE":
            size = [float(light.size), float(light.size_y)]
        else:
            size = [float(light.size), float(light.size)]
        return {
            "type": "area_rect",
            "transform": mw,
            "size": size,
            "color": col,
            "power": float(light.energy),
        }
    return None


def _export_world(world, buf: bytearray, textures: list) -> dict:
    if world is None or not world.use_nodes:
        col = list(world.color)[:3] if world else [0.05, 0.05, 0.05]
        return {"type": "constant", "color": col, "strength": 1.0}

    out = world.node_tree.nodes.get("World Output")
    if out is None:
        out = next(
            (n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"),
            None,
        )
    if out is None or not out.inputs["Surface"].is_linked:
        return {"type": "constant", "color": [0, 0, 0], "strength": 0.0}

    src = out.inputs["Surface"].links[0].from_node
    if src.bl_idname == "ShaderNodeBackground":
        strength = src.inputs["Strength"].default_value
        color_sock = src.inputs["Color"]
        if color_sock.is_linked:
            linked = color_sock.links[0].from_node
            if linked.bl_idname == "ShaderNodeTexEnvironment" and linked.image is not None:
                tex_id = material_export.export_image_texture(
                    linked.image, buf, textures, colorspace="linear"
                )
                rotation_z_rad = 0.0
                return {
                    "type": "envmap",
                    "texture": tex_id,
                    "rotation_z_rad": rotation_z_rad,
                    "strength": float(strength),
                }
        col = list(color_sock.default_value)[:3]
        return {"type": "constant", "color": col, "strength": float(strength)}

    return {"type": "constant", "color": [0, 0, 0], "strength": 0.0}


def export_scene(
    depsgraph: bpy.types.Depsgraph,
    json_path: Path,
    bin_path: Path,
):
    scene = depsgraph.scene_eval
    rd = scene.render
    width = int(rd.resolution_x * rd.resolution_percentage / 100.0)
    height = int(rd.resolution_y * rd.resolution_percentage / 100.0)
    aspect = width / height if height > 0 else 1.0

    buf = bytearray()
    textures: list[dict] = []
    materials: list[dict] = []
    material_index: dict[str, int] = {}

    def resolve_material(mat) -> int:
        if mat is None:
            # Default diffuse grey
            name = "__default__"
            if name in material_index:
                return material_index[name]
            mid = len(materials)
            materials.append(
                {"base_color": [0.8, 0.8, 0.8], "metallic": 0.0, "roughness": 0.5}
            )
            material_index[name] = mid
            return mid
        if mat.name in material_index:
            return material_index[mat.name]
        exported = material_export.export_material(mat, buf, textures)
        mid = len(materials)
        materials.append(exported)
        material_index[mat.name] = mid
        return mid

    meshes: list[dict] = []
    objects: list[dict] = []
    mesh_cache: dict[str, int] = {}

    cam_obj = None
    lights_json: list[dict] = []

    for inst in depsgraph.object_instances:
        obj = inst.object
        if inst.is_instance:
            obj_eval = obj
        else:
            obj_eval = obj.evaluated_get(depsgraph)

        if obj_eval.type == "MESH":
            mkey = f"{obj_eval.data.name}#{obj_eval.name}"
            mesh_id = mesh_cache.get(mkey)
            if mesh_id is None:
                mesh = _export_mesh(obj_eval, buf)
                if mesh is None:
                    continue
                mesh_id = len(meshes)
                meshes.append(mesh)
                mesh_cache[mkey] = mesh_id
            slot_mat_ids: list[int] = []
            if obj_eval.material_slots:
                for slot in obj_eval.material_slots:
                    slot_mat_ids.append(resolve_material(slot.material))
            if not slot_mat_ids:
                slot_mat_ids = [resolve_material(None)]
            obj_desc = {
                "mesh": mesh_id,
                "material": slot_mat_ids[0],
                "transform": _matrix_to_row_major(inst.matrix_world),
            }
            if len(slot_mat_ids) > 1:
                obj_desc["materials"] = slot_mat_ids
            objects.append(obj_desc)
        elif obj_eval.type == "LIGHT":
            light = _export_light(obj_eval, buf, textures)
            if light is not None:
                lights_json.append(light)
        elif obj_eval.type == "CAMERA" and obj_eval == scene.camera.evaluated_get(
            depsgraph
        ):
            cam_obj = obj_eval

    if cam_obj is None and scene.camera is not None:
        cam_obj = scene.camera.evaluated_get(depsgraph)
    if cam_obj is None:
        raise RuntimeError("No active camera in scene")

    world = _export_world(scene.world, buf, textures)

    # samples-per-pixel: prefer a custom scene prop if set, else fall back to Cycles if present, else 64
    spp = 64
    vibrt_sp = scene.get("vibrt_spp")
    if vibrt_sp is not None:
        spp = max(1, int(vibrt_sp))
    elif hasattr(scene, "cycles"):
        try:
            spp = max(1, int(scene.cycles.samples))
        except AttributeError:
            pass
    scene_json = {
        "version": 1,
        "binary": bin_path.name,
        "render": {
            "width": width,
            "height": height,
            "spp": spp,
            "max_depth": 8,
        },
        "camera": _export_camera(scene, cam_obj, aspect),
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "objects": objects,
        "lights": lights_json,
        "world": world,
    }

    bin_path.write_bytes(bytes(buf))
    json_path.write_text(json.dumps(scene_json, indent=2))
