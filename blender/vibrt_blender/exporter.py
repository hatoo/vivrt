"""Export the evaluated Blender scene to scene.json + scene.bin.

Blender uses right-handed, Z-up, metres. vibrt uses the same, so no
axis conversion is required — only matrix transposition (Blender matrices are
column-major via mathutils, our JSON schema expects row-major 16-float arrays).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import bpy

from . import material_export


def _matrix_to_row_major(m) -> list[float]:
    return [m[i][j] for i in range(4) for j in range(4)]


def _write_f32(buf: bytearray, xs) -> dict:
    """Append a float32 blob to `buf`. Accepts numpy arrays (zero-copy view)
    or sequences (coerced via np.asarray). Saves a full-buffer copy vs
    `.tobytes()` — for ~1GB of textures that's ~200ms.
    """
    import numpy as np
    arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    off = len(buf)
    buf.extend(memoryview(arr).cast("B"))
    pad = (-len(buf)) & 15
    if pad:
        buf.extend(b"\x00" * pad)
    return {"offset": off, "len": int(arr.nbytes)}


def _write_u32(buf: bytearray, xs) -> dict:
    import numpy as np
    arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.uint32)
    arr = np.ascontiguousarray(arr, dtype=np.uint32)
    off = len(buf)
    buf.extend(memoryview(arr).cast("B"))
    pad = (-len(buf)) & 15
    if pad:
        buf.extend(b"\x00" * pad)
    return {"offset": off, "len": int(arr.nbytes)}


def _find_displacement(obj_eval, buf, textures):
    """Return (tex_id, strength) for the first material slot's Displacement
    output that drives a single TexImage; (None, 0.0) otherwise.
    """
    for slot in obj_eval.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        out = next(
            (n for n in mat.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputMaterial"
             and n.is_active_output),
            None,
        )
        if out is None:
            continue
        disp = out.inputs.get("Displacement")
        if disp is None or not disp.is_linked:
            continue
        src = disp.links[0].from_node
        strength = 1.0
        if src.bl_idname == "ShaderNodeDisplacement":
            if "Scale" in src.inputs:
                strength = float(src.inputs["Scale"].default_value)
            height_sock = src.inputs.get("Height")
            if height_sock is None or not height_sock.is_linked:
                continue
            upstream = height_sock.links[0].from_node
            if upstream.bl_idname == "ShaderNodeTexImage" and upstream.image is not None:
                from . import material_export
                tex_id = material_export.export_image_texture(
                    upstream.image, buf, textures, "linear"
                )
                return tex_id, strength
        elif src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
            from . import material_export
            tex_id = material_export.export_image_texture(
                src.image, buf, textures, "linear"
            )
            return tex_id, strength
    return None, 0.0


def _export_mesh(obj_eval, buf: bytearray, textures: list) -> dict | None:
    """Export a mesh + per-triangle material indices.

    Returns a dict suitable for a `MeshDesc` — always includes `material_indices`
    when there's more than one slot, otherwise the entry is omitted.

    Vertex / normal / UV / index arrays are built through `foreach_get` into
    numpy buffers. A Python-level per-triangle loop was the dominant export
    cost on non-trivial meshes (seconds per 100k-tri object vs milliseconds).
    """
    import numpy as np

    mesh = obj_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        ntri = len(mesh.loop_triangles)
        if ntri == 0:
            return None
        # Split normals are auto-computed in Blender 4.2+; older branches still
        # need the explicit call.
        if hasattr(mesh, "calc_normals_split"):
            try:
                mesh.calc_normals_split()
            except RuntimeError:
                pass
        num_slots = len(obj_eval.material_slots)

        tri_vert_idx = np.empty(ntri * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_vert_idx)
        tri_loop_idx = np.empty(ntri * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loop_idx)
        split_normals = np.empty(ntri * 9, dtype=np.float32)
        mesh.loop_triangles.foreach_get("split_normals", split_normals)

        nverts = len(mesh.vertices)
        vcos = np.empty(nverts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vcos)
        vcos = vcos.reshape(-1, 3)
        positions = np.ascontiguousarray(vcos[tri_vert_idx].reshape(-1))

        uv_layer = mesh.uv_layers.active
        if uv_layer is not None:
            nloops = len(mesh.loops)
            uv_all = np.empty(nloops * 2, dtype=np.float32)
            uv_layer.data.foreach_get("uv", uv_all)
            uv_all = uv_all.reshape(-1, 2)
            uvs = np.ascontiguousarray(uv_all[tri_loop_idx].reshape(-1))
        else:
            uvs = None

        indices = np.arange(ntri * 3, dtype=np.uint32)

        desc = {
            "vertices": _write_f32(buf, positions),
            "normals": _write_f32(buf, split_normals),
            "uvs": _write_f32(buf, uvs) if uvs is not None else None,
            "indices": _write_u32(buf, indices),
        }
        if num_slots > 1:
            tri_mat_idx = np.empty(ntri, dtype=np.int32)
            mesh.loop_triangles.foreach_get("material_index", tri_mat_idx)
            desc["material_indices"] = _write_u32(buf, tri_mat_idx)
        disp_tex, disp_strength = _find_displacement(obj_eval, buf, textures)
        if disp_tex is not None and disp_strength != 0.0:
            desc["displacement_tex"] = disp_tex
            desc["displacement_strength"] = disp_strength
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
    t0 = time.perf_counter()
    material_export.reset_stats()
    mesh_s = 0.0
    slow_meshes: list[tuple] = []
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
                t_mesh = time.perf_counter()
                mesh = _export_mesh(obj_eval, buf, textures)
                dt_mesh = time.perf_counter() - t_mesh
                mesh_s += dt_mesh
                if mesh is None:
                    continue
                if dt_mesh > 0.25:
                    ntri = mesh["indices"]["len"] // 12  # 4 bytes/u32, 3 per tri
                    slow_meshes.append((obj_eval.name, ntri, dt_mesh))
                    slow_meshes.sort(key=lambda e: e[2], reverse=True)
                    del slow_meshes[5:]
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

    t_world = time.perf_counter()
    world = _export_world(scene.world, buf, textures)
    world_s = time.perf_counter() - t_world

    spp = max(1, int(scene.vibrt_spp))
    print(f"[vibrt] spp={spp}")
    scene_json = {
        "version": 1,
        "binary": bin_path.name,
        "render": {
            "width": width,
            "height": height,
            "spp": spp,
            "max_depth": 8,
            "clamp_indirect": float(scene.vibrt_clamp_indirect),
        },
        "camera": _export_camera(scene, cam_obj, aspect),
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "objects": objects,
        "lights": lights_json,
        "world": world,
    }

    t_bin = time.perf_counter()
    # `write_bytes(bytes(buf))` would copy the whole bytearray once more before
    # hitting the OS write; for ~1GB scenes that's ~200ms of pure memcpy.
    with bin_path.open("wb") as f:
        f.write(buf)
    bin_write_s = time.perf_counter() - t_bin

    t_dump = time.perf_counter()
    json_blob = json.dumps(scene_json, indent=2)
    json_dump_s = time.perf_counter() - t_dump

    t_jw = time.perf_counter()
    json_path.write_text(json_blob)
    json_write_s = time.perf_counter() - t_jw

    dt = time.perf_counter() - t0
    stats = material_export.pop_stats()
    material_s = stats["material_self_s"]
    texture_s = stats["texture_self_s"]
    pixel_read_s = stats["pixel_read_s"]
    bake_chain_s = stats["bake_chain_s"]
    accounted = (
        mesh_s + material_s + texture_s + pixel_read_s + bake_chain_s
        + world_s + json_dump_s + bin_write_s + json_write_s
    )
    other_s = max(0.0, dt - accounted)
    print(
        f"[vibrt] export {dt:.2f}s "
        f"({len(meshes)} mesh, {len(objects)} obj, "
        f"{len(textures)} tex, {len(materials)} mat, "
        f"{len(buf)/1024/1024:.1f}MB bin, {stats['pixel_bytes']/1024/1024:.1f}MB px)"
    )
    print(
        f"[vibrt]   mesh={mesh_s:.2f}s  material={material_s:.2f}s  "
        f"texture={texture_s:.2f}s  pixel_read={pixel_read_s:.2f}s  "
        f"bake_chain={bake_chain_s:.2f}s"
    )
    print(
        f"[vibrt]   world={world_s:.2f}s  json_dump={json_dump_s:.2f}s  "
        f"bin_write={bin_write_s:.2f}s  json_write={json_write_s:.2f}s  "
        f"other={other_s:.2f}s"
    )
    if slow_meshes:
        desc = ", ".join(f"{n}({t:.0f}k tri, {d:.2f}s)" for n, t, d in
                         ((nm, ntri / 1000.0, dd) for nm, ntri, dd in slow_meshes))
        print(f"[vibrt]   slow meshes: {desc}")
    if stats["slow_textures"]:
        desc = ", ".join(
            f"{n}({w}x{h}, {d:.2f}s)" for n, w, h, d in stats["slow_textures"]
        )
        print(f"[vibrt]   slow textures: {desc}")
    if stats["slow_materials"]:
        desc = ", ".join(f"{n}({d:.2f}s)" for n, d in stats["slow_materials"])
        print(f"[vibrt]   slow materials: {desc}")
