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


def _export_mesh(
    obj_eval, buf: bytearray, textures: list, vc_attr_name: str | None = None
) -> dict | None:
    """Export a mesh + per-triangle material indices.

    Returns a dict suitable for a `MeshDesc` — always includes `material_indices`
    when there's more than one slot, otherwise the entry is omitted.

    Vertex / normal / UV / index arrays are built through `foreach_get` into
    numpy buffers. A Python-level per-triangle loop was the dominant export
    cost on non-trivial meshes (seconds per 100k-tri object vs milliseconds).

    If `vc_attr_name` is given and the mesh has a matching colour attribute,
    emits f32x3 per emitted vertex (one per triangle corner, matching the
    duplicated-vertex layout) so the renderer can interpolate it.
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
        if vc_attr_name:
            color_attrs = getattr(mesh, "color_attributes", None) or []
            target = next((ca for ca in color_attrs if ca.name == vc_attr_name), None)
            if target is not None:
                n = len(target.data)
                rgba = np.empty(n * 4, dtype=np.float32)
                target.data.foreach_get("color", rgba)
                rgba = rgba.reshape(n, 4)[:, :3]
                if target.domain == "CORNER":
                    per_corner = rgba[tri_loop_idx]
                elif target.domain == "POINT":
                    per_corner = rgba[tri_vert_idx]
                else:
                    per_corner = None
                if per_corner is not None:
                    desc["vertex_colors"] = _write_f32(
                        buf, np.ascontiguousarray(per_corner.reshape(-1))
                    )
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


def _light_node_emission(light) -> tuple[float, tuple[float, float, float]]:
    """Return (strength, color) contributed by an Emission shader driving the
    Light Output, as a multiplier on top of `light.energy`/`light.color`.

    Cycles evaluates the light's shader tree and multiplies its output with
    `light.energy * light.color`, so an Emission node whose Strength was edited
    directly on the node (rather than via the UI's Power field) still influences
    the render. classroom.blend's `blackBoard_light` and `sun` both rely on
    this: their `light.energy` is leftover UI value, and the node's Strength is
    the real source of truth.

    Returns (1.0, (1, 1, 1)) when there's no usable Emission — callers then see
    `light.energy`/`light.color` unchanged.
    """
    if not light.use_nodes or light.node_tree is None:
        return 1.0, (1.0, 1.0, 1.0)
    out = next(
        (n for n in light.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputLight" and n.is_active_output),
        None,
    )
    if out is None:
        out = next(
            (n for n in light.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputLight"),
            None,
        )
    if out is None:
        return 1.0, (1.0, 1.0, 1.0)
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return 1.0, (1.0, 1.0, 1.0)
    src = surf.links[0].from_node
    if src.bl_idname != "ShaderNodeEmission":
        return 1.0, (1.0, 1.0, 1.0)
    s_sock = src.inputs.get("Strength")
    c_sock = src.inputs.get("Color")
    if s_sock is None:
        strength = 1.0
    elif s_sock.is_linked:
        # Light Falloff drives Emission.Strength on classroom's blackBoard_light:
        # its own Strength input is the pre-falloff multiplier. Our renderer
        # already does physical inverse-square on area/point lights, so we use
        # that raw value and ignore which falloff output was picked.
        up = s_sock.links[0].from_node
        if up.bl_idname == "ShaderNodeLightFalloff":
            inner = up.inputs.get("Strength")
            strength = float(inner.default_value) if inner is not None and not inner.is_linked else 1.0
        else:
            strength = 1.0
    else:
        strength = float(s_sock.default_value)
    if c_sock is not None and not c_sock.is_linked:
        cv = c_sock.default_value
        color = (float(cv[0]), float(cv[1]), float(cv[2]))
    else:
        color = (1.0, 1.0, 1.0)
    return strength, color


def _export_light(obj, buf: bytearray, textures: list) -> dict | None:
    light = obj.data
    node_strength, node_color = _light_node_emission(light)
    energy = float(light.energy) * node_strength
    col = [light.color[0] * node_color[0],
           light.color[1] * node_color[1],
           light.color[2] * node_color[2]]
    mw = _matrix_to_row_major(obj.matrix_world)
    if light.type == "POINT":
        position = [obj.matrix_world[i][3] for i in range(3)]
        return {
            "type": "point",
            "position": position,
            "color": col,
            "power": energy,
            "radius": max(light.shadow_soft_size, 0.005),
        }
    if light.type == "SUN":
        # Blender sun: local -Z is light direction in world
        direction = [-obj.matrix_world[i][2] for i in range(3)]
        return {
            "type": "sun",
            "direction": direction,
            "color": col,
            "strength": energy,
            "angle_rad": float(light.angle),
        }
    if light.type == "SPOT":
        return {
            "type": "spot",
            "transform": mw,
            "color": col,
            "power": energy,
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
        # Blender area lights emit along local -Z; vibrt's area_rect expects
        # emission along local +Z (see cornell/make_scene.py). Flip the Z axis
        # (third column) so the emission direction survives the round trip.
        mw_flipz = list(mw)
        mw_flipz[2] = -mw_flipz[2]
        mw_flipz[6] = -mw_flipz[6]
        mw_flipz[10] = -mw_flipz[10]
        mw_flipz[14] = -mw_flipz[14]
        return {
            "type": "area_rect",
            "transform": mw_flipz,
            "size": size,
            "color": col,
            "power": energy,
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


def _collect_attribute_means(depsgraph) -> dict:
    """Mean colour per (material_name, attribute_name) across mesh objects.

    Feeds `material_export.set_attribute_means`. ShaderNodeAttribute nodes
    otherwise fold to neutral white, which breaks shaders (classroom
    wallClock_darkWood) where the attribute is the dominant colour input to a
    Mix. Using the mesh's actual mean keeps the baked colour close to what
    Cycles would render.

    Imprecise on meshes with multiple material slots — the mesh's overall
    attribute mean is credited to every referenced material rather than split
    per-slot. Good enough for this export path, where the typical use is a
    single-material asset (the clock frame) with a darkening mask attribute.
    """
    import numpy as np
    acc: dict = {}
    seen_meshes: set = set()
    for inst in depsgraph.object_instances:
        obj = inst.object
        obj_eval = obj if inst.is_instance else obj.evaluated_get(depsgraph)
        if obj_eval.type != "MESH":
            continue
        me = obj_eval.data
        if me is None:
            continue
        mesh_ptr = me.as_pointer()
        if mesh_ptr in seen_meshes:
            continue
        seen_meshes.add(mesh_ptr)
        col_attrs = getattr(me, "color_attributes", None)
        if not col_attrs:
            continue
        mats_used = {s.material.name for s in obj_eval.material_slots if s.material}
        if not mats_used:
            continue
        for ca in col_attrs:
            n = len(ca.data)
            if n == 0:
                continue
            buf = np.empty(n * 4, dtype=np.float32)
            try:
                ca.data.foreach_get("color", buf)
            except Exception:
                continue
            rgb = buf.reshape(n, 4)[:, :3]
            sum_rgb = rgb.sum(axis=0)
            for mat_name in mats_used:
                key = (mat_name, ca.name)
                e = acc.get(key)
                if e is None:
                    acc[key] = [float(sum_rgb[0]), float(sum_rgb[1]), float(sum_rgb[2]), n]
                else:
                    e[0] += float(sum_rgb[0])
                    e[1] += float(sum_rgb[1])
                    e[2] += float(sum_rgb[2])
                    e[3] += n
    return {
        key: [r / n, g / n, b / n]
        for key, (r, g, b, n) in acc.items()
        if n > 0
    }


def export_scene(
    depsgraph: bpy.types.Depsgraph,
    json_path: Path,
    bin_path: Path,
):
    t0 = time.perf_counter()
    material_export.reset_stats()
    material_export.set_attribute_means(_collect_attribute_means(depsgraph))
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
    # Materials that drive base_color from a ShaderNodeAttribute flag the
    # attribute name on their exported dict under `_vertex_color_attr`. We pop
    # it here so the material JSON stays clean, and remember it so meshes that
    # reference the material can emit the matching vertex-colour blob.
    material_vc_attr: dict[str, str] = {}

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
        vc_attr = exported.pop("_vertex_color_attr", None)
        if vc_attr:
            material_vc_attr[mat.name] = vc_attr
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

        # Skip meshes that Cycles hides from camera — they're light portals
        # (e.g. classroom's `windows` dayLight_portal) meant to inject light
        # but never appear as geometry on the rendered image. Without this
        # check the emissive portal bleeds over the ceiling / walls.
        if obj_eval.type == "MESH" and not getattr(obj, "visible_camera", True):
            continue

        if obj_eval.type == "MESH":
            # Resolve materials before mesh export — _export_mesh needs to know
            # which vertex-colour attribute (if any) a bound material references.
            slot_mat_ids: list[int] = []
            vc_attr_name: str | None = None
            if obj_eval.material_slots:
                for slot in obj_eval.material_slots:
                    slot_mat_ids.append(resolve_material(slot.material))
                    if slot.material is not None and vc_attr_name is None:
                        vc_attr_name = material_vc_attr.get(slot.material.name)
            if not slot_mat_ids:
                slot_mat_ids = [resolve_material(None)]

            # Same mesh data exported with / without vertex colour would need
            # separate blobs, so fold the attribute name into the cache key.
            mkey = f"{obj_eval.data.name}#{obj_eval.name}#vc={vc_attr_name}"
            mesh_id = mesh_cache.get(mkey)
            if mesh_id is None:
                t_mesh = time.perf_counter()
                mesh = _export_mesh(obj_eval, buf, textures, vc_attr_name)
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
            obj_desc = {
                "mesh": mesh_id,
                "material": slot_mat_ids[0],
                "transform": _matrix_to_row_major(inst.matrix_world),
            }
            if len(slot_mat_ids) > 1:
                obj_desc["materials"] = slot_mat_ids
            # Cycles' object Ray Visibility → Shadow. The source object carries
            # the flag (instances inherit it). Classroom's paper-lantern shades
            # use this to let the inner point light escape the drum; without
            # the hint the drum occludes NEE and we get characteristic
            # rectangular shadow-cutouts on the ceiling.
            if not getattr(obj, "visible_shadow", True):
                obj_desc["cast_shadow"] = False
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
