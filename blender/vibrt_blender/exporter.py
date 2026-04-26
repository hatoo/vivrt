"""Export the evaluated Blender scene to (json, bin, texture_arrays) for the
in-process FFI render path.

Blender uses right-handed, Z-up, metres. vibrt uses the same, so no
axis conversion is required — only matrix transposition (Blender matrices are
column-major via mathutils, our JSON schema expects row-major 16-float arrays).

The bin contains mesh / index / vertex-color / colour-graph LUT blobs.
Texture pixels travel as a separate `list[ndarray]` and are referenced
from `TextureDesc.array_index` in the JSON; they cross PyO3 as
`Vec<PyBuffer<f32>>` instead of being concatenated into the bin.
"""

from __future__ import annotations

import json
import math
import time

import bpy

from . import _log, hair_export, material_export
from ._log import log as _emit


def _matrix_to_row_major(m) -> list[float]:
    return [m[i][j] for i in range(4) for j in range(4)]


def _try_emissive_quad_as_rect_light(obj, obj_eval, mat_params):
    """If obj_eval is a single-quad mesh with pure-emissive material, return
    an AreaRect light JSON dict. Otherwise None.

    Pure emissive = emission > 0 with base_color ≈ 0, no transmission, no
    textures/graphs tinting the lobes. vibrt only importance-samples light
    objects; emissive-mesh surfaces are hit via BSDF sampling alone, so
    small bright panels (BMW27's hood light, classroom blackboard lights)
    under-contribute to indirect illumination. Converting them to
    `area_rect` routes them through NEE, recovering the several-× drop in
    ambient lift vs Cycles.

    The returned dict's `camera_visible` field mirrors Blender's
    Ray Visibility → Camera flag so that authored "invisible light panel"
    meshes (BMW27's `Light`) still contribute via NEE but aren't drawn as
    a bright rectangle in the final image.
    """
    if mat_params is None:
        return None
    emission = mat_params.get("emission", [0, 0, 0])
    if not any(e > 1e-6 for e in emission):
        return None
    bc = mat_params.get("base_color", [0, 0, 0])
    if max(bc) > 0.01:
        return None
    if mat_params.get("transmission", 0.0) > 0.0:
        return None
    for k in ("base_color_tex", "color_graph", "normal_tex", "roughness_tex",
              "metallic_tex", "bump_tex"):
        if k in mat_params:
            return None

    mesh = obj_eval.data
    if len(mesh.polygons) != 1:
        return None
    poly = mesh.polygons[0]
    if len(poly.vertices) != 4:
        return None

    verts = [mesh.vertices[vi].co.copy() for vi in poly.vertices]
    # Treat the quad's edge[0]=v1-v0 and edge[3]=v3-v0 as the U/V axes.
    # Assumes a reasonably rectangular quad — sheared/twisted ones fold to
    # an approximate axis-aligned rect, which is fine for light sampling
    # purposes (the emission and area are preserved).
    u_edge = verts[1] - verts[0]
    v_edge = verts[3] - verts[0]
    if u_edge.length < 1e-6 or v_edge.length < 1e-6:
        return None

    mw = obj_eval.matrix_world
    m3 = mw.to_3x3()
    center_w = mw @ ((verts[0] + verts[1] + verts[2] + verts[3]) * 0.25)
    u_edge_w = m3 @ u_edge
    v_edge_w = m3 @ v_edge
    size_u = u_edge_w.length
    size_v = v_edge_w.length
    if size_u < 1e-6 or size_v < 1e-6:
        return None
    u_axis_w = u_edge_w / size_u
    v_axis_w = v_edge_w / size_v
    # Cross product gives the face normal; consistent with Blender's poly
    # winding for a single 4-vert face. No need for the inverse-transpose
    # trick (that only matters for shading normals under non-uniform scale).
    normal_w = u_axis_w.cross(v_axis_w).normalized()

    # vibrt's area_rect transform: columns are (U, V, N) axes in world space
    # with translation at rect centre. scene_loader.rs reconstructs the rect
    # by transforming (±0.5·size_u, ±0.5·size_v, 0) corners.
    transform = [
        u_axis_w.x, v_axis_w.x, normal_w.x, center_w.x,
        u_axis_w.y, v_axis_w.y, normal_w.y, center_w.y,
        u_axis_w.z, v_axis_w.z, normal_w.z, center_w.z,
        0.0, 0.0, 0.0, 1.0,
    ]

    # Pass emission through 1:1. scene_loader.rs computes
    # stored_emission = color · power / (area · π); choosing power = area·π
    # makes the coefficient 1 so the rect radiance equals the source
    # material's emission radiance.
    area = size_u * size_v
    return {
        "type": "area_rect",
        "transform": transform,
        "size": [size_u, size_v],
        "color": list(emission),
        "power": area * math.pi,
        "camera_visible": 1 if getattr(obj, "visible_camera", True) else 0,
        # Blender emissive meshes radiate from both faces by default; a
        # one-sided rect here would black-out whichever face Blender's
        # vertex winding happens to orient away from the scene.
        "two_sided": 1,
    }


from .bin_writer import BinWriter  # re-export for the rest of this module


def _find_displacement(obj_eval, writer, textures):
    """Return (tex_id, strength) for the first material slot's Displacement
    output that drives a single TexImage; (None, 0.0) otherwise.

    Walks through bakeable nodes between Displacement.Height and the TexImage
    so that classroom-style chains
    (Displacement ← Math(scale) ← Bump ← TexImage) resolve correctly. The
    chain's colour transforms bake into the height texture's pixels; any
    scalar scale found on Math / Bump multiplies into `strength`.
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
        tag = f"{obj_eval.name!r}/{mat.name!r}"
        strength = 1.0
        if src.bl_idname == "ShaderNodeDisplacement":
            if "Scale" in src.inputs:
                if src.inputs["Scale"].is_linked:
                    _emit(
                        f"[vibrt] warn: {tag}: Displacement Scale input is "
                        f"linked — using constant default ({src.inputs['Scale'].default_value})"
                    )
                strength = float(src.inputs["Scale"].default_value)
            height_sock = src.inputs.get("Height")
            if height_sock is None or not height_sock.is_linked:
                _emit(
                    f"[vibrt] warn: {tag}: Displacement.Height is not connected "
                    f"— displacement ignored"
                )
                continue
            tex_id, extra = _resolve_height_texture(
                height_sock, writer, textures, tag
            )
            if tex_id is not None:
                return tex_id, strength * extra
        elif src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
            tex_id = material_export.export_image_texture(
                src.image, writer, textures, "linear"
            )
            return tex_id, strength
        else:
            _emit(
                f"[vibrt] warn: {tag}: Displacement output driven by "
                f"{src.bl_idname} (expected ShaderNodeDisplacement or "
                f"ShaderNodeTexImage) — displacement ignored"
            )
    return None, 0.0


_PROCEDURAL_HEIGHT_NODES = frozenset((
    "ShaderNodeTexNoise",
    "ShaderNodeTexWave",
    "ShaderNodeTexVoronoi",
    "ShaderNodeTexMusgrave",
    "ShaderNodeTexMagic",
    "ShaderNodeTexWhiteNoise",
    "ShaderNodeTexChecker",
    "ShaderNodeTexGradient",
))


def _resolve_height_texture(sock, writer, textures, tag, depth: int = 0):
    """Walk a Height-style chain (Math, Bump) back to a TexImage.

    Returns (tex_id, extra_strength). `extra_strength` multiplies into the
    Displacement.Scale so a Bump.Distance or Math multiplier between the
    TexImage and the Displacement node doesn't get silently dropped.
    `(None, 1.0)` when the chain can't be resolved.
    """
    if sock is None or not sock.is_linked or depth > 4:
        return None, 1.0
    up = sock.links[0].from_node
    if up.bl_idname == "ShaderNodeBump":
        # Bump wraps a Height texture with Strength * Distance. Fold both
        # constants into the returned strength multiplier and continue.
        height = up.inputs.get("Height")
        strength = _const_f(up.inputs.get("Strength"), 1.0)
        distance = _const_f(up.inputs.get("Distance"), 1.0)
        tex_id, extra = _resolve_height_texture(
            height, writer, textures, tag, depth + 1
        )
        if tex_id is None:
            return None, 1.0
        return tex_id, extra * strength * distance
    if up.bl_idname in _PROCEDURAL_HEIGHT_NODES:
        # Pure procedural height (noise, wave, ...) — the exporter has no
        # per-object displacement baker, so the best we can do is drop the
        # displacement. Silent: we already flag this to the artist as "no
        # image to export" rather than as an export error.
        return None, 1.0
    if up.bl_idname == "ShaderNodeMath":
        # Only commutative / chain-preserving ops fold cleanly into a scalar
        # multiplier on Displacement.Scale. MULTIPLY / ADD of a TexImage with
        # a constant is the common pattern (classroom's woodPlanks does
        # Math(TexImage, 0.1)); other ops would distort the signal.
        op = getattr(up, "operation", "MULTIPLY")
        inputs = list(up.inputs)
        chain_idx = next(
            (i for i, inp in enumerate(inputs) if inp.is_linked), -1
        )
        if chain_idx < 0:
            return None, 1.0
        if op == "MULTIPLY":
            others = [
                _const_f(inp, None) for i, inp in enumerate(inputs)
                if i != chain_idx
            ]
            if None in others:
                _emit(
                    f"[vibrt] warn: {tag}: Displacement.Height Math({op}) "
                    f"has non-constant side — displacement ignored"
                )
                return None, 1.0
            k = 1.0
            for v in others:
                k *= v
            tex_id, extra = _resolve_height_texture(
                inputs[chain_idx], writer, textures, tag, depth + 1
            )
            if tex_id is None:
                return None, 1.0
            return tex_id, extra * k
        # ADD of a constant shifts the height reference; the renderer already
        # treats 0.5 as the neutral point, so swallow the shift silently as
        # long as the other side is constant and the texture can be found.
        if op == "ADD":
            others = [
                _const_f(inp, None) for i, inp in enumerate(inputs)
                if i != chain_idx
            ]
            if None in others:
                return None, 1.0
            return _resolve_height_texture(
                inputs[chain_idx], writer, textures, tag, depth + 1
            )
        _emit(
            f"[vibrt] warn: {tag}: Displacement.Height Math op={op!r} not "
            f"handled (only MULTIPLY / ADD) — displacement ignored"
        )
        return None, 1.0
    # Fall through to the material-export chain walker: TexImage, plus any
    # colour transform between it and here (HueSaturation / RGBCurve / Gamma /
    # Invert / Clamp / ColorRamp / BrightContrast) bakes into the height
    # texture's pixels. The chain walker also follows through Mapping /
    # Mix / Separate nodes as pass-throughs, matching colour-chain semantics.
    img, chain = material_export._socket_linked_image_with_chain(sock)
    if img is None:
        _emit(
            f"[vibrt] warn: {tag}: Displacement.Height driven by "
            f"{up.bl_idname} — couldn't resolve an image, displacement ignored"
        )
        return None, 1.0
    return material_export.export_image_texture(
        img, writer, textures, "linear", chain=chain
    ), 1.0


def _const_f(sock, default):
    if sock is None:
        return default
    if sock.is_linked:
        return default
    try:
        return float(sock.default_value)
    except (TypeError, ValueError):
        return default


def _export_mesh(
    obj_eval, writer, textures: list, vc_attr_name: str | None = None
) -> dict | None:
    """Export a mesh + per-triangle material indices.

    Returns a dict suitable for a `MeshDesc` — always includes `material_indices`
    when there's more than one slot, otherwise the entry is omitted.

    Vertex / normal / UV / index arrays are built through `foreach_get` into
    numpy buffers. A Python-level per-triangle loop was the dominant export
    cost on non-trivial meshes (seconds per 100k-tri object vs milliseconds).

    If `vc_attr_name` is given and the mesh has a matching colour attribute,
    emits f32x3 per emitted vertex so the renderer can interpolate it against
    a material with `use_vertex_color=True`.
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
            except RuntimeError as exc:
                _emit(
                    f"[vibrt] warn: mesh {obj_eval.name!r}: "
                    f"calc_normals_split failed ({exc}) — using existing split normals"
                )
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
            "vertices": writer.write_f32(positions),
            "normals": writer.write_f32(split_normals),
            "uvs": writer.write_f32(uvs) if uvs is not None else None,
            "indices": writer.write_u32(indices),
        }
        if num_slots > 1:
            tri_mat_idx = np.empty(ntri, dtype=np.int32)
            mesh.loop_triangles.foreach_get("material_index", tri_mat_idx)
            desc["material_indices"] = writer.write_u32(tri_mat_idx)
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
                    desc["vertex_colors"] = writer.write_f32(
                        np.ascontiguousarray(per_corner.reshape(-1))
                    )
        disp_tex, disp_strength = _find_displacement(obj_eval, writer, textures)
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
    if cam.dof.use_dof:
        # Cycles convention: aperture radius [m] = (focal_length_mm / 1000) / (2 * fstop)
        fstop = cam.dof.aperture_fstop
        lens_radius = (cam.lens * 1e-3) / (2.0 * fstop) if fstop > 0.0 else 0.0
        # If focus_object is set, Cycles uses the distance from the camera to the
        # object along the camera's view axis (-Z in camera space); otherwise the
        # numeric focus_distance field. focus_distance defaults to 0 when an
        # object is in use, so falling back to it would collapse DOF to ~0m.
        focus_obj = cam.dof.focus_object
        if focus_obj is not None:
            cam_mat = cam_obj.matrix_world
            cam_pos = cam_mat.translation
            forward = -cam_mat.col[2].to_3d().normalized()  # camera looks down -Z
            delta = focus_obj.matrix_world.translation - cam_pos
            focal_distance = max(delta.dot(forward), 0.001)
        else:
            focal_distance = max(cam.dof.focus_distance, 0.001)
    else:
        lens_radius = 0.0
        focal_distance = 1.0
    return {
        "transform": matrix,
        "fov_y_rad": fov_y,
        "lens_radius": lens_radius,
        "focal_distance": focal_distance,
        # Cycles' "Clip Start" — primary rays don't intersect anything
        # closer than this. Lone_monk has 1.12m clip_start to skip a
        # close-up alcove wall the camera is positioned just behind;
        # without honouring this the wall fills the entire frame.
        "clip_start": float(cam.clip_start),
        "clip_end": float(cam.clip_end),
        # Lens shift (in sensor units; multiply by 2/sensor for NDC
        # offset). Also drives Cycles' lone_monk view: shift_y=0.07
        # tilts the rendered frame slightly upward so the arches above
        # the alcove come into view.
        "shift_x": float(cam.shift_x),
        "shift_y": float(cam.shift_y),
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
        _emit(
            f"[vibrt] warn: light {light.name!r}: Surface driven by "
            f"{src.bl_idname} (expected ShaderNodeEmission) — using "
            f"light.energy × light.color unchanged"
        )
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
            _emit(
                f"[vibrt] warn: light {light.name!r}: Emission Strength is "
                f"driven by {up.bl_idname} (expected ShaderNodeLightFalloff) "
                f"— using strength=1.0"
            )
            strength = 1.0
    else:
        strength = float(s_sock.default_value)
    if c_sock is None:
        color = (1.0, 1.0, 1.0)
    elif c_sock.is_linked:
        # Resolve via the same constant-folding machinery used for material
        # Emission.Color, so a Blackbody / Attribute / procedural drop-in
        # collapses to its neutral constant instead of being thrown away.
        rgb = material_export._socket_constant_rgb(c_sock)
        if rgb is None:
            _emit(
                f"[vibrt] warn: light {light.name!r}: Emission Color chain "
                f"isn't foldable to a constant — using light.color unchanged"
            )
            color = (1.0, 1.0, 1.0)
        else:
            color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    else:
        cv = c_sock.default_value
        color = (float(cv[0]), float(cv[1]), float(cv[2]))
    return strength, color


def _export_light(obj, writer, textures: list) -> dict | None:
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
            _emit(
                f"[vibrt] warn: area light {obj.name!r}: shape={light.shape!r} "
                f"not supported (only SQUARE/RECTANGLE) — approximating as SQUARE"
            )
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
            # Cycles' Ray Visibility → Camera flag. Junk Shop authors area
            # lights that are supposed to illuminate the scene but never
            # appear as bright rectangles in the final frame (e.g. `Area.027`,
            # a 9×10 m panel over the roof). Without this flag the default
            # on the Rust side (camera_visible=1) draws them as geometry.
            "camera_visible": 1 if getattr(obj, "visible_camera", True) else 0,
        }
    _emit(
        f"[vibrt] warn: light {obj.name!r}: type={light.type!r} not supported "
        f"(only POINT/SUN/SPOT/AREA) — light dropped"
    )
    return None


# Cache populated by `prebake_sky_envmaps_for_world` (called from
# `RenderEngine.update()` — see engine.py). `_export_world` consumes
# entries here when it sees a `ShaderNodeTexSky` node.
#
# Why a cache? The bake uses `bpy.ops.render.render` against a temp scene,
# which Blender silently turns into an all-zero render when invoked from
# inside `RenderEngine.render()`. Calling it from `update()` instead
# (which Blender invokes BEFORE handing control to render()) works fine.
# The exporter, which runs inside render(), reads the cache to get
# already-baked pixels.
_SKY_BAKE_CACHE: dict[str, tuple] = {}


def _sky_node_cache_key(world, sky_node) -> str:
    """Stable key tying a baked envmap to (world, all Sky Texture controls).

    Anything that affects the rendered sky pixels participates in the key
    so a scene that flips `sun_elevation` or `turbidity` between renders
    re-bakes correctly.
    """
    return (
        f"__sky__{world.name}__{sky_node.sky_type}__1024x512__"
        f"se{sky_node.sun_elevation:.5f}_"
        f"sr{sky_node.sun_rotation:.5f}_"
        f"t{sky_node.turbidity:.4f}_"
        f"a{sky_node.altitude:.2f}_"
        f"ad{sky_node.air_density:.4f}_"
        f"dd{sky_node.aerosol_density:.4f}_"
        f"od{sky_node.ozone_density:.4f}_"
        f"sd{int(sky_node.sun_disc)}_"
        f"ss{sky_node.sun_size:.6f}_"
        f"si{sky_node.sun_intensity:.4f}_"
        f"ga{sky_node.ground_albedo:.4f}"
    )


def _world_sky_node(world):
    """Return the ShaderNodeTexSky driving Background.Color, or None."""
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    out = world.node_tree.nodes.get("World Output")
    if out is None:
        out = next(
            (n for n in world.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputWorld"),
            None,
        )
    if out is None:
        return None
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return None
    bg = surf.links[0].from_node
    if bg.bl_idname != "ShaderNodeBackground":
        return None
    col = bg.inputs.get("Color")
    if col is None or not col.is_linked:
        return None
    src = col.links[0].from_node
    if src.bl_idname != "ShaderNodeTexSky":
        return None
    return src


def prebake_sky_envmaps_for_world(world, w: int = 1024, h: int = 512) -> None:
    """If `world`'s Background.Color is driven by a ShaderNodeTexSky, bake
    it into `_SKY_BAKE_CACHE`. Safe to call repeatedly: returns early on
    cache hit. Must be called from `RenderEngine.update()` (or other
    non-render context) — see _SKY_BAKE_CACHE comment.

    No-op for worlds that don't use Sky Texture, so engine.py can call
    this unconditionally before every render.
    """
    sky = _world_sky_node(world)
    if sky is None:
        return
    key = _sky_node_cache_key(world, sky)
    if key in _SKY_BAKE_CACHE:
        return
    t0 = time.perf_counter()
    try:
        baked = _bake_sky_world_to_pixels(world, w=w, h=h)
    except Exception as ex:
        _emit(
            f"[vibrt] warn: world {world.name!r}: failed to bake "
            f"ShaderNodeTexSky ({sky.sky_type}) via Cycles in update(): "
            f"{ex} — world background will fall back to constant"
        )
        return
    if baked is None:
        _emit(
            f"[vibrt] warn: world {world.name!r}: Sky Texture "
            f"({sky.sky_type}) bake produced no pixels — world background "
            f"will fall back to constant"
        )
        return
    # Split the sun out of the bake: super-bright pixels (the disc) are
    # impossible to importance-sample cleanly through an equirect — they
    # blow up NEE/BSDF MIS into vertical firefly streaks. Cycles handles
    # this internally by emitting the sun as a dedicated delta-direction
    # sun light *in addition* to the procedural sky; we mirror that.
    rgb, bw, bh = baked
    sun_light = _extract_sun_from_bake_inplace(rgb, bw, bh, sky, world.name)
    _SKY_BAKE_CACHE[key] = (rgb, bw, bh, sun_light)
    _emit(
        f"[vibrt] world {world.name!r}: pre-baked ShaderNodeTexSky "
        f"({sky.sky_type}) to a {baked[1]}x{baked[2]} envmap in "
        f"{time.perf_counter() - t0:.2f}s"
    )


def _extract_sun_from_bake_inplace(rgb, w: int, h: int, sky_node, world_name: str):
    """If the bake contains a sun disc (a tight cluster of very-high-
    radiance pixels), clip those pixels down to roughly diffuse-sky levels
    and return a `sun` LightDesc carrying the missing flux as a separate
    delta light.

    Modifies `rgb` in place. Returns None when no sun is present (sky_disc
    off, or scattered radiance not concentrated enough to justify a delta
    light).

    Why bother splitting? Cycles' Sky Texture renders the sun disc *both*
    into the procedural envmap and as a dedicated delta sun light, so its
    importance sampler can find the sun directly. We can't replicate the
    delta path through a baked equirect — the disc ends up as a few
    super-bright pixels that punch through MIS as vertical fireflies. The
    fix mirrors Cycles: strip the disc out of the bake and emit it as a
    separate sun light. The bake retains the diffuse sky only.
    """
    import numpy as np
    import math
    if not getattr(sky_node, "sun_disc", False):
        return None
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    lum_max = float(lum.max())
    # Diffuse Nishita sky tops out around 10-30. Anything noticeably
    # brighter than the brightest bright-but-diffuse pixel is the sun
    # disc; threshold at 5x the 99.5th percentile gives a robust cutoff
    # without per-scene tuning. Bail when the bake is uniform-ish (no
    # sun_disc, overcast sky, hand-edited HDRI).
    # Diffuse Nishita sky tops out around 5-15 W/sr/m². The disc punches
    # through that by 4-5 orders of magnitude. Threshold at 3x the 99th
    # percentile catches the disc's wing without clipping bright horizon
    # scattering; floor at 30 so a hand-edited HDRI without a tight disc
    # doesn't accidentally have its highlights pulled out as a "sun".
    p99 = float(np.percentile(lum, 99.0))
    threshold = max(30.0, p99 * 3.0)
    if lum_max < threshold:
        return None
    bright_mask = lum > threshold
    n_bright = int(bright_mask.sum())
    if n_bright == 0:
        return None
    ys, xs = np.where(bright_mask)
    # (theta, phi) of bright pixels. Equirect convention: y=0 is +Z (zenith).
    theta = (ys.astype(np.float32) + 0.5) * (math.pi / h)
    phi = (xs.astype(np.float32) + 0.5) * (2.0 * math.pi / w)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    # Per-pixel solid angle (equirect Jacobian).
    omega = (2.0 * math.pi * math.pi / (w * h)) * sin_t
    weights = lum[ys, xs] * omega
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return None
    dx = float((sin_t * cos_p * weights).sum() / total_weight)
    dy = float((sin_t * sin_p * weights).sum() / total_weight)
    dz = float((cos_t * weights).sum() / total_weight)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-6:
        return None
    sun_dir = (dx / norm, dy / norm, dz / norm)
    # Flux above the diffuse-sky threshold (per channel, in irradiance
    # units = radiance × sr).
    rgb_bright = rgb[ys, xs, :]
    lum_bright = lum[ys, xs]
    flux_rgb = ((rgb_bright * (lum_bright - threshold)[:, None]
                 / np.maximum(lum_bright[:, None], 1e-6))
                * omega[:, None]).sum(axis=0)
    # Replace bright pixels with the threshold luminance (preserving each
    # pixel's hue) so the bake's residual max equals `threshold`.
    scale = threshold / np.maximum(lum_bright, 1e-6)
    rgb[ys, xs, :] = rgb_bright * scale[:, None]
    sun_size = max(float(sky_node.sun_size), 1e-4)
    flux_max = max(float(flux_rgb[0]), float(flux_rgb[1]),
                   float(flux_rgb[2]), 1e-6)
    color = [float(c) / flux_max for c in flux_rgb]
    # No cap on the sun light's irradiance — letting the disc deliver its
    # full Nishita flux (~120-200 W/m²) is what gives sun-lit surfaces
    # their proper contrast against the diffuse-sky shadows. The fireflies
    # that prompted earlier capping came from the *bake* itself sampling
    # the disc through MIS; pulling the disc out of the bake (residual sky
    # tops out at `threshold` above) is what handles that.
    _emit(
        f"[vibrt] world {world_name!r}: split {n_bright} sun-disc pixels "
        f"(peak L={lum_max:.0f}, residual sky max={threshold:.0f}) → "
        f"sun light dir={sun_dir} E={flux_max:.2f} "
        f"angle={math.degrees(sun_size):.2f}°"
    )
    return {
        "type": "sun",
        # Direction the light *travels*: opposite of the "comes-from" vector.
        "direction": [-sun_dir[0], -sun_dir[1], -sun_dir[2]],
        "color": color,
        "strength": flux_max,
        "angle_rad": sun_size,
    }


def clear_sky_bake_cache() -> None:
    """Drop all cached envmap pixel buffers. Called between renders so
    edits to the Sky Texture are picked up; the cache is per-process and
    would otherwise stay alive forever."""
    _SKY_BAKE_CACHE.clear()


def _bake_sky_world_to_pixels(world, w: int = 1024, h: int = 512):
    """Bake the world's procedural Sky Texture (Nishita / Hosek-Wilkie /
    Preetham) into a numpy `(h, w, 3)` float32 linear-RGB array.

    `ShaderNodeTexSky` has no Python eval entry-point, so we ask Cycles to
    do the work: a temp scene with a PANO/EQUIRECTANGULAR camera at the
    origin renders the world to a 32-bit-float OpenEXR, which we load
    back in and read into numpy. The temp scene contains zero geometry,
    so this is fast (~0.25 s for 1024×512 in a CPU build) and doesn't
    depend on the user's scene contents. Cycles' equirectangular camera
    uses the same spherical convention as ShaderNodeTexEnvironment, so
    the result plugs into the existing envmap branch with
    `rotation_z_rad = 0.0`, matching how an authored env-map .exr would
    behave. (`sun_rotation` and the rest of the Sky Texture's controls
    are already folded into the rendered pixels.)

    Returns `(rgb, w, h)` where `rgb` is contiguous float32 of shape
    `(h, w, 3)` in linear RGB. Returns `None` on any failure path; the
    caller is expected to log a warning and fall back.
    """
    import os
    import tempfile
    import numpy as np

    tmp = bpy.data.scenes.new("__vibrt_sky_bake_scene")
    tmp.world = world
    tmp.render.engine = "CYCLES"
    # Sky Texture is smooth — modest spp + no adaptive sampling so the
    # disc gets sampled enough to land in the centroid we extract; at
    # adaptive Cycles cuts off as soon as the diffuse sky converges,
    # leaving the disc dim.
    try:
        tmp.cycles.samples = 64
        tmp.cycles.use_denoising = False
        tmp.cycles.use_adaptive_sampling = False
    except Exception:
        pass
    tmp.render.resolution_x = int(w)
    tmp.render.resolution_y = int(h)
    tmp.render.resolution_percentage = 100
    tmp.render.image_settings.file_format = 'OPEN_EXR'
    tmp.render.image_settings.color_mode = 'RGB'
    tmp.render.image_settings.color_depth = '32'
    # Don't punch through the world background — we want to capture it.
    tmp.render.film_transparent = False
    # Output linear-light pixels straight from the renderer (no display
    # transform / Filmic curve / exposure tweaks). The downstream texture
    # pipeline marks the texture `colorspace="linear"` and the engine
    # samples it as raw radiance.
    try:
        tmp.view_settings.view_transform = 'Raw'
        tmp.view_settings.look = 'None'
        tmp.view_settings.exposure = 0.0
        tmp.view_settings.gamma = 1.0
    except Exception:
        pass

    # PANO/EQUIRECTANGULAR camera at world origin. Cycles uses the
    # "physics" sphere convention: u sweeps phi around the up-axis (Z),
    # v sweeps theta from north (+Z) to south (-Z). Camera default
    # orientation (0,0,0 Euler) has -Z forward / +Y up — exactly the
    # frame ShaderNodeTexEnvironment expects, so no rotation is needed
    # here.
    cam_data = bpy.data.cameras.new("__vibrt_sky_bake_cam")
    cam_data.type = 'PANO'
    # `panorama_type` lives on Camera directly in Blender 4.x+ (it used to be
    # under `cam_data.cycles.panorama_type` in 3.x, but that attribute is gone
    # by 4.x — silently falling back to the default `FISHEYE_EQUISOLID` here
    # would render only the camera's front hemisphere into a circular disc,
    # leaving ~80% of the equirect grid black and dropping the sun disc
    # entirely. So we set it on the top-level Camera and treat any failure as
    # a hard error (the addon's required Blender version is 4.x+).
    cam_data.panorama_type = 'EQUIRECTANGULAR'
    cam = bpy.data.objects.new("__vibrt_sky_bake_cam", cam_data)
    cam.location = (0.0, 0.0, 0.0)
    # Rotate the camera so its local +Y (Cycles' equirect "latitude axis")
    # points at world -Z. Without this rotation the bake puts the sky on
    # the equator (Cycles' default panorama latitude axis is camera +Y,
    # which under identity rotation = world +Y → not the zenith axis the
    # kernel expects). Rotating by -90° around world X takes camera +Y
    # → world -Z (the nadir direction), which after Cycles' bottom-up
    # storage and our `world_background()` indexing leaves world +Z at
    # buffer y=0 — i.e. the zenith maps to image-y=0 just like the
    # kernel's `theta = v * π` mapping reads it.
    import math
    cam.rotation_euler = (-math.pi / 2.0, 0.0, 0.0)
    tmp.collection.objects.link(cam)
    tmp.camera = cam

    # Render to a unique temp .exr; we read the pixels back into numpy
    # and unlink.
    fd, exr_path = tempfile.mkstemp(prefix="vibrt_sky_", suffix=".exr")
    os.close(fd)
    tmp.render.filepath = exr_path
    img_loaded = None
    rgb = None
    try:
        # Pass the scene name explicitly so the operator targets our temp
        # scene instead of bpy.context.scene (we don't switch the window
        # context, so this is the only way to direct the render). Note:
        # Blender silently produces an all-zero image when this runs from
        # inside `RenderEngine.render()` — the bake MUST be called from
        # `RenderEngine.update()` (or another non-rendering context). The
        # caller (engine.py) drives that.
        bpy.ops.render.render(scene=tmp.name, write_still=True)
        img_loaded = bpy.data.images.load(exr_path)
        # Mark linear so colourspace conversion is skipped on read.
        img_loaded.colorspace_settings.name = 'Non-Color'
        iw, ih = int(img_loaded.size[0]), int(img_loaded.size[1])
        if iw == 0 or ih == 0:
            return None
        ch = int(img_loaded.channels) or 4
        # Read RGBA into a flat numpy buffer, then drop the alpha channel
        # — envmaps are always 3-channel radiance for vibrt. We do this
        # while the .exr is still on disk, because Blender's image data
        # is lazily decoded; deleting the file before this read can yield
        # an "image has no data" error on subsequent renders.
        flat = np.empty(iw * ih * ch, dtype=np.float32)
        img_loaded.pixels.foreach_get(flat)
        arr = flat.reshape((ih, iw, ch))
        rgb = np.ascontiguousarray(arr[..., :3], dtype=np.float32)
    finally:
        try:
            if img_loaded is not None:
                bpy.data.images.remove(img_loaded, do_unlink=True)
        except Exception:
            pass
        # Tear down the temp scene and camera.
        try:
            bpy.data.objects.remove(cam, do_unlink=True)
        except Exception:
            pass
        try:
            bpy.data.cameras.remove(cam_data, do_unlink=True)
        except Exception:
            pass
        try:
            bpy.data.scenes.remove(tmp, do_unlink=True)
        except Exception:
            pass
        try:
            os.unlink(exr_path)
        except Exception:
            pass
    if rgb is None:
        return None
    return rgb, int(w), int(h)


def _export_world(world, writer, textures: list, lights_json: list | None = None) -> dict:
    if world is None or not world.use_nodes:
        col = list(world.color)[:3] if world else [0.05, 0.05, 0.05]
        return {"type": "constant", "color": col, "strength": 1.0}

    out = world.node_tree.nodes.get("World Output")
    if out is None:
        out = next(
            (n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"),
            None,
        )
    if out is None:
        _emit(
            f"[vibrt] warn: world {world.name!r}: no ShaderNodeOutputWorld — "
            f"world treated as black"
        )
        return {"type": "constant", "color": [0, 0, 0], "strength": 0.0}
    if not out.inputs["Surface"].is_linked:
        _emit(
            f"[vibrt] warn: world {world.name!r}: World Output Surface is "
            f"not connected — world treated as black"
        )
        return {"type": "constant", "color": [0, 0, 0], "strength": 0.0}

    src = out.inputs["Surface"].links[0].from_node
    if src.bl_idname == "ShaderNodeBackground":
        strength_sock = src.inputs["Strength"]
        if strength_sock.is_linked:
            _emit(
                f"[vibrt] warn: world {world.name!r}: Background Strength is "
                f"linked — using constant default ({strength_sock.default_value})"
            )
        strength = strength_sock.default_value
        color_sock = src.inputs["Color"]
        col = list(color_sock.default_value)[:3]
        if color_sock.is_linked:
            linked = color_sock.links[0].from_node
            if linked.bl_idname == "ShaderNodeTexEnvironment":
                if linked.image is None:
                    _emit(
                        f"[vibrt] warn: world {world.name!r}: "
                        f"ShaderNodeTexEnvironment has no image assigned — "
                        f"falling back to constant background"
                    )
                else:
                    tex_id = material_export.export_image_texture(
                        linked.image, writer, textures, colorspace="linear"
                    )
                    rotation_z_rad = 0.0
                    return {
                        "type": "envmap",
                        "texture": tex_id,
                        "rotation_z_rad": rotation_z_rad,
                        "strength": float(strength),
                    }
            elif linked.bl_idname == "ShaderNodeTexSky":
                # Procedural sky (Nishita / Hosek-Wilkie / Preetham). The
                # actual bake — a tiny equirectangular Cycles render of
                # the sky world — runs in `RenderEngine.update()` (see
                # engine.py and `prebake_sky_envmaps_for_world`); it
                # CANNOT run inside the exporter because we're already
                # inside `RenderEngine.render()`, and Cycles silently
                # produces an all-zero image for nested renders. Here we
                # just look up the cached pixels and route them through
                # `_PreBakedTexture` so they land on the world dict as a
                # regular envmap.
                key = _sky_node_cache_key(world, linked)
                baked = _SKY_BAKE_CACHE.get(key)
                if baked is None:
                    _emit(
                        f"[vibrt] warn: world {world.name!r}: "
                        f"ShaderNodeTexSky ({linked.sky_type}) was not "
                        f"pre-baked (key={key!r}) — falling back to "
                        f"constant background. Was the engine.update() "
                        f"hook invoked?"
                    )
                else:
                    rgb, bw, bh, sun_light = baked
                    pb = material_export._PreBakedTexture(
                        rgb=rgb, w=bw, h=bh, cache_key=key,
                    )
                    tex_id = material_export.export_image_texture(
                        pb, writer, textures, colorspace="linear"
                    )
                    # Sun disc lives as a separate delta light next to the
                    # diffuse-sky envmap (Cycles works the same way).
                    # Strength is scaled by the world's Background.Strength
                    # so dimming the world also dims the sun.
                    if sun_light is not None and lights_json is not None:
                        scaled = dict(sun_light)
                        scaled["strength"] = (
                            scaled["strength"] * float(strength)
                        )
                        lights_json.append(scaled)
                    return {
                        "type": "envmap",
                        "texture": tex_id,
                        "rotation_z_rad": 0.0,
                        "strength": float(strength),
                    }
            elif linked.bl_idname == "ShaderNodeRGB":
                # Constant colour picker — read its Color output.
                col = list(linked.outputs["Color"].default_value)[:3]
            elif linked.bl_idname == "ShaderNodeBlackbody":
                # Deterministic RGB from a constant temperature — fold to
                # the exact linear colour Cycles would produce. We don't
                # accept lossy chains (textures, procedurals): the renderer
                # has no way to carry per-pixel variation into a constant
                # background, and silently averaging would hide the drop.
                temp_sock = linked.inputs.get("Temperature")
                if temp_sock is not None and not temp_sock.is_linked:
                    col = material_export._blackbody_to_linear_rgb(
                        float(temp_sock.default_value)
                    )
                else:
                    _emit(
                        f"[vibrt] warn: world {world.name!r}: Blackbody "
                        f"Temperature is linked — using constant default"
                    )
            else:
                _emit(
                    f"[vibrt] warn: world {world.name!r}: Background Color "
                    f"driven by {linked.bl_idname} (expected "
                    f"ShaderNodeTexEnvironment) — using constant default"
                )
        return {"type": "constant", "color": col, "strength": float(strength)}

    _emit(
        f"[vibrt] warn: world {world.name!r}: World Output Surface driven by "
        f"{src.bl_idname} (expected ShaderNodeBackground) — world treated as black"
    )
    return {"type": "constant", "color": [0, 0, 0], "strength": 0.0}


def _export_world_volume(world) -> dict | None:
    """Resolve the optional ShaderNodeOutputWorld → Volume socket into a
    VolumeParams dict, or None when the atmosphere is clear (current
    behaviour). Atmospheric haze / fog scenes get one global homogeneous
    volume that fills the entire world; the renderer always treats it as
    sitting at the bottom of the volume stack.
    """
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    out = next(
        (n for n in world.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output),
        None,
    )
    if out is None:
        out = next(
            (n for n in world.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputWorld"),
            None,
        )
    if out is None:
        return None
    return material_export._resolve_volume(out.inputs.get("Volume"))


def export_scene_to_memory(
    depsgraph: bpy.types.Depsgraph,
    texture_pct: int | None = None,
) -> tuple[str, list, list]:
    """Build (scene.json, mesh_blobs, texture_arrays) in RAM.

    All three feed `vibrt_native.render` directly:
    - `json_str`: the scene description. Every BlobRef has been replaced
      with an `array_index: u32` referencing the corresponding entry in
      one of the two array lists.
    - `mesh_blobs`: contiguous numpy arrays for mesh / index / vertex-
      colour / colour-graph LUT data. Mixed dtypes (f32 / u32). PyO3
      borrows each via the buffer protocol; the Rust loader picks the
      right `bytemuck` cast based on what the field declares.
    - `texture_arrays`: per-texture float32 RGBA buffers, kept separate
      from mesh blobs so the Rust side can take them as a typed
      `Vec<PyBuffer<f32>>` for the GPU upload path.
    """
    writer = BinWriter()
    scene_dict = _export_into(depsgraph, writer, texture_pct)
    return (
        json.dumps(scene_dict, indent=2),
        writer.mesh_blobs,
        writer.texture_arrays,
    )


def _export_into(
    depsgraph: bpy.types.Depsgraph,
    writer: "BinWriter",
    texture_pct: int | None,
) -> dict:
    """Drive the export, writing blobs through `writer`. Returns the
    JSON-serialisable scene dict.
    """
    t0 = time.perf_counter()
    material_export.reset_stats()
    mesh_s = 0.0
    hair_s = 0.0
    iter_body_s = 0.0  # cumulative time spent inside the depsgraph loop body
    alloc_s = 0.0      # begin_export / end_export buffer init
    pre_loop_s = 0.0   # everything before the depsgraph iteration
    post_loop_s = 0.0  # everything after the loop end
    t_pre = time.perf_counter()
    slow_meshes: list[tuple] = []
    scene = depsgraph.scene_eval
    rd = scene.render
    width = int(rd.resolution_x * rd.resolution_percentage / 100.0)
    height = int(rd.resolution_y * rd.resolution_percentage / 100.0)
    aspect = width / height if height > 0 else 1.0

    textures: list[dict] = []
    materials: list[dict] = []
    material_index: dict[str, int] = {}
    # Materials whose Base Color is driven by a ShaderNodeAttribute stash the
    # attribute name on their exported dict under `_vertex_color_attr`. We pop
    # it so the material JSON stays clean and record it per-material so meshes
    # that reference the material can emit the matching vertex-colour blob.
    material_vc_attr: dict[str, str] = {}

    meshes: list[dict] = []
    objects: list[dict] = []
    mesh_cache: dict[str, int] = {}

    cam_obj = None
    lights_json: list[dict] = []

    # The reusable foreach_get buffer is now grown lazily on first use, so we
    # no longer pre-walk `bpy.data.images`. That walk used to cost ~7 s on
    # junk_shop because reading `img.size` materialises image headers from
    # disk for every datablock — including unused ones (orphan textures,
    # the Render Result / Viewer Node images Blender always keeps around).
    t_alloc = time.perf_counter()
    material_export.begin_export(texture_pct=texture_pct)
    alloc_s += time.perf_counter() - t_alloc

    try:
        # Single-arm `try/finally` so we always reach `material_export.end_export()`.
        if True:

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
                exported = material_export.export_material(mat, writer, textures)
                vc_attr = exported.pop("_vertex_color_attr", None)
                if vc_attr:
                    material_vc_attr[mat.name] = vc_attr
                mid = len(materials)
                materials.append(exported)
                material_index[mat.name] = mid
                return mid

            pre_loop_s = time.perf_counter() - t_pre
            t_loop_start = time.perf_counter()
            for inst in depsgraph.object_instances:
                t_iter = time.perf_counter()
                obj = inst.object
                if inst.is_instance:
                    obj_eval = obj
                else:
                    obj_eval = obj.evaluated_get(depsgraph)

                if obj_eval.type == "MESH":
                    slot_mat_ids: list[int] = []
                    vc_attr_name: str | None = None
                    if obj_eval.material_slots:
                        for slot in obj_eval.material_slots:
                            slot_mat_ids.append(resolve_material(slot.material))
                            if slot.material is not None and vc_attr_name is None:
                                vc_attr_name = material_vc_attr.get(slot.material.name)
                    if not slot_mat_ids:
                        slot_mat_ids = [resolve_material(None)]

                    # Promote single-quad pure-emissive meshes to area_rect lights
                    # so NEE can find them. Runs before the visible_camera skip
                    # because Blender scenes routinely use camera-hidden meshes as
                    # emissive panels (BMW27's `Light`) — without promotion their
                    # contribution is entirely lost. The rect honours the original
                    # visible_camera flag so hidden panels stay hidden from the
                    # final image.
                    if len(slot_mat_ids) == 1 and not inst.is_instance:
                        rect = _try_emissive_quad_as_rect_light(
                            obj, obj_eval, materials[slot_mat_ids[0]]
                        )
                        if rect is not None:
                            lights_json.append(rect)
                            continue

                    # Skip meshes that Cycles hides from camera — they're light
                    # portals (e.g. classroom's `windows` dayLight_portal) meant
                    # to inject light but never appear as geometry on the
                    # rendered image. Without this check the emissive portal
                    # bleeds over the ceiling / walls.
                    if not getattr(obj, "visible_camera", True):
                        continue

                    mkey = f"{obj_eval.data.name}#{obj_eval.name}#vc={vc_attr_name}"
                    mesh_id = mesh_cache.get(mkey)
                    if mesh_id is None:
                        t_mesh = time.perf_counter()
                        mesh = _export_mesh(obj_eval, writer, textures, vc_attr_name)
                        dt_mesh = time.perf_counter() - t_mesh
                        mesh_s += dt_mesh
                        if mesh is None:
                            continue
                        if dt_mesh > 0.25:
                            ntri = writer.mesh_blobs[mesh["indices"]].nbytes // 12  # 4 bytes/u32, 3 per tri
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

                    # Hair particle systems on this object: tessellate to
                    # ribbons + emit a sibling mesh/object. Skipped for
                    # instance copies — each particle system is owned by the
                    # source object and we'd otherwise duplicate the hair
                    # for every linked instance.
                    if not inst.is_instance and any(
                        ps.settings.type == "HAIR"
                        for ps in obj_eval.particle_systems
                    ):
                        t_hair = time.perf_counter()
                        hair_pair = hair_export.export_hair(
                            obj,
                            obj_eval,
                            scene,
                            writer,
                            resolve_material,
                            obj_eval.name,
                        )
                        hair_s += time.perf_counter() - t_hair
                        if hair_pair is not None:
                            hair_mesh, hair_obj = hair_pair
                            hair_mesh_id = len(meshes)
                            meshes.append(hair_mesh)
                            hair_obj["mesh"] = hair_mesh_id
                            objects.append(hair_obj)
                elif obj_eval.type == "LIGHT":
                    light = _export_light(obj_eval, writer, textures)
                    if light is not None:
                        lights_json.append(light)
                elif obj_eval.type == "CAMERA" and obj_eval == scene.camera.evaluated_get(
                    depsgraph
                ):
                    cam_obj = obj_eval
                iter_body_s += time.perf_counter() - t_iter
            loop_total_s = time.perf_counter() - t_loop_start
            depsgraph_iter_s = max(0.0, loop_total_s - iter_body_s)
            t_post = time.perf_counter()

            if cam_obj is None and scene.camera is not None:
                cam_obj = scene.camera.evaluated_get(depsgraph)
            if cam_obj is None:
                raise RuntimeError("No active camera in scene")

            t_world = time.perf_counter()
            world = _export_world(scene.world, writer, textures, lights_json)
            world_volume = _export_world_volume(scene.world)
            world_s = time.perf_counter() - t_world

            bin_size = sum(b.nbytes for b in writer.mesh_blobs)
    finally:
        t_dealloc = time.perf_counter()
        material_export.end_export()
        alloc_s += time.perf_counter() - t_dealloc

    bin_write_s = 0.0

    spp = max(1, int(scene.vibrt_spp))
    _emit(f"[vibrt] spp={spp}")
    scene_json = {
        "version": 1,
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
    if world_volume is not None:
        scene_json["world_volume"] = world_volume

    # JSON serialise + write happens in the caller now; keep the dump out of
    # this hot path so `_export_into` can be reused for both file and memory
    # call sites.
    json_dump_s = 0.0
    json_write_s = 0.0

    post_loop_s = time.perf_counter() - t_post
    dt = time.perf_counter() - t0
    stats = material_export.pop_stats()
    material_s = stats["material_self_s"]
    texture_s = stats["texture_self_s"]
    pixel_read_s = stats["pixel_read_s"]
    bake_chain_s = stats["bake_chain_s"]
    accounted = (
        mesh_s + material_s + texture_s + pixel_read_s + bake_chain_s
        + hair_s + alloc_s + depsgraph_iter_s + pre_loop_s + post_loop_s
        + world_s + json_dump_s + bin_write_s + json_write_s
    )
    other_s = max(0.0, dt - accounted)
    # `loop_overhead` is the time spent inside the depsgraph loop body that
    # *isn't* mesh / hair / material / texture / pixel / bake. It's the
    # iteration scaffolding itself: obj.evaluated_get, material_slots access,
    # _try_emissive_quad_as_rect_light, dict appends, etc. Useful for
    # pinpointing whether a slow export is texture-bound vs scaffolding-bound.
    loop_children = (
        mesh_s + hair_s + material_s + texture_s + pixel_read_s + bake_chain_s
    )
    loop_overhead_s = max(0.0, iter_body_s - loop_children)
    tex_pct_str = (
        f", texture_pct={texture_pct}%"
        if texture_pct is not None and texture_pct != 100 else ""
    )
    _emit(
        f"[vibrt] export {dt:.2f}s "
        f"({len(meshes)} mesh, {len(objects)} obj, "
        f"{len(textures)} tex, {len(materials)} mat, "
        f"{bin_size/1024/1024:.1f}MB bin, {stats['pixel_bytes']/1024/1024:.1f}MB px"
        f"{tex_pct_str})"
    )
    _emit(
        f"[vibrt]   mesh={mesh_s:.2f}s  material={material_s:.2f}s  "
        f"texture={texture_s:.2f}s  pixel_read={pixel_read_s:.2f}s  "
        f"bake_chain={bake_chain_s:.2f}s"
    )
    _emit(
        f"[vibrt]   hair={hair_s:.2f}s  loop_overhead={loop_overhead_s:.2f}s  "
        f"depsgraph_iter={depsgraph_iter_s:.2f}s  alloc={alloc_s:.2f}s  "
        f"pre_loop={pre_loop_s:.2f}s  post_loop={post_loop_s:.2f}s  "
        f"world={world_s:.2f}s  other={other_s:.2f}s"
    )
    if slow_meshes:
        desc = ", ".join(f"{n}({t:.0f}k tri, {d:.2f}s)" for n, t, d in
                         ((nm, ntri / 1000.0, dd) for nm, ntri, dd in slow_meshes))
        _emit(f"[vibrt]   slow meshes: {desc}")
    if stats["slow_textures"]:
        desc = ", ".join(
            f"{n}({w}x{h}, {d:.2f}s)" for n, w, h, d in stats["slow_textures"]
        )
        _emit(f"[vibrt]   slow textures: {desc}")
    if stats["slow_materials"]:
        desc = ", ".join(f"{n}({d:.2f}s)" for n, d in stats["slow_materials"])
        _emit(f"[vibrt]   slow materials: {desc}")
    return scene_json
