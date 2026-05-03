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

from . import _log, hair_export, ies, material_export
from ._log import log as _emit


def _matrix_to_row_major(m) -> list[float]:
    return [m[i][j] for i in range(4) for j in range(4)]


def _approx_mesh_as_rect(mesh, m3, mw):
    """Build an oriented rect that approximates a multi-poly emissive mesh
    for NEE sampling. Returns (center_w, u_axis_w, v_axis_w, normal_w,
    size_u, size_v) or None if the mesh is degenerate.

    Strategy: walk every triangle, accumulate area-weighted centroid +
    normal in world space, then build orthonormal basis (U, V) in the
    plane perpendicular to the average normal. Set the rect size so its
    area equals the mesh's full surface area (two-sided emission keeps
    the integrated radiated power roughly equal to a thin emissive sheet
    of the same surface area).
    """
    from mathutils import Vector
    total_area = 0.0
    centroid = Vector((0.0, 0.0, 0.0))
    avg_n = Vector((0.0, 0.0, 0.0))
    for poly in mesh.polygons:
        vs = [mw @ mesh.vertices[vi].co for vi in poly.vertices]
        # Triangulate fan-style around vs[0].
        for i in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[i], vs[i + 1]
            cross = (b - a).cross(c - a)
            tri_area = cross.length * 0.5
            if tri_area < 1e-12:
                continue
            tri_n = cross.normalized()
            tri_centroid = (a + b + c) / 3.0
            total_area += tri_area
            centroid += tri_centroid * tri_area
            avg_n += tri_n * tri_area
    if total_area < 1e-12 or avg_n.length < 1e-9:
        return None
    centroid /= total_area
    normal_w = avg_n.normalized()
    # Pick a U axis that's perpendicular to normal_w. Use the world axis
    # least aligned with normal_w as a seed for stable orthonormalisation.
    seed = Vector((1.0, 0.0, 0.0)) if abs(normal_w.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    u_axis_w = (seed - normal_w * seed.dot(normal_w)).normalized()
    v_axis_w = normal_w.cross(u_axis_w).normalized()
    # Square rect with area = total mesh surface area. (For a flat mesh
    # this matches the mesh's projected area exactly; for a curved one it
    # over-estimates the visible cross-section but preserves total power.)
    side = math.sqrt(total_area)
    return centroid, u_axis_w, v_axis_w, normal_w, side, side


def _try_mixed_emissive_proxy(obj, obj_eval, mat_params):
    """For meshes whose material has both emission AND a non-emissive lobe
    (Principled BSDF with `emission_strength != 0` plus a non-zero
    base_color, or a MixShader of an emissive Principled and a Translucent
    fabric), return a `camera_visible=False` rect light dict that
    approximates the mesh's emission for NEE. The caller should keep the
    mesh in the geometry export so primary visibility, BSDF reflections,
    and the diffuse half of the surface are unchanged.

    The kernel's "add emission only on bounce==0 or last_specular" gate
    keeps double-counting under control: at diffuse bounces the BSDF-
    sampled emission is suppressed, so the rect proxy is the only
    emission contributor — exactly what we want. At specular bounces
    (mirror reflection of the lamp) the mesh's emission is what gets
    added, and the proxy doesn't fire because NEE skips delta lobes.

    Returns None when emission is effectively zero or the mesh is
    geometrically degenerate. Pure-emissive meshes are still routed
    through `_try_emissive_quad_as_rect_light` first (full replacement)
    — this function is the second-chance path for everything else.
    """
    if mat_params is None:
        return None
    emission = mat_params.get("emission", [0, 0, 0])
    if not any(e > 1e-6 for e in emission):
        return None
    # Skip materials whose emission comes from a per-pixel texture — a
    # uniform rect proxy would average over the texture's spatial detail
    # and inject a uniform glow where the texture has cold areas.
    if "emission_tex" in mat_params:
        return None

    mesh = obj_eval.data
    if len(mesh.polygons) == 0:
        return None
    mw = obj_eval.matrix_world
    m3 = mw.to_3x3()
    rect_info = _approx_mesh_as_rect(mesh, m3, mw)
    if rect_info is None:
        return None
    center_w, u_axis_w, v_axis_w, normal_w, size_u, size_v = rect_info

    transform = [
        u_axis_w.x, v_axis_w.x, normal_w.x, center_w.x,
        u_axis_w.y, v_axis_w.y, normal_w.y, center_w.y,
        u_axis_w.z, v_axis_w.z, normal_w.z, center_w.z,
        0.0, 0.0, 0.0, 1.0,
    ]
    area = size_u * size_v
    return {
        "type": "area_rect",
        "transform": transform,
        "size": [size_u, size_v],
        "color": list(emission),
        "power": area * math.pi,
        # Critical: NEE-only proxy. Camera and specular rays still see
        # the original mesh, so emission isn't double-counted.
        "camera_visible": 0,
        "two_sided": 1,
    }


def _try_emissive_quad_as_rect_light(obj, obj_eval, mat_params):
    """If obj_eval is a pure-emissive mesh, return an AreaRect light JSON
    dict that vibrt's NEE can sample. Otherwise None.

    Single-quad meshes use the quad's edges as the rect's U/V axes
    (preserving the source rectangle exactly). Multi-poly meshes fall
    through `_approx_mesh_as_rect`: a square rect placed at the
    area-weighted centroid, oriented along the area-weighted average
    normal, with side = √(total surface area). The mesh is then dropped
    from the geometry export (the rect takes its place in the scene), so
    the only visible footprint is the rect itself — fine for emissive
    panels of any topology, less faithful for curved/long emissive
    bodies whose silhouette matters.

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
              "metallic_tex", "bump_tex", "emission_tex"):
        if k in mat_params:
            # An emission texture (cloud panel, billboard, screen) carries
            # per-pixel detail the constant-radiance area_rect can't
            # reproduce. Promoting it to a rect light would render it as a
            # uniform glowing rectangle and drop the cloud / image.
            return None

    mesh = obj_eval.data
    if len(mesh.polygons) == 0:
        return None
    mw = obj_eval.matrix_world
    m3 = mw.to_3x3()
    if len(mesh.polygons) == 1 and len(mesh.polygons[0].vertices) == 4:
        # Exact single-quad fast path: take the quad's two edges as the
        # rect's U/V axes (preserves shape exactly).
        poly = mesh.polygons[0]
        verts = [mesh.vertices[vi].co.copy() for vi in poly.vertices]
        u_edge = verts[1] - verts[0]
        v_edge = verts[3] - verts[0]
        if u_edge.length < 1e-6 or v_edge.length < 1e-6:
            return None
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
    else:
        # Multi-poly fallback: approximate the mesh's emission with a rect
        # placed at the area-weighted centroid, oriented along the
        # area-weighted average normal, sized so its area matches the
        # mesh's actual surface area / 2 (two-sided, so each face emits π·Le
        # and total radiated power matches a thin emissive sheet of the
        # same surface area). Accuracy degrades on highly anisotropic
        # shapes (long tubes, ribbons), but it's a strict improvement over
        # the BSDF-only path that emissive multi-poly geometry was on
        # before — NEE on shaded surfaces nearby now finds the lamp.
        rect_info = _approx_mesh_as_rect(mesh, m3, mw)
        if rect_info is None:
            return None
        center_w, u_axis_w, v_axis_w, normal_w, size_u, size_v = rect_info

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
        # Cycles only moves real geometry when `displacement_method` is
        # DISPLACEMENT or BOTH; the default BUMP just feeds the
        # Displacement node into the shader's normal channel and leaves
        # the mesh unchanged. Apply the same gate so we don't distort
        # terrains whose author meant the height map purely as a bump
        # cue (pabellon's `grass` material on `tree_scatter_plane` is
        # exactly this — scale=0.1 in local space × the plane's 44.94
        # world scale would otherwise smear the ground by ~4.5 m).
        disp_method = (getattr(mat, "displacement_method", None)
                       or getattr(getattr(mat, "cycles", None),
                                  "displacement_method", "BUMP"))
        if str(disp_method).upper() == "BUMP":
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
    # Derive the actual visible frame from Blender's view_frame(), which
    # already folds in sensor_fit (AUTO/HORIZONTAL/VERTICAL), sensor size,
    # lens, render resolution, pixel aspect, and lens shift. The four
    # returned corners live in camera-local space at z=-clip_start; we
    # normalise to the z=-1 plane and read off half-extents + center.
    #
    # Doing it this way is mandatory for sensor_fit=AUTO scenes where the
    # image aspect crosses the sensor aspect (lone_monk: 1440x1080 image
    # with a 36x24 sensor → AUTO resolves to HORIZONTAL, so cam.angle_y
    # is NOT the actual vertical FOV — using it shrinks the frame by
    # ~12%). The center cx/cy encodes shift in the same view-space units,
    # which avoids re-deriving Blender's "shift is fraction of the larger
    # sensor dimension" convention by hand (off by sw/sh on lone_monk).
    frame = cam.view_frame(scene=scene)
    norm = [(p.x / -p.z, p.y / -p.z) for p in frame]
    xs = [n[0] for n in norm]
    ys = [n[1] for n in norm]
    half_w = (max(xs) - min(xs)) * 0.5
    half_h = (max(ys) - min(ys)) * 0.5
    cx = (max(xs) + min(xs)) * 0.5
    cy = (max(ys) + min(ys)) * 0.5
    fov_y = 2.0 * math.atan(half_h)
    # Device pinhole: px = 2*idx.x/dim.x - 1 + 2*shift_x, then dir = U*px +
    # V*py + W where U = right*half_w (and half_w = half_h*aspect on the
    # device), V = up*half_h. Centre pixel (px=2*shift_x) lands in
    # view-space at right*half_w*(2*shift_x), so to put the centre at
    # (cx, cy) we send shift = c / (2 * half_extent).
    shift_x_ndc = cx / (2.0 * half_w) if half_w > 0.0 else 0.0
    shift_y_ndc = cy / (2.0 * half_h) if half_h > 0.0 else 0.0
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
        # NDC shift: half the centre offset of the view frame, expressed
        # as a fraction of the half-extent on each axis. The device adds
        # `2 * shift` to the NDC coordinate before projecting, so this
        # form lands a centre pixel at (cx, cy) in view-space.
        "shift_x": float(shift_x_ndc),
        "shift_y": float(shift_y_ndc),
    }


def _resolve_ies_table(ies_node) -> dict | None:
    """Parse a `ShaderNodeTexIES` into the JSON `ies` block expected by
    `LightDesc.{Point,Spot,AreaRect}.ies` (`scene_format::IesProfile`).

    Returns None — with a warning — if the node's IES content is
    unreadable: external file missing or unreadable, internal Text
    datablock empty, or the IES content fails the LM-63 parse.
    """
    text = None
    label = ies_node.name
    if ies_node.mode == "INTERNAL" and ies_node.ies is not None:
        try:
            text = ies_node.ies.as_string()
        except Exception as ex:
            _emit(
                f"[vibrt] warn: IES node {label!r}: failed to read internal "
                f"Text datablock {ies_node.ies.name!r}: {ex}"
            )
            return None
    elif ies_node.mode == "EXTERNAL" and ies_node.filepath:
        path = bpy.path.abspath(ies_node.filepath)
        try:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()
        except OSError as ex:
            _emit(
                f"[vibrt] warn: IES node {label!r}: cannot open external file "
                f"{path!r}: {ex} — light will not get an IES profile"
            )
            return None
    else:
        _emit(
            f"[vibrt] warn: IES node {label!r}: mode={ies_node.mode!r} but "
            f"neither internal Text nor external filepath has IES content — "
            f"light will not get an IES profile"
        )
        return None
    try:
        table = ies.parse_ies(text)
    except Exception as ex:
        _emit(
            f"[vibrt] warn: IES node {label!r}: parse error: {ex} — "
            f"light will not get an IES profile"
        )
        return None
    return {
        "thetas_deg": list(table.thetas_deg),
        "phis_deg": list(table.phis_deg),
        "candelas": list(table.candelas),
        "peak_candela": float(table.peak_candela),
        # Solid-angle integral of the [0, 1]-normalised table. The
        # renderer divides 4π by this to compensate for the IES being
        # concentrated into a beam: an isotropic Point uses
        # `power / (4π)` so flux is preserved; an IES Point should use
        # `power / integral_norm` for the same reason. Sent across as
        # a precomputed scalar so the GPU doesn't have to integrate.
        "integral_norm": float(table.integral_normalised()),
    }


def _follow_through_group(start_node, start_socket_name):
    """Walk into a `ShaderNodeGroup` to find what drives its
    `start_socket_name` output (the canonical IES idiom is to wrap the
    Emission + IES Texture inside a NodeGroup, exposed as Color /
    Strength inputs and an Emission output). Returns the (inner_node,
    inner_input_socket_name_for_strength_lookup, group_input_strength_default,
    group_input_color_default) tuple, or None if the group's contents
    don't match the expected pattern.
    """
    if start_node.bl_idname != "ShaderNodeGroup" or start_node.node_tree is None:
        return None
    tree = start_node.node_tree
    out = next(
        (n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"
         and getattr(n, "is_active_output", True)),
        None,
    )
    if out is None:
        return None
    sock = out.inputs.get(start_socket_name)
    if sock is None:
        # Some Light Outputs use "Surface", group output uses "Emission".
        # Try the first connected input.
        for s in out.inputs:
            if s.is_linked:
                sock = s
                break
        if sock is None:
            return None
    if not sock.is_linked:
        return None
    inner = sock.links[0].from_node
    return inner


def _light_node_emission(light) -> tuple[float, tuple[float, float, float], dict | None]:
    """Return `(strength, color, ies)` contributed by the light's shader
    tree, as multipliers on top of `light.energy`/`light.color`.

    Cycles evaluates the light's shader tree and multiplies its output
    with `light.energy * light.color`. Three patterns we recognise:

    1. `Output ← Emission(Strength=K, Color=C)` — classic case;
       returns (K, C, None).
    2. `Output ← Emission(Strength=Light Falloff(Strength=K), Color=C)`
       — classroom's blackBoard_light; vibrt does its own physical
       falloff so we just fold K and ignore the Falloff output choice.
    3. `Output ← NodeGroup(Color=C, Strength=K) → ... → Emission` with
       an inner `ShaderNodeTexIES` driving `Emission.Strength` — the
       canonical IES Texture idiom (Blender's official ies_light test
       scene uses exactly this). The IES table becomes an attached
       photometric profile on the resulting light; `K` becomes the
       outer multiplier on light.energy. Without recognising this
       pattern, the IES factor is dropped entirely (the wrapping
       NodeGroup confuses the simple Emission-only walk).

    Returns `(1.0, (1, 1, 1), None)` when there's no usable Emission —
    the callers then see `light.energy`/`light.color` unchanged.
    """
    if not light.use_nodes or light.node_tree is None:
        return 1.0, (1.0, 1.0, 1.0), None
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
        return 1.0, (1.0, 1.0, 1.0), None
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return 1.0, (1.0, 1.0, 1.0), None
    src = surf.links[0].from_node

    # Pattern 3: NodeGroup wrapping IES Texture. Detect by descending
    # through the group, finding the inner Emission, and inspecting its
    # Strength chain for a ShaderNodeTexIES.
    if src.bl_idname == "ShaderNodeGroup":
        inner = _follow_through_group(src, "Emission")
        if inner is not None and inner.bl_idname == "ShaderNodeEmission":
            s_sock = inner.inputs.get("Strength")
            c_sock = inner.inputs.get("Color")
            ies_table = None
            ies_strength_mult = 1.0
            color_mult = (1.0, 1.0, 1.0)
            if s_sock is not None and s_sock.is_linked:
                up = s_sock.links[0].from_node
                if up.bl_idname == "ShaderNodeTexIES":
                    ies_table = _resolve_ies_table(up)
                    # IES Texture's own Strength input is a 1.0-default
                    # multiplier. Inside a NodeGroup it's typically driven
                    # by Group Input → outer Group node; resolve to a
                    # constant by querying the outer Group input.
                    ies_strength_mult = _socket_value_via_group(
                        up.inputs.get("Strength"), src, default=1.0
                    )
            if c_sock is not None:
                color_mult = _socket_color_via_group(c_sock, src,
                                                    default=(1.0, 1.0, 1.0))
            if ies_table is not None:
                return ies_strength_mult, color_mult, ies_table
        # Fall through with a warning — group present but not the IES
        # idiom we recognise.
        _emit(
            f"[vibrt] warn: light {light.name!r}: Surface driven by "
            f"{src.bl_idname} (NodeGroup but not the IES Texture idiom) "
            f"— using light.energy × light.color unchanged"
        )
        return 1.0, (1.0, 1.0, 1.0), None

    if src.bl_idname != "ShaderNodeEmission":
        _emit(
            f"[vibrt] warn: light {light.name!r}: Surface driven by "
            f"{src.bl_idname} (expected ShaderNodeEmission) — using "
            f"light.energy × light.color unchanged"
        )
        return 1.0, (1.0, 1.0, 1.0), None

    # Patterns 1 and 2: bare Emission, possibly with Light Falloff or
    # IES Texture (no NodeGroup) on Strength.
    s_sock = src.inputs.get("Strength")
    c_sock = src.inputs.get("Color")
    ies_table = None
    if s_sock is None:
        strength = 1.0
    elif s_sock.is_linked:
        up = s_sock.links[0].from_node
        if up.bl_idname == "ShaderNodeLightFalloff":
            inner = up.inputs.get("Strength")
            strength = float(inner.default_value) if inner is not None and not inner.is_linked else 1.0
        elif up.bl_idname == "ShaderNodeTexIES":
            # Bare IES Texture (not wrapped in a NodeGroup). Its Strength
            # input acts as a multiplier on the IES table; bake it into
            # the light's overall strength.
            ies_table = _resolve_ies_table(up)
            ies_in = up.inputs.get("Strength")
            strength = (float(ies_in.default_value)
                        if ies_in is not None and not ies_in.is_linked
                        else 1.0)
        else:
            _emit(
                f"[vibrt] warn: light {light.name!r}: Emission Strength is "
                f"driven by {up.bl_idname} (expected ShaderNodeLightFalloff "
                f"or ShaderNodeTexIES) — using strength=1.0"
            )
            strength = 1.0
    else:
        strength = float(s_sock.default_value)
    if c_sock is None:
        color = (1.0, 1.0, 1.0)
    elif c_sock.is_linked:
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
    return strength, color, ies_table


def _socket_value_via_group(socket, group_node, *, default: float) -> float:
    """Resolve a scalar socket inside a NodeGroup by hopping out to the
    enclosing group node's matching input, used by the IES idiom: the
    `IES Texture.Strength` input is wired to a `Group Input`, whose
    real value lives on the outer Group node's input slot."""
    if socket is None:
        return default
    if not socket.is_linked:
        return float(socket.default_value)
    up = socket.links[0].from_node
    if up.bl_idname == "NodeGroupInput":
        outer = group_node.inputs.get(socket.links[0].from_socket.name)
        if outer is None:
            return default
        if outer.is_linked:
            # Could chain further; for simplicity just take its default.
            return default
        return float(outer.default_value)
    return default


def _socket_color_via_group(socket, group_node, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Same hop for an RGB Color socket — the NodeGroup's `Color` input
    is exposed and the outer Group node's matching input carries the
    artist-set value."""
    if socket is None:
        return default
    if not socket.is_linked:
        cv = socket.default_value
        return (float(cv[0]), float(cv[1]), float(cv[2]))
    up = socket.links[0].from_node
    if up.bl_idname == "NodeGroupInput":
        outer = group_node.inputs.get(socket.links[0].from_socket.name)
        if outer is None:
            return default
        if outer.is_linked:
            rgb = material_export._socket_constant_rgb(outer)
            if rgb is None:
                return default
            return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        cv = outer.default_value
        return (float(cv[0]), float(cv[1]), float(cv[2]))
    return default


def _export_light(obj, writer, textures: list) -> dict | None:
    light = obj.data
    node_strength, node_color, ies_table = _light_node_emission(light)
    energy = float(light.energy) * node_strength
    col = [light.color[0] * node_color[0],
           light.color[1] * node_color[1],
           light.color[2] * node_color[2]]
    mw = _matrix_to_row_major(obj.matrix_world)
    if light.type == "POINT":
        position = [obj.matrix_world[i][3] for i in range(3)]
        # Pull the 3×3 rotation out of the world matrix so the kernel
        # can sample the IES table in the light's local frame. Without
        # IES the rotation is invisible (Point is isotropic) but we
        # always emit it so the schema stays uniform.
        m = obj.matrix_world
        light_rotation = [
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
        ]
        out = {
            "type": "point",
            "position": position,
            "color": col,
            "power": energy,
            "radius": max(light.shadow_soft_size, 0.005),
            "light_rotation": light_rotation,
        }
        if ies_table is not None:
            out["ies"] = ies_table
        return out
    if light.type == "SUN":
        # Blender sun: local -Z is the direction photons travel (i.e.
        # the forward / away-from-sun direction in world). Export that;
        # `scene_loader.rs` negates to the toward-sun direction the
        # kernel ultimately uses as `wi`.
        direction = [-obj.matrix_world[i][2] for i in range(3)]
        return {
            "type": "sun",
            "direction": direction,
            "color": col,
            "strength": energy,
            "angle_rad": float(light.angle),
        }
    if light.type == "SPOT":
        out = {
            "type": "spot",
            "transform": mw,
            "color": col,
            "power": energy,
            "cone_rad": float(light.spot_size) * 0.5,
            "blend": float(light.spot_blend),
        }
        if ies_table is not None:
            out["ies"] = ies_table
        return out
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
        out = {
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
        if ies_table is not None:
            out["ies"] = ies_table
        return out
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


def _world_generic_cache_key(world) -> str:
    """Cache key for a "bake the entire world" envmap. Used when
    `_export_world` can't recognise the world's shader graph (MixShader,
    nested colour math, etc.). The cache is cleared between renders so
    we just key on the world name — graph mutations within one render
    are unsupported anyway.
    """
    return f"__world_full__{world.name}__1024x512"


def _sky_node_alone_cache_key(world, sky_node) -> str:
    """Cache key for a per-Sky-node bake (a Sky Texture inside a
    MixShader, baked in isolation so option-A's Mixed envmap path can
    sample each layer at native resolution without paying for a
    combined-world bake)."""
    sun_rot = float(getattr(sky_node, "sun_rotation", 0.0))
    sun_elev = float(getattr(sky_node, "sun_elevation", 0.0))
    sun_size = float(getattr(sky_node, "sun_size", 0.0))
    sun_disc = bool(getattr(sky_node, "sun_disc", False))
    sky_type = getattr(sky_node, "sky_type", "?")
    return (f"__sky_alone__{world.name}__{sky_node.name}__{sky_type}__"
            f"{sun_rot}__{sun_elev}__{sun_size}__{sun_disc}__1024x512")


def _detect_mix_world_camera_ray_split(world):
    """Detect `MixShader(Background_a, Background_b, fac=Light Path.Is Camera Ray)`.

    This is the canonical archiviz idiom for separating a high-strength
    ambient lighting envmap from a low-strength backplate visible to
    the camera (and to specular reflections). The MixShader's factor is
    `is_camera_ray`, so:
      - fac=0 (non-camera rays: NEE / indirect / diffuse) → input 1 (lighting)
      - fac=1 (camera + specular-chain rays)              → input 2 (camera)

    Returns `(bg_lighting, bg_camera)` matching that mapping, or None
    if any of the topology requirements aren't met. The caller then
    bakes each branch separately.
    """
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    out = world.node_tree.nodes.get("World Output") or next(
        (n for n in world.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputWorld"),
        None,
    )
    if out is None:
        return None
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return None
    src = surf.links[0].from_node
    if src.bl_idname != "ShaderNodeMixShader":
        return None
    fac_sock = src.inputs[0]
    if not fac_sock.is_linked:
        return None
    fac_link = fac_sock.links[0]
    if fac_link.from_node.bl_idname != "ShaderNodeLightPath":
        return None
    if fac_link.from_socket.name != "Is Camera Ray":
        return None
    in1 = src.inputs[1]
    in2 = src.inputs[2]
    if not in1.is_linked or not in2.is_linked:
        return None
    bg_lighting = in1.links[0].from_node
    bg_camera = in2.links[0].from_node
    if bg_lighting.bl_idname != "ShaderNodeBackground":
        return None
    if bg_camera.bl_idname != "ShaderNodeBackground":
        return None
    return bg_lighting, bg_camera


def _world_branch_cache_key(world, branch_node, role: str) -> str:
    """Stable cache key for a Light-Path camera-ray-split branch bake.
    `role` is "lighting" or "camera" so the two halves stay distinct."""
    return f"__world_split__{world.name}__{role}__{branch_node.name}__2048x1024"


def _detect_mix_world(world):
    """If the world's Surface is `MixShader(Background, Background, fac=const)`,
    return `(bg_a, bg_b, fac)`. Otherwise None.

    Both Backgrounds must drive their Color from a recognised source
    (`ShaderNodeTexEnvironment` with an image, or `ShaderNodeTexSky`).
    Linked / non-foldable Mix factors fall through to None — the caller
    then takes the existing full-bake path.
    """
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    out = world.node_tree.nodes.get("World Output") or next(
        (n for n in world.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputWorld"),
        None,
    )
    if out is None:
        return None
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return None
    src = surf.links[0].from_node
    if src.bl_idname != "ShaderNodeMixShader":
        return None
    fac_sock = src.inputs[0]
    if fac_sock.is_linked:
        return None
    fac = float(fac_sock.default_value)
    in1 = src.inputs[1]
    in2 = src.inputs[2]
    if not in1.is_linked or not in2.is_linked:
        return None
    bg_a = in1.links[0].from_node
    bg_b = in2.links[0].from_node
    if bg_a.bl_idname != "ShaderNodeBackground":
        return None
    if bg_b.bl_idname != "ShaderNodeBackground":
        return None
    return bg_a, bg_b, fac


def _walk_world_for_sky_nodes(world):
    """Return every `ShaderNodeTexSky` reachable from the world's surface
    output. Used so the prebake step can render each Sky in isolation
    when it's wrapped inside a MixShader (pabellon's sunset world
    blends a Sky Texture with an HDRI backplate)."""
    if world is None or not world.use_nodes or world.node_tree is None:
        return []
    return [n for n in world.node_tree.nodes
            if n.bl_idname == "ShaderNodeTexSky"]


def _euler_xyz_to_matrix3(rx, ry, rz):
    """Compose a 3×3 row-major rotation matrix from intrinsic XYZ Euler
    angles (Blender's default for Mapping rotation). Matches
    `mathutils.Euler((rx,ry,rz), 'XYZ').to_matrix()`."""
    import math
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz · Ry · Rx (intrinsic XYZ = Rx then Ry then Rz applied to vector)
    # Combined manually:
    m = [
        cy * cz,        sx * sy * cz - cx * sz,  cx * sy * cz + sx * sz,
        cy * sz,        sx * sy * sz + cx * cz,  cx * sy * sz - sx * cz,
        -sy,            sx * cy,                  cx * cy,
    ]
    return [float(v) for v in m]


_IDENTITY_3X3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _mapping_rotation_matrix(mapping_node):
    """Extract a 3×3 row-major rotation matrix from a `ShaderNodeMapping`
    node, or identity if the node has any non-rotation transform we
    can't fold (location / non-unit scale)."""
    if mapping_node is None:
        return list(_IDENTITY_3X3)
    rot_sock = mapping_node.inputs.get("Rotation")
    if rot_sock is None:
        return list(_IDENTITY_3X3)
    rx, ry, rz = rot_sock.default_value
    return _euler_xyz_to_matrix3(float(rx), float(ry), float(rz))


def _try_export_layer_from_background(world, bg_node, writer, textures):
    """Convert one `ShaderNodeBackground` into the dict shape that
    `WorldDesc::Mixed.{a,b}` expects (texture id + 3×3 rotation +
    strength). Returns None when the background's source isn't one of
    the recognised options (TexEnvironment with image, TexSky cached)."""
    strength_sock = bg_node.inputs.get("Strength")
    if strength_sock is not None and strength_sock.is_linked:
        # A linked Strength can't be folded into a constant — bail to
        # the full-bake path so the artist's animation still reads.
        return None
    strength = (float(strength_sock.default_value)
                if strength_sock is not None else 1.0)
    color_sock = bg_node.inputs.get("Color")
    if color_sock is None or not color_sock.is_linked:
        return None
    src = color_sock.links[0].from_node
    if src.bl_idname in ("ShaderNodeTexEnvironment", "ShaderNodeTexImage"):
        # Both nodes feed an image into the Color socket of a Background.
        # ShaderNodeTexEnvironment is always equirect-style (sphere
        # mapping); ShaderNodeTexImage carries an explicit `projection`
        # that drives a different direction → UV function. Pabellon's
        # sunset world routes a regular JPG photo through TexImage with
        # `projection='FLAT'`, which is fundamentally different from
        # equirect and must not be sampled as one.
        if src.image is None:
            return None
        # Walk back through an optional Mapping → TexCoord chain to
        # collect the rotation. Cycles Mapping (type='Point' default)
        # applies `R · v` to the input vector, so the kernel — which
        # also pre-multiplies the sample direction by `rotation` —
        # gets the same `R` we extract here. Non-unit scale and
        # non-zero location aren't representable as a 3×3 alone and
        # are silently ignored (rare on world-shader Mapping nodes).
        rotation = list(_IDENTITY_3X3)
        vec_sock = src.inputs.get("Vector")
        if vec_sock is not None and vec_sock.is_linked:
            mapping = vec_sock.links[0].from_node
            if mapping.bl_idname == "ShaderNodeMapping":
                rotation = _mapping_rotation_matrix(mapping)
        # Linear colorspace for HDRI assets that ship as EXR / HDR;
        # SRGB images get linearised at scene-load.
        colorspace = (
            "srgb" if src.image.colorspace_settings.name.lower().startswith("srgb")
            else "linear"
        )
        # Determine the projection mode we tell the host rasteriser.
        if src.bl_idname == "ShaderNodeTexEnvironment":
            env_proj = getattr(src, "projection", "EQUIRECTANGULAR")
            if env_proj == "EQUIRECTANGULAR":
                projection = "equirect"
            else:
                _emit(
                    f"[vibrt] warn: world {world.name!r}: TexEnvironment "
                    f"projection={env_proj!r} unsupported — falling back "
                    f"to equirect (image '{src.image.name}' will sample "
                    f"with a different mapping than Cycles)"
                )
                projection = "equirect"
            extension = "repeat"
        else:
            img_proj = getattr(src, "projection", "FLAT")
            proj_map = {
                "FLAT": "flat",
                # SPHERE / TUBE / BOX would need extra device-side code;
                # fall back to FLAT with a visible warning so the bug is
                # locatable rather than silently mis-sampling as equirect.
            }
            if img_proj not in proj_map:
                _emit(
                    f"[vibrt] warn: world {world.name!r}: TexImage "
                    f"projection={img_proj!r} unsupported — treating "
                    f"as FLAT (image '{src.image.name}' may render "
                    f"differently than Cycles)"
                )
            projection = proj_map.get(img_proj, "flat")
            ext_map = {"REPEAT": "repeat", "EXTEND": "extend", "CLIP": "clip"}
            ext_raw = getattr(src, "extension", "REPEAT")
            extension = ext_map.get(ext_raw, "repeat")
        tex_id = material_export.export_image_texture(
            src.image, writer, textures, colorspace=colorspace
        )
        return {
            "texture": int(tex_id),
            "rotation": rotation,
            "strength": strength,
            "projection": projection,
            "extension": extension,
        }
    if src.bl_idname == "ShaderNodeTexSky":
        key = _sky_node_alone_cache_key(world, src)
        baked = _SKY_BAKE_CACHE.get(key)
        if baked is None:
            return None
        rgb, bw, bh, _sun = baked
        pb = material_export._PreBakedTexture(
            rgb=rgb, w=bw, h=bh, cache_key=key,
        )
        tex_id = material_export.export_image_texture(
            pb, writer, textures, colorspace="linear"
        )
        return {
            "texture": int(tex_id),
            "rotation": list(_IDENTITY_3X3),
            "strength": strength,
        }
    return None


def _try_export_mixed_world(world, mix_pat, writer, textures, lights_json):
    """Emit `WorldDesc::Mixed` for a world whose surface matches
    `MixShader(Background_a, Background_b, fac=const)`. Returns None
    when either layer can't be converted, so the caller falls back to
    the legacy combined-bake path."""
    bg_a, bg_b, fac = mix_pat
    layer_a = _try_export_layer_from_background(world, bg_a, writer, textures)
    if layer_a is None:
        return None
    layer_b = _try_export_layer_from_background(world, bg_b, writer, textures)
    if layer_b is None:
        return None
    # Sun lights extracted from per-Sky bakes ride alongside the Mixed
    # envmap; their strength was already scaled by the Sky's
    # Background.Strength when extracted, so no further scaling here.
    if lights_json is not None:
        for bg in (bg_a, bg_b):
            color_sock = bg.inputs.get("Color")
            if color_sock is None or not color_sock.is_linked:
                continue
            src = color_sock.links[0].from_node
            if src.bl_idname != "ShaderNodeTexSky":
                continue
            key = _sky_node_alone_cache_key(world, src)
            baked = _SKY_BAKE_CACHE.get(key)
            if baked is None:
                continue
            _rgb, _bw, _bh, sun_light = baked
            if sun_light is not None:
                lights_json.append(dict(sun_light))
    _emit(
        f"[vibrt] world {world.name!r}: emitting Mixed envmap "
        f"(a={'sky' if bg_a.inputs['Color'].links[0].from_node.bl_idname == 'ShaderNodeTexSky' else 'image'} "
        f"strength={layer_a['strength']:.2f}, "
        f"b={'sky' if bg_b.inputs['Color'].links[0].from_node.bl_idname == 'ShaderNodeTexSky' else 'image'} "
        f"strength={layer_b['strength']:.2f}, fac={fac:.3f})"
    )
    return {
        "type": "mixed",
        "a": layer_a,
        "b": layer_b,
        "fac": float(fac),
    }


def _try_export_split_world(world, split_pat, writer, textures):
    """Emit a Mixed envmap with `split_by_camera_ray=true` from a
    Light-Path-driven Mix Shader. Returns None when the per-branch
    bakes aren't in `_SKY_BAKE_CACHE` (e.g. update() didn't run, or a
    bake failed) so the caller falls back to the combined-bake path.

    Layer convention: `a` = lighting branch (drives the CDF / NEE),
    `b` = camera branch (camera-visible backplate). The bakes already
    fold Background.Strength + any upstream Mapping rotation into the
    pixels, so each layer carries strength=1.0 and identity rotation.
    """
    bg_lighting, bg_camera = split_pat
    key_a = _world_branch_cache_key(world, bg_lighting, "lighting")
    key_b = _world_branch_cache_key(world, bg_camera, "camera")
    baked_a = _SKY_BAKE_CACHE.get(key_a)
    baked_b = _SKY_BAKE_CACHE.get(key_b)
    if baked_a is None or baked_b is None:
        _emit(
            f"[vibrt] warn: world {world.name!r}: camera-ray split "
            f"detected but per-branch bakes were not cached "
            f"(lighting={'OK' if baked_a is not None else 'MISSING'}, "
            f"camera={'OK' if baked_b is not None else 'MISSING'}) — "
            f"falling back to legacy combined bake. Was the engine "
            f"update() hook invoked?"
        )
        return None
    rgb_a, aw, ah, _ = baked_a
    rgb_b, bw, bh, _ = baked_b
    pb_a = material_export._PreBakedTexture(rgb=rgb_a, w=aw, h=ah, cache_key=key_a)
    pb_b = material_export._PreBakedTexture(rgb=rgb_b, w=bw, h=bh, cache_key=key_b)
    tex_a = material_export.export_image_texture(
        pb_a, writer, textures, colorspace="linear"
    )
    tex_b = material_export.export_image_texture(
        pb_b, writer, textures, colorspace="linear"
    )
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    _emit(
        f"[vibrt] world {world.name!r}: emitting camera-ray split "
        f"envmap (lighting={aw}x{ah}, camera={bw}x{bh})"
    )
    return {
        "type": "mixed",
        "a": {
            "texture": tex_a,
            "rotation": identity,
            "strength": 1.0,
            "projection": "equirect",
            "extension": "repeat",
        },
        "b": {
            "texture": tex_b,
            "rotation": identity,
            "strength": 1.0,
            "projection": "equirect",
            "extension": "repeat",
        },
        "fac": 0.0,
        "split_by_camera_ray": True,
    }


def _world_needs_full_bake(world) -> bool:
    """True when the world's shader graph isn't a plain
    `ShaderNodeBackground -> Output.Surface`. Catches the
    `MixShader(Background_a, Background_b)` pattern that scenes like
    pabellon_barcelona use to blend a Sky Texture with a curve-shifted
    duplicate, and falls through any other unrecognised topology to a
    Cycles-evaluated equirect bake instead of the
    "treat-world-as-black" branch in `_export_world`.
    """
    if world is None or not world.use_nodes or world.node_tree is None:
        return False
    out = world.node_tree.nodes.get("World Output") or next(
        (n for n in world.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputWorld"),
        None,
    )
    if out is None:
        return False
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return False
    return surf.links[0].from_node.bl_idname != "ShaderNodeBackground"


def prebake_sky_envmaps_for_world(world, w: int = 2048, h: int = 1024) -> None:
    """Bake a world's environment into `_SKY_BAKE_CACHE` so the exporter
    has equirect pixels ready by the time it runs. Two paths share this
    cache:

    1. The world's Background.Color is driven by a ShaderNodeTexSky →
       bake the procedural sky and split out the sun disc as a delta
       sun light (firefly mitigation).
    2. The world's Surface is driven by anything else we can't fold
       host-side (MixShader, nested colour math, …) → bake the whole
       world via Cycles and use the result as a plain envmap.
       pabellon_barcelona's `mid day` / `sunset` / `night` worlds all
       hit this path because their surface is a MixShader of two
       Backgrounds, not a single Background.

    Safe to call repeatedly: returns early on cache hit. Must be called
    from `RenderEngine.update()` (or other non-render context) — Blender
    silently produces an all-zero image for nested renders started from
    inside `render()`.
    """
    sky = _world_sky_node(world)
    if sky is not None:
        _prebake_sky_texture(world, sky, w=w, h=h)
        return
    # Light-Path camera-ray split: bake each branch in isolation so the
    # camera-visible backplate and the lighting envmap stay separate
    # (Cycles' single-equirect render of this graph would only ever
    # capture is_camera_ray=1, dropping the lighting branch entirely).
    split = _detect_mix_world_camera_ray_split(world)
    if split is not None:
        bg_lighting, bg_camera = split
        _prebake_world_branch_alone(
            world, bg_lighting, w=w, h=h,
            cache_key=_world_branch_cache_key(world, bg_lighting, "lighting"),
            role="lighting",
        )
        _prebake_world_branch_alone(
            world, bg_camera, w=w, h=h,
            cache_key=_world_branch_cache_key(world, bg_camera, "camera"),
            role="camera",
        )
        return
    # If the world surface is `MixShader(Background_a, Background_b, fac=const)`,
    # bake each Sky Texture present in either branch separately so the
    # exporter's option-A path can emit `WorldDesc::Mixed` with per-layer
    # native-resolution textures. The full-world bake fallback below is
    # still kept for any topology we don't recognise.
    mix_pat = _detect_mix_world(world)
    if mix_pat is not None:
        for sky_node in _walk_world_for_sky_nodes(world):
            _prebake_sky_node_alone(world, sky_node, w=w, h=h)
        return
    if _world_needs_full_bake(world):
        _prebake_world_full(world, w=w, h=h)


def _prebake_world_branch_alone(world, branch_bg, *, w, h, cache_key, role):
    """Bake one Background branch of a Light-Path-driven world Mix in
    isolation. Clones the world, rewires the clone's World Output to the
    branch's Background output (bypassing the MixShader), bakes via
    `_bake_sky_world_to_pixels`, then removes the clone. The bake captures
    Background.Strength and any upstream Mapping rotation in the pixels,
    so the resulting EnvmapLayer carries strength=1.0 and identity rotation.
    """
    if cache_key in _SKY_BAKE_CACHE:
        return
    if branch_bg is None or branch_bg.bl_idname != "ShaderNodeBackground":
        return
    t0 = time.perf_counter()
    tmp_world = world.copy()
    tmp_world.name = f"__vibrt_world_branch_{role}_{branch_bg.name}"
    nt = tmp_world.node_tree
    out_n = nt.nodes.get("World Output") or next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld"),
        None,
    )
    if out_n is None:
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    # `world.copy()` preserves node names, so the cloned counterpart of
    # the branch's Background is reachable by name.
    tmp_branch = nt.nodes.get(branch_bg.name)
    if tmp_branch is None:
        _emit(
            f"[vibrt] warn: world {world.name!r}: failed to find cloned "
            f"branch node {branch_bg.name!r} in temp world — skipping "
            f"camera-ray split bake for {role!r}"
        )
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    surf_sock = out_n.inputs.get("Surface")
    if surf_sock is None:
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    # Clear any existing links on Surface (the cloned MixShader chain) so
    # the rewired Background drives the bake by itself.
    for lk in list(surf_sock.links):
        nt.links.remove(lk)
    nt.links.new(tmp_branch.outputs["Background"], surf_sock)
    try:
        baked = _bake_sky_world_to_pixels(tmp_world, w=w, h=h)
    except Exception as ex:
        _emit(
            f"[vibrt] warn: world {world.name!r}: failed to bake "
            f"camera-ray split branch {role!r} ({branch_bg.name!r}): "
            f"{ex} — split path will fall back to combined bake"
        )
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    if baked is None:
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    rgb, bw, bh = baked
    _SKY_BAKE_CACHE[cache_key] = (rgb, bw, bh, None)
    _emit(
        f"[vibrt] world {world.name!r}: pre-baked {role!r} branch "
        f"{branch_bg.name!r} (camera-ray split) to a {bw}x{bh} envmap "
        f"in {time.perf_counter() - t0:.2f}s"
    )
    bpy.data.worlds.remove(tmp_world, do_unlink=True)


def _prebake_sky_node_alone(world, sky_node, w: int, h: int) -> None:
    """Bake a single Sky Texture node in isolation (its own temp world
    with `Output → Background → Sky`) so option-A's Mixed envmap can
    pick it up as one layer without re-rendering the whole world graph.
    Cached by Sky-node identity in `_SKY_BAKE_CACHE`."""
    key = _sky_node_alone_cache_key(world, sky_node)
    if key in _SKY_BAKE_CACHE:
        return
    t0 = time.perf_counter()
    # Build a temp world with a single Background → this Sky Texture.
    # We can't render `world` directly because it's a MixShader of two
    # Backgrounds; we want only the Sky's contribution, so a fresh tiny
    # world tree gets fed to `_bake_sky_world_to_pixels`.
    tmp_world = bpy.data.worlds.new(f"__vibrt_sky_alone_{sky_node.name}")
    tmp_world.use_nodes = True
    nt = tmp_world.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    sky_clone = nt.nodes.new("ShaderNodeTexSky")
    # Copy the Sky Texture's relevant attributes onto the clone so the
    # bake reproduces what the original node would emit.
    for attr in ("sky_type", "sun_disc", "sun_size", "sun_intensity",
                 "sun_elevation", "sun_rotation", "altitude",
                 "air_density", "dust_density", "ozone_density",
                 "ground_albedo", "turbidity"):
        if hasattr(sky_node, attr) and hasattr(sky_clone, attr):
            try:
                setattr(sky_clone, attr, getattr(sky_node, attr))
            except (TypeError, AttributeError):
                pass
    nt.links.new(sky_clone.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out_n.inputs["Surface"])
    try:
        baked = _bake_sky_world_to_pixels(tmp_world, w=w, h=h)
    except Exception as ex:
        _emit(
            f"[vibrt] warn: world {world.name!r}: failed to bake Sky "
            f"node {sky_node.name!r} alone: {ex} — Mixed envmap will "
            f"miss this layer's contribution"
        )
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    if baked is None:
        bpy.data.worlds.remove(tmp_world, do_unlink=True)
        return
    rgb, bw, bh = baked
    sun_light = _extract_sun_from_bake_inplace(rgb, bw, bh, sky_node, world.name)
    _SKY_BAKE_CACHE[key] = (rgb, bw, bh, sun_light)
    _emit(
        f"[vibrt] world {world.name!r}: pre-baked Sky node {sky_node.name!r} "
        f"alone ({sky_node.sky_type}) to {bw}x{bh} envmap in "
        f"{time.perf_counter() - t0:.2f}s"
        + (" + extracted sun" if sun_light is not None else "")
    )
    bpy.data.worlds.remove(tmp_world, do_unlink=True)


def _prebake_sky_texture(world, sky, w: int, h: int) -> None:
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
    rgb, bw, bh = baked
    sun_light = _extract_sun_from_bake_inplace(rgb, bw, bh, sky, world.name)
    _SKY_BAKE_CACHE[key] = (rgb, bw, bh, sun_light)
    _emit(
        f"[vibrt] world {world.name!r}: pre-baked ShaderNodeTexSky "
        f"({sky.sky_type}) to a {baked[1]}x{baked[2]} envmap in "
        f"{time.perf_counter() - t0:.2f}s"
    )


def _prebake_world_full(world, w: int, h: int) -> None:
    key = _world_generic_cache_key(world)
    if key in _SKY_BAKE_CACHE:
        return
    t0 = time.perf_counter()
    try:
        baked = _bake_sky_world_to_pixels(world, w=w, h=h)
    except Exception as ex:
        _emit(
            f"[vibrt] warn: world {world.name!r}: failed to bake "
            f"complex world graph via Cycles in update(): {ex} — "
            f"world background will fall back to constant"
        )
        return
    if baked is None:
        _emit(
            f"[vibrt] warn: world {world.name!r}: complex world bake "
            f"produced no pixels — world background will fall back to "
            f"constant"
        )
        return
    rgb, bw, bh = baked
    # Walk the world graph for any Sky Texture with `sun_disc=True`. If
    # one is present (typical: a MixShader blends a Nishita sky with a
    # backplate HDRI — pabellon's sunset world), use it to drive the sun
    # extraction; the bright-pixel heuristic + MixShader-strength
    # weighting matches the disc that's actually baked into the
    # equirect. Without this, complex worlds drop the sun entirely and
    # NEE on shaded surfaces (pebbles below the pool plane) only sees
    # the diffuse-sky envmap, which is too dim to compete with the
    # camera-side Fresnel reflection on the water.
    sun_light = None
    sky_node = _find_first_sky_node(world)
    if sky_node is not None and getattr(sky_node, "sun_disc", False):
        sun_light = _extract_sun_from_bake_inplace(rgb, bw, bh, sky_node, world.name)
    _SKY_BAKE_CACHE[key] = (rgb, bw, bh, sun_light)
    _emit(
        f"[vibrt] world {world.name!r}: pre-baked complex world graph "
        f"to a {bw}x{bh} envmap in {time.perf_counter() - t0:.2f}s"
        + (f" + extracted sun" if sun_light is not None else "")
    )


def _find_first_sky_node(world):
    """Walk the world's node tree for the first ShaderNodeTexSky regardless
    of where it sits in the graph (direct Background.Color, behind a
    MixShader, inside a NodeGroup). Used to seed sun extraction for the
    full-world-bake path."""
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    for n in world.node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexSky":
            return n
    return None


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
    # Threshold for "this is the sun disc, not just bright sky".
    # Nishita sun discs punch through diffuse sky by 4-5 orders of
    # magnitude. Preetham / sunset bakes put the disc only ~2-3×
    # brighter than the brightest cloud. We rely on a percentile-
    # relative threshold (the disc is always the brightest pixel
    # cluster) and require the ratio of max to median luminance to
    # be high enough that the bake genuinely has a hot spot — pure
    # overcast skies have a very low max/median ratio (≈ 1.5), suns
    # push it to 3+. Skips both the hand-edited HDRI overcast case
    # and the very-dim-sun edge cases that would otherwise produce a
    # garbage delta light.
    p_med = float(np.percentile(lum, 50.0))
    if p_med <= 0.0 or lum_max / max(p_med, 1e-4) < 3.0:
        return None
    p99 = float(np.percentile(lum, 99.0))
    threshold = p99 * 1.5
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
    # Empirical Cycles-parity boost. A 3-way comparison on lone_monk
    # (Cycles direct Nishita vs Cycles with our bake EXR as world vs
    # vibrt with bake_residual + extracted sun) showed Cycles' direct
    # Sky Texture rendering delivers ~2.6× more flux than the same
    # bake re-rendered as a generic Environment Texture. Source:
    # Cycles' `kbackground->use_sun_guiding` path
    # (`c:/tmp/cycles-src/src/kernel/light/background.h`,
    # `src/scene/light.cpp:1290`) fires when the world is a Sky Texture
    # with sun_disc, dedicates `sun_weight=4.0` (vs `map_weight=1.0`)
    # of background NEE samples to a uniform-cone sampler around the
    # known sun direction, and uses `sun_average_radiance` as the
    # sample value — neither of which fires for a baked equirect. The
    # ~2× scale here is a coarse empirical match (Cycles' sun_weight=4
    # split + the `0.8*map + 0.2*sun` MIS heuristic) that lifts
    # lone_monk's render close to the Cycles reference. Specific to
    # ShaderNodeTexSky + sun_disc; HDRI environments don't go through
    # this path so they're unaffected.
    sun_guiding_boost = 2.0
    flux_max_boosted = flux_max * sun_guiding_boost
    _emit(
        f"[vibrt] world {world_name!r}: split {n_bright} sun-disc pixels "
        f"(peak L={lum_max:.0f}, residual sky max={threshold:.0f}) → "
        f"sun light dir={sun_dir} E={flux_max_boosted:.2f} "
        f"(raw={flux_max:.2f} × {sun_guiding_boost} sun_guiding boost) "
        f"angle={math.degrees(sun_size):.2f}°"
    )
    return {
        "type": "sun",
        # Forward / photon-travel direction: opposite of the centroid
        # (which points toward the bright bake pixels = toward the
        # sun). `scene_loader.rs` negates this back to the toward-sun
        # direction the kernel uses as `wi`. Keep the convention in
        # sync with `_export_light` for SUN lamps.
        "direction": [-sun_dir[0], -sun_dir[1], -sun_dir[2]],
        "color": color,
        "strength": flux_max_boosted,
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
        # Sun discs in Sky Texture / HDRI hot spots cover tiny solid angles;
        # 64 spp doesn't hit them often enough for the average pixel to
        # carry the sun's full luminance. 256 spp + denoising disabled is
        # a reasonable trade — adds a few extra seconds to the one-time
        # update-step bake. 1024 spp doesn't help meaningfully on tested
        # scenes (pabellon's HDRI is already SDR-clipped and Nishita
        # converges by 256), so the extra cost isn't worth it.
        tmp.cycles.samples = 256
        tmp.cycles.use_denoising = False
        tmp.cycles.use_adaptive_sampling = False
        # No pixel reconstruction filter for the bake. The bake is treated
        # as a direct radiance grid by the importance sampler — Cycles' default
        # Blackman-Harris 1.5px filter would convolve each sample's value
        # into multiple pixels, which is what you want for a viewing image
        # but blurs sharp features (sun discs, HDRI hot spots) for a
        # radiance lookup. BOX width=1.0 makes each sample contribute only
        # to its own pixel. (Empirically didn't move lone_monk's |diff| —
        # the bake at 256 spp was already converged on the sun pixels —
        # but the change keeps the bake conceptually correct as a radiance
        # source.)
        tmp.cycles.pixel_filter_type = 'BOX'
        tmp.cycles.filter_width = 1.0
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
    # Rotate the bake camera so its +Y_local axis (Cycles equirect's
    # latitude axis) points at world -Z. With foreach_get's bottom-up
    # storage that puts buffer y=0 at the world +Z direction (zenith),
    # which is what our kernel's `world_background` reads `theta=0` as.
    # The longitude axis ends up offset / mirrored — we correct that
    # below by rolling and flipping the bake to match the kernel's
    # `phi = atan2(dir.y_world, dir.x_world)` convention.
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
        # Roll the bake +90° in azimuth (no mirror). Empirically validated
        # via `scripts/_test_bake_orientation.py`, which builds a debug
        # world that paints `+X→red, +Y→green, +Z→blue` and confirms
        # cardinal-direction lookups match.
        #
        # Why: the bake camera uses `rotation_euler=(-π/2, 0, 0)` so its
        # latitude axis lines up with the world Z axis (kernel's `theta=0`
        # is +Z, the zenith). That same rotation lands the camera-forward
        # `-Z_local` along world `+Y`, while Cycles' equirect places
        # `phi_local=0` at camera-forward — so `u=0.5` in the bake stores
        # world `+Y`, but the kernel reads `phi = atan2(dir.y, dir.x)`
        # which expects `u=0` at world `+X` (and `u=0.25` at world `+Y`).
        # Rolling the array right by `bw/4` shifts world `+Y` from u=0.5
        # back to u=0.25, restoring the kernel's convention. The earlier
        # `arr[:, ::-1]` mirror lined up Sky Texture's sun_rotation by
        # accident but flipped the y axis (made +Y read as -Y from the
        # kernel) for any general world graph.
        bw = iw
        roll = bw // 4
        rgb = np.ascontiguousarray(np.roll(arr[..., :3], roll, axis=1),
                                   dtype=np.float32)
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

    # Light-Path camera-ray split: emit a Mixed envmap with
    # `split_by_camera_ray=true` so the kernel picks layer b for camera-
    # visible misses and layer a for lighting / NEE / indirect rays.
    # Both layers come from per-branch Cycles bakes populated by
    # `_prebake_world_branch_alone`.
    split_pat = _detect_mix_world_camera_ray_split(world)
    if split_pat is not None:
        split = _try_export_split_world(world, split_pat, writer, textures)
        if split is not None:
            return split
        # If the per-branch bake didn't land in cache, fall through to
        # the legacy full-world bake (which captures only the camera
        # branch — same behaviour as before this change).

    # Option A: detect `MixShader(Background_a, Background_b, fac=const)`
    # and emit `WorldDesc::Mixed` so each layer can be sampled at its
    # native resolution (HDRI direct, Sky Texture per-node bake) instead
    # of the combined world bake we used to fall back to.
    mix_pat = _detect_mix_world(world)
    if mix_pat is not None:
        mixed = _try_export_mixed_world(world, mix_pat, writer, textures,
                                         lights_json)
        if mixed is not None:
            return mixed
        # If conversion failed (unsupported source on a layer), fall
        # through to the legacy combined-bake path below.

    # Surface is driven by something we don't fold host-side (MixShader,
    # nested colour math, …). Look for a generic full-world bake the
    # prebake step should have left behind, and emit it as a plain
    # envmap. Strength=1 because the bake captured the world's
    # Cycles-evaluated output already.
    full_key = _world_generic_cache_key(world)
    full_baked = _SKY_BAKE_CACHE.get(full_key)
    if full_baked is not None:
        rgb, bw, bh, sun_light = full_baked
        pb = material_export._PreBakedTexture(
            rgb=rgb, w=bw, h=bh, cache_key=full_key,
        )
        tex_id = material_export.export_image_texture(
            pb, writer, textures, colorspace="linear"
        )
        if sun_light is not None and lights_json is not None:
            lights_json.append(dict(sun_light))
        return {
            "type": "envmap",
            "texture": tex_id,
            "rotation_z_rad": 0.0,
            "strength": 1.0,
        }

    _emit(
        f"[vibrt] warn: world {world.name!r}: World Output Surface driven by "
        f"{src.bl_idname} (expected ShaderNodeBackground) and no generic "
        f"bake was cached — world treated as black"
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
                        # Mixed-emissive (mesh has emission + a non-zero
                        # diffuse lobe / textures). Add a NEE-only rect proxy
                        # so diffuse receivers can sample the lamp; keep the
                        # mesh visible to the camera so its actual silhouette
                        # / fabric / specular response renders normally.
                        # flat_archiviz's Lamp Gubi (Principled with
                        # emission_strength=2 mixed with Translucent fabric)
                        # is the canonical case.
                        proxy = _try_mixed_emissive_proxy(
                            obj, obj_eval, materials[slot_mat_ids[0]]
                        )
                        if proxy is not None:
                            lights_json.append(proxy)

                    # Skip meshes that Cycles hides from camera — they're light
                    # portals (e.g. classroom's `windows` dayLight_portal) meant
                    # to inject light but never appear as geometry on the
                    # rendered image. Without this check the emissive portal
                    # bleeds over the ceiling / walls.
                    if not getattr(obj, "visible_camera", True):
                        continue

                    # Particle scatter emitters: when the source object has
                    # particle systems and `show_instancer_for_render = False`
                    # Cycles renders only the instances and hides the emitter
                    # mesh itself. Without this skip pabellon's
                    # `lotus_scattering_plane` / `pebbles_scatter` (both with
                    # empty material lists) cover the pond as bright opaque
                    # planes, drowning the water beneath.
                    if (not inst.is_instance
                            and obj_eval.particle_systems
                            and not getattr(obj, "show_instancer_for_render", True)):
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
    # Read Cycles' bounce caps directly so a scene authored with e.g.
    # `diffuse_bounces=2` (lone_monk) doesn't quietly get 8 diffuse
    # bounces' worth of warm-brick indirect light. Falls back to
    # generous defaults when Cycles' settings are missing (e.g. a
    # vibrt-only scene without the cycles addon active).
    cy = getattr(scene, "cycles", None)
    # Off-by-one between Cycles and vibrt: Cycles' `max_bounces=N` means
    # at most N *secondary* bounces (the camera ray + first-hit NEE
    # always run, even at N=0), while vibrt's path-tracer loop iterates
    # `bounce in 0..max_depth` and does NEE inside the loop body. So
    # `max_depth = max_bounces + 1` keeps "first iteration = primary +
    # direct" intact at small N. Without this, a regression scene with
    # `cycles.max_bounces=0` (Blender's official ies_light test) renders
    # all-black because the loop never executes. Larger production caps
    # (12+) drift by one but the difference is invisible there.
    cy_mb = int(getattr(cy, "max_bounces", 12)) if cy is not None else 12
    max_depth = cy_mb + 1
    max_diffuse = int(getattr(cy, "diffuse_bounces", max_depth)) if cy is not None else max_depth
    max_glossy = int(getattr(cy, "glossy_bounces", max_depth)) if cy is not None else max_depth
    max_transmission = int(getattr(cy, "transmission_bounces", max_depth)) if cy is not None else max_depth
    # Cycles' Light Paths > Clamping. Pabellon ships with both clamps at
    # 1.0; without honouring them vibrt over-bright Fresnel reflections of
    # the bright HDRI peaks (foreground pool floor read 50% brighter than
    # Cycles' reference). 0 = no clamp on the Cycles side; the vibrt scene
    # property still acts as a per-render override and falls back to the
    # default (10.0) when Cycles is unavailable.
    cy_clamp_indirect = float(getattr(cy, "sample_clamp_indirect", 0.0)) if cy is not None else 0.0
    cy_clamp_direct = float(getattr(cy, "sample_clamp_direct", 0.0)) if cy is not None else 0.0
    # Cycles' "Filter Glossy" parameter (cycles.blur_glossy). Pabellon
    # ships it at 5.0; default for new scenes is 1.0. Used in the kernel
    # to inflate subsequent BSDFs' alpha after glossy / transmission
    # bounces, suppressing fireflies on caustic-prone paths.
    cy_filter_glossy = float(getattr(cy, "blur_glossy", 0.0)) if cy is not None else 0.0
    user_clamp = float(scene.vibrt_clamp_indirect)
    # Pick the tightest non-zero clamp from Cycles + the vibrt user
    # property. 0 means "no clamp" in either system, so skip those.
    clamp_indirect = min(
        c for c in (cy_clamp_indirect, user_clamp) if c > 0.0
    ) if any(c > 0.0 for c in (cy_clamp_indirect, user_clamp)) else 0.0
    _emit(
        f"[vibrt] bounces total={max_depth} "
        f"diffuse={max_diffuse} glossy={max_glossy} transmission={max_transmission}"
    )
    scene_json = {
        "version": 1,
        "render": {
            "width": width,
            "height": height,
            "spp": spp,
            "max_depth": max_depth,
            "max_diffuse_bounces": max_diffuse,
            "max_glossy_bounces": max_glossy,
            "max_transmission_bounces": max_transmission,
            "clamp_indirect": clamp_indirect,
            "clamp_direct": cy_clamp_direct,
            "filter_glossy": cy_filter_glossy,
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
