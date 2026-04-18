"""Map Blender shader graphs to vibrt's Principled material JSON.

Supports common non-Principled patterns used by legacy Cycles scenes:
Diffuse, Glossy, Diffuse+Glossy Mix, Glass, Emission, and Group nodes.
All collapsed onto the renderer's Principled-only parameter set.
"""

from __future__ import annotations

import math
import struct

import bpy


def _pack_f32(xs) -> bytes:
    return struct.pack(f"<{len(xs)}f", *xs)


def _write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def export_image_texture(
    image: bpy.types.Image,
    buf: bytearray,
    textures: list,
    colorspace: str | None = None,
) -> int:
    """Serialize an image into scene.bin and register a TextureDesc entry.

    Returns the texture index. Reuses existing entries by image name.
    """
    key = f"__image__{image.name}"
    for i, t in enumerate(textures):
        if t.get("_key") == key:
            return i
    if image.size[0] == 0 or image.size[1] == 0:
        image.update()
    width, height = image.size[0], image.size[1]
    if width == 0 or height == 0:
        width, height = 1, 1
        pixels = [1.0, 1.0, 1.0, 1.0]
    else:
        pixels = list(image.pixels[:])  # RGBA f32
    if colorspace is None:
        colorspace = (
            "srgb"
            if image.colorspace_settings.name.lower().startswith("srgb")
            else "linear"
        )
    desc = {
        "width": int(width),
        "height": int(height),
        "channels": 4,
        "colorspace": colorspace,
        "pixels": _write_blob(buf, _pack_f32(pixels)),
        "_key": key,
    }
    textures.append(desc)
    return len(textures) - 1


def _socket_rgb(sock) -> list[float]:
    v = sock.default_value
    return [float(v[0]), float(v[1]), float(v[2])]


def _socket_f(sock) -> float:
    return float(sock.default_value)


_IDENTITY_UV = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _first_mapping_node(node_tree) -> "bpy.types.Node | None":
    """Find any ShaderNodeMapping feeding a TexImage inside this node tree."""
    if node_tree is None:
        return None
    for n in node_tree.nodes:
        if n.bl_idname == "ShaderNodeMapping":
            return n
    return None


def _mapping_to_affine(node) -> list[float]:
    """Convert a Mapping node (type=POINT) to a 2x3 affine matrix over UV.

    Other types (TEXTURE/VECTOR/NORMAL) fall back to identity with a warning.
    """
    if node is None:
        return list(_IDENTITY_UV)
    vector_type = getattr(node, "vector_type", "POINT")
    if vector_type != "POINT":
        return list(_IDENTITY_UV)

    def _vec3(sock):
        v = sock.default_value
        return [float(v[0]), float(v[1]), float(v[2])]

    loc = _vec3(node.inputs["Location"])
    rot = _vec3(node.inputs["Rotation"])
    scl = _vec3(node.inputs["Scale"])

    sx, sy = scl[0], scl[1]
    theta = rot[2]  # UV is 2D; only Z rotation matters
    tx, ty = loc[0], loc[1]
    c = math.cos(theta)
    s = math.sin(theta)
    # M = T · R · S, row-major 2x3:
    return [sx * c, -sy * s, tx,
            sx * s,  sy * c, ty]


# Node types that pass a TexImage through unchanged in semantic terms: Mapping
# only transforms the Vector side, so seeing through it is safe. MixRGB /
# ColorRamp / Gamma / etc. change colour values; treating them as passthrough
# visibly darkens or tints the result, so they're omitted.
_PASSTHROUGH_NODES = {
    "ShaderNodeMapping",
}


def _socket_linked_image(sock, depth: int = 0) -> bpy.types.Image | None:
    """Return the image feeding a socket.

    Sees through color-math / UV-transform intermediates (MixRGB, ColorRamp,
    Mapping, etc.). The actual color math or UV transform is discarded — the
    renderer can't replicate it — but picking up the underlying image is still
    better than exporting a flat default colour.
    """
    if not sock.is_linked or depth > 6:
        return None
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image
    if src.bl_idname in _PASSTHROUGH_NODES:
        for inp in src.inputs:
            if not inp.is_linked:
                continue
            img = _socket_linked_image(inp, depth + 1)
            if img is not None:
                return img
    return None


def _normal_perturbation(normal_sock):
    """Inspect a Normal input socket and return (normal_img, strength,
    bump_img, bump_strength).

    Any can be None. Tangent-space normal maps go through ShaderNodeNormalMap;
    heightmaps come through ShaderNodeBump's Height input.
    """
    if normal_sock is None or not normal_sock.is_linked:
        return None, 1.0, None, 1.0
    src = normal_sock.links[0].from_node
    if src.bl_idname == "ShaderNodeNormalMap":
        img = _socket_linked_image(src.inputs["Color"])
        strength = _socket_f(src.inputs["Strength"]) if "Strength" in src.inputs else 1.0
        return img, strength, None, 1.0
    if src.bl_idname == "ShaderNodeBump":
        h_img = _socket_linked_image(src.inputs["Height"])
        strength = _socket_f(src.inputs["Strength"]) if "Strength" in src.inputs else 1.0
        distance = _socket_f(src.inputs["Distance"]) if "Distance" in src.inputs else 1.0
        return None, 1.0, h_img, strength * distance
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image, 1.0, None, 1.0
    return None, 1.0, None, 1.0


def _normal_map_image(normal_sock) -> bpy.types.Image | None:
    img, _, _, _ = _normal_perturbation(normal_sock)
    return img


def _apply_normal_perturbation(params: dict, normal_sock, buf, textures) -> None:
    """Resolve the Normal input and assign normal_tex / bump_tex + strengths."""
    n_img, n_strength, b_img, b_strength = _normal_perturbation(normal_sock)
    if n_img is not None:
        params["normal_tex"] = export_image_texture(n_img, buf, textures, "linear")
        if n_strength != 1.0:
            params["normal_strength"] = n_strength
    if b_img is not None:
        params["bump_tex"] = export_image_texture(b_img, buf, textures, "linear")
        if b_strength != 1.0:
            params["bump_strength"] = b_strength


def _default_params() -> dict:
    return {
        "base_color": [0.8, 0.8, 0.8],
        "metallic": 0.0,
        "roughness": 0.5,
        "ior": 1.45,
        "transmission": 0.0,
        "emission": [0.0, 0.0, 0.0],
    }


def _resolve_shader(node, buf, textures, mat_name: str) -> dict:
    """Return Principled-equivalent params for any surface-shader node."""
    bl = node.bl_idname
    if bl == "ShaderNodeBsdfPrincipled":
        return _from_principled(node, buf, textures)
    if bl == "ShaderNodeBsdfDiffuse":
        return _from_diffuse(node, buf, textures)
    if bl == "ShaderNodeBsdfGlossy" or bl == "ShaderNodeBsdfAnisotropic":
        return _from_glossy(node, buf, textures)
    if bl == "ShaderNodeBsdfTranslucent":
        return _from_diffuse(node, buf, textures)
    if bl == "ShaderNodeBsdfTransparent":
        return _from_transparent(node, buf, textures)
    if bl == "ShaderNodeMixShader":
        return _from_mix(node, buf, textures, mat_name)
    if bl == "ShaderNodeBsdfGlass":
        return _from_glass(node, buf, textures)
    if bl == "ShaderNodeBsdfRefraction":
        return _from_refraction(node, buf, textures)
    if bl == "ShaderNodeEmission":
        return _from_emission(node, buf, textures)
    if bl == "ShaderNodeAddShader":
        return _from_add(node, buf, textures, mat_name)
    if bl == "ShaderNodeGroup":
        return _from_group(node, buf, textures, mat_name)
    if bl == "ShaderNodeVolumeAbsorption" or bl == "ShaderNodeVolumeScatter" or bl == "ShaderNodeVolumePrincipled":
        print(f"[vibrt] ignoring volume shader {bl} in material {mat_name!r}")
        return _default_params()
    print(f"[vibrt] unsupported shader node: {bl} in material {mat_name!r}")
    return _default_params()


def _from_principled(node, buf, textures) -> dict:
    p = _default_params()
    bc = node.inputs["Base Color"]
    p["base_color"] = _socket_rgb(bc)
    img = _socket_linked_image(bc)
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")

    p["metallic"] = _socket_f(node.inputs["Metallic"])
    img = _socket_linked_image(node.inputs["Metallic"])
    if img is not None:
        p["metallic_tex"] = export_image_texture(img, buf, textures, "linear")

    p["roughness"] = _socket_f(node.inputs["Roughness"])
    img = _socket_linked_image(node.inputs["Roughness"])
    if img is not None:
        p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")

    if "IOR" in node.inputs:
        p["ior"] = _socket_f(node.inputs["IOR"])
    trans_name = (
        "Transmission Weight" if "Transmission Weight" in node.inputs
        else "Transmission" if "Transmission" in node.inputs
        else None
    )
    if trans_name:
        p["transmission"] = _socket_f(node.inputs[trans_name])

    if "Emission" in node.inputs:
        p["emission"] = _socket_rgb(node.inputs["Emission"])
    elif "Emission Color" in node.inputs:
        color = _socket_rgb(node.inputs["Emission Color"])
        strength = (
            _socket_f(node.inputs["Emission Strength"])
            if "Emission Strength" in node.inputs else 1.0
        )
        p["emission"] = [c * strength for c in color]

    _apply_normal_perturbation(p, node.inputs.get("Normal"), buf, textures)

    # Anisotropy (Blender Principled: "Anisotropic" + "Anisotropic Rotation").
    for name in ("Anisotropic", "Anisotropy"):
        if name in node.inputs:
            a = _socket_f(node.inputs[name])
            if a != 0.0:
                p["anisotropy"] = a
            break
    for name in ("Anisotropic Rotation", "Anisotropy Rotation"):
        if name in node.inputs:
            r = _socket_f(node.inputs[name])
            if r != 0.0:
                p["tangent_rotation"] = r * 2.0 * math.pi
            break

    # Coat (Blender 4.x: "Coat Weight" / "Coat Roughness" / "Coat IOR";
    # legacy 3.x: "Clearcoat" / "Clearcoat Roughness").
    for name in ("Coat Weight", "Clearcoat"):
        if name in node.inputs:
            w = _socket_f(node.inputs[name])
            if w > 0.0:
                p["coat_weight"] = w
            break
    for name in ("Coat Roughness", "Clearcoat Roughness"):
        if name in node.inputs:
            r = _socket_f(node.inputs[name])
            if r != 0.03:
                p["coat_roughness"] = r
            break
    if "Coat IOR" in node.inputs:
        ior = _socket_f(node.inputs["Coat IOR"])
        if ior != 1.5:
            p["coat_ior"] = ior

    # Sheen (Blender 4.x: "Sheen Weight" / "Sheen Roughness" / "Sheen Tint";
    # legacy: "Sheen" / "Sheen Tint" scalar).
    for name in ("Sheen Weight", "Sheen"):
        if name in node.inputs:
            w = _socket_f(node.inputs[name])
            if w > 0.0:
                p["sheen_weight"] = w
            break
    if "Sheen Roughness" in node.inputs:
        r = _socket_f(node.inputs["Sheen Roughness"])
        if r != 0.5:
            p["sheen_roughness"] = r
    if "Sheen Tint" in node.inputs:
        v = node.inputs["Sheen Tint"].default_value
        if hasattr(v, "__len__") and len(v) >= 3:
            t = [float(v[0]), float(v[1]), float(v[2])]
            if t != [1.0, 1.0, 1.0]:
                p["sheen_tint"] = t

    # Subsurface (Blender 4.x: "Subsurface Weight" / "Subsurface Radius"
    # / "Subsurface Anisotropy"; 3.x: "Subsurface" / "Subsurface Radius").
    for name in ("Subsurface Weight", "Subsurface"):
        if name in node.inputs:
            w = _socket_f(node.inputs[name])
            if w > 0.0:
                p["sss_weight"] = w
            break
    if "Subsurface Radius" in node.inputs:
        v = node.inputs["Subsurface Radius"].default_value
        if hasattr(v, "__len__") and len(v) >= 3:
            r = [float(v[0]), float(v[1]), float(v[2])]
            if r != [1.0, 0.2, 0.1]:
                p["sss_radius"] = r
    if "Subsurface Anisotropy" in node.inputs:
        a = _socket_f(node.inputs["Subsurface Anisotropy"])
        if a != 0.0:
            p["sss_anisotropy"] = a

    alpha_sock = node.inputs.get("Alpha")
    if alpha_sock is not None:
        alpha_val = _socket_f(alpha_sock)
        if alpha_sock.is_linked:
            p["alpha_threshold"] = 0.5
            if "base_color_tex" not in p:
                img = _socket_linked_image(alpha_sock)
                if img is not None:
                    p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
        elif alpha_val < 1.0:
            p["alpha_threshold"] = alpha_val
    return p


def _from_diffuse(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["roughness"] = 1.0
    p["metallic"] = 0.0
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    _apply_normal_perturbation(p, node.inputs.get("Normal"), buf, textures)
    return p


def _from_glossy(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["metallic"] = 1.0
    if "Roughness" in node.inputs:
        p["roughness"] = _socket_f(node.inputs["Roughness"])
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
    if "Anisotropy" in node.inputs:
        a = _socket_f(node.inputs["Anisotropy"])
        if a != 0.0:
            p["anisotropy"] = a
    if "Rotation" in node.inputs:
        r = _socket_f(node.inputs["Rotation"])
        if r != 0.0:
            p["tangent_rotation"] = r * 2.0 * math.pi
    _apply_normal_perturbation(p, node.inputs.get("Normal"), buf, textures)
    return p


def _from_glass(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["transmission"] = 1.0
    if "Roughness" in node.inputs:
        p["roughness"] = _socket_f(node.inputs["Roughness"])
    if "IOR" in node.inputs:
        p["ior"] = _socket_f(node.inputs["IOR"])
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
    return p


def _from_refraction(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["transmission"] = 1.0
    p["metallic"] = 0.0
    if "Roughness" in node.inputs:
        p["roughness"] = _socket_f(node.inputs["Roughness"])
    if "IOR" in node.inputs:
        p["ior"] = _socket_f(node.inputs["IOR"])
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
    return p


def _from_add(node, buf, textures, mat_name: str) -> dict:
    """Add Shader: sum of two branches. Average surface-like params, sum emission."""
    in1, in2 = node.inputs[0], node.inputs[1]
    n1 = in1.links[0].from_node if in1.is_linked else None
    n2 = in2.links[0].from_node if in2.is_linked else None
    p1 = _resolve_shader(n1, buf, textures, mat_name) if n1 else _default_params()
    p2 = _resolve_shader(n2, buf, textures, mat_name) if n2 else _default_params()
    out = dict(p1)
    out["emission"] = [a + b for a, b in zip(p1["emission"], p2["emission"])]
    out["transmission"] = max(p1["transmission"], p2["transmission"])
    # Prefer the non-emission branch's surface params when mixing with an Emission.
    if sum(p1["emission"]) > 0 and sum(p2["emission"]) == 0:
        out["base_color"] = p2["base_color"]
        out["metallic"] = p2["metallic"]
        out["roughness"] = p2["roughness"]
        for k in ("base_color_tex", "normal_tex", "roughness_tex",
                  "metallic_tex", "bump_tex", "normal_strength",
                  "bump_strength", "uv_transform", "alpha_threshold"):
            if k in p2:
                out[k] = p2[k]
    return out


def _from_transparent(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["transmission"] = 1.0
    p["ior"] = 1.0  # no refraction
    p["roughness"] = 0.0
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
        p["alpha_threshold"] = 0.5
    return p


def _from_emission(node, buf, textures) -> dict:
    p = _default_params()
    color = _socket_rgb(node.inputs["Color"])
    strength = (
        _socket_f(node.inputs["Strength"])
        if "Strength" in node.inputs else 1.0
    )
    p["emission"] = [c * strength for c in color]
    p["base_color"] = [0.0, 0.0, 0.0]
    # Allow a color texture to be picked up as base_color_tex (shaded on the
    # emissive surface below the emission term).
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    return p


def _from_mix(node, buf, textures, mat_name: str) -> dict:
    in1, in2 = node.inputs[1], node.inputs[2]
    n1 = in1.links[0].from_node if in1.is_linked else None
    n2 = in2.links[0].from_node if in2.is_linked else None
    p1 = _resolve_shader(n1, buf, textures, mat_name) if n1 else _default_params()
    p2 = _resolve_shader(n2, buf, textures, mat_name) if n2 else _default_params()
    t1 = n1.bl_idname if n1 else None
    t2 = n2.bl_idname if n2 else None

    diffuse_like = {"ShaderNodeBsdfDiffuse", "ShaderNodeBsdfTranslucent"}
    glossy_like = {"ShaderNodeBsdfGlossy", "ShaderNodeBsdfAnisotropic"}
    if (t1 in diffuse_like and t2 in glossy_like) or (t2 in diffuse_like and t1 in glossy_like):
        diff = p1 if t1 in diffuse_like else p2
        gloss = p1 if t1 in glossy_like else p2
        out = dict(diff)
        out["roughness"] = gloss["roughness"]
        out["metallic"] = 0.0
        if "roughness_tex" in gloss:
            out["roughness_tex"] = gloss["roughness_tex"]
        for k in ("normal_tex", "normal_strength", "bump_tex",
                  "bump_strength", "alpha_threshold"):
            if k not in out and k in gloss:
                out[k] = gloss[k]
        return out

    fac_sock = node.inputs[0]
    fac = _socket_f(fac_sock) if not fac_sock.is_linked else 0.5
    primary = p1 if fac < 0.5 else p2
    other = p2 if fac < 0.5 else p1
    out = dict(primary)
    out["emission"] = [max(a, b) for a, b in zip(primary["emission"], other["emission"])]
    out["transmission"] = max(primary["transmission"], other["transmission"])
    return out


def _from_group(node, buf, textures, mat_name: str) -> dict:
    tree = node.node_tree
    if tree is None:
        print(f"[vibrt] empty group in material {mat_name!r}")
        return _default_params()
    group_output = next(
        (n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"),
        None,
    )
    if group_output is None:
        print(f"[vibrt] group {tree.name!r} has no output in material {mat_name!r}")
        return _default_params()
    for sock in group_output.inputs:
        if sock.is_linked:
            inner = sock.links[0].from_node
            return _resolve_shader(inner, buf, textures, mat_name)
    print(f"[vibrt] group {tree.name!r} output has no linked input in {mat_name!r}")
    return _default_params()


def export_material(
    mat: bpy.types.Material, buf: bytearray, textures: list
) -> dict:
    """Dispatch to the surface-node handler; fall back to diffuse_color."""
    if not mat.use_nodes:
        p = _default_params()
        p["base_color"] = list(mat.diffuse_color)[:3]
        return p

    out_node = next(
        (n for n in mat.node_tree.nodes
         if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output),
        None,
    )
    if out_node is None:
        out_node = next(
            (n for n in mat.node_tree.nodes
             if n.bl_idname == "ShaderNodeOutputMaterial"),
            None,
        )
    if out_node is None or not out_node.inputs["Surface"].is_linked:
        return _default_params()

    volume = out_node.inputs.get("Volume")
    if volume is not None and volume.is_linked:
        print(f"[vibrt] ignoring Volume shader in material {mat.name!r}")
    displacement = out_node.inputs.get("Displacement")
    if displacement is not None and displacement.is_linked:
        # Displacement is consumed by the mesh exporter in a later phase.
        pass

    surface = out_node.inputs["Surface"].links[0].from_node
    params = _resolve_shader(surface, buf, textures, mat.name)

    # Per-material UV transform: take the first Mapping node found.
    mapping = _first_mapping_node(mat.node_tree)
    if mapping is not None:
        affine = _mapping_to_affine(mapping)
        if affine != _IDENTITY_UV:
            params["uv_transform"] = affine
    return params
