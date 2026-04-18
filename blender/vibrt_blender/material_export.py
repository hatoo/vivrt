"""Map Blender shader graphs to vibrt's Principled material JSON.

Supports common non-Principled patterns used by legacy Cycles scenes:
Diffuse, Glossy, Diffuse+Glossy Mix, Glass, Emission, and Group nodes.
All collapsed onto the renderer's Principled-only parameter set.
"""

from __future__ import annotations

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


def _socket_linked_image(sock) -> bpy.types.Image | None:
    """Return the image feeding a socket, seeing through a Mapping node."""
    if not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image
    # Rare on color paths but harmless: follow through a Mapping node,
    # dropping the UV transform (renderer doesn't honor it).
    if src.bl_idname == "ShaderNodeMapping":
        for inp in src.inputs:
            if not inp.is_linked:
                continue
            upstream = inp.links[0].from_node
            if upstream.bl_idname == "ShaderNodeTexImage" and upstream.image is not None:
                return upstream.image
    return None


def _normal_map_image(normal_sock) -> bpy.types.Image | None:
    """Return a tangent-space normal map image feeding a Normal input, or None.

    Bump nodes (height-based) are intentionally skipped — the renderer only
    supports tangent-space normal maps.
    """
    if normal_sock is None or not normal_sock.is_linked:
        return None
    src = normal_sock.links[0].from_node
    if src.bl_idname == "ShaderNodeNormalMap":
        return _socket_linked_image(src.inputs["Color"])
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image
    return None


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
    if bl == "ShaderNodeEmission":
        return _from_emission(node, buf, textures)
    if bl == "ShaderNodeGroup":
        return _from_group(node, buf, textures, mat_name)
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

    img = _normal_map_image(node.inputs.get("Normal"))
    if img is not None:
        p["normal_tex"] = export_image_texture(img, buf, textures, "linear")
    return p


def _from_diffuse(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["roughness"] = 1.0
    p["metallic"] = 0.0
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
    img = _normal_map_image(node.inputs.get("Normal"))
    if img is not None:
        p["normal_tex"] = export_image_texture(img, buf, textures, "linear")
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
    img = _normal_map_image(node.inputs.get("Normal"))
    if img is not None:
        p["normal_tex"] = export_image_texture(img, buf, textures, "linear")
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


def _from_transparent(node, buf, textures) -> dict:
    p = _default_params()
    p["base_color"] = _socket_rgb(node.inputs["Color"])
    p["transmission"] = 1.0
    p["ior"] = 1.0  # no refraction
    p["roughness"] = 0.0
    img = _socket_linked_image(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb")
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
        if "normal_tex" not in out and "normal_tex" in gloss:
            out["normal_tex"] = gloss["normal_tex"]
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

    surface = out_node.inputs["Surface"].links[0].from_node
    return _resolve_shader(surface, buf, textures, mat.name)
