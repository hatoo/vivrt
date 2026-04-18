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
    chain: tuple = (),
) -> int:
    """Serialize an image into scene.bin and register a TextureDesc entry.

    `chain` is a sequence of `(id_tuple, apply_fn)` color transforms (typically
    from `_socket_linked_image_with_chain`) to bake into the pixels. When
    non-empty, pixels are linearised, transformed, and stored as linear.

    Returns the texture index. Reuses existing entries by (image, chain) key.
    """
    chain_key = "" if not chain else "|" + repr(tuple(x[0] for x in chain))
    key = f"__image__{image.name}{chain_key}"
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
    if chain:
        pixels, colorspace = _bake_chain(pixels, width, height, colorspace, chain)
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


def _srgb_to_linear_np(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def _bake_chain(pixels, w, h, colorspace, chain):
    """Apply `chain` color transforms to RGBA f32 pixels.

    Transforms operate in linear space (matching Cycles' shader semantics: the
    TexImage node converts sRGB→linear before its Color output). If the source
    is sRGB, we linearise before applying. Returns the new pixel list and the
    output colorspace string ("linear" once anything is baked in).
    """
    import numpy as np
    arr = np.asarray(pixels, dtype=np.float32).reshape((h, w, 4))
    rgb = arr[..., :3].copy()
    if colorspace.lower() == "srgb":
        rgb = _srgb_to_linear_np(rgb).astype(np.float32)
    for _id, apply_fn in chain:
        rgb = apply_fn(rgb)
    arr[..., :3] = rgb
    return arr.reshape(-1).tolist(), "linear"


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


# Node types we traverse to reach an underlying TexImage, mapped to the input
# socket names that carry the dominant colour data (in priority order). We
# accept the tinting / remapping these nodes apply as the price for not losing
# the texture entirely — a flat default colour is usually worse than an
# approximately-correct texture. `None` means "any linked input, in order".
_PASSTHROUGH_INPUTS: dict[str, tuple[str, ...] | None] = {
    "ShaderNodeMapping": None,             # vector-side only, safe
    "ShaderNodeMix": ("A", "B"),           # new-style universal Mix
    "ShaderNodeMixRGB": ("Color1", "Color2"),
    "ShaderNodeRGBCurve": ("Color",),
    "ShaderNodeGamma": ("Color",),
    "ShaderNodeHueSaturation": ("Color",),
    "ShaderNodeBrightContrast": ("Color",),
    "ShaderNodeInvert": ("Color",),
    "ShaderNodeValToRGB": ("Fac",),        # ColorRamp: scalar-in, colour-out
    "ShaderNodeMath": None,                # scalar ops (Invert, Multiply, ...)
    "ShaderNodeClamp": ("Value",),
    "ShaderNodeSeparateColor": ("Color",), # packed ORM: Color -> R/G/B channels
    "ShaderNodeSeparateRGB": ("Image",),   # legacy pre-3.3 variant
    "ShaderNodeSeparateXYZ": ("Vector",),
}


def _socket_linked_image(sock, depth: int = 0) -> bpy.types.Image | None:
    """Return the image feeding a socket.

    Sees through common colour-math / UV-transform / scalar-math intermediates
    listed in `_PASSTHROUGH_INPUTS`. Most node behaviour is discarded; for
    base-color inputs use `_socket_linked_image_with_chain` instead so RGBCurve
    transforms can be baked into the texture pixels.
    """
    img, _ = _socket_linked_image_with_chain(sock, depth)
    return img


def _socket_linked_image_with_chain(sock, depth: int = 0):
    """Return (image, transforms_chain) feeding a color socket.

    `transforms_chain` is a tuple of (id_tuple, apply_fn) pairs that should be
    applied to the texture's pixels in order (texture-first, BSDF-last) to
    reproduce the effect of any RGBCurve / similar nodes between the texture
    and `sock`. Other pass-through nodes (Mix, Hue, Gamma, ...) are still
    ignored — same approximation as before.
    """
    if not sock.is_linked or depth > 6:
        return None, ()
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        return src.image, ()
    if src.bl_idname not in _PASSTHROUGH_INPUTS:
        return None, ()
    preferred = _PASSTHROUGH_INPUTS[src.bl_idname]
    if preferred is None:
        candidates = list(src.inputs)
    else:
        candidates = [src.inputs[n] for n in preferred if n in src.inputs]
    for inp in candidates:
        if not inp.is_linked:
            continue
        img, sub_chain = _socket_linked_image_with_chain(inp, depth + 1)
        if img is None:
            continue
        xform = _node_color_transform(src)
        # Inner (closer to texture) first, then outer.
        chain = sub_chain + ((xform,) if xform is not None else ())
        return img, chain
    return None, ()


def _node_color_transform(node):
    """Return (id_tuple, apply_fn) for a colour-modifying node, or None.

    `id_tuple` is a hashable identifier used in the texture cache key so that
    the same image with two different transforms gets two cache entries.
    `apply_fn(rgb)` takes a numpy float32 array of shape (..., 3) in linear
    space and returns the transformed array.
    """
    if node.bl_idname == "ShaderNodeRGBCurve":
        return _rgbcurve_transform(node)
    return None


def _rgbcurve_transform(node):
    import numpy as np
    cm = node.mapping
    cm.update()
    pts_id = tuple(
        tuple((round(p.location[0], 6), round(p.location[1], 6))
              for p in cm.curves[ci].points)
        for ci in range(4)
    )
    bw_min, bw_max = float(cm.black_level[0]), float(cm.white_level[0])
    id_tuple = ("RGBCurve", pts_id, round(bw_min, 6), round(bw_max, 6))
    # Build a 1024-entry LUT per channel that includes the composite (C) curve
    # applied first, then the per-channel R/G/B curve (matches Blender).
    N = 1024
    luts = []
    composite = cm.curves[3]
    for ci in range(3):
        channel = cm.curves[ci]
        lut = np.array(
            [cm.evaluate(channel, cm.evaluate(composite, i / (N - 1)))
             for i in range(N)],
            dtype=np.float32,
        )
        luts.append(lut)

    def apply(rgb):
        out = np.empty_like(rgb)
        for ch in range(3):
            lut = luts[ch]
            v = np.clip(rgb[..., ch], 0.0, 1.0) * (N - 1)
            i0 = v.astype(np.int32)
            i1 = np.minimum(i0 + 1, N - 1)
            f = v - i0
            out[..., ch] = lut[i0] * (1.0 - f) + lut[i1] * f
        return out

    return id_tuple, apply


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
    img, chain = _socket_linked_image_with_chain(bc)
    if img is not None:
        # Linked socket: the link drives the colour, the default RGB is unused
        # in Cycles. The renderer multiplies base_color × texture, so neutralise
        # the factor to avoid tinting the texture.
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(bc)

    img = _socket_linked_image(node.inputs["Metallic"])
    if img is not None:
        p["metallic_tex"] = export_image_texture(img, buf, textures, "linear")
        p["metallic"] = 1.0
    else:
        p["metallic"] = _socket_f(node.inputs["Metallic"])

    img = _socket_linked_image(node.inputs["Roughness"])
    if img is not None:
        p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
        p["roughness"] = 1.0
    else:
        p["roughness"] = _socket_f(node.inputs["Roughness"])

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
    p["roughness"] = 1.0
    p["metallic"] = 0.0
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    _apply_normal_perturbation(p, node.inputs.get("Normal"), buf, textures)
    return p


def _from_glossy(node, buf, textures) -> dict:
    p = _default_params()
    p["metallic"] = 1.0
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
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
    p["transmission"] = 1.0
    if "IOR" in node.inputs:
        p["ior"] = _socket_f(node.inputs["IOR"])
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
    return p


def _from_refraction(node, buf, textures) -> dict:
    p = _default_params()
    p["transmission"] = 1.0
    p["metallic"] = 0.0
    if "IOR" in node.inputs:
        p["ior"] = _socket_f(node.inputs["IOR"])
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        img = _socket_linked_image(node.inputs["Roughness"])
        if img is not None:
            p["roughness_tex"] = export_image_texture(img, buf, textures, "linear")
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
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
    p["transmission"] = 1.0
    p["ior"] = 1.0  # no refraction
    p["roughness"] = 0.0
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, buf, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
        p["alpha_threshold"] = 0.5
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
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
