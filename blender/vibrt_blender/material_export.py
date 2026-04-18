"""Map Blender shader graphs to vibrt's Principled material JSON.

Supports common non-Principled patterns used by legacy Cycles scenes:
Diffuse, Glossy, Diffuse+Glossy Mix, Glass, Emission, and Group nodes.
All collapsed onto the renderer's Principled-only parameter set.
"""

from __future__ import annotations

import math
import struct

import bpy


_LOGGED_WARNINGS: set = set()
_CURRENT_MATERIAL: str = ""


def _warn(key: str, msg: str) -> None:
    """Print a one-line warning, deduped within the current material.

    `key` is the dedup key; `msg` is the human message. Reset per-material by
    `export_material`, so the same issue recurring across materials still
    prints (once per material), but a single material doesn't spam the log.
    """
    full_key = (_CURRENT_MATERIAL, key)
    if full_key in _LOGGED_WARNINGS:
        return
    _LOGGED_WARNINGS.add(full_key)
    prefix = f"material {_CURRENT_MATERIAL!r}: " if _CURRENT_MATERIAL else ""
    print(f"[vibrt] warn: {prefix}{msg}")


def _node_tag(node) -> str:
    """`Name (bl_idname)` for log messages — lets users locate the node."""
    return f"{node.name!r} ({node.bl_idname})"


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


def _socket_constant_rgb(sock) -> list[float] | None:
    """Return a constant RGB triple if the socket is effectively constant.

    Accepts unlinked sockets (uses default_value) and sockets linked to a
    `ShaderNodeRGB`. Returns None for any other linked source — signalling to
    callers that the value depends on a runtime-computed chain and can't be
    folded into a bake.
    """
    if not sock.is_linked:
        return _socket_rgb(sock)
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeRGB":
        v = src.outputs[0].default_value
        return [float(v[0]), float(v[1]), float(v[2])]
    return None


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
        _warn(
            f"map-vtype:{node.as_pointer()}",
            f"Mapping {_node_tag(node)}: vector_type={vector_type!r} not "
            f"supported (only POINT) — UV transform ignored",
        )
        return list(_IDENTITY_UV)

    for n in ("Location", "Rotation", "Scale"):
        if n in node.inputs and node.inputs[n].is_linked:
            _warn(
                f"map-linked:{node.as_pointer()}:{n}",
                f"Mapping {_node_tag(node)}: {n} is linked — UV transform "
                f"uses default constant only",
            )

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
    # Note: `ShaderNodeMix` is handled specially by `_chain_candidates` so the
    # RGBA-typed A/B sockets are picked (inputs["A"] returns the float one).
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
    """Return the image feeding a non-base-color socket.

    Sees through common colour-math / UV-transform / scalar-math intermediates
    listed in `_PASSTHROUGH_INPUTS`. Chain transforms are *dropped* here
    (scalar texture paths aren't baked), so we warn if any were detected so
    the user knows the effect on roughness/metallic/normal is missing.
    """
    img, chain = _socket_linked_image_with_chain(sock, depth)
    if img is not None and chain:
        _warn(
            f"scalar-drop:{id(sock)}",
            f"socket {sock.name!r}: colour transform(s) between TexImage and "
            f"this input were dropped (scalar-path bake not implemented)",
        )
    return img


def _socket_linked_image_with_chain(sock, depth: int = 0):
    """Return (image, transforms_chain) feeding a color socket.

    `transforms_chain` is a tuple of (id_tuple, apply_fn) pairs that should be
    applied to the texture's pixels in order (texture-first, BSDF-last) to
    reproduce the effect of colour-modifying nodes between the texture and
    `sock`. Nodes whose effect can't be baked are passed through with a
    warning via `_warn`.
    """
    if not sock.is_linked:
        return None, ()
    if depth > 6:
        _warn(
            f"depth:{id(sock)}",
            f"socket {sock.name!r}: chain deeper than 6 nodes — giving up",
        )
        return None, ()
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeTexImage":
        if src.image is None:
            _warn(
                f"texnoimage:{src.as_pointer()}",
                f"TexImage {src.name!r} has no image assigned",
            )
            return None, ()
        return src.image, ()
    candidates = _chain_candidates(src)
    if candidates is None:
        _warn(
            f"nonpassthru:{src.bl_idname}",
            f"node {_node_tag(src)} in colour chain is not recognised — "
            f"stopping traversal (the texture behind it will not be used)",
        )
        return None, ()
    for name, inp in candidates:
        if not inp.is_linked:
            continue
        img, sub_chain = _socket_linked_image_with_chain(inp, depth + 1)
        if img is None:
            continue
        xform = _node_color_transform(src, chain_input_name=name)
        # Inner (closer to texture) first, then outer.
        chain = sub_chain + ((xform,) if xform is not None else ())
        return img, chain
    return None, ()


def _chain_candidates(src):
    """Return `[(name, socket), ...]` to follow through `src`, or None if `src`
    isn't a recognised pass-through node.

    Special-cases `ShaderNodeMix` because its inputs list contains duplicate
    A/B sockets for each data_type — we only want the RGBA ones.
    """
    bl = src.bl_idname
    if bl == "ShaderNodeMix":
        data_type = getattr(src, "data_type", "RGBA")
        if data_type != "RGBA":
            _warn(
                f"mix-dt:{src.as_pointer()}",
                f"node {_node_tag(src)}: data_type={data_type!r} not "
                f"supported (only RGBA)",
            )
            return None
        _, a, b = _mix_rgba_sockets(src)
        if a is None or b is None:
            _warn(
                f"mix-sock:{src.as_pointer()}",
                f"node {_node_tag(src)}: could not locate RGBA A/B sockets",
            )
            return None
        return [("A", a), ("B", b)]
    if bl not in _PASSTHROUGH_INPUTS:
        return None
    preferred = _PASSTHROUGH_INPUTS[bl]
    if preferred is None:
        return [(inp.name, inp) for inp in src.inputs]
    return [(n, src.inputs[n]) for n in preferred if n in src.inputs]


def _mix_rgba_sockets(node):
    """Pick the Factor/A/B sockets of a new-style ShaderNodeMix in RGBA mode.

    `inputs["A"]` returns the first socket named "A" — which is the Float one
    (index 2), not the Color one (index 6). Iterate by `type` to disambiguate.
    """
    fac = a = b = None
    for inp in node.inputs:
        t = inp.type
        if inp.name == "Factor" and t == "VALUE" and fac is None:
            fac = inp
        elif inp.name == "A" and t == "RGBA" and a is None:
            a = inp
        elif inp.name == "B" and t == "RGBA" and b is None:
            b = inp
    return fac, a, b


def _node_color_transform(node, chain_input_name=None):
    """Return (id_tuple, apply_fn) for a colour-modifying node, or None.

    `id_tuple` is a hashable identifier used in the texture cache key so that
    the same image with two different transforms gets two cache entries.
    `apply_fn(rgb)` takes a numpy float32 array of shape (..., 3) in linear
    space and returns the transformed array. `chain_input_name` identifies
    which socket the texture chain comes in on (matters for two-input nodes
    like Mix — the "other" side has to be constant to be bakeable).
    """
    bl = node.bl_idname
    if bl == "ShaderNodeRGBCurve":
        return _rgbcurve_transform(node)
    if bl == "ShaderNodeGamma":
        return _gamma_transform(node)
    if bl == "ShaderNodeBrightContrast":
        return _brightcontrast_transform(node)
    if bl == "ShaderNodeInvert":
        return _invert_transform(node)
    if bl == "ShaderNodeHueSaturation":
        return _huesat_transform(node)
    if bl == "ShaderNodeValToRGB":
        return _colorramp_transform(node)
    if bl == "ShaderNodeClamp":
        return _clamp_transform(node)
    if bl in ("ShaderNodeMix", "ShaderNodeMixRGB"):
        return _mix_transform(node, chain_input_name)
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


def _gamma_transform(node):
    import numpy as np
    if node.inputs["Gamma"].is_linked:
        _warn(
            f"gamma-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: Gamma input is linked — effect not baked",
        )
        return None
    gamma = _socket_f(node.inputs["Gamma"])
    id_tuple = ("Gamma", round(gamma, 6))

    def apply(rgb):
        return np.power(np.maximum(rgb, 0.0), gamma).astype(np.float32)

    return id_tuple, apply


def _brightcontrast_transform(node):
    # Blender formula: a = 1 + contrast; b = 0.5*(1 - a) + bright;
    # out = max(a*col + b, 0).
    import numpy as np
    if node.inputs["Bright"].is_linked or node.inputs["Contrast"].is_linked:
        _warn(
            f"brcon-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: Bright/Contrast input is linked — effect not baked",
        )
        return None
    bright = _socket_f(node.inputs["Bright"])
    contrast = _socket_f(node.inputs["Contrast"])
    a = 1.0 + contrast
    b = 0.5 * (1.0 - a) + bright
    id_tuple = ("BrightContrast", round(bright, 6), round(contrast, 6))

    def apply(rgb):
        return np.maximum(rgb * a + b, 0.0).astype(np.float32)

    return id_tuple, apply


def _invert_transform(node):
    import numpy as np
    if node.inputs["Fac"].is_linked:
        _warn(
            f"inv-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: Fac input is linked — effect not baked",
        )
        return None
    fac = _socket_f(node.inputs["Fac"])
    id_tuple = ("Invert", round(fac, 6))

    def apply(rgb):
        return (rgb * (1.0 - fac) + (1.0 - rgb) * fac).astype(np.float32)

    return id_tuple, apply


def _huesat_transform(node):
    import numpy as np
    for n in ("Hue", "Saturation", "Value", "Fac"):
        if n in node.inputs and node.inputs[n].is_linked:
            _warn(
                f"hsv-linked:{node.as_pointer()}:{n}",
                f"{_node_tag(node)}: {n} input is linked — effect not baked",
            )
            return None
    hue = _socket_f(node.inputs["Hue"])
    sat = _socket_f(node.inputs["Saturation"])
    val = _socket_f(node.inputs["Value"])
    fac = _socket_f(node.inputs["Fac"]) if "Fac" in node.inputs else 1.0
    id_tuple = (
        "HSV",
        round(hue, 6), round(sat, 6), round(val, 6), round(fac, 6),
    )

    def apply(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = np.max(rgb, axis=-1)
        mn = np.min(rgb, axis=-1)
        d = mx - mn
        dsafe = np.maximum(d, 1e-12)
        v = mx
        s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0.0)
        rc = (mx - r) / dsafe
        gc = (mx - g) / dsafe
        bc = (mx - b) / dsafe
        h = np.where(r == mx, bc - gc,
            np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
        h = np.where(d > 0, (h / 6.0) % 1.0, 0.0)

        h = (h + hue - 0.5) % 1.0
        s = np.clip(s * sat, 0.0, 1.0)
        v = v * val

        i = np.floor(h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        r2 = np.where(i == 0, v, np.where(i == 1, q,
             np.where(i == 2, p, np.where(i == 3, p,
             np.where(i == 4, t, v)))))
        g2 = np.where(i == 0, t, np.where(i == 1, v,
             np.where(i == 2, v, np.where(i == 3, q,
             np.where(i == 4, p, p)))))
        b2 = np.where(i == 0, p, np.where(i == 1, p,
             np.where(i == 2, t, np.where(i == 3, v,
             np.where(i == 4, v, q)))))
        hsv_rgb = np.stack([r2, g2, b2], axis=-1)
        return (rgb * (1.0 - fac) + hsv_rgb * fac).astype(np.float32)

    return id_tuple, apply


def _colorramp_transform(node):
    # ColorRamp takes a scalar. When fed a colour through a chain we collapse
    # to luminance — same heuristic Blender uses for Colour→Value conversion.
    import numpy as np
    ramp = node.color_ramp
    N = 1024
    lut = np.array(
        [list(ramp.evaluate(i / (N - 1)))[:3] for i in range(N)],
        dtype=np.float32,
    )
    elems_id = tuple(
        (round(e.position, 6), tuple(round(c, 6) for c in e.color[:4]))
        for e in ramp.elements
    )
    id_tuple = (
        "ColorRamp",
        getattr(ramp, "interpolation", "LINEAR"),
        getattr(ramp, "color_mode", "RGB"),
        elems_id,
    )

    def apply(rgb):
        y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        y = np.clip(y, 0.0, 1.0) * (N - 1)
        i0 = y.astype(np.int32)
        i1 = np.minimum(i0 + 1, N - 1)
        f = (y - i0)[..., None]
        return (lut[i0] * (1.0 - f) + lut[i1] * f).astype(np.float32)

    return id_tuple, apply


def _clamp_transform(node):
    import numpy as np
    if node.inputs["Min"].is_linked or node.inputs["Max"].is_linked:
        _warn(
            f"clamp-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: Min/Max input is linked — effect not baked",
        )
        return None
    mn = _socket_f(node.inputs["Min"])
    mx = _socket_f(node.inputs["Max"])
    mode = getattr(node, "clamp_type", "MINMAX")
    if mode == "RANGE" and mn > mx:
        mn, mx = mx, mn
    id_tuple = ("Clamp", mode, round(mn, 6), round(mx, 6))

    def apply(rgb):
        return np.clip(rgb, mn, mx).astype(np.float32)

    return id_tuple, apply


def _mix_transform(node, chain_input_name):
    """Bake a Mix node when the non-chain input is a constant colour.

    Supports MIX / MULTIPLY / ADD / SUBTRACT blend types. If the factor, or the
    non-chain input, is linked, we can't bake (return None and the node's
    effect is silently dropped — same as pre-bake behaviour).
    """
    import numpy as np
    if node.bl_idname == "ShaderNodeMixRGB":
        fac_sock = node.inputs["Fac"]
        a_sock = node.inputs["Color1"]
        b_sock = node.inputs["Color2"]
        blend = node.blend_type
        is_a_chain = chain_input_name == "Color1"
    else:
        if getattr(node, "data_type", "RGBA") != "RGBA":
            _warn(
                f"mix-dt2:{node.as_pointer()}",
                f"{_node_tag(node)}: data_type not RGBA — effect not baked",
            )
            return None
        fac_sock, a_sock, b_sock = _mix_rgba_sockets(node)
        if fac_sock is None or a_sock is None or b_sock is None:
            _warn(
                f"mix-sock2:{node.as_pointer()}",
                f"{_node_tag(node)}: could not locate RGBA sockets — effect not baked",
            )
            return None
        blend = getattr(node, "blend_type", "MIX")
        is_a_chain = chain_input_name == "A"

    if blend not in ("MIX", "MULTIPLY", "ADD", "SUBTRACT"):
        _warn(
            f"mix-blend:{node.as_pointer()}:{blend}",
            f"{_node_tag(node)}: blend_type={blend!r} not supported — "
            f"effect passed through without baking",
        )
        return None
    if fac_sock.is_linked:
        _warn(
            f"mix-fac-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: Factor is linked — effect not baked",
        )
        return None
    other_sock = b_sock if is_a_chain else a_sock
    other_const = _socket_constant_rgb(other_sock)
    if other_const is None:
        _warn(
            f"mix-other-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: non-chain input {other_sock.name!r} is "
            f"driven by another chain (two-texture mix) — effect not baked",
        )
        return None

    fac = _socket_f(fac_sock)
    use_clamp = bool(
        getattr(node, "use_clamp", False) or getattr(node, "clamp_result", False)
    )
    other_rgb = np.array(other_const, dtype=np.float32)
    id_tuple = (
        "Mix", blend, is_a_chain, round(fac, 6),
        tuple(round(float(c), 6) for c in other_rgb.tolist()),
        use_clamp,
    )

    def _clamp01(x):
        return np.clip(x, 0.0, 1.0) if use_clamp else x

    if blend == "MIX":
        if is_a_chain:
            def apply(rgb):
                return _clamp01(rgb * (1.0 - fac) + other_rgb * fac).astype(np.float32)
        else:
            def apply(rgb):
                return _clamp01(other_rgb * (1.0 - fac) + rgb * fac).astype(np.float32)
    elif blend == "MULTIPLY":
        if is_a_chain:
            def apply(rgb):
                return _clamp01(rgb * (1.0 - fac + fac * other_rgb)).astype(np.float32)
        else:
            def apply(rgb):
                return _clamp01(other_rgb * (1.0 - fac + fac * rgb)).astype(np.float32)
    elif blend == "ADD":
        if is_a_chain:
            def apply(rgb):
                return _clamp01(rgb + fac * other_rgb).astype(np.float32)
        else:
            def apply(rgb):
                return _clamp01(other_rgb + fac * rgb).astype(np.float32)
    else:  # SUBTRACT
        if is_a_chain:
            def apply(rgb):
                return _clamp01(rgb - fac * other_rgb).astype(np.float32)
        else:
            def apply(rgb):
                return _clamp01(other_rgb - fac * rgb).astype(np.float32)

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
    _warn(
        f"normal-src:{src.bl_idname}",
        f"Normal input driven by {_node_tag(src)} — not handled (expected "
        f"NormalMap, Bump, or TexImage); normal perturbation ignored",
    )
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
        _warn(f"vol-shader:{bl}", f"ignoring volume shader {bl}")
        return _default_params()
    _warn(f"unsup-shader:{bl}", f"unsupported shader node: {bl}")
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
        _warn(f"group-empty:{node.as_pointer()}", f"empty node group {_node_tag(node)}")
        return _default_params()
    group_output = next(
        (n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"),
        None,
    )
    if group_output is None:
        _warn(f"group-noout:{tree.name}", f"group {tree.name!r} has no output node")
        return _default_params()
    for sock in group_output.inputs:
        if sock.is_linked:
            inner = sock.links[0].from_node
            return _resolve_shader(inner, buf, textures, mat_name)
    _warn(
        f"group-nolink:{tree.name}",
        f"group {tree.name!r} output has no linked input",
    )
    return _default_params()


def export_material(
    mat: bpy.types.Material, buf: bytearray, textures: list
) -> dict:
    """Dispatch to the surface-node handler; fall back to diffuse_color."""
    global _CURRENT_MATERIAL
    # Reset dedup state so each material emits its own set of warnings.
    _LOGGED_WARNINGS.clear()
    _CURRENT_MATERIAL = mat.name
    try:
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
        if out_node is None:
            _warn("no-output", "no ShaderNodeOutputMaterial — using default params")
            return _default_params()
        if not out_node.inputs["Surface"].is_linked:
            _warn(
                "no-surface",
                "output Surface is not connected — using default params",
            )
            return _default_params()

        volume = out_node.inputs.get("Volume")
        if volume is not None and volume.is_linked:
            _warn("vol-output", "Volume shader on output is ignored")
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
    finally:
        _CURRENT_MATERIAL = ""
