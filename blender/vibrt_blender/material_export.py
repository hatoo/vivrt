"""Map Blender shader graphs to vibrt's Principled material JSON.

Supports common non-Principled patterns used by legacy Cycles scenes:
Diffuse, Glossy, Diffuse+Glossy Mix, Glass, Emission, and Group nodes.
All collapsed onto the renderer's Principled-only parameter set.
"""

from __future__ import annotations

import math
import time

import bpy


_LOGGED_WARNINGS: set = set()
_CURRENT_MATERIAL: str = ""


# Phase timing accumulators — populated by the hot paths in this module and
# drained by exporter.py after each `export_scene` call. Self-times subtract
# nested child timings so the breakdown in the summary log is additive.
_STATS: dict = {
    "material_self_s": 0.0,
    "texture_self_s": 0.0,
    "pixel_read_s": 0.0,
    "bake_chain_s": 0.0,
    "texture_count": 0,
    "pixel_bytes": 0,
    "slow_textures": [],  # list of (name, w, h, seconds), only > 0.25s
    "slow_materials": [],  # list of (name, self_seconds), only > 0.10s
    "premix_cache_hits": 0,
    "premix_cache_misses": 0,
    "linear_rgb_cache_hits": 0,
    "linear_rgb_cache_misses": 0,
}


# Session-scoped cache for `_try_premix_two_textures`. Same Mix(TexA, TexB,
# fac, blend) may be reached repeatedly from different sockets on the same
# material (profile showed 2× per material on classroom.blend), and sometimes
# across materials that reuse textures. The bake is deterministic, so we can
# reuse the result keyed by the same identity string the function already
# computed for `_PreBakedTexture.cache_key`. Cleared in `reset_stats()`.
_PREMIX_CACHE: dict = {}


# Session-scoped cache for `_image_to_linear_rgb`. The result is a pure
# function of the image's pixels + colorspace, but every Mix-bake path
# re-reads the pixels and re-runs the sRGB→linear conversion. Keyed by
# `image.as_pointer()` since that's stable for the lifetime of the export.
# Values are stored read-only so any accidental mutation fails loudly.
_LINEAR_RGB_CACHE: dict = {}


# Session-scoped cache for `_image_mean_rgb`. Used when a constant-folding
# chain (MixShader factor, Emission colour, ...) needs the image reduced to
# a single RGBA value rather than its per-pixel array.
_MEAN_RGBA_CACHE: dict = {}


def reset_stats() -> None:
    _STATS["material_self_s"] = 0.0
    _STATS["texture_self_s"] = 0.0
    _STATS["pixel_read_s"] = 0.0
    _STATS["bake_chain_s"] = 0.0
    _STATS["texture_count"] = 0
    _STATS["pixel_bytes"] = 0
    _STATS["slow_textures"].clear()
    _STATS["slow_materials"].clear()
    _STATS["premix_cache_hits"] = 0
    _STATS["premix_cache_misses"] = 0
    _STATS["linear_rgb_cache_hits"] = 0
    _STATS["linear_rgb_cache_misses"] = 0
    _PREMIX_CACHE.clear()
    _LINEAR_RGB_CACHE.clear()
    _MEAN_RGBA_CACHE.clear()


def pop_stats() -> dict:
    """Snapshot the current stats — caller is expected to `reset_stats()` first."""
    return {
        "material_self_s": _STATS["material_self_s"],
        "texture_self_s": _STATS["texture_self_s"],
        "pixel_read_s": _STATS["pixel_read_s"],
        "bake_chain_s": _STATS["bake_chain_s"],
        "texture_count": _STATS["texture_count"],
        "pixel_bytes": _STATS["pixel_bytes"],
        "slow_textures": list(_STATS["slow_textures"]),
        "slow_materials": list(_STATS["slow_materials"]),
        "premix_cache_hits": _STATS["premix_cache_hits"],
        "premix_cache_misses": _STATS["premix_cache_misses"],
        "linear_rgb_cache_hits": _STATS["linear_rgb_cache_hits"],
        "linear_rgb_cache_misses": _STATS["linear_rgb_cache_misses"],
    }


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


# Reusable flat float32 buffer shared across `export_image_texture` calls
# within a single `export_scene`. Sized up front to the largest image datablock
# so each texture's foreach_get writes into a slice view instead of allocating
# a fresh per-texture buffer. Not a cache — contents are overwritten on every
# texture and the buffer is released when `end_export()` runs. Blender's
# RenderEngine.render is serial on the main thread, so no locking needed.
_REUSABLE_PIXELS = None

# Integer percentage applied to final-written texture dimensions, or None/100
# for no-op. Intermediate caches (linear RGB, mean RGBA, premix) still operate
# at full resolution; the downsample only affects what lands in scene.bin.
_TEXTURE_PCT = None


def begin_export(max_pixel_count: int, texture_pct: int | None = None) -> None:
    """Allocate the reusable foreach_get buffer for the upcoming export."""
    import numpy as np
    global _REUSABLE_PIXELS, _TEXTURE_PCT
    # `max(1, ...)` so a scene with no images still produces a valid view
    # should export_image_texture be invoked for an empty image.
    n = max(1, int(max_pixel_count)) * 4
    _REUSABLE_PIXELS = np.empty(n, dtype=np.float32)
    _TEXTURE_PCT = texture_pct


def end_export() -> None:
    """Release the reusable buffer so peak RAM drops back once export is done."""
    global _REUSABLE_PIXELS, _TEXTURE_PCT
    _REUSABLE_PIXELS = None
    _TEXTURE_PCT = None


class _PreBakedTexture:
    """A synthetic, fully-resolved linear RGB texture produced offline (e.g. by
    pre-mixing the two sides of a two-texture Mix node).

    Carries `rgb` as a numpy (h, w, 3) float32 array and a `cache_key` so the
    main exporter can dedupe and write it without going through `bpy.Image`.
    """
    __slots__ = ("rgb", "w", "h", "cache_key")

    def __init__(self, rgb, w: int, h: int, cache_key: str):
        self.rgb = rgb
        self.w = w
        self.h = h
        self.cache_key = cache_key


def export_image_texture(
    image,
    writer,
    textures: list,
    colorspace: str | None = None,
    chain: tuple = (),
) -> int:
    """Serialize an image into scene.bin and register a TextureDesc entry.

    Accepts either a `bpy.types.Image` (chain transforms baked into its pixels)
    or a `_PreBakedTexture` (already in linear space; chain ignored, must be
    empty). Returns the texture index. Reuses existing entries by cache key.
    """
    import numpy as np
    if isinstance(image, _PreBakedTexture):
        return _export_prebaked(image, writer, textures)
    chain_key = "" if not chain else "|" + repr(tuple(x[0] for x in chain))
    key = f"__image__{image.name}{chain_key}"
    for i, t in enumerate(textures):
        if t.get("_key") == key:
            return i
    t_enter = time.perf_counter()
    pixel_before = _STATS["pixel_read_s"]
    bake_before = _STATS["bake_chain_s"]
    if image.size[0] == 0 or image.size[1] == 0:
        image.update()
    width, height = image.size[0], image.size[1]
    if width == 0 or height == 0:
        width, height = 1, 1
        pixels = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    else:
        # foreach_get writes into a view of the module-level reusable buffer so
        # we skip the per-image `np.empty(w*h*4)` allocation. Fall back to a
        # fresh alloc if export_image_texture was reached without a surrounding
        # `begin_export()` (e.g. test harness calling it directly).
        n = width * height * 4
        if _REUSABLE_PIXELS is not None and _REUSABLE_PIXELS.size >= n:
            pixels = _REUSABLE_PIXELS[:n]
        else:
            pixels = np.empty(n, dtype=np.float32)
        t_px = time.perf_counter()
        image.pixels.foreach_get(pixels)
        _STATS["pixel_read_s"] += time.perf_counter() - t_px
        _STATS["pixel_bytes"] += pixels.nbytes
    if colorspace is None:
        colorspace = (
            "srgb"
            if image.colorspace_settings.name.lower().startswith("srgb")
            else "linear"
        )
    if chain:
        pixels, colorspace = _bake_chain(pixels, width, height, colorspace, chain)
    if _TEXTURE_PCT is not None and _TEXTURE_PCT != 100:
        new_w = max(1, int(round(width * _TEXTURE_PCT / 100.0)))
        new_h = max(1, int(round(height * _TEXTURE_PCT / 100.0)))
        if (new_w, new_h) != (width, height):
            rgba = pixels.reshape((height, width, 4))
            rgba = _resample_bilinear(rgba, new_h, new_w)
            pixels = np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)
            width, height = new_w, new_h
    desc = {
        "width": int(width),
        "height": int(height),
        "channels": 4,
        "colorspace": colorspace,
        "pixels": writer.write_f32(pixels),
        "_key": key,
    }
    textures.append(desc)
    dt = time.perf_counter() - t_enter
    children = (_STATS["pixel_read_s"] - pixel_before) + (_STATS["bake_chain_s"] - bake_before)
    _STATS["texture_self_s"] += max(0.0, dt - children)
    _STATS["texture_count"] += 1
    if dt > 0.25:
        slow = _STATS["slow_textures"]
        slow.append((image.name, int(width), int(height), dt))
        slow.sort(key=lambda e: e[3], reverse=True)
        del slow[5:]
    return len(textures) - 1


def _export_prebaked(pb: "_PreBakedTexture", writer, textures: list) -> int:
    import numpy as np
    for i, t in enumerate(textures):
        if t.get("_key") == pb.cache_key:
            return i
    rgb = pb.rgb
    w, h = int(pb.w), int(pb.h)
    if _TEXTURE_PCT is not None and _TEXTURE_PCT != 100:
        new_w = max(1, int(round(w * _TEXTURE_PCT / 100.0)))
        new_h = max(1, int(round(h * _TEXTURE_PCT / 100.0)))
        if (new_w, new_h) != (w, h):
            rgb = _resample_bilinear(rgb, new_h, new_w)
            w, h = new_w, new_h
    rgba = np.concatenate(
        [rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1
    )
    pixels = np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)
    desc = {
        "width": w,
        "height": h,
        "channels": 4,
        "colorspace": "linear",
        "pixels": writer.write_f32(pixels),
        "_key": pb.cache_key,
    }
    textures.append(desc)
    return len(textures) - 1


def _image_to_linear_rgb(image):
    """Decode a `bpy.types.Image` into a (h, w, 3) numpy array in linear space."""
    import numpy as np
    ptr = image.as_pointer()
    hit = _LINEAR_RGB_CACHE.get(ptr)
    if hit is not None:
        _STATS["linear_rgb_cache_hits"] += 1
        return hit
    _STATS["linear_rgb_cache_misses"] += 1
    if image.size[0] == 0 or image.size[1] == 0:
        image.update()
    w, h = int(image.size[0]), int(image.size[1])
    if w == 0 or h == 0:
        rgb = np.ones((1, 1, 3), dtype=np.float32)
        rgb.setflags(write=False)
        result = (rgb, 1, 1)
        _LINEAR_RGB_CACHE[ptr] = result
        return result
    px = np.empty(w * h * 4, dtype=np.float32)
    t_px = time.perf_counter()
    image.pixels.foreach_get(px)
    _STATS["pixel_read_s"] += time.perf_counter() - t_px
    _STATS["pixel_bytes"] += px.nbytes
    px = px.reshape((h, w, 4))
    rgb = np.ascontiguousarray(px[..., :3], dtype=np.float32)
    if image.colorspace_settings.name.lower().startswith("srgb"):
        rgb = _srgb_to_linear_np(rgb)
    # Freeze so downstream chain / resample steps can't accidentally mutate the
    # shared buffer. All known apply_fn/resample/mix helpers return new arrays.
    rgb.setflags(write=False)
    result = (rgb, w, h)
    _LINEAR_RGB_CACHE[ptr] = result
    return result


def _bake_source_to_linear(source, chain):
    """Resolve any texture source (`bpy.Image` or `_PreBakedTexture`) plus chain
    into a linear (h, w, 3) numpy array.
    """
    if isinstance(source, _PreBakedTexture):
        # PreBaked is already linear; chains on top would be unusual.
        rgb = source.rgb
        for _id, apply_fn in chain:
            rgb = apply_fn(rgb)
        return rgb, source.w, source.h
    rgb, w, h = _image_to_linear_rgb(source)
    for _id, apply_fn in chain:
        rgb = apply_fn(rgb)
    return rgb, w, h


def _resample_bilinear(rgb, target_h: int, target_w: int):
    """Separable bilinear resample — X pass first, then Y. The 4-corner form
    allocated 4 × (target_h, target_w, c) intermediates per call; classroom's
    700 → 2048 upsample made that ~200MB and ~300ms. Separable drops the
    heaviest pass to (h, target_w, c).
    """
    import numpy as np
    h, w, _ = rgb.shape
    if h == target_h and w == target_w:
        return rgb
    # X pass: interpolate columns, shape (h, target_w, c).
    xs = np.linspace(0.0, w - 1, target_w, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (xs - x0)[None, :, None]
    a = rgb[:, x0, :]
    b = rgb[:, x1, :]
    row_interp = a + (b - a) * fx
    # Y pass: interpolate rows, shape (target_h, target_w, c).
    ys = np.linspace(0.0, h - 1, target_h, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    fy = (ys - y0)[:, None, None]
    c0 = row_interp[y0]
    c1 = row_interp[y1]
    out = c0 + (c1 - c0) * fy
    return out.astype(np.float32, copy=False)


def _srgb_to_linear_np(x):
    """sRGB → linear conversion, kept in float32 end-to-end.

    The previous `np.power(..., 2.4)` with a Python-float exponent silently
    promoted the whole image to f64 and `np.where` over f32+f64 operands
    produced an f64 intermediate, so a 2K image would allocate ~50MB of f64
    it didn't need. Using an explicit `np.float32(2.4)` + `dtype=np.float32`
    keeps memory and arithmetic at half that.
    """
    import numpy as np
    xf = x if x.dtype == np.float32 else x.astype(np.float32)
    lin = xf * np.float32(1.0 / 12.92)
    gamma = np.power(
        (xf + np.float32(0.055)) * np.float32(1.0 / 1.055),
        np.float32(2.4),
        dtype=np.float32,
    )
    return np.where(xf <= np.float32(0.04045), lin, gamma)


def _bake_chain(pixels, w, h, colorspace, chain):
    """Apply `chain` color transforms to RGBA f32 pixels.

    Transforms operate in linear space (matching Cycles' shader semantics: the
    TexImage node converts sRGB→linear before its Color output). If the source
    is sRGB, we linearise before applying. Returns the new pixel list and the
    output colorspace string ("linear" once anything is baked in).
    """
    import numpy as np
    t_enter = time.perf_counter()
    try:
        arr = np.asarray(pixels, dtype=np.float32).reshape((h, w, 4)).copy()
        rgb = arr[..., :3].copy()
        if colorspace.lower() == "srgb":
            rgb = _srgb_to_linear_np(rgb).astype(np.float32)
        for _id, apply_fn in chain:
            rgb = apply_fn(rgb)
        arr[..., :3] = rgb
        # Return numpy directly — _pack_f32 handles it without a list round-trip.
        return arr.reshape(-1), "linear"
    finally:
        _STATS["bake_chain_s"] += time.perf_counter() - t_enter


def _socket_rgb(sock) -> list[float]:
    v = sock.default_value
    return [float(v[0]), float(v[1]), float(v[2])]


def _socket_f(sock) -> float:
    return float(sock.default_value)


def _warn_linked_scalar(node, input_name: str) -> float:
    """Return the effective constant value for a scalar input socket.

    If the socket is unlinked, returns its default. If linked, tries an
    *exact* fold (Value / Math / Clamp / scalar Mix / Invert of a constant) —
    those chains compute a single deterministic number, so we honour them
    silently. Lossy paths (texture means, procedural mean-colour
    substitution) still warn, because the renderer's scalar slot can't
    carry per-pixel variation and hiding that would silently drop the
    effect.
    """
    sock = node.inputs.get(input_name)
    if sock is None:
        return 0.0
    if not sock.is_linked:
        return _socket_f(sock)
    resolved = _resolve_exact_scalar(sock)
    if resolved is not None:
        return resolved
    src = sock.links[0].from_node
    _warn(
        f"linked-scalar:{node.as_pointer()}:{input_name}",
        f"{_node_tag(node)}: input {input_name!r} linked to {src.bl_idname} "
        f"but not baked — using constant default ({sock.default_value})",
    )
    return _socket_f(sock)


def _resolve_exact_scalar(sock, depth: int = 0) -> float | None:
    """Strict scalar fold: refuses texture means / procedural mean-colour
    approximations so callers can distinguish a real constant from a guess.

    Handles ShaderNodeValue, Math (via `_eval_math_constant`), Clamp,
    scalar Mix, and Invert — all evaluating to a single deterministic float.
    Anything else (TexImage, Attribute, Fresnel, ...) returns None.
    """
    if depth > 8:
        return None
    if not sock.is_linked:
        dv = getattr(sock, "default_value", None)
        if dv is None:
            return None
        try:
            return float(dv)
        except TypeError:
            # RGBA socket default — collapse to sRGB luminance so Invert on
            # an unlinked Color default still folds.
            try:
                return (0.2126 * float(dv[0]) + 0.7152 * float(dv[1])
                        + 0.0722 * float(dv[2]))
            except (TypeError, IndexError):
                return None
    link = sock.links[0]
    src = link.from_node
    bl = src.bl_idname
    if bl == "ShaderNodeValue":
        return float(src.outputs[0].default_value)
    if bl == "ShaderNodeMath":
        return _eval_math_constant(src, depth + 1)
    if bl == "ShaderNodeClamp":
        v = _resolve_exact_scalar(src.inputs["Value"], depth + 1)
        if v is None:
            return None
        mn = _resolve_exact_scalar(src.inputs["Min"], depth + 1)
        mx = _resolve_exact_scalar(src.inputs["Max"], depth + 1)
        if mn is None or mx is None:
            return None
        mode = getattr(src, "clamp_type", "MINMAX")
        if mode == "RANGE" and mn > mx:
            mn, mx = mx, mn
        return max(mn, min(mx, v))
    if bl == "ShaderNodeMix" and getattr(src, "data_type", "RGBA") == "FLOAT":
        fac_sock = next((i for i in src.inputs
                         if i.name == "Factor" and i.type == "VALUE"), None)
        a_sock = next((i for i in src.inputs
                       if i.name == "A" and i.type == "VALUE"), None)
        b_sock = next((i for i in src.inputs
                       if i.name == "B" and i.type == "VALUE"), None)
        if fac_sock is None or a_sock is None or b_sock is None:
            return None
        fa = _resolve_exact_scalar(fac_sock, depth + 1)
        va = _resolve_exact_scalar(a_sock, depth + 1)
        vb = _resolve_exact_scalar(b_sock, depth + 1)
        if fa is None or va is None or vb is None:
            return None
        if bool(getattr(src, "clamp_factor", True)):
            fa = max(0.0, min(1.0, fa))
        out = va * (1.0 - fa) + vb * fa
        if bool(getattr(src, "clamp_result", False)):
            out = max(0.0, min(1.0, out))
        return out
    if bl == "ShaderNodeInvert":
        fac_sock = src.inputs.get("Fac")
        col_sock = src.inputs.get("Color")
        if col_sock is None:
            return None
        if fac_sock is None:
            fac = 1.0
        else:
            fac = _resolve_exact_scalar(fac_sock, depth + 1)
            if fac is None:
                return None
        x = _resolve_exact_scalar(col_sock, depth + 1)
        if x is None:
            return None
        return x * (1.0 - fac) + (1.0 - x) * fac
    return None


def _socket_constant_rgb(sock) -> list[float] | None:
    """Return a constant RGB triple if the socket is effectively constant.

    Walks through bakeable colour-modifying nodes (RGBCurve, Gamma, Mix when
    both sides resolve to constants, ColorRamp with unlinked Fac, ...) so that
    a sub-tree with no TexImage anywhere collapses cleanly. Returns None if
    any node on the path is non-bakeable or any leaf is a TexImage (which
    isn't a constant — callers should treat it as a real texture chain).
    """
    return _resolve_constant_socket(sock, depth=0)


def _image_mean_rgba(image) -> tuple[list[float], float] | None:
    """Return (mean_linear_rgb, mean_alpha) for `image`, or None when pixels
    can't be read. Cached per-image. Used only in constant-folding paths —
    the full texture pipeline reads pixels directly via foreach_get.
    """
    import numpy as np
    ptr = image.as_pointer()
    hit = _MEAN_RGBA_CACHE.get(ptr)
    if hit is not None:
        return hit
    if image.size[0] == 0 or image.size[1] == 0:
        image.update()
    w, h = int(image.size[0]), int(image.size[1])
    if w == 0 or h == 0:
        return None
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    _STATS["pixel_bytes"] += px.nbytes
    px = px.reshape((-1, 4))
    rgb = px[..., :3]
    if image.colorspace_settings.name.lower().startswith("srgb"):
        rgb = _srgb_to_linear_np(rgb)
    mean_rgb = [float(rgb[..., c].mean()) for c in range(3)]
    mean_a = float(px[..., 3].mean())
    result = (mean_rgb, mean_a)
    _MEAN_RGBA_CACHE[ptr] = result
    return result


def _resolve_constant_socket(sock, depth: int = 0):
    if depth > 8:
        return None
    if not sock.is_linked:
        return _socket_rgb(sock)
    link = sock.links[0]
    src = link.from_node
    mean = _node_mean_color(src, link.from_socket)
    if mean is not None:
        return mean
    if src.bl_idname == "ShaderNodeRGB":
        v = src.outputs[0].default_value
        return [float(v[0]), float(v[1]), float(v[2])]
    if src.bl_idname == "ShaderNodeBlackbody":
        temp_sock = src.inputs.get("Temperature")
        if temp_sock is None or temp_sock.is_linked:
            return None
        return _blackbody_to_linear_rgb(_socket_f(temp_sock))
    if src.bl_idname == "ShaderNodeTexImage":
        # A real texture doesn't have a single colour — but when a caller has
        # no per-pixel path (MixShader factor, Light colour, Emission input),
        # folding to the mean pixel value is the faithful host-side reduction.
        # Callers that DO have a per-pixel path resolve the image directly via
        # `_socket_linked_image_with_chain`, so this only fires when baking
        # isn't an option.
        if src.image is None:
            return None
        from_name = link.from_socket.name
        if from_name == "Alpha":
            mean = _image_mean_rgba(src.image)
            if mean is None:
                return None
            a = mean[1]
            return [a, a, a]
        mean = _image_mean_rgba(src.image)
        if mean is None:
            return None
        return list(mean[0])
    if src.bl_idname == "ShaderNodeValToRGB":
        # ColorRamp: walk the Fac input to a scalar constant, then evaluate.
        fac_sock = src.inputs.get("Fac")
        if fac_sock is None:
            return None
        if fac_sock.is_linked:
            inner = _resolve_constant_socket(fac_sock, depth + 1)
            if inner is None:
                return None
            fac = float(inner[0])  # collapse to scalar via R channel
        else:
            fac = _socket_f(fac_sock)
        ramp_color = list(src.color_ramp.evaluate(max(0.0, min(1.0, fac))))[:3]
        return [float(c) for c in ramp_color]
    if src.bl_idname in ("ShaderNodeMix", "ShaderNodeMixRGB"):
        fac_sock, a_sock, b_sock, blend, use_clamp, ok = _mix_node_params(src)
        if ok and blend in _SUPPORTED_MIX_BLENDS:
            if fac_sock.is_linked:
                fac = _resolve_constant_scalar(fac_sock, depth + 1)
                if fac is None:
                    return None
            else:
                fac = _socket_f(fac_sock)
            a_const = _resolve_constant_socket(a_sock, depth + 1)
            b_const = _resolve_constant_socket(b_sock, depth + 1)
            if a_const is None or b_const is None:
                return None
            import numpy as np
            a = np.array([a_const], dtype=np.float32)
            b = np.array([b_const], dtype=np.float32)
            out = _apply_mix_blend(a, b, fac, blend, use_clamp)
            return [float(c) for c in out[0]]
        # Non-RGBA Mix: fall through to None. `_resolve_constant_scalar`
        # handles the FLOAT case directly.
    # Try transforms that take a single Color input (Gamma, BrightContrast,
    # Invert, HueSaturation, RGBCurve, Clamp). Walk their input to a constant
    # and apply the transform's LUT.
    candidates = _chain_candidates(src)
    if candidates is None:
        return None
    for name, inp in candidates:
        # Skip vector inputs (e.g. Mapping's "Vector"); we want colour ones.
        if inp.type not in ("RGBA", "VALUE"):
            continue
        if inp.is_linked:
            inner = _resolve_constant_socket(inp, depth + 1)
            if inner is None:
                continue
        else:
            try:
                inner = _socket_rgb(inp)
            except Exception as exc:
                _warn(
                    f"resolve_const:{_node_tag(src)}:{name}",
                    f"could not read default of socket {name!r} on "
                    f"{_node_tag(src)} ({exc}) — skipping constant-fold candidate",
                )
                continue
        xform = _node_color_transform(src, chain_input_name=name)
        if xform is None:
            return inner
        import numpy as np
        arr = np.array([[inner]], dtype=np.float32)
        arr = xform[1](arr)
        return [float(c) for c in arr[0, 0]]
    return None


def _view_dependent_ior(sock, depth: int = 0) -> float | None:
    """Return the effective Fresnel IOR of a Fresnel/LayerWeight factor
    chain, or None when the chain isn't view-angle driven.

    Blender's canonical clearcoat / "Fresnel over base" idiom wires such
    a factor into a MixShader. `_resolve_constant_scalar` would fold the
    leaf to 0 (via `_node_mean_color`) and silently drop shader2; callers
    use this IOR instead to route shader2 through the coat lobe with a
    matching Fresnel curve.

    - `ShaderNodeFresnel`: IOR input directly (default 1.45).
    - `ShaderNodeLayerWeight`: Cycles uses `eta = 1/(1-Blend)` for its
      Fresnel output, so Blend=0.5 → ior=2 (F0≈0.111).
    Math/Clamp/Mix wrappers are passed through by searching their VALUE
    inputs; if multiple branches are view-dependent we take the first
    (rare enough in practice to not matter).
    """
    if depth > 8 or not sock.is_linked:
        return None
    src = sock.links[0].from_node
    bl = src.bl_idname
    if bl == "ShaderNodeFresnel":
        ior_sock = src.inputs.get("IOR")
        if ior_sock is not None and not ior_sock.is_linked:
            return float(ior_sock.default_value)
        return 1.45
    if bl == "ShaderNodeLayerWeight":
        blend_sock = src.inputs.get("Blend")
        if blend_sock is not None and not blend_sock.is_linked:
            blend = max(0.0, min(0.9999, float(blend_sock.default_value)))
        else:
            blend = 0.5
        return 1.0 / max(1e-4, 1.0 - blend)
    if bl in ("ShaderNodeMath", "ShaderNodeClamp", "ShaderNodeMapRange") or (
        bl == "ShaderNodeMix" and getattr(src, "data_type", "RGBA") == "FLOAT"
    ):
        for i in src.inputs:
            if i.type != "VALUE":
                continue
            ior = _view_dependent_ior(i, depth + 1)
            if ior is not None:
                return ior
    return None


def _resolve_constant_scalar(sock, depth: int = 0) -> float | None:
    """Fold a VALUE socket to a single float, walking Math/Clamp/Value.

    Returns None when we hit a node we can't evaluate — no "representative
    mean" substitution for non-foldable leaves (Random, procedural noise).
    Callers must handle the None by warning and dropping the effect.
    """
    if depth > 8:
        return None
    if not sock.is_linked:
        dv = getattr(sock, "default_value", None)
        if dv is None:
            return None
        try:
            return float(dv)
        except TypeError:
            return None
    link = sock.links[0]
    src = link.from_node
    if src.bl_idname == "ShaderNodeValue":
        return float(src.outputs[0].default_value)
    if src.bl_idname == "ShaderNodeMath":
        return _eval_math_constant(src, depth + 1)
    if src.bl_idname == "ShaderNodeClamp":
        v = _resolve_constant_scalar(src.inputs["Value"], depth + 1)
        if v is None:
            return None
        mn = _resolve_constant_scalar(src.inputs["Min"], depth + 1)
        mx = _resolve_constant_scalar(src.inputs["Max"], depth + 1)
        if mn is None or mx is None:
            return None
        mode = getattr(src, "clamp_type", "MINMAX")
        if mode == "RANGE" and mn > mx:
            mn, mx = mx, mn
        return max(mn, min(mx, v))
    if src.bl_idname == "ShaderNodeMix" and getattr(src, "data_type", "RGBA") == "FLOAT":
        # Scalar Mix: inputs are scalar A/B with scalar Factor. Walk each.
        fac_sock = next((i for i in src.inputs
                         if i.name == "Factor" and i.type == "VALUE"), None)
        a_sock = next((i for i in src.inputs
                       if i.name == "A" and i.type == "VALUE"), None)
        b_sock = next((i for i in src.inputs
                       if i.name == "B" and i.type == "VALUE"), None)
        if fac_sock is None or a_sock is None or b_sock is None:
            return None
        def _val(s):
            if s.is_linked:
                return _resolve_constant_scalar(s, depth + 1)
            return float(s.default_value)
        fa, va, vb = _val(fac_sock), _val(a_sock), _val(b_sock)
        if fa is None or va is None or vb is None:
            return None
        if bool(getattr(src, "clamp_factor", True)):
            fa = max(0.0, min(1.0, fa))
        out = va * (1.0 - fa) + vb * fa
        if bool(getattr(src, "clamp_result", False)):
            out = max(0.0, min(1.0, out))
        return out
    # Approximated / procedural / per-geometry leaves — fold to the luminance
    # of the node's mean colour so a surrounding Mix Factor / Light Strength
    # can still bake.
    mean = _node_mean_color(src, link.from_socket)
    if mean is not None:
        return 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    # Any colour-producing source the RGB resolver understands (TexImage
    # mean, Invert, Mix, HueSat, RGBCurve, ...) can feed a scalar socket via
    # luminance — Cycles' Colour→Value conversion uses the same weights.
    rgb = _resolve_constant_socket(sock, depth)
    if rgb is not None:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return None


def _eval_math_constant(node, depth: int) -> float | None:
    """Evaluate a ShaderNodeMath with constant-foldable inputs. Returns None
    when any needed input is non-foldable or the op is unsupported.
    """
    op = getattr(node, "operation", "ADD")
    inputs = node.inputs
    a = _resolve_constant_scalar(inputs[0], depth) if len(inputs) > 0 else None
    b = _resolve_constant_scalar(inputs[1], depth) if len(inputs) > 1 else None
    c = _resolve_constant_scalar(inputs[2], depth) if len(inputs) > 2 else None
    if a is None:
        return None
    try:
        if op == "ADD":
            if b is None:
                return None
            r = a + b
        elif op == "SUBTRACT":
            if b is None:
                return None
            r = a - b
        elif op == "MULTIPLY":
            if b is None:
                return None
            r = a * b
        elif op == "DIVIDE":
            if b is None:
                return None
            r = a / b if b != 0 else 0.0
        elif op == "MULTIPLY_ADD":
            if b is None or c is None:
                return None
            r = a * b + c
        elif op == "POWER":
            if b is None:
                return None
            r = a ** b if a >= 0 or float(b).is_integer() else 0.0
        elif op == "LOGARITHM":
            if b is None or a <= 0 or b <= 0 or b == 1:
                return None
            r = math.log(a, b)
        elif op == "MINIMUM":
            if b is None:
                return None
            r = min(a, b)
        elif op == "MAXIMUM":
            if b is None:
                return None
            r = max(a, b)
        elif op == "LESS_THAN":
            if b is None:
                return None
            r = 1.0 if a < b else 0.0
        elif op == "GREATER_THAN":
            if b is None:
                return None
            r = 1.0 if a > b else 0.0
        elif op == "COMPARE":
            if b is None:
                return None
            eps = c if c is not None else 0.0
            r = 1.0 if abs(a - b) <= eps else 0.0
        elif op == "MODULO":
            if b is None:
                return None
            r = math.fmod(a, b) if b != 0 else 0.0
        elif op == "SMOOTH_MIN":
            if b is None:
                return None
            r = min(a, b)
        elif op == "SMOOTH_MAX":
            if b is None:
                return None
            r = max(a, b)
        elif op == "ARCTAN2":
            if b is None:
                return None
            r = math.atan2(a, b)
        elif op == "SQRT":
            r = math.sqrt(max(a, 0.0))
        elif op == "INVERSE_SQRT":
            r = 1.0 / math.sqrt(a) if a > 0 else 0.0
        elif op == "ABSOLUTE":
            r = abs(a)
        elif op == "EXPONENT":
            r = math.exp(a)
        elif op == "SIGN":
            r = float((a > 0) - (a < 0))
        elif op == "ROUND":
            r = float(round(a))
        elif op == "FLOOR":
            r = math.floor(a)
        elif op == "CEIL":
            r = math.ceil(a)
        elif op == "FRACT":
            r = a - math.floor(a)
        elif op == "TRUNC":
            r = math.trunc(a)
        elif op == "SINE":
            r = math.sin(a)
        elif op == "COSINE":
            r = math.cos(a)
        elif op == "TANGENT":
            r = math.tan(a)
        elif op == "ARCSINE":
            r = math.asin(max(-1.0, min(1.0, a)))
        elif op == "ARCCOSINE":
            r = math.acos(max(-1.0, min(1.0, a)))
        elif op == "ARCTANGENT":
            r = math.atan(a)
        elif op == "HYPERBOLIC_SINE":
            r = math.sinh(a)
        elif op == "HYPERBOLIC_COSINE":
            r = math.cosh(a)
        elif op == "HYPERBOLIC_TANGENT":
            r = math.tanh(a)
        elif op == "RADIANS":
            r = math.radians(a)
        elif op == "DEGREES":
            r = math.degrees(a)
        else:
            _warn(
                f"math-fold-op:{op}",
                f"{_node_tag(node)}: op={op!r} can't be constant-folded — "
                f"chain stopped at this node",
            )
            return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if getattr(node, "use_clamp", False):
        r = max(0.0, min(1.0, r))
    return float(r)


def _blackbody_to_linear_rgb(kelvin: float) -> list[float]:
    """Tanner-Helland blackbody → sRGB approximation, then sRGB→linear.

    Matches Blender's Blackbody node within a few percent in the 1000-12000K
    range, which is enough for a baked constant going into a Mix or BSDF.
    """
    t = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    rgb_srgb = [max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b)]

    def s2l(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return [s2l(c) for c in rgb_srgb]


_IDENTITY_UV = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _first_teximage_uv_transform(node_tree) -> list[float] | None:
    """Return the first non-identity UV affine attached to a TexImage's Vector
    input in the tree, or None. Scanning the TexImage nodes directly avoids
    picking up Mapping nodes that only drive procedurals (e.g. Noise Texture
    for bump) — applying such a Mapping to an unlinked TexImage would stretch
    it spuriously.
    """
    if node_tree is None:
        return None
    for n in node_tree.nodes:
        if n.bl_idname != "ShaderNodeTexImage":
            continue
        affine = _tex_node_uv_transform(n)
        if affine != list(_IDENTITY_UV):
            return affine
    return None


def _invert_affine_2x3(m: list[float]) -> list[float]:
    """Invert a row-major 2x3 affine matrix [a, b, tx, c, d, ty].

    Non-invertible inputs (det ≈ 0) return the identity so the caller gets a
    safe fallback instead of NaNs.
    """
    a, b, tx, c, d, ty = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        return list(_IDENTITY_UV)
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    return [inv_a, inv_b, -(inv_a * tx + inv_b * ty),
            inv_c, inv_d, -(inv_c * tx + inv_d * ty)]


def _mapping_to_affine(node) -> list[float]:
    """Convert a Mapping node to a 2x3 affine matrix over UV.

    Supports POINT (apply transform to input coords) and TEXTURE (apply the
    inverse — see Blender's vector_math_mapping). VECTOR/NORMAL fall back to
    identity with a warning.
    """
    if node is None:
        return list(_IDENTITY_UV)
    vector_type = getattr(node, "vector_type", "POINT")
    if vector_type not in ("POINT", "TEXTURE"):
        _warn(
            f"map-vtype:{node.as_pointer()}",
            f"Mapping {_node_tag(node)}: vector_type={vector_type!r} not "
            f"supported (only POINT/TEXTURE) — UV transform ignored",
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
    point = [sx * c, -sy * s, tx,
             sx * s,  sy * c, ty]
    if vector_type == "TEXTURE":
        # TEXTURE applies the inverse transform — see blender's node_shader_mapping.
        return _invert_affine_2x3(point)
    return point


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
    """Return just the image feeding a non-base-color socket.

    Kept as a thin wrapper for callers that don't bake the chain (currently
    only the alpha-cutout path in `_from_principled`). Prefer
    `_export_linked_scalar_texture` for roughness / metallic / bump inputs.
    """
    img, _ = _socket_linked_image_with_chain(sock, depth)
    return img


def _export_linked_scalar_texture(sock, writer, textures):
    """Resolve a scalar input socket to a (linear) baked texture.

    The chain transforms still apply per-channel; the GPU only reads the R
    channel for scalar inputs (roughness/metallic/bump), so baking RGB is
    fine — R after the bake is what the GPU will sample. Returns texture id
    or None.
    """
    img, chain = _socket_linked_image_with_chain(sock)
    if img is None:
        return None
    return export_image_texture(img, writer, textures, "linear", chain=chain)


_CONSTANT_LEAF_NODES = frozenset((
    # Exact-constant sources: `_socket_constant_rgb` folds these into bakeable
    # Mix operands.
    "ShaderNodeRGB",
    "ShaderNodeBlackbody",
    "ShaderNodeWavelength",
    "ShaderNodeValue",
    # Per-geometry / per-object / procedural signals we can't bake at export
    # time. `_node_mean_color` gives each a neutral constant so the enclosing
    # Mix / Emission / MixShader can still fold — the stochastic variation is
    # lost but the surrounding colour chain survives instead of the whole
    # texture being dropped. Kept in this set so chain walking treats them as
    # dead-ends silently (no "not recognised" warning).
    "ShaderNodeAttribute",
    "ShaderNodeObjectInfo",
    "ShaderNodeNewGeometry",
    "ShaderNodeTexCoord",
    "ShaderNodeTexNoise",
    "ShaderNodeTexWave",
    "ShaderNodeTexVoronoi",
    "ShaderNodeTexMusgrave",
    "ShaderNodeTexMagic",
    "ShaderNodeTexChecker",
    "ShaderNodeTexBrick",
    "ShaderNodeTexGradient",
    "ShaderNodeTexWhiteNoise",
    "ShaderNodeTexSky",
    "ShaderNodeTexEnvironment",
    "ShaderNodeWireframe",
    "ShaderNodeLightPath",
    # Incidence-dependent — the renderer's own Principled Fresnel replaces
    # these; folding to the normal-incidence value steers Mix/MixShader paths
    # to the "body" side (shader1), which matches how these factors are
    # typically wired (Fresnel-blended clearcoat / edge tint).
    "ShaderNodeFresnel",
    "ShaderNodeLayerWeight",
))


def _node_mean_color(node, output_sock) -> list[float] | None:
    """Best-effort constant colour for a non-chain node's output socket.

    Returns None when the node isn't one we know how to approximate. Used by
    the constant-folding path when a Mix / Emission / Light Color chain ends
    on an Attribute / ObjectInfo / procedural / Fresnel leaf — folding to
    their mean lets the rest of the chain collapse instead of being dropped.
    """
    bl = node.bl_idname
    name = output_sock.name if output_sock is not None else ""
    if bl == "ShaderNodeAttribute":
        # Missing attribute at render-time defaults to white; use that so a
        # surrounding Mix passes the other operand through untinted.
        if name == "Alpha":
            return [1.0, 1.0, 1.0]
        if name in ("Fac", "Vector"):
            return [0.0, 0.0, 0.0]
        return [1.0, 1.0, 1.0]
    if bl == "ShaderNodeObjectInfo":
        if name == "Color":
            return [1.0, 1.0, 1.0]
        if name == "Random":
            return [0.5, 0.5, 0.5]
        if name == "Alpha":
            return [1.0, 1.0, 1.0]
        return [0.0, 0.0, 0.0]
    if bl == "ShaderNodeNewGeometry":
        return [0.0, 0.0, 0.0] if "Position" in name else [0.5, 0.5, 0.5]
    if bl == "ShaderNodeTexCoord":
        return [0.5, 0.5, 0.5]
    if bl in ("ShaderNodeTexNoise", "ShaderNodeTexMusgrave",
              "ShaderNodeTexMagic", "ShaderNodeTexWhiteNoise",
              "ShaderNodeTexWave", "ShaderNodeTexVoronoi",
              "ShaderNodeTexGradient"):
        return [0.5, 0.5, 0.5]
    if bl == "ShaderNodeTexChecker":
        def _col(sock):
            if sock is None or sock.is_linked:
                return [0.5, 0.5, 0.5]
            v = sock.default_value
            return [float(v[0]), float(v[1]), float(v[2])]
        a = _col(node.inputs.get("Color1"))
        b = _col(node.inputs.get("Color2"))
        return [0.5 * (a[i] + b[i]) for i in range(3)]
    if bl == "ShaderNodeTexBrick":
        def _col(sock, default):
            if sock is None or sock.is_linked:
                return default
            v = sock.default_value
            return [float(v[0]), float(v[1]), float(v[2])]
        c1 = _col(node.inputs.get("Color1"), [0.8, 0.8, 0.8])
        c2 = _col(node.inputs.get("Color2"), [0.2, 0.2, 0.2])
        mortar = _col(node.inputs.get("Mortar"), [0.0, 0.0, 0.0])
        # Roughly 80% brick, 20% mortar by default geometry.
        return [0.4 * c1[i] + 0.4 * c2[i] + 0.2 * mortar[i] for i in range(3)]
    if bl == "ShaderNodeFresnel":
        # ≈ Schlick at normal incidence for IOR 1.5.
        return [0.04, 0.04, 0.04]
    if bl == "ShaderNodeLayerWeight":
        # "Fresnel"/"Facing" outputs are 0 at normal incidence; folding to 0
        # routes MixShader(factor=LayerWeight) through shader1.
        return [0.0, 0.0, 0.0]
    if bl == "ShaderNodeWireframe":
        # Per-pixel edge detector: 0 except on triangle edges, so the mean is
        # effectively 0. Folding to 0 picks shader1 in the common
        # MixShader(Wireframe, edge, body) wiring.
        return [0.0, 0.0, 0.0]
    if bl == "ShaderNodeLightPath":
        # "Is Camera Ray" is 1 for primary rays (what the user sees); the
        # other path flags (Shadow/Diffuse/Glossy/Transmission) are 0 for
        # camera rays. "Ray Depth"/"Ray Length" fold to 0 / 0 as a neutral.
        if name == "Is Camera Ray":
            return [1.0, 1.0, 1.0]
        return [0.0, 0.0, 0.0]
    if bl in ("ShaderNodeTexSky", "ShaderNodeTexEnvironment"):
        # Without evaluating the sky model, daylight-sky zenith colour is a
        # reasonable average for folding into a MixShader / Emission.
        return [0.5, 0.65, 0.85]
    return None


# Stack of ShaderNodeGroup nodes currently being resolved by _from_group.
# Lets chain traversal jump back out through NodeGroupInput → parent socket
# without threading the stack through every BSDF helper.
_GROUP_STACK: list = []


def _socket_linked_image_with_chain(sock, depth: int = 0, group_stack=None):
    """Return (image, transforms_chain) feeding a color socket.

    `transforms_chain` is a tuple of (id_tuple, apply_fn) pairs that should be
    applied to the texture's pixels in order (texture-first, BSDF-last) to
    reproduce the effect of colour-modifying nodes between the texture and
    `sock`. Nodes whose effect can't be baked are passed through with a
    warning via `_warn`.

    `group_stack` tracks nested ShaderNodeGroup traversals so a chain that
    enters a NodeGroupInput can follow back out to the corresponding socket on
    the parent group node. Defaults to the live `_GROUP_STACK` so BSDF helpers
    don't need to thread it through.
    """
    if group_stack is None:
        group_stack = tuple(_GROUP_STACK)
    if not sock.is_linked:
        return None, ()
    if depth > 12:
        _warn(
            f"depth:{id(sock)}",
            f"socket {sock.name!r}: chain deeper than 12 nodes — giving up",
        )
        return None, ()
    link = sock.links[0]
    src = link.from_node
    from_socket = link.from_socket
    if src.bl_idname == "ShaderNodeTexImage":
        if src.image is None:
            _warn(
                f"texnoimage:{src.as_pointer()}",
                f"TexImage {src.name!r} has no image assigned",
            )
            return None, ()
        return src.image, ()
    if src.bl_idname in _CONSTANT_LEAF_NODES:
        # Constant-colour source — there's no texture behind it. Caller's bake
        # path (e.g. Mix) handles the constant via _socket_constant_rgb.
        return None, ()
    if src.bl_idname == "ShaderNodeGroup":
        return _follow_into_group(src, from_socket, depth, group_stack)
    if src.bl_idname == "NodeGroupInput":
        return _follow_out_of_group(from_socket, depth, group_stack)
    if src.bl_idname in ("ShaderNodeMix", "ShaderNodeMixRGB"):
        # Both sides may be textures — collapse the whole Mix into one offline
        # baked texture so the chain returning from here is empty.
        prebaked = _try_premix_two_textures(src, group_stack)
        if prebaked is not None:
            return prebaked, ()
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
        img, sub_chain = _socket_linked_image_with_chain(inp, depth + 1, group_stack)
        if img is None:
            continue
        xform = _node_color_transform(src, chain_input_name=name)
        # Inner (closer to texture) first, then outer.
        chain = sub_chain + ((xform,) if xform is not None else ())
        return img, chain
    return None, ()


def _follow_into_group(group_node, output_sock, depth, group_stack):
    """Enter a ShaderNodeGroup: find the NodeGroupOutput input matching the
    socket whose value the chain wants, then keep following backwards.
    """
    tree = group_node.node_tree
    if tree is None:
        return None, ()
    group_output = next(
        (n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"), None
    )
    if group_output is None:
        return None, ()
    inner = None
    for inp in group_output.inputs:
        if inp.name == output_sock.name and (inp.type == output_sock.type
                                              or inp.type in ("RGBA", "VALUE")):
            inner = inp
            break
    if inner is None:
        return None, ()
    return _socket_linked_image_with_chain(
        inner, depth + 1, group_stack + (group_node,)
    )


def _follow_out_of_group(output_sock, depth, group_stack):
    """Exit a ShaderNodeGroup via NodeGroupInput: look up the matching socket
    on the most recent parent group node and continue the chain there.
    """
    if not group_stack:
        _warn(
            f"groupin-orphan:{output_sock.name}",
            f"NodeGroupInput.{output_sock.name!r} encountered without an "
            f"outer group context — chain stopped",
        )
        return None, ()
    parent = group_stack[-1]
    external = parent.inputs.get(output_sock.name)
    if external is None:
        return None, ()
    if external.is_linked:
        return _socket_linked_image_with_chain(external, depth + 1, group_stack[:-1])
    # Unlinked: the group's external default could still be a baked-in colour,
    # but there's no texture chain to walk. Returning None lets the caller
    # fall back to its own constant handling (`_socket_constant_rgb`).
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
    if bl == "ShaderNodeMath":
        return _math_transform(node, chain_input_name)
    return None


def _math_transform(node, chain_input_name=None):
    """Bake a scalar Math node into an RGB transform when every non-chain
    input folds to a constant.

    Acts on the three `Value` sockets in order (socket names are all "Value",
    so identify by `identifier` — Blender suffixes them "Value_001" etc.).
    The chain feeds one socket with a colour; we convert the other inputs to
    scalars and run the op per-channel so bumps / factors keep their
    pre-chain scale (classroom's paintedCeiling multiplies the inverted
    ceiling_AO by 0.8 — without this the factor saturates to 1 and the baked
    texture darkens twice as much as Cycles).
    """
    import numpy as np
    op = getattr(node, "operation", "ADD")
    use_clamp = bool(getattr(node, "use_clamp", False))
    inputs = list(node.inputs)
    chain_idx = next(
        (i for i, inp in enumerate(inputs) if inp.identifier == chain_input_name),
        -1,
    )
    if chain_idx < 0:
        chain_idx = next((i for i, inp in enumerate(inputs) if inp.is_linked), -1)
    if chain_idx < 0:
        return None
    other_vals: list[float] = []
    for i, inp in enumerate(inputs):
        if i == chain_idx:
            continue
        v = _resolve_constant_scalar(inp) if inp.is_linked else _socket_f(inp)
        if v is None:
            _warn(
                f"math-chain:{node.as_pointer()}:{op}",
                f"{_node_tag(node)}: op={op} non-chain input {inp.name!r} not "
                f"foldable to a constant — effect not baked",
            )
            return None
        other_vals.append(v)

    a_name = inputs[chain_idx].identifier  # stable key across Blender versions
    id_tuple = (
        "Math", op, use_clamp, a_name,
        tuple(round(float(v), 6) for v in other_vals),
    )

    def _finish(out):
        if use_clamp:
            out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)

    # Chain side is always the "A" operand in the op; the other constant(s)
    # become "B" (and "C" for ternary ops). Non-commutative ops split on
    # chain_idx so SUBTRACT(chain=0) stays "chain - k" but SUBTRACT(chain=1)
    # becomes "k - chain".
    b = other_vals[0] if other_vals else 0.0
    if op == "ADD":
        def apply(rgb):
            return _finish(rgb + b)
    elif op == "SUBTRACT":
        if chain_idx == 0:
            def apply(rgb):
                return _finish(rgb - b)
        else:
            def apply(rgb):
                return _finish(b - rgb)
    elif op == "MULTIPLY":
        def apply(rgb):
            return _finish(rgb * b)
    elif op == "DIVIDE":
        if chain_idx == 0:
            def apply(rgb):
                denom = b if abs(b) > 1e-12 else 1.0
                return _finish(rgb / denom)
        else:
            def apply(rgb):
                return _finish(np.where(rgb == 0.0, 0.0, b / np.where(rgb == 0.0, 1.0, rgb)))
    elif op == "POWER":
        if chain_idx == 0:
            def apply(rgb):
                return _finish(np.power(np.maximum(rgb, 0.0), b))
        else:
            def apply(rgb):
                return _finish(np.power(max(b, 0.0), rgb))
    elif op == "MULTIPLY_ADD" and len(other_vals) >= 2:
        c = other_vals[1]
        def apply(rgb):
            return _finish(rgb * b + c)
    elif op == "MINIMUM":
        def apply(rgb):
            return _finish(np.minimum(rgb, b))
    elif op == "MAXIMUM":
        def apply(rgb):
            return _finish(np.maximum(rgb, b))
    else:
        _warn(
            f"math-op:{op}",
            f"{_node_tag(node)}: op={op!r} not baked — chain short-circuited",
        )
        return None
    return id_tuple, apply


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


_SUPPORTED_MIX_BLENDS = frozenset((
    "MIX", "MULTIPLY", "ADD", "SUBTRACT", "SCREEN", "DIVIDE", "DIFFERENCE",
    "DARKEN", "LIGHTEN", "OVERLAY", "SOFT_LIGHT", "LINEAR_LIGHT",
))


def _find_base_color_attribute(sock, depth: int = 0) -> str | None:
    """Return the attribute name if `sock` is driven — directly or through any
    chain of Mix / MixRGB nodes — by a ShaderNodeAttribute. Otherwise None.

    Used by the BSDF helpers to spot a per-vertex colour mask sitting in the
    Base Color chain (e.g. classroom's `wallClock_darkWood` wood mask). The
    renderer multiplies base_color × vertex_color at the hit point, so
    callers set base_color=(1,1,1) and let the per-hit attribute drive the
    colour unmodulated — no tint inference, no procedural substitution, no
    mean-colour fallback. When the artist also blended in a procedural or
    texture alongside the Attribute, that side is dropped: picking up the
    spatial vertex-mask signal is more faithful than synthesising a constant
    stand-in for the non-attribute side.
    """
    if depth > 6 or not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src.bl_idname == "ShaderNodeAttribute":
        name = getattr(src, "attribute_name", "") or ""
        return name or None
    if src.bl_idname in ("ShaderNodeMix", "ShaderNodeMixRGB"):
        _fac, a_sock, b_sock, _blend, _clamp, ok = _mix_node_params(src)
        if not ok:
            return None
        # Prefer B (Blender convention: shader A = body, shader B = overlay)
        # but try both so the detection is symmetric.
        return (
            _find_base_color_attribute(b_sock, depth + 1)
            or _find_base_color_attribute(a_sock, depth + 1)
        )
    return None


def _mix_node_params(node):
    """Return `(fac_sock, a_sock, b_sock, blend, use_clamp, ok)` for a Mix node,
    or `(*_, False)` if the node isn't a bakeable RGBA-mode Mix.
    """
    if node.bl_idname == "ShaderNodeMixRGB":
        fac_sock = node.inputs["Fac"]
        a_sock = node.inputs["Color1"]
        b_sock = node.inputs["Color2"]
        blend = node.blend_type
        use_clamp = bool(getattr(node, "use_clamp", False))
        return fac_sock, a_sock, b_sock, blend, use_clamp, True
    if getattr(node, "data_type", "RGBA") != "RGBA":
        return None, None, None, "MIX", False, False
    fac_sock, a_sock, b_sock = _mix_rgba_sockets(node)
    if fac_sock is None or a_sock is None or b_sock is None:
        return None, None, None, "MIX", False, False
    blend = getattr(node, "blend_type", "MIX")
    use_clamp = bool(
        getattr(node, "clamp_result", False) or getattr(node, "use_clamp", False)
    )
    return fac_sock, a_sock, b_sock, blend, use_clamp, True


def _apply_mix_blend(col1, col2, fac, blend: str, use_clamp: bool):
    """Blender ramp_blend applied to two arrays (or array + broadcast scalar).

    `fac` is either a Python float or a numpy array broadcastable against
    `col1`/`col2` (e.g. `(h, w, 1)` for a textured Factor).
    """
    import numpy as np
    facm = 1.0 - fac
    if blend == "MIX":
        out = col1 * facm + col2 * fac
    elif blend == "MULTIPLY":
        out = col1 * (facm + fac * col2)
    elif blend == "ADD":
        out = col1 + fac * col2
    elif blend == "SUBTRACT":
        out = col1 - fac * col2
    elif blend == "SCREEN":
        out = 1.0 - (facm + fac * (1.0 - col2)) * (1.0 - col1)
    elif blend == "DIVIDE":
        col2_safe = np.where(col2 == 0.0, 1.0, col2)
        out = np.where(col2 == 0.0, col1, facm * col1 + fac * col1 / col2_safe)
    elif blend == "DIFFERENCE":
        out = facm * col1 + fac * np.abs(col1 - col2)
    elif blend == "DARKEN":
        out = facm * col1 + fac * np.minimum(col1, col2)
    elif blend == "LIGHTEN":
        # Blender's asymmetric formula: max(col1, col2*fac).
        out = np.maximum(col1, col2 * fac)
    elif blend == "OVERLAY":
        lo = col1 * (facm + 2.0 * fac * col2)
        hi = 1.0 - (facm + 2.0 * fac * (1.0 - col2)) * (1.0 - col1)
        out = np.where(col1 < 0.5, lo, hi)
    elif blend == "SOFT_LIGHT":
        scr = 1.0 - (1.0 - col2) * (1.0 - col1)
        out = facm * col1 + fac * ((1.0 - col1) * col2 * col1 + col1 * scr)
    elif blend == "LINEAR_LIGHT":
        out = col1 + fac * (2.0 * col2 - 1.0)
    else:
        raise AssertionError(f"unreachable blend: {blend}")
    if use_clamp:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _mix_transform(node, chain_input_name):
    """Bake a Mix node when the non-chain input is a constant colour.

    For two-texture mixes (both inputs driven by chains) the pre-bake step in
    `_try_premix_two_textures` collapses the entire Mix into a synthetic
    texture, so the chain reaching this point sees an unlinked operand.
    """
    import numpy as np
    fac_sock, a_sock, b_sock, blend, use_clamp, ok = _mix_node_params(node)
    if not ok:
        _warn(
            f"mix-dt2:{node.as_pointer()}",
            f"{_node_tag(node)}: data_type not RGBA / sockets unresolved — "
            f"effect not baked",
        )
        return None
    is_a_chain = chain_input_name in ("A", "Color1")

    if blend not in _SUPPORTED_MIX_BLENDS:
        _warn(
            f"mix-blend:{node.as_pointer()}:{blend}",
            f"{_node_tag(node)}: blend_type={blend!r} not supported — "
            f"effect passed through without baking",
        )
        return None
    if fac_sock.is_linked:
        fac = _resolve_constant_scalar(fac_sock)
        if fac is None:
            _warn(
                f"mix-fac-linked:{node.as_pointer()}",
                f"{_node_tag(node)}: Factor is linked and not foldable to a "
                f"constant — effect not baked",
            )
            return None
    else:
        fac = _socket_f(fac_sock)
    other_sock = b_sock if is_a_chain else a_sock
    other_const = _socket_constant_rgb(other_sock)
    if other_const is None:
        _warn(
            f"mix-other-linked:{node.as_pointer()}",
            f"{_node_tag(node)}: non-chain input {other_sock.name!r} is "
            f"driven by another chain (two-texture mix) — effect not baked",
        )
        return None
    other_rgb = np.array(other_const, dtype=np.float32)
    id_tuple = (
        "Mix", blend, is_a_chain, round(fac, 6),
        tuple(round(float(c), 6) for c in other_rgb.tolist()),
        use_clamp,
    )

    if is_a_chain:
        def apply(rgb):
            return _apply_mix_blend(rgb, other_rgb, fac, blend, use_clamp)
    else:
        def apply(rgb):
            return _apply_mix_blend(other_rgb, rgb, fac, blend, use_clamp)

    return id_tuple, apply


def _try_premix_two_textures(mix_node, group_stack):
    """Pre-bake a Mix into one `_PreBakedTexture` when the shape can't be
    collapsed by the in-chain `_mix_transform` path.

    Handles two cases:
      - Both colour sides are textures (const factor): the classic two-texture
        mix; the in-chain path can't bake it because it only threads one side.
      - Factor is linked to a texture chain: requires per-pixel fac, so we
        premix regardless of whether the other colour side is a texture or a
        resolvable constant.

    Returns None when the Mix is in an unsupported mode, the blend isn't
    bakeable, a required side (colour or fac) can't be resolved, or there's
    nothing texture-shaped left to bake (all operands are constants, which the
    constant-folding path will pick up instead).
    """
    import numpy as np
    fac_sock, a_sock, b_sock, blend, use_clamp, ok = _mix_node_params(mix_node)
    if not ok or blend not in _SUPPORTED_MIX_BLENDS:
        return None
    img_a, chain_a = _socket_linked_image_with_chain(a_sock, group_stack=group_stack)
    img_b, chain_b = _socket_linked_image_with_chain(b_sock, group_stack=group_stack)
    if fac_sock.is_linked:
        img_f, chain_f = _socket_linked_image_with_chain(
            fac_sock, group_stack=group_stack
        )
        if img_f is not None:
            fac_const = None
            # A textured factor forces a full premix, so it's fine if one
            # colour side is only a resolvable constant.
            const_a = None if img_a is not None else _socket_constant_rgb(a_sock)
            const_b = None if img_b is not None else _socket_constant_rgb(b_sock)
            if img_a is None and const_a is None:
                return None
            if img_b is None and const_b is None:
                return None
        else:
            # No texture behind the factor chain; fold Math/Clamp/Value to a
            # real constant. `_mix_transform` picks up tex-vs-const shapes, so
            # only claim the premix when both colour sides are textures.
            chain_f = ()
            fac_scalar = _resolve_constant_scalar(fac_sock)
            if fac_scalar is None or img_a is None or img_b is None:
                return None
            fac_const = fac_scalar
            const_a = const_b = None
    else:
        img_f, chain_f = None, ()
        fac_const = _socket_f(fac_sock)
        # Constant factor: only worth a full premix when both sides are
        # textures. Otherwise the cheaper in-chain `_mix_transform` path
        # already handles tex-vs-const and we'd be baking an extra copy.
        if img_a is None or img_b is None:
            return None
        const_a = const_b = None

    chain_a_id = tuple(x[0] for x in chain_a)
    chain_b_id = tuple(x[0] for x in chain_b)
    chain_f_id = tuple(x[0] for x in chain_f)

    def _src_id(src):
        return src.cache_key if isinstance(src, _PreBakedTexture) else f"img:{src.name}"

    def _side_key(img, chain_id, const):
        if img is not None:
            return f"tex:{_src_id(img)}|{chain_id!r}"
        return f"const:{tuple(round(float(c), 6) for c in const)}"

    fac_key = (
        f"tex:{_src_id(img_f)}|{chain_f_id!r}" if img_f is not None
        else f"const:{round(fac_const, 6)}"
    )
    cache_key = (
        f"__mix2tex__|{_side_key(img_a, chain_a_id, const_a)}|"
        f"{_side_key(img_b, chain_b_id, const_b)}|{blend}|{fac_key}|{use_clamp}"
    )
    hit = _PREMIX_CACHE.get(cache_key)
    if hit is not None:
        _STATS["premix_cache_hits"] += 1
        return hit
    _STATS["premix_cache_misses"] += 1

    baked_a = _bake_source_to_linear(img_a, chain_a) if img_a is not None else None
    baked_b = _bake_source_to_linear(img_b, chain_b) if img_b is not None else None
    baked_f = _bake_source_to_linear(img_f, chain_f) if img_f is not None else None
    sizes = [(b[2], b[1]) for b in (baked_a, baked_b, baked_f) if b is not None]
    # Guaranteed non-empty: at least one of the three must be a texture, else
    # we'd have returned above.
    out_h = max(h for h, _ in sizes)
    out_w = max(w for _, w in sizes)

    def _resolve_side(baked, const):
        if baked is not None:
            return _resample_bilinear(baked[0], out_h, out_w)
        return np.broadcast_to(
            np.asarray(const, dtype=np.float32), (out_h, out_w, 3)
        ).astype(np.float32, copy=True)

    rgb_a = _resolve_side(baked_a, const_a)
    rgb_b = _resolve_side(baked_b, const_b)
    if baked_f is not None:
        rgb_f = _resample_bilinear(baked_f[0], out_h, out_w)
        # VALUE sockets receive an RGB as Rec.709 luminance (matches Cycles'
        # rgb_to_bw). Keep a trailing axis so the (h, w, 1) factor broadcasts
        # against the (h, w, 3) colour operands in `_apply_mix_blend`.
        fac = (
            np.float32(0.2126) * rgb_f[..., 0:1]
            + np.float32(0.7152) * rgb_f[..., 1:2]
            + np.float32(0.0722) * rgb_f[..., 2:3]
        ).astype(np.float32)
    else:
        fac = fac_const
    mixed = _apply_mix_blend(rgb_a, rgb_b, fac, blend, use_clamp)
    result = _PreBakedTexture(mixed, out_w, out_h, cache_key)
    _PREMIX_CACHE[cache_key] = result
    return result


_GRAPH_MAX_NODES = 32

_GRAPH_BLEND_MODES = frozenset((
    "mix", "multiply", "add", "subtract", "screen", "divide", "difference",
    "darken", "lighten", "overlay", "soft_light", "linear_light",
))

_GRAPH_MATH_OPS = frozenset((
    "add", "subtract", "multiply", "divide", "power",
    "multiply_add", "minimum", "maximum",
))


def _tex_node_uv_transform(tex_node) -> list[float]:
    """Extract the UV affine feeding a TexImage.Vector. Returns identity when
    the Vector is unlinked (TexImage falls back to the mesh's default UV)."""
    vec_sock = tex_node.inputs.get("Vector")
    if vec_sock is None or not vec_sock.is_linked:
        return list(_IDENTITY_UV)
    src = vec_sock.links[0].from_node
    if src.bl_idname == "ShaderNodeMapping":
        return _mapping_to_affine(src)
    if src.bl_idname == "ShaderNodeTexCoord":
        return list(_IDENTITY_UV)
    # Anything else (procedurals, wrangler groups) loses the transform — the
    # graph evaluator still works, just without its UV tweak.
    return list(_IDENTITY_UV)


def _try_emit_color_graph(sock, writer, textures, group_stack=None, vc_attrs=None) -> dict | None:
    """Walk the shader graph feeding `sock` and emit a `color_graph` dict that
    the GPU evaluator can run per-pixel. Returns None if the graph uses a
    node we can't yet translate (ShaderNodeValToRGB, ShaderNodeTexNoise, ...);
    callers fall back to the existing bake path in that case.

    Only handles a focused subset today — ImageTex, Mix, Invert, Math,
    HueSat (treated as pass-through until HSV lands on the GPU), plus
    constant-RGB leaves. classroom's paintedCeiling comes out as a 7-node
    graph that mirrors the Cycles shader exactly except for the HSV
    saturation bump.

    `vc_attrs`, when provided, is a list populated with the vertex colour
    attribute names referenced by emitted `vertex_color` nodes. Callers use
    this to set `_vertex_color_attr` on the material dict so the mesh
    exporter ships the matching per-vertex colour blob.
    """
    if group_stack is None:
        group_stack = tuple(_GROUP_STACK)
    if not sock.is_linked:
        # Just a constant colour — no graph needed, caller can use base_color.
        return None
    nodes: list[dict] = []
    memo: dict = {}

    def emit(emit_sock) -> int | None:
        if len(nodes) >= _GRAPH_MAX_NODES:
            return None
        if not emit_sock.is_linked:
            rgb = _socket_rgb(emit_sock)
            idx = len(nodes)
            nodes.append({"type": "const", "rgb": rgb})
            return idx
        link = emit_sock.links[0]
        src = link.from_node
        key = (src.as_pointer(), link.from_socket.name)
        if key in memo:
            return memo[key]

        bl = src.bl_idname
        if bl == "ShaderNodeTexImage":
            if src.image is None:
                return None
            tex_id = export_image_texture(src.image, writer, textures, "srgb")
            uv = _tex_node_uv_transform(src)
            idx = len(nodes)
            nodes.append({"type": "image_tex", "tex": tex_id, "uv": uv})
            memo[key] = idx
            return idx

        if bl in ("ShaderNodeMix", "ShaderNodeMixRGB"):
            fac_sock, a_sock, b_sock, blend, clamp, ok = _mix_node_params(src)
            if not ok:
                return None
            blend_lc = blend.lower()
            if blend_lc not in _GRAPH_BLEND_MODES:
                return None
            ai = emit(a_sock)
            if ai is None:
                return None
            bi = emit(b_sock)
            if bi is None:
                return None
            if fac_sock.is_linked:
                # Prefer a real per-pixel node so the factor can vary with
                # texture content. Fall back to host-side constant folding
                # so chains like `ObjectInfo.Random → Math(GREATER_THAN, …)`
                # (unsupported by the graph evaluator) collapse to the
                # representative constant instead of dropping the whole
                # graph — classroom's schoolDesk_wood picks its texture
                # variant this way.
                fi = emit(fac_sock)
                if fi is not None:
                    fac_spec: float | dict = {"node": fi}
                else:
                    fac_const = _resolve_constant_scalar(fac_sock)
                    if fac_const is None:
                        return None
                    fac_spec = fac_const
            else:
                fac_spec = _socket_f(fac_sock)
            idx = len(nodes)
            nodes.append({
                "type": "mix", "a": ai, "b": bi,
                "fac": fac_spec, "blend": blend_lc, "clamp": bool(clamp),
            })
            memo[key] = idx
            return idx

        if bl == "ShaderNodeInvert":
            fac_sock = src.inputs.get("Fac")
            col_sock = src.inputs.get("Color")
            if col_sock is None:
                return None
            if fac_sock is not None and fac_sock.is_linked:
                _warn(
                    f"invert_textured_fac:{_node_tag(src)}",
                    f"{_node_tag(src)}: textured Fac on Invert is not yet "
                    "supported — node will be ignored in the color graph",
                )
                return None
            fac = _socket_f(fac_sock) if fac_sock is not None else 1.0
            ci = emit(col_sock)
            if ci is None:
                return None
            idx = len(nodes)
            nodes.append({"type": "invert", "input": ci, "fac": fac})
            memo[key] = idx
            return idx

        if bl == "ShaderNodeMath":
            op = getattr(src, "operation", "MULTIPLY").lower()
            if op not in _GRAPH_MATH_OPS:
                return None
            use_clamp = bool(getattr(src, "use_clamp", False))
            inputs = list(src.inputs)
            chain_idx = next(
                (i for i, inp in enumerate(inputs) if inp.is_linked),
                -1,
            )
            if chain_idx < 0:
                return None
            ci = emit(inputs[chain_idx])
            if ci is None:
                return None
            # Non-chain inputs must fold to constants.
            others: list[float] = []
            for i, inp in enumerate(inputs):
                if i == chain_idx:
                    continue
                if inp.is_linked:
                    v = _resolve_constant_scalar(inp)
                    if v is None:
                        return None
                else:
                    v = _socket_f(inp)
                others.append(v)
            b = others[0] if others else 0.0
            c = others[1] if len(others) > 1 else 0.0
            # Non-commutative ops with the chain on input[1] swap operands
            # on the device (so "subtract" evaluates b - rgb, matching
            # ShaderNodeMath). Commutative ops don't care.
            swap = chain_idx == 1 and op in ("subtract", "divide", "power")
            idx = len(nodes)
            node = {
                "type": "math", "input": ci, "op": op,
                "b": b, "c": c, "clamp": use_clamp,
            }
            if swap:
                node["swap"] = True
            nodes.append(node)
            memo[key] = idx
            return idx

        if bl == "ShaderNodeHueSaturation":
            col_sock = src.inputs.get("Color")
            if col_sock is None:
                return None
            # Constants only for Hue/Sat/Val/Fac — matches Blender's common
            # use where these are UI-tweaked sliders rather than driven.
            for n in ("Hue", "Saturation", "Value", "Fac"):
                s = src.inputs.get(n)
                if s is not None and s.is_linked:
                    return None
            ci = emit(col_sock)
            if ci is None:
                return None
            hue = _socket_f(src.inputs["Hue"]) if "Hue" in src.inputs else 0.5
            sat = _socket_f(src.inputs["Saturation"]) if "Saturation" in src.inputs else 1.0
            val = _socket_f(src.inputs["Value"]) if "Value" in src.inputs else 1.0
            fac = _socket_f(src.inputs["Fac"]) if "Fac" in src.inputs else 1.0
            idx = len(nodes)
            nodes.append({
                "type": "hue_sat", "input": ci,
                "hue": hue, "saturation": sat, "value": val, "fac": fac,
            })
            memo[key] = idx
            return idx

        if bl == "ShaderNodeRGBCurve":
            col_sock = src.inputs.get("Color")
            fac_sock = src.inputs.get("Fac")
            if col_sock is None:
                return None
            # Textured Fac would need a per-pixel lerp; require a constant
            # so the blend folds into the LUT at export time.
            if fac_sock is not None and fac_sock.is_linked:
                return None
            fac_val = _socket_f(fac_sock) if fac_sock is not None else 1.0
            ci = emit(col_sock)
            if ci is None:
                return None
            # Bake the curves into a 256x3 LUT. Blender stacks per-channel
            # curves and the combined curve — evaluate combined(per_channel(x))
            # per sample so the LUT captures the full effect. Non-unity Fac
            # is folded in too: `out = lerp(x, curve(x), fac)` lets classroom's
            # RGB-Curves-with-Fac-0.9 wood tint stay in the graph instead of
            # falling out into a baked texture.
            cm = src.mapping
            cm.update()
            curves = cm.curves
            if len(curves) < 4:
                return None
            lut: list[float] = []
            for ch in range(3):
                for i in range(256):
                    x = i / 255.0
                    # CurveMap is evaluated via the parent CurveMapping.
                    y = cm.evaluate(curves[ch], x)
                    y = cm.evaluate(curves[3], y)
                    y = x * (1.0 - fac_val) + y * fac_val
                    lut.append(max(0.0, min(1.0, float(y))))
            idx = len(nodes)
            nodes.append({"type": "rgb_curve", "input": ci, "lut": lut})
            memo[key] = idx
            return idx

        if bl == "ShaderNodeBrightContrast":
            col_sock = src.inputs.get("Color")
            br_sock = src.inputs.get("Bright")
            co_sock = src.inputs.get("Contrast")
            if col_sock is None:
                return None
            if (br_sock is not None and br_sock.is_linked) or (
                co_sock is not None and co_sock.is_linked):
                return None
            ci = emit(col_sock)
            if ci is None:
                return None
            bright = _socket_f(br_sock) if br_sock is not None else 0.0
            contrast = _socket_f(co_sock) if co_sock is not None else 0.0
            idx = len(nodes)
            nodes.append({
                "type": "bright_contrast", "input": ci,
                "bright": bright, "contrast": contrast,
            })
            memo[key] = idx
            return idx

        if bl == "ShaderNodeAttribute":
            # Per-vertex colour attribute feeding the chain (classroom's
            # schoolDesk_wood OVERLAY-mixes a per-instance tint this way).
            # Only GEOMETRY-type attributes are real vertex data — Cycles'
            # Instancer / Object attributes don't map to anything we ship.
            attr_type = getattr(src, "attribute_type", "GEOMETRY")
            if attr_type != "GEOMETRY":
                return None
            attr_name = getattr(src, "attribute_name", "") or ""
            if not attr_name:
                return None
            if vc_attrs is not None and attr_name not in vc_attrs:
                vc_attrs.append(attr_name)
            idx = len(nodes)
            nodes.append({"type": "vertex_color"})
            memo[key] = idx
            return idx

        if bl == "ShaderNodeValToRGB":
            # ColorRamp: scalar in, RGB out. We fold it to a Const node when
            # the Fac chain reduces to a constant (noise / procedural leaves
            # fold to their mean, Math/Clamp fold through
            # `_resolve_constant_scalar`, Mix(RGBA) folds through
            # `_socket_constant_rgb`). Non-foldable inputs (live textures
            # driving the ramp) would need a per-pixel LUT eval node,
            # deferred until the GPU grows one.
            in_sock = src.inputs.get("Fac")
            if in_sock is None:
                return None
            scalar = _resolve_constant_scalar(in_sock)
            if scalar is None:
                rgb = _socket_constant_rgb(in_sock)
                if rgb is None:
                    return None
                scalar = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            scalar = max(0.0, min(1.0, float(scalar)))
            color = list(src.color_ramp.evaluate(scalar))[:3]
            idx = len(nodes)
            nodes.append({"type": "const", "rgb": color})
            memo[key] = idx
            return idx

        # Fallback: any node whose sub-chain folds to a constant RGB (plain
        # RGB / Blackbody / all-const Mix / TexImage mean, …) becomes a Const
        # leaf so the surrounding graph can still emit. Specific handlers
        # above (Attribute → vertex_color) already claim the nodes where a
        # mean-colour fold would be less faithful than a per-pixel reader.
        fallback_rgb = _socket_constant_rgb(emit_sock)
        if fallback_rgb is not None:
            idx = len(nodes)
            nodes.append({"type": "const", "rgb": fallback_rgb})
            memo[key] = idx
            return idx
        return None

    out_idx = emit(sock)
    if out_idx is None or not nodes:
        return None
    return {"nodes": nodes, "output": out_idx}


def _normal_perturbation(normal_sock):
    """Inspect a Normal input socket and return (normal_sock_or_None,
    n_strength, bump_sock_or_None, b_strength).

    The returned sockets are the upstream Color/Height inputs that ultimately
    feed the perturbation; callers run them through the chain machinery to
    bake any colour transforms into the resulting linear texture.
    """
    if normal_sock is None or not normal_sock.is_linked:
        return None, 1.0, None, 1.0
    src = normal_sock.links[0].from_node
    if src.bl_idname == "ShaderNodeNormalMap":
        strength = _socket_f(src.inputs["Strength"]) if "Strength" in src.inputs else 1.0
        return src.inputs["Color"], strength, None, 1.0
    if src.bl_idname == "ShaderNodeBump":
        strength = _socket_f(src.inputs["Strength"]) if "Strength" in src.inputs else 1.0
        distance = _socket_f(src.inputs["Distance"]) if "Distance" in src.inputs else 1.0
        return None, 1.0, src.inputs["Height"], strength * distance
    if src.bl_idname == "ShaderNodeTexImage" and src.image is not None:
        # Direct TexImage on Normal input: synthesise a fake socket-like by
        # wrapping the node's output back into normal_sock itself.
        return normal_sock, 1.0, None, 1.0
    _warn(
        f"normal-src:{src.bl_idname}",
        f"Normal input driven by {_node_tag(src)} — not handled (expected "
        f"NormalMap, Bump, or TexImage); normal perturbation ignored",
    )
    return None, 1.0, None, 1.0


def _normal_map_image(normal_sock) -> bpy.types.Image | None:
    n_sock, _, b_sock, _ = _normal_perturbation(normal_sock)
    src = n_sock if n_sock is not None else b_sock
    if src is None:
        return None
    return _socket_linked_image(src)


def _apply_normal_perturbation(params: dict, normal_sock, writer, textures) -> None:
    """Resolve the Normal input and assign normal_tex / bump_tex + strengths."""
    n_sock, n_strength, b_sock, b_strength = _normal_perturbation(normal_sock)
    if n_sock is not None:
        tex = _export_linked_scalar_texture(n_sock, writer, textures)
        if tex is not None:
            params["normal_tex"] = tex
            if n_strength != 1.0:
                params["normal_strength"] = n_strength
    if b_sock is not None:
        tex = _export_linked_scalar_texture(b_sock, writer, textures)
        if tex is not None:
            params["bump_tex"] = tex
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


def _resolve_shader(node, writer, textures, mat_name: str) -> dict:
    """Return Principled-equivalent params for any surface-shader node."""
    bl = node.bl_idname
    if bl == "ShaderNodeBsdfPrincipled":
        return _from_principled(node, writer, textures)
    if bl == "ShaderNodeBsdfDiffuse":
        return _from_diffuse(node, writer, textures)
    if bl == "ShaderNodeBsdfGlossy" or bl == "ShaderNodeBsdfAnisotropic":
        return _from_glossy(node, writer, textures)
    if bl == "ShaderNodeBsdfTranslucent":
        return _from_diffuse(node, writer, textures)
    if bl == "ShaderNodeBsdfTransparent":
        return _from_transparent(node, writer, textures)
    if bl == "ShaderNodeBsdfSheen" or bl == "ShaderNodeBsdfVelvet":
        # Sheen has no native slot in the simplified Principled; treat the
        # Color input as a rough diffuse so the surface isn't blacked out.
        return _from_diffuse(node, writer, textures)
    if bl == "ShaderNodeBsdfHair":
        return _from_hair(node, writer, textures)
    if bl == "ShaderNodeMixShader":
        return _from_mix(node, writer, textures, mat_name)
    if bl == "ShaderNodeBsdfGlass":
        return _from_glass(node, writer, textures)
    if bl == "ShaderNodeBsdfRefraction":
        return _from_refraction(node, writer, textures)
    if bl == "ShaderNodeEmission":
        return _from_emission(node, writer, textures)
    if bl == "ShaderNodeAddShader":
        return _from_add(node, writer, textures, mat_name)
    if bl == "ShaderNodeGroup":
        return _from_group(node, writer, textures, mat_name)
    if bl == "ShaderNodeVolumeAbsorption" or bl == "ShaderNodeVolumeScatter" or bl == "ShaderNodeVolumePrincipled":
        _warn(f"vol-shader:{bl}", f"ignoring volume shader {bl}")
        return _default_params()
    _warn(f"unsup-shader:{bl}", f"unsupported shader node: {bl}")
    return _default_params()


def _from_principled(node, writer, textures) -> dict:
    p = _default_params()
    bc = node.inputs["Base Color"]
    vc_attrs: list[str] = []
    graph = (
        _try_emit_color_graph(bc, writer, textures, vc_attrs=vc_attrs)
        if bc.is_linked else None
    )
    if graph is not None:
        p["color_graph"] = graph
        p["base_color"] = [1.0, 1.0, 1.0]
        if vc_attrs:
            # The graph samples v.vc directly — don't also enable the
            # base_color × v.vc multiplier (would double-apply the
            # attribute).
            p["_vertex_color_attr"] = vc_attrs[0]
    else:
        img, chain = _socket_linked_image_with_chain(bc)
        if img is not None:
            # Linked socket: the link drives the colour, the default RGB is unused
            # in Cycles. The renderer multiplies base_color × texture, so neutralise
            # the factor to avoid tinting the texture.
            p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
            p["base_color"] = [1.0, 1.0, 1.0]
        elif bc.is_linked:
            attr = _find_base_color_attribute(bc)
            if attr is not None:
                p["use_vertex_color"] = True
                p["_vertex_color_attr"] = attr
                p["base_color"] = [1.0, 1.0, 1.0]
            else:
                p["base_color"] = _socket_rgb(bc)
        else:
            p["base_color"] = _socket_rgb(bc)

    tex = _export_linked_scalar_texture(node.inputs["Metallic"], writer, textures)
    if tex is not None:
        p["metallic_tex"] = tex
        p["metallic"] = 1.0
    else:
        p["metallic"] = _socket_f(node.inputs["Metallic"])

    tex = _export_linked_scalar_texture(node.inputs["Roughness"], writer, textures)
    if tex is not None:
        p["roughness_tex"] = tex
        p["roughness"] = 1.0
    else:
        p["roughness"] = _socket_f(node.inputs["Roughness"])

    if "IOR" in node.inputs:
        p["ior"] = _warn_linked_scalar(node, "IOR")
    trans_name = (
        "Transmission Weight" if "Transmission Weight" in node.inputs
        else "Transmission" if "Transmission" in node.inputs
        else None
    )
    if trans_name:
        tex = _export_linked_scalar_texture(
            node.inputs[trans_name], writer, textures,
        )
        if tex is not None:
            p["transmission_tex"] = tex
            p["transmission"] = 1.0
        else:
            p["transmission"] = _warn_linked_scalar(node, trans_name)

    # Principled emission: same texture-driven idiom as the standalone
    # Emission node (e.g. Cycles 4.x Emissive Color = Image Texture). Prefer
    # the per-pixel `emission_tex` path so a baked display / billboard image
    # keeps its spatial detail; otherwise fold to a constant.
    emit_sock = None
    strength = 1.0
    if "Emission" in node.inputs:
        emit_sock = node.inputs["Emission"]
    elif "Emission Color" in node.inputs:
        emit_sock = node.inputs["Emission Color"]
        if "Emission Strength" in node.inputs:
            strength = _warn_linked_scalar(node, "Emission Strength")
    if emit_sock is not None:
        emit_img = None
        emit_chain = ()
        if emit_sock.is_linked:
            emit_img, emit_chain = _socket_linked_image_with_chain(emit_sock)
        if emit_img is not None:
            p["emission_tex"] = export_image_texture(
                emit_img, writer, textures, "srgb", chain=emit_chain
            )
            p["emission"] = [strength, strength, strength]
        else:
            color = _emission_constant_color(node, emit_sock)
            p["emission"] = [c * strength for c in color]

    _apply_normal_perturbation(p, node.inputs.get("Normal"), writer, textures)

    # Anisotropy (Blender Principled: "Anisotropic" + "Anisotropic Rotation").
    for name in ("Anisotropic", "Anisotropy"):
        if name in node.inputs:
            a = _warn_linked_scalar(node, name)
            if a != 0.0:
                p["anisotropy"] = a
            break
    for name in ("Anisotropic Rotation", "Anisotropy Rotation"):
        if name in node.inputs:
            r = _warn_linked_scalar(node, name)
            if r != 0.0:
                p["tangent_rotation"] = r * 2.0 * math.pi
            break

    # Coat (Blender 4.x: "Coat Weight" / "Coat Roughness" / "Coat IOR";
    # legacy 3.x: "Clearcoat" / "Clearcoat Roughness").
    for name in ("Coat Weight", "Clearcoat"):
        if name in node.inputs:
            w = _warn_linked_scalar(node, name)
            if w > 0.0:
                p["coat_weight"] = w
            break
    for name in ("Coat Roughness", "Clearcoat Roughness"):
        if name in node.inputs:
            r = _warn_linked_scalar(node, name)
            if r != 0.03:
                p["coat_roughness"] = r
            break
    if "Coat IOR" in node.inputs:
        ior = _warn_linked_scalar(node, "Coat IOR")
        if ior != 1.5:
            p["coat_ior"] = ior

    # Sheen (Blender 4.x: "Sheen Weight" / "Sheen Roughness" / "Sheen Tint";
    # legacy: "Sheen" / "Sheen Tint" scalar).
    for name in ("Sheen Weight", "Sheen"):
        if name in node.inputs:
            w = _warn_linked_scalar(node, name)
            if w > 0.0:
                p["sheen_weight"] = w
            break
    if "Sheen Roughness" in node.inputs:
        r = _warn_linked_scalar(node, "Sheen Roughness")
        if r != 0.5:
            p["sheen_roughness"] = r
    if "Sheen Tint" in node.inputs:
        sheen_tint_sock = node.inputs["Sheen Tint"]
        if sheen_tint_sock.is_linked:
            _warn(
                f"sheen-tint-linked:{node.as_pointer()}",
                f"{_node_tag(node)}: Sheen Tint is linked but not baked — "
                f"using constant default",
            )
        v = sheen_tint_sock.default_value
        if hasattr(v, "__len__") and len(v) >= 3:
            t = [float(v[0]), float(v[1]), float(v[2])]
            if t != [1.0, 1.0, 1.0]:
                p["sheen_tint"] = t

    # Subsurface (Blender 4.x: "Subsurface Weight" / "Subsurface Radius";
    # 3.x: "Subsurface" / "Subsurface Radius"). Anisotropy is not consumed
    # by the simplified wrap-Lambert SSS in the renderer, so we don't emit it.
    for name in ("Subsurface Weight", "Subsurface"):
        if name in node.inputs:
            w = _warn_linked_scalar(node, name)
            if w > 0.0:
                p["sss_weight"] = w
            break
    if "Subsurface Radius" in node.inputs:
        sss_r_sock = node.inputs["Subsurface Radius"]
        if sss_r_sock.is_linked:
            _warn(
                f"sss-radius-linked:{node.as_pointer()}",
                f"{_node_tag(node)}: Subsurface Radius is linked but not baked "
                f"— using constant default",
            )
        v = sss_r_sock.default_value
        if hasattr(v, "__len__") and len(v) >= 3:
            r = [float(v[0]), float(v[1]), float(v[2])]
            if r != [1.0, 0.2, 0.1]:
                p["sss_radius"] = r

    alpha_sock = node.inputs.get("Alpha")
    if alpha_sock is not None:
        alpha_val = _socket_f(alpha_sock)
        if alpha_sock.is_linked:
            p["alpha_threshold"] = 0.5
            if "base_color_tex" not in p:
                img = _socket_linked_image(alpha_sock)
                if img is not None:
                    p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb")
                else:
                    _warn(
                        f"alpha-noimg:{node.as_pointer()}",
                        f"{_node_tag(node)}: Alpha is linked but no TexImage "
                        f"found upstream — alpha threshold is effectively "
                        f"disabled (no per-pixel alpha to compare against)",
                    )
        elif alpha_val < 1.0:
            p["alpha_threshold"] = alpha_val
    return p


def _from_diffuse(node, writer, textures) -> dict:
    p = _default_params()
    p["roughness"] = 1.0
    p["metallic"] = 0.0
    color_sock = node.inputs["Color"]
    vc_attrs: list[str] = []
    graph = (
        _try_emit_color_graph(color_sock, writer, textures, vc_attrs=vc_attrs)
        if color_sock.is_linked else None
    )
    if graph is not None:
        p["color_graph"] = graph
        p["base_color"] = [1.0, 1.0, 1.0]
        if vc_attrs:
            p["_vertex_color_attr"] = vc_attrs[0]
    else:
        img, chain = _socket_linked_image_with_chain(color_sock)
        if img is not None:
            p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
            p["base_color"] = [1.0, 1.0, 1.0]
        elif color_sock.is_linked:
            attr = _find_base_color_attribute(color_sock)
            if attr is not None:
                p["use_vertex_color"] = True
                p["_vertex_color_attr"] = attr
                p["base_color"] = [1.0, 1.0, 1.0]
            else:
                p["base_color"] = _socket_rgb(color_sock)
        else:
            p["base_color"] = _socket_rgb(color_sock)
    _apply_normal_perturbation(p, node.inputs.get("Normal"), writer, textures)
    return p


def _from_hair(node, writer, textures) -> dict:
    """Map Cycles' classic Hair BSDF (Kajiya-Kay) to Principled params.

    Sets `hair_weight = 1.0` so the kernel takes the Kajiya-Kay eval/sample
    branch instead of the standard diffuse+specular path. Color feeds
    `base_color` (with texture support). Offset / RoughnessU / RoughnessV are
    read as scalar constants; linked inputs fall back through
    `_warn_linked_scalar` (exact fold or warn). Tangent is intentionally
    ignored — the kernel uses the per-hit tangent frame built from Ns, which
    doesn't follow the strand direction but gives a qualitatively hair-like
    look on triangle-mesh "hair caps". Proper UV-based tangent is follow-up.
    """
    p = _default_params()
    p["hair_weight"] = 1.0
    color_sock = node.inputs["Color"]
    img, chain = _socket_linked_image_with_chain(color_sock)
    if img is not None:
        p["base_color_tex"] = export_image_texture(
            img, writer, textures, "srgb", chain=chain
        )
        p["base_color"] = [1.0, 1.0, 1.0]
    elif color_sock.is_linked:
        rgb = _socket_constant_rgb(color_sock)
        if rgb is not None:
            p["base_color"] = rgb
        else:
            p["base_color"] = _socket_rgb(color_sock)
    else:
        p["base_color"] = _socket_rgb(color_sock)
    if "Offset" in node.inputs:
        p["hair_offset"] = _warn_linked_scalar(node, "Offset")
    if "RoughnessU" in node.inputs:
        p["hair_roughness_u"] = _warn_linked_scalar(node, "RoughnessU")
    if "RoughnessV" in node.inputs:
        p["hair_roughness_v"] = _warn_linked_scalar(node, "RoughnessV")
    return p


def _from_glossy(node, writer, textures) -> dict:
    p = _default_params()
    p["metallic"] = 1.0
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        tex = _export_linked_scalar_texture(node.inputs["Roughness"], writer, textures)
        if tex is not None:
            p["roughness_tex"] = tex
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
    if "Anisotropy" in node.inputs:
        a = _warn_linked_scalar(node, "Anisotropy")
        if a != 0.0:
            p["anisotropy"] = a
    if "Rotation" in node.inputs:
        r = _warn_linked_scalar(node, "Rotation")
        if r != 0.0:
            p["tangent_rotation"] = r * 2.0 * math.pi
    _apply_normal_perturbation(p, node.inputs.get("Normal"), writer, textures)
    return p


def _from_glass(node, writer, textures) -> dict:
    p = _default_params()
    p["transmission"] = 1.0
    if "IOR" in node.inputs:
        p["ior"] = _warn_linked_scalar(node, "IOR")
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        tex = _export_linked_scalar_texture(node.inputs["Roughness"], writer, textures)
        if tex is not None:
            p["roughness_tex"] = tex
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
    return p


def _from_refraction(node, writer, textures) -> dict:
    p = _default_params()
    p["transmission"] = 1.0
    p["metallic"] = 0.0
    if "IOR" in node.inputs:
        p["ior"] = _warn_linked_scalar(node, "IOR")
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    if "Roughness" in node.inputs:
        tex = _export_linked_scalar_texture(node.inputs["Roughness"], writer, textures)
        if tex is not None:
            p["roughness_tex"] = tex
            p["roughness"] = 1.0
        else:
            p["roughness"] = _socket_f(node.inputs["Roughness"])
    return p


def _from_add(node, writer, textures, mat_name: str) -> dict:
    """Add Shader: sum of emission, max of transmission, p1's surface params.

    The renderer has one BSDF slot, so we can't actually add two lobes — the
    best faithful fallback is to keep p1's surface and sum the emissive
    contributions. Transmission takes the max so Add(Glass, Emission) still
    refracts.
    """
    in1, in2 = node.inputs[0], node.inputs[1]
    n1 = in1.links[0].from_node if in1.is_linked else None
    n2 = in2.links[0].from_node if in2.is_linked else None
    p1 = _resolve_shader(n1, writer, textures, mat_name) if n1 else _default_params()
    p2 = _resolve_shader(n2, writer, textures, mat_name) if n2 else _default_params()
    out = dict(p1)
    out["emission"] = [a + b for a, b in zip(p1["emission"], p2["emission"])]
    out["transmission"] = max(p1["transmission"], p2["transmission"])
    e1 = p1.get("emission_tex")
    e2 = p2.get("emission_tex")
    if e1 is not None and e2 is not None:
        _warn(
            f"add-emit-tex:{node.as_pointer()}",
            f"{_node_tag(node)}: both Add Shader inputs have an emission "
            "texture — only the first is kept (one emission_tex slot)",
        )
        out["emission_tex"] = e1
    elif e2 is not None:
        out["emission_tex"] = e2
    return out


def _from_transparent(node, writer, textures) -> dict:
    p = _default_params()
    p["transmission"] = 1.0
    p["ior"] = 1.0  # no refraction
    p["roughness"] = 0.0
    img, chain = _socket_linked_image_with_chain(node.inputs["Color"])
    if img is not None:
        p["base_color_tex"] = export_image_texture(img, writer, textures, "srgb", chain=chain)
        p["base_color"] = [1.0, 1.0, 1.0]
        p["alpha_threshold"] = 0.5
    else:
        p["base_color"] = _socket_rgb(node.inputs["Color"])
    return p


def _from_emission(node, writer, textures) -> dict:
    p = _default_params()
    color_sock = node.inputs["Color"]
    if "Strength" in node.inputs:
        strength = _warn_linked_scalar(node, "Strength")
    else:
        strength = 1.0
    # Per-pixel path first: a Color chain that resolves to a single image
    # texture (typical for billboards, displays, emissive labels) drives
    # `emission_tex` so the spatial detail survives. Otherwise we collapse
    # the chain to a constant — `_emission_constant_color` warns when the
    # collapse is lossy.
    if color_sock.is_linked:
        img, chain = _socket_linked_image_with_chain(color_sock)
        if img is not None:
            p["emission_tex"] = export_image_texture(
                img, writer, textures, "srgb", chain=chain
            )
            p["emission"] = [strength, strength, strength]
            p["base_color"] = [0.0, 0.0, 0.0]
            return p
    color = _emission_constant_color(node, color_sock)
    p["emission"] = [c * strength for c in color]
    p["base_color"] = [0.0, 0.0, 0.0]
    return p


def _emission_constant_color(node, color_sock) -> list[float]:
    """Resolve an Emission-style Color socket to a constant RGB.

    Walks the chain through bakeable transforms + procedural/leaf folding. If
    the chain isn't foldable (e.g. a live TexImage drives the emission), we
    warn and fall back to the socket's default. The renderer stores emission
    as a constant RGB, so there's no "bake to texture" option here.
    """
    if not color_sock.is_linked:
        return _socket_rgb(color_sock)
    rgb = _socket_constant_rgb(color_sock)
    if rgb is not None:
        return rgb
    _warn(
        f"emit-color-linked:{node.as_pointer()}",
        f"{_node_tag(node)}: Color chain isn't foldable to a constant "
        f"(textured emission) — using socket default",
    )
    return _socket_rgb(color_sock)


def _from_mix(node, writer, textures, mat_name: str) -> dict:
    """Shader Mix: lerp Principled params by the constant Factor.

    A linked Factor is not bakeable into scalar params, so we warn and lerp at
    fac=0.5 — the least-biased collapse (and no side is silently dropped).
    """
    in1, in2 = node.inputs[1], node.inputs[2]
    n1 = in1.links[0].from_node if in1.is_linked else None
    n2 = in2.links[0].from_node if in2.is_linked else None
    p1 = _resolve_shader(n1, writer, textures, mat_name) if n1 else _default_params()
    p2 = _resolve_shader(n2, writer, textures, mat_name) if n2 else _default_params()

    fac_sock = node.inputs[0]
    view_ior = _view_dependent_ior(fac_sock) if fac_sock.is_linked else None
    if view_ior is not None:
        # Fresnel/LayerWeight-driven MixShader is Blender's clearcoat
        # idiom: shader1 is the base lobe, shader2 dominates at grazing.
        # Decompose shader2 onto the Principled coat lobe above shader1:
        #   coat_ior       ← factor's Fresnel IOR (matches `fac(c)` in
        #                    Cycles' `(1-fac)·F₁ + fac·F₂` mix shape)
        #   coat_weight    ← luminance(shader2.base_color) — Cycles
        #                    Glossy has F=Schlick(F₀=Color), so shader2's
        #                    peak reflectance at normal scales the lobe
        #   coat_roughness ← shader2.roughness
        # Loses shader2's hue (coat is a white dielectric lobe) but
        # captures the view-angle falloff and highlight intensity —
        # what actually sells the metallic paint. Requires the renderer
        # to importance-sample the coat lobe (see `sample_bsdf`); a pure
        # eval-only coat over a broad base lobe under-samples the sharp
        # clearcoat highlight.
        if (p1.get("coat_weight", 0.0) == 0.0
                and p2.get("emission", [0, 0, 0]) == [0, 0, 0]
                and p2.get("transmission", 0.0) == 0.0):
            bc2 = p2.get("base_color", [1.0, 1.0, 1.0])
            lum2 = 0.2126 * bc2[0] + 0.7152 * bc2[1] + 0.0722 * bc2[2]
            out = dict(p1)
            out["coat_weight"] = max(0.0, min(1.0, lum2))
            out["coat_roughness"] = p2.get("roughness", 0.03)
            out["coat_ior"] = view_ior
            return out
        return dict(p1)
    if fac_sock.is_linked:
        # Try to fold the factor to a scalar (constant-driven Math /
        # procedural → mean).
        fac = _resolve_constant_scalar(fac_sock)
        if fac is None:
            _warn(
                f"mixshader-fac-linked:{node.as_pointer()}",
                f"{_node_tag(node)}: MixShader Factor is linked and not "
                f"foldable — keeping shader1's params (per-pixel mix not "
                f"bakeable)",
            )
            return dict(p1)
    else:
        fac = _socket_f(fac_sock)
    primary = p1 if fac < 0.5 else p2
    out = dict(primary)
    # Scalar params lerp; textures / graphs / normal maps come from `primary`
    # (they can't be weighted on the host).
    if "base_color_tex" not in out and "color_graph" not in out:
        bc1, bc2 = p1["base_color"], p2["base_color"]
        out["base_color"] = [
            bc1[i] * (1.0 - fac) + bc2[i] * fac for i in range(3)
        ]
    if "roughness_tex" not in out:
        out["roughness"] = p1["roughness"] * (1.0 - fac) + p2["roughness"] * fac
    if "metallic_tex" not in out:
        out["metallic"] = p1["metallic"] * (1.0 - fac) + p2["metallic"] * fac
    out["emission"] = [
        a * (1.0 - fac) + b * fac for a, b in zip(p1["emission"], p2["emission"])
    ]
    out["transmission"] = (
        p1["transmission"] * (1.0 - fac) + p2["transmission"] * fac
    )
    return out


def _from_group(node, writer, textures, mat_name: str) -> dict:
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
            # Push so chain traversal inside the inner BSDF can follow
            # NodeGroupInput → external socket on this group node.
            _GROUP_STACK.append(node)
            try:
                return _resolve_shader(inner, writer, textures, mat_name)
            finally:
                _GROUP_STACK.pop()
    _warn(
        f"group-nolink:{tree.name}",
        f"group {tree.name!r} output has no linked input",
    )
    return _default_params()


def export_material(
    mat: bpy.types.Material, writer, textures: list
) -> dict:
    """Dispatch to the surface-node handler; fall back to diffuse_color."""
    global _CURRENT_MATERIAL
    # Reset dedup state so each material emits its own set of warnings.
    _LOGGED_WARNINGS.clear()
    _CURRENT_MATERIAL = mat.name
    # Exclude nested texture/bake children so material_self_s is additive with
    # texture_self_s + pixel_read_s + bake_chain_s in the summary log.
    t_enter = time.perf_counter()
    tex_before = _STATS["texture_self_s"]
    pixel_before = _STATS["pixel_read_s"]
    bake_before = _STATS["bake_chain_s"]
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
        params = _resolve_shader(surface, writer, textures, mat.name)

        # Per-material UV transform: read it off the TexImage nodes rather
        # than picking up any Mapping node (which may drive a Noise procedural,
        # not the actual image sample).
        affine = _first_teximage_uv_transform(mat.node_tree)
        if affine is not None:
            params["uv_transform"] = affine
        return params
    finally:
        _CURRENT_MATERIAL = ""
        dt = time.perf_counter() - t_enter
        children = (
            (_STATS["texture_self_s"] - tex_before)
            + (_STATS["pixel_read_s"] - pixel_before)
            + (_STATS["bake_chain_s"] - bake_before)
        )
        self_s = max(0.0, dt - children)
        _STATS["material_self_s"] += self_s
        if self_s > 0.10:
            slow = _STATS["slow_materials"]
            slow.append((mat.name, self_s))
            slow.sort(key=lambda e: e[1], reverse=True)
            del slow[5:]
