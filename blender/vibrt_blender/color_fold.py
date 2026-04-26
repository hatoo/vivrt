"""Export-time constant folding for the JSON `color_graph`.

Walks the graph in topological order. A node whose every input resolves to
a `{"type": "const"}` is evaluated on the host and rewritten in-place as a
`const`; graph length and `output` index are preserved so downstream
indices never need remapping. The GPU interpreter (`eval_color_graph` in
`devicecode.cu`) sees a shorter chain of work and the texture/colour-graph
upload path can take the const fast path.

Op evaluators mirror the GPU code in `devicecode.cu` (`mix_blend_rgb`,
`math_apply_rgb`, `rgb_to_hsv_bl`, `hsv_to_rgb_bl`, etc.) branch-for-branch
and are unit-tested in `test_color_fold.py`.

This module has no bpy imports, so the tests can run under plain CPython.
"""

from __future__ import annotations

import math
from typing import Any


# --- Blend / math op tables (must match `principled.rs::parse_blend` etc.) ---

_BLEND_OPS = {
    "mix": 0,
    "multiply": 1,
    "add": 2,
    "subtract": 3,
    "screen": 4,
    "divide": 5,
    "difference": 6,
    "darken": 7,
    "lighten": 8,
    "overlay": 9,
    "soft_light": 10,
    "linear_light": 11,
}

_MATH_OPS = {
    "add": 0,
    "subtract": 1,
    "multiply": 2,
    "divide": 3,
    "power": 4,
    "multiply_add": 5,
    "minimum": 6,
    "maximum": 7,
}


def fold_constants(graph: dict) -> dict:
    """Return a copy of `graph` with constant-foldable nodes rewritten as
    `{"type": "const", "rgb": [...]}`. Non-foldable inputs (image_tex,
    vertex_color, or any op with a non-const ancestor) pass through
    unchanged. Output index is preserved."""
    in_nodes = graph.get("nodes", [])
    out_nodes: list[dict] = []
    for node in in_nodes:
        folded = _try_fold(node, out_nodes)
        out_nodes.append(folded if folded is not None else dict(node))
    result = {"nodes": out_nodes}
    if "output" in graph and graph["output"] is not None:
        result["output"] = graph["output"]
    return result


# --- Internal helpers ---


def _as_const(prev: list[dict], idx: int) -> list[float] | None:
    if idx < 0 or idx >= len(prev):
        return None
    n = prev[idx]
    if n.get("type") == "const":
        return list(n["rgb"])
    return None


def _const_node(rgb: list[float]) -> dict:
    return {"type": "const", "rgb": [float(rgb[0]), float(rgb[1]), float(rgb[2])]}


def _luminance(c: list[float]) -> float:
    # Rec.709 — must match devicecode.cu `luminance()`.
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _clamp01(c: list[float]) -> list[float]:
    return [min(max(c[0], 0.0), 1.0), min(max(c[1], 0.0), 1.0), min(max(c[2], 0.0), 1.0)]


def _try_fold(node: dict, prev: list[dict]) -> dict | None:
    t = node.get("type")
    if t in ("const", "image_tex", "vertex_color", None):
        return None

    if t == "mix":
        a = _as_const(prev, int(node["a"]))
        if a is None:
            return None
        b = _as_const(prev, int(node["b"]))
        if b is None:
            return None
        fac_field = node.get("fac", 0.5)
        if isinstance(fac_field, dict):
            fnode = _as_const(prev, int(fac_field["node"]))
            if fnode is None:
                return None
            fac = _luminance(fnode)
        else:
            fac = float(fac_field)
        blend = _BLEND_OPS.get(node.get("blend", "mix"))
        if blend is None:
            return None
        return _const_node(_eval_mix(a, b, fac, blend, bool(node.get("clamp", False))))

    if t == "invert":
        c = _as_const(prev, int(node["input"]))
        if c is None:
            return None
        fac = float(node.get("fac", 1.0))
        return _const_node(_eval_invert(c, fac))

    if t == "math":
        inp = _as_const(prev, int(node["input"]))
        if inp is None:
            return None
        op = _MATH_OPS.get(node.get("op", "multiply"))
        if op is None:
            return None
        b = float(node.get("b", 0.0))
        c = float(node.get("c", 0.0))
        swap = bool(node.get("swap", False))
        out = _eval_math(inp, op, b, c, swap)
        if bool(node.get("clamp", False)):
            out = _clamp01(out)
        return _const_node(out)

    if t == "hue_sat":
        src = _as_const(prev, int(node["input"]))
        if src is None:
            return None
        return _const_node(_eval_hue_sat(
            src,
            float(node.get("hue", 0.5)),
            float(node.get("saturation", 1.0)),
            float(node.get("value", 1.0)),
            float(node.get("fac", 1.0)),
        ))

    if t == "rgb_curve":
        src = _as_const(prev, int(node["input"]))
        if src is None:
            return None
        lut = node.get("lut", [])
        if len(lut) != 768:
            # Validation will fire later with a clearer error from the loader.
            return None
        return _const_node(_eval_rgb_curve(src, lut))

    if t == "bright_contrast":
        c = _as_const(prev, int(node["input"]))
        if c is None:
            return None
        return _const_node(_eval_bright_contrast(
            c,
            float(node.get("bright", 0.0)),
            float(node.get("contrast", 0.0)),
        ))

    return None


# --- Op evaluators (mirror devicecode.cu branch-for-branch) ---


def _eval_mix(a: list[float], b: list[float], fac: float, blend: int, clamp_out: bool) -> list[float]:
    facm = 1.0 - fac
    if blend == 0:  # MIX
        out = [a[0] * facm + b[0] * fac, a[1] * facm + b[1] * fac, a[2] * facm + b[2] * fac]
    elif blend == 1:  # MULTIPLY
        out = [
            a[0] * (facm + b[0] * fac),
            a[1] * (facm + b[1] * fac),
            a[2] * (facm + b[2] * fac),
        ]
    elif blend == 2:  # ADD
        out = [a[0] + b[0] * fac, a[1] + b[1] * fac, a[2] + b[2] * fac]
    elif blend == 3:  # SUBTRACT
        out = [a[0] - b[0] * fac, a[1] - b[1] * fac, a[2] - b[2] * fac]
    elif blend == 4:  # SCREEN
        out = [
            1.0 - (facm + (1.0 - b[0]) * fac) * (1.0 - a[0]),
            1.0 - (facm + (1.0 - b[1]) * fac) * (1.0 - a[1]),
            1.0 - (facm + (1.0 - b[2]) * fac) * (1.0 - a[2]),
        ]
    elif blend == 5:  # DIVIDE — per-channel, guard div-by-zero
        out = [
            a[i] if b[i] == 0.0 else a[i] * facm + a[i] / b[i] * fac
            for i in range(3)
        ]
    elif blend == 6:  # DIFFERENCE
        out = [a[i] * facm + abs(a[i] - b[i]) * fac for i in range(3)]
    elif blend == 7:  # DARKEN
        out = [a[i] * facm + min(a[i], b[i]) * fac for i in range(3)]
    elif blend == 8:  # LIGHTEN (Blender's asymmetric form)
        out = [max(a[i], b[i] * fac) for i in range(3)]
    elif blend == 9:  # OVERLAY per-channel
        def _ov(ax: float, bx: float) -> float:
            if ax < 0.5:
                return ax * (facm + 2.0 * fac * bx)
            return 1.0 - (facm + 2.0 * fac * (1.0 - bx)) * (1.0 - ax)
        out = [_ov(a[0], b[0]), _ov(a[1], b[1]), _ov(a[2], b[2])]
    elif blend == 10:  # SOFT_LIGHT
        scr = [1.0 - (1.0 - b[i]) * (1.0 - a[i]) for i in range(3)]
        inner = [(1.0 - a[i]) * b[i] * a[i] + a[i] * scr[i] for i in range(3)]
        out = [a[i] * facm + inner[i] * fac for i in range(3)]
    elif blend == 11:  # LINEAR_LIGHT
        out = [a[i] + (b[i] * 2.0 - 1.0) * fac for i in range(3)]
    else:  # fall back to MIX
        out = [a[0] * facm + b[0] * fac, a[1] * facm + b[1] * fac, a[2] * facm + b[2] * fac]
    if clamp_out:
        out = _clamp01(out)
    return out


def _eval_math(inp: list[float], op: int, b: float, c: float, swap: bool) -> list[float]:
    if op == 0:  # ADD
        return [inp[i] + b for i in range(3)]
    if op == 1:  # SUBTRACT
        if swap:
            return [b - inp[i] for i in range(3)]
        return [inp[i] - b for i in range(3)]
    if op == 2:  # MULTIPLY
        return [inp[i] * b for i in range(3)]
    if op == 3:  # DIVIDE
        if swap:
            return [0.0 if inp[i] == 0.0 else b / inp[i] for i in range(3)]
        if b == 0.0:
            return [0.0, 0.0, 0.0]
        return [inp[i] / b for i in range(3)]
    if op == 4:  # POWER
        if swap:
            base = max(b, 0.0)
            return [base ** inp[i] for i in range(3)]
        return [max(inp[i], 0.0) ** b for i in range(3)]
    if op == 5:  # MULTIPLY_ADD
        return [inp[i] * b + c for i in range(3)]
    if op == 6:  # MINIMUM
        return [min(inp[i], b) for i in range(3)]
    if op == 7:  # MAXIMUM
        return [max(inp[i], b) for i in range(3)]
    return [inp[i] + b for i in range(3)]  # fallback


def _eval_invert(c: list[float], fac: float) -> list[float]:
    facm = 1.0 - fac
    return [c[i] * facm + (1.0 - c[i]) * fac for i in range(3)]


def _rgb_to_hsv_bl(c: list[float]) -> list[float]:
    cmax = max(c[0], c[1], c[2])
    cmin = min(c[0], c[1], c[2])
    v = cmax
    d = cmax - cmin
    s = d / cmax if cmax > 0.0 else 0.0
    h = 0.0
    if d > 0.0:
        if cmax == c[0]:
            h = (c[1] - c[2]) / d + (6.0 if c[1] < c[2] else 0.0)
        elif cmax == c[1]:
            h = (c[2] - c[0]) / d + 2.0
        else:
            h = (c[0] - c[1]) / d + 4.0
        h *= 1.0 / 6.0
    return [h, s, v]


def _hsv_to_rgb_bl(c: list[float]) -> list[float]:
    h = c[0] - math.floor(c[0])  # wrap to [0, 1)
    s = min(max(c[1], 0.0), 1.0)
    v = c[2]
    i = math.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    ii = int(i) % 6
    if ii < 0:
        ii += 6
    if ii == 0:
        return [v, t, p]
    if ii == 1:
        return [q, v, p]
    if ii == 2:
        return [p, v, t]
    if ii == 3:
        return [p, q, v]
    if ii == 4:
        return [t, p, v]
    return [v, p, q]


def _eval_hue_sat(src: list[float], hue: float, sat: float, val: float, fac: float) -> list[float]:
    hsv = _rgb_to_hsv_bl(src)
    hsv[0] += hue - 0.5
    hsv[1] = min(max(hsv[1] * sat, 0.0), 1.0)
    hsv[2] *= val
    shifted = _hsv_to_rgb_bl(hsv)
    facm = 1.0 - fac
    return [src[i] * facm + shifted[i] * fac for i in range(3)]


def _eval_rgb_curve(src: list[float], lut: list[float]) -> list[float]:
    # LUT layout: channel-major, R[0..256], G[0..256], B[0..256].
    def fetch(channel: int, x: float) -> float:
        xc = min(max(x, 0.0), 1.0)
        fx = xc * 255.0
        i0 = int(math.floor(fx))
        i1 = i0 + 1 if i0 < 255 else 255
        t = fx - i0
        base = channel * 256
        return lut[base + i0] * (1.0 - t) + lut[base + i1] * t

    return [fetch(0, src[0]), fetch(1, src[1]), fetch(2, src[2])]


def _eval_bright_contrast(c: list[float], bright: float, contrast: float) -> list[float]:
    a = 1.0 + contrast
    b = bright - 0.5 * contrast
    return [max(a * c[i] + b, 0.0) for i in range(3)]
