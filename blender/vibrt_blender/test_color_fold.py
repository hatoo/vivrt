"""Unit tests for `color_fold`. Runs under plain CPython — no Blender
needed, since `color_fold` has no bpy imports.

Run with:
    py blender/vibrt_blender/test_color_fold.py

The sibling `__init__.py` imports `bpy`, so we load `color_fold.py`
directly via importlib instead of going through the package.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "vibrt_color_fold", _HERE / "color_fold.py"
)
_color_fold = importlib.util.module_from_spec(_spec)
sys.modules["vibrt_color_fold"] = _color_fold
_spec.loader.exec_module(_color_fold)
fold_constants = _color_fold.fold_constants


def _const(rgb):
    return {"type": "const", "rgb": list(rgb)}


def _out_rgb(graph):
    nodes = graph["nodes"]
    idx = graph.get("output")
    if idx is None:
        idx = len(nodes) - 1
    n = nodes[idx]
    assert n["type"] == "const", f"output node is not const: {n!r}"
    return n["rgb"]


def _approx(case, a, b, eps=1e-6):
    case.assertEqual(len(a), len(b))
    for i in range(len(a)):
        case.assertAlmostEqual(a[i], b[i], delta=eps,
                               msg=f"component {i}: {a[i]} vs {b[i]}")


class FoldTests(unittest.TestCase):
    # --- Invert ---
    def test_invert_full(self):
        g = {"nodes": [
            _const([0.25, 0.5, 0.75]),
            {"type": "invert", "input": 0, "fac": 1.0},
        ]}
        f = fold_constants(g)
        self.assertEqual(len(f["nodes"]), 2)
        _approx(self, _out_rgb(f), [0.75, 0.5, 0.25])

    def test_invert_partial(self):
        # fac=0.5 → out = 0.5*rgb + 0.5*(1-rgb) = 0.5
        g = {"nodes": [
            _const([0.1, 0.9, 0.3]),
            {"type": "invert", "input": 0, "fac": 0.5},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.5, 0.5, 0.5])

    # --- Math ---
    def test_math_multiply(self):
        g = {"nodes": [
            _const([0.1, 0.2, 0.3]),
            {"type": "math", "input": 0, "op": "multiply", "b": 2.0, "c": 0.0,
             "clamp": False, "swap": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.2, 0.4, 0.6])

    def test_math_subtract_swap(self):
        # swap=True: out = b - input
        g = {"nodes": [
            _const([0.25, 0.5, 0.75]),
            {"type": "math", "input": 0, "op": "subtract", "b": 1.0, "c": 0.0,
             "clamp": False, "swap": True},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.75, 0.5, 0.25])

    def test_math_clamp(self):
        # multiply by 10, clamp on → values past 1 saturate.
        g = {"nodes": [
            _const([0.1, 0.5, 0.9]),
            {"type": "math", "input": 0, "op": "multiply", "b": 10.0, "c": 0.0,
             "clamp": True, "swap": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [1.0, 1.0, 1.0])

    def test_math_power_swap(self):
        # swap=True: out = b^input. With b=2, input=[1,2,3] → [2,4,8].
        g = {"nodes": [
            _const([1.0, 2.0, 3.0]),
            {"type": "math", "input": 0, "op": "power", "b": 2.0, "c": 0.0,
             "clamp": False, "swap": True},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [2.0, 4.0, 8.0], eps=1e-5)

    def test_math_divide_zero_b(self):
        # b=0, swap=False → output is all-zero (matches devicecode.cu fast path).
        g = {"nodes": [
            _const([1.0, 2.0, 3.0]),
            {"type": "math", "input": 0, "op": "divide", "b": 0.0, "c": 0.0,
             "clamp": False, "swap": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.0, 0.0, 0.0])

    # --- HueSat ---
    def test_huesat_identity(self):
        # hue=0.5, sat=1, val=1, fac=1 is identity (Blender's neutral hue).
        g = {"nodes": [
            _const([0.3, 0.6, 0.9]),
            {"type": "hue_sat", "input": 0, "hue": 0.5, "saturation": 1.0,
             "value": 1.0, "fac": 1.0},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.3, 0.6, 0.9], eps=1e-5)

    def test_huesat_hue_rotate_red_to_green(self):
        # Red (1,0,0) with hue += 1/3 should land on green (0,1,0).
        g = {"nodes": [
            _const([1.0, 0.0, 0.0]),
            {"type": "hue_sat", "input": 0, "hue": 0.5 + 1.0 / 3.0,
             "saturation": 1.0, "value": 1.0, "fac": 1.0},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.0, 1.0, 0.0], eps=1e-5)

    def test_huesat_desaturate(self):
        # saturation=0 → grey (luminance-ish single value).
        g = {"nodes": [
            _const([1.0, 0.0, 0.0]),
            {"type": "hue_sat", "input": 0, "hue": 0.5, "saturation": 0.0,
             "value": 1.0, "fac": 1.0},
        ]}
        out = _out_rgb(fold_constants(g))
        # Grey: all channels equal to v (=1.0 here, since saturation=0 leaves
        # value unchanged at the rgb max).
        self.assertAlmostEqual(out[0], out[1], delta=1e-5)
        self.assertAlmostEqual(out[1], out[2], delta=1e-5)

    # --- BrightContrast ---
    def test_bright_contrast_bright_only(self):
        g = {"nodes": [
            _const([0.2, 0.4, 0.6]),
            {"type": "bright_contrast", "input": 0, "bright": 0.1, "contrast": 0.0},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.3, 0.5, 0.7])

    def test_bright_contrast_floor_at_zero(self):
        # bright=-0.5, contrast=0: a=1, b=-0.5, out = rgb - 0.5 (clamped >= 0).
        g = {"nodes": [
            _const([0.1, 0.4, 0.9]),
            {"type": "bright_contrast", "input": 0, "bright": -0.5, "contrast": 0.0},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.0, 0.0, 0.4])

    # --- Mix ---
    def test_mix_midpoint(self):
        g = {"nodes": [
            _const([0.0, 0.0, 0.0]),
            _const([1.0, 1.0, 1.0]),
            {"type": "mix", "a": 0, "b": 1, "fac": 0.5, "blend": "mix",
             "clamp": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.5, 0.5, 0.5])

    def test_mix_fac_node_luminance(self):
        # fac node grey (0.5,0.5,0.5) → luminance = 0.5.
        g = {"nodes": [
            _const([0.0, 0.0, 0.0]),
            _const([1.0, 1.0, 1.0]),
            _const([0.5, 0.5, 0.5]),
            {"type": "mix", "a": 0, "b": 1, "fac": {"node": 2},
             "blend": "mix", "clamp": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.5, 0.5, 0.5])

    def test_mix_multiply(self):
        # Multiply with fac=1: out = a*b per channel.
        g = {"nodes": [
            _const([0.5, 0.5, 0.5]),
            _const([0.4, 0.6, 0.8]),
            {"type": "mix", "a": 0, "b": 1, "fac": 1.0,
             "blend": "multiply", "clamp": False},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.2, 0.3, 0.4])

    def test_mix_screen(self):
        # Screen with fac=1: out = 1 - (1-a)(1-b).
        g = {"nodes": [
            _const([0.2, 0.5, 0.8]),
            _const([0.4, 0.4, 0.4]),
            {"type": "mix", "a": 0, "b": 1, "fac": 1.0,
             "blend": "screen", "clamp": False},
        ]}
        # 1 - (1 - 0.2)*(1 - 0.4) = 1 - 0.48 = 0.52
        _approx(self, _out_rgb(fold_constants(g)), [0.52, 0.7, 0.88], eps=1e-6)

    def test_mix_clamp_overflow(self):
        g = {"nodes": [
            _const([2.0, 2.0, 2.0]),
            _const([3.0, 3.0, 3.0]),
            {"type": "mix", "a": 0, "b": 1, "fac": 0.5, "blend": "mix",
             "clamp": True},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [1.0, 1.0, 1.0])

    # --- RGBCurve ---
    def test_rgb_curve_identity(self):
        # Identity LUT: lut[channel*256 + i] = i / 255.
        lut = []
        for _ in range(3):
            for i in range(256):
                lut.append(i / 255.0)
        g = {"nodes": [
            _const([0.2, 0.5, 0.8]),
            {"type": "rgb_curve", "input": 0, "lut": lut},
        ]}
        _approx(self, _out_rgb(fold_constants(g)), [0.2, 0.5, 0.8],
                eps=1.0 / 255.0)

    def test_rgb_curve_invalid_length_skipped(self):
        # Wrong-sized LUT (validation will fire later); fold silently skips.
        g = {"nodes": [
            _const([0.2, 0.5, 0.8]),
            {"type": "rgb_curve", "input": 0, "lut": [0.0] * 100},
        ]}
        f = fold_constants(g)
        self.assertEqual(f["nodes"][1]["type"], "rgb_curve")

    # --- Non-foldable leaves block propagation ---
    def test_no_fold_image_tex_chain(self):
        g = {"nodes": [
            {"type": "image_tex", "tex": 0,
             "uv": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            {"type": "math", "input": 0, "op": "multiply", "b": 2.0,
             "c": 0.0, "clamp": False, "swap": False},
        ]}
        f = fold_constants(g)
        self.assertEqual(f["nodes"][0]["type"], "image_tex")
        self.assertEqual(f["nodes"][1]["type"], "math")

    def test_no_fold_vertex_color_chain(self):
        g = {"nodes": [
            {"type": "vertex_color"},
            {"type": "hue_sat", "input": 0, "hue": 0.5, "saturation": 1.0,
             "value": 1.0, "fac": 1.0},
        ]}
        f = fold_constants(g)
        self.assertEqual(f["nodes"][0]["type"], "vertex_color")
        self.assertEqual(f["nodes"][1]["type"], "hue_sat")

    # --- Deep chain + output preservation ---
    def test_deep_chain_collapses(self):
        # Const -> Math(*2) -> Math(+0.1) -> Mix(_, Const, fac=0.5)
        g = {"nodes": [
            _const([0.1, 0.2, 0.4]),
            {"type": "math", "input": 0, "op": "multiply", "b": 2.0,
             "c": 0.0, "clamp": False, "swap": False},  # (0.2, 0.4, 0.8)
            {"type": "math", "input": 1, "op": "add", "b": 0.1,
             "c": 0.0, "clamp": False, "swap": False},  # (0.3, 0.5, 0.9)
            _const([0.1, 0.1, 0.1]),
            {"type": "mix", "a": 2, "b": 3, "fac": 0.5, "blend": "mix",
             "clamp": False},  # average → (0.2, 0.3, 0.5)
        ]}
        f = fold_constants(g)
        for i, n in enumerate(f["nodes"]):
            self.assertEqual(n["type"], "const",
                             msg=f"node {i} should have folded to const")
        _approx(self, _out_rgb(f), [0.2, 0.3, 0.5])

    def test_output_index_preserved(self):
        g = {"nodes": [
            _const([0.1, 0.1, 0.1]),
            {"type": "math", "input": 0, "op": "add", "b": 0.5,
             "c": 0.0, "clamp": False, "swap": False},
            _const([0.9, 0.9, 0.9]),
        ], "output": 1}
        f = fold_constants(g)
        self.assertEqual(f.get("output"), 1)
        _approx(self, _out_rgb(f), [0.6, 0.6, 0.6])

    # --- Unsupported op falls through ---
    def test_unknown_blend_skipped(self):
        g = {"nodes": [
            _const([0.0, 0.0, 0.0]),
            _const([1.0, 1.0, 1.0]),
            {"type": "mix", "a": 0, "b": 1, "fac": 0.5, "blend": "tropical",
             "clamp": False},
        ]}
        f = fold_constants(g)
        self.assertEqual(f["nodes"][2]["type"], "mix")

    def test_does_not_mutate_input(self):
        g = {"nodes": [
            _const([0.5, 0.5, 0.5]),
            {"type": "invert", "input": 0, "fac": 1.0},
        ]}
        before = repr(g)
        fold_constants(g)
        self.assertEqual(repr(g), before, "input graph was mutated")


if __name__ == "__main__":
    unittest.main()
