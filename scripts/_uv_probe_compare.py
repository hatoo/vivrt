"""Blender-side comparison half of scripts/uv_probe.py.

Loads probe_cycles.png and probe_vibrt.png, samples them at known UV
locations, and prints a side-by-side table. Exits non-zero on any corner
mismatch. Run under:
  blender --background --factory-startup --python scripts/_uv_probe_compare.py
"""
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "test_scenes" / "uv_probe"


def load(p: Path):
    img = bpy.data.images.load(str(p), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    # Blender gives us pixels bottom-up; flip to human (row 0 = top) order so
    # "v" below reads as "fraction from top", matching how the human sees
    # the rendered PNG.
    return px.reshape((h, w, 4))[::-1], w, h


def label(rgb) -> str:
    """Classify the dominant channel. A color-managed Cycles render applies
    Filmic, which desaturates pure (0,1,0) to something like (0.56,0.83,0.49)
    — so we can't use a fixed per-channel threshold to split R/G/B from RGB.
    Instead we compare each channel against the other two's mean, and call
    a color "R" when R clearly beats (G+B)/2, etc.
    """
    r, g, b = [float(c) for c in rgb]
    mx = max(r, g, b)
    if mx < 0.2:
        return "K"
    others = sorted([("R", r), ("G", g), ("B", b)], key=lambda kv: kv[1], reverse=True)
    top, second = others[0][1], others[1][1]
    if top - second < 0.1:
        # Not clearly dominated by any channel — near-neutral grey / white.
        return "RGB"
    return others[0][0]


c, w, h = load(OUT / "probe_cycles.png")
v, _, _ = load(OUT / "probe_vibrt.png")

probes = [
    # (name, u, v-from-top). Stay a few % away from the image edge so the
    # swatches (each 64/512 ≈ 12.5% wide) are sampled well inside, not right
    # at the corner where filtering + grey background bleed in.
    ("top-left",     0.07, 0.07),
    ("top-right",    0.93, 0.07),
    ("bottom-left",  0.07, 0.93),
    ("bottom-right", 0.93, 0.93),
    ("F centre",     0.30, 0.50),
]

print(f"{'probe':<16}{'uv':<14}{'Cycles':<22}{'vibrt':<22}match?")
print("-" * 80)
mismatches = 0
for name, u, vv in probes:
    cy, cx = int(vv * (h - 1)), int(u * (w - 1))
    cr = c[cy, cx, :3]
    vr = v[cy, cx, :3]
    cl, vl = label(cr), label(vr)
    ok = cl == vl
    mismatches += (not ok)
    print(f"{name:<16}({u:.2f},{vv:.2f})   "
          f"{cr[0]:.2f},{cr[1]:.2f},{cr[2]:.2f} [{cl:<4}]   "
          f"{vr[0]:.2f},{vr[1]:.2f},{vr[2]:.2f} [{vl:<4}]   "
          f"{'OK' if ok else 'MISMATCH'}")

if mismatches:
    print(f"\nFAIL: {mismatches} mismatch(es) — UV orientation drift between Cycles and vibrt")
    import sys
    sys.exit(1)
print("\nOK: corner labels match — UV axes agree between Cycles and vibrt")
