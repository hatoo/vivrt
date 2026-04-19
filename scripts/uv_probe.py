"""Controlled A/B test for texture UV handling: render the same scene in
Cycles and vibrt, sample matching pixels, and flag any orientation drift.

Puts a plane in an empty scene, textures it with a probe PNG whose four
corners and edges are uniquely identifiable (colored swatches + an
asymmetric "F" glyph), fits the camera to the plane exactly, then renders
with both engines and compares pixel values at known UV locations. Corner
label mismatch means U/V/axis-swap.

Outputs in ``test_scenes/uv_probe/``:
  probe.png          — the probe texture
  uv_probe.blend     — the scene (for manual inspection)
  probe_cycles.png   — Cycles render
  probe_vibrt.png    — vibrt render
  scene.json/bin     — vibrt-exported scene

Run:
  python scripts/uv_probe.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
DEFAULT_VIBRT = str(REPO / "target" / "release" / "vibrt.exe")
OUT_DIR = REPO / "test_scenes" / "uv_probe"


def _run(cmd: list[str], label: str) -> int:
    print(f"[{label}] {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    ap.add_argument("--vibrt", default=os.environ.get("VIBRT_EXECUTABLE", DEFAULT_VIBRT))
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = _run(
        [args.blender, "--background", "--factory-startup",
         "--python", str(HERE / "_uv_probe_build.py")],
        "build+export",
    )
    if rc != 0:
        return rc

    rc = _run(
        [args.vibrt, str(OUT_DIR / "scene.json"),
         "-o", str(OUT_DIR / "probe_vibrt.png"),
         "--spp", str(args.spp),
         "--width", str(args.width), "--height", str(args.height)],
        "vibrt",
    )
    if rc != 0:
        return rc

    return _run(
        [args.blender, "--background", "--factory-startup",
         "--python", str(HERE / "_uv_probe_compare.py")],
        "compare",
    )


if __name__ == "__main__":
    sys.exit(main())
