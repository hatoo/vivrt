"""Render a .blend file with Blender's Cycles engine (reference output).

Usage:
  python scripts/render_cycles.py scene.blend -o cycles.png --spp 128

Mirrors scripts/render_blend.py but produces a Cycles reference image for
side-by-side comparison with vibrt output.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blend", type=Path, help="input .blend file")
    ap.add_argument("--output", "-o", type=Path, default=Path("cycles.png"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--percentage", type=int, default=None,
                    help="override render resolution percentage (e.g. 25 for 1/4 size)")
    ap.add_argument("--scene", default=None,
                    help="Blender scene name to use (default: active scene)")
    ap.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()

    blend = args.blend.resolve()
    if not blend.exists():
        print(f"error: not found: {blend}", file=sys.stderr)
        return 1
    if not Path(args.blender).exists():
        print(f"error: blender not found: {args.blender}", file=sys.stderr)
        return 1

    inner = HERE / "_blender_cycles.py"
    inner_argv = ["--output", str(args.output.resolve()),
                  "--spp", str(args.spp),
                  "--device", args.device]
    if args.percentage is not None:
        inner_argv += ["--percentage", str(args.percentage)]
    if args.scene is not None:
        inner_argv += ["--scene", args.scene]

    cmd = [args.blender, "-b", str(blend),
           "--python", str(inner),
           "--", *inner_argv]
    subprocess.run(cmd, check=True)
    print(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
