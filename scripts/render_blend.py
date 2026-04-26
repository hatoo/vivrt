"""Render a .blend file via vibrt from the command line.

The renderer runs in-process inside Blender via the bundled
`vibrt_native.pyd` extension — there is no standalone vibrt binary any
more. This script spawns Blender headless (`-b`), force-loads the
working-tree copy of the addon, and triggers a single F12 render.

Usage:
  python scripts/render_blend.py scene.blend -o out.png --spp 128
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
    ap.add_argument("--output", "-o", type=Path, default=Path("render.png"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--max-depth", type=int, default=None,
                    help="override max ray depth (default: scene's vibrt_spp / 8)")
    ap.add_argument("--denoise", action="store_true",
                    help="run the OptiX AI denoiser on the final image")
    ap.add_argument("--percentage", type=int, default=None,
                    help="override render resolution percentage (e.g. 25 for 1/4 size)")
    ap.add_argument("--scene", default=None,
                    help="Blender scene name (default: active scene)")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()

    blend = args.blend.resolve()
    if not blend.exists():
        print(f"error: not found: {blend}", file=sys.stderr)
        return 1
    if not Path(args.blender).exists():
        print(f"error: blender not found: {args.blender}", file=sys.stderr)
        return 1

    inner = HERE / "_blender_render.py"
    inner_argv = [
        "--output", str(args.output.resolve()),
        "--spp", str(args.spp),
    ]
    if args.percentage is not None:
        inner_argv += ["--percentage", str(args.percentage)]
    if args.max_depth is not None:
        inner_argv += ["--max-depth", str(args.max_depth)]
    if args.denoise:
        inner_argv += ["--denoise"]
    if args.scene is not None:
        inner_argv += ["--scene", args.scene]

    cmd = [
        args.blender, "-b", str(blend),
        "--python", str(inner),
        "--", *inner_argv,
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
