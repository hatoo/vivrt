"""Render a .blend file via vibrt from the command line.

Usage:
  python scripts/render_blend.py scene.blend -o out.png --spp 128

Blender is invoked in background mode to export the scene to
vibrt's JSON+binary format, then the vibrt CLI is run on the result.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
DEFAULT_VIBRT = str(REPO / "target" / "release" / "vibrt.exe")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blend", type=Path, help="input .blend file")
    ap.add_argument("--output", "-o", type=Path, default=Path("render.png"))
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--denoise", action="store_true",
                    help="run the OptiX AI denoiser on the final image")
    ap.add_argument("--percentage", type=int, default=None,
                    help="override render resolution percentage (e.g. 25 for 1/4 size)")
    ap.add_argument("--texture-pct", type=int, default=None,
                    help="downsample exported textures to N%% of original size "
                         "(e.g. 25 for quarter-res); defaults to full size")
    ap.add_argument("--scene", default=None,
                    help="Blender scene name to use (default: active scene)")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    ap.add_argument("--vibrt", default=os.environ.get("VIBRT_EXECUTABLE", DEFAULT_VIBRT))
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    blend = args.blend.resolve()
    if not blend.exists():
        print(f"error: not found: {blend}", file=sys.stderr)
        return 1
    if not Path(args.blender).exists():
        print(f"error: blender not found: {args.blender}", file=sys.stderr)
        return 1
    if not Path(args.vibrt).exists():
        print(f"error: vibrt binary not found: {args.vibrt}", file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="vibrt_render_"))
    try:
        print(f"[1/2] Exporting {blend.name} via Blender...")
        inner = HERE / "_blender_export.py"
        export_argv = ["--workdir", str(workdir)]
        if args.percentage is not None:
            export_argv += ["--percentage", str(args.percentage)]
        if args.texture_pct is not None:
            export_argv += ["--texture-pct", str(args.texture_pct)]
        if args.scene is not None:
            export_argv += ["--scene", args.scene]
        cmd = [
            args.blender, "-b", str(blend),
            "--python", str(inner),
            "--", *export_argv,
        ]
        subprocess.run(cmd, check=True)

        scene_json = workdir / "scene.json"
        if not scene_json.exists():
            print(f"error: export failed; {scene_json} not produced", file=sys.stderr)
            return 1

        print(f"[2/2] Running vibrt (spp={args.spp})...")
        vibrt_cmd = [args.vibrt, str(scene_json),
                     "--spp", str(args.spp),
                     "--output", str(args.output.resolve())]
        if args.max_depth is not None:
            vibrt_cmd += ["--depth", str(args.max_depth)]
        if args.denoise:
            vibrt_cmd += ["--denoise"]
        subprocess.run(vibrt_cmd, check=True)
        print(f"Done: {args.output}")
        return 0
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"(kept workdir: {workdir})")


if __name__ == "__main__":
    sys.exit(main())
