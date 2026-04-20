"""Invoked inside `blender -b ... --python` to render the scene with Cycles.

Not intended to be run directly — use scripts/render_cycles.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--percentage", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    args = ap.parse_args(argv)

    if args.scene is not None:
        if args.scene not in bpy.data.scenes:
            raise SystemExit(f"scene not found: {args.scene!r} "
                             f"(available: {[s.name for s in bpy.data.scenes]})")
        bpy.context.window.scene = bpy.data.scenes[args.scene]

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # Prefer OPTIX (matches vibrt), fall back to CUDA.
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                break
            except TypeError:
                continue
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("OPTIX", "CUDA")
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"

    if args.spp is not None:
        scene.cycles.samples = args.spp
        scene.cycles.use_adaptive_sampling = False
    if args.percentage is not None:
        scene.render.resolution_percentage = args.percentage

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Render to a temp dir then move, to sidestep Blender's frame-number /
    # extension handling on scene.render.filepath.
    with tempfile.TemporaryDirectory(prefix="cycles_render_") as td:
        scene.render.filepath = str(Path(td) / "frame")
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        bpy.ops.render.render(write_still=True)
        produced = next(iter(Path(td).glob("*.png")), None)
        if produced is None:
            raise SystemExit(f"[cycles] no PNG produced in {td}")
        shutil.move(str(produced), str(args.output))

    print(f"[cycles] rendered to {args.output}")


main()
