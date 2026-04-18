"""Invoked inside `blender -b ... --python` to export the scene to vibrt format.

Not intended to be run directly — use scripts/render_blend.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))

# An installed addon of the same name may already be in sys.modules — evict it
# so the repo source is loaded instead.
for mod_name in list(sys.modules):
    if mod_name == "vibrt_blender" or mod_name.startswith("vibrt_blender."):
        del sys.modules[mod_name]

from vibrt_blender import exporter  # noqa: E402

expected = str(REPO / "blender")
actual = Path(exporter.__file__).resolve().parent.parent
if str(actual) != expected:
    raise SystemExit(
        f"[vibrt] exporter loaded from unexpected path: {actual} (expected {expected})"
    )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--percentage", type=int, default=None)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args(argv)

    if args.scene is not None:
        if args.scene not in bpy.data.scenes:
            raise SystemExit(f"scene not found: {args.scene!r} "
                             f"(available: {[s.name for s in bpy.data.scenes]})")
        bpy.context.window.scene = bpy.data.scenes[args.scene]

    if args.percentage is not None:
        bpy.context.scene.render.resolution_percentage = args.percentage

    args.workdir.mkdir(parents=True, exist_ok=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    exporter.export_scene(
        depsgraph,
        args.workdir / "scene.json",
        args.workdir / "scene.bin",
    )
    print(f"[vibrt] exported to {args.workdir}")


main()
