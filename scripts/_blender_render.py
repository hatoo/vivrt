"""Inside `blender -b ... --python` — render the active scene with vibrt.

Forces a fresh import of the working-tree addon (any installed copy on
sys.modules is evicted), registers it, sets `render.engine = VIBRT`,
and calls `bpy.ops.render.render(write_still=True)` so the output PNG
lands at `args.output`.

Not intended to be invoked directly — use `scripts/render_blend.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))

# Force-fresh import of the source addon (the dev-junctioned addon may
# already be on sys.modules — evict any cached bytes).
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

import vibrt_blender  # noqa: E402
from vibrt_blender import runner  # noqa: E402

if runner.find_native_module() is None:
    raise SystemExit(
        "vibrt_native not importable — run `make dev-install` (or `make "
        "addon-with-native`) and try again."
    )


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--percentage", type=int, default=None)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args(argv)

    if args.scene is not None:
        if args.scene not in bpy.data.scenes:
            raise SystemExit(
                f"scene not found: {args.scene!r} "
                f"(available: {[s.name for s in bpy.data.scenes]})"
            )
        bpy.context.window.scene = bpy.data.scenes[args.scene]

    scene = bpy.context.scene
    scene.render.engine = "VIBRT"
    if args.percentage is not None:
        scene.render.resolution_percentage = args.percentage
    scene.vibrt_spp = args.spp
    scene.vibrt_denoise = bool(args.denoise)
    scene.render.filepath = args.output
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.render.render(write_still=True)
    print(f"[vibrt] wrote {args.output}")


vibrt_blender.register()
try:
    main()
finally:
    vibrt_blender.unregister()
