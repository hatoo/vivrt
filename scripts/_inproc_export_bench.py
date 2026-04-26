"""Benchmark `export_scene_to_memory` (the in-process FFI path) on the
currently-loaded .blend file. Optionally also do a 1-spp render via
vibrt_native to verify the array-list path works end-to-end. Run via:

    blender -b path/to/scene.blend --python scripts/_inproc_export_bench.py

The exporter prints its own per-bucket breakdown; this script reports the
total wall time, bin-buffer size, and (if requested) render time.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))
# Evict any installed addon copy so we always exercise the working tree.
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

import bpy  # noqa: E402

from vibrt_blender import exporter  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
print("=== in-process: export_scene_to_memory ===")
t0 = time.perf_counter()
json_str, buf, texture_arrays = exporter.export_scene_to_memory(depsgraph, texture_pct=100)
dt = time.perf_counter() - t0
total_tex_mb = sum(a.nbytes for a in texture_arrays) / 1024 / 1024
print(f"[bench] total wall = {dt:.2f}s, bin = {len(buf) / 1024 / 1024:.1f}MB, "
      f"json = {len(json_str) / 1024:.1f}KB, "
      f"{len(texture_arrays)} texture arrays = {total_tex_mb:.1f}MB")

# Optional render — skipped if vibrt_native isn't bundled.
try:
    sys.path.insert(0, str(REPO / "blender" / "vibrt_blender"))
    import vibrt_native  # type: ignore
except ImportError:
    print("[bench] vibrt_native not importable — skipping render")
    sys.exit(0)

print("\n=== render via vibrt_native ===")
def log_cb(s):
    print(f"  {s.rstrip()}")
def cancel_cb():
    return False
opts = {"spp": 1, "width": 200, "height": 100}
t0 = time.perf_counter()
pixels = vibrt_native.render(
    json_str, buf, opts, log_cb, cancel_cb,
    texture_arrays=texture_arrays,
)
dt = time.perf_counter() - t0
print(f"[bench] render wall = {dt:.2f}s, output shape = {pixels.shape}")
