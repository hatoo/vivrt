"""Quick bench: export + render junk_shop, dump pixel stats."""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blender"))
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from vibrt_blender import exporter, runner  # noqa: E402

depsgraph = bpy.context.evaluated_depsgraph_get()
print("=== export ===")
t0 = time.perf_counter()
json_str, mesh_blobs, texture_arrays = exporter.export_scene_to_memory(depsgraph, texture_pct=10)
print(f"[bench] export = {time.perf_counter() - t0:.2f}s")
print(f"[bench] mesh_blobs: {len(mesh_blobs)} entries, "
      f"total = {sum(b.nbytes for b in mesh_blobs) / 1024:.1f} KB")
print(f"[bench] texture_arrays: {len(texture_arrays)} entries")

# Inspect a few blobs
import json as _json
d = _json.loads(json_str)
print(f"[bench] meshes in json: {len(d.get('meshes', []))}")
if d.get("meshes"):
    m0 = d["meshes"][0]
    print(f"[bench] mesh[0] keys: {list(m0.keys())}")
    print(f"[bench] mesh[0].vertices = {m0.get('vertices')}  indices = {m0.get('indices')}")

# Render
sys.path.insert(0, str(REPO / "blender" / "vibrt_blender"))
import vibrt_native  # type: ignore  # noqa: E402

print("\n=== render ===")
def log_cb(s): print(f"  {s.rstrip()}")
def cancel_cb(): return False

# Mimic runner.run_render_inproc's .view(np.uint8) reinterpret
blobs_u8 = [b if b.dtype == np.uint8 else b.view(np.uint8) for b in mesh_blobs]
opts = {"denoise": False, "spp": 16, "width": 200, "height": 100}
pixels = vibrt_native.render(
    json_str, blobs_u8, opts, log_cb, cancel_cb,
    texture_arrays=texture_arrays,
)
print(f"\n[bench] pixels shape={pixels.shape} dtype={pixels.dtype}")
print(f"[bench] mean RGB = {pixels[..., :3].mean(axis=(0,1))}")
print(f"[bench] max RGB = {pixels[..., :3].max(axis=(0,1))}")
print(f"[bench] min RGB = {pixels[..., :3].min(axis=(0,1))}")
print(f"[bench] (0, 0) = {pixels[0, 0]}")
print(f"[bench] (h//2, w//2) = {pixels[pixels.shape[0]//2, pixels.shape[1]//2]}")
