"""Blender-side half of scripts/uv_probe.py.

Builds a minimal one-plane scene, writes a probe texture, renders it in
Cycles, then exports the same scene for vibrt. Run under:
  blender --background --factory-startup --python scripts/_uv_probe_build.py
"""
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "test_scenes" / "uv_probe"
ADDON_SRC = REPO / "blender"

OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ADDON_SRC))
# Evict any pre-loaded installed addon of the same name so the repo source wins.
for _m in list(sys.modules):
    if _m == "vibrt_blender" or _m.startswith("vibrt_blender."):
        del sys.modules[_m]

# ---- probe texture ----------------------------------------------------------
W = H = 512
# arr[0] is the first row passed to foreach_set. Blender stores pixels
# bottom-up, so arr[0] lands at the BOTTOM of the saved PNG. Corner labels
# are chosen so the on-disk PNG matches a human's mental map:
#   top-left = blue  top-right = white
#   bottom-left = red  bottom-right = green
arr = np.full((H, W, 4), [0.5, 0.5, 0.5, 1.0], dtype=np.float32)
S = 64
arr[0:S,     0:S    ] = [1, 0, 0, 1]   # RED
arr[0:S,     W - S:W] = [0, 1, 0, 1]   # GREEN
arr[H - S:H, 0:S    ] = [0, 0, 1, 1]   # BLUE
arr[H - S:H, W - S:W] = [1, 1, 1, 1]   # WHITE
# Asymmetric "F" glyph: the only way a V-flip or U-flip can hide is if the
# glyph is symmetric. An F is not.
arr[80:400,  80:120 ] = [0, 0, 0, 1]   # vertical stroke
arr[80:120,  80:320 ] = [0, 0, 0, 1]   # longer horizontal (numpy-"top")
arr[200:240, 80:260 ] = [0, 0, 0, 1]   # shorter horizontal ("middle")

probe_path = str(OUT / "probe.png")
img = bpy.data.images.new("probe", W, H, alpha=True, float_buffer=True)
img.colorspace_settings.name = "Linear Rec.709"
img.pixels.foreach_set(arr.reshape(-1))
img.update()
img.file_format = "PNG"
img.save(filepath=probe_path)

# ---- scene ------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
plane = bpy.context.active_object

# Perspective camera sized to exactly frame the 2x2 plane. vibrt doesn't
# support ORTHO, so we use perspective on both sides and compute distance:
# hFOV = 2·atan(36/(2·50)) ≈ 39.6° → d = 1/tan(hFOV/2) ≈ 2.7777.
bpy.ops.object.camera_add(location=(0, 0, 2.7777), rotation=(0, 0, 0))
cam = bpy.context.active_object
cam.data.lens = 50.0
cam.data.sensor_width = 36.0
cam.data.sensor_height = 36.0
# Force VERTICAL fit so Cycles matches vibrt's fov_y-based camera exactly
# (vibrt has no concept of sensor_fit; it reads fov_y_rad directly).
cam.data.sensor_fit = "VERTICAL"
bpy.context.scene.camera = cam

bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
bpy.context.active_object.data.energy = 5.0

mat = bpy.data.materials.new("ProbeMat")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(probe_path, check_existing=False)
tex.image.colorspace_settings.name = "sRGB"
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
bsdf.inputs["Roughness"].default_value = 0.8
plane.data.materials.append(mat)

scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.render.resolution_x = 512
scn.render.resolution_y = 512
scn.cycles.samples = 32
scn.cycles.use_denoising = False
scn.world = bpy.data.worlds.new("World")
scn.world.use_nodes = True
bg = scn.world.node_tree.nodes.get("Background")
bg.inputs["Strength"].default_value = 0.0

bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "uv_probe.blend"))

scn.render.filepath = str(OUT / "probe_cycles.png")
scn.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)

# ---- vibrt export -----------------------------------------------------------
import vibrt_blender
vibrt_blender.register()
import vibrt_blender.exporter as vexp
vexp.export_scene(
    bpy.context.evaluated_depsgraph_get(),
    OUT / "scene.json",
    OUT / "scene.bin",
)
print(f"[uv_probe] wrote {OUT / 'scene.json'}")
