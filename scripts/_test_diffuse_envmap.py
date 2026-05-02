"""Diagnostic: render a grey diffuse plane (albedo=0.5) lit by a uniform
envmap of strength=1.0. The expected pixel value is exactly albedo*L = 0.5
(since for a Lambertian surface lit by uniform L, integrated reflectance
equals albedo*L). Any deviation indicates a normalization bug in either
the envmap CDF or the MIS weighting.

We run two variants:
  - constant-color world (no bake, no CDF)
  - envmap world (forces a Cycles bake → CDF) at the same brightness
"""
import bpy, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "blender"))
import vibrt_blender  # noqa
try:
    bpy.ops.preferences.addon_enable(module="vibrt_blender")
except Exception:
    pass

# Wipe scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat, do_unlink=True)
for me in list(bpy.data.meshes):
    bpy.data.meshes.remove(me, do_unlink=True)
for w in list(bpy.data.worlds):
    bpy.data.worlds.remove(w, do_unlink=True)

scene = bpy.context.scene

# Grey diffuse plane (albedo = 0.5) facing up.
mat = bpy.data.materials.new("grey")
mat.use_nodes = True
mat.node_tree.nodes.clear()
out_n = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
diff = mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
diff.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
mat.node_tree.links.new(diff.outputs[0], out_n.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
plane = bpy.context.object
plane.data.materials.append(mat)

# Camera looking straight down so cosine factor = 1 and we sample mostly
# the upper hemisphere of the environment.
import math
bpy.ops.object.camera_add(location=(0, 0, 5),
                          rotation=(0, 0, 0))
scene.camera = bpy.context.object

# Constant-color world, strength 1.0 → uniform L = 1.
w_const = bpy.data.worlds.new("const_world")
w_const.use_nodes = True
w_const.node_tree.nodes.clear()
out_w = w_const.node_tree.nodes.new("ShaderNodeOutputWorld")
bg = w_const.node_tree.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs["Strength"].default_value = 1.0
w_const.node_tree.links.new(bg.outputs[0], out_w.inputs["Surface"])

# Envmap-driven world: a uniform-grey 2k×1k texture (no Cycles bake needed
# because the world surface is a plain Background → TexEnvironment, which
# `_export_world` recognizes directly and ships as the envmap with
# strength=1).
img = bpy.data.images.new("uniform_env", width=2048, height=1024,
                           alpha=False, float_buffer=True)
flat = np.ones(2048 * 1024 * 4, dtype=np.float32)
img.pixels.foreach_set(flat)
img.colorspace_settings.name = "Non-Color"

w_env = bpy.data.worlds.new("env_world")
w_env.use_nodes = True
w_env.node_tree.nodes.clear()
out_w2 = w_env.node_tree.nodes.new("ShaderNodeOutputWorld")
bg2 = w_env.node_tree.nodes.new("ShaderNodeBackground")
bg2.inputs["Strength"].default_value = 1.0
tex = w_env.node_tree.nodes.new("ShaderNodeTexEnvironment")
tex.image = img
w_env.node_tree.links.new(tex.outputs["Color"], bg2.inputs["Color"])
w_env.node_tree.links.new(bg2.outputs[0], out_w2.inputs["Surface"])

# Use in-memory image directly. Verify pixels first.
print(f"image channels={img.channels}")
verify = np.empty(2048 * 1024 * img.channels, dtype=np.float32)
img.pixels.foreach_get(verify)
print(f"in-memory image (after foreach_set ones): "
      f"min={verify.min()} max={verify.max()} mean={verify.mean()}")
# Re-set explicitly to make sure all channels are 1.
img.pixels.foreach_set(np.ones(2048 * 1024 * 4, dtype=np.float32))

# Also build a 2nd envmap with a SINGLE bright spot at zenith. The
# expected diffuse-plane brightness is `albedo · ∫_hemisphere L cos θ
# dω / π` — for a delta light at zenith this is `albedo · L_total` where
# L_total = pixel_radiance × pixel_solid_angle. We pick a 16×16 pixel
# patch at the bake's bottom row (zenith in our convention) at a
# luminance such that the integrated irradiance is 0.5; reflectance off
# albedo=0.5 should then read 0.25.
img_hot = bpy.data.images.new("hot_env", width=2048, height=1024,
                                alpha=False, float_buffer=True)
hot = np.zeros(2048 * 1024 * 4, dtype=np.float32)
# Set a 16×16 cluster of pixels at the bottom (= zenith in our equirect
# convention, since foreach_get is bottom-up and the bake camera points
# zenith at the bottom of the buffer). cos(zenith) = 1, so the
# irradiance integral = L × Ω where Ω is the cluster's solid angle.
patch_size = 16
W, H = 2048, 1024
patch_x = (W - patch_size) // 2
patch_y = 0  # bottom of buffer (zenith)
# Per-pixel solid angle at row y in equirect: 2π² sin(θ) / (W·H).
# At y=0 in buffer = zenith, in our convention theta_kernel = 0. But
# the ROW of the bottom pixel in equirect-Cycles convention is theta=π
# (opposite end of the v axis from kernel). The bottom pixel's solid
# angle in equirect is what matters: Ω_pixel = 2π²·sin(π·v_pixel)/(W·H).
# v=0 → sin(0)=0 (degenerate at exact pole). So pick row y=h//2 (middle
# of buffer) instead — that's near horizon (theta=π/2, sin=1).
patch_y = H // 2
for dy in range(patch_size):
    for dx in range(patch_size):
        idx = ((patch_y + dy) * W + (patch_x + dx)) * 4
        hot[idx + 0] = 100.0  # bright cluster
        hot[idx + 1] = 100.0
        hot[idx + 2] = 100.0
        hot[idx + 3] = 1.0
img_hot.pixels.foreach_set(hot)
img_hot.colorspace_settings.name = "Non-Color"

w_hot = bpy.data.worlds.new("hot_world")
w_hot.use_nodes = True
w_hot.node_tree.nodes.clear()
out_w3 = w_hot.node_tree.nodes.new("ShaderNodeOutputWorld")
bg3 = w_hot.node_tree.nodes.new("ShaderNodeBackground")
bg3.inputs["Strength"].default_value = 1.0
tex3 = w_hot.node_tree.nodes.new("ShaderNodeTexEnvironment")
tex3.image = img_hot
w_hot.node_tree.links.new(tex3.outputs["Color"], bg3.inputs["Color"])
w_hot.node_tree.links.new(bg3.outputs[0], out_w3.inputs["Surface"])

# Compute expected reflectance:
# Each patch pixel covers Ω_pixel = 2π² sin(θ) / (W·H)
# At y=H/2, theta_buffer=π/2, sin=1. So Ω_pixel = 2π² / (W·H) = 9.42e-6
# 16×16 patch total Ω = 256 * 9.42e-6 = 2.41e-3 sr
# Patch is at theta_buffer=π/2 (= horizon in our equirect-bake
# convention). For a plane facing up, NoL = cos(theta_world) where the
# patch's world direction has theta_world = π/2 (horizon). So NoL = 0
# at the horizon — patch contributes nothing to a top-facing diffuse!
# Expected reflectance = albedo · L · Ω · NoL · INV_PI = 0.
print()
print("Note: Hot patch at horizon → NoL=0 for top-facing plane → "
      "diffuse reflectance should be ~0.")
verify = np.empty(2048 * 1024 * img.channels, dtype=np.float32)
img.pixels.foreach_get(verify)
print(f"after explicit set: min={verify.min()} max={verify.max()} mean={verify.mean()}")

scene.render.engine = "VIBRT"
scene.vibrt_spp = 256
scene.vibrt_clamp_indirect = 0.0  # disable clamp for diagnostic
scene.render.resolution_x = 200
scene.render.resolution_y = 200
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "16"
scene.view_settings.view_transform = "Raw"  # avoid sRGB clamp
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

results = {}
for name, world in [("const", w_const), ("envmap", w_env)]:
    scene.world = world
    out = os.path.join(HERE, "test_scenes", f"_diffuse_{name}.png")
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    # Read the centre pixel (over the plane, not the world)
    rendered = bpy.data.images.load(out)
    rendered.colorspace_settings.name = "Non-Color"
    w_, h_ = int(rendered.size[0]), int(rendered.size[1])
    flat = np.empty(w_ * h_ * 4, dtype=np.float32)
    rendered.pixels.foreach_get(flat)
    arr = flat.reshape((h_, w_, 4))[..., :3]
    # Centre 100×100
    cx, cy = w_ // 2, h_ // 2
    box = arr[cy - 50:cy + 50, cx - 50:cx + 50, :]
    mean = tuple(float(box[..., c].mean()) for c in range(3))
    print(f"\n=== {name}: world strength=1.0, plane albedo=0.5 ===")
    print(f"  Expected: albedo * L = 0.5 * 1.0 = 0.500")
    print(f"  Measured: rgb={tuple(round(c, 4) for c in mean)}")
    print(f"  Ratio: {mean[0] / 0.5:.3f}× expected")
    results[name] = mean

print("\nIf any ratio is > 1.05× or < 0.95×, there's a normalization bug.")
