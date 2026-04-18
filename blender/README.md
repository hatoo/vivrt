# vibrt Blender Addon

A Blender render engine that delegates rendering to the `vibrt` CLI.

## Installation

1. Build the renderer:

   ```
   cargo build --release -p vibrt
   ```

   The binary will be at `target/release/vibrt` (`vibrt.exe` on Windows).

2. Build the addon zip:

   ```
   make addon
   ```

   (or, without make: `py blender/build_addon.py`). This writes `blender/vibrt_blender.zip`.

3. Open Blender → `Edit` → `Preferences` → `Add-ons` → `Install...` and select the zip.

4. Enable the `Render: vibrt` addon.

5. Open the addon preferences panel and set the **vibrt executable** path to the binary built above. Alternatively, set `$VIBRT_EXECUTABLE` or put the binary on `PATH`.

### Fast iteration (development)

```
make dev-install
```

creates a junction/symlink from `blender/vibrt_blender/` into Blender's user addons directory, so edits to the Python sources are picked up after a Blender restart without rezipping.

## Usage

1. In the Render Properties panel, set **Render Engine** to `vibrt`.

2. Add geometry, lights (Point, Sun, Spot, or Area), and a camera.

3. Assign materials using the **Principled BSDF** shader.

4. Optionally set a World background with an **Environment Texture** (HDRI).

5. In the Sampling panel, set **Samples** (`vibrt_spp`) and **Clamp Indirect** (`vibrt_clamp_indirect`).

6. Press `F12` (or `Render > Render Image`). The addon will:
   - Export the scene to a temporary `scene.json` + `scene.bin` under Blender's `tempdir`.
   - Spawn `vibrt` as a subprocess.
   - Load the resulting `.raw` float image back into the image editor.

## Supported features

- Camera: perspective only (no DoF yet).
- Meshes: triangulated, per-corner normals and UVs, multi-material slots (per-triangle material index).
- Materials: Principled BSDF — base color, metallic, roughness, IOR, transmission, emission, anisotropy (+ rotation), coat (weight / roughness / IOR), sheen (weight / roughness / tint), subsurface (weight / radius / anisotropy), alpha cutout, normal map (with strength), bump, displacement. Image textures on base color / normal / roughness / metallic. Colour- and scalar-math node chains between TexImage and the BSDF are traversed.
- Lights: Point, Sun, Spot, Area (square/rectangle).
- World: constant colour background or Environment Texture (importance-sampled).
- Transparent shadows through transmissive / alpha-cutout surfaces.
- Output: linear float image loaded into Blender's image editor.

## Not supported yet

- Viewport IPR (live preview).
- Depth of field, motion blur, volumes.
- Thin-film, true SSS (subsurface is currently diffuse-blend approximation).
