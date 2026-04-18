# vibrt-blender Blender Addon

A Blender render engine that delegates rendering to the `vibrt-blender` CLI.

## Installation

1. Build the renderer:

   ```
   cargo build --release -p vibrt-blender
   ```

   The binary will be at `target/release/vibrt-blender` (`vibrt-blender.exe` on Windows).

2. Zip the `blender/vibrt_blender/` directory into `vibrt_blender.zip`.

3. Open Blender → `Edit` → `Preferences` → `Add-ons` → `Install...` and select the zip.

4. Enable the `Render: vibrt-blender` addon.

5. Open the addon preferences panel and set the **vibrt-blender executable** path to the binary built above. Alternatively, set `$VIBRT_BLENDER_EXECUTABLE` or put the binary on `PATH`.

## Usage

1. In the Render Properties panel, set **Render Engine** to `vibrt-blender`.

2. Add geometry, lights (Point, Sun, Spot, or Area), and a camera.

3. Assign materials using the **Principled BSDF** shader (base color, metallic, roughness, IOR, transmission, and emission are exported).

4. Optionally set a World background with an **Environment Texture** (HDRI).

5. Press `F12` (or `Render > Render Image`). The addon will:
   - Export the scene to a temporary `scene.json` + `scene.bin` under Blender's `tempdir`.
   - Spawn `vibrt-blender` as a subprocess.
   - Load the resulting EXR back into the image editor.

## Supported features (Stage 1)

- Camera: perspective only (no DoF yet).
- Meshes: triangulated, per-corner normals and UVs.
- Materials: Principled BSDF — base color, metallic, roughness, IOR, transmission, emission, and image textures for base color / normal / roughness / metallic.
- Lights: Point, Sun, Spot, Area (square/rectangle).
- World: constant colour background or Environment Texture.
- Output: EXR loaded back into Blender's compositor/image-editor.

## Not supported yet

- Viewport IPR (live preview).
- Depth of field, motion blur, volumes.
- Sheen, clearcoat, subsurface, anisotropy, thin-film.
- Multi-material meshes (only material slot 0 is used per object).
