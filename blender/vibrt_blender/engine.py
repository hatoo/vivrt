"""Blender `RenderEngine` subclass for vibrt (final-frame F12 only).

Uses the bundled `vibrt_native.pyd` PyO3 extension exclusively — there's
no subprocess fallback. If the extension isn't importable the addon
errors out at render time with instructions on how to build it. Subprocess
rendering through the standalone `vibrt.exe` binary is still available
for CLI tooling (`scripts/render_blend.py`, `make <scene>-preview`) but
is not used by the addon itself.
"""

from __future__ import annotations

import time

import bpy

from . import _log, exporter, runner


class VibrtRenderEngine(bpy.types.RenderEngine):
    bl_idname = "VIBRT"
    bl_label = "vibrt"
    bl_use_preview = False
    bl_use_shading_nodes_custom = False

    def render(self, depsgraph: bpy.types.Depsgraph):
        scene = depsgraph.scene_eval
        rd = scene.render
        width = int(rd.resolution_x * rd.resolution_percentage / 100.0)
        height = int(rd.resolution_y * rd.resolution_percentage / 100.0)
        denoise = bool(getattr(scene, "vibrt_denoise", False))

        if runner.find_native_module() is None:
            self.report(
                {"ERROR"},
                "vibrt_native extension not bundled. Run `make dev-install` "
                "(or `make addon-with-native`) and restart Blender.",
            )
            return

        try:
            self._render_in_process(depsgraph, width, height, denoise)
        except KeyboardInterrupt:
            self.report({"WARNING"}, "Render cancelled")

    def _render_in_process(
        self,
        depsgraph: bpy.types.Depsgraph,
        width: int,
        height: int,
        denoise: bool,
    ) -> None:
        # Surface every log line — exporter timing breakdowns,
        # material/mesh warnings, the renderer's per-bucket timings — into
        # Blender's Info panel and stdout. `self.report` mirrors INFO
        # messages to stdout for free.
        def _to_info(msg: str) -> None:
            s = msg.rstrip()
            if s:
                self.report({"INFO"}, s)

        with _log.redirect(_to_info):
            self.update_stats("vibrt", "Exporting scene...")
            self.report({"INFO"}, f"vibrt: rendering {width}x{height} in-process")
            t_export = time.perf_counter()
            json_str, mesh_blobs, texture_arrays = (
                exporter.export_scene_to_memory(depsgraph)
            )
            n_blobs = len(mesh_blobs)
            n_arrays = len(texture_arrays)
            mesh_mb = sum(b.nbytes for b in mesh_blobs) / 1024 / 1024
            tex_mb = sum(a.nbytes for a in texture_arrays) / 1024 / 1024
            self.report(
                {"INFO"},
                f"vibrt: export {time.perf_counter() - t_export:.2f}s "
                f"({n_blobs} mesh blobs / {mesh_mb:.1f} MB, "
                f"{n_arrays} textures / {tex_mb:.1f} MB)",
            )

            self.update_stats("vibrt", "Rendering...")
            t_render = time.perf_counter()
            pixels = runner.run_render_inproc(
                json_str,
                mesh_blobs,
                self.report,
                self.test_break,
                denoise=denoise,
                texture_arrays=texture_arrays,
            )
            self.report(
                {"INFO"},
                f"vibrt: render returned {pixels.shape} in "
                f"{time.perf_counter() - t_render:.2f}s",
            )

            # `pixels` is float32 (h, w, 4), bottom-left origin.
            self.update_stats("vibrt", "Loading result...")
            t_blit = time.perf_counter()
            _push_pixels_into_render_result(self, pixels, width, height)
            self.report(
                {"INFO"},
                f"vibrt: blit to Combined pass {time.perf_counter() - t_blit:.2f}s",
            )


def _push_pixels_into_render_result(engine, pixels, width: int, height: int):
    """Copy a float32 (h, w, 4) ndarray into the Combined pass."""
    arr = pixels
    if arr.shape[0] != height or arr.shape[1] != width or arr.shape[2] != 4:
        engine.report(
            {"ERROR"},
            f"native render returned shape {arr.shape}, expected ({height}, {width}, 4)",
        )
        return

    result = engine.begin_result(0, 0, width, height)
    render_layer = result.layers[0]
    combined = None
    try:
        combined = render_layer.passes.find_by_name("Combined", "")
    except TypeError:
        combined = render_layer.passes.find_by_name("Combined")
    if combined is None:
        for p in render_layer.passes:
            if p.name == "Combined":
                combined = p
                break
    if combined is None:
        engine.end_result(result)
        engine.report({"ERROR"}, "Combined pass not found in render result")
        return
    flat = arr.ravel()
    if hasattr(combined.rect, "foreach_set"):
        combined.rect.foreach_set(flat)
    else:
        combined.rect = [tuple(flat[i:i + 4]) for i in range(0, len(flat), 4)]
    engine.end_result(result)


def register():
    bpy.utils.register_class(VibrtRenderEngine)
    _add_ui_compatibility()


def unregister():
    bpy.utils.unregister_class(VibrtRenderEngine)


def _add_ui_compatibility():
    """Mark standard Blender render panels as visible for our engine."""
    from bl_ui import (
        properties_render,
        properties_output,
        properties_view_layer,
        properties_world,
        properties_material,
        properties_data_camera,
        properties_data_light,
    )

    # Panels whose class name contains these tokens are engine-specific sampling
    # controls we provide ourselves (see properties.VIBRT_PT_sampling). Letting
    # e.g. EEVEE's sampling panel appear for VIBRT caused the exported spp to
    # silently disagree with the visible "Samples" field.
    exclude_tokens = ("sampling",)

    for module in (
        properties_render,
        properties_output,
        properties_view_layer,
        properties_world,
        properties_material,
        properties_data_camera,
        properties_data_light,
    ):
        for clsname in dir(module):
            cls = getattr(module, clsname)
            if not hasattr(cls, "COMPAT_ENGINES"):
                continue
            if any(tok in clsname.lower() for tok in exclude_tokens):
                continue
            compat = getattr(cls, "COMPAT_ENGINES")
            if isinstance(compat, set):
                compat.add(VibrtRenderEngine.bl_idname)
