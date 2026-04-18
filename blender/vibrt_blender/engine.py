"""Blender `RenderEngine` subclass for vibrt-blender (final-frame F12 only)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import bpy

from . import exporter
from . import runner


class VibrtRenderEngine(bpy.types.RenderEngine):
    bl_idname = "VIBRT_BLENDER"
    bl_label = "vibrt-blender"
    bl_use_preview = False
    bl_use_shading_nodes_custom = False

    def render(self, depsgraph: bpy.types.Depsgraph):
        scene = depsgraph.scene_eval
        rd = scene.render
        width = int(rd.resolution_x * rd.resolution_percentage / 100.0)
        height = int(rd.resolution_y * rd.resolution_percentage / 100.0)

        exe = runner.find_executable()
        if exe is None:
            self.report(
                {"ERROR"},
                "vibrt-blender executable not found (set in addon preferences or $VIBRT_BLENDER_EXECUTABLE or PATH)",
            )
            return

        work = Path(bpy.app.tempdir) / f"vibrt_blender_{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True, exist_ok=True)
        json_path = work / "scene.json"
        bin_path = work / "scene.bin"
        exr_path = work / "output.exr"

        self.update_stats("vibrt-blender", "Exporting scene...")
        try:
            exporter.export_scene(depsgraph, json_path, bin_path)
        except Exception as e:
            self.report({"ERROR"}, f"Export failed: {e}")
            raise

        self.update_stats("vibrt-blender", "Rendering...")
        code = runner.run_render(
            exe,
            json_path,
            exr_path,
            self.report,
            self.test_break,
        )
        if code != 0:
            self.report({"ERROR"}, f"vibrt-blender exited with code {code}")
            return
        if not exr_path.exists():
            self.report({"ERROR"}, f"No output produced at {exr_path}")
            return

        self.update_stats("vibrt-blender", "Loading result...")
        _load_exr_into_render_result(self, exr_path, width, height)

    # Fallback: this addon does not support viewport IPR (yet).


def _load_exr_into_render_result(engine, exr_path: Path, width: int, height: int):
    # Load via Blender's own EXR loader for fidelity.
    result = engine.begin_result(0, 0, width, height)
    layer = result.layers[0].passes.find_by_name("Combined")
    if layer is None:
        engine.end_result(result)
        engine.report({"ERROR"}, "Combined pass not found in render result")
        return
    # Use Blender's image load to get pixel data.
    img = bpy.data.images.load(str(exr_path), check_existing=False)
    try:
        pixels = list(img.pixels[:])
        # Blender layer pass wants RGBA flat
        layer.rect = [
            (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
            for i in range(0, len(pixels), 4)
        ]
    finally:
        bpy.data.images.remove(img)
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
            compat = getattr(cls, "COMPAT_ENGINES")
            if isinstance(compat, set):
                compat.add(VibrtRenderEngine.bl_idname)
