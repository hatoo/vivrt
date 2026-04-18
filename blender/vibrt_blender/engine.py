"""Blender `RenderEngine` subclass for vibrt (final-frame F12 only)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import bpy

from . import exporter
from . import runner


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

        exe = runner.find_executable()
        if exe is None:
            self.report(
                {"ERROR"},
                "vibrt executable not found (set in addon preferences or $VIBRT_EXECUTABLE or PATH)",
            )
            return

        work = Path(bpy.app.tempdir) / f"vibrt_{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True, exist_ok=True)
        json_path = work / "scene.json"
        bin_path = work / "scene.bin"
        # Use `.raw` so we can bypass Blender's EXR loader, which emits
        # multi-layer-channel warnings on single-layer files and can return
        # empty pixel buffers.
        exr_path = work / "output.raw"

        self.update_stats("vibrt", "Exporting scene...")
        try:
            exporter.export_scene(depsgraph, json_path, bin_path)
        except Exception as e:
            self.report({"ERROR"}, f"Export failed: {e}")
            raise

        self.update_stats("vibrt", "Rendering...")
        code = runner.run_render(
            exe,
            json_path,
            exr_path,
            self.report,
            self.test_break,
        )
        if code != 0:
            self.report({"ERROR"}, f"vibrt exited with code {code}")
            return
        if not exr_path.exists():
            self.report({"ERROR"}, f"No output produced at {exr_path}")
            return

        self.update_stats("vibrt", "Loading result...")
        _load_exr_into_render_result(self, exr_path, width, height)

    # Fallback: this addon does not support viewport IPR (yet).


def _load_exr_into_render_result(engine, raw_path: Path, width: int, height: int):
    """Read our `.raw` float RGBA file and copy pixels into the Combined pass."""
    import struct

    data = raw_path.read_bytes()
    if len(data) < 16 or data[:4] != b"VBLT":
        engine.report({"ERROR"}, f"invalid raw file header in {raw_path}")
        return
    file_w, file_h, ch = struct.unpack("<III", data[4:16])
    if ch != 4 or file_w != width or file_h != height:
        engine.report(
            {"ERROR"},
            f"raw file dims mismatch (file {file_w}x{file_h}x{ch}, expected {width}x{height}x4)",
        )
        return
    expected = width * height * 4
    pixels = struct.unpack(f"<{expected}f", data[16:16 + expected * 4])

    result = engine.begin_result(0, 0, width, height)
    render_layer = result.layers[0]
    # Blender 5.x requires a `view` argument; fall back to scanning for older API.
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
    if hasattr(combined.rect, "foreach_set"):
        combined.rect.foreach_set(pixels)
    else:
        combined.rect = [
            tuple(pixels[i : i + 4]) for i in range(0, len(pixels), 4)
        ]
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
