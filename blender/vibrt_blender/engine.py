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
    # Run Blender's compositor over our output. We populate Combined plus
    # Mist (from primary-ray hit distance + world.mist_settings), Z (same
    # depth), and Noisy Image (= Combined, so any Mix(Image, Noisy Image)
    # idiom is a no-op). lone_monk's tree (Mist factor mix → Glare → Lens
    # Distortion) needs these to actually produce its hazy/airy look; with
    # them present, scenes whose compositor is empty / trivial render the
    # same way they did when this flag was False.
    bl_use_postprocess = True

    def update(self, data=None, depsgraph=None):
        # `update()` is Blender's "prepare-to-render" hook. We use it to
        # pre-bake any procedural Sky Texture (Nishita / Hosek-Wilkie /
        # Preetham) the user has on their World — that involves running a
        # nested `bpy.ops.render.render` against a temp scene, which
        # Blender silently turns into an all-zero render if invoked from
        # `render()` but works fine here. Pixels land in the
        # `_SKY_BAKE_CACHE` keyed by the Sky Texture's controls; the
        # exporter (which runs inside render()) reads from the cache.
        scene = bpy.context.scene if data is None else (
            getattr(data, "scene", None) or bpy.context.scene
        )
        try:
            exporter.clear_sky_bake_cache()
            exporter.prebake_sky_envmaps_for_world(scene.world)
        except Exception as ex:
            # Don't take down the whole render if the bake fails — emit a
            # visible warning and let _export_world fall back to constant.
            self.report(
                {"WARNING"},
                f"vibrt: sky pre-bake in update() failed: {ex}",
            )

    def update_render_passes(self, scene=None, renderlayer=None):
        """Tell Blender which passes we'll write so they exist on the
        RenderResult and the compositor's `Render Layers` node exposes them.

        Combined is registered unconditionally; Mist / Z follow the
        view-layer's `use_pass_*` flags (mirroring Cycles); Noisy Image is
        registered when denoising is on so any `Mix(Image, Noisy Image)`
        compositor idiom finds a real socket. Falling through silently
        when an attribute isn't present keeps this safe across Blender
        versions where the API spelling drifts.
        """
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", "COLOR")

        rl = renderlayer
        try:
            if getattr(rl, "use_pass_mist", False):
                self.register_pass(scene, renderlayer, "Mist", 1, "Z", "VALUE")
            if getattr(rl, "use_pass_z", False):
                self.register_pass(scene, renderlayer, "Z", 1, "Z", "VALUE")
        except Exception:
            pass

        # Always expose Noisy Image (= Combined) so Cycles-authored
        # `Mix(Image, Noisy Image)` denoise-blend nodes don't end up
        # mixing Combined with all-zero. Cheap: it's the same buffer.
        self.register_pass(
            scene, renderlayer, "Noisy Image", 4, "RGBA", "COLOR"
        )

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

        # If the scene's compositor wants Render-Layers passes we don't
        # emit (AO / Shadow / Emit / Diffuse Color / ...), running it
        # would mix Combined with all-zero buffers and ruin the output
        # (classroom.blend's tree is a 100-node chain that does exactly
        # this). Disable compositing for the render in that case and tell
        # the user. We toggle on the original (non-eval) scene because
        # `scene_eval.render` is read-only.
        original_scene = depsgraph.scene
        prev_use_compositing = original_scene.render.use_compositing
        unsupported = _unsupported_compositor_passes(original_scene)
        compositor_disabled_here = False
        if unsupported and prev_use_compositing:
            self.report(
                {"WARNING"},
                f"vibrt: compositor uses unsupported passes "
                f"{sorted(unsupported)} — disabling compositor for this "
                f"render. Re-enable manually if you've authored a tree "
                f"that doesn't depend on those.",
            )
            original_scene.render.use_compositing = False
            compositor_disabled_here = True

        try:
            self._render_in_process(depsgraph, width, height, denoise)
        except KeyboardInterrupt:
            self.report({"WARNING"}, "Render cancelled")
        finally:
            if compositor_disabled_here:
                original_scene.render.use_compositing = prev_use_compositing

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
            pixels, depth = runner.run_render_inproc(
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

            # `pixels` is float32 (h, w, 4), `depth` is float32 (h, w).
            self.update_stats("vibrt", "Loading result...")
            t_blit = time.perf_counter()
            _push_pixels_into_render_result(
                self,
                pixels,
                depth,
                width,
                height,
                depsgraph.scene_eval,
            )
            self.report(
                {"INFO"},
                f"vibrt: blit to passes {time.perf_counter() - t_blit:.2f}s",
            )


_SUPPORTED_RENDER_LAYER_OUTPUTS = frozenset({
    # Sockets we populate (and aliases Blender exposes for the same pass).
    "Image", "Combined",
    "Noisy Image",
    "Mist",
    "Z", "Depth",
    # Static / always-zero sockets that are safe even when un-driven.
    "Alpha",
})


def _unsupported_compositor_passes(scene) -> set:
    """Return the set of `Render Layers` output sockets the compositor reads
    that we don't emit. Empty result means the compositor is safe to run.

    classroom.blend's compositor reaches into AO / Shadow / Emit / Diffuse
    Color and similar passes; running its tree on top of our Combined
    (with those passes all zero) collapses the image to black. Returning
    a non-empty set lets the caller decide to skip the compositor and
    surface a clear warning.
    """
    if not getattr(scene, "render", None):
        return set()
    if not getattr(scene.render, "use_compositing", False):
        return set()
    tree = getattr(scene, "compositing_node_group", None) or getattr(
        scene, "node_tree", None
    )
    if tree is None or not hasattr(tree, "nodes"):
        return set()
    unsupported: set = set()
    for n in tree.nodes:
        if n.bl_idname != "CompositorNodeRLayers":
            continue
        for sock in n.outputs:
            if not sock.is_linked:
                continue
            if sock.name in _SUPPORTED_RENDER_LAYER_OUTPUTS:
                continue
            unsupported.add(sock.name)
    return unsupported


def _push_pixels_into_render_result(engine, pixels, depth, width, height, scene):
    """Copy `pixels` (h, w, 4) and `depth` (h, w) into the available passes.

    Populates Combined unconditionally; Mist / Z / Noisy Image are written
    only when the corresponding view-layer flag (or denoising-data
    request) made Blender register the pass. Mist comes from
    `world.mist_settings` applied to the per-pixel hit distance; Noisy
    Image is the same buffer as Combined so the lone_monk-style
    `Mix(Image, Noisy Image)` compositor idiom is a no-op.
    """
    if (pixels.shape[0] != height or pixels.shape[1] != width
            or pixels.shape[2] != 4):
        engine.report(
            {"ERROR"},
            f"native render returned pixels shape {pixels.shape}, "
            f"expected ({height}, {width}, 4)",
        )
        return
    if depth.shape[0] != height or depth.shape[1] != width:
        engine.report(
            {"ERROR"},
            f"native render returned depth shape {depth.shape}, "
            f"expected ({height}, {width})",
        )
        return

    result = engine.begin_result(0, 0, width, height)
    try:
        render_layer = result.layers[0]
        passes = {p.name: p for p in render_layer.passes}

        flat_rgba = pixels.ravel()

        combined = passes.get("Combined")
        if combined is None:
            engine.report({"ERROR"}, "Combined pass not found in render result")
            return
        _set_pass(combined, flat_rgba)

        # Cycles' compositor idiom for denoising mixes Image↔Noisy Image;
        # writing the same buffer to both makes the mix produce Combined
        # regardless of factor, which is what we want here.
        noisy = passes.get("Noisy Image")
        if noisy is not None:
            _set_pass(noisy, flat_rgba)

        z_pass = passes.get("Z")
        if z_pass is not None:
            _set_pass(z_pass, depth.ravel())

        mist_pass = passes.get("Mist")
        if mist_pass is not None:
            mist = _compute_mist(depth, scene)
            _set_pass(mist_pass, mist.ravel())
    finally:
        engine.end_result(result)


def _set_pass(rpass, flat) -> None:
    if hasattr(rpass.rect, "foreach_set"):
        rpass.rect.foreach_set(flat)
    else:
        ch = rpass.channels
        rpass.rect = [tuple(flat[i:i + ch]) for i in range(0, len(flat), ch)]


def _compute_mist(depth, scene):
    """Cycles-style Mist factor: t = clamp((dist - start)/depth, 0, 1) with
    LINEAR / QUADRATIC / INVERSE_QUADRATIC falloff. `world.mist_settings`
    drives start/depth/falloff; vibrt's per-pixel hit distance plays the
    role of Cycles' camera-space ray length.
    """
    import numpy as np

    world = scene.world
    if world is None or not hasattr(world, "mist_settings"):
        return np.zeros_like(depth)
    ms = world.mist_settings
    start = float(ms.start)
    drange = max(float(ms.depth), 1e-6)
    falloff = getattr(ms, "falloff", "LINEAR")

    t = np.clip((depth - start) / drange, 0.0, 1.0).astype(np.float32)
    if falloff == "QUADRATIC":
        return t * t
    if falloff == "INVERSE_QUADRATIC":
        u = 1.0 - t
        return (1.0 - u * u).astype(np.float32)
    return t


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
