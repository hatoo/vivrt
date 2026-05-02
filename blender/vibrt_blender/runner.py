"""Discover and invoke `vibrt_native` — the in-process PyO3 extension.

The addon renders exclusively through the bundled `vibrt_native.pyd`. The
standalone `vibrt.exe` binary still exists for CLI tooling but is not
invoked from the addon itself.
"""

from __future__ import annotations


def find_native_module():
    """Import `vibrt_native` if it's bundled with the addon. Returns the
    module on success, or None on `ImportError` (extension not built yet,
    binary missing, ABI mismatch). Callers report a clear error and stop —
    there is no subprocess fallback.
    """
    try:
        from . import vibrt_native  # type: ignore  # ships as a .pyd next to __init__.py
        return vibrt_native
    except ImportError:
        return None


def run_render_inproc(
    scene_json: str,
    mesh_blobs,
    report,
    is_break,
    denoise: bool = False,
    texture_arrays=None,
):
    """Render `(scene.json, mesh_blobs, texture_arrays)` in-process via
    `vibrt_native`.

    Returns a `(pixels, depth)` tuple where `pixels` is a
    `(height, width, 4)` float32 ndarray (linear RGBA, top-left origin)
    and `depth` is a `(height, width)` float32 ndarray of per-pixel
    primary-ray hit distance from the camera, in metres (misses store
    the camera's `clip_end`). The addon turns `depth` into Mist / Z
    passes for the compositor. Raises `ImportError` if the extension
    isn't available; raises `RuntimeError` for vibrt errors; raises
    `KeyboardInterrupt` if the user aborted via Esc.

    `mesh_blobs` is the per-blob byte-buffer list (mesh / index / vertex-
    color / colour-graph LUT data). Each entry is a numpy array; the Rust
    side picks them by `MeshDesc.array_index` references in the JSON.
    `texture_arrays` is the parallel list for `TextureDesc.array_index`.
    """
    import numpy as np
    native = find_native_module()
    if native is None:
        raise ImportError("vibrt_native not available (build with --features python)")

    def log_cb(msg: str) -> None:
        # Filter out empty lines so the Info panel stays tidy. Strip CR
        # which Windows can introduce when stdout-style messages arrive.
        s = msg.rstrip()
        if s:
            report({"INFO"}, s)

    def cancel_cb() -> bool:
        # `is_break()` is the engine's `test_break` — flips when the user
        # hits Esc. Wrap it because PyO3 wants a plain truthiness check.
        try:
            return bool(is_break())
        except Exception:
            return False

    # Reinterpret each typed (f32 / u32) blob as a uint8 view so PyO3 can
    # take it as `PyBuffer<u8>`. `.view(np.uint8)` is a no-copy reinterpret.
    blobs_u8 = [b if b.dtype == np.uint8 else b.view(np.uint8) for b in mesh_blobs]

    opts = {"denoise": bool(denoise)}
    return native.render(
        scene_json, blobs_u8, opts, log_cb, cancel_cb,
        texture_arrays=texture_arrays,
    )
