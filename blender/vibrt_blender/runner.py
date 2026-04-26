"""Discover and invoke the vibrt renderer.

Two paths exist:

- `run_render_inproc` calls the bundled `vibrt_native` PyO3 extension directly,
  handing it the in-memory `(scene.json, scene.bin)` buffers produced by
  `exporter.export_scene_to_memory`. No disk roundtrip, no subprocess.
- `run_render` is the legacy fallback: write the buffers to a temp dir, spawn
  `vibrt.exe`, stream its stdout into Blender's Info panel. Used when the
  native extension isn't bundled (e.g. fresh checkout that hasn't run
  `cargo build --features python` yet) or when in-process rendering errors
  out and the engine wants a second chance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import bpy


def find_executable() -> str | None:
    prefs = bpy.context.preferences.addons[__package__].preferences
    if prefs.vibrt_executable and Path(prefs.vibrt_executable).exists():
        return prefs.vibrt_executable
    env = os.environ.get("VIBRT_EXECUTABLE")
    if env and Path(env).exists():
        return env
    which = shutil.which("vibrt")
    if which:
        return which
    return None


def find_native_module():
    """Import `vibrt_native` if it's bundled with the addon. Returns the
    module on success, or None on `ImportError` (extension not built yet,
    binary missing, ABI mismatch). Callers fall back to `run_render` (the
    subprocess path) when this is None.
    """
    try:
        from . import vibrt_native  # type: ignore  # ships as a .pyd next to __init__.py
        return vibrt_native
    except ImportError:
        return None


def run_render_inproc(
    scene_json: str,
    scene_bin: bytes,
    report,
    is_break,
    denoise: bool = False,
    texture_arrays=None,
):
    """Render `(scene.json, scene.bin)` in-process via `vibrt_native`.

    Returns a `(height, width, 4)` float32 numpy ndarray (linear RGBA, the
    same buffer the GPU writes — bottom-left origin matching Blender's
    `Image.pixels`). Raises `ImportError` if the extension isn't available;
    raises `RuntimeError` for vibrt errors; raises `KeyboardInterrupt` if
    the user aborted via Esc.

    `texture_arrays`, when supplied, is the per-texture pixel-array list
    produced by `exporter.export_scene_to_memory`. The Rust loader resolves
    each `TextureDesc.array_index` against it, so texture pixels can be
    handed across PyO3 directly instead of being concatenated into the bin.
    """
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

    opts = {"denoise": bool(denoise)}
    return native.render(
        scene_json, scene_bin, opts, log_cb, cancel_cb,
        texture_arrays=texture_arrays,
    )


def run_render(
    exe: str,
    scene_json: Path,
    output_path: Path,
    report,
    is_break,
    denoise: bool = False,
) -> int:
    """Run vibrt; return exit code. Pipes stderr lines to `report`."""
    cmd = [exe, str(scene_json), "--output", str(output_path)]
    if denoise:
        cmd.append("--denoise")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
        universal_newlines=True,
    )
    stderr_tail: list[str] = []
    try:
        while True:
            if proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    report({"INFO"}, line.rstrip())
                    continue
            if proc.poll() is not None:
                break
            if is_break():
                proc.terminate()
                break
    finally:
        if proc.stderr is not None:
            err = proc.stderr.read()
            if err:
                stderr_tail = err.splitlines()[-20:]
                for ln in stderr_tail:
                    report({"ERROR"}, ln)
        proc.wait()
    return proc.returncode if proc.returncode is not None else 1
