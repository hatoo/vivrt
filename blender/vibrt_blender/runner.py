"""Discover and invoke the vibrt-blender binary."""

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
    env = os.environ.get("VIBRT_BLENDER_EXECUTABLE")
    if env and Path(env).exists():
        return env
    which = shutil.which("vibrt-blender")
    if which:
        return which
    return None


def run_render(
    exe: str,
    scene_json: Path,
    output_path: Path,
    report,
    is_break,
) -> int:
    """Run vibrt-blender; return exit code. Pipes stderr lines to `report`."""
    cmd = [exe, str(scene_json), "--output", str(output_path)]
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
