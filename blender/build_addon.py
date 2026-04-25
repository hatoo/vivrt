"""Build vibrt_blender.zip for Blender installation.

Run: python blender/build_addon.py [--with-native] [--skip-build]

By default the zip contains the addon's Python sources only — Blender will
fall back to the subprocess `vibrt.exe` path. Pass `--with-native` to also
build and bundle `vibrt_native.pyd`, the PyO3 extension that the addon
imports for the in-process render path (skips the scene.bin disk roundtrip,
~20% faster on junk_shop-class scenes).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
SRC = HERE / "vibrt_blender"
OUT = HERE / "vibrt_blender.zip"

# Cargo's cdylib output on Windows / Linux / macOS for the `vibrt_native`
# library (set via `[lib] name = "vibrt_native"` in vibrt/Cargo.toml).
if sys.platform == "win32":
    NATIVE_BUILT = REPO / "target" / "release" / "vibrt_native.dll"
    NATIVE_BUNDLED = SRC / "vibrt_native.pyd"
elif sys.platform == "darwin":
    NATIVE_BUILT = REPO / "target" / "release" / "libvibrt_native.dylib"
    NATIVE_BUNDLED = SRC / "vibrt_native.so"
else:
    NATIVE_BUILT = REPO / "target" / "release" / "libvibrt_native.so"
    NATIVE_BUNDLED = SRC / "vibrt_native.so"


def _cargo_build_native() -> None:
    """Invoke `cargo build --release --features python` against the vibrt
    crate. Honours `PYO3_PYTHON` from the environment (or PYO3_NO_PYTHON for
    bring-your-own-Python builds with abi3); errors out if neither is set
    and cargo can't locate a Python interpreter on its own.
    """
    env = os.environ.copy()
    if "PYO3_PYTHON" not in env and "PYO3_NO_PYTHON" not in env:
        # PyO3 needs *some* Python install for abi3 builds (to find
        # python3.lib on Windows). Hint at the launcher-installed CPython.
        py_launcher = shutil.which("py") or shutil.which("python")
        if py_launcher:
            env["PYO3_PYTHON"] = py_launcher
    cmd = [
        "cargo", "build", "--release",
        "--features", "python",
        "--manifest-path", str(REPO / "vibrt" / "Cargo.toml"),
    ]
    print(f"[build_addon] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _stage_native(skip_build: bool) -> None:
    if not skip_build:
        _cargo_build_native()
    if not NATIVE_BUILT.exists():
        sys.exit(
            f"[build_addon] expected cargo output not found: {NATIVE_BUILT}\n"
            "(rerun without --skip-build, or check that cargo build --features python succeeded)"
        )
    NATIVE_BUNDLED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NATIVE_BUILT, NATIVE_BUNDLED)
    print(f"[build_addon] staged {NATIVE_BUNDLED.relative_to(REPO)} ({NATIVE_BUNDLED.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-native",
        action="store_true",
        help="build vibrt_native.pyd via cargo and bundle it into the zip",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="skip the cargo build step (assumes target/release/vibrt_native.* is already present)",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help=(
            "build and stage vibrt_native.pyd into the addon source dir but "
            "skip writing the zip — used by `make dev-install` so the dev "
            "junctioned addon picks up the freshly built extension"
        ),
    )
    args = parser.parse_args()

    if args.with_native or args.stage_only:
        _stage_native(args.skip_build)

    if args.stage_only:
        return

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SRC.rglob("*"):
            if f.is_dir() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            z.write(f, f.relative_to(HERE))
    print(f"Wrote {OUT.relative_to(HERE.parent)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
