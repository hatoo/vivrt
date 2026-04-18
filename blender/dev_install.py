"""Symlink vibrt_blender into Blender's user-addons dir for fast iteration.

Run: python blender/dev_install.py [--version X.Y]

After running, edit files in blender/vibrt_blender/ and hit F3 ->
"Reload Scripts" in Blender (or restart). No zip + addon-dialog dance.

On Windows a directory junction (`mklink /J`) is used — no admin required.
On Linux/macOS a symlink is used.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _blender_config_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Blender Foundation" / "Blender"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Blender"
    return Path.home() / ".config" / "blender"


def _safe_remove(p: Path) -> None:
    if not os.path.lexists(p):
        return
    if p.is_symlink():
        p.unlink()
        return
    try:
        os.rmdir(p)  # empty dir or Windows junction (does not touch target)
    except OSError:
        shutil.rmtree(p)


def _link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
        )
    else:
        os.symlink(target, link, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Blender version dir name (e.g. 5.1). Defaults to the newest one found.",
    )
    args = parser.parse_args()

    source = Path(__file__).resolve().parent / "vibrt_blender"
    if not source.is_dir():
        sys.exit(f"source dir not found: {source}")

    root = _blender_config_root()
    if not root.is_dir():
        sys.exit(f"Blender config dir not found: {root}")

    if args.version:
        versions = [root / args.version]
    else:
        versions = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name[:1].isdigit()),
            key=lambda p: tuple(int(x) for x in p.name.split(".") if x.isdigit()),
        )
        versions = versions[-1:]  # newest

    if not versions or not versions[0].is_dir():
        sys.exit(f"no matching Blender version under {root}")

    for v in versions:
        link = v / "scripts" / "addons" / "vibrt_blender"
        _safe_remove(link)
        _link(link, source)
        print(f"linked: {link} -> {source}")


if __name__ == "__main__":
    main()
