"""Pluggable logger for the addon's Python side.

By default lines go to stdout via `print` — that's what headless users
running through `scripts/render_blend.py` (or invoking
`export_scene_to_memory` from a script) want to see in their terminal.

Inside Blender, `engine._render_in_process` swaps the sink to
`self.report({"INFO"}, msg)` so the same messages also surface in
Blender's Info panel and the bottom status bar. Both fan out to stdout:
`self.report` mirrors INFO messages there for free.

Use:
    from . import _log
    _log.log(f"...")  # always works
    with _log.redirect(my_cb):
        ...  # `_log.log` calls go through `my_cb` instead of print
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

_CALLBACK: Optional[Callable[[str], None]] = None


def log(msg: str) -> None:
    """Emit a single log line. Goes through whatever callback the
    enclosing `redirect()` context set, or `print` when none is active."""
    cb = _CALLBACK
    if cb is None:
        print(msg)
    else:
        cb(msg)


@contextmanager
def redirect(cb: Callable[[str], None]):
    """Route `log` calls through `cb` for the duration of the block.
    Nesting is supported: the previous callback is restored on exit."""
    global _CALLBACK
    prev = _CALLBACK
    _CALLBACK = cb
    try:
        yield
    finally:
        _CALLBACK = prev
