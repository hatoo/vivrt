"""Per-blob writer for the in-process FFI export path.

There is no longer a "bin" — the previous version concatenated every
blob into one giant `bytearray` and returned `{"offset", "len"}`
references. That forced Python to grow a single buffer through log-N
reallocs (with all the memoryview-pinning gymnastics that implies) and
forced Rust to resolve each blob against the same byte slice with
`bytemuck::cast_slice`. Both are unnecessary: each blob is independently
usable.

`write_f32` / `write_u32` append a contiguous numpy array to
`mesh_blobs` and return `{"array_index": i}`. The Rust loader picks the
slice by index. `write_texture_pixels` does the same for f32 RGBA
texture data, but parks into a separate list (`texture_arrays`) because
Rust receives them as a typed `Vec<PyBuffer<f32>>` for the GPU upload
path.

This module has no bpy imports, so tests can import and exercise it
under plain CPython (see `test_bin_writer.py`).
"""

from __future__ import annotations

import numpy as np


class BinWriter:
    __slots__ = ("_blobs", "_texture_arrays")

    def __init__(self):
        self._blobs: list = []
        self._texture_arrays: list = []

    @property
    def mesh_blobs(self) -> list:
        """Mesh / index / vertex-color / colour-graph LUT numpy arrays.
        Index matches the `array_index` returned by `write_f32` / `write_u32`."""
        return self._blobs

    @property
    def texture_arrays(self) -> list:
        """f32 RGBA numpy arrays accumulated by `write_texture_pixels`."""
        return self._texture_arrays

    def write_f32(self, xs) -> int:
        """Append a contiguous float32 blob and return its array_index.

        Mesh / index / vertex-color fields on `MeshDesc` are plain `u32`
        on the Rust side, so the index is returned bare — JSON ends up
        with `"vertices": 5` not `"vertices": {"array_index": 5}`."""
        arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.float32)
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        i = len(self._blobs)
        self._blobs.append(arr)
        return i

    def write_u32(self, xs) -> int:
        """Append a contiguous uint32 blob and return its array_index."""
        arr = xs if isinstance(xs, np.ndarray) else np.asarray(xs, dtype=np.uint32)
        if arr.dtype != np.uint32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.uint32)
        i = len(self._blobs)
        self._blobs.append(arr)
        return i

    def write_texture_pixels(self, arr) -> dict:
        """Park a contiguous float32 RGBA buffer into the per-texture array
        list. The texture data crosses PyO3 as a typed `PyBuffer<f32>` (no
        u8 view trickery) so it's bagged separately from mesh blobs.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float32)
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        i = len(self._texture_arrays)
        self._texture_arrays.append(arr)
        return {"array_index": i}
