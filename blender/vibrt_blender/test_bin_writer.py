"""Unit tests for `bin_writer.BinWriter`. Runs under plain CPython.

Run with:
    py blender/vibrt_blender/test_bin_writer.py

The sibling `__init__.py` imports `bpy`, so we load `bin_writer.py`
directly via importlib instead of going through the package.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "vibrt_bin_writer", _HERE / "bin_writer.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["vibrt_bin_writer"] = _mod
_spec.loader.exec_module(_mod)
BinWriter = _mod.BinWriter


class BinWriterTests(unittest.TestCase):
    def test_empty_writer(self):
        w = BinWriter()
        self.assertEqual(w.mesh_blobs, [])
        self.assertEqual(w.texture_arrays, [])

    def test_write_f32_returns_array_index(self):
        # Mesh writer returns a bare int — `MeshDesc.vertices` etc. are
        # `u32` fields on the Rust side, no enclosing dict.
        w = BinWriter()
        self.assertEqual(w.write_f32([1.0, 2.0, 3.0]), 0)

    def test_write_f32_indexes_increment(self):
        w = BinWriter()
        self.assertEqual(w.write_f32([0.0]), 0)
        self.assertEqual(w.write_f32([1.0]), 1)
        self.assertEqual(w.write_f32([2.0]), 2)
        self.assertEqual(len(w.mesh_blobs), 3)

    def test_write_u32_indexes_increment_with_f32(self):
        # f32 and u32 share the mesh_blobs index space.
        w = BinWriter()
        self.assertEqual(w.write_f32([0.0]), 0)
        self.assertEqual(w.write_u32([1, 2, 3]), 1)
        self.assertEqual(w.write_f32([2.0]), 2)

    def test_write_f32_preserves_data(self):
        w = BinWriter()
        w.write_f32([0.25, 0.5, 0.75])
        np.testing.assert_array_equal(w.mesh_blobs[0],
                                      np.array([0.25, 0.5, 0.75], dtype=np.float32))

    def test_write_u32_preserves_data(self):
        w = BinWriter()
        w.write_u32([10, 20, 30])
        np.testing.assert_array_equal(w.mesh_blobs[0],
                                      np.array([10, 20, 30], dtype=np.uint32))

    def test_write_f32_dtype_coerced(self):
        # Integer input gets coerced to float32 (matches old BinWriter behaviour).
        w = BinWriter()
        w.write_f32(np.array([1, 2, 3], dtype=np.int32))
        self.assertEqual(w.mesh_blobs[0].dtype, np.float32)

    def test_write_u32_dtype_coerced(self):
        w = BinWriter()
        w.write_u32(np.array([1, 2, 3], dtype=np.int64))
        self.assertEqual(w.mesh_blobs[0].dtype, np.uint32)

    def test_write_f32_existing_contig_no_copy(self):
        # When the input is already contiguous f32, the writer should keep
        # the same numpy array (no extra copy) — the array becomes the blob.
        w = BinWriter()
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w.write_f32(a)
        self.assertIs(w.mesh_blobs[0], a)

    def test_write_f32_makes_contiguous_when_strided(self):
        w = BinWriter()
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        # Take a strided view (column 1) and pass it.
        col = a[:, 1]
        self.assertFalse(col.flags.c_contiguous)
        w.write_f32(col)
        self.assertTrue(w.mesh_blobs[0].flags.c_contiguous)
        np.testing.assert_array_equal(w.mesh_blobs[0],
                                      np.array([2.0, 5.0], dtype=np.float32))

    def test_write_texture_pixels_separate_index_space(self):
        # texture_arrays is independent of mesh_blobs.
        w = BinWriter()
        self.assertEqual(w.write_f32([0.0]), 0)
        # write_texture_pixels returns a dict because TextureDesc spreads it.
        self.assertEqual(w.write_texture_pixels(np.zeros(16, dtype=np.float32)),
                         {"array_index": 0})  # texture index, not mesh
        self.assertEqual(w.write_u32([1, 2]), 1)
        self.assertEqual(w.write_texture_pixels(np.ones(16, dtype=np.float32)),
                         {"array_index": 1})

    def test_write_texture_pixels_contig_f32(self):
        w = BinWriter()
        w.write_texture_pixels([0.1, 0.2, 0.3, 1.0])
        arr = w.texture_arrays[0]
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(arr.flags.c_contiguous)

    def test_mesh_blob_view_as_uint8_works(self):
        # The runner's PyO3 path does `.view(np.uint8)` on each blob.
        # Verify that round-trip preserves bytes.
        w = BinWriter()
        w.write_f32([1.0, 2.0, 3.0, 4.0])
        w.write_u32([100, 200, 300])
        for blob in w.mesh_blobs:
            view = blob.view(np.uint8)
            self.assertEqual(view.nbytes, blob.nbytes)
            self.assertTrue(view.flags.c_contiguous)

    def test_distinct_blobs_dont_alias(self):
        # Each blob should be its own array — writing different data twice
        # must not leave a single shared buffer (this caught the
        # _REUSABLE_PIXELS aliasing regression that gave junk_shop a
        # purple cast on every texture).
        w = BinWriter()
        w.write_f32([0.0, 0.0, 0.0])
        w.write_f32([1.0, 2.0, 3.0])
        a, b = w.mesh_blobs
        self.assertIsNot(a, b)
        np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(b, np.array([1.0, 2.0, 3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
