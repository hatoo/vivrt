"""Pure-Python tests for ies.py — runs under plain CPython (no Blender).
Picked up automatically by `make test` since the file matches `test_*.py`.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ies import parse_ies, lookup_normalised, IesParseError


# Minimal LM-63 profile: 1 lamp, 1 lumen, 1 multiplier, 3 vertical angles,
# 1 horizontal angle (radially symmetric), simple downward beam.
_MINIMAL_IES = """\
IESNA:LM-63-1995
[TEST] synthetic
TILT=NONE
1 1 1 3 1 1 2 0 0 0
1 1 1
0 45 90
0
100 50 0
"""

# Three Lobe Vee from Pixar Public Domain repository (the one bundled
# with the Blender Lighting Bundle and referenced by flat_archiviz).
_THREE_LOBE_VEE = """\
IESNA:LM-63-1995
[TEST] 100069_0 BY: ERCO / LUM650
TILT=NONE
1 4850 1 19 1 1 2 -.097 0 0
1.00 1.00 70
0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90
0
68000 34008 6853 3982 936 582 543 514 349 189 92 58 44 34 0 0 0 0 0
"""


class TestIesParse(unittest.TestCase):
    def test_minimal(self):
        t = parse_ies(_MINIMAL_IES)
        self.assertEqual(t.thetas_deg, [0.0, 45.0, 90.0])
        self.assertEqual(t.phis_deg, [0.0])
        self.assertEqual(t.candelas, [100.0, 50.0, 0.0])
        self.assertEqual(t.peak_candela, 100.0)
        self.assertEqual(t.lumens_per_lamp, 1.0)
        self.assertEqual(t.multiplier, 1.0)
        self.assertEqual(t.n_v, 3)
        self.assertEqual(t.n_h, 1)

    def test_three_lobe_vee(self):
        t = parse_ies(_THREE_LOBE_VEE)
        self.assertEqual(t.n_v, 19)
        self.assertEqual(t.n_h, 1)
        self.assertEqual(len(t.candelas), 19)
        self.assertEqual(t.thetas_deg[0], 0.0)
        self.assertEqual(t.thetas_deg[-1], 90.0)
        self.assertEqual(t.candelas[0], 68000.0)  # peak straight-down
        self.assertEqual(t.candelas[-1], 0.0)     # zero at horizon
        self.assertEqual(t.peak_candela, 68000.0)
        self.assertEqual(t.lumens_per_lamp, 4850.0)

    def test_missing_tilt_raises(self):
        with self.assertRaises(IesParseError):
            parse_ies("IESNA:LM-63-1995\n1 1 1 3 1 1 2 0 0 0\n")

    def test_truncated_arrays_raise(self):
        bad = "TILT=NONE\n1 1 1 3 1 1 2 0 0 0\n1 1 1\n0 45 90\n0\n100 50\n"
        with self.assertRaises(IesParseError):
            parse_ies(bad)

    def test_non_monotonic_thetas_raise(self):
        bad = (
            "TILT=NONE\n1 1 1 3 1 1 2 0 0 0\n1 1 1\n"
            "0 90 45\n0\n100 50 0\n"
        )
        with self.assertRaises(IesParseError):
            parse_ies(bad)

    def test_multi_line_arrays_concatenated(self):
        # IES files routinely break long arrays across lines; the parser
        # should ignore line boundaries entirely after TILT=.
        wrapped = """\
TILT=NONE
1 1 1
3 1
1 2 0 0 0
1
1 1
0
45 90
0
100
50 0
"""
        t = parse_ies(wrapped)
        self.assertEqual(t.candelas, [100.0, 50.0, 0.0])


class TestIesLookup(unittest.TestCase):
    def test_lookup_endpoints(self):
        t = parse_ies(_MINIMAL_IES)
        self.assertAlmostEqual(lookup_normalised(t, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(lookup_normalised(t, 90.0, 0.0), 0.0)
        # Mid-range linear interp: candela@45° = 50, peak = 100 → 0.5
        self.assertAlmostEqual(lookup_normalised(t, 45.0, 0.0), 0.5)

    def test_lookup_clamps_outside_theta(self):
        t = parse_ies(_MINIMAL_IES)
        # Below 0 → 0; above 90 → 0 (last sample). No reflection / wrap
        # because LM-63 vertical range is well-defined.
        self.assertAlmostEqual(lookup_normalised(t, -10.0, 0.0), 1.0)
        self.assertAlmostEqual(lookup_normalised(t, 200.0, 0.0), 0.0)

    def test_radial_symmetry_ignores_phi(self):
        t = parse_ies(_THREE_LOBE_VEE)
        a = lookup_normalised(t, 30.0, 0.0)
        b = lookup_normalised(t, 30.0, 90.0)
        c = lookup_normalised(t, 30.0, 270.0)
        self.assertEqual(a, b)
        self.assertEqual(b, c)

    def test_three_lobe_vee_curve_shape(self):
        t = parse_ies(_THREE_LOBE_VEE)
        # Expect a fast falloff: 1.0 at 0°, ~0.5 at 5° (34008/68000),
        # ~0.10 at 10° (6853/68000), ≈0 by 70°+.
        self.assertAlmostEqual(lookup_normalised(t, 0.0, 0.0), 1.0)
        self.assertAlmostEqual(
            lookup_normalised(t, 5.0, 0.0), 34008.0 / 68000.0, places=5
        )
        self.assertAlmostEqual(
            lookup_normalised(t, 10.0, 0.0), 6853.0 / 68000.0, places=5
        )
        self.assertAlmostEqual(lookup_normalised(t, 80.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
