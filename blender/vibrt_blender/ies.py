"""IES (Illuminating Engineering Society) photometric profile parser.

Parses LM-63-1995 / 2002 IES files into a structured table that the
renderer can sample. The output is designed to be embedded directly in
`scene.json` (the candela tables are small — typically 20-200 floats).

The parser is intentionally lenient about whitespace, tabs, and stray
non-ASCII bytes (some IES files in the wild are Latin-1 / cp1252 with
accented manufacturer names) — the numeric block is what matters.

Cycles' Blender ShaderNodeTexIES has two modes:
- EXTERNAL: `node.filepath` points to a .ies on disk
- INTERNAL: `node.ies` is a Text datablock holding the IES content

Use `parse_ies(text: str) -> IesTable` for either path; the caller is
responsible for reading the file or `.ies.as_string()`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class IesTable:
    """Parsed IES profile.

    `thetas_deg` are vertical angles (0° = down the light's local -Z, 90°
    = horizontal, 180° = up the local +Z) per IES Type C convention.

    `phis_deg` are horizontal angles around the light's local Z axis.
    A single-element `phis_deg` (the IES "1 horizontal angle" case)
    means the profile is radially symmetric.

    `candelas` is row-major over `phis_deg × thetas_deg`. So
    `candelas[h * V + v]` is intensity at (phis_deg[h], thetas_deg[v]).
    """
    thetas_deg: list[float]
    phis_deg: list[float]
    candelas: list[float]
    lumens_per_lamp: float
    multiplier: float

    @property
    def n_v(self) -> int:
        return len(self.thetas_deg)

    @property
    def n_h(self) -> int:
        return len(self.phis_deg)

    @property
    def peak_candela(self) -> float:
        return max(self.candelas) if self.candelas else 0.0

    def integral_normalised(self) -> float:
        """Solid-angle integral of `candela / peak_candela` over the
        sphere, in steradians.

        This is what Cycles' Point/Spot lights divide the user-set Power
        by to preserve total emitted flux when an IES is attached:
        without the IES the per-direction intensity is `power / (4π)`
        (isotropic), with the IES it's `power / integral_norm` so that
        `∫ I dΩ = power` regardless of how concentrated the beam is.

        Computed as a piecewise-constant 2D integral over the table
        bins, taking phi-bin angular widths from the file (mirroring
        the LM-63 fold conventions: a profile that covers only [0,90]
        is implicitly quadrant-symmetric, etc.).
        """
        import math
        peak = self.peak_candela
        if peak <= 0.0 or not self.candelas:
            return 0.0
        ts = self.thetas_deg
        ps = self.phis_deg
        n_v = len(ts)
        n_h = len(ps)
        # Determine the full phi span the table represents. LM-63
        # convention: a single phi=0 entry → full radial symmetry
        # (effective span 360°); a 0..90 table → quadrant symmetric
        # (full span 360°, the table covers 1/4 by mirror); 0..180 →
        # bilateral (full span 360°, table covers 1/2); 0..360 → full.
        if n_h == 1:
            phi_span_total = 2.0 * math.pi
        else:
            phi_max = ps[-1]
            if phi_max <= 90.0:
                # Quadrant symmetric. Each table value represents the
                # same intensity in 4 mirrored quadrants.
                full_factor = 4.0
            elif phi_max <= 180.0:
                full_factor = 2.0
            else:
                full_factor = 1.0
            # We'll integrate the table directly and multiply by
            # full_factor at the end.
            phi_span_total = full_factor * (math.radians(ps[-1] - ps[0])
                                            if n_h > 1 else 0.0)
        # 2D piecewise-trapezoidal: ∫∫ f(θ, φ) sin θ dθ dφ
        # We average f over each (θ, φ) bin and weight by the bin's
        # solid-angle area.
        if n_h == 1:
            # Radially symmetric: integrate over θ and multiply by 2π.
            total = 0.0
            for i in range(n_v - 1):
                t0 = math.radians(ts[i])
                t1 = math.radians(ts[i + 1])
                f0 = self.candelas[i] / peak
                f1 = self.candelas[i + 1] / peak
                avg = 0.5 * (f0 * math.sin(t0) + f1 * math.sin(t1))
                total += avg * (t1 - t0)
            return 2.0 * math.pi * total
        # Multi-phi: integrate over both axes.
        total = 0.0
        for j in range(n_h - 1):
            p0 = math.radians(ps[j])
            p1 = math.radians(ps[j + 1])
            dphi = p1 - p0
            for i in range(n_v - 1):
                t0 = math.radians(ts[i])
                t1 = math.radians(ts[i + 1])
                dtheta = t1 - t0
                # Average sin θ over [t0, t1] bin endpoints.
                f00 = self.candelas[j * n_v + i] / peak
                f01 = self.candelas[j * n_v + (i + 1)] / peak
                f10 = self.candelas[(j + 1) * n_v + i] / peak
                f11 = self.candelas[(j + 1) * n_v + (i + 1)] / peak
                # bilinear average × sin θ at bin centres
                # Use sin at endpoints and average to keep it simple.
                avg = 0.25 * (f00 * math.sin(t0) + f01 * math.sin(t1)
                              + f10 * math.sin(t0) + f11 * math.sin(t1))
                total += avg * dtheta * dphi
        # Apply mirror multiplier for partial-coverage tables.
        phi_max = ps[-1]
        if phi_max <= 90.0:
            total *= 4.0
        elif phi_max <= 180.0:
            total *= 2.0
        return total


class IesParseError(ValueError):
    pass


def _strip_keyword_block(lines: list[str]) -> list[str]:
    """Drop the file's keyword/header block, returning only the lines from
    `TILT=...` onward. Real-world IES files have variable preambles
    (IESNA: marker, [TEST], [MANUFAC], free-form comments before LM-63
    standardisation), so anchoring on TILT= is the most reliable cut.
    """
    for i, ln in enumerate(lines):
        if ln.lstrip().upper().startswith("TILT="):
            return lines[i:]
    raise IesParseError("missing 'TILT=' line")


def _flatten_numbers(lines: list[str]) -> list[str]:
    """Re-tokenise multi-number lines into a flat list. IES files
    routinely break long numeric arrays across multiple lines and pack
    multiple numbers per line, so we ignore line boundaries entirely
    after the TILT= directive."""
    out: list[str] = []
    for ln in lines:
        out.extend(ln.split())
    return out


_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _take_floats(tokens: list[str], n: int, where: str) -> list[float]:
    if len(tokens) < n:
        raise IesParseError(
            f"{where}: expected {n} numbers, only {len(tokens)} left in stream"
        )
    out = []
    for i in range(n):
        t = tokens[i]
        if not _NUM_RE.match(t):
            raise IesParseError(f"{where}: non-numeric token {t!r}")
        out.append(float(t))
    del tokens[:n]
    return out


def _take_floats_int(tokens: list[str], n: int, where: str) -> list[int]:
    return [int(x) for x in _take_floats(tokens, n, where)]


def parse_ies(text: str) -> IesTable:
    """Parse an IES (LM-63) string into a numeric table.

    Raises IesParseError on malformed input. Tolerates inline TILT
    blocks (TILT=INCLUDE) by silently skipping them — they affect a
    real-world fixture's mounting orientation but not the photometric
    distribution we sample.
    """
    # Normalise line endings, drop trailing empties.
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        raise IesParseError("empty IES content")

    body = _strip_keyword_block(lines)
    # body[0] is the TILT= directive
    tilt_line = body[0].strip()
    after_tilt = body[1:]
    tilt_val = tilt_line.split("=", 1)[1].strip().upper()
    if tilt_val == "NONE":
        rest = after_tilt
    elif tilt_val == "INCLUDE":
        # Inline tilt block: <lamp_to_lum_geometry> <num_tilt_angles>
        # <num_tilt_angles> angles ... <num_tilt_angles> multipliers ...
        # We don't apply tilt — its effect is geometric (mounting), so
        # skip past it. Use _flatten_numbers to find the count.
        flat = _flatten_numbers(after_tilt)
        if len(flat) < 2:
            raise IesParseError("TILT=INCLUDE: tilt block truncated")
        # lamp_to_lum_geometry, num_tilt_angles
        _ = flat[0]
        n_tilt = int(float(flat[1]))
        skip = 2 + 2 * n_tilt  # angles + multipliers
        if len(flat) < skip:
            raise IesParseError("TILT=INCLUDE: tilt arrays truncated")
        # Re-emit remaining numbers as a single space-separated line so
        # the regular flow takes over. (We've discarded line structure
        # but the rest of the parser doesn't care about it.)
        rest = [" ".join(flat[skip:])]
    else:
        # External tilt file. We treat as NONE — the path-resolved file
        # is rarely shipped and not relevant to photometric shape.
        rest = after_tilt

    tokens = _flatten_numbers(rest)
    # Standard line: 10 values
    h1 = _take_floats(tokens, 10, "header line 1")
    num_lamps = int(h1[0])
    lumens_per_lamp = h1[1]
    multiplier = h1[2]
    n_v = int(h1[3])
    n_h = int(h1[4])
    photometric_type = int(h1[5])
    units_type = int(h1[6])
    # h1[7..9] = width, length, height (unused — geometry of the luminaire)
    if num_lamps < 1 or n_v < 1 or n_h < 1:
        raise IesParseError(
            f"invalid header: num_lamps={num_lamps} n_v={n_v} n_h={n_h}"
        )
    # Standard line: 3 values (ballast_factor, future_use, input_watts) —
    # discarded.
    _ = _take_floats(tokens, 3, "header line 2")

    thetas = _take_floats(tokens, n_v, "vertical angles")
    phis = _take_floats(tokens, n_h, "horizontal angles")
    candelas = _take_floats(tokens, n_v * n_h, "candela values")

    # Sanity: angles must be monotonic.
    if any(thetas[i] >= thetas[i + 1] for i in range(n_v - 1)):
        raise IesParseError("vertical angles not strictly increasing")
    if n_h > 1 and any(phis[i] >= phis[i + 1] for i in range(n_h - 1)):
        raise IesParseError("horizontal angles not strictly increasing")

    return IesTable(
        thetas_deg=thetas,
        phis_deg=phis,
        candelas=candelas,
        lumens_per_lamp=lumens_per_lamp,
        multiplier=multiplier,
    )


def lookup_normalised(table: IesTable, theta_deg: float, phi_deg: float) -> float:
    """Bilinear lookup, returning candela / peak_candela (range [0, 1]).

    `theta_deg` is clamped to the table's vertical range; `phi_deg` is
    wrapped per the file's horizontal symmetry rules: 1 angle → radially
    symmetric (phi ignored); spans up to 90/180/360 → mirrored or wrapped.

    This is the host-side reference implementation used by tests and by
    occasional pre-bake / sanity passes; the GPU does the same logic in
    `devicecode.cu`.
    """
    if not table.candelas:
        return 0.0
    peak = table.peak_candela
    if peak <= 0.0:
        return 0.0

    thetas = table.thetas_deg
    n_v = len(thetas)
    # Clamp theta to table range.
    if theta_deg <= thetas[0]:
        v_lo, v_hi, tv = 0, 0, 0.0
    elif theta_deg >= thetas[-1]:
        v_lo, v_hi, tv = n_v - 1, n_v - 1, 0.0
    else:
        v_lo = 0
        for i in range(n_v - 1):
            if thetas[i] <= theta_deg < thetas[i + 1]:
                v_lo = i
                break
        v_hi = v_lo + 1
        span = thetas[v_hi] - thetas[v_lo]
        tv = (theta_deg - thetas[v_lo]) / span if span > 0 else 0.0

    phis = table.phis_deg
    n_h = len(phis)
    if n_h == 1:
        h_lo = h_hi = 0
        th = 0.0
    else:
        # Phi range determines symmetry per LM-63: 0..90 = quadrant
        # symmetric, 0..180 = bilateral, 0..360 = full. Fold accordingly.
        phi_max = phis[-1]
        p = phi_deg % 360.0
        if phi_max <= 90.0:
            # Quadrant — fold to [0, 90].
            p = p % 180.0
            if p > 90.0:
                p = 180.0 - p
            if p > 90.0:  # safety
                p = 90.0
        elif phi_max <= 180.0:
            if p > 180.0:
                p = 360.0 - p
        # else full range, no fold

        if p <= phis[0]:
            h_lo = h_hi = 0
            th = 0.0
        elif p >= phis[-1]:
            h_lo = h_hi = n_h - 1
            th = 0.0
        else:
            h_lo = 0
            for i in range(n_h - 1):
                if phis[i] <= p < phis[i + 1]:
                    h_lo = i
                    break
            h_hi = h_lo + 1
            span = phis[h_hi] - phis[h_lo]
            th = (p - phis[h_lo]) / span if span > 0 else 0.0

    def at(h: int, v: int) -> float:
        return table.candelas[h * n_v + v] / peak

    a = at(h_lo, v_lo)
    b = at(h_lo, v_hi)
    c = at(h_hi, v_lo)
    d = at(h_hi, v_hi)
    ab = a * (1 - tv) + b * tv
    cd = c * (1 - tv) + d * tv
    return ab * (1 - th) + cd * th
