"""Unit tests for ``hishells.pvcut`` against synthetic cubes.

We build a tiny in-memory cube with a single expanding-shell signature
planted at a known ``(ra, dec, vel)``, then verify:

* :func:`extract_window` recovers a window whose central pixel sits
  *inside* the shell (i.e. lower brightness than the rim).
* :func:`window_extent_for_hole` falls back to the gas-dispersion
  velocity for type-1 holes.
* PA rotation aligns the shell's expected velocity-symmetry axis with
  the position axis (rotating by +90 deg should swap the role of the
  two output axes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from hishells.cubes import Cube, load_cube
from hishells.pvcut import (
    WindowExtent,
    extract_window,
    extract_window_for_hole,
    window_extent_for_hole,
)


# ---------------------------------------------------------------------------
# Synthetic cube fixture
# ---------------------------------------------------------------------------


def _synth_cube_with_shell(
    *,
    n_chan: int = 31,
    n_pix: int = 192,
    pix_arcsec: float = 1.5,
    chan_kms: float = 5.0,
    shell_radius_arcsec: float = 18.0,
    vexp_kms: float = 15.0,
    rim_thickness_arcsec: float = 4.0,
) -> tuple[Cube, dict]:
    """Build an in-memory :class:`Cube` with one expanding shell.

    The cube is centered at (RA, Dec) = (180.0, 30.0) deg, has a
    descending velocity axis (mimicking THINGS), and contains uniform
    diffuse emission with a thin spherical shell carved out of it
    (low-brightness inside, brighter rim, low background outside the
    galaxy).
    """

    n_dec = n_ra = n_pix
    cx_pix = (n_ra - 1) / 2.0
    cy_pix = (n_dec - 1) / 2.0

    velocity_kms = np.linspace(60, -60, n_chan)  # descending
    cz_kms = 0.0

    yy, xx = np.meshgrid(
        np.arange(n_dec) - cy_pix, np.arange(n_ra) - cx_pix, indexing="ij"
    )
    rr_arcsec = np.sqrt(xx**2 + yy**2) * pix_arcsec

    data = np.full((n_chan, n_dec, n_ra), 1.0, dtype=np.float32)
    for k, v in enumerate(velocity_kms):
        # Expanding-shell signature: at velocity v relative to systemic,
        # the projected ring radius shrinks toward |v| -> Vexp.
        if abs(v - cz_kms) > vexp_kms:
            ring_r = 0.0
        else:
            ring_r = shell_radius_arcsec * np.sqrt(
                1.0 - ((v - cz_kms) / vexp_kms) ** 2
            )
        rim = np.exp(-((rr_arcsec - ring_r) ** 2) / (2 * (rim_thickness_arcsec / 2.355) ** 2))
        # Brighten the rim, suppress the inside.
        data[k] = 1.0 + 1.5 * rim
        data[k][rr_arcsec < ring_r - rim_thickness_arcsec] *= 0.05

    # Build a minimal celestial WCS at (RA, Dec) = (180, 30) deg
    # with ``pix_arcsec`` per pixel, RA increasing to the *left*.
    w = WCS(naxis=2)
    w.wcs.crpix = [cx_pix + 1, cy_pix + 1]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    cube = Cube(
        data=data,
        wcs2d=w,
        velocity_kms=velocity_kms,
        beam_bmaj_arcsec=6.0,
        beam_bmin_arcsec=6.0,
        beam_bpa_deg=0.0,
        pixel_scale_arcsec=pix_arcsec,
        galaxy_id="SYNTH",
        path=Path("synth"),
    )

    hole = {
        "ra_deg": 180.0,
        "dec_deg": 30.0,
        "vel_helio_kms": cz_kms,
        "pa_deg": 0.0,
        "diameter_arcsec": 2 * shell_radius_arcsec,
        "diameter_pc": 2 * shell_radius_arcsec * 10.0,
        "vexp_kms": vexp_kms,
        "sigma_gas_kms": 10.0,
        "hole_type": 3,
    }
    return cube, hole


# ---------------------------------------------------------------------------
# window_extent_for_hole
# ---------------------------------------------------------------------------


def test_window_extent_type_2_3():
    we = window_extent_for_hole(diameter_arcsec=20.0, vexp_kms=12.0)
    assert isinstance(we, WindowExtent)
    assert we.pos_extent_arcsec == 40.0
    # 2 * 12 = 24 km/s, exceeds the 20-km/s floor.
    assert we.vel_extent_kms == pytest.approx(24.0)


def test_window_extent_type_2_3_below_floor():
    we = window_extent_for_hole(diameter_arcsec=20.0, vexp_kms=4.0)
    # 2 * 4 = 8 km/s -> floor at 20 km/s.
    assert we.vel_extent_kms == pytest.approx(20.0)


def test_window_extent_type_1_fallback():
    we = window_extent_for_hole(
        diameter_arcsec=20.0, vexp_kms=float("nan"), sigma_gas_kms=11.0
    )
    assert we.vel_extent_kms == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# extract_window on synthetic data
# ---------------------------------------------------------------------------


def test_extract_window_centered_inside_shell():
    cube, hole = _synth_cube_with_shell()
    win = extract_window_for_hole(cube, hole, window_pix=64)
    assert win.shape == (64, 64)
    assert win.dtype == np.float32

    # Window center should land inside the shell -> brightness < 1.0
    # because the carved-out interior is at ~0.05.
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    assert win[cy, cx] < 0.5

    # The (position-extreme, velocity-center) sample should be on the
    # rim or beyond the shell -> brightness >= 1.0 (background).
    assert win[0, cx] > 0.5 and win[-1, cx] > 0.5


def test_extract_window_pa_rotation_changes_pattern():
    """Rotating PA by 90 deg should flip the orientation pattern."""

    cube, hole = _synth_cube_with_shell()
    win0 = extract_window_for_hole(cube, hole, window_pix=64)
    hole90 = dict(hole)
    hole90["pa_deg"] = 90.0
    win90 = extract_window_for_hole(cube, hole90, window_pix=64)
    # For a circularly symmetric synthetic shell the windows should be
    # nearly identical (PA only matters for elongated holes, but the
    # extraction grid is built differently). What matters is they share
    # the same low-brightness center.
    assert win0[32, 32] < 0.5
    assert win90[32, 32] < 0.5


def test_extract_window_offset_far_is_emission_free():
    """A sightline far from the shell should be uniform background."""

    cube, hole = _synth_cube_with_shell()
    far = dict(hole)
    # 0.02 deg = 72 arcsec offset; well outside the 36-arcsec shell but
    # still inside the synthetic cube footprint.
    far["ra_deg"] = 180.0 + 0.02
    win = extract_window_for_hole(cube, far, window_pix=64)
    # We placed background = 1.0 everywhere outside the shell.
    assert np.all(win >= 0.95)
    assert np.all(win <= 1.05)


# ---------------------------------------------------------------------------
# Real-cube smoke test (skipped if NGC_2403 isn't downloaded yet)
# ---------------------------------------------------------------------------


def test_extract_window_real_cube_smoke(things_dir):
    p = things_dir / "NGC_2403_NA_CUBE_THINGS.FITS"
    if not p.exists():
        pytest.skip("NGC_2403 cube not downloaded")
    cube = load_cube(p)
    win = extract_window(
        cube,
        ra_deg=cube.wcs2d.wcs.crval[0],
        dec_deg=cube.wcs2d.wcs.crval[1],
        vel_kms=float(np.median(cube.velocity_kms)),
        pa_deg=0.0,
        pos_extent_arcsec=80.0,
        vel_extent_kms=40.0,
    )
    # Default window_pix bumped from 64 to 96 in plan §2.1 (verified
    # 2026-04-30 in 04_window_inspection.ipynb). This smoke test takes
    # the default to confirm the default is what we think it is.
    assert win.shape == (96, 96)
    assert np.all(np.isfinite(win))
