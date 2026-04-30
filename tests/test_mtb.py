"""Smoke tests for ``hishells.baselines.mtb`` against synthetic shells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from hishells.baselines.mtb import (
    MTBGrid,
    best_score_at,
    normalise_by_rho_99,
    score_table,
    shell_template,
)
from hishells.baselines.trivial import score_table as trivial_score
from hishells.cubes import Cube


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _shell_cube(
    R_arcsec: float = 18.0,
    Vexp_kms: float = 15.0,
    n_pix: int = 96,
    n_chan: int = 31,
    pix_arcsec: float = 1.5,
) -> Cube:
    cy = (n_pix - 1) / 2.0
    cx = (n_pix - 1) / 2.0
    velocity_kms = np.linspace(60, -60, n_chan)

    yy, xx = np.meshgrid(
        np.arange(n_pix) - cy, np.arange(n_pix) - cx, indexing="ij"
    )
    rho_arcsec = np.sqrt(xx**2 + yy**2) * pix_arcsec
    inside = rho_arcsec < R_arcsec

    data = np.full((n_chan, n_pix, n_pix), 1.0, dtype=np.float32)
    rim_thickness_arcsec = 4.0
    sv = rim_thickness_arcsec / 2.355
    for k, v in enumerate(velocity_kms):
        if abs(v) > Vexp_kms:
            ring_r = 0.0
        else:
            ring_r = R_arcsec * np.sqrt(1.0 - (v / Vexp_kms) ** 2)
        rim = np.exp(-((rho_arcsec - ring_r) ** 2) / (2 * sv**2))
        data[k] = 1.0 + 1.5 * rim
        data[k][rho_arcsec < ring_r - rim_thickness_arcsec] *= 0.05

    w = WCS(naxis=2)
    w.wcs.crpix = [cx + 1, cy + 1]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return Cube(
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


class _OneCubeStore:
    def __init__(self, cube: Cube):
        self._cube = cube

    def __call__(self, gid: str) -> Cube:
        return self._cube


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def test_shell_template_zero_outside():
    tmpl = shell_template(R_pix=8.0, Vexp_chan=4.0, n_pix=21, n_chan=11)
    # Corner of the (n_pix, n_pix) grid is outside the shell -> 0.
    assert tmpl[5, 0, 0] == 0.0
    # Centre of the (n_pix, n_pix) grid is inside the shell ->
    # nonzero only at +/- Vexp_chan from cz.
    cz = (11 - 1) // 2
    cx = cy = (21 - 1) // 2
    centre_spectrum = tmpl[:, cy, cx]
    assert centre_spectrum[cz - 4] > 0.5
    assert centre_spectrum[cz + 4] > 0.5
    assert centre_spectrum[cz] < 0.5


# ---------------------------------------------------------------------------
# best_score_at: shell-cube center is the maximum
# ---------------------------------------------------------------------------


def test_best_score_at_centred_shell_high():
    cube = _shell_cube()
    rho, R_best, V_best = best_score_at(
        cube, ra_deg=180.0, dec_deg=30.0, vel_kms=0.0
    )
    # Shell + matched template -> very high anti-correlation.
    assert rho > 0.3
    # The best R should be in the same ballpark as the planted radius.
    assert 10.0 <= R_best <= 35.0


def test_best_score_at_off_centre_low():
    cube = _shell_cube()
    rho, _, _ = best_score_at(
        cube, ra_deg=180.0 + 0.012, dec_deg=30.0, vel_kms=0.0
    )
    # Off-cavity sightline should score lower than at the centre.
    assert rho < 0.3


# ---------------------------------------------------------------------------
# score_table end-to-end + rho_99 normalisation
# ---------------------------------------------------------------------------


def _table_pos_and_negs(cube: Cube) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "galaxy_id": "SYNTH",
            "ra_deg": 180.0,
            "dec_deg": 30.0,
            "vel_helio_kms": 0.0,
            "pa_deg": 0.0,
            "diameter_arcsec": 36.0,
            "vexp_kms": 15.0,
            "sigma_gas_kms": 10.0,
            "label": 1,
            "neg_kind": pd.NA,
        }
    ]
    for _ in range(15):
        rows.append(
            {
                "galaxy_id": "SYNTH",
                "ra_deg": float(180.0 + rng.uniform(0.005, 0.015)),
                "dec_deg": float(30.0 + rng.uniform(-0.01, 0.01)),
                "vel_helio_kms": float(rng.uniform(-50, 50)),
                "pa_deg": 0.0,
                "diameter_arcsec": 36.0,
                "vexp_kms": 15.0,
                "sigma_gas_kms": 10.0,
                "label": 0,
                "neg_kind": "hard",
            }
        )
    return pd.DataFrame(rows)


def test_score_table_separates_positive_from_negatives():
    cube = _shell_cube()
    table = _table_pos_and_negs(cube)
    scores, labels = score_table(table, _OneCubeStore(cube))  # type: ignore[arg-type]
    assert scores[labels == 1].max() > scores[labels == 0].max()


def test_normalise_by_rho_99_above_one_for_strong_positive():
    cube = _shell_cube()
    table = _table_pos_and_negs(cube)
    scores, labels = score_table(table, _OneCubeStore(cube))  # type: ignore[arg-type]
    norm = normalise_by_rho_99(scores, labels)
    # Strong positive should sit above rho_99.
    assert norm[labels == 1][0] > 1.0


# ---------------------------------------------------------------------------
# Trivial baseline as the floor
# ---------------------------------------------------------------------------


def test_trivial_baseline_runs_and_varies():
    """Trivial baseline is just the floor; on this synthetic cube the
    *direction* of the deficit signal is dominated by the per-window
    sigma_rms, which is unrealistic. We only check that the function
    runs end-to-end, returns the right shapes, and produces non-constant
    scores; the real evaluation happens against THINGS data in
    ``scripts/run_baseline.py``.
    """

    cube = _shell_cube()
    table = _table_pos_and_negs(cube)
    scores, labels = trivial_score(table, _OneCubeStore(cube))  # type: ignore[arg-type]
    assert scores.shape == (len(table),)
    assert labels.shape == (len(table),)
    assert np.unique(scores).size > 1
