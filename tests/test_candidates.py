"""Tests for ``hishells.candidates``.

We build a synthetic 2-D moment-0 with planted local minima and
verify that ``mom0_minima`` recovers them. The MTB-side and the
DBSCAN dedup are smoke-tested with a tiny synthetic cube.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from hishells.candidates import (
    Candidate,
    dedupe_candidates,
    enumerate_candidates,
    mom0_minima,
)


# ---------------------------------------------------------------------------
# Synthetic MOM0 with planted minima
# ---------------------------------------------------------------------------


@pytest.fixture
def planted_mom0(tmp_path: Path) -> tuple[Path, list[tuple[int, int]]]:
    n = 200
    rng = np.random.default_rng(0)
    bg = 10.0 + rng.normal(0, 0.2, size=(n, n)).astype(np.float32)
    # Plant 5 local minima at known pixel positions.
    sites = [(40, 40), (40, 160), (100, 100), (160, 40), (160, 160)]
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for cy, cx in sites:
        rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
        bg -= 8.0 * np.exp(-rr2 / (2 * 6.0**2))
    # Add a bright source elsewhere so the percentile cap is realistic.
    bg += 30.0 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 4.0**2))

    w = WCS(naxis=2)
    w.wcs.crpix = [n / 2, n / 2]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdu = fits.PrimaryHDU(data=bg, header=w.to_header())
    out = tmp_path / "synth_mom0.fits"
    hdu.writeto(out)
    return out, sites


def test_mom0_minima_recovers_planted(planted_mom0):
    path, sites = planted_mom0
    cands = mom0_minima(path, footprint_arcsec=15.0, relative_depth=0.95)
    assert len(cands) > 0
    # Each planted site (in pixel space) should match one returned
    # candidate within 5 pixels (= 7.5 arcsec) on the sky.
    with fits.open(path) as hdul:
        wcs = WCS(hdul[0].header).celestial
    for cy, cx in sites:
        ra_target, dec_target = wcs.wcs_pix2world(cx, cy, 0)
        sep = np.array(
            [
                np.hypot(c.ra_deg - float(ra_target), c.dec_deg - float(dec_target))
                for c in cands
            ]
        )
        assert sep.min() < 0.005  # 18 arcsec at most


# ---------------------------------------------------------------------------
# DBSCAN dedup
# ---------------------------------------------------------------------------


def test_dedupe_collapses_close_candidates():
    cands = [
        Candidate(180.000, 30.0, 0.0, 0.5, "mom0"),
        # 2" away (well under the 6" THINGS beam) -> same cluster.
        Candidate(180.000 + 2.0 / 3600.0, 30.0, 0.0, 0.7, "mtb"),
        # 30" away -> separate cluster.
        Candidate(180.000 + 30.0 / 3600.0, 30.0, 0.0, 0.6, "mom0"),
    ]
    out = dedupe_candidates(cands, eps_arcsec=6.0)
    assert len(out) == 2
    # The first cluster's representative should be the higher-scoring one.
    rep = max(out, key=lambda c: c.score)
    assert rep.score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# enumerate_candidates end-to-end on a tiny in-memory cube
# ---------------------------------------------------------------------------


def test_enumerate_candidates_runs_on_real_cube(things_dir, tmp_path):
    p = things_dir / "DDO53_NA_CUBE_THINGS.FITS"
    if not p.exists():
        pytest.skip("DDO53 cube not downloaded")
    df = enumerate_candidates(
        p,
        mom0_path=None,
        mtb_kwargs=dict(spatial_stride_arcsec=120.0, top_k=20),
    )
    # Should produce *some* candidates and the schema columns.
    assert set(df.columns) >= {"ra_deg", "dec_deg", "vel_kms", "score_seed", "source"}
