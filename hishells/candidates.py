"""Inference-time candidate enumeration (plan \u00a71.3).

For a never-before-seen cube there's no B11 catalog, so the CNN
needs *something* to score. v1 default per the plan is the union of
two cheap candidate generators, deduplicated by DBSCAN at the
THINGS beam scale (eps = 6\"):

* :func:`mom0_minima` -- local minima of the integrated-intensity
  map (the "holes \u21d4 depressions" intuition). Fast: ~10\u00b2-10\u00b3
  candidates per cube.
* :func:`mtb_candidates` -- top-N candidates from the MTB template
  matcher (plan \u00a79 Row 1). Free correlation with the baseline.

The DBSCAN dedup runs in (RA, Dec) projected onto a small angular
neighbourhood (we don't need a full 3-D dedup since velocity is much
better resolved than position; cluster representatives keep the best
score).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import minimum_filter

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None  # type: ignore[assignment]

from . import THINGS_BEAM_ARCSEC
from .baselines.mtb import MTBGrid, best_score_at
from .cubes import Cube, channel_width_kms, load_cube, pix_to_world


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    ra_deg: float
    dec_deg: float
    vel_kms: float
    score: float
    source: str  # "mom0" | "mtb"


# ---------------------------------------------------------------------------
# MOM0 local minima
# ---------------------------------------------------------------------------


def mom0_minima(
    mom0_path: str | Path,
    *,
    footprint_arcsec: float = 30.0,
    relative_depth: float = 0.5,
) -> list[Candidate]:
    """Return ``Candidate`` rows at MOM0 local minima.

    A pixel is a candidate if it equals its
    ``minimum_filter(footprint)`` value AND its intensity is below
    ``relative_depth * percentile_95(mom0)``. The score is
    ``1 - mom0_value / percentile_95``, so deeper minima score higher.
    The velocity is filled with NaN; callers wrap into a small grid
    of ``(vel_systemic +/- delta)`` priors before scoring with the CNN.
    """

    p = Path(mom0_path)
    with fits.open(p) as hdul:
        data = np.asarray(hdul[0].data).astype(np.float32)
        header = hdul[0].header
    if data.ndim == 3:
        data = data[0]
    elif data.ndim == 4:
        data = data[0, 0]

    wcs = WCS(header).celestial
    pix_arcsec = float(np.mean([abs(c) * 3600.0 for c in wcs.wcs.cdelt]))
    footprint_pix = max(3, int(round(footprint_arcsec / pix_arcsec)))

    nan_mask = np.isnan(data)
    safe = np.where(nan_mask, np.nanmax(data) + 1.0, data)
    local_min = minimum_filter(safe, size=footprint_pix)
    is_min = (safe == local_min) & ~nan_mask

    p95 = float(np.nanpercentile(data, 95))
    if p95 <= 0 or not np.isfinite(p95):
        return []
    depth_threshold = relative_depth * p95
    is_dark = data < depth_threshold

    yy, xx = np.where(is_min & is_dark)
    if xx.size == 0:
        return []
    ra_arr, dec_arr = wcs.wcs_pix2world(xx, yy, 0)
    scores = 1.0 - data[yy, xx] / p95
    return [
        Candidate(
            ra_deg=float(ra_arr[i]),
            dec_deg=float(dec_arr[i]),
            vel_kms=float("nan"),
            score=float(scores[i]),
            source="mom0",
        )
        for i in range(xx.size)
    ]


# ---------------------------------------------------------------------------
# MTB-derived candidates
# ---------------------------------------------------------------------------


def mtb_candidates(
    cube: Cube,
    *,
    grid: MTBGrid | None = None,
    spatial_stride_arcsec: float = 30.0,
    spectral_stride_kms: float | None = None,
    top_k: int = 200,
) -> list[Candidate]:
    """Top-``k`` MTB scores over a regular (RA, Dec, vel) grid."""

    grid = grid or MTBGrid()
    pix_per_arcsec = 1.0 / cube.pixel_scale_arcsec
    stride_pix = max(2, int(round(spatial_stride_arcsec * pix_per_arcsec)))
    chw = channel_width_kms(cube)
    stride_chan = (
        max(1, int(round((spectral_stride_kms or 4.0) / chw))) if np.isfinite(chw) else 1
    )

    margin = stride_pix
    xs = np.arange(margin, cube.shape[2] - margin, stride_pix)
    ys = np.arange(margin, cube.shape[1] - margin, stride_pix)
    zs = np.arange(0, cube.n_chan, stride_chan)

    cands: list[Candidate] = []
    for x in xs:
        for y in ys:
            for z in zs:
                ra, dec = pix_to_world(cube, np.array([x]), np.array([y]))
                rho, _, _ = best_score_at(
                    cube,
                    ra_deg=float(ra[0]),
                    dec_deg=float(dec[0]),
                    vel_kms=float(cube.velocity_kms[z]),
                    grid=grid,
                )
                if rho <= 0:
                    continue
                cands.append(
                    Candidate(
                        ra_deg=float(ra[0]),
                        dec_deg=float(dec[0]),
                        vel_kms=float(cube.velocity_kms[z]),
                        score=float(rho),
                        source="mtb",
                    )
                )
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:top_k]


# ---------------------------------------------------------------------------
# DBSCAN deduplication
# ---------------------------------------------------------------------------


def dedupe_candidates(
    cands: list[Candidate],
    *,
    eps_arcsec: float = THINGS_BEAM_ARCSEC,
    min_samples: int = 1,
) -> list[Candidate]:
    """Cluster candidates within ``eps_arcsec`` on the sky and pick the
    highest-scoring representative per cluster.

    Note: declination scaling on RA is approximated with
    ``cos(dec_mean)``; for THINGS galaxies (|dec| < 75 deg) this is
    fine to ~10% on cluster boundaries, and DBSCAN is robust to that.
    """

    if not cands:
        return []
    if DBSCAN is None:  # pragma: no cover
        return cands

    arr = np.array([[c.ra_deg, c.dec_deg] for c in cands])
    cos_dec = np.cos(np.deg2rad(np.mean(arr[:, 1])))
    # Convert to a local cartesian frame in arcsec.
    xy = np.column_stack(
        [
            (arr[:, 0] - arr[0, 0]) * 3600.0 * cos_dec,
            (arr[:, 1] - arr[0, 1]) * 3600.0,
        ]
    )
    db = DBSCAN(eps=eps_arcsec, min_samples=min_samples).fit(xy)
    out: list[Candidate] = []
    for label in set(db.labels_):
        if label == -1:
            # DBSCAN noise points: keep individually so we don't lose
            # bona-fide isolated detections.
            for i, lab in enumerate(db.labels_):
                if lab == -1:
                    out.append(cands[i])
            continue
        members = [cands[i] for i, lab in enumerate(db.labels_) if lab == label]
        out.append(max(members, key=lambda c: c.score))
    return out


# ---------------------------------------------------------------------------
# Top-level enumeration
# ---------------------------------------------------------------------------


def enumerate_candidates(
    cube_path: str | Path,
    mom0_path: str | Path | None = None,
    *,
    velocity_grid_kms: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0),
    mom0_kwargs: dict | None = None,
    mtb_kwargs: dict | None = None,
    eps_arcsec: float = THINGS_BEAM_ARCSEC,
) -> pd.DataFrame:
    """Build the union of MOM0-minima and MTB candidates.

    Parameters
    ----------
    cube_path
        Path to the THINGS NA cube.
    mom0_path
        Optional pre-computed MOM0 FITS path. If omitted we compute
        the moment-0 in-memory from the cube.
    velocity_grid_kms
        Each MOM0 candidate is replicated across this set of velocity
        offsets relative to the cube's *systemic* velocity (median of
        the cube's velocity axis); this is the "small grid of
        (Vexp, vel_systemic) priors" from plan \u00a71.3 Option C.
    mom0_kwargs, mtb_kwargs
        Forwarded to :func:`mom0_minima` / :func:`mtb_candidates`.
    eps_arcsec
        DBSCAN ``eps`` for deduplication. Default is the THINGS beam.
    """

    cube = load_cube(cube_path)
    sys_v = float(np.nanmedian(cube.velocity_kms))

    # -- MOM0 candidates --
    mom0_cands: list[Candidate] = []
    if mom0_path is None:
        # Build a temp MOM0 array in-memory and reuse the same loader.
        from .cubes import moment0

        m0 = moment0(cube)
        # mom0_minima needs a FITS header; bypass by replicating its
        # logic here on the in-memory array.
        mom0_cands = _mom0_minima_from_array(m0, cube)
    else:
        mom0_cands = mom0_minima(mom0_path, **(mom0_kwargs or {}))

    # Replicate each MOM0 candidate over the velocity grid.
    vel_grid = np.asarray(velocity_grid_kms) + sys_v
    expanded: list[Candidate] = []
    for c in mom0_cands:
        for v in vel_grid:
            expanded.append(
                Candidate(
                    ra_deg=c.ra_deg,
                    dec_deg=c.dec_deg,
                    vel_kms=float(v),
                    score=c.score,
                    source="mom0",
                )
            )

    # -- MTB candidates --
    mtb = mtb_candidates(cube, **(mtb_kwargs or {}))

    union = dedupe_candidates(expanded + mtb, eps_arcsec=eps_arcsec)
    return pd.DataFrame(
        [
            {
                "ra_deg": c.ra_deg,
                "dec_deg": c.dec_deg,
                "vel_kms": c.vel_kms,
                "score_seed": c.score,
                "source": c.source,
            }
            for c in union
        ]
    )


def _mom0_minima_from_array(
    mom0: np.ndarray,
    cube: Cube,
    *,
    footprint_arcsec: float = 30.0,
    relative_depth: float = 0.5,
) -> list[Candidate]:
    pix_arcsec = cube.pixel_scale_arcsec
    footprint_pix = max(3, int(round(footprint_arcsec / pix_arcsec)))
    nan_mask = np.isnan(mom0)
    safe = np.where(nan_mask, np.nanmax(mom0) + 1.0, mom0)
    local_min = minimum_filter(safe, size=footprint_pix)
    is_min = (safe == local_min) & ~nan_mask
    p95 = float(np.nanpercentile(mom0, 95))
    if p95 <= 0 or not np.isfinite(p95):
        return []
    is_dark = mom0 < relative_depth * p95
    yy, xx = np.where(is_min & is_dark)
    if xx.size == 0:
        return []
    ra_arr, dec_arr = pix_to_world(cube, xx.astype(float), yy.astype(float))
    scores = 1.0 - mom0[yy, xx] / p95
    return [
        Candidate(
            ra_deg=float(ra_arr[i]),
            dec_deg=float(dec_arr[i]),
            vel_kms=float("nan"),
            score=float(scores[i]),
            source="mom0",
        )
        for i in range(xx.size)
    ]
