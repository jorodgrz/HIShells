"""Per-cube window normalisation and negative sampling.

Two responsibilities (plan \u00a72.1 and \u00a72.2):

* :func:`normalize_window` -- per-window scale-free normalisation:
  subtract the per-window median and divide by the per-cube
  ``sigma_rms``. This is what feeds the CNN in
  :class:`hishells.data.ShellWindowDataset`.

* :func:`sample_negatives` -- generate non-shell candidate sightlines
  for one cube. Two-source per plan \u00a72.2: ~75% "hard" negatives
  (offset by \u2265 ``2 \u00d7 d_arcsec`` and \u2265 ``Vexp`` from any positive)
  and ~25% "easy" negatives (random sightlines through emission-free
  regions, identified via a 5-sigma-clipped intensity mask). The
  caller supplies the ratio (default 5 negatives per positive).

The sampler emits dicts with the same keys the
:class:`hishells.pvcut` API consumes, so a downstream PyTorch dataset
can call :func:`hishells.pvcut.extract_window_for_hole` on either
positives or negatives uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .cubes import Cube, moment0, pix_to_world, sigma_rms, world_to_pix


# ---------------------------------------------------------------------------
# Per-window normalisation
# ---------------------------------------------------------------------------


def normalize_window(
    window: np.ndarray,
    cube_sigma: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-window normalisation per plan \u00a72.1.

    ``(window - median(window)) / max(cube_sigma, eps)``. Returns a
    float32 array of the same shape.
    """

    if cube_sigma is None or not np.isfinite(cube_sigma) or cube_sigma <= 0:
        cube_sigma = float(eps)
    med = np.nanmedian(window)
    out = (window - med) / max(float(cube_sigma), eps)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Emission-free mask (for "easy" negatives)
# ---------------------------------------------------------------------------


def emission_free_mask(
    cube: Cube,
    *,
    sigma_threshold: float = 5.0,
    cube_sigma: float | None = None,
) -> np.ndarray:
    """Boolean ``(n_dec, n_ra)`` mask of pixels that are emission-free.

    True where the moment-0 magnitude is below ``sigma_threshold`` times
    the per-cube ``sigma_rms`` * sqrt(n_chan) (the noise level after
    summing along the spectral axis). Used to source "easy" negatives
    that don't accidentally land on a real shell that B11 missed.
    """

    if cube_sigma is None:
        cube_sigma = sigma_rms(cube)
    m0 = moment0(cube)
    threshold = sigma_threshold * float(cube_sigma) * np.sqrt(cube.n_chan)
    return np.abs(m0) < threshold


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegSampleConfig:
    """Knobs for :func:`sample_negatives`.

    Defaults match plan \u00a72.2: 5 negatives per positive, 75% hard
    negatives, exclude any candidate that lands within
    ``min_sep_factor \u00d7 d_arcsec`` and ``min_vel_sep_factor \u00d7
    Vexp_kms`` of an existing positive.
    """

    ratio: float = 5.0
    hard_frac: float = 0.75
    min_sep_factor: float = 2.0
    min_vel_sep_factor: float = 1.0
    sigma_threshold: float = 5.0
    rng_seed: int | None = None


def sample_negatives(
    cube: Cube,
    positives: pd.DataFrame,
    config: NegSampleConfig | None = None,
    *,
    cube_sigma: float | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of negative sightlines for ``cube``.

    Columns match :class:`hishells.pvcut.extract_window_for_hole`'s
    ``hole`` dict spec: ``ra_deg``, ``dec_deg``, ``vel_helio_kms``,
    ``pa_deg``, ``diameter_arcsec``, ``vexp_kms``, ``sigma_gas_kms``.

    Negatives inherit ``diameter_arcsec`` / ``vexp_kms`` /
    ``sigma_gas_kms`` from a randomly-chosen positive in the same
    galaxy; that keeps the window extents matched to the positive
    distribution so the model can't separate classes by window scale.
    """

    cfg = config or NegSampleConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    if cube_sigma is None:
        cube_sigma = sigma_rms(cube)

    # Restrict to positives belonging to this cube. Caller usually
    # filters already, but be defensive.
    pos = positives[positives["galaxy_id"] == cube.galaxy_id].reset_index(drop=True)
    if len(pos) == 0:
        return pd.DataFrame(columns=list(positives.columns))

    n_total_neg = int(round(cfg.ratio * len(pos)))
    n_hard = int(round(cfg.hard_frac * n_total_neg))
    n_easy = n_total_neg - n_hard

    pos_xy = world_to_pix(cube, pos["ra_deg"].values, pos["dec_deg"].values)
    pos_x = np.asarray(pos_xy[0])
    pos_y = np.asarray(pos_xy[1])
    pos_v = pos["vel_helio_kms"].values.astype(float)
    pos_d_pix = (
        pos["diameter_arcsec"].values.astype(float) / cube.pixel_scale_arcsec
    )
    pos_vexp = pos["vexp_kms"].values.astype(float)
    # Substitute sigma_gas for type-1 holes (Vexp NaN/0)
    sigma_gas_arr = pos["sigma_gas_kms"].values.astype(float)
    pos_vexp_eff = np.where(
        np.isfinite(pos_vexp) & (pos_vexp > 0), pos_vexp, sigma_gas_arr
    )

    nx_max = cube.shape[2]
    ny_max = cube.shape[1]
    nz_max = cube.n_chan

    rows: list[dict] = []

    def _too_close_to_positive(x: float, y: float, v: float, ref_d_pix: float) -> bool:
        sep_pix = np.hypot(pos_x - x, pos_y - y)
        sep_v = np.abs(pos_v - v)
        forbid_pix = cfg.min_sep_factor * np.maximum(pos_d_pix, ref_d_pix)
        forbid_v = cfg.min_vel_sep_factor * pos_vexp_eff
        return bool(np.any((sep_pix < forbid_pix) & (sep_v < forbid_v)))

    # ----- Hard negatives -----
    attempts = 0
    while len(rows) < n_hard and attempts < n_hard * 50:
        attempts += 1
        # Inherit window-extent params from a random positive.
        ref = pos.iloc[int(rng.integers(0, len(pos)))]
        d_arcsec = float(ref["diameter_arcsec"])
        d_pix = d_arcsec / cube.pixel_scale_arcsec
        # Pixel inside the cube footprint with margin equal to the
        # candidate's own diameter (so the window doesn't run off the
        # edge during extraction).
        margin = max(int(np.ceil(d_pix)), 8)
        x = float(rng.integers(margin, max(nx_max - margin, margin + 1)))
        y = float(rng.integers(margin, max(ny_max - margin, margin + 1)))
        z_idx = int(rng.integers(0, nz_max))
        v = float(cube.velocity_kms[z_idx])
        if _too_close_to_positive(x, y, v, d_pix):
            continue
        ra, dec = pix_to_world(cube, np.array([x]), np.array([y]))
        rows.append(
            {
                "galaxy_id": cube.galaxy_id,
                "hole_idx": -(len(rows) + 1),
                "ra_deg": float(ra[0]),
                "dec_deg": float(dec[0]),
                "vel_helio_kms": v,
                "pa_deg": float(rng.uniform(0, 180)),
                "diameter_arcsec": d_arcsec,
                "diameter_pc": float(ref["diameter_pc"]),
                "vexp_kms": float(ref["vexp_kms"]) if np.isfinite(ref["vexp_kms"]) else 0.0,
                "sigma_gas_kms": float(ref["sigma_gas_kms"]),
                "hole_type": 0,
                "label": 0,
                "neg_kind": "hard",
            }
        )

    # ----- Easy negatives -----
    mask = emission_free_mask(cube, cube_sigma=cube_sigma, sigma_threshold=cfg.sigma_threshold)
    ys_free, xs_free = np.where(mask)
    if len(ys_free) == 0:
        # Cube is bright everywhere; degrade easy -> hard to keep counts.
        n_hard_extra = n_easy
        n_easy = 0
    else:
        n_hard_extra = 0
    attempts = 0
    while len(rows) < n_hard + n_easy and attempts < (n_easy + 1) * 50 and len(ys_free) > 0:
        attempts += 1
        idx = int(rng.integers(0, len(ys_free)))
        x = float(xs_free[idx])
        y = float(ys_free[idx])
        z_idx = int(rng.integers(0, nz_max))
        v = float(cube.velocity_kms[z_idx])
        ref = pos.iloc[int(rng.integers(0, len(pos)))]
        d_arcsec = float(ref["diameter_arcsec"])
        d_pix = d_arcsec / cube.pixel_scale_arcsec
        if _too_close_to_positive(x, y, v, d_pix):
            continue
        ra, dec = pix_to_world(cube, np.array([x]), np.array([y]))
        rows.append(
            {
                "galaxy_id": cube.galaxy_id,
                "hole_idx": -(len(rows) + 1),
                "ra_deg": float(ra[0]),
                "dec_deg": float(dec[0]),
                "vel_helio_kms": v,
                "pa_deg": float(rng.uniform(0, 180)),
                "diameter_arcsec": d_arcsec,
                "diameter_pc": float(ref["diameter_pc"]),
                "vexp_kms": float(ref["vexp_kms"]) if np.isfinite(ref["vexp_kms"]) else 0.0,
                "sigma_gas_kms": float(ref["sigma_gas_kms"]),
                "hole_type": 0,
                "label": 0,
                "neg_kind": "easy",
            }
        )

    # If the easy-mask was empty we promised to top up with extra hard.
    while n_hard_extra > 0 and attempts < n_total_neg * 50:
        attempts += 1
        ref = pos.iloc[int(rng.integers(0, len(pos)))]
        d_arcsec = float(ref["diameter_arcsec"])
        d_pix = d_arcsec / cube.pixel_scale_arcsec
        margin = max(int(np.ceil(d_pix)), 8)
        x = float(rng.integers(margin, max(nx_max - margin, margin + 1)))
        y = float(rng.integers(margin, max(ny_max - margin, margin + 1)))
        z_idx = int(rng.integers(0, nz_max))
        v = float(cube.velocity_kms[z_idx])
        if _too_close_to_positive(x, y, v, d_pix):
            continue
        ra, dec = pix_to_world(cube, np.array([x]), np.array([y]))
        rows.append(
            {
                "galaxy_id": cube.galaxy_id,
                "hole_idx": -(len(rows) + 1),
                "ra_deg": float(ra[0]),
                "dec_deg": float(dec[0]),
                "vel_helio_kms": v,
                "pa_deg": float(rng.uniform(0, 180)),
                "diameter_arcsec": d_arcsec,
                "diameter_pc": float(ref["diameter_pc"]),
                "vexp_kms": float(ref["vexp_kms"]) if np.isfinite(ref["vexp_kms"]) else 0.0,
                "sigma_gas_kms": float(ref["sigma_gas_kms"]),
                "hole_type": 0,
                "label": 0,
                "neg_kind": "hard",
            }
        )
        n_hard_extra -= 1

    return pd.DataFrame(rows)


def label_positives(positives: pd.DataFrame) -> pd.DataFrame:
    """Tag a positive-row DataFrame with ``label=1`` and ``neg_kind=NA``.

    Convenience so the rest of the pipeline can concatenate positives
    and negatives into a single DataFrame.
    """

    out = positives.copy()
    out["label"] = 1
    out["neg_kind"] = pd.NA
    return out


def build_window_table(
    catalog: pd.DataFrame,
    cubes_by_galaxy: dict[str, Cube],
    config: NegSampleConfig | None = None,
    *,
    cube_sigmas: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Concatenate positives + sampled negatives across all cubes.

    Used by :class:`hishells.data.ShellWindowDataset` to enumerate the
    full per-fold window list before training. ``cubes_by_galaxy`` is a
    ``{galaxy_id: Cube}`` dict; ``cube_sigmas`` (optional) lets the
    caller pass cached sigma_rms values to avoid re-computing.
    """

    parts: list[pd.DataFrame] = [label_positives(catalog)]
    for gid, cube in cubes_by_galaxy.items():
        sub = catalog[catalog["galaxy_id"] == gid]
        if len(sub) == 0:
            continue
        sigma = (cube_sigmas or {}).get(gid)
        negs = sample_negatives(cube, sub, config, cube_sigma=sigma)
        if len(negs) > 0:
            parts.append(negs)
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out
