"""Mashchenko-Thilker-Braun template-matching shell finder.

Implements the 5-parameter expanding-spherical-shell template
``(R, V_exp, X0, Y0, Z0)`` from Mashchenko, Thilker & Braun 1999
(A&A 343, 352) and Mashchenko & St-Louis 2000. The template at
local offset ``(dx, dy, dv)`` from the candidate centre, with shell
radius ``R`` (px) and expansion velocity ``V_exp`` (channels), is

::

    rho_pix = sqrt(dx**2 + dy**2)
    inside  = rho_pix < R
    dv_ring = V_exp * sqrt(1 - (rho_pix / R)**2) when inside
    template[dy, dx, dv] = exp(-(|dv| - dv_ring)**2 / (2 * sigma_v**2))

with ``sigma_v`` set to one channel by default (matching the THINGS
channel width). Inside the shell at impact parameter ``rho_pix`` the
template lights up at the two ring velocities ``+/- dv_ring``;
outside the shell it's zero. Scores are Pearson correlation
``rho_xy = cov(cube_sub, template) / (sigma_cube * sigma_tmpl)``,
maximised over a small grid of ``(R, V_exp)``.

Per plan \u00a79 Row 1 the score we report per candidate is
``rho / rho_99``, where ``rho_99`` is the 99-th percentile of
candidate scores in the same fold. We expose the raw ``rho`` here
and let :mod:`hishells.eval` (or the caller) divide by ``rho_99``
when evaluating against B11.

This implementation prioritises legibility over speed; for the full
\u00a711 LOGO sweep we expect ~10-min/galaxy, which is acceptable for a
one-off baseline row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cubes import Cube, channel_width_kms, sigma_rms, world_to_pix
from ..data import CubeStore


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def shell_template(
    R_pix: float,
    Vexp_chan: float,
    n_pix: int,
    n_chan: int,
    sigma_v_chan: float = 1.0,
) -> np.ndarray:
    """Build the (n_chan, n_pix, n_pix) MTB template.

    ``n_pix`` and ``n_chan`` should each be at least
    ``ceil(2 * max(R_pix, Vexp_chan) + 1)``.
    """

    cy = (n_pix - 1) / 2.0
    cx = (n_pix - 1) / 2.0
    cz = (n_chan - 1) / 2.0

    yy, xx = np.meshgrid(
        np.arange(n_pix) - cy, np.arange(n_pix) - cx, indexing="ij"
    )
    rho = np.hypot(xx, yy)
    inside = rho < R_pix
    dv_ring = np.zeros_like(rho)
    if R_pix > 0:
        dv_ring[inside] = Vexp_chan * np.sqrt(
            np.clip(1.0 - (rho[inside] / R_pix) ** 2, 0.0, 1.0)
        )

    zs = np.arange(n_chan) - cz  # signed dv
    # Two ring velocities per pixel: +dv_ring and -dv_ring.
    tmpl = np.zeros((n_chan, n_pix, n_pix), dtype=np.float32)
    sv2 = 2.0 * float(sigma_v_chan) ** 2
    for k, dv in enumerate(zs):
        for sign in (+1.0, -1.0):
            tmpl[k] += np.exp(-(((dv - sign * dv_ring) ** 2) / sv2))
    tmpl[:, ~inside] = 0.0
    return tmpl


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------------


@dataclass
class MTBGrid:
    """Search grid for ``(R, Vexp)``.

    Defaults match B11 holes' range (R \u2208 [50 pc, 1 kpc]; Vexp \u2208
    [4, 36] km/s). The grid is expressed in *physical* units so the
    same configuration applies across cubes with different pixel
    scales / channel widths.
    """

    radii_arcsec: tuple[float, ...] = (5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0)
    vexp_kms: tuple[float, ...] = (4.0, 8.0, 12.0, 18.0, 25.0)
    sigma_v_chan: float = 1.0


def best_score_at(
    cube: Cube,
    ra_deg: float,
    dec_deg: float,
    vel_kms: float,
    grid: MTBGrid | None = None,
) -> tuple[float, float, float]:
    """Best Pearson ``rho`` over the (R, Vexp) grid at one sightline.

    Returns ``(rho, R_arcsec, Vexp_kms)`` for the maximising template.
    """

    grid = grid or MTBGrid()
    pix_arcsec = cube.pixel_scale_arcsec
    chw = channel_width_kms(cube)
    if not (np.isfinite(pix_arcsec) and np.isfinite(chw) and pix_arcsec > 0 and chw > 0):
        return 0.0, float("nan"), float("nan")

    cx, cy = world_to_pix(cube, np.array([ra_deg]), np.array([dec_deg]))
    cx_i = int(round(float(cx[0])))
    cy_i = int(round(float(cy[0])))
    cz_pix = np.interp(
        vel_kms,
        cube.velocity_kms[::-1] if cube.velocity_kms[0] > cube.velocity_kms[-1] else cube.velocity_kms,
        np.arange(cube.n_chan)[::-1] if cube.velocity_kms[0] > cube.velocity_kms[-1] else np.arange(cube.n_chan),
    )
    cz_i = int(round(float(cz_pix)))

    best = (0.0, float("nan"), float("nan"))
    for R_arcsec in grid.radii_arcsec:
        R_pix = R_arcsec / pix_arcsec
        for V_kms in grid.vexp_kms:
            V_chan = V_kms / chw
            n_pix = int(np.ceil(2 * R_pix)) + 1
            n_chan = int(np.ceil(2 * V_chan)) + 1
            if n_pix < 5 or n_chan < 3:
                continue
            # Clip the local subcube to the same shape as the template.
            x0 = cx_i - n_pix // 2
            y0 = cy_i - n_pix // 2
            z0 = cz_i - n_chan // 2
            x1 = x0 + n_pix
            y1 = y0 + n_pix
            z1 = z0 + n_chan
            if (
                x0 < 0
                or y0 < 0
                or z0 < 0
                or x1 > cube.shape[2]
                or y1 > cube.shape[1]
                or z1 > cube.shape[0]
            ):
                continue
            sub = cube.data[z0:z1, y0:y1, x0:x1]
            if sub.size == 0:
                continue
            tmpl = shell_template(R_pix, V_chan, n_pix, n_chan, grid.sigma_v_chan)
            # Negate the cube: a *cavity* is anti-correlated with the
            # bright-rim template, so we want the correlation between
            # the template and the *flux deficit*. Maximise over
            # |rho|, but keep sign so wrong-sign matches don't win.
            rho = -_pearson(sub, tmpl)
            if rho > best[0]:
                best = (rho, R_arcsec, V_kms)
    return best


def score_table(
    table: pd.DataFrame,
    cubes: CubeStore,
    *,
    grid: MTBGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every row in ``table`` with the MTB template matcher.

    Returns ``(scores, labels)`` with ``scores = rho`` per row. The
    caller is expected to normalise by ``np.percentile(scores[labels==0],
    99)`` to obtain ``rho/rho_99``; we don't bake that in so the same
    raw scores can be re-used for different normalisation choices.
    """

    grid = grid or MTBGrid()
    scores = np.zeros(len(table), dtype=np.float64)
    for i, (_, row) in enumerate(table.iterrows()):
        cube = cubes(str(row["galaxy_id"]))
        rho, _, _ = best_score_at(
            cube,
            ra_deg=float(row["ra_deg"]),
            dec_deg=float(row["dec_deg"]),
            vel_kms=float(row["vel_helio_kms"]),
            grid=grid,
        )
        scores[i] = rho
    labels = table["label"].values.astype(np.int64)
    return scores, labels


def normalise_by_rho_99(
    scores: np.ndarray, labels: np.ndarray, *, percentile: float = 99.0
) -> np.ndarray:
    """Divide ``scores`` by the ``percentile``-th percentile of the
    negative-class scores. Returns ``rho / rho_99`` per the MTB convention.
    """

    neg = scores[labels == 0]
    if neg.size == 0:
        return scores.copy()
    rho99 = float(np.percentile(neg, percentile))
    if rho99 <= 0 or not np.isfinite(rho99):
        return scores.copy()
    return scores / rho99
