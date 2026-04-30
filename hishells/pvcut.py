"""Position-velocity (p-v) cut window extraction.

Given an HI cube and a candidate sightline ``(ra, dec, vel, pa,
diameter, vexp)``, produce a fixed-size 2-D window with the position
axis along the sightline (rotated by ``pa``) and the velocity axis
along the cube's spectral direction. The window is the diagnostic
representation a CNN classifies as shell vs. non-shell.

Conventions per ``plan.md`` \u00a72.1:

* ``window_pix = 96`` square output by default. Verified in
  ``notebooks/04_window_inspection.ipynb``: 64 px covered only ~91.7%
  of B11 type-{2,3} holes at native sampling, below the \u00a712 \u226595%
  gate; 96 px clears it. The CNN architecture in \u00a73.1 is fully
  convolutional with global average pooling, so this is a runtime
  cost change only -- weight count is unaffected.
* Position extent ``= 2 * d_arcsec`` (``d`` from B11), centered.
* Velocity extent ``= max(2 * V_exp, 20)`` km/s for hole types 2/3;
  for type 1 (no measurable Vexp) the fallback is
  ``2 * sigma_gas`` from :data:`hishells.catalog.SIGMA_GAS_KMS_BY_STEM`.
* PA is treated north-of-east-of-positive-RA-axis (B11's convention,
  matching standard astronomical position angle).
* Resampling uses :func:`scipy.ndimage.map_coordinates` (cubic
  spline, ``cval=0.0`` outside the cube).

The signature is intentionally stateless / numpy-only so it fits
inside a ``torch.utils.data.Dataset.__getitem__`` cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from .cubes import Cube, world_to_pix


# ---------------------------------------------------------------------------
# Per-hole window-extent helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowExtent:
    """Physical span of one window in arcsec / km/s.

    ``pos_extent_arcsec`` is the half-extent of the position axis (the
    full window spans ``[-pos_extent, +pos_extent]``); same for
    ``vel_extent_kms`` on the velocity axis.
    """

    pos_extent_arcsec: float
    vel_extent_kms: float


def window_extent_for_hole(
    diameter_arcsec: float,
    vexp_kms: float | None,
    sigma_gas_kms: float = 10.0,
    *,
    pos_factor: float = 2.0,
    vel_factor: float = 2.0,
    vel_floor_kms: float = 20.0,
) -> WindowExtent:
    """Compute (pos_extent, vel_extent) for a single B11 hole row.

    ``vexp_kms`` may be ``None`` or ``NaN`` for type-1 holes; in that
    case the velocity extent falls back to ``vel_factor *
    sigma_gas_kms`` per plan \u00a72.1.
    """

    pos = float(pos_factor) * float(diameter_arcsec)
    if vexp_kms is None or not np.isfinite(vexp_kms) or vexp_kms <= 0:
        vel = float(vel_factor) * float(sigma_gas_kms)
    else:
        vel = max(float(vel_factor) * float(vexp_kms), float(vel_floor_kms))
    return WindowExtent(pos_extent_arcsec=pos, vel_extent_kms=vel)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------


def _sightline_pix(
    cube: Cube, ra_deg: float, dec_deg: float, pa_deg: float, n_pos: int, half_extent_pix: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates ``(x_pix, y_pix)`` along a 1-D sightline.

    The sightline is centered on ``(ra, dec)``, has total angular span
    ``2 * half_extent_pix`` pixels, contains ``n_pos`` samples, and
    points along the position-angle direction ``pa_deg`` (north of
    east, i.e. measured counter-clockwise from the +RA axis).
    """

    cx, cy = world_to_pix(cube, np.array([ra_deg]), np.array([dec_deg]))
    cx = float(cx[0])
    cy = float(cy[0])

    # PA is "north through east", i.e. measured from +Dec (numpy axis 1)
    # toward +RA. Convert to a unit vector in (x, y) pixel space, where
    # x = RA pixel, y = Dec pixel.
    theta = np.deg2rad(float(pa_deg))
    dx = np.sin(theta)
    dy = np.cos(theta)

    s = np.linspace(-half_extent_pix, half_extent_pix, n_pos)
    x_pix = cx + s * dx
    y_pix = cy + s * dy
    return x_pix, y_pix


def extract_window(
    cube: Cube,
    ra_deg: float,
    dec_deg: float,
    vel_kms: float,
    pa_deg: float,
    pos_extent_arcsec: float,
    vel_extent_kms: float,
    window_pix: int = 96,
    *,
    order: int = 3,
    cval: float = 0.0,
) -> np.ndarray:
    """Extract one ``(window_pix, window_pix)`` p-v cut from ``cube``.

    Parameters
    ----------
    cube
        Source data cube.
    ra_deg, dec_deg
        Sky position of the sightline center.
    vel_kms
        Velocity of the sightline center (heliocentric or LSR; must
        match the cube's spectral axis).
    pa_deg
        Position angle of the sightline (B11 convention: north of
        east of the +RA axis).
    pos_extent_arcsec, vel_extent_kms
        Half-extents of the output window in arcsec / km/s. Use
        :func:`window_extent_for_hole` for B11-driven defaults.
    window_pix
        Output array side length.
    order, cval
        Forwarded to :func:`scipy.ndimage.map_coordinates`. ``order=3``
        gives cubic-spline resampling; ``cval=0.0`` zero-pads outside
        the cube's footprint.

    Returns
    -------
    np.ndarray
        ``(window_pix, window_pix)`` float32 array. Axis 0 is the
        position axis (sightline), axis 1 is the velocity axis (and
        therefore axis 1 increases with channel index, *not*
        necessarily with km/s if the cube's spectral axis is
        descending). Downstream code that cares about absolute velocity
        ordering should consult ``cube.velocity_kms``.
    """

    half_pos_arcsec = float(pos_extent_arcsec)
    half_pos_pix = half_pos_arcsec / cube.pixel_scale_arcsec
    x_pix, y_pix = _sightline_pix(
        cube, ra_deg, dec_deg, pa_deg, window_pix, half_pos_pix
    )

    # Velocity-axis coordinates: ``window_pix`` channel-index samples
    # spanning ``[vel - vel_extent, vel + vel_extent]`` km/s.
    v_targets = np.linspace(
        vel_kms - vel_extent_kms, vel_kms + vel_extent_kms, window_pix
    )
    z_pix = np.interp(
        v_targets,
        cube.velocity_kms[::-1] if cube.velocity_kms[0] > cube.velocity_kms[-1] else cube.velocity_kms,
        np.arange(cube.n_chan)[::-1] if cube.velocity_kms[0] > cube.velocity_kms[-1] else np.arange(cube.n_chan),
    )

    # Build the (W_pos, W_vel) coordinate grid; ``map_coordinates``
    # expects ``[axis0_coords, axis1_coords, axis2_coords]``.
    pos_grid = np.broadcast_to(
        np.arange(window_pix)[:, None], (window_pix, window_pix)
    )
    vel_grid = np.broadcast_to(
        np.arange(window_pix)[None, :], (window_pix, window_pix)
    )

    # axis0 (channel) varies along the velocity (column) axis.
    z_grid = z_pix[vel_grid]
    # axis1 (Dec) and axis2 (RA) vary along the position (row) axis.
    y_grid = y_pix[pos_grid]
    x_grid = x_pix[pos_grid]

    coords = np.stack([z_grid, y_grid, x_grid], axis=0)
    win = map_coordinates(
        cube.data, coords, order=order, mode="constant", cval=cval
    )
    return np.ascontiguousarray(win, dtype=np.float32)


def extract_window_for_hole(
    cube: Cube,
    hole: dict,
    *,
    window_pix: int = 96,
    pos_factor: float = 2.0,
    vel_factor: float = 2.0,
    vel_floor_kms: float = 20.0,
) -> np.ndarray:
    """Convenience: pull all the window-extent params from a B11 row.

    ``hole`` is a mapping with the keys produced by
    :func:`hishells.catalog.load_holes`: ``ra_deg``, ``dec_deg``,
    ``vel_helio_kms``, ``pa_deg``, ``diameter_arcsec``, ``vexp_kms``,
    ``sigma_gas_kms``.
    """

    ext = window_extent_for_hole(
        diameter_arcsec=hole["diameter_arcsec"],
        vexp_kms=hole["vexp_kms"],
        sigma_gas_kms=hole.get("sigma_gas_kms", 10.0),
        pos_factor=pos_factor,
        vel_factor=vel_factor,
        vel_floor_kms=vel_floor_kms,
    )
    return extract_window(
        cube,
        ra_deg=hole["ra_deg"],
        dec_deg=hole["dec_deg"],
        vel_kms=hole["vel_helio_kms"],
        pa_deg=hole["pa_deg"] if np.isfinite(hole["pa_deg"]) else 0.0,
        pos_extent_arcsec=ext.pos_extent_arcsec,
        vel_extent_kms=ext.vel_extent_kms,
        window_pix=window_pix,
    )
