"""Thin loader / helpers around THINGS HI cubes.

THINGS NA cubes are 4-D FITS files with a degenerate Stokes axis,
spectral axis in radio velocity (m/s, usually LSRK), and per-pixel
brightness in Jy/beam. This module wraps the load + axis squeeze +
unit normalisation in one place so the rest of ``hishells`` can treat
a cube as a plain ``(n_chan, n_dec, n_ra)`` numpy array + a velocity
vector in km/s + a 2-D celestial :class:`astropy.wcs.WCS` + beam
metadata.

We deliberately avoid pulling in ``spectral-cube`` as a hard
dependency in the hot path: the THINGS cubes are large enough that we
want explicit memory-map control, and ``astropy.io.fits``+``WCS`` is
sufficient. If a caller has ``spectral-cube`` installed they can still
load any cube via :func:`load_spectral_cube` for plotting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.wcs import WCS


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class Cube:
    """In-memory HI data cube with normalised metadata.

    Attributes
    ----------
    data
        ``(n_chan, n_dec, n_ra)`` float32 array of brightness
        (Jy/beam for THINGS NA cubes).
    wcs2d
        2-D celestial WCS describing the spatial axes of ``data``
        (i.e. axes 1 and 2 of the array; axis 0 is spectral).
    velocity_kms
        ``(n_chan,)`` heliocentric (or LSR, depending on the cube
        header) velocity of each channel in km/s.
    beam_bmaj_arcsec, beam_bmin_arcsec, beam_bpa_deg
        Restoring-beam major/minor FWHM (arcsec) and PA (deg).
    pixel_scale_arcsec
        Mean ``|CDELT|`` of the celestial axes, in arcsec/pixel.
    galaxy_id, path
        Bookkeeping; ``galaxy_id`` is the THINGS filename stem
        (e.g. ``"NGC_2403"``).
    """

    data: np.ndarray
    wcs2d: WCS
    velocity_kms: np.ndarray
    beam_bmaj_arcsec: float
    beam_bmin_arcsec: float
    beam_bpa_deg: float
    pixel_scale_arcsec: float
    galaxy_id: str
    path: Path

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_chan(self) -> int:
        return self.data.shape[0]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def galaxy_id_from_path(path: str | Path) -> str:
    """``"Data/THINGS/NGC_2403_NA_CUBE_THINGS.FITS" -> "NGC_2403"``."""

    name = Path(path).name
    return name.split("_NA_")[0].split("_RO_")[0]


def _ctype_unit(header: fits.Header, axis: int) -> u.Unit:
    """Best-effort ``CUNIT<axis>`` -> :class:`astropy.units.Unit`."""

    cunit = header.get(f"CUNIT{axis}", "").strip().lower()
    if cunit in ("m/s", "ms-1"):
        return u.m / u.s
    if cunit in ("km/s", "kms-1"):
        return u.km / u.s
    # THINGS spectral axis is m/s but some headers omit CUNIT.
    return u.m / u.s


def _spectral_axis_kms(header: fits.Header, n_chan: int) -> np.ndarray:
    """Build the spectral axis in km/s from the FITS WCS keywords.

    Assumes a linear axis (``CRPIX3``, ``CRVAL3``, ``CDELT3``); THINGS
    cubes satisfy this. Pixels are 1-indexed in FITS; channel ``i``
    (0-indexed) has axis value ``CRVAL3 + (i+1 - CRPIX3) * CDELT3``.
    """

    crpix = float(header.get("CRPIX3", 1.0))
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    unit = _ctype_unit(header, 3)
    pix = np.arange(n_chan, dtype=np.float64)
    axis = crval + (pix + 1.0 - crpix) * cdelt
    return (axis * unit).to(u.km / u.s).value.astype(np.float64)


_AIPS_BEAM_RE = __import__("re").compile(
    r"BMAJ\s*=\s*([0-9eE+\-.]+)\s+BMIN\s*=\s*([0-9eE+\-.]+)\s+BPA\s*=\s*([0-9eE+\-.]+)"
)


def _beam_from_header(header: fits.Header) -> tuple[float, float, float]:
    """Return ``(bmaj_arcsec, bmin_arcsec, bpa_deg)``.

    THINGS headers carry beam info as either a top-level
    ``BMAJ``/``BMIN``/``BPA`` triple (degrees) or, in the case of the
    AIPS-produced THINGS NA cubes, as a ``HISTORY`` line of the form
    ``AIPS CLEAN BMAJ=  2.43E-03 BMIN=  2.12E-03 BPA=  25.16``
    (BMAJ/BMIN in degrees, BPA in degrees). We try the keyword form
    first, then walk the HISTORY records, and fall back to NaN if
    nothing is found.
    """

    bmaj = header.get("BMAJ")
    bmin = header.get("BMIN")
    bpa = header.get("BPA")
    if bmaj is not None and bmin is not None and bpa is not None:
        return float(bmaj) * 3600.0, float(bmin) * 3600.0, float(bpa)

    # Walk HISTORY for the AIPS beam line; the most-recent line wins.
    last = None
    for h in header.get("HISTORY", []):
        m = _AIPS_BEAM_RE.search(str(h))
        if m:
            last = m
    if last is not None:
        bmaj_deg, bmin_deg, bpa_deg = (float(x) for x in last.groups())
        return bmaj_deg * 3600.0, bmin_deg * 3600.0, bpa_deg

    return float("nan"), float("nan"), float("nan")


def _pixel_scale_arcsec(wcs2d: WCS) -> float:
    """Mean of ``|CDELT|`` along the celestial axes, in arcsec/pix."""

    try:
        cdelt = wcs2d.proj_plane_pixel_scales()
        return float(np.mean([c.to(u.arcsec).value for c in cdelt]))
    except Exception:
        # Fallback for headers without CD matrix metadata.
        sx = abs(float(wcs2d.wcs.cdelt[0])) * 3600.0
        sy = abs(float(wcs2d.wcs.cdelt[1])) * 3600.0
        return 0.5 * (sx + sy)


def _squeeze_stokes(data: np.ndarray) -> np.ndarray:
    """Drop a leading degenerate Stokes axis if present."""

    if data.ndim == 4:
        if data.shape[0] != 1:
            raise ValueError(
                f"Expected degenerate Stokes axis, got shape {data.shape}"
            )
        return data[0]
    if data.ndim != 3:
        raise ValueError(f"Cube must be 3-D or 4-D; got shape {data.shape}")
    return data


def load_cube(path: str | Path, *, dtype: type = np.float32) -> Cube:
    """Load a THINGS-style FITS cube into a :class:`Cube`.

    Memory-mapped via ``astropy.io.fits`` (``memmap=True``); the
    returned ``data`` is a copy cast to ``dtype`` so the caller can
    close the HDU without keeping the file open.
    """

    p = Path(path)
    with fits.open(p, memmap=True) as hdul:
        hdu = hdul[0]
        header = hdu.header
        raw = np.asarray(hdu.data)
    data = np.ascontiguousarray(_squeeze_stokes(raw)).astype(dtype, copy=False)

    full_wcs = WCS(header)
    # 2-D celestial WCS for spatial-only operations (overlays etc.).
    wcs2d = full_wcs.celestial

    velocity_kms = _spectral_axis_kms(header, data.shape[0])
    bmaj, bmin, bpa = _beam_from_header(header)
    pix = _pixel_scale_arcsec(wcs2d)

    return Cube(
        data=data,
        wcs2d=wcs2d,
        velocity_kms=velocity_kms,
        beam_bmaj_arcsec=bmaj,
        beam_bmin_arcsec=bmin,
        beam_bpa_deg=bpa,
        pixel_scale_arcsec=pix,
        galaxy_id=galaxy_id_from_path(p),
        path=p,
    )


def load_spectral_cube(path: str | Path):
    """Convenience wrapper returning a :class:`spectral_cube.SpectralCube`.

    Used by notebooks / plotting; not on any hot path.
    """

    from spectral_cube import SpectralCube  # type: ignore

    sc = SpectralCube.read(str(path))
    if sc.ndim == 4:
        sc = sc[0]
    return sc.with_spectral_unit(u.km / u.s, velocity_convention="radio")


# ---------------------------------------------------------------------------
# Per-cube derived quantities
# ---------------------------------------------------------------------------


def channel_width_kms(cube: Cube) -> float:
    """Mean absolute channel spacing in km/s."""

    if cube.velocity_kms.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(cube.velocity_kms))))


def beam_pix(cube: Cube) -> float:
    """Beam major-axis FWHM in pixels."""

    if not np.isfinite(cube.beam_bmaj_arcsec):
        return float("nan")
    return float(cube.beam_bmaj_arcsec / cube.pixel_scale_arcsec)


def sigma_rms(
    cube: Cube,
    *,
    n_edge_channels: int = 5,
    sigma: float = 3.0,
) -> float:
    """Per-cube noise estimate from emission-free edge channels.

    Uses the lowest ``n_edge_channels`` and highest ``n_edge_channels``
    velocity channels (THINGS cubes are line-free at the spectral
    edges) and reports the sigma-clipped MAD-based standard deviation.
    Returned in the brightness units of the cube (Jy/beam for THINGS).
    """

    chans = np.concatenate(
        [
            np.arange(n_edge_channels),
            np.arange(cube.n_chan - n_edge_channels, cube.n_chan),
        ]
    )
    chans = np.unique(chans[(chans >= 0) & (chans < cube.n_chan)])
    sample = cube.data[chans].ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    _, _, sd = sigma_clipped_stats(sample, sigma=sigma)
    if not np.isfinite(sd) or sd == 0:
        sd = float(mad_std(sample, ignore_nan=True))
    return float(sd)


def world_to_pix(
    cube: Cube,
    ra_deg: float | np.ndarray,
    dec_deg: float | np.ndarray,
    vel_kms: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert world coordinates to (x, y[, z]) pixel coordinates.

    ``x`` corresponds to the RA axis (numpy axis 2), ``y`` to the Dec
    axis (numpy axis 1), ``z`` to the spectral axis (numpy axis 0).
    """

    x, y = cube.wcs2d.wcs_world2pix(np.atleast_1d(ra_deg), np.atleast_1d(dec_deg), 0)
    if vel_kms is None:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    v = np.atleast_1d(vel_kms).astype(np.float64)
    # ``np.interp`` requires monotonically increasing xp; THINGS cubes
    # have a descending velocity axis, so flip both the velocity vector
    # and its corresponding channel indices before interpolating.
    chans = np.arange(cube.n_chan, dtype=np.float64)
    if cube.velocity_kms[0] > cube.velocity_kms[-1]:
        z = np.interp(v, cube.velocity_kms[::-1], chans[::-1])
    else:
        z = np.interp(v, cube.velocity_kms, chans)
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        z,
    )


def pix_to_world(
    cube: Cube,
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`world_to_pix`."""

    ra, dec = cube.wcs2d.wcs_pix2world(np.atleast_1d(x), np.atleast_1d(y), 0)
    if z is None:
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)
    z_arr = np.atleast_1d(z).astype(np.float64)
    chans = np.arange(cube.n_chan, dtype=np.float64)
    # Channel index is monotonically increasing; just look up.
    vel = np.interp(z_arr, chans, cube.velocity_kms)
    return (
        np.asarray(ra, dtype=np.float64),
        np.asarray(dec, dtype=np.float64),
        vel,
    )


def moment0(cube: Cube) -> np.ndarray:
    """Sum the cube along the spectral axis.

    Returns a 2-D ``(n_dec, n_ra)`` array in (Jy/beam * channel) units.
    NaNs in the cube are treated as zero.
    """

    return np.nansum(cube.data, axis=0)


# ---------------------------------------------------------------------------
# Cube-stats cache (results/cube_stats.json)
# ---------------------------------------------------------------------------


def cube_stats(
    cube: Cube,
    *,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Compute (and optionally cache) per-cube derived stats.

    The dict has keys ``galaxy_id``, ``n_chan``, ``n_pix_dec``,
    ``n_pix_ra``, ``pixel_scale_arcsec``, ``channel_width_kms``,
    ``beam_bmaj_arcsec``, ``beam_bmin_arcsec``, ``beam_pix``,
    ``sigma_rms``, ``vel_min_kms``, ``vel_max_kms``.
    """

    cache: dict[str, dict[str, Any]] = {}
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists() and not refresh:
            cache = json.loads(cp.read_text())
            if cube.galaxy_id in cache:
                return cache[cube.galaxy_id]

    stats = {
        "galaxy_id": cube.galaxy_id,
        "n_chan": int(cube.n_chan),
        "n_pix_dec": int(cube.shape[1]),
        "n_pix_ra": int(cube.shape[2]),
        "pixel_scale_arcsec": cube.pixel_scale_arcsec,
        "channel_width_kms": channel_width_kms(cube),
        "beam_bmaj_arcsec": cube.beam_bmaj_arcsec,
        "beam_bmin_arcsec": cube.beam_bmin_arcsec,
        "beam_pix": beam_pix(cube),
        "sigma_rms": sigma_rms(cube),
        "vel_min_kms": float(np.nanmin(cube.velocity_kms)),
        "vel_max_kms": float(np.nanmax(cube.velocity_kms)),
    }

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cache[cube.galaxy_id] = stats
        cp.write_text(json.dumps(cache, indent=2, sort_keys=True))

    return stats
