# src/utils/wcs_tools.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from astropy.constants import c

def open_cube(cube_path: str | Path):
    """
    Open a FITS spectral cube. Returns (cube, header_of_data, wcs, hdulist).
    Expects data in (V, Y, X) or (Z, Y, X) order.
    """
    p = Path(cube_path)
    hdul = fits.open(p)
    # Pick first HDU with data
    hdu = None
    for h in hdul:
        if h.data is not None:
            hdu = h
            break
    if hdu is None:
        raise ValueError(f"No image data in {p}")
    data = np.asarray(hdu.data, dtype=np.float32)
    # Squeeze any singleton axes
    data = np.squeeze(data)
    # Heuristic: ensure (V, Y, X)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube (found shape {data.shape})")
    hdr = hdu.header
    wcs = WCS(hdr)
    return data, hdr, wcs, hdul

def pixel_scales_arcsec(wcs: WCS):
    """
    Return (x_scale_arcsec_per_pix, y_scale_arcsec_per_pix) in the image plane.
    """
    # proj_plane_pixel_scales returns degrees for celestial axes
    scales = proj_plane_pixel_scales(wcs)  # degrees per pixel for each axis
    # We want image plane (X,Y) scales; take last two axes
    if len(scales) >= 2:
        x_deg = float(scales[-1])
        y_deg = float(scales[-2])
    else:
        # Fallback
        x_deg = y_deg = 1.0 / 3600.0
    return x_deg * 3600.0, y_deg * 3600.0  # arcsec/pix

def radec_to_xy(wcs: WCS, ra_deg: float, dec_deg: float):
    """
    Convert RA/Dec (deg) to pixel (x,y) in the image plane.
    """
    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg, 0)[:2] if wcs.naxis >= 2 else wcs.world_to_pixel(ra_deg, dec_deg)[:2]
    return float(x), float(y)

def velocity_axis_kms(hdr):
    """
    Construct the spectral axis in km/s.
    Handles VELO axis directly; if FREQ axis present and RESTFRQ/RESTFRQZ available,
    converts to (radio) velocity: v = c * (1 - nu/nu0).
    """
    # Index of spectral axis: assume axis 0 is spectral after we squeezed to (V,Y,X)
    n = int(hdr.get("NAXIS1", 0))
    # But NAXIS keywords are per axis; we'll read CRVAL3/2/1 by looking for spectral CTYPE
    # Build via WCS to be safe:
    w = WCS(hdr)
    # Prepare pixel coordinates along spectral axis (0..NV-1)
    # Find spectral axis index in WCS:
    spec_axis = None
    for i in range(w.naxis):
        ctype = (hdr.get(f"CTYPE{i+1}", "") or "").upper()
        if "VELO" in ctype or "VRAD" in ctype or "FREQ" in ctype or "WAVE" in ctype:
            spec_axis = i
            break
    if spec_axis is None:
        # Fallback: assume axis 0
        spec_axis = 0
    nv = int(hdr.get(f"NAXIS{spec_axis+1}", 0))
    pix = np.arange(nv, dtype=float)

    # Build full pixel grid only along spectral axis:
    # WCS expects N-dim; create zeros for others and fill spec coordinate
    pix_coords = [np.zeros(nv, dtype=float)] * w.naxis
    pix_coords[spec_axis] = pix
    world = w.all_pix2world(*pix_coords, 0)
    # Extract spectral world
    spec_world = world[spec_axis]

    cunit = (hdr.get(f"CUNIT{spec_axis+1}", "") or "").lower()

    # If already velocity-like
    if "m/s" in cunit or "ms-1" in cunit or "km/s" in cunit or "kms-1" in cunit:
        q = u.Quantity(spec_world, u.Unit(cunit))
        return q.to(u.km/u.s).value.astype(np.float32)

    # Frequency axis to radio velocity
    if "freq" in (hdr.get(f"CTYPE{spec_axis+1}", "") or "").lower() or "hz" in cunit:
        nu = u.Quantity(spec_world, u.Hz)
        # Rest frequency from header (various keywords used in the wild)
        nu0 = None
        for k in ("RESTFRQ", "RESTFREQ", "RESTFR", "RESTF"):
            if k in hdr:
                try:
                    nu0 = (hdr[k] * u.Hz)
                    break
                except Exception:
                    pass
        if nu0 is None:
            # As a last resort, assume already in velocity pixels (not ideal)
            # Build linear axis using WCS CD/CRVAL:
            crval = hdr.get(f"CRVAL{spec_axis+1}", 0.0)
            cdelt = hdr.get(f"CDELT{spec_axis+1}", 1.0)
            v = (crval + cdelt * (pix - hdr.get(f"CRPIX{spec_axis+1}", 1.0) + 1.0)) * u.m/u.s
            return v.to(u.km/u.s).value.astype(np.float32)
        v = (c * (1 - (nu/nu0))).to(u.km/u.s)
        return v.value.astype(np.float32)

    # Wavelength axis or unknown: do a linear convert assuming units convertible to velocity (rare)
    try:
        q = u.Quantity(spec_world, u.Unit(cunit))
        return q.to(u.km/u.s).value.astype(np.float32)
    except Exception:
        # Final fallback: linear pixels as km/s step=1
        return (pix - pix.mean()).astype(np.float32)

def unit_vectors_for_pa(pa_deg: float, convention: str = "astro"):
    """
    Return two orthonormal 2D unit vectors in image pixel space:
      - dvec_major: along the PA direction
      - nvec_major: +90° CCW from dvec_major
    Conventions:
      'astro': PA measured from North (+Y) through East (+X).
               PA=0 -> +Y; PA=90 -> +X.
               dvec = (sin(PA), cos(PA))
      'image': PA measured from +X toward +Y.
               PA=0 -> +X; PA=90 -> +Y.
               dvec = (cos(PA), sin(PA))
    """
    th = np.deg2rad(float(pa_deg))
    if convention.lower() == "astro":
        dx, dy = np.sin(th), np.cos(th)
    else:  # 'image'
        dx, dy = np.cos(th), np.sin(th)
    # +90° CCW
    nx, ny = -dy, dx
    # Normalize (just in case)
    d = np.hypot(dx, dy); n = np.hypot(nx, ny)
    return (dx/d, dy/d), (nx/n, ny/n)