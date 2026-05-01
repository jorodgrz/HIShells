# src/utils/wcs_tools.py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# ---- locate axes in header ----
def _ctype_list(hdr):
    n = hdr.get("NAXIS", 0)
    return [str(hdr.get(f"CTYPE{i}", "")).upper() for i in range(1, n+1)]

def _is_spec_ctype(ct: str) -> bool:
    ct = ct.upper()
    # cover common velocity/frequency/wavelength names & legacy labels
    keys = ("VELO", "VRAD", "VOPT", "FELO", "FREQ", "WAVE", "AWAV", "ZOPT")
    return any(k in ct for k in keys)

def _find_axis_numbers(hdr):
    """
    Return 1-based FITS axis numbers for RA, DEC, and SPECTRAL.
    """
    ctypes = _ctype_list(hdr)
    ra = dec = spec = None
    for i, ct in enumerate(ctypes, start=1):
        if ra is None and ("RA" in ct or "GLON" in ct):
            ra = i
        if dec is None and ("DEC" in ct or "GLAT" in ct):
            dec = i
        if spec is None and _is_spec_ctype(ct):
            spec = i
    return {"ra": ra, "dec": dec, "spec": spec, "naxis": len(ctypes)}

def _axisnum_to_numpy_index(axisnum, naxis):
    return None if axisnum is None else (naxis - axisnum)

# ---- public API ----
def open_cube(path):
    """
    Open a FITS cube and return (data_vyx, header, full_wCS, hdul).
    Handles 3D or 4D cubes (e.g., STOKES × V × Y × X). Extra axes are dropped by taking index 0.
    """
    hdul = fits.open(path)
    hdr = hdul[0].header
    data = hdul[0].data
    if data is None:
        for h in hdul[1:]:
            if h.data is not None:
                data = h.data
                hdr = h.header
                break
    if data is None:
        raise ValueError(f"No data found in FITS: {path}")

    wcs = WCS(hdr)
    info = _find_axis_numbers(hdr)
    nax = info["naxis"]
    arr = np.asarray(data)

    spec_idx = _axisnum_to_numpy_index(info["spec"], nax)
    ra_idx   = _axisnum_to_numpy_index(info["ra"],   nax)
    dec_idx  = _axisnum_to_numpy_index(info["dec"],  nax)

    # Fallback: squeeze & assume last 3 are (v,y,x)
    if spec_idx is None or ra_idx is None or dec_idx is None or arr.ndim < 3:
        arr = np.squeeze(arr)
        if arr.ndim < 3:
            raise ValueError(f"Cube must be at least 3D after squeeze; got {arr.shape}")
        v, y, x = arr.shape[-3], arr.shape[-2], arr.shape[-1]
        arr = arr.reshape((-1, v, y, x))[-1]  # drop leading extras
        return arr.astype(np.float32), hdr, wcs, hdul

    order = [spec_idx, dec_idx, ra_idx] + [i for i in range(arr.ndim) if i not in (spec_idx, dec_idx, ra_idx)]
    arr = np.transpose(arr, order)

    while arr.ndim > 3:
        arr = arr.take(indices=0, axis=3)

    return arr.astype(np.float32), hdr, wcs, hdul

def pixel_scales_arcsec(wcs: WCS):
    """Return (arcsec_per_x, arcsec_per_y) from celestial WCS."""
    cel = wcs.celestial
    scales = proj_plane_pixel_scales(cel)  # deg/pix (x,y)
    return float(abs(scales[0])*3600.0), float(abs(scales[1])*3600.0)

def radec_to_xy(wcs: WCS, ra_deg: float, dec_deg: float):
    """RA/Dec (deg) -> pixel (x,y) using celestial WCS."""
    cel = wcs.celestial
    x, y = cel.world_to_pixel_values(ra_deg, dec_deg)
    return float(x), float(y)

def velocity_axis_kms(hdr):
    """
    Build spectral axis in km/s regardless of which CTYPE# it lives on.
    If CUNIT is frequency, raise (safer than guessing).
    """
    info = _find_axis_numbers(hdr)
    spec_ax = info["spec"]
    if spec_ax is None:
        # Final fallback: if there are 3+ axes, try the last non-spatial axis
        ctypes = _ctype_list(hdr)
        for i, ct in enumerate(ctypes, start=1):
            if not ("RA" in ct or "DEC" in ct or "GLON" in ct or "GLAT" in ct):
                if _is_spec_ctype(ct):
                    spec_ax = i
                    break
        if spec_ax is None:
            raise ValueError("Could not find spectral axis in CTYPE keywords.")

    nv = int(hdr.get(f"NAXIS{spec_ax}"))
    crval = float(hdr.get(f"CRVAL{spec_ax}"))
    cdelt = float(hdr.get(f"CDELT{spec_ax}"))
    crpix = float(hdr.get(f"CRPIX{spec_ax}"))
    cunit = str(hdr.get(f"CUNIT{spec_ax}", "")).lower()

    idx = np.arange(nv, dtype=np.float64)
    world = crval + (idx + 1 - crpix) * cdelt  # FITS 1-indexed CRPIX

    if "m/s" in cunit or "ms-1" in cunit or "m s-1" in cunit:
        world /= 1000.0
    elif "km/s" in cunit or "kms-1" in cunit or "km s-1" in cunit:
        pass
    elif "hz" in cunit or "ghz" in cunit or "mhz" in cunit:
        raise ValueError(
            f"SPECTRAL axis is in frequency ({cunit}); convert to velocity first or extend converter."
        )
    else:
        # Unknown or blank units — assume km/s but warn
        print(f"[velocity_axis_kms] Warning: unknown CUNIT{spec_ax}='{cunit}', assuming km/s.")

    return world.astype(np.float32)

# ---- geometry helpers needed by make_pv.py ----
def line_extent_to_bounds(cx, cy, dx, dy, nx, ny):
    """
    For a line p(t)=(cx,cy)+t*(dx,dy), return (tmin, tmax) that stay in [0..nx-1]×[0..ny-1].
    """
    ts = []
    if dx != 0:
        ts += [(0 - cx)/dx, ((nx-1) - cx)/dx]
    if dy != 0:
        ts += [(0 - cy)/dy, ((ny-1) - cy)/dy]
    t_candidates = []
    for t in ts:
        x = cx + t*dx; y = cy + t*dy
        if -1 <= x <= nx and -1 <= y <= ny:
            t_candidates.append(t)
    if not t_candidates:
        return -0.0, 0.0
    tmin, tmax = min(t_candidates), max(t_candidates)
    return tmin, tmax

def rotate_xy(x, y, pa_deg: float, convention: str = "astro"):
    """
    Rotate points by +PA according to the convention above.
    Returns (xr, yr) in the same pixel frame.
    """
    th = np.deg2rad(float(pa_deg))
    c, s = np.cos(th), np.sin(th)
    if convention == "astro":
        # +PA rotates +y toward +x
        # Rotate standard (image) axes so that a positive PA brings +y toward +x
        # Equivalent to using the (sin, cos) mapping used in unit_vectors_for_pa
        xr =  x *  c + y * s
        yr = -x *  s + y * c
    else:  # "image"
        xr =  x *  c - y * s
        yr =  x *  s + y * c
    return xr, yr

def unit_vectors_for_pa(pa_deg: float, convention: str = "astro") -> tuple[tuple[float,float], tuple[float,float]]:
    """
    Returns (dvec, nvec):
      dvec: unit vector along the PA direction on the sky (projected in image pix coords)
      nvec: unit vector perpendicular to dvec (rotate CCW by +90°)
    convention:
      "astro" -> PA measured east of north (CCW from +y axis)
      "image" -> PA measured CCW from +x axis (typical math/image)
    """
    th = np.deg2rad(float(pa_deg))
    if convention == "astro":
        # 0° = +y, 90° = +x
        # A rotation by +th from +y toward +x
        dx =  np.sin(th)
        dy =  np.cos(th)
    elif convention == "image":
        # 0° = +x, 90° = +y
        dx =  np.cos(th)
        dy =  np.sin(th)
    else:
        raise ValueError(f"unknown convention: {convention}")
    # perpendicular, CCW +90°
    nx = -dy
    ny =  dx
    return (dx, dy), (nx, ny)