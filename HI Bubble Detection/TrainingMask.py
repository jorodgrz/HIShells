#!/usr/bin/env python3
# TrainingData.py — No-CLI version with celestial-only WCS + robust world→pixel.

import os, math
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord

# ========= CONFIG (edit these) =========
FITS_PATH    = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/NGC_2403_NA_MOM0_THINGS.FITS"
TABLE7_PATH  = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/table7.dat"
TABLE2_PATH  = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/table2.dat"
GALAXY       = "NGC 2403"
OUTDIR       = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/masks_out"
RIM_WIDTH_PX = 3.0
DEBUG        = True  # set False after it works
# ======================================

# Fixed-width specs
COLS7 = [(0,11),(12,15),(16,18),(19,21),(22,26),(27,28),(28,30),(31,33),(34,38),
         (39,43),(44,45),(46,50),(51,53),(54,57),(58,61),(62,66),(67,71),(72,75),(76,80),(81,85)]
N7 = ['Name','Seq','RAh','RAm','RAs','DEsign','DEd','DEm','DEs','HV','Type','Diameter',
      'Vexp','PA','AxialRatio','Rgal','nHI','tkin','logE','logMHI']

COLS2 = [(0,11),(12,21),(22,24),(25,27),(28,32),(33,34),(34,36),(37,39),(40,44),
         (45,55),(56,60),(61,63),(64,67),(68,72),(73,78),(80,84),(85,88)]
N2 = ['Name','OName','RAh','RAm','RAs','DEsign','DEd','DEm','DEs','Type','Dist','Incl','PA',
      'MHI','logSFR','logD25','Res']

def read_tables(t7_path, t2_path, galaxy):
    df7 = pd.read_fwf(t7_path, colspecs=COLS7, names=N7, na_values=['-', '—', '...', 'NaN'])
    df7 = df7[df7['Name'].astype(str).str.strip() == galaxy].copy()

    df2 = pd.read_fwf(t2_path, colspecs=COLS2, names=N2, na_values=['-', '—', '...', 'NaN'])
    row = df2[df2['Name'].astype(str).str.strip() == galaxy]
    if row.empty:
        raise ValueError(f"No distance row found for galaxy '{galaxy}' in table2.dat")
    dist_val = row['Dist'].values[0]
    dist_mpc = float(str(dist_val).strip())
    if not np.isfinite(dist_mpc) or dist_mpc <= 0:
        raise ValueError(f"Invalid distance for '{galaxy}': {dist_val!r}")
    return df7, dist_mpc

def row_to_deg(row):
    ra_str  = f"{int(row['RAh'])}h{int(row['RAm'])}m{float(row['RAs']):.3f}s"
    ded = float(row['DEd'])
    sgn = '-' if ded < 0 or str(row.get('DEsign','')).strip() == '-' else '+'
    dec_str = f"{sgn}{int(abs(ded))}d{int(row['DEm'])}m{float(row['DEs']):.3f}s"
    c = SkyCoord(ra=ra_str, dec=dec_str, frame='icrs')
    return c.ra.deg, c.dec.deg

def arcsec_per_pix_from_wcs(wcs):
    scales_deg = proj_plane_pixel_scales(wcs)  # deg/pix
    try:
        scales_deg = scales_deg.value
    except AttributeError:
        pass
    scales_arcsec = np.array(scales_deg) * 3600.0  # "/pix
    return float(np.sqrt(scales_arcsec[0] * scales_arcsec[1]))

def rasterize_ellipse(mask, cx, cy, a_pix, b_pix, theta_rad, value):
    from astropy.modeling.functional_models import Ellipse2D
    H, W = mask.shape
    y, x = np.mgrid[0:H, 0:W]
    model = Ellipse2D(amplitude=1.0, x_0=cx, y_0=cy, a=a_pix, b=b_pix, theta=theta_rad)
    img = model(x, y)
    mask[img >= 0.5] = value

def make_rim(shape, cx, cy, a_pix, b_pix, theta_rad, rim_px):
    inner = np.zeros(shape, dtype=np.uint8)
    outer = np.zeros(shape, dtype=np.uint8)
    rasterize_ellipse(outer, cx, cy, a_pix, b_pix, theta_rad, 1)
    rasterize_ellipse(inner, cx, cy, max(1.0, a_pix - rim_px), max(1.0, b_pix - rim_px), theta_rad, 1)
    return (outer == 1) & (inner == 0)

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load FITS + build celestial-only WCS
    with fits.open(FITS_PATH) as hdul:
        data_hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim >= 2)
        data = np.squeeze(data_hdu.data)
        header = data_hdu.header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image; got {data.shape}")

    wcs_full = WCS(header)
    wcs = wcs_full.celestial if hasattr(wcs_full, "celestial") else WCS(header, naxis=2)

    H, W = data.shape
    aspix = arcsec_per_pix_from_wcs(wcs)

    # Load tables + distance (Mpc)
    df7, dist_mpc = read_tables(TABLE7_PATH, TABLE2_PATH, GALAXY)
    if DEBUG:
        print(f"[INFO] {GALAXY}: shells={len(df7)}, dist={dist_mpc:.3f} Mpc, aspix={aspix:.3f}\"/pix")

    mask_region = np.zeros((H, W), dtype=np.uint8)
    mask_rim    = np.zeros((H, W), dtype=np.uint8)
    instances   = np.zeros((H, W), dtype=np.int32)

    derived = []
    label_id = 1
    skipped = dict(invalid_diameter=0, invalid_axes=0, wcs_error=0, draw_error=0)

    for _, row in df7.iterrows():
        try:
            diameter_pc = pd.to_numeric(row['Diameter'], errors='coerce')
            if not np.isfinite(diameter_pc) or diameter_pc <= 0:
                skipped['invalid_diameter'] += 1
                continue

            major_arcsec = (diameter_pc / (dist_mpc * 1e6)) * 206265.0

            axial_ratio = pd.to_numeric(row.get('AxialRatio', 1.0), errors='coerce')
            axial_ratio = 1.0 if not np.isfinite(axial_ratio) or axial_ratio <= 0 else float(axial_ratio)
            minor_arcsec = major_arcsec * axial_ratio

            pa_deg = pd.to_numeric(row.get('PA', 0.0), errors='coerce')
            pa_deg = 0.0 if not np.isfinite(pa_deg) else float(pa_deg)

            a_pix = (major_arcsec / 2.0) / aspix
            b_pix = (minor_arcsec / 2.0) / aspix
            if not (np.isfinite(a_pix) and np.isfinite(b_pix)) or a_pix <= 0 or b_pix <= 0:
                skipped['invalid_axes'] += 1
                continue

            try:
                ra_deg, dec_deg = row_to_deg(row)
                x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)  # origin=0 for numpy
            except Exception:
                skipped['wcs_error'] += 1
                continue

            theta = math.radians(90.0 - pa_deg)  # PA (E of N) → Ellipse2D theta

            try:
                rasterize_ellipse(mask_region, x, y, a_pix, b_pix, theta, 1)
                rim = make_rim(mask_region.shape, x, y, a_pix, b_pix, theta, RIM_WIDTH_PX)
                mask_rim[rim] = 1

                temp = np.zeros_like(mask_region, dtype=np.uint8)
                rasterize_ellipse(temp, x, y, a_pix, b_pix, theta, 1)
                instances[(temp == 1) & (instances == 0)] = label_id
            except Exception:
                skipped['draw_error'] += 1
                continue

            derived.append({
                "id": int(row['Seq']) if pd.notnull(row['Seq']) else label_id,
                "ra_deg": ra_deg, "dec_deg": dec_deg,
                "major_arcsec": major_arcsec, "minor_arcsec": minor_arcsec, "pa_deg": pa_deg
            })
            label_id += 1

        except Exception:
            continue

    # Write outputs
    hdr = header.copy(); hdr["BUNIT"] = "mask"
    fits.PrimaryHDU(mask_region, hdr).writeto(os.path.join(OUTDIR, "mask_region.fits"), overwrite=True)
    fits.PrimaryHDU(mask_rim,    hdr).writeto(os.path.join(OUTDIR, "mask_rim.fits"),    overwrite=True)
    fits.PrimaryHDU(instances,   hdr).writeto(os.path.join(OUTDIR, "instances.fits"),   overwrite=True)

    if derived:
        pd.DataFrame(derived).to_csv(os.path.join(OUTDIR, "derived_catalog.csv"), index=False)

    placed = label_id - 1
    print(f"Done. Placed {placed} shells. Output → {OUTDIR}")
    if DEBUG:
        print(f"[DEBUG] Skips: {skipped}")
        if placed == 0:
            print(df7.head(3).to_string(index=False))

if __name__ == "__main__":
    main()