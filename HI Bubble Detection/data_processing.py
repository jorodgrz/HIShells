#!/usr/bin/env python3
# data_processing_masked.py — tiles + masks with HI emission "valid" mask

import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage.draw import ellipse

# ---- hard-coded paths ----
CATALOG_CSV = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/bagetakos_ngc2403.csv"
FITS_PATH   = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/NGC_2403_NA_MOM0_THINGS.FITS"
GALAXY_NAME = "NGC 2403"
OUT_DIR     = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/patches_256s192_masked"
TILE, STRIDE = 256, 192
KEEP_EMPTY_IN_GAL = 0.15    # keep some all-negative tiles inside galaxy
MIN_EMISSION_COVER = 0.05   # require ≥5% of tile covered by emission mask
MIN_DIAM_ARCSEC = 0.0       # keep all catalogue shells

# ---------------- utils ----------------
def load_fits(path):
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim >= 2)
        img = np.squeeze(hdu.data).astype(np.float32)
        hdr = hdu.header.copy()
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img, WCS(hdr)

def robust_norm(x):
    v = x[np.isfinite(x)]
    if v.size == 0: return np.zeros_like(x, np.float32)
    lo, hi = np.percentile(v, [1, 99])
    x = np.clip(x, lo, hi)
    med = np.median(x); q25, q75 = np.percentile(x, [25, 75]); iqr = max(q75-q25, 1e-6)
    return ((x - med) / iqr).astype(np.float32)

def build_emission_mask(img):
    # threshold smoothed moment-0 to get galaxy footprint
    x = cv2.GaussianBlur(img, (0,0), 2.0)
    pos = x[x > 0]
    if pos.size == 0:
        thr = np.percentile(x, 95)
    else:
        # combine percentile with moment stats to be robust
        thr = max(np.percentile(pos, 60), np.median(pos) + 0.3*np.std(pos))
    m = (x >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    return m

def load_catalog(csv_path, galaxy):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "galaxy" in df.columns:
        df = df[df["galaxy"].str.strip().str.lower() == galaxy.strip().lower()].reset_index(drop=True)
    # column normalization
    col_map = {}
    for c in df.columns:
        if c in ("ra","ra_deg","ra (deg)"): col_map[c] = "ra_deg"
        if c in ("dec","dec_deg","dec (deg)"): col_map[c] = "dec_deg"
        if c in ("diameter","diameter_arcsec","d (arcsec)"): col_map[c] = "diameter_arcsec"
        if c in ("major","major_arcsec","major_diameter_arcsec"): col_map[c] = "major_arcsec"
        if c in ("minor","minor_arcsec","minor_diameter_arcsec"): col_map[c] = "minor_arcsec"
    df = df.rename(columns=col_map)
    if not {"ra_deg","dec_deg"}.issubset(df.columns):
        raise ValueError("Catalogue missing RA/Dec columns.")
    return df

def draw_holes_mask(df, shape, wcs, min_diam_arcsec=0.0):
    ny, nx = shape
    mask = np.zeros((ny, nx), np.uint8)
    # estimate pixels per arcsec
    scale = np.abs(wcs.pixel_scale_matrix)
    if np.all(scale == 0) or np.isnan(scale).any():
        cdelt = np.abs(wcs.wcs.cdelt[:2]) if wcs.wcs.cdelt is not None else np.array([1/3600,1/3600])
        arcsec_per_pix = float(np.mean(cdelt) * 3600.0)
    else:
        from astropy.wcs.utils import proj_plane_pixel_scales
        arcsec_per_pix = float(np.mean(proj_plane_pixel_scales(wcs) * 3600.0))
    pix_per_arcsec = 1.0 / max(arcsec_per_pix, 1e-6)

    for _, r in df.iterrows():
        if "diameter_arcsec" in r and pd.notnull(r["diameter_arcsec"]):
            dmaj = float(r["diameter_arcsec"]); dmin = dmaj
        else:
            dmaj = float(r.get("major_arcsec", 0) or 0)
            dmin = float(r.get("minor_arcsec", dmaj) or dmaj)
        if dmaj < min_diam_arcsec: continue
        sky = SkyCoord(r["ra_deg"]*u.deg, r["dec_deg"]*u.deg)
        x, y = wcs.world_to_pixel(sky)
        a = int(round((dmaj/2.0) * pix_per_arcsec))
        b = int(round((dmin/2.0) * pix_per_arcsec))
        if a <= 0 or b <= 0: continue
        rr, cc = ellipse(int(round(y)), int(round(x)), b, a, shape=mask.shape)
        mask[rr, cc] = 1
    return mask

def save_npz(idx, img_tile, mask_tile, valid_tile, out_dir):
    out = Path(out_dir) / f"tile_{idx:05d}.npz"
    np.savez_compressed(out, image=img_tile.astype(np.float32),
                             mask=mask_tile.astype(np.uint8),
                             valid=valid_tile.astype(np.uint8))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img, wcs = load_fits(FITS_PATH)
    z = robust_norm(img)
    valid = build_emission_mask(img)
    df = load_catalog(CATALOG_CSV, GALAXY_NAME)
    mask = draw_holes_mask(df, img.shape, wcs, min_diam_arcsec=MIN_DIAM_ARCSEC)

    # write a simple index.csv (path, has_pos, pos_frac, valid_frac)
    rows = []
    ny, nx = img.shape
    idx = 0
    for y in range(0, max(1, ny - TILE + 1), STRIDE):
        for x in range(0, max(1, nx - TILE + 1), STRIDE):
            iy, ix = y + TILE, x + TILE
            img_t   = z[y:iy, x:ix]
            mask_t  = mask[y:iy, x:ix]
            valid_t = valid[y:iy, x:ix]
            # pad at borders
            if img_t.shape != (TILE, TILE):
                pad = np.zeros((TILE, TILE), np.float32); pad[:img_t.shape[0], :img_t.shape[1]] = img_t; img_t = pad
                pad = np.zeros((TILE, TILE), np.uint8);   pad[:mask_t.shape[0], :mask_t.shape[1]] = mask_t; mask_t = pad
                pad = np.zeros((TILE, TILE), np.uint8);   pad[:valid_t.shape[0], :valid_t.shape[1]] = valid_t; valid_t = pad

            vfrac = valid_t.mean()
            if vfrac < MIN_EMISSION_COVER:
                continue  # skip outside-galaxy tiles entirely

            has_pos = int(mask_t.sum() > 0)
            if has_pos == 0:
                # keep only a fraction of empty tiles *inside* galaxy
                if np.random.default_rng(0).random() > KEEP_EMPTY_IN_GAL:
                    continue

            save_npz(idx, img_t, mask_t, valid_t, OUT_DIR)
            rows.append((f"tile_{idx:05d}.npz", has_pos, float(mask_t.mean()), float(vfrac)))
            idx += 1

    import csv
    with open(Path(OUT_DIR,"index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","has_pos","pos_frac","valid_frac"])
        w.writerows(rows)
    print(f"[OK] saved {len(rows)} tiles → {OUT_DIR}")

if __name__ == "__main__":
    main()