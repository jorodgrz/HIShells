#!/usr/bin/env python3
# MakePatches.py — tile image + masks into training patches (.npz)

import os, math, csv, random
import numpy as np
import pandas as pd
from astropy.io import fits

# ========= CONFIG =========
FITS_PATH       = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/NGC_2403_NA_MOM0_THINGS.FITS"
MASK_DIR        = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/masks_out"
OUT_DIR         = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/patches_256s192"
PATCH_SIZE      = 256
STRIDE          = 192        # 25% overlap for 256
POS_FRAC_THRESH = 0.01       # save patch if region coverage ≥ 1%
NEG_SAMPLE_RATE = 0.25       # sample this fraction of negatives
RNG_SEED        = 42
# =========================

def load_fits(path):
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None and h.data.ndim >= 2)
        data = np.squeeze(hdu.data).astype(np.float32)
    # Fill NaNs/Infs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def offsets(n, win, stride):
    xs = list(range(0, max(1, n - win + 1), stride))
    if xs[-1] != n - win:
        xs.append(n - win)
    return xs

def main():
    random.seed(RNG_SEED)
    ensure_dir(OUT_DIR)

    img = load_fits(FITS_PATH)
    mask_region = load_fits(os.path.join(MASK_DIR, "mask_region.fits")).astype(np.uint8)
    mask_rim    = load_fits(os.path.join(MASK_DIR, "mask_rim.fits")).astype(np.uint8)
    instances   = load_fits(os.path.join(MASK_DIR, "instances.fits")).astype(np.int32)

    H, W = img.shape
    assert mask_region.shape == (H, W)
    assert mask_rim.shape == (H, W)
    assert instances.shape == (H, W)

    xs = offsets(W, PATCH_SIZE, STRIDE)
    ys = offsets(H, PATCH_SIZE, STRIDE)

    # global stats (use robust clip to avoid crazy outliers)
    finite = np.isfinite(img)
    v = img[finite]
    if v.size == 0:
        v = np.array([0.0], dtype=np.float32)
    lo, hi = np.percentile(v, [1, 99])
    v_clipped = np.clip(v, lo, hi)
    mean, std = float(v_clipped.mean()), float(v_clipped.std() + 1e-6)

    index_rows = []
    saved = 0
    total = 0

    for y0 in ys:
        for x0 in xs:
            total += 1
            y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE

            im_patch  = img[y0:y1, x0:x1]
            reg_patch = mask_region[y0:y1, x0:x1]
            rim_patch = mask_rim[y0:y1, x0:x1]
            ins_patch = instances[y0:y1, x0:x1]

            # sanity for border conditions
            if im_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue

            # keep if positive enough, else sample as negative
            frac_region = float(reg_patch.sum()) / float(PATCH_SIZE * PATCH_SIZE)
            is_pos = frac_region >= POS_FRAC_THRESH
            if not is_pos and random.random() > NEG_SAMPLE_RATE:
                continue

            # (Optional) normalize to ~N(0,1) using global robust stats
            im_norm = (im_patch - mean) / std
            im_norm = im_norm.astype(np.float32)

            # Save
            fid = f"y{y0:05d}_x{x0:05d}"
            out_path = os.path.join(OUT_DIR, f"{fid}.npz")
            np.savez_compressed(
                out_path,
                image=im_norm,
                region=reg_patch.astype(np.uint8),
                rim=rim_patch.astype(np.uint8),
                instances=ins_patch.astype(np.int32),
                x0=np.int32(x0),
                y0=np.int32(y0),
                mean=np.float32(mean),
                std=np.float32(std),
            )
            index_rows.append({
                "id": fid, "x0": x0, "y0": y0,
                "frac_region": frac_region, "is_positive": int(is_pos),
                "path": out_path
            })
            saved += 1

    # write index
    idx_path = os.path.join(OUT_DIR, "index.csv")
    pd.DataFrame(index_rows).to_csv(idx_path, index=False)

    # write stats
    with open(os.path.join(OUT_DIR, "stats.txt"), "w") as f:
        f.write(f"H={H}, W={W}\n")
        f.write(f"patch={PATCH_SIZE}, stride={STRIDE}\n")
        f.write(f"saved={saved} / candidates={total}\n")
        f.write(f"global_mean={mean:.6g}, global_std={std:.6g}, clip=[{lo:.6g},{hi:.6g}]\n")

    print(f"[OK] Saved {saved} patches to {OUT_DIR}")
    print(f"Index: {idx_path}")

if __name__ == "__main__":
    main()