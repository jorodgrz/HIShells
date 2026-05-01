#!/usr/bin/env python3
# detect_shells_v2.py — PyTorch inference for big HI shells (matches model_best.pth)

import os, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt

# ========= PATHS (edit if needed) =========
FITS_PATH = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/NGC_2403_NA_MOM0_THINGS.FITS"
CKPT_PATH = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/model_best.pth"
OUT_DIR   = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/infer_v2_out"
# ==========================================

# ========= INFERENCE SETTINGS =========
TILE   = 256
STRIDE = 192
BATCH  = 8

# multiscale + TTA
SCALES = [1.0, 0.75, 0.5]   # fewer scales → faster; add 0.35 if you want more recall
USE_TTA_FLIPS = True        # horizontal/vertical flip TTA

# adaptive hysteresis
FIXED_HI  = 0.55            # lower bound on the high threshold
Q_HI      = 0.92            # quantile-based hi threshold; hi = max(FIXED_HI, quantile(Q_HI))
LO_FRAC   = 0.6             # low = hi * LO_FRAC (0.5–0.7 reasonable)

# morphology
OPEN_K  = 5                 # remove specks
CLOSE_K = 9                 # connect arcs

# size prior (favor big structures)
MIN_AREA_PX      = 300
MIN_MAJOR_ARCSEC = 18.0
# fallback if nothing is kept
FALLBACK_MIN_AREA_PX      = 150
FALLBACK_MIN_MAJOR_ARCSEC = 12.0

# rim export (from filled detection)
RIM_W = 3
# =====================================


# ======= MODEL (GroupNorm U-Net, same family you trained) =======
def GN(ch: int) -> nn.GroupNorm:
    for g in (8,4,2,1):
        if ch % g == 0: return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), GN(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), GN(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels*2, base_channels*4); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels*4, base_channels*8)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2); self.dec3 = DoubleConv(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2); self.dec2 = DoubleConv(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2);  self.dec1 = DoubleConv(base_channels*2, base_channels)
        self.final = nn.Conv2d(base_channels, 1, 1)
    def forward(self,x):
        x1=self.enc1(x); x2=self.enc2(self.pool1(x1)); x3=self.enc3(self.pool2(x2))
        x4=self.bottleneck(self.pool3(x3))
        x=self.up3(x4); x=torch.cat([x,x3],1); x=self.dec3(x)
        x=self.up2(x);  x=torch.cat([x,x2],1); x=self.dec2(x)
        x=self.up1(x);  x=torch.cat([x,x1],1); x=self.dec1(x)
        return self.final(x)

# =================== IO / UTILS ===================
def load_fits(path):
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h,"data",None) is not None and h.data.ndim>=2)
        img = np.squeeze(hdu.data).astype(np.float32)
        hdr = hdu.header.copy()
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    wcs = WCS(hdr); wcs = wcs.celestial if hasattr(wcs,"celestial") else WCS(hdr, naxis=2)
    scales = proj_plane_pixel_scales(wcs) * 3600.0  # arcsec/pix
    aspix = float(np.sqrt(scales[0]*scales[1]))
    return img, wcs, aspix

def robust_norm(img):
    v = img[np.isfinite(img)]
    if v.size==0: return np.zeros_like(img, np.float32)
    lo,hi = np.percentile(v, [1,99])
    x = np.clip(img, lo, hi)
    med = np.median(x); q25,q75 = np.percentile(x, [25,75]); iqr = max(q75-q25, 1e-6)
    return ((x - med) / iqr).astype(np.float32)

def make_offsets(L, win, stride):
    xs = list(range(0, max(1, L - win + 1), stride))
    if xs[-1] != L - win: xs.append(max(0, L - win))
    return xs

@torch.no_grad()
def predict_full(img_z, model, device):
    H,W = img_z.shape
    ys = make_offsets(H, TILE, STRIDE)
    xs = make_offsets(W, TILE, STRIDE)
    out = np.zeros((H,W), np.float32)
    cnt = np.zeros((H,W), np.float32)
    batch=[]; locs=[]
    for y in ys:
        for x in xs:
            patch = img_z[y:y+TILE, x:x+TILE]
            if patch.shape != (TILE,TILE):
                pad = np.zeros((TILE,TILE), np.float32)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            batch.append(patch[None,None,...]); locs.append((y,x))
            if len(batch)==BATCH:
                inp=torch.from_numpy(np.concatenate(batch,0)).to(device)
                prob=torch.sigmoid(model(inp)).cpu().numpy()[:,0]
                for (yy,xx), pr in zip(locs, prob):
                    h2=min(TILE,H-yy); w2=min(TILE,W-xx)
                    out[yy:yy+h2, xx:xx+w2] += pr[:h2,:w2]; cnt[yy:yy+h2, xx:xx+w2] += 1.0
                batch=[]; locs=[]
    if batch:
        inp=torch.from_numpy(np.concatenate(batch,0)).to(device)
        prob=torch.sigmoid(model(inp)).cpu().numpy()[:,0]
        for (yy,xx), pr in zip(locs, prob):
            h2=min(TILE,H-yy); w2=min(TILE,W-xx)
            out[yy:yy+h2, xx:xx+w2] += pr[:h2,:w2]; cnt[yy:yy+h2, xx:xx+w2] += 1.0
    return np.divide(out, cnt, out=np.zeros_like(out), where=cnt>0)

def predict_multiscale(img_z, model, device, scales, tta_flips=True):
    H,W = img_z.shape
    acc = np.full((H,W), -np.inf, np.float32)

    def run_once(arr):
        pr = predict_full(arr, model, device)
        return pr

    def apply_tta(arr):
        outs = []
        # original
        outs.append(run_once(arr))
        if not tta_flips: return outs
        # horizontal
        a = np.fliplr(arr).copy(); outs.append(np.fliplr(run_once(a)))
        # vertical
        a = np.flipud(arr).copy(); outs.append(np.flipud(run_once(a)))
        # both
        a = np.flipud(np.fliplr(arr)).copy(); outs.append(np.flipud(np.fliplr(run_once(a))))
        return outs

    for s in scales:
        h2,w2 = max(1,int(round(H*s))), max(1,int(round(W*s)))
        im_s = cv2.resize(img_z, (w2,h2), interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
        outs = apply_tta(im_s)
        pr_s = np.mean(outs, axis=0)
        pr_up = cv2.resize(pr_s, (W,H), interpolation=cv2.INTER_LINEAR)
        acc = np.maximum(acc, pr_up)
    return np.where(np.isfinite(acc), acc, 0.0)

def hysteresis_from_prob(prob, fixed_hi=0.55, q_hi=0.92, lo_frac=0.6):
    hi_q = float(np.quantile(prob, q_hi))
    t_hi = max(fixed_hi, hi_q)
    t_lo = max(0.05, min(0.95, t_hi * lo_frac))
    hi = (prob >= t_hi).astype(np.uint8)
    lo = (prob >= t_lo).astype(np.uint8)
    if OPEN_K>1:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(OPEN_K,OPEN_K))
        hi=cv2.morphologyEx(hi, cv2.MORPH_OPEN, k, 1)
        lo=cv2.morphologyEx(lo, cv2.MORPH_OPEN, k, 1)
    if CLOSE_K>1:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLOSE_K,CLOSE_K))
        hi=cv2.morphologyEx(hi, cv2.MORPH_CLOSE, k, 1)
    num, labels, _, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
    keep = np.zeros_like(lo)
    for i in range(1, num):
        comp = (labels==i)
        if (hi[comp]>0).any():
            keep[comp]=1
    return keep, t_hi, t_lo

def filter_by_size(mask, min_area_px, min_major_arcsec, aspix_arcsec):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, np.uint8); kept=0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px: continue
        cnts,_ = cv2.findContours((labels==i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 5: continue
        (_, _), (maj, _), _ = cv2.fitEllipse(c)
        if (maj * aspix_arcsec) < min_major_arcsec: continue
        out[labels==i]=1; kept+=1
    return out, kept

def mask_to_rim(mask, rim_w=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rim_w+1, 2*rim_w+1))
    dil = cv2.dilate(mask, k, 1); ero = cv2.erode(mask, k, 1)
    return cv2.subtract(dil, ero)

def export_ds9(mask, wcs, aspix_arcsec, out_dir):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows=[]
    for c in cnts:
        if len(c) < 5: continue
        (cx,cy),(maj,minr),ang = cv2.fitEllipse(c)
        ra_deg, dec_deg = wcs.all_pix2world([[cx, cy]], 0)[0]
        pa = (90.0 - ang) % 180.0  # DS9 PA: N->E
        rows.append((ra_deg, dec_deg, float(maj*aspix_arcsec), float(minr*aspix_arcsec), float(pa)))
    if rows:
        with open(Path(out_dir,"detections_fk5.reg"),"w") as f:
            f.write("# Region file format: DS9\n"); f.write("global color=cyan width=1\nfk5\n")
            for ra,dec,a,b,pa in rows:
                f.write(f'ellipse({ra:.6f},{dec:.6f},{a:.2f}",{b:.2f}",{pa:.1f})\n')
        import pandas as pd
        pd.DataFrame(rows, columns=["ra_deg","dec_deg","major_arcsec","minor_arcsec","pa_deg"])\
          .to_csv(Path(out_dir,"detections.csv"), index=False)
    return len(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    try:
        # load & norm
        print("[INFO] load FITS …")
        img, wcs, aspix = load_fits(FITS_PATH)
        print(f"[INFO] image={img.shape}, scale≈{aspix:.3f}\"/px")
        z = robust_norm(img)

        # model
        print("[INFO] load model …")
        model = UNet(in_channels=1, base_channels=24)
        state = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(state); model.eval(); model.to(device)

        # inference
        print(f"[INFO] infer: TILE={TILE} STRIDE={STRIDE} BATCH={BATCH} SCALES={SCALES} TTA={USE_TTA_FLIPS}")
        t0 = time.time()
        prob = predict_multiscale(z, model, device, SCALES, tta_flips=USE_TTA_FLIPS)
        dt = time.time()-t0
        print(f"[INFO] inference {dt:.1f}s | prob min/max={prob.min():.3f}/{prob.max():.3f}")
        plt.imsave(str(Path(OUT_DIR,"prob.png")), prob, cmap="hot")

        # hysteresis
        mask_hys, t_hi, t_lo = hysteresis_from_prob(prob, FIXED_HI, Q_HI, LO_FRAC)
        print(f"[INFO] hysteresis: hi={t_hi:.3f} lo={t_lo:.3f}")
        plt.imsave(str(Path(OUT_DIR,"mask_hysteresis.png")), (mask_hys*255).astype(np.uint8))

        # size prior
        print(f"[INFO] size prior: area≥{MIN_AREA_PX}px, major≥{MIN_MAJOR_ARCSEC:.1f}\"")
        mask_big, kept = filter_by_size(mask_hys, MIN_AREA_PX, MIN_MAJOR_ARCSEC, aspix)
        if kept == 0:
            print(f"[FALLBACK] relax to area≥{FALLBACK_MIN_AREA_PX}px, major≥{FALLBACK_MIN_MAJOR_ARCSEC:.1f}\"")
            mask_big, kept = filter_by_size(mask_hys, FALLBACK_MIN_AREA_PX, FALLBACK_MIN_MAJOR_ARCSEC, aspix)
        plt.imsave(str(Path(OUT_DIR,"mask_big.png")), (mask_big*255).astype(np.uint8))

        # rim + overlay
        rim = mask_to_rim(mask_big, RIM_W)
        cv2.imwrite(str(Path(OUT_DIR,"rim.png")), (rim*255).astype(np.uint8))
        plt.figure(figsize=(6,6)); plt.imshow(z, cmap="gray"); plt.contour(mask_big, colors="lime", linewidths=0.6)
        plt.contour(rim, colors="r", linewidths=0.5); plt.axis("off"); plt.tight_layout()
        plt.savefig(str(Path(OUT_DIR,"overlay.png")), dpi=220); plt.close()

        # DS9 export
        n = export_ds9(rim, wcs, aspix, OUT_DIR)
        print(f"[RESULT] components_kept={kept} | DS9 ellipses={n}")
        print(f"[OK] wrote → {OUT_DIR}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] partial outputs saved in:", OUT_DIR)
        sys.exit(130)

if __name__ == "__main__":
    main()