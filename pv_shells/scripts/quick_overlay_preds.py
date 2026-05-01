#!/usr/bin/env python3
# --- repo-root bootstrap ---
import os, sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f"[bootstrap] ROOT={ROOT}")

# --- real work ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils.io import load_yaml

def main(cfg_path, maxn, thresh):
    cfg = load_yaml("data/_resolved_config.yaml") if Path("data/_resolved_config.yaml").exists() else load_yaml(cfg_path)
    pv_dir  = Path(cfg["output_root"])/"pv"
    pred_dir= Path(cfg["output_root"])/"pred"
    out     = Path(cfg["output_root"])/"qa_pred"; out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(pred_dir.glob("*.npy")))[:maxn]
    if not files:
        print("[quick_overlay_preds] No predictions found in data/pred. Run inference first.")
        return
    for f in files:
        pv  = np.load(pv_dir/f.name)
        prob= np.load(f)
        mask= (prob >= thresh).astype(np.uint8)
        plt.figure()
        plt.imshow(pv, origin="lower", aspect="auto")
        plt.imshow(np.ma.masked_where(mask==0, mask), origin="lower", aspect="auto", alpha=0.35)
        plt.title(f"{f.name} thr={thresh}")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(out/f"{f.stem}_thr{thresh:.2f}.png", dpi=150)
        plt.close()
    print(f"[quick_overlay_preds] wrote {len(files)} overlays to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    main(args.config, args.max, args.thresh)