# src/infer/infer_pv.py
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.utils.io import load_yaml
from src.pv.dataset import _resolve_cfg

def main(cfg_path: str, model_path: str, out_dir: str | None):
    cfg = _resolve_cfg(cfg_path)
    root = Path(cfg["output_root"])
    pv_dir = root / "pv"
    out = Path(out_dir) if out_dir else (root / "pred")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[infer] loading model: {model_path}")
    m = tf.keras.models.load_model(model_path, compile=False)

    files = sorted([p for p in pv_dir.glob("*.npy") if not p.name.endswith("_posxy.npy")])
    assert files, f"No PV files in {pv_dir}"
    count = 0
    for p in files:
        x = np.load(p)  # (V,S)
        x_norm = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
        xin = x_norm[np.newaxis, ..., np.newaxis]  # (1,V,S,1)
        y = m.predict(xin, verbose=0)[0, ..., 0]   # (V,S)
        np.save(out / p.name, y.astype(np.float32))
        count += 1
    print(f"[infer] wrote {count} prob-maps to {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)  # e.g. data/exp/<run>/ckpt_best.h5 or final_model
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.config, args.model, args.out)