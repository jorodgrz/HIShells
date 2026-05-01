# src/pv/dataset.py
from __future__ import annotations
import random
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.utils.io import load_yaml
from src.utils.config import resolve_config

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------- basics -------------------------------

def _read_manifest(path: Path) -> List[str]:
    assert path.exists(), f"Manifest missing: {path}"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _resolve_cfg(cfg_path: str) -> Dict:
    """Load resolved config from cache if present, else resolve and write it."""
    res = Path("data/_resolved_config.yaml")
    return load_yaml(res) if res.exists() else resolve_config(cfg_path, write_resolved=True)


def _load_pair(root: Path, fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one PV/label pair (both .npy) and assert same shape."""
    pv_path  = root / "pv" / fname
    lab_path = root / "labels" / fname
    if not pv_path.exists() or not lab_path.exists():
        raise FileNotFoundError(f"Missing PV/label pair: {pv_path.name}")
    pv = np.load(pv_path)         # (V, S)
    lab = np.load(lab_path)       # (V, S)
    assert pv.shape == lab.shape, f"shape mismatch for {fname}: {pv.shape} vs {lab.shape}"
    return pv, lab


# ---------------------------- normalization ---------------------------

def _zscore_galaxy_only(pv: np.ndarray) -> np.ndarray:
    x = pv[np.isfinite(pv)]
    if x.size == 0:
        return pv.astype(np.float32)
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-6)
    return ((pv - mu) / sigma).astype(np.float32)


def _normalize(pv: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore_galaxy_only":
        return _zscore_galaxy_only(pv)
    elif method in ("none", None):
        return pv.astype(np.float32)
    else:
        raise ValueError(f"Unknown norm_method: {method}")


# ----------------------------- patching --------------------------------

def _pad_to(pv: np.ndarray, lab: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad to at least (ph, pw). PV: edge pad; label: zeros."""
    v, s = pv.shape
    dv = max(0, ph - v)
    ds = max(0, pw - s)
    if dv == 0 and ds == 0:
        return pv, lab
    pv = np.pad(
        pv, ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)), mode="edge"
    )
    lab = np.pad(
        lab, ((dv // 2, dv - dv // 2), (ds // 2, ds - ds // 2)),
        mode="constant", constant_values=0
    )
    return pv, lab


def _choose_patch(
    v: int, s: int,
    pv: np.ndarray, lab: np.ndarray,
    pos_frac: float, ph: int, pw: int,
    rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose a (ph×pw) crop; bias to positives by pos_frac if any labels exist."""
    want_pos = rng.random() < pos_frac
    if want_pos and lab.any():
        ys, xs = np.where(lab > 0)
        k = rng.randrange(len(ys))
        cy, cx = int(ys[k]), int(xs[k])
        y0 = max(0, min(cy - ph // 2, v - ph))
        x0 = max(0, min(cx - pw // 2, s - pw))
    else:
        y0 = rng.randrange(0, max(1, v - ph + 1))
        x0 = rng.randrange(0, max(1, s - pw + 1))
    y1, x1 = y0 + ph, x0 + pw
    return pv[y0:y1, x0:x1], lab[y0:y1, x0:x1]


def _gen_samples(cfg: Dict, split: str, seed: int = 1337):
    """Python generator that yields (x,y) patch pairs as np.float32 with channel dim."""
    rng = random.Random(seed)
    root = Path(cfg["output_root"])
    man_path = root / "splits" / f"{split}_manifest.txt"
    files = _read_manifest(man_path)

    norm = cfg["train"]["norm_method"]
    pos_frac = float(cfg["train"]["pos_fraction"])
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])

    for fname in files:
        if fname.endswith("_posxy.npy"):  # sidecar, skip
            continue
        try:
            pv, lab = _load_pair(root, fname)
        except FileNotFoundError as e:
            print(f"[dataset] warning: {e} ; skipping")
            continue

        # normalize then pad to requested patch size
        pv = _normalize(pv, norm)
        pv, lab = _pad_to(pv, lab, ph, pw)

        v, s = pv.shape
        # sample multiple patches per PV
        n_per = max(4, (v * s) // (ph * pw))  # small PVs still yield at least 4
        for _ in range(n_per):
            x, y = _choose_patch(v, s, pv, lab, pos_frac, ph, pw, rng)
            # add channel dim
            x = x[..., np.newaxis].astype(np.float32)
            y = y[..., np.newaxis].astype(np.float32)
            yield x, y


# ------------------------------ dataset --------------------------------

def build_dataset(
    cfg_path: str,
    split: str,
    batch_size: int,
    seed: int = 1337,
    repeat: bool = False
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline.
    - train: repeat+shuffle+batch+prefetch
    - val/test: no repeat, no shuffle (deterministic)
    """
    cfg = _resolve_cfg(cfg_path)
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])

    elemspec = (
        tf.TensorSpec(shape=(ph, pw, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(ph, pw, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _gen_samples(cfg, split=split, seed=seed),
        output_signature=elemspec
    )

    if split == "train":
        if repeat:
            ds = ds.repeat()
        ds = ds.shuffle(buffer_size=max(batch_size * 8, 64), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# -------------------------- utilities / CLI ----------------------------

def _selftest(cfg_path: str):
    cfg = _resolve_cfg(cfg_path)
    bs = max(1, int(cfg["optim"]["batch_size"]))
    ds = build_dataset(cfg_path, split="train", batch_size=bs, seed=42, repeat=False)
    it = iter(ds)
    x, y = next(it)
    print("[dataset.selftest] batch shapes:", x.shape, y.shape)
    print("[dataset.selftest] x mean/std:",
          tf.math.reduce_mean(x).numpy(),
          tf.math.reduce_std(x).numpy())
    print("[dataset.selftest] y unique:",
          tf.unique(tf.reshape(tf.cast(y > 0.5, tf.int32), [-1]))[0].numpy())
    print("[dataset.selftest] OK")


def estimate_num_patches(cfg_path: str, split: str) -> int:
    """Deterministically estimate how many patches the generator will yield for a split."""
    cfg = _resolve_cfg(cfg_path)
    root = Path(cfg["output_root"])
    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])
    files = _read_manifest(root / "splits" / f"{split}_manifest.txt")

    total = 0
    for fname in files:
        if fname.endswith("_posxy.npy"):  # skip meta sidecars
            continue
        pv_path = root / "pv" / fname
        if not pv_path.exists():
            continue
        v, s = np.load(pv_path, mmap_mode="r").shape  # no RAM spike
        # account for padding to at least (ph, pw)
        v = max(v, ph); s = max(s, pw)
        n_per = max(4, (v * s) // (ph * pw))
        total += n_per
    return total


def estimate_steps(cfg_path: str, split: str, batch_size: int) -> int:
    n = estimate_num_patches(cfg_path, split)
    return max(1, int(ceil(n / max(1, batch_size))))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.config)
    else:
        print("Use --selftest to run a quick dataset check.")