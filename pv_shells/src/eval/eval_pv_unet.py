# src/eval/eval_pv_unet.py
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils.config import resolve_config
from src.pv.dataset import build_dataset


def _len_of_split(cfg_path: str, split: str) -> int:
    splits_dir = Path("data") / "splits"
    fname = {"train": "train.json", "val": "val.json", "test": "test.json"}[split]
    p = splits_dir / fname
    if p.exists():
        try:
            return len(json.loads(p.read_text()))
        except Exception:
            pass
    return 0


def _build_eval_dataset(cfg_path: str, split: str, batch_size: int):
    """Non-repeating dataset for evaluation (no shuffle arg)."""
    return build_dataset(
        cfg_path,
        split,
        batch_size=batch_size,
        seed=0,
        repeat=False,   # finite dataset to avoid OUT_OF_RANGE spam
    )


def _collect_logits_and_labels(model: keras.Model, ds) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in ds:              # iterate to exhaustion
        p = model(x, training=False)
        ys.append(y.numpy())
        ps.append(p.numpy())
    if not ys:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    y_true = np.concatenate(ys, axis=0)[..., 0].ravel().astype(np.float32)
    y_score = np.concatenate(ps, axis=0)[..., 0].ravel().astype(np.float32)
    return y_true, y_score


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    m = tf.keras.metrics.AUC(curve="PR", summation_method="interpolation")
    m.update_state(y_true, y_score)
    return float(m.result().numpy())


def _precision_recall_at_thresh(y_true: np.ndarray, y_score: np.ndarray, thresh: float):
    if y_true.size == 0:
        return float("nan"), float("nan"), 0, 0, 0, 0
    y_pred = (y_score >= thresh).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn, tn


def evaluate(cfg_path: str,
             run_dir: str | Path,
             split: str = "test",
             batch_size: int = 4,
             thresh: float = 0.5,
             out_json: str | None = None):
    _ = resolve_config(cfg_path, write_resolved=False)

    run_dir = Path(run_dir)
    model_path = run_dir / "best_model.keras"
    print(f"[eval] loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    ds = _build_eval_dataset(cfg_path, split, batch_size)
    n_items = _len_of_split(cfg_path, split)
    approx_steps = math.ceil(max(1, n_items) / batch_size) if n_items else "unknown"
    print(f"[eval] split={split}, n_items≈{n_items}, batch_size={batch_size}, approx_steps={approx_steps}")

    y_true, y_score = _collect_logits_and_labels(model, ds)

    prauc = _pr_auc(y_true, y_score)
    precision, recall, tp, fp, fn, tn = _precision_recall_at_thresh(y_true, y_score, thresh)

    results = {
        "split": split,
        "n_examples": int(y_true.size),
        "batch_size": batch_size,
        "threshold": float(thresh),
        "pr_auc": prauc,
        "precision_at_thresh": precision,
        "recall_at_thresh": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    print("[eval] PR-AUC={:.4f} | @thresh {:.2f}: precision={:.4f}, recall={:.4f} "
          "(tp={}, fp={}, fn={}, tn={})".format(
              prauc, thresh, precision, recall, tp, fp, fn, tn))

    out_path = Path(out_json) if out_json else (run_dir / f"eval_{split}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evaluate(args.config, args.run_dir, args.split, args.batch_size, args.thresh, args.out)