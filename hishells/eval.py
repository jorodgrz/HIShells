"""Per-fold metrics + operating-point selection (plan \u00a76.1).

The single entry point :func:`compute_metrics` returns the 14-column
dict that every \u00a79 ablation row writes to ``results/ablations.csv``.
The rule is:

1. On the within-fold *validation* set, find the lowest threshold
   ``tau`` whose recall is ``>= recall_floor`` (default 0.70 per
   plan).
2. If no such threshold exists, fall back to the threshold that
   maximises validation F1 and flag the row in ``notes``.
3. Apply the chosen ``tau`` to the test-galaxy scores and report
   ``recall_at_op``, ``fp_per_galaxy_at_op``, ``precision_at_op``,
   ``F1`` (all at the operating point) plus the threshold-free
   ``AUC_PR``, ``AUC_ROC``, and ``ECE``.

This module is pure-numpy / scikit-learn so the same helper drives
both the CNN folds and the MTB / trivial / CASI baseline rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # sklearn is in environment.yml but be defensive
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    average_precision_score = None  # type: ignore[assignment]
    precision_recall_curve = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Operating-point selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    recall: float
    precision: float
    f1: float
    notes: str = ""


def _safe_div(num: float, den: float) -> float:
    if den <= 0 or not np.isfinite(den):
        return 0.0
    return float(num) / float(den)


def _binary_metrics_at(threshold: float, scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, int]:
    """Return ``(recall, precision, f1, n_fp)`` at ``threshold``."""

    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return recall, precision, f1, fp


def select_operating_point(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    *,
    recall_floor: float = 0.70,
) -> OperatingPoint:
    """Implement plan \u00a76.1's operating-point rule on the validation set."""

    if precision_recall_curve is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for operating-point selection")
    if val_labels.sum() == 0:
        return OperatingPoint(
            threshold=0.5,
            recall=0.0,
            precision=0.0,
            f1=0.0,
            notes="no positives in validation set",
        )

    precision, recall, thresh = precision_recall_curve(val_labels, val_scores)
    # ``precision_recall_curve`` returns ``thresholds`` of length n-1
    # corresponding to ``precision[1:]`` and ``recall[1:]``. Walk from
    # high-recall (low-threshold) to low-recall (high-threshold) and
    # pick the lowest threshold whose recall meets the floor.
    rec_aligned = recall[1:]
    prec_aligned = precision[1:]
    candidates = np.where(rec_aligned >= recall_floor)[0]
    if candidates.size > 0:
        # Among thresholds meeting the floor, take the lowest threshold
        # (== highest recall point that still meets the floor).
        idx = candidates[np.argmin(thresh[candidates])]
        tau = float(thresh[idx])
        rec, prec, f1, _ = _binary_metrics_at(tau, val_scores, val_labels)
        return OperatingPoint(threshold=tau, recall=rec, precision=prec, f1=f1)

    # Fallback: max-F1 threshold, flag in notes.
    f1s = 2 * prec_aligned * rec_aligned / np.clip(prec_aligned + rec_aligned, 1e-12, None)
    if f1s.size == 0:
        return OperatingPoint(threshold=0.5, recall=0.0, precision=0.0, f1=0.0, notes="empty PR curve")
    best = int(np.argmax(f1s))
    tau = float(thresh[best])
    rec, prec, f1, _ = _binary_metrics_at(tau, val_scores, val_labels)
    return OperatingPoint(
        threshold=tau,
        recall=rec,
        precision=prec,
        f1=f1,
        notes=f"no tau achieves val recall {recall_floor:.2f}; using max-F1 fallback",
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Compute ECE per Naeini, Cooper, Hauskrecht 2015.

    Bins the predicted probabilities into ``n_bins`` equal-width bins,
    weights the absolute gap between bin-mean predicted probability
    and bin-mean accuracy by bin occupancy, and sums.
    """

    if scores.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = scores.size
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = float(labels[mask].mean())
        bin_conf = float(scores[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# 14-column metric helper
# ---------------------------------------------------------------------------


METRIC_COLUMNS: tuple[str, ...] = (
    "name",
    "fold",
    "threshold",
    "recall_at_op",
    "fp_per_galaxy_at_op",
    "precision_at_op",
    "F1",
    "AUC_PR",
    "AUC_ROC",
    "ECE",
    "n_train",
    "n_test",
    "seed",
    "notes",
)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    *,
    name: str,
    fold: str,
    n_train: int,
    seed: int,
    recall_floor: float = 0.70,
) -> dict:
    """Compute the full 14-column dict for one (ablation, fold).

    Parameters
    ----------
    scores, labels
        Test-galaxy scores in [0, 1] and {0, 1} labels.
    val_scores, val_labels
        Within-fold validation scores / labels (used for operating-
        point selection only).
    name
        Ablation row name (e.g. ``"v1_baseline"``).
    fold
        Held-out galaxy stem (e.g. ``"NGC_2403"``).
    n_train
        Number of training windows (positives + negatives).
    seed
        RNG seed for the run.
    recall_floor
        Plan default 0.70.
    """

    op = select_operating_point(val_scores, val_labels, recall_floor=recall_floor)
    rec_t, prec_t, f1_t, n_fp = _binary_metrics_at(op.threshold, scores, labels)

    if average_precision_score is None or roc_auc_score is None:  # pragma: no cover
        auc_pr = float("nan")
        auc_roc = float("nan")
    else:
        # AUC scores require both classes present; default to NaN if
        # the held-out galaxy has all-positive or all-negative scores.
        if labels.sum() == 0 or labels.sum() == labels.size:
            auc_pr = float("nan")
            auc_roc = float("nan")
        else:
            auc_pr = float(average_precision_score(labels, scores))
            auc_roc = float(roc_auc_score(labels, scores))

    ece = expected_calibration_error(scores, labels)

    return {
        "name": name,
        "fold": fold,
        "threshold": float(op.threshold),
        "recall_at_op": float(rec_t),
        "fp_per_galaxy_at_op": int(n_fp),
        "precision_at_op": float(prec_t),
        "F1": float(f1_t),
        "AUC_PR": float(auc_pr) if np.isfinite(auc_pr) else float("nan"),
        "AUC_ROC": float(auc_roc) if np.isfinite(auc_roc) else float("nan"),
        "ECE": float(ece),
        "n_train": int(n_train),
        "n_test": int(labels.size),
        "seed": int(seed),
        "notes": op.notes,
    }


# ---------------------------------------------------------------------------
# Aggregation across folds (used by scripts/eval_logo.py)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(lo), float(hi)


def aggregate_folds(rows: Iterable[dict]) -> dict:
    """Aggregate per-fold metric dicts to mean +/- bootstrapped CI.

    Returned keys: ``mean_recall_at_op``, ``ci_recall_at_op``,
    ``mean_fp_per_galaxy_at_op``, ``ci_fp_per_galaxy_at_op``,
    ``mean_AUC_PR``, ``mean_AUC_ROC``, ``mean_ECE``,
    ``passes_recall_target``, ``passes_fp_target``.
    """

    rows = list(rows)
    recalls = [r["recall_at_op"] for r in rows]
    fps = [r["fp_per_galaxy_at_op"] for r in rows]
    rec_ci = bootstrap_ci(recalls)
    fp_ci = bootstrap_ci(fps)
    return {
        "mean_recall_at_op": float(np.nanmean(recalls)) if recalls else float("nan"),
        "ci_recall_at_op": rec_ci,
        "mean_fp_per_galaxy_at_op": float(np.nanmean(fps)) if fps else float("nan"),
        "ci_fp_per_galaxy_at_op": fp_ci,
        "mean_AUC_PR": float(np.nanmean([r["AUC_PR"] for r in rows])) if rows else float("nan"),
        "mean_AUC_ROC": float(np.nanmean([r["AUC_ROC"] for r in rows])) if rows else float("nan"),
        "mean_ECE": float(np.nanmean([r["ECE"] for r in rows])) if rows else float("nan"),
        "passes_recall_target": bool(rec_ci[0] >= 0.70),
        "passes_fp_target": bool(fp_ci[1] <= 5.0),
    }
