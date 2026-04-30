"""Aggregate the LOGO sweep results (plan §6.1, §11 step 10).

Reads ``results/ablations.csv`` (one row per ``(name, fold)``),
bootstraps a 95% CI for the ``recall_at_op`` and
``fp_per_galaxy_at_op`` columns, writes a per-ablation summary table
to ``results/summary.csv``, and saves a small set of diagnostic
figures to ``results/figures/``:

* ``recall_at_op_per_fold__<name>.png`` -- bar chart of the 19 fold
  recalls with the bootstrapped mean line and 95% CI band.
* ``fp_per_galaxy_at_op__<name>.png`` -- same for false-positive count.
* ``proposal_targets.png`` -- a single figure summarising every
  ablation against the §6 acceptance bar (``recall ≥ 0.70`` lower CI;
  ``fp/galaxy ≤ 5`` upper CI).

The summary table flags ``passes_recall_target`` /
``passes_fp_target`` so a CI step (or eyeball) can answer "did the
v1 baseline meet the proposal goals?" without re-running anything.

Usage:

    python scripts/eval_logo.py
    python scripts/eval_logo.py --csv results/ablations.csv --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hishells.eval import aggregate_folds, bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_ablations(df: pd.DataFrame) -> pd.DataFrame:
    """One summary row per ``name``, with bootstrapped CIs.

    Mirrors :func:`hishells.eval.aggregate_folds` but flattens the
    nested CI tuples into separate columns so the table can be saved
    directly to CSV.
    """

    rows = []
    for name, sub in df.groupby("name", sort=True):
        agg = aggregate_folds(sub.to_dict(orient="records"))
        rows.append(
            {
                "name": name,
                "n_folds": int(len(sub)),
                "mean_recall_at_op": agg["mean_recall_at_op"],
                "ci_recall_lo": agg["ci_recall_at_op"][0],
                "ci_recall_hi": agg["ci_recall_at_op"][1],
                "mean_fp_per_galaxy_at_op": agg["mean_fp_per_galaxy_at_op"],
                "ci_fp_lo": agg["ci_fp_per_galaxy_at_op"][0],
                "ci_fp_hi": agg["ci_fp_per_galaxy_at_op"][1],
                "mean_AUC_PR": agg["mean_AUC_PR"],
                "mean_AUC_ROC": agg["mean_AUC_ROC"],
                "mean_ECE": agg["mean_ECE"],
                "passes_recall_target": agg["passes_recall_target"],
                "passes_fp_target": agg["passes_fp_target"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _per_fold_bar(
    sub: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    target: float | None,
    target_kind: str,
    out_path: Path,
) -> None:
    """One figure: per-fold metric + bootstrapped mean ± CI."""

    import matplotlib.pyplot as plt

    sub = sub.sort_values("fold").reset_index(drop=True)
    values = sub[metric].to_numpy(dtype=np.float64)
    folds = sub["fold"].astype(str).tolist()
    lo, hi = bootstrap_ci(values)
    mean = float(np.nanmean(values)) if values.size else float("nan")

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * max(len(folds), 1) + 2), 4))
    ax.bar(folds, values, color="#1f77b4", alpha=0.85)
    ax.axhline(mean, color="black", linestyle="--", linewidth=1.0, label=f"mean={mean:.3f}")
    ax.axhspan(lo, hi, color="black", alpha=0.10, label=f"95% CI=[{lo:.3f}, {hi:.3f}]")
    if target is not None:
        ax.axhline(target, color="red", linestyle=":", linewidth=1.0, label=f"target {target_kind} {target}")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("fold (held-out galaxy)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _proposal_targets_figure(summary: pd.DataFrame, out_path: Path) -> None:
    """Single figure: every ablation's recall/FP CI vs proposal targets."""

    import matplotlib.pyplot as plt

    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4 + 0.2 * len(summary)))
    names = summary["name"].tolist()
    y = np.arange(len(names))

    # Left: recall_at_op
    ax = axes[0]
    ax.errorbar(
        summary["mean_recall_at_op"],
        y,
        xerr=[
            summary["mean_recall_at_op"] - summary["ci_recall_lo"],
            summary["ci_recall_hi"] - summary["mean_recall_at_op"],
        ],
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    ax.axvline(0.70, color="red", linestyle=":", linewidth=1.0, label="target ≥ 0.70")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("recall @ operating point (mean ± 95% CI)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize="small")

    # Right: fp_per_galaxy_at_op
    ax = axes[1]
    ax.errorbar(
        summary["mean_fp_per_galaxy_at_op"],
        y,
        xerr=[
            summary["mean_fp_per_galaxy_at_op"] - summary["ci_fp_lo"],
            summary["ci_fp_hi"] - summary["mean_fp_per_galaxy_at_op"],
        ],
        fmt="o",
        color="#ff7f0e",
        capsize=3,
    )
    ax.axvline(5.0, color="red", linestyle=":", linewidth=1.0, label="target ≤ 5")
    ax.set_yticks(y)
    ax.set_yticklabels(["" for _ in names])
    ax.set_xlabel("FP / galaxy @ op (mean ± 95% CI)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize="small")

    fig.suptitle("LOGO sweep vs proposal targets")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=REPO / "results" / "ablations.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "figures")
    ap.add_argument("--summary", type=Path, default=REPO / "results" / "summary.csv")
    ap.add_argument("--name-filter", default=None, help="Only aggregate ablations whose name contains this substring.")
    ap.add_argument("--no-figures", action="store_true", help="Compute summary CSV only; skip matplotlib.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.csv.exists():
        print(f"!! {args.csv} not found; nothing to aggregate", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if args.name_filter:
        df = df[df["name"].str.contains(args.name_filter, na=False)]
    if df.empty:
        print("!! no rows after filtering", file=sys.stderr)
        return 1

    summary = aggregate_ablations(df)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"-- wrote summary: {args.summary}")
    print(summary.to_string(index=False))

    if args.no_figures:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, sub in df.groupby("name", sort=True):
        _per_fold_bar(
            sub,
            "recall_at_op",
            title=f"recall@op per fold ({name})",
            ylabel="recall @ op",
            target=0.70,
            target_kind=">=",
            out_path=args.out_dir / f"recall_at_op_per_fold__{name}.png",
        )
        _per_fold_bar(
            sub,
            "fp_per_galaxy_at_op",
            title=f"FP/galaxy@op per fold ({name})",
            ylabel="FP / galaxy",
            target=5.0,
            target_kind="<=",
            out_path=args.out_dir / f"fp_per_galaxy_at_op_per_fold__{name}.png",
        )
    _proposal_targets_figure(summary, args.out_dir / "proposal_targets.png")
    print(f"-- wrote figures to {args.out_dir}")

    # Tiny JSON summary so downstream CI / notebooks can parse it cheaply.
    targets = {
        "n_ablations": int(len(summary)),
        "passing_recall": int(summary["passes_recall_target"].sum()),
        "passing_fp": int(summary["passes_fp_target"].sum()),
        "passing_both": int((summary["passes_recall_target"] & summary["passes_fp_target"]).sum()),
    }
    with (args.out_dir / "proposal_targets.json").open("w") as f:
        json.dump(targets, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
