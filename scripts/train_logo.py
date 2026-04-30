"""Run a leave-one-galaxy-out (LOGO) sweep across the 19 B11 galaxies.

Each fold:

1. Build the per-fold window table (positives from B11 + negatives
   sampled per :class:`hishells.windows.NegSampleConfig`).
2. Construct ``ShellWindowDataset`` for train / val / test using the
   :class:`hishells.data.LOGOSplitter` indices.
3. Train one model with :func:`hishells.train.train_one_fold`.
4. Score the test galaxy + the within-fold val set, compute the
   14-column metric dict via :func:`hishells.eval.compute_metrics`,
   append it to ``results/ablations.csv``.
5. Save the model checkpoint to ``results/checkpoints/<name>/<fold>.pt``.

CLI flags expose every \u00a79 ablation knob (loss, neg ratio, hole
types, augmentation toggles, architecture) so a single shell loop can
fan out the experiment grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hishells.augment import AugmentConfig, no_augment  # noqa: E402
from hishells.catalog import LOGO_GALAXIES_19, load_catalog  # noqa: E402
from hishells.cubes import sigma_rms  # noqa: E402
from hishells.data import (  # noqa: E402
    CubeStore,
    DatasetConfig,
    LOGOSplitter,
    ShellWindowDataset,
    make_subset,
)
from hishells.eval import METRIC_COLUMNS, compute_metrics  # noqa: E402
from hishells.loss import build_loss  # noqa: E402
from hishells.model import build_model  # noqa: E402
from hishells.train import (  # noqa: E402
    TrainConfig,
    predict_dataset,
    save_checkpoint,
    train_one_fold,
)
from hishells.windows import NegSampleConfig, build_window_table  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--name", required=True, help="Ablation row name (e.g. v1_baseline).")
    ap.add_argument("--catalog-dir", type=Path, default=REPO / "Data" / "J_AJ_141_23")
    ap.add_argument("--cube-dir", type=Path, default=REPO / "Data" / "THINGS")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--galaxies", nargs="*", default=None, help="Subset of LOGO galaxies (default: all 19).")
    ap.add_argument("--hole-types", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--neg-hard-frac", type=float, default=0.75)
    ap.add_argument("--model", default="small", choices=["small", "resnet18"])
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true", help="Disable all augmentations (\u00a79 Row 5).")
    ap.add_argument("--no-mixed-precision", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if omitted).")
    ap.add_argument("--limit-folds", type=int, default=None, help="Only run the first N folds (debug).")
    return ap.parse_args(argv)


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRIC_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", module="astropy")

    cat = load_catalog(args.catalog_dir)
    holes = cat.filter(
        hole_types=tuple(args.hole_types),
        downloaded_in=args.cube_dir,
    )
    galaxies = tuple(args.galaxies) if args.galaxies else LOGO_GALAXIES_19
    galaxies = tuple(g for g in galaxies if g in set(holes["galaxy_id"]))
    print(f"-- training '{args.name}' on {len(galaxies)} galaxies, {len(holes)} positives")

    cubes = CubeStore(args.cube_dir, max_cubes=2)

    # Pre-compute sigma_rms per galaxy once (cheap relative to training).
    sigma_rms_by_galaxy = {}
    for g in galaxies:
        sigma_rms_by_galaxy[g] = sigma_rms(cubes(g))

    aug = no_augment() if args.no_augment else AugmentConfig()
    ds_cfg = DatasetConfig(window_pix=64, augment=aug, rng_seed=args.seed)
    val_ds_cfg = DatasetConfig(window_pix=64, augment=no_augment(), rng_seed=args.seed)

    csv_path = args.results_dir / "ablations.csv"
    ckpt_dir = args.results_dir / "checkpoints" / args.name
    log_dir = args.results_dir / "logs" / args.name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build the full per-galaxy window table once (deterministic via seed).
    cubes_by_g = {g: cubes(g) for g in galaxies}
    pos_table = holes[holes["galaxy_id"].isin(galaxies)]
    neg_cfg = NegSampleConfig(
        ratio=args.neg_ratio,
        hard_frac=args.neg_hard_frac,
        rng_seed=args.seed,
    )
    full_table = build_window_table(pos_table, cubes_by_g, neg_cfg, cube_sigmas=sigma_rms_by_galaxy)
    splitter = LOGOSplitter(
        full_table,
        galaxies=galaxies,
        val_frac=0.10,
        rng_seed=args.seed,
    )

    folds_run = 0
    for fold in splitter:
        if args.limit_folds is not None and folds_run >= args.limit_folds:
            break
        folds_run += 1
        print(f"\n=== fold {folds_run}/{len(galaxies)}: {fold.test_galaxy} ===")

        train_pos = fold.train_idx[full_table["label"].values[fold.train_idx] == 1].size
        train_neg = fold.train_idx[full_table["label"].values[fold.train_idx] == 0].size
        loss_fn = build_loss(
            args.loss,
            n_pos=train_pos,
            n_neg=train_neg,
            label_smoothing=args.label_smoothing,
        )
        model = build_model(args.model)

        parent_ds = ShellWindowDataset(
            table=full_table,
            cubes=cubes,
            sigma_rms_by_galaxy=sigma_rms_by_galaxy,
            config=ds_cfg,
        )
        val_parent_ds = ShellWindowDataset(
            table=full_table,
            cubes=cubes,
            sigma_rms_by_galaxy=sigma_rms_by_galaxy,
            config=val_ds_cfg,
        )
        train_set = make_subset(parent_ds, fold.train_idx)
        val_set = make_subset(val_parent_ds, fold.val_idx)
        test_set = make_subset(val_parent_ds, fold.test_idx)

        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stop_patience=args.patience,
            num_workers=args.num_workers,
            device=args.device,
            mixed_precision=not args.no_mixed_precision,
            log_every=5,
            seed=args.seed,
        )
        model, fold_result = train_one_fold(
            model, train_set, val_set, loss_fn, config=cfg
        )

        # Test-set scoring + metrics.
        test_scores, test_labels, _ = predict_dataset(model, test_set, batch_size=cfg.batch_size)
        val_scores = (
            fold_result.val_scores
            if fold_result.val_scores is not None
            else np.zeros(0)
        )
        val_labels = (
            fold_result.val_labels
            if fold_result.val_labels is not None
            else np.zeros(0)
        )
        row = compute_metrics(
            scores=test_scores,
            labels=test_labels,
            val_scores=val_scores,
            val_labels=val_labels,
            name=args.name,
            fold=fold.test_galaxy,
            n_train=int(fold.train_idx.size),
            seed=args.seed,
        )
        append_csv_row(csv_path, row)
        save_checkpoint(
            ckpt_dir / f"{fold.test_galaxy}.pt",
            model=model,
            config=cfg,
            fold=fold.test_galaxy,
            seed=args.seed,
            extras={"history": [r.__dict__ for r in fold_result.history]},
        )
        with (log_dir / f"{fold.test_galaxy}.json").open("w") as f:
            json.dump({"row": row, "best_val_pr_auc": fold_result.best_val_pr_auc}, f, indent=2)
        print(
            f"   -> recall@op={row['recall_at_op']:.2f} fp/galaxy={row['fp_per_galaxy_at_op']} "
            f"AUC_PR={row['AUC_PR']:.3f}"
        )

    print(f"\nDone. Wrote {folds_run} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
