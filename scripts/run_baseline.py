"""Run the classical baselines under the same LOGO geometry (plan §11 step 11).

Drives :mod:`hishells.baselines.{trivial,mtb,casi}` through the same
:class:`hishells.data.LOGOSplitter` used by ``scripts/train_logo.py``,
so each baseline produces one row per ``(name, fold)`` in
``results/ablations.csv`` directly comparable with the CNN folds.

Per fold:

1. Build the per-fold window table (positives + sampled negatives,
   identical to the CNN code path).
2. Score every row with the chosen baseline; treat the within-fold
   *validation* split as the operating-point selector exactly like
   the CNN run.
3. Hand the test scores + val scores to
   :func:`hishells.eval.compute_metrics` and append the row.

CASI is optional: if ``CASI_HOME`` is unset the corresponding fold
rows are written with ``threshold=NaN`` and ``notes='CASI unavailable'``
so the LOGO sweep still completes for the other baselines.

Usage:

    python scripts/run_baseline.py --baseline trivial --name trivial
    python scripts/run_baseline.py --baseline mtb --name mtb_v1
    python scripts/run_baseline.py --baseline casi --name casi_v1
    python scripts/run_baseline.py --baseline all --name baseline_v1
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hishells.baselines import casi as casi_mod  # noqa: E402
from hishells.baselines import mtb as mtb_mod  # noqa: E402
from hishells.baselines import trivial as trivial_mod  # noqa: E402
from hishells.catalog import LOGO_GALAXIES_19, load_catalog  # noqa: E402
from hishells.cubes import sigma_rms  # noqa: E402
from hishells.data import CubeStore, LOGOSplitter  # noqa: E402
from hishells.eval import METRIC_COLUMNS, compute_metrics  # noqa: E402
from hishells.windows import NegSampleConfig, build_window_table  # noqa: E402


# ---------------------------------------------------------------------------
# Per-baseline scoring
# ---------------------------------------------------------------------------


def _score_trivial(table, cubes, sigma_rms_by_galaxy, **_):
    return trivial_mod.score_table(
        table, cubes, sigma_rms_by_galaxy=sigma_rms_by_galaxy
    )


def _score_mtb(table, cubes, **_):
    return mtb_mod.score_table(table, cubes)


def _score_casi(table, cubes, *, mom0_dir=None, casi_cli=None, casi_extra=None, **_):
    return casi_mod.score_table(
        table,
        cubes,
        mom0_dir=mom0_dir,
        cli=casi_cli,
        extra_args=casi_extra,
    )


_SCORERS = {
    "trivial": _score_trivial,
    "mtb": _score_mtb,
    "casi": _score_casi,
}


# ---------------------------------------------------------------------------
# CSV writer (matches scripts/train_logo.py exactly)
# ---------------------------------------------------------------------------


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRIC_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


def _placeholder_row(name: str, fold: str, n_train: int, n_test: int, seed: int, note: str) -> dict:
    return {
        "name": name,
        "fold": fold,
        "threshold": float("nan"),
        "recall_at_op": float("nan"),
        "fp_per_galaxy_at_op": -1,
        "precision_at_op": float("nan"),
        "F1": float("nan"),
        "AUC_PR": float("nan"),
        "AUC_ROC": float("nan"),
        "ECE": float("nan"),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "seed": int(seed),
        "notes": note,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--baseline", choices=list(_SCORERS) + ["all"], default="trivial")
    ap.add_argument("--name", required=True, help="Ablation row name (e.g. 'trivial_v1', 'mtb_v1', 'baseline_all').")
    ap.add_argument("--catalog-dir", type=Path, default=REPO / "Data" / "J_AJ_141_23")
    ap.add_argument("--cube-dir", type=Path, default=REPO / "Data" / "THINGS")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--galaxies", nargs="*", default=None, help="Subset of LOGO galaxies (default: all 19).")
    ap.add_argument("--hole-types", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--neg-hard-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-folds", type=int, default=None)
    ap.add_argument("--mom0-dir", type=Path, default=None, help="Cache dir for per-galaxy MOM0 FITS (CASI only).")
    ap.add_argument("--casi-cli", type=Path, default=None, help="Override path to the CASI predict script.")
    ap.add_argument("--casi-extra", nargs="*", default=None, help="Extra CLI args forwarded to CASI.")
    return ap.parse_args(argv)


def _baselines(args) -> list[str]:
    return list(_SCORERS) if args.baseline == "all" else [args.baseline]


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
    if not galaxies:
        print("!! no galaxies have downloaded cubes; aborting", file=sys.stderr)
        return 1
    print(f"-- baselines={_baselines(args)} on {len(galaxies)} galaxies, {len(holes)} positives")

    cubes = CubeStore(args.cube_dir, max_cubes=2)

    sigma_rms_by_galaxy: dict[str, float] = {}
    for g in galaxies:
        sigma_rms_by_galaxy[g] = sigma_rms(cubes(g))

    cubes_by_g = {g: cubes(g) for g in galaxies}
    pos_table = holes[holes["galaxy_id"].isin(galaxies)]
    neg_cfg = NegSampleConfig(
        ratio=args.neg_ratio, hard_frac=args.neg_hard_frac, rng_seed=args.seed
    )
    full_table = build_window_table(
        pos_table, cubes_by_g, neg_cfg, cube_sigmas=sigma_rms_by_galaxy
    )
    splitter = LOGOSplitter(
        full_table, galaxies=galaxies, val_frac=0.10, rng_seed=args.seed
    )

    csv_path = args.results_dir / "ablations.csv"
    casi_cache: dict[str, np.ndarray] = {}

    folds_run = 0
    for fold in splitter:
        if args.limit_folds is not None and folds_run >= args.limit_folds:
            break
        folds_run += 1
        print(f"\n=== fold {folds_run}/{len(galaxies)}: {fold.test_galaxy} ===")

        train_table = full_table.iloc[fold.train_idx].reset_index(drop=True)
        val_table = full_table.iloc[fold.val_idx].reset_index(drop=True)
        test_table = full_table.iloc[fold.test_idx].reset_index(drop=True)

        for bname in _baselines(args):
            row_name = args.name if args.baseline != "all" else f"{args.name}__{bname}"
            scorer = _SCORERS[bname]
            try:
                test_scores, test_labels = scorer(
                    test_table,
                    cubes,
                    sigma_rms_by_galaxy=sigma_rms_by_galaxy,
                    mom0_dir=args.mom0_dir,
                    casi_cli=args.casi_cli,
                    casi_extra=args.casi_extra,
                    cache=casi_cache if bname == "casi" else None,
                )
                # Validation pool = train_pos held out + a small slice of
                # train_negs, matching the CNN's val composition (positives
                # only) but with negatives appended so PR-curve thresholds
                # span both classes.
                val_pos = val_table
                if (train_table["label"] == 0).any():
                    n_val_neg = min(len(val_pos) * 5, int((train_table["label"] == 0).sum()))
                    val_neg = train_table[train_table["label"] == 0].sample(
                        n=n_val_neg, random_state=args.seed
                    )
                else:
                    val_neg = val_pos.iloc[0:0]
                val_combined = pd.concat([val_pos, val_neg], ignore_index=True)
                val_scores, val_labels = scorer(
                    val_combined,
                    cubes,
                    sigma_rms_by_galaxy=sigma_rms_by_galaxy,
                    mom0_dir=args.mom0_dir,
                    casi_cli=args.casi_cli,
                    casi_extra=args.casi_extra,
                    cache=casi_cache if bname == "casi" else None,
                )

                # MTB convention: report rho/rho_99 using negatives in
                # the *validation* set (we don't have access to the test
                # set's negatives at decision time).
                if bname == "mtb":
                    test_scores = mtb_mod.normalise_by_rho_99(test_scores, test_labels)
                    val_scores = mtb_mod.normalise_by_rho_99(val_scores, val_labels)

                row = compute_metrics(
                    scores=test_scores,
                    labels=test_labels,
                    val_scores=val_scores,
                    val_labels=val_labels,
                    name=row_name,
                    fold=fold.test_galaxy,
                    n_train=int(fold.train_idx.size),
                    seed=args.seed,
                )
            except casi_mod.CASINotInstalledError as exc:
                print(f"   {bname}: SKIP -- {exc}")
                row = _placeholder_row(
                    row_name,
                    fold.test_galaxy,
                    n_train=int(fold.train_idx.size),
                    n_test=int(fold.test_idx.size),
                    seed=args.seed,
                    note="CASI unavailable",
                )
            except Exception as exc:  # noqa: BLE001 -- baseline failures shouldn't crash the sweep
                print(f"   {bname}: ERROR -- {exc}")
                row = _placeholder_row(
                    row_name,
                    fold.test_galaxy,
                    n_train=int(fold.train_idx.size),
                    n_test=int(fold.test_idx.size),
                    seed=args.seed,
                    note=f"{bname} failed: {exc}",
                )
            append_csv_row(csv_path, row)
            print(
                f"   {bname:7s}: recall@op={row.get('recall_at_op')!s:>5s} "
                f"fp/gal={row.get('fp_per_galaxy_at_op')!s:>3s} AUC_PR={row.get('AUC_PR')!s:>5s}"
            )

    print(f"\nDone. Wrote {folds_run} folds × {len(_baselines(args))} baselines to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
