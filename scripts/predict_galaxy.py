"""Per-galaxy MC-dropout inference (plan §11 step 10 / §7).

Given a single THINGS cube and a trained checkpoint, this script:

1. Enumerates candidate sightlines via
   :func:`hishells.candidates.enumerate_candidates` (MOM0 minima ∪ MTB,
   DBSCAN-deduplicated at the THINGS beam scale).
2. Builds a window table whose rows match the dataset schema in
   :class:`hishells.data.ShellWindowDataset`: each candidate inherits a
   default ``diameter_arcsec`` / ``vexp_kms`` / ``sigma_gas_kms`` so
   the p-v window extents use the same per-galaxy priors that training
   did. Defaults are the median over the B11 type-2/3 holes for the
   target galaxy if the catalog is available, else fall back to the
   plan §1.3 priors (250 pc, 12 km/s).
3. Runs ``T`` MC-dropout passes through the loaded model.
4. Writes a FITS binary table to ``--out`` with the §7 columns
   (``ra``, ``dec``, ``vel``, ``score_mean``, ``score_std``,
   ``score_q05``, ``score_q95``, ``n_passes``, ``galaxy``), plus
   ``score_seed`` and ``source`` carried through from the candidate
   enumerator for downstream debugging.

Usage:

    python scripts/predict_galaxy.py \
        --cube Data/THINGS/NGC_2403_NA_CUBE_THINGS.FITS \
        --checkpoint results/checkpoints/v1_baseline/NGC_2403.pt \
        --out results/per_galaxy/NGC_2403.fits

The checkpoint format is the one written by
:func:`hishells.train.save_checkpoint`. The model architecture is
inferred from the checkpoint's ``config`` (or pass ``--model`` to
override).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from hishells.candidates import enumerate_candidates  # noqa: E402
from hishells.catalog import (  # noqa: E402
    SIGMA_GAS_KMS_BY_STEM,
    THINGS_STEM_TO_NAME,
    load_catalog,
)
from hishells.cubes import load_cube, sigma_rms  # noqa: E402
from hishells.data import (  # noqa: E402
    DatasetConfig,
    ShellWindowDataset,
)
from hishells.model import build_model  # noqa: E402
from hishells.predict import (  # noqa: E402
    MCResult,
    apply_temperature,
    fit_temperature,
    predict_dataset_mc,
    write_candidates_fits,
)
from hishells.train import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Galaxy stem inference + per-galaxy priors
# ---------------------------------------------------------------------------


def _infer_galaxy_id(cube_path: Path) -> str:
    """Strip ``_<weighting>_CUBE_THINGS.FITS`` off a THINGS filename."""

    stem = cube_path.stem
    for suffix in ("_NA_CUBE_THINGS", "_RO_CUBE_THINGS"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _priors_for_galaxy(
    galaxy_id: str,
    catalog_dir: Path | None,
    pixel_scale_arcsec: float,
) -> dict:
    """Median ``(diameter_arcsec, vexp_kms, sigma_gas_kms)`` for a galaxy.

    Falls back to plan §1.3 priors if the catalog is unavailable or the
    galaxy is not represented (e.g. running on a non-THINGS cube). The
    diameter prior is anchored at 250 pc -> arcsec via the per-galaxy
    distance from B11; if we don't have it we use a coarse default of
    20 arcsec (~250 pc at ~3 Mpc).
    """

    sigma_gas = SIGMA_GAS_KMS_BY_STEM.get(galaxy_id, 10.0)
    diameter_arcsec_default = 20.0
    vexp_kms_default = 12.0
    if catalog_dir is None or not catalog_dir.exists():
        return {
            "diameter_arcsec": diameter_arcsec_default,
            "vexp_kms": vexp_kms_default,
            "sigma_gas_kms": sigma_gas,
        }
    try:
        cat = load_catalog(catalog_dir)
    except Exception:
        return {
            "diameter_arcsec": diameter_arcsec_default,
            "vexp_kms": vexp_kms_default,
            "sigma_gas_kms": sigma_gas,
        }
    holes = cat.holes
    sub = holes[(holes["galaxy_id"] == galaxy_id) & (holes["hole_type"].isin([2, 3]))]
    if len(sub) == 0:
        return {
            "diameter_arcsec": diameter_arcsec_default,
            "vexp_kms": vexp_kms_default,
            "sigma_gas_kms": sigma_gas,
        }
    diam_arcsec = float(np.nanmedian(sub["diameter_arcsec"]))
    vexp = float(np.nanmedian(sub["vexp_kms"]))
    return {
        "diameter_arcsec": diam_arcsec if np.isfinite(diam_arcsec) else diameter_arcsec_default,
        "vexp_kms": vexp if np.isfinite(vexp) and vexp > 0 else vexp_kms_default,
        "sigma_gas_kms": sigma_gas,
    }


# ---------------------------------------------------------------------------
# Window-table builder (candidates -> ShellWindowDataset rows)
# ---------------------------------------------------------------------------


def _candidates_to_window_table(
    cands_df: pd.DataFrame,
    galaxy_id: str,
    priors: dict,
) -> pd.DataFrame:
    """Promote the candidates DataFrame to the dataset schema."""

    rows = []
    for i, row in cands_df.reset_index(drop=True).iterrows():
        rows.append(
            {
                "galaxy_id": galaxy_id,
                "hole_idx": -(i + 1),
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "vel_helio_kms": float(row["vel_kms"]),
                "pa_deg": 0.0,
                "diameter_arcsec": float(priors["diameter_arcsec"]),
                "diameter_pc": float("nan"),
                "vexp_kms": float(priors["vexp_kms"]),
                "sigma_gas_kms": float(priors["sigma_gas_kms"]),
                "hole_type": 0,
                "label": 0,
                "neg_kind": "candidate",
                "score_seed": float(row.get("score_seed", float("nan"))),
                "source": str(row.get("source", "candidate")),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cube", type=Path, required=True, help="Path to the THINGS NA cube FITS.")
    ap.add_argument("--mom0", type=Path, default=None, help="Optional pre-computed MOM0 FITS.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.pt).")
    ap.add_argument("--out", type=Path, required=True, help="Output FITS table path.")
    ap.add_argument("--catalog-dir", type=Path, default=REPO / "Data" / "J_AJ_141_23")
    ap.add_argument("--galaxy-id", default=None, help="Override galaxy stem (default: infer from --cube).")
    ap.add_argument("--model", default=None, choices=[None, "small", "resnet18"], help="Model arch (default: from checkpoint).")
    ap.add_argument("--T", dest="n_passes", type=int, default=50, help="Number of MC-dropout passes.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if omitted).")
    ap.add_argument("--temperature", type=float, default=None, help="Pre-fit temperature; if set, scores are calibrated as sigmoid(logit(score) / T).")
    ap.add_argument("--temperature-from", type=Path, default=None, help="JSON file with {'temperature': T} produced by notebook 07.")
    ap.add_argument("--velocity-grid-kms", default="-30,-15,0,15,30", help="Comma-separated velocity offsets relative to systemic for MOM0 candidates.")
    ap.add_argument("--mtb-top-k", type=int, default=200, help="MTB candidates retained per galaxy.")
    ap.add_argument("--mom0-footprint-arcsec", type=float, default=30.0)
    ap.add_argument("--mom0-relative-depth", type=float, default=0.5)
    ap.add_argument("--score-min", type=float, default=0.0, help="Drop output rows with score_mean below this (default: keep all).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", module="astropy")

    galaxy_id = args.galaxy_id or _infer_galaxy_id(args.cube)
    galaxy_label = THINGS_STEM_TO_NAME.get(galaxy_id, galaxy_id)
    print(f"-- predicting on galaxy {galaxy_id} ({galaxy_label}) from {args.cube}")

    cube = load_cube(args.cube)
    sigma = sigma_rms(cube)
    priors = _priors_for_galaxy(
        galaxy_id, args.catalog_dir if args.catalog_dir.exists() else None, cube.pixel_scale_arcsec
    )
    print(f"   priors: diameter={priors['diameter_arcsec']:.1f}\" vexp={priors['vexp_kms']:.1f} km/s sigma_gas={priors['sigma_gas_kms']:.1f}")

    velocity_grid = tuple(float(v) for v in args.velocity_grid_kms.split(",") if v.strip())
    cands_df = enumerate_candidates(
        args.cube,
        mom0_path=args.mom0,
        velocity_grid_kms=velocity_grid,
        mom0_kwargs={
            "footprint_arcsec": args.mom0_footprint_arcsec,
            "relative_depth": args.mom0_relative_depth,
        },
        mtb_kwargs={"top_k": args.mtb_top_k},
    )
    print(f"   {len(cands_df)} candidate sightlines after dedup")
    if len(cands_df) == 0:
        print("   nothing to score; writing empty table")
        write_candidates_fits(
            args.out,
            ra_deg=np.empty(0),
            dec_deg=np.empty(0),
            vel_kms=np.empty(0),
            mc=MCResult(
                score_mean=np.empty(0),
                score_std=np.empty(0),
                score_q05=np.empty(0),
                score_q95=np.empty(0),
                n_passes=int(args.n_passes),
            ),
            galaxy_id=galaxy_id,
        )
        return 0

    table = _candidates_to_window_table(cands_df, galaxy_id, priors)

    # Wrap the already-loaded cube in a tiny callable matching the
    # ``CubeStore`` protocol so :class:`ShellWindowDataset` doesn't try
    # to read the FITS twice.
    class _SingleCubeStore:
        def __init__(self, c):
            self._cube = c

        def __call__(self, galaxy_id):  # noqa: ARG002
            return self._cube

    store_proxy = _SingleCubeStore(cube)

    ds = ShellWindowDataset(
        table=table,
        cubes=store_proxy,
        sigma_rms_by_galaxy={galaxy_id: float(sigma)},
        config=DatasetConfig(window_pix=96, augment=None),
    )

    # Build & load the model. Architecture preference: --model flag,
    # else checkpoint extras['arch'], else 'small'.
    arch = args.model
    if arch is None:
        try:
            tmp = torch.load(args.checkpoint, map_location="cpu")
            arch = (tmp.get("config") or {}).get("arch") or (tmp.get("extras") or {}).get("arch") or "small"
        except Exception:
            arch = "small"
    model = build_model(arch)
    load_checkpoint(args.checkpoint, model)

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"   running {args.n_passes} MC-dropout passes on {device}")
    mc, _, _ = predict_dataset_mc(
        model, ds, T=int(args.n_passes), batch_size=args.batch_size, device=torch.device(device)
    )

    # Optional temperature calibration. We re-derive the logits from
    # MC mean (a small approximation: ``logit(mean(sigmoid(z)))`` is
    # within ~5% of ``mean(z)`` for the regimes we care about, and the
    # per-pass logits aren't kept around).
    T_used = None
    if args.temperature is not None:
        T_used = float(args.temperature)
    elif args.temperature_from is not None and args.temperature_from.exists():
        with args.temperature_from.open() as f:
            T_used = float(json.load(f).get("temperature", 1.0))
    if T_used is not None and T_used != 1.0:
        eps = 1e-6
        logits_eq = np.log(np.clip(mc.score_mean, eps, 1 - eps) / np.clip(1 - mc.score_mean, eps, 1 - eps))
        mc.score_mean = apply_temperature(logits_eq, T_used)
        print(f"   applied temperature scaling T={T_used:.3f}")

    keep = mc.score_mean >= float(args.score_min)
    n_kept = int(keep.sum())
    print(f"   kept {n_kept}/{len(table)} candidates with score_mean >= {args.score_min}")
    if n_kept == 0:
        keep = np.ones(len(mc.score_mean), dtype=bool)

    src_arr = np.array([str(s) for s in table["source"].to_numpy()[keep]], dtype="U8")
    extra_cols = {
        "score_seed": table["score_seed"].to_numpy(dtype=np.float64)[keep],
        "source": src_arr,
    }

    write_candidates_fits(
        args.out,
        ra_deg=table["ra_deg"].to_numpy(dtype=np.float64)[keep],
        dec_deg=table["dec_deg"].to_numpy(dtype=np.float64)[keep],
        vel_kms=table["vel_helio_kms"].to_numpy(dtype=np.float64)[keep],
        mc=type(mc)(
            score_mean=mc.score_mean[keep],
            score_std=mc.score_std[keep],
            score_q05=mc.score_q05[keep],
            score_q95=mc.score_q95[keep],
            n_passes=mc.n_passes,
        ),
        galaxy_id=galaxy_id,
        extra_cols=extra_cols,
    )
    print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
