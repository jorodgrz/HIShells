# src/eval/viz_pv_unet.py
from __future__ import annotations
import re
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from src.utils.config import resolve_config
from src.pv.dataset import build_dataset, estimate_steps



def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_numpy(t):
    try:
        return t.numpy()
    except AttributeError:
        return np.asarray(t)


def _load_names(manifest_path: Path | None, split: str) -> list[str] | None:
    """
    Load one PV filename per line from a manifest file.
    Defaults to data/splits/<split>.txt if not provided.
    Returns None if file not found or empty.
    """
    if manifest_path is None:
        manifest_path = Path("data/splits") / f"{split}.txt"
    if not manifest_path.exists():
        print(f"[viz] no manifest found at {manifest_path} — will use index-based names.")
        return None
    names = [ln.strip() for ln in manifest_path.read_text().splitlines() if ln.strip()]
    if not names:
        print(f"[viz] manifest {manifest_path} is empty — will use index-based names.")
        return None
    print(f"[viz] loaded {len(names)} names from {manifest_path}")
    return names


def _safe_stub(name: str) -> str:
    """Make a filesystem-safe short stub from a filename."""
    # take the basename without directories
    base = Path(name).name
    # drop any extension like .npy
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    # replace non-word chars with underscores
    base = re.sub(r"[^\w\-\.]+", "_", base)
    # trim long stubs
    return base[:120] if len(base) > 120 else base


def visualize(
    cfg_path: str,
    run_dir: str,
    split: str = "test",
    outdir: str | None = None,
    batch_size: int = 4,
    thresh: float = 0.5,
    limit: int = 24,
    seed: int = 1234,
    manifest: str | None = None,
):
    """
    Make overlays: PV (grayscale) + predicted (solid) + ground-truth (dashed).
    File names and titles will include the PV filename from the split manifest.
    """
    resolve_config(cfg_path, write_resolved=False)  # keeps behavior consistent

    run_dir = Path(run_dir)
    outdir = Path(outdir) if outdir is not None else run_dir / f"vis_{split}"
    _ensure_dir(outdir)

    # Load best or final model
    model_path = run_dir / "best_model.keras"
    if not model_path.exists():
        model_path = run_dir / "final_model.keras"
    print(f"[viz] loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    # Dataset (non-repeating for deterministic pass)
    print(f"[viz] preparing dataset split={split}, batch_size={batch_size}")
    ds = build_dataset(cfg_path, split, batch_size=batch_size, seed=seed, repeat=False)
    steps = estimate_steps(cfg_path, split, batch_size)
    names = _load_names(Path(manifest) if manifest else None, split)

    saved = 0
    idx_global = 0
    for step, (x_batch, y_batch) in enumerate(ds):
        x = _to_numpy(x_batch)  # (B, V, S, 1)
        y = _to_numpy(y_batch)  # (B, V, S, 1)

        p = model.predict(x, verbose=0)
        p_bin = (p >= thresh).astype(np.uint8)

        B = x.shape[0]
        for b in range(B):
            if saved >= limit:
                print(f"[viz] wrote {saved} figures to {outdir}")
                return

            # Pick name for this item
            if names is not None and idx_global < len(names):
                pretty = names[idx_global]
                stub = _safe_stub(pretty)
            else:
                pretty = f"{split}_{idx_global:04d}"
                stub = pretty

            img, gt, pred, score = x[b, :, :, 0], y[b, :, :, 0], p_bin[b, :, :, 0], p[b, :, :, 0]

            # Overlay figure
            fig = plt.figure(figsize=(7, 3))
            ax = fig.add_subplot(111)
            ax.imshow(img, cmap="gray", origin="lower", interpolation="nearest")
            ax.contour(pred, levels=[0.5], linewidths=1.2)  # predicted (solid)
            if (gt > 0.5).any():
                ax.contour(gt, levels=[0.5], linewidths=1.2, linestyles="--")  # GT (dashed)

            ax.set_title(f"{pretty}  (thr={thresh:.2f})")
            ax.set_xlabel("Position (px)")
            ax.set_ylabel("Velocity (px)")
            ax.set_xlim(0, img.shape[1]-1)
            ax.set_ylim(0, img.shape[0]-1)
            fig.tight_layout()
            fig.savefig(outdir / f"{stub}.png", dpi=150)
            plt.close(fig)

            # Probability heatmap
            heat_fig = plt.figure(figsize=(7, 3))
            heat_ax = heat_fig.add_subplot(111)
            im = heat_ax.imshow(score, origin="lower", interpolation="nearest")
            heat_ax.set_title(f"Prob map — {pretty}")
            heat_ax.set_xlabel("Position (px)")
            heat_ax.set_ylabel("Velocity (px)")
            heat_fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
            heat_fig.tight_layout()
            heat_fig.savefig(outdir / f"{stub}_prob.png", dpi=150)
            plt.close(heat_fig)

            saved += 1
            idx_global += 1

        if steps is not None and (step + 1) >= steps:
            break

    print(f"[viz] wrote {saved} figures to {outdir}")

def plot_overlay(img, y_pred_mask, y_true_mask, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray', origin='lower')

    # Predicted shells: solid red
    pred_contours = plt.contour(
        y_pred_mask.squeeze(), levels=[0.5],
        colors='red', linewidths=1.5, linestyles='solid'
    )

    # Ground-truth shells: dashed blue
    true_contours = plt.contour(
        y_true_mask.squeeze(), levels=[0.5],
        colors='blue', linewidths=1.5, linestyles='dashed'
    )

    ax.set_title("Overlay: red=model prediction, blue=ground truth")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--manifest", default=None, help="Optional path to a split manifest (one PV filename per line). Defaults to data/splits/<split>.txt")
    args = ap.parse_args()

    visualize(
        cfg_path=args.config,
        run_dir=args.run_dir,
        split=args.split,
        outdir=args.outdir,
        batch_size=args.batch_size,
        thresh=args.thresh,
        limit=args.limit,
        seed=args.seed,
        manifest=args.manifest,
    )