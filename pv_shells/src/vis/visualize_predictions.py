# src/vis/visualize_predictions.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tensorflow as tf
from tensorflow import keras

# Project imports
from src.utils.config import resolve_config
from src.pv.dataset import build_dataset, estimate_steps


# ----------------------------
# Visualization helpers
# ----------------------------

def _ensure_uint8(img: np.ndarray, lo=2, hi=98) -> np.ndarray:
    """Percentile stretch to [0,255] for nicer grayscale rendering."""
    x = img.astype(np.float32)
    vmin, vmax = np.percentile(x, lo), np.percentile(x, hi)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
    return (x * 255).astype(np.uint8)


def _draw_legend(ax, pred_color, gt_color, lw):
    legend_handles = [
        Line2D([0], [0], color=pred_color, linewidth=lw, linestyle='solid',  label='Prediction'),
        Line2D([0], [0], color=gt_color,   linewidth=lw, linestyle='dashed', label='Ground truth'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=8)


def _safe_contour(
    ax,
    arr: np.ndarray,
    level: float,
    color: str,
    linestyle: str,
    linewidth: float = 2.2,
    label: str | None = None,
    alpha: float = 1.0,
    zorder: int = 10,
) -> bool:
    """
    Draw a single-level contour; return True if anything was drawn.
    Uses high zorder so contours sit above the heatmap/image.
    """
    try:
        cs = ax.contour(
            arr,
            levels=[float(level)],
            colors=[color],
            linestyles=[linestyle],
            linewidths=linewidth,
            origin='lower',
            alpha=alpha,
            zorder=zorder,
        )
        # Check if any paths were produced
        drawn = any(len(coll.get_paths()) > 0 for coll in cs.collections)
        if not drawn:
            for coll in cs.collections:
                coll.remove()
        return drawn
    except Exception:
        return False


def _pick_threshold(prob_map: np.ndarray, user_thr: float, thr_auto: bool) -> float:
    """
    If thr_auto is True and the requested level is degenerate, pick a robust level
    from the prediction’s distribution (e.g., 90th percentile).
    """
    if not thr_auto:
        return float(user_thr)
    # If user_thr is fine (0<thr<1), keep it; else pick p90
    if 0.0 < user_thr < 1.0:
        return float(user_thr)
    # robust fallback
    return float(np.clip(np.percentile(prob_map, 90.0), 0.05, 0.95))


def _plot_overlay(
    pv_img,
    prob_map,
    y_true,
    thr,
    out_path_png,
    title_suffix="",
    pred_color="red",
    gt_color="blue",
    lw=2.2,
    vh_scale=1.6,
):
    """
    PV slice (grayscale) with red solid predicted mask contours and blue dashed GT contours.
    Taller figure (vh_scale) reduces the "long thin rectangle" look.
    """
    fig, ax = plt.subplots(figsize=(7, max(4, 4 * vh_scale)))
    ax.imshow(_ensure_uint8(pv_img), cmap='gray', origin='lower')
    # Contours
    _safe_contour(ax, prob_map, level=thr, color=pred_color, linestyle='solid',  linewidth=lw, alpha=0.95, zorder=10)
    _safe_contour(ax, y_true,    level=0.5, color=gt_color,   linestyle='dashed', linewidth=lw, alpha=0.95, zorder=11)

    _draw_legend(ax, pred_color, gt_color, lw)
    ax.set_title(f"PV overlay {title_suffix} (red=pred, blue=GT, thr={thr:.2f})", fontsize=10)
    ax.set_xlabel("position (S)")
    ax.set_ylabel("velocity (V)")

    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close(fig)


def _plot_probmap(
    prob_map,
    y_true,
    thr,
    out_path_png,
    title_suffix="",
    pred_color="red",
    gt_color="blue",
    lw=2.2,
    vh_scale=1.6,
):
    """
    Probability heatmap (inferno) with overlaid contours: red (pred thr) and blue (GT).
    """
    fig, ax = plt.subplots(figsize=(7, max(4, 4 * vh_scale)))
    im = ax.imshow(prob_map, cmap='inferno', origin='lower', vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(shell)")
    # Contours
    _safe_contour(ax, prob_map, level=thr, color=pred_color, linestyle='solid',  linewidth=lw, alpha=0.95, zorder=10)
    _safe_contour(ax, y_true,    level=0.5, color=gt_color,   linestyle='dashed', linewidth=lw, alpha=0.95, zorder=11)

    _draw_legend(ax, pred_color, gt_color, lw)
    ax.set_title(f"Probability map {title_suffix} (red=pred thr, blue=GT)", fontsize=10)
    ax.set_xlabel("position (S)")
    ax.set_ylabel("velocity (V)")

    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close(fig)


# ----------------------------
# Main routine
# ----------------------------

def visualize(
    cfg_path: str,
    run_dir: str,
    split: str = "test",
    batch_size: int = 4,
    max_items: int | None = None,
    thresh: float = 0.5,
    out_dir: str | None = None,
    pred_color: str = "red",
    gt_color: str = "blue",
    linewidth: float = 2.2,
    vh_scale: float = 1.6,
    thr_auto: bool = False,
):
    """
    Load the best model, run it on the chosen split, and write:
      - *_overlay.png  : PV slice with contours (red=pred, blue=GT)
      - *_prob.png     : Heatmap of probabilities with the same contours

    Parameters
    ----------
    vh_scale : multiply the figure height (use >1.0 to elongate vertical axis)
    thr_auto : if True, auto-picks a robust threshold (p90) when the provided
               --thresh would draw nothing / be unhelpful.
    """
    run_dir = Path(run_dir)
    model_path = run_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _ = resolve_config(cfg_path, write_resolved=False)

    # Dataset: deterministic (no shuffle/repeat); we’ll cap with max_items
    ds = build_dataset(cfg_path, split, batch_size=batch_size, seed=123, repeat=False)
    # If user didn’t set max_items, try to cover the full split approximately
    steps = estimate_steps(cfg_path, split, batch_size)
    total_target = (steps * batch_size) if (max_items is None) else max_items

    model = keras.models.load_model(model_path, compile=False)

    # Output directory
    if out_dir is None:
        out_dir = run_dir / f"vis_{split}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx_global = 0
    for batch_idx, (x, y_true) in enumerate(ds):
        # x: (B, V, S, 1), y_true: (B, V, S, 1)
        prob = model.predict(x, verbose=0)  # (B, V, S, 1)
        x_np = x.numpy()
        y_np = y_true.numpy()
        p_np = prob

        batch_size_eff = x_np.shape[0]
        for i in range(batch_size_eff):
            if saved >= total_target:
                break

            pv_img   = x_np[i, :, :, 0]                      # (V, S)
            prob_map = np.clip(p_np[i, :, :, 0], 0.0, 1.0)   # (V, S)
            gt_mask  = (y_np[i, :, :, 0] > 0.5).astype(np.float32)  # (V, S)

            # Choose the working threshold (optionally auto)
            thr = _pick_threshold(prob_map, thresh, thr_auto)

            base = f"{split}_{idx_global:04d}"
            png_overlay = out_dir / f"{base}_overlay.png"
            png_prob    = out_dir / f"{base}_prob.png"

            _plot_overlay(
                pv_img, prob_map, gt_mask, thr, png_overlay, title_suffix=f"[{base}]",
                pred_color=pred_color, gt_color=gt_color, lw=linewidth, vh_scale=vh_scale
            )
            _plot_probmap(
                prob_map, gt_mask, thr, png_prob, title_suffix=f"[{base}]",
                pred_color=pred_color, gt_color=gt_color, lw=linewidth, vh_scale=vh_scale
            )

            saved += 1
            idx_global += 1

        if saved >= total_target:
            break

    print(f"[vis] wrote {saved} pairs to: {out_dir}")


# ----------------------------
# CLI
# ----------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to pv_config.yaml")
    ap.add_argument("--run_dir", required=True, help="Run directory containing best_model.keras")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=None, help="Cap the number of items to visualize")
    ap.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for prediction contours")
    ap.add_argument("--thr_auto", action="store_true", help="Auto-pick threshold when --thresh is unhelpful")
    ap.add_argument("--out_dir", type=str, default=None, help="Write images here (default: <run_dir>/vis_<split>)")

    # Styling / readability knobs
    ap.add_argument("--pred_color", type=str, default="red")
    ap.add_argument("--gt_color", type=str, default="blue")
    ap.add_argument("--linewidth", type=float, default=2.2)
    ap.add_argument("--vh_scale", type=float, default=1.6, help="Multiply figure height to elongate vertical axis")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize(
        cfg_path=args.config,
        run_dir=args.run_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_items=args.max_items,
        thresh=args.thresh,
        out_dir=args.out_dir,
        pred_color=args.pred_color,
        gt_color=args.gt_color,
        linewidth=args.linewidth,
        vh_scale=args.vh_scale,
        thr_auto=args.thr_auto,
    )