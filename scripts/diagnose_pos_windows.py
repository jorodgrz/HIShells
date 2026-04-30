"""One-off diagnostic for the notebook 04 POS-window issue.

Implements all phases of `diagnose_pos_windows_nb04` in a single
script so we can inspect the output PNGs directly. Outputs go to
``HIShells/results/diagnostics/`` (gitignored).

Phase A    -- POS vs NEG panel for NGC 2403 (replicates cell 12).
Phase B.1  -- Gate-2 MOM0 overlay sanity for NGC 2403.
Phase B.2  -- Velocity-frame channel-image diagnostic.
Phase B.3  -- PA sweep (same hole at PA=0 / hole.pa_deg / +90).
Phase B.4  -- Single window with explicit (pos arcsec) vs (vel km/s)
              axes plus colorbar.

Each phase prints a short text summary so the diagnostic is judgable
from the terminal even before viewing the PNGs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hishells.catalog import load_catalog  # noqa: E402
from hishells.cubes import (  # noqa: E402
    Cube,
    channel_width_kms,
    load_cube,
    moment0,
    sigma_rms,
    world_to_pix,
)
from hishells.pvcut import (  # noqa: E402
    extract_window_for_hole,
    window_extent_for_hole,
)
from hishells.windows import (  # noqa: E402
    NegSampleConfig,
    normalize_window,
    sample_negatives,
)

OUT = REPO / "results" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "Data" / "THINGS"
GID = "NGC_2403"


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_state() -> tuple[Cube, pd.DataFrame, float]:
    cube = load_cube(DATA / f"{GID}_NA_CUBE_THINGS.FITS")
    cat = load_catalog(REPO / "Data" / "J_AJ_141_23")
    holes = cat.holes[
        (cat.holes["galaxy_id"] == GID)
        & (cat.holes["hole_type"].isin([2, 3]))
    ].reset_index(drop=True)
    sigma = sigma_rms(cube)
    return cube, holes, sigma


def phase_A_pos_vs_neg(cube: Cube, holes: pd.DataFrame, sigma: float) -> None:
    header("Phase A: NGC 2403 POS vs NEG panel (cell-12 replication)")
    print(f"  type-{{2,3}} holes: {len(holes)} (was 3 for DDO 154)")

    negs = sample_negatives(
        cube, holes, NegSampleConfig(ratio=2.0, rng_seed=0), cube_sigma=sigma
    )
    pos_six = holes.head(6)
    neg_six = negs.head(6)

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.5))
    for ax, (_, h) in zip(axes[0], pos_six.iterrows()):
        win = normalize_window(
            extract_window_for_hole(cube, h.to_dict(), window_pix=96), sigma
        )
        ax.imshow(win, origin="lower", cmap="magma", vmin=-2, vmax=8)
        ax.set_title(f"POS hole {int(h.hole_idx)} t{int(h.hole_type)}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, (_, n) in zip(axes[1], neg_six.iterrows()):
        win = normalize_window(
            extract_window_for_hole(cube, n.to_dict(), window_pix=96), sigma
        )
        ax.imshow(win, origin="lower", cmap="magma", vmin=-2, vmax=8)
        ax.set_title(f"NEG ({n.neg_kind})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{GID}: POS vs NEG, window_pix=96")
    fig.tight_layout()
    fig.savefig(OUT / "phase_A_pos_vs_neg_panel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'phase_A_pos_vs_neg_panel.png'}")

    # Quantitative shell-signature check on the 6 positives:
    # a real shell window has a darker median region with a brighter
    # rim. A useful proxy is "is the central 1/3 darker than the outer
    # 2/3"? If so, signature is at least *present* even if noisy.
    interior_minus_rim = []
    for _, h in pos_six.iterrows():
        win = normalize_window(
            extract_window_for_hole(cube, h.to_dict(), window_pix=96), sigma
        )
        n = win.shape[0]
        c0, c1 = n // 3, 2 * n // 3
        interior = float(np.median(win[c0:c1, c0:c1]))
        rim_mask = np.ones_like(win, dtype=bool)
        rim_mask[c0:c1, c0:c1] = False
        rim = float(np.median(win[rim_mask]))
        interior_minus_rim.append(interior - rim)
        print(
            f"  hole {int(h.hole_idx):4d} t{int(h.hole_type)}: "
            f"interior_median={interior:+.2f} sigma  "
            f"rim_median={rim:+.2f} sigma  "
            f"diff={interior - rim:+.2f}  "
            f"(<0 means cavity-like)"
        )
    print(
        f"  summary: {sum(d < 0 for d in interior_minus_rim)}/6 holes "
        f"show interior < rim (cavity-like)"
    )


def phase_B1_mom0_overlay(cube: Cube, holes: pd.DataFrame) -> None:
    header("Phase B.1: gate-2 MOM0 overlay sanity for NGC 2403")
    m0 = moment0(cube)
    xs, ys = world_to_pix(cube, holes["ra_deg"].values, holes["dec_deg"].values)
    radii_pix = (holes["diameter_arcsec"].values / 2.0) / cube.pixel_scale_arcsec

    # In-cube fraction (sanity: did world_to_pix put them inside the array?)
    H, W = m0.shape
    in_cube = ((xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)).mean()
    print(f"  {len(holes)} type-{{2,3}} holes; {in_cube:.0%} land inside the cube footprint")
    print(f"  mom0 shape = {m0.shape}, pixel scale = {cube.pixel_scale_arcsec:.2f} arcsec/pix")
    print(f"  beam = {cube.beam_bmaj_arcsec:.1f} x {cube.beam_bmin_arcsec:.1f} arcsec")

    # Ratio of mom0 inside the circle vs ring around it. Cavities should
    # have hole_inside / hole_ring < 1 (less integrated emission inside).
    ratios = []
    for x, y, r in zip(xs, ys, radii_pix):
        if not (0 <= x < W and 0 <= y < H):
            continue
        yy, xx = np.indices(m0.shape)
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        inside = m0[d2 <= r**2]
        ring = m0[(d2 > r**2) & (d2 <= (2 * r) ** 2)]
        if inside.size == 0 or ring.size == 0:
            continue
        ratios.append(np.nanmedian(inside) / max(np.nanmedian(ring), 1e-12))
    ratios = np.asarray(ratios)
    print(f"  inside/ring mom0 ratio: median={np.median(ratios):.2f}  ")
    print(f"     (<1 means catalog circles sit in MOM0 cavities, as expected)")
    print(f"     fraction with ratio<1: {(ratios < 1).mean():.0%}")

    fig, ax = plt.subplots(figsize=(11, 9))
    vmin, vmax = np.nanpercentile(m0, [2, 99.5])
    ax.imshow(m0, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    for x, y, r, ht in zip(xs, ys, radii_pix, holes["hole_type"]):
        color = {1: "cyan", 2: "lime", 3: "red"}[int(ht)]
        ax.add_patch(
            Circle((x, y), r, fill=False, edgecolor=color, lw=0.6, alpha=0.8)
        )
    ax.set_title(
        f"{GID} MOM0 with B11 type-{{2,3}} holes overlaid "
        f"(lime=t2, red=t3) -- circles should sit in dark cavities"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "phase_B1_mom0_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'phase_B1_mom0_overlay.png'}")


def phase_B2_channel_diag(cube: Cube, holes: pd.DataFrame) -> None:
    header("Phase B.2: velocity-frame channel diagnostic")

    type3 = holes[holes["hole_type"] == 3].reset_index(drop=True)
    if len(type3) == 0:
        print("  no type-3 holes; skipping")
        return
    h = type3.iloc[0]
    ix = int(np.argmin(np.abs(cube.velocity_kms - h["vel_helio_kms"])))
    chw = float(channel_width_kms(cube))
    print(f"  hole vel_helio = {h['vel_helio_kms']:.2f} km/s")
    print(f"  cube[{ix}] vel  = {cube.velocity_kms[ix]:.2f} km/s")
    print(f"  miss           = {abs(cube.velocity_kms[ix] - h['vel_helio_kms']):.2f} km/s")
    print(f"  channel width  = {chw:.2f} km/s  ({abs(cube.velocity_kms[ix] - h['vel_helio_kms']) / chw:.2f} channels)")
    print(
        f"  cube vel range = [{cube.velocity_kms.min():.1f}, "
        f"{cube.velocity_kms.max():.1f}] km/s"
    )
    print(f"  (B11 publishes heliocentric km/s; THINGS CTYPE3='FELO-HEL')")

    xs, ys = world_to_pix(
        cube, np.array([h["ra_deg"]]), np.array([h["dec_deg"]])
    )
    cx, cy = float(xs[0]), float(ys[0])
    r_pix = (h["diameter_arcsec"] / 2.0) / cube.pixel_scale_arcsec
    half_box = max(int(3 * r_pix), 40)
    H, W = cube.data[ix].shape
    x0 = max(0, int(cx - half_box))
    x1 = min(W, int(cx + half_box))
    y0 = max(0, int(cy - half_box))
    y1 = min(H, int(cy + half_box))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for off, ax in zip([-1, 0, 1], axes):
        ix2 = max(0, min(cube.n_chan - 1, ix + off))
        chan = cube.data[ix2, y0:y1, x0:x1]
        vmin, vmax = np.nanpercentile(chan, [2, 99])
        ax.imshow(chan, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.add_patch(
            Circle(
                (cx - x0, cy - y0),
                r_pix,
                fill=False,
                edgecolor="red",
                lw=2,
            )
        )
        ax.set_title(
            f"chan {ix2} v={cube.velocity_kms[ix2]:.1f} km/s"
            + (" *" if off == 0 else "")
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{GID} hole {int(h.hole_idx)} t{int(h.hole_type)} "
        f"(vel_helio={h['vel_helio_kms']:.0f} km/s, d={h['diameter_arcsec']:.0f}\"); "
        f"red circle should be in a CAVITY at the central panel"
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase_B2_channel_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'phase_B2_channel_diagnostic.png'}")

    # quantitative cavity check
    yy, xx = np.indices(cube.data[ix].shape)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = cube.data[ix][d2 <= r_pix**2]
    ring = cube.data[ix][(d2 > r_pix**2) & (d2 <= (2 * r_pix) ** 2)]
    print(
        f"  channel inside/ring median: "
        f"{np.nanmedian(inside):.4f} / {np.nanmedian(ring):.4f}  "
        f"(ratio={np.nanmedian(inside) / max(np.nanmedian(ring), 1e-12):.2f}; <1 is cavity)"
    )


def phase_B3_pa_sweep(cube: Cube, holes: pd.DataFrame, sigma: float) -> None:
    header("Phase B.3: PA sweep on a known type-3 hole")
    type3 = holes[holes["hole_type"] == 3].dropna(subset=["pa_deg"]).reset_index(drop=True)
    if len(type3) == 0:
        print("  no type-3 holes with PA; skipping")
        return
    h = type3.iloc[0]
    print(f"  hole {int(h.hole_idx)} t{int(h.hole_type)}, B11 PA = {h['pa_deg']:.1f} deg")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, pa in zip(axes, [0.0, float(h["pa_deg"]), float(h["pa_deg"]) + 90.0]):
        h2 = h.to_dict()
        h2["pa_deg"] = pa
        win = normalize_window(
            extract_window_for_hole(cube, h2, window_pix=96), sigma
        )
        ax.imshow(win, origin="lower", cmap="magma", vmin=-2, vmax=8)
        ax.set_title(f"PA={pa:.0f} deg")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{GID} hole {int(h.hole_idx)}: PA sweep (B11 PA={h['pa_deg']:.0f}); "
        f"walls should rotate visibly between panels"
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase_B3_pa_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'phase_B3_pa_sweep.png'}")


def phase_B4_labeled_window(cube: Cube, holes: pd.DataFrame, sigma: float) -> None:
    header("Phase B.4: single window with labeled axes + colorbar")
    type3 = holes[holes["hole_type"] == 3].reset_index(drop=True)
    if len(type3) == 0:
        print("  no type-3 holes; skipping")
        return
    h = type3.iloc[0]
    ext = window_extent_for_hole(
        diameter_arcsec=float(h["diameter_arcsec"]),
        vexp_kms=float(h["vexp_kms"]) if pd.notna(h["vexp_kms"]) else None,
        sigma_gas_kms=float(h["sigma_gas_kms"]),
    )
    win = normalize_window(
        extract_window_for_hole(cube, h.to_dict(), window_pix=96), sigma
    )
    # win.shape = (window_pix, window_pix); axis 0 = position, axis 1 = velocity
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        win,
        origin="lower",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        extent=[
            -ext.vel_extent_kms,
            ext.vel_extent_kms,
            -ext.pos_extent_arcsec,
            ext.pos_extent_arcsec,
        ],
        aspect="auto",
    )
    ax.set_xlabel("velocity offset (km/s)")
    ax.set_ylabel("position along PA (arcsec)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_title(
        f"{GID} hole {int(h.hole_idx)} t{int(h.hole_type)} "
        f"(d={h['diameter_arcsec']:.0f}\", Vexp={h['vexp_kms']:.0f} km/s)"
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("normalized flux (sigma units)")
    fig.tight_layout()
    fig.savefig(OUT / "phase_B4_labeled_window.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'phase_B4_labeled_window.png'}")
    print(
        f"  expected: bright walls near (vel=+/-{h['vexp_kms']:.0f} km/s, "
        f"pos=+/-{h['diameter_arcsec'] / 2:.0f}\"); darker interior."
    )


def main() -> int:
    cube, holes, sigma = load_state()
    print(f"Loaded {GID}: {cube.data.shape} cube, sigma_rms={sigma:.4f}")
    print(f"  {len(holes)} type-{{2,3}} B11 holes for this galaxy")
    phase_A_pos_vs_neg(cube, holes, sigma)
    phase_B1_mom0_overlay(cube, holes)
    phase_B2_channel_diag(cube, holes)
    phase_B3_pa_sweep(cube, holes, sigma)
    phase_B4_labeled_window(cube, holes, sigma)
    print("\nAll phases complete. PNGs in:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
