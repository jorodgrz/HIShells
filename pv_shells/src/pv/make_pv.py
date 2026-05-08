# src/pv/make_pv.py
# Grid-aligned PV slicer (major/minor axes), with legacy spoke/ring code preserved (commented).
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # go up from src/pv/ → project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.utils.io import load_yaml, dumps_json
from src.utils.config import resolve_config
from src.utils.wcs_tools import (
    open_cube, pixel_scales_arcsec, radec_to_xy,
    velocity_axis_kms,
    unit_vectors_for_pa, rotate_xy
)

__VERSION__ = "make_pv.py@grid-1.1"

# -----------------------------------------------------------------------------
# Config loader (uses resolved config if present)
# -----------------------------------------------------------------------------
def load_cfg(cfg_path: str):
    return resolve_config(cfg_path, write_resolved=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _moment0(cube):
    m0 = np.nansum(cube, axis=0).astype(np.float32)
    mx = np.nanmax(m0) if np.isfinite(m0).any() else 0.0
    if mx > 0:
        m0 = m0 / mx
    return np.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, perp_vec):
    """
    Average spectrum across a thin slit centered at (x,y) perpendicular to the line direction.
    Nearest-neighbor sampling for robustness / speed.
    """
    vdim = cube.shape[0]
    sx, sy = perp_vec
    acc = []
    for k in range(-slit_half, slit_half + 1):
        xi = int(round(x + k * sx))
        yi = int(round(y + k * sy))
        if 0 <= xi < nx and 0 <= yi < ny:
            acc.append(cube[:, yi, xi])
    if not acc:
        return np.zeros(vdim, dtype=np.float32)
    return np.mean(acc, axis=0)

def _draw_endcap(ax, x0, y0, sx, sy, slit_half, lw=1.0):
    x1, y1 = x0 - sy * slit_half, y0 + sx * slit_half
    x2, y2 = x0 + sy * slit_half, y0 - sx * slit_half
    ax.plot([x1, x2], [y1, y2], lw=lw)

# -----------------------------------------------------------------------------
# NEW: Major/Minor Grid PV slicing (with PA convention controls)
# -----------------------------------------------------------------------------
def make_grid_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path):
    """
    Uniform, galaxy-aligned (or RA/Dec) grid of PV slices.

    Returns:
      {"major": n_major, "minor": n_minor, "total": n_total}
      where:
        - "major"  = number of cuts whose *direction* is along galaxy major axis (x′)
        - "minor"  = number of cuts whose *direction* is along galaxy minor axis (y′)
    """
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    v = velocity_axis_kms(hdr)
    ax_as, ay_as = pixel_scales_arcsec(wcs)  # arcsec/pix (x,y)
    ra0, dec0 = cfg["galaxy"]["ra_deg"], cfg["galaxy"]["dec_deg"]
    cx, cy = radec_to_xy(wcs, ra0, dec0)

    grid = cfg["pv"]["grid"]
    frame = str(grid.get("frame", "galaxy")).lower()

    # NEW: orientation controls
    conv = str(grid.get("pa_convention", "astro")).lower()  # "astro" | "image"
    pa_delta = float(grid.get("pa_delta_deg", 0.0))
    pa_eff = float(cfg["galaxy"]["pa_deg"]) + pa_delta

    # extents & steps (arcsec)
    x_extent_as = float(grid["x_extent_arcsec"])
    y_extent_as = float(grid["y_extent_arcsec"])
    x_step_as   = float(grid["x_step_arcsec"])
    y_step_as   = float(grid["y_step_arcsec"])
    margin_as   = float(grid.get("margin_arcsec", 0.0))
    slit_w      = int(grid.get("slit_width_pix", 3))
    slit_half   = max(0, slit_w // 2)

    # sampling stride along the slice in pixels
    pos_step_pix = float(cfg["pv"]["grid"].get("pos_step_pix", 1.0))

    # Define x′ (major) and y′ (minor) directions in image-pixel space
    if frame == "galaxy":
        # dvec_major: along the provided PA; nvec_major: CCW +90°
        dvec_major, nvec_major = unit_vectors_for_pa(pa_eff, convention=conv)
        ux, uy = dvec_major   # x′ (major-axis direction)
        vx, vy = nvec_major   # y′ (minor-axis direction)
        # conversion arcsec -> pix along image x,y axes
        x_pix_per_as = 1.0 / ax_as
        y_pix_per_as = 1.0 / ay_as
    elif frame == "radec":
        # Align with image axes: x′ ≡ +x (RA to the right in pixel space), y′ ≡ +y
        ux, uy = 1.0, 0.0
        vx, vy = 0.0, 1.0
        x_pix_per_as = 1.0 / ax_as
        y_pix_per_as = 1.0 / ay_as
    else:
        raise ValueError(f"pv.grid.frame must be 'galaxy' or 'radec', got {frame!r}")

    # extents/steps in pixels (shrink by margin)
    x_extent_pix = max(0.0, (x_extent_as - margin_as) * x_pix_per_as)
    y_extent_pix = max(0.0, (y_extent_as - margin_as) * y_pix_per_as)
    x_step_pix   = max(1.0, x_step_as * x_pix_per_as)
    y_step_pix   = max(1.0, y_step_as * y_pix_per_as)

    m0 = _moment0(cube)

    def _save_line_pv(
        stem_base,
        x0,
        y0,
        dir_x,
        dir_y,
        perp_x,
        perp_y,
        half_len_pix,
        *,
        axis_name,
        offset_arcsec,
    ):
        # sample positions along the line
        npos = int(2 * half_len_pix / pos_step_pix) + 1
        if npos < 9:
            return False
        ts = np.linspace(-half_len_pix, +half_len_pix, npos, dtype=np.float32)
        xs = x0 + ts * dir_x
        ys = y0 + ts * dir_y

        # clip to in-bounds
        mask = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
        ts = ts[mask]
        xs, ys = xs[mask], ys[mask]
        if xs.size < 8:
            return False

        # build PV
        pv = np.zeros((len(v), xs.size), dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            pv[:, i] = _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, (perp_x, perp_y))

        # save arrays + sidecars
        np.save(out_dir / f"{stem_base}.npy", pv)
        np.save(out_dir / f"{stem_base}_posxy.npy", np.stack([xs, ys], axis=1).astype(np.float32))
        meta = {
            "script": __VERSION__, "cfg_hash": cfg["_meta"]["_hash"],
            "type": "grid",
            "frame": frame,
            "grid_axis": axis_name,
            "offset_arcsec": float(offset_arcsec),
            "pa_convention": conv,
            "pa_eff_deg": float(pa_eff),
            "center_pix": [float(x0), float(y0)],
            "dir_pix": [float(dir_x), float(dir_y)],     # along the line
            "perp_pix": [float(perp_x), float(perp_y)],  # across the slit
            "slit_width_pix": int(slit_w),
            "pos_step_pix": float(pos_step_pix),
            "pos_pix": [float(ts[0]), float(ts[-1]), float(pos_step_pix)],
            "pos_axis_pix": [float(t) for t in ts],
            "npos": int(pv.shape[1]),
            "nv": int(pv.shape[0]),
            "vel_kms": [float(v[0]), float(v[-1]), float(v[1]-v[0])]
        }
        dumps_json(meta, out_dir / f"{stem_base}.json")

        # spatial overlay
        plt.figure(figsize=(6, 5))
        plt.imshow(m0, origin="lower", cmap="gray")
        plt.plot(xs, ys, lw=1.0)
        # endcaps (show slit width at both ends)
        if xs.size >= 2:
            _draw_endcap(plt.gca(), xs[0],  ys[0],  perp_x, perp_y, slit_half, lw=1.0)
            _draw_endcap(plt.gca(), xs[-1], ys[-1], perp_x, perp_y, slit_half, lw=1.0)
        plt.title(stem_base)
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_spatial.png", dpi=140)
        plt.close()

        # PV preview
        plt.figure(figsize=(7, 4))
        plt.imshow(pv, origin="lower", aspect="auto")
        plt.colorbar(); plt.title(f"PV (v×pos): {stem_base}")
        plt.tight_layout()
        plt.savefig(overlays_dir / f"{stem_base}_pv.png", dpi=140)
        plt.close()
        return True

    n_minor, n_major = 0, 0  # minor=lines directed along y′ ; major=lines directed along x′

    # “Vertical” cuts: constant x′; direction is y′ (minor axis)
    # x′ offsets in arcsec -> pixels along the x image axis scaling
    x_positions_pix = np.arange(-x_extent_pix, x_extent_pix + 1e-6, x_step_pix, dtype=np.float32)
    for xprime in x_positions_pix:
        # point on the central line at offset x′ from galaxy center
        x0 = cx + xprime * ux
        y0 = cy + xprime * uy
        # human-friendly arcsec value for naming (approx via x scaling)
        xprime_as = xprime / x_pix_per_as if x_pix_per_as != 0 else 0.0
        stem = f"grid_xp_{int(round(xprime_as))}as"
        if _save_line_pv(
            stem,
            x0,
            y0,
            dir_x=vx,
            dir_y=vy,
            perp_x=ux,
            perp_y=uy,
            half_len_pix=y_extent_pix,
            axis_name="minor",
            offset_arcsec=xprime_as,
        ):
            n_minor += 1

    # “Horizontal” cuts: constant y′; direction is x′ (major axis)
    y_positions_pix = np.arange(-y_extent_pix, y_extent_pix + 1e-6, y_step_pix, dtype=np.float32)
    for yprime in y_positions_pix:
        x0 = cx + yprime * vx
        y0 = cy + yprime * vy
        yprime_as = yprime / y_pix_per_as if y_pix_per_as != 0 else 0.0
        stem = f"grid_yp_{int(round(yprime_as))}as"
        if _save_line_pv(
            stem,
            x0,
            y0,
            dir_x=ux,
            dir_y=uy,
            perp_x=vx,
            perp_y=vy,
            half_len_pix=x_extent_pix,
            axis_name="major",
            offset_arcsec=yprime_as,
        ):
            n_major += 1

    return {"major": n_major, "minor": n_minor, "total": n_major + n_minor}

# -----------------------------------------------------------------------------
# LEGACY (preserved but disabled): spokes & rings
# -----------------------------------------------------------------------------
"""
def make_spoke_pv(...):
    ...
def make_ring_pv(...):
    ...
"""

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(cfg):
    outdir   = Path(cfg["output_root"]) / "pv"
    outdir.mkdir(parents=True, exist_ok=True)
    overlays = Path(cfg["output_root"]) / "qa_pv"
    overlays.mkdir(parents=True, exist_ok=True)

    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])

    counts = make_grid_pv(cube, hdr, wcs, cfg, outdir, overlays)

    # Backward-compatible handling
    if isinstance(counts, int):
        counts = {"major": 0, "minor": 0, "total": counts}

    print(
        f"[make_pv] wrote {counts.get('major',0)} major-axis and "
        f"{counts.get('minor',0)} minor-axis cuts "
        f"(total={counts.get('total', counts.get('major',0)+counts.get('minor',0))}) to {outdir}"
    )
    print(f"[make_pv] overlays saved to {overlays}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    # sanity: require PA & INC for meaningful major/minor axes
    if cfg["galaxy"].get("pa_deg") is None and str(cfg["pv"]["grid"].get("frame","galaxy")).lower()=="galaxy":
        raise SystemExit("galaxy.pa_deg must be set in YAML for grid slicing with frame='galaxy'.")
    if "pv" not in cfg or "grid" not in cfg["pv"]:
        raise SystemExit("pv.grid block missing in YAML.")
    main(cfg)
