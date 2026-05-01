from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import resolve_config
from src.utils.io import load_yaml, dumps_json
from src.utils.wcs_tools import open_cube, radec_to_xy, pixel_scales_arcsec, velocity_axis_kms

__VERSION__ = "label_pv.py@pv-oval-broadcast-1.0"

TARGETGALAXY = "NGC 628"

# -----------------------------
# Config loader
# -----------------------------
def load_cfg(cfg_path: str) -> Dict:
    res = Path("data_NGC628/_resolved_config.yaml")
    return load_yaml(res) if res.exists() else resolve_config(cfg_path, write_resolved=True)


# -----------------------------
# Bagetakos Table 7 parser
# -----------------------------
def _hms_to_deg(h: int, m: int, s: float) -> float:
    return 15.0 * (float(h) + float(m) / 60.0 + float(s) / 3600.0)

def _dms_to_deg(sign_char, d: int, m: int, s: float) -> float:
    sign = -1.0 if (isinstance(sign_char, str) and sign_char.strip() == "-") else 1.0
    return sign * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)

def load_bagetakos_table7(path: str, target_galaxy: str, keep_types=(2,3)) -> pd.DataFrame:
    colspecs = [
        (0, 11),(12, 15),(16, 18),(19, 21),(22, 26),
        (27, 28),(28, 30),(31, 33),(34, 38),
        (39, 43),(44, 45),(46, 50),(51, 53),(54, 57),
        (58, 61),(62, 66),(67, 71),(72, 75),(76, 80),(81, 85),
    ]
    names = ["Name","Seq","RAh","RAm","RAs","DEsign","DEd","DEm","DEs",
             "HV","Type","d_pc","Vexp","PA","Ratio","R_kpc","nHI","tkin","logE","logMHI"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None)

    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Name"].str.upper() == str(target_galaxy).upper()].copy()

    df["Type"] = pd.to_numeric(df["Type"], errors="coerce")
    df = df[df["Type"].isin(keep_types)].copy()

    for col in ("RAh","RAm","DEd","DEm"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("RAs","DEs","Ratio"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ra_deg"]  = [_hms_to_deg(h, m, s) for h, m, s in zip(df["RAh"], df["RAm"], df["RAs"])]
    df["dec_deg"] = [_dms_to_deg(sign, d, m, s) for sign, d, m, s in zip(df["DEsign"], df["DEd"], df["DEm"], df["DEs"])]

    for col in ("PA","HV","Vexp","d_pc"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Ratio"] = df["Ratio"].fillna(1.0).clip(lower=1e-3)

    return df[["Name","Seq","Type","ra_deg","dec_deg","PA","HV","Vexp","d_pc","Ratio"]].reset_index(drop=True)


# -----------------------------
# Utility: velocity index band (vector-friendly)
# -----------------------------
def vel_band_indices_mask(v_axis: np.ndarray, hv: float, vhalf_by_pos: np.ndarray, dilate_ch: int = 0) -> np.ndarray:
    """
    Build a boolean mask of shape (V, S) where True iff |v - hv| <= v_half[pos].
    vhalf_by_pos: shape (S,), non-negative. Will broadcast to (V, S).
    Optionally dilate by +/- 'dilate_ch' channels after discretization.
    """
    # Continuous mask with broadcasting
    V = v_axis[:, None]  # (V,1)
    cont = np.abs(V - hv) <= (vhalf_by_pos[None, :])  # (V,S)

    if dilate_ch <= 0:
        return cont

    # Discretize & dilate by shifting masks up/down in velocity index
    # Build a simple max over shifted versions
    out = cont.copy()
    for k in range(1, dilate_ch + 1):
        up = np.zeros_like(out, dtype=bool)
        dn = np.zeros_like(out, dtype=bool)
        up[:-k, :] = cont[k:, :]
        dn[k:, :]  = cont[:-k, :]
        out |= up | dn
    return out


# -----------------------------
# Core: vectorized ellipse/circle painting in PV space
# -----------------------------
def paint_shell_on_slice(
    posxy: np.ndarray,              # (S,2) sample coordinates along the PV slice
    meta: Dict,                     # contains center_pix, dir_pix, perp_pix
    v_axis: np.ndarray,             # (V,)
    xc: float, yc: float,           # shell center in image pixels
    a_pix: float, b_pix: float,     # ellipse semi-axes (pix) (if circle, a=b)
    pa_deg: float,                  # ellipse PA in image pixel frame (deg, CCW from +x)
    hv: float,                      # systemic (catalog) velocity (cube units after scaling)
    vexp: float | None,             # expansion velocity (km/s scaled to cube units)
    vexp_fallback_channels: int,    # used only if vexp is missing/<=0
    dilate_vel_channels: int,       # small dilation along velocity axis (indices)
) -> np.ndarray:
    """
    Return an (V,S) boolean mask for this *one* shell on this *one* PV slice,
    by evaluating the inequality |v - hv| <= vexp * sqrt(1 - rho^2) at all positions.
    """
    S = posxy.shape[0]
    V = v_axis.shape[0]
    mask = np.zeros((V, S), dtype=bool)

    # Rotation terms to ellipse-aligned (x′ along major axis)
    th = np.deg2rad(pa_deg)
    c, s = np.cos(th), np.sin(th)

    # Coordinates of slice samples relative to ellipse center
    dx = posxy[:, 0] - xc  # (S,)
    dy = posxy[:, 1] - yc  # (S,)

    # Rotate to ellipse frame
    xp =  c * dx + s * dy  # (S,)
    yp = -s * dx + c * dy  # (S,)

    # Ellipse membership
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = (xp / max(a_pix, 1e-6)) ** 2 + (yp / max(b_pix, 1e-6)) ** 2  # (S,)

    inside = rho2 <= 1.0
    if not np.any(inside):
        return mask  # nothing to paint on this slice

    # Per-position half-height in velocity space
    dv_native = np.abs(v_axis[1] - v_axis[0]) if len(v_axis) > 1 else 1.0
    if vexp is not None and np.isfinite(vexp) and vexp > 0:
        vhalf = np.zeros(S, dtype=float)
        vhalf[inside] = vexp * np.sqrt(np.maximum(0.0, 1.0 - rho2[inside]))
    else:
        # fallback (channels) but still tapered by sqrt(1-rho2)
        vhalf = np.zeros(S, dtype=float)
        vhalf[inside] = (vexp_fallback_channels * dv_native) * np.sqrt(np.maximum(0.0, 1.0 - rho2[inside]))

    # Broadcast inequality across (V,S)
    mask |= vel_band_indices_mask(v_axis, hv, vhalf, dilate_ch=dilate_vel_channels)
    return mask


# -----------------------------
# Build labels for GRID PV (vectorized)
# -----------------------------
def build_labels_for_grid_pv(
    pv: np.ndarray,
    meta: Dict,
    posxy: np.ndarray,
    shells_pix: List[Dict],
    v_axis: np.ndarray,
    opts: Dict
) -> np.ndarray:
    """
    Vectorized painter: union of all shell masks on this slice.
    opts:
      - min_axis_ratio_circular (float)
      - vexp_fallback_channels (int)
      - dilate_vel_channels (int)
    """
    S = posxy.shape[0]
    V = v_axis.shape[0]
    lab = np.zeros((V, S), dtype=np.uint8)

    min_axis_ratio_circ = float(opts.get("min_axis_ratio_circular", 0.8))
    vexp_fallback_ch    = int(opts.get("vexp_fallback_channels", 3))
    dilate_ch           = int(opts.get("dilate_vel_channels", 1))

    for sh in shells_pix:
        a = float(sh["a_pix"]); b = float(sh["b_pix"])
        xc, yc = float(sh["xc"]), float(sh["yc"])
        pa     = float(sh["pa_deg"])
        hv     = float(sh["hv"]) if np.isfinite(sh["hv"]) else np.nan
        vexp   = sh["vexp"]

        if not (np.isfinite(a) and np.isfinite(b) and a > 1 and b > 1 and np.isfinite(hv)):
            continue

        # If "nearly circular", force a=b to paint circles
        if b / a >= min_axis_ratio_circ:
            b = a

        # Paint one shell
        shell_mask = paint_shell_on_slice(
            posxy=posxy, meta=meta, v_axis=v_axis,
            xc=xc, yc=yc, a_pix=a, b_pix=b, pa_deg=pa,
            hv=hv, vexp=vexp,
            vexp_fallback_channels=vexp_fallback_ch,
            dilate_vel_channels=dilate_ch
        )
        lab[shell_mask] = 1

    return lab


# -----------------------------
# Main
# -----------------------------
def main(cfg: Dict):
    if cfg["galaxy"].get("distance_mpc") is None:
        raise SystemExit("Please add galaxy.distance_mpc to your YAML (needed to convert pc → arcsec).")

    out_lab = Path(cfg["output_root"]) / "labels"; out_lab.mkdir(parents=True, exist_ok=True)
    pv_dir  = Path(cfg["output_root"]) / "pv"
    qa_dir  = Path(cfg["output_root"]) / "qa_labels"; qa_dir.mkdir(parents=True, exist_ok=True)

    # Open cube and axes
    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
    v_axis = velocity_axis_kms(hdr)  # try to produce km/s; may be m/s in some cubes

    # === Catalog HV scale/offset handling ===
    # auto: if |v| medians look like thousands, treat cube as m/s and scale HV by 1000
    cfg_scale = cfg.get("catalogs", {}).get("hv_scale", "auto")
    if isinstance(cfg_scale, str) and cfg_scale.lower() == "auto":
        hv_scale = 1000.0 if np.nanmedian(np.abs(v_axis)) > 1_000.0 else 1.0
    else:
        hv_scale = float(cfg_scale)
    hv_offset = float(cfg.get("catalogs", {}).get("hv_offset_kms", 0.0))

    # Label options
    label_opts = cfg.get("pv", {}).get("label", {}) if "pv" in cfg else {}
    min_axis_ratio_circ = float(label_opts.get("min_axis_ratio_circular", 0.8))
    keep_types = tuple(label_opts.get("keep_types", [2, 3]))
    dilate_vel_channels = int(label_opts.get("dilate_vel_channels", 1))
    vexp_fallback_channels = int(label_opts.get("vexp_fallback_channels", 3))

    # Load catalog
    holes_dat = cfg.get("catalogs", {}).get("holes_dat", None)
    if not holes_dat or not Path(holes_dat).exists():
        raise SystemExit("catalogs.holes_dat missing or not found; needed for labeling.")
    target_gal = (cfg.get("catalogs", {}).get("target_galaxy") or TARGETGALAXY)
    df = load_bagetakos_table7(holes_dat, target_gal, keep_types=keep_types)

    # World→pixel conversions for ellipse sizes
    ax_as, ay_as = pixel_scales_arcsec(wcs)
    pix_per_as = 1.0 / max(ax_as, 1e-9)  # assume near-square beam/pixel scales

    D_mpc = float(cfg["galaxy"]["distance_mpc"])
    D_pc  = D_mpc * 1.0e6

    shells_pix: List[Dict] = []
    for _, row in df.iterrows():
        ra = float(row["ra_deg"]); dec = float(row["dec_deg"])
        x, y = radec_to_xy(wcs, ra, dec)

        if not pd.notna(row["d_pc"]):
            continue

        a_arcsec = ((row["d_pc"] / 2.0) / D_pc) * 206265.0
        b_arcsec = a_arcsec * (row["Ratio"] if pd.notna(row["Ratio"]) else 1.0)
        a_pix = a_arcsec * pix_per_as
        b_pix = b_arcsec * pix_per_as
        pa_deg = float(row["PA"]) if pd.notna(row["PA"]) else 0.0

        hv_raw = float(row["HV"]) if pd.notna(row["HV"]) else np.nan
        hv = hv_raw * hv_scale + hv_offset if np.isfinite(hv_raw) else np.nan

        vexp = float(row["Vexp"]) if pd.notna(row["Vexp"]) else None
        if vexp is not None and np.isfinite(vexp) and hv_scale == 1000.0:
            # If hv scaled to m/s, scale Vexp likewise
            vexp *= 1000.0

        shells_pix.append(dict(
            xc=float(x), yc=float(y),
            a_pix=float(a_pix), b_pix=float(b_pix),
            pa_deg=float(pa_deg),
            hv=float(hv), vexp=(float(vexp) if vexp is not None else None),
            Type=int(row["Type"]) if pd.notna(row["Type"]) else None,
            Seq=int(row["Seq"])  if pd.notna(row["Seq"])  else None
        ))

    print(f"[label_pv] v_axis ~ [{float(v_axis.min()):.1f}, {float(v_axis.max()):.1f}] (cube units)")
    print(f"[label_pv] using hv_scale={hv_scale:g}, hv_offset_kms={hv_offset:+.1f}, "
          f"min_axis_ratio_circular={min_axis_ratio_circ:.2f}, keep_types={keep_types}")
    print(f"[label_pv] catalog rows kept: {len(shells_pix)}")

    # Iterate PV slices (grid only)
    pv_files = sorted([p for p in pv_dir.glob("*.npy") if not p.name.endswith("_posxy.npy")])
    written = 0
    for pv_path in pv_files:
        stem = pv_path.stem
        meta_path = pv_dir / f"{stem}.json"
        pos_path  = pv_dir / f"{stem}_posxy.npy"
        if not meta_path.exists() or not pos_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        if meta.get("type") != "grid":
            continue

        pv   = np.load(pv_path)           # (V, S)
        pos  = np.load(pos_path)          # (S, 2)

        # Build label with vectorized ellipse painter
        lab = build_labels_for_grid_pv(
            pv=pv, meta=meta, posxy=pos, shells_pix=shells_pix, v_axis=v_axis,
            opts=dict(
                min_axis_ratio_circular=min_axis_ratio_circ,
                vexp_fallback_channels=vexp_fallback_channels,
                dilate_vel_channels=dilate_vel_channels
            )
        )

        # Save mask + sidecar
        out_mask = out_lab / f"{stem}.npy"
        np.save(out_mask, lab.astype(np.uint8))
        dumps_json({
            "script": __VERSION__,
            "cfg_hash": cfg["_meta"]["_hash"],
            "pv_file": pv_path.name,
            "label_shape": list(lab.shape),
            "n_shells": int(len(shells_pix)),
            "params": {
                "hv_scale": float(hv_scale),
                "hv_offset_kms": float(hv_offset),
                "min_axis_ratio_circular": float(min_axis_ratio_circ),
                "vexp_fallback_channels": int(vexp_fallback_channels),
                "dilate_vel_channels": int(dilate_vel_channels),
            },
            "note": "Label is union over shells of {(x',v): (x',y') inside ellipse AND |v-HV| <= vexp*sqrt(1-rho^2)}."
        }, out_lab / f"{stem}.json")
        written += 1

        # QA overlay
        try:
            plt.figure(figsize=(7,4))
            plt.imshow(pv, origin="lower", aspect="auto", cmap="gray")
            mm = np.ma.masked_where(lab == 0, lab)
            plt.imshow(mm, origin="lower", aspect="auto", alpha=0.35, cmap="autumn")
            plt.title(f"{stem} : PV label overlay (oval fill)")
            plt.tight_layout()
            plt.savefig(qa_dir / f"{stem}_label_overlay.png", dpi=140)
            plt.close()
        except Exception as e:
            print(f"[label_pv] overlay failed for {stem}: {e}")

    print(f"[label_pv] wrote {written} label masks to {out_lab}")
    print(f"[label_pv] overlays → {qa_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    main(cfg)