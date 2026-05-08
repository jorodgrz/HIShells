from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.pv.shell_catalog import catalog_to_pixel_shells, load_bagetakos_table7
from src.utils.config import resolve_config
from src.utils.io import dumps_json
from src.utils.wcs_tools import open_cube, unit_vectors_for_pa, velocity_axis_kms

__VERSION__ = "label_pv.py@grid-catalog-types-2.0"


def load_cfg(cfg_path: str) -> Dict:
    return resolve_config(cfg_path, write_resolved=True)


def _velocity_scale(v_axis: np.ndarray, cfg: Dict) -> tuple[float, float]:
    """Return catalog velocity scale/offset needed to match the cube axis."""
    cfg_scale = cfg.get("catalogs", {}).get("hv_scale", "auto")
    if isinstance(cfg_scale, str) and cfg_scale.lower() == "auto":
        hv_scale = 1000.0 if np.nanmedian(np.abs(v_axis)) > 1_000.0 else 1.0
    else:
        hv_scale = float(cfg_scale)
    hv_offset = float(cfg.get("catalogs", {}).get("hv_offset_kms", 0.0))
    return hv_scale, hv_offset


def _velocity_band_mask(
    v_axis: np.ndarray,
    v_center: float,
    vhalf_by_pos: np.ndarray,
    *,
    dilate_ch: int = 0,
) -> np.ndarray:
    """Boolean ``(V,S)`` mask where ``|v - center| <= half_width[pos]``."""
    cont = np.abs(v_axis[:, None] - float(v_center)) <= vhalf_by_pos[None, :]
    if dilate_ch <= 0:
        return cont

    out = cont.copy()
    for k in range(1, int(dilate_ch) + 1):
        up = np.zeros_like(out, dtype=bool)
        dn = np.zeros_like(out, dtype=bool)
        up[:-k, :] = cont[k:, :]
        dn[k:, :] = cont[:-k, :]
        out |= up | dn
    return out


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]


def _ellipse_rho2_for_slice(
    posxy: np.ndarray,
    shell: Dict,
    *,
    pa_convention: str,
    min_axis_ratio_circular: float,
    spatial_pad_fraction: float,
) -> tuple[np.ndarray, float, float]:
    """Project PV cut samples into the shell ellipse frame."""
    a = float(shell["a_pix"])
    b = float(shell["b_pix"])
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return np.full(posxy.shape[0], np.inf), a, b

    if min(a, b) / max(a, b) >= float(min_axis_ratio_circular):
        a = b = max(a, b)

    pad = max(a, b) * float(spatial_pad_fraction)
    a = max(a + pad, 1e-6)
    b = max(b + pad, 1e-6)

    (mx, my), (nx, ny) = unit_vectors_for_pa(float(shell["pa_deg"]), convention=pa_convention)
    dx = posxy[:, 0] - float(shell["xc"])
    dy = posxy[:, 1] - float(shell["yc"])
    major_coord = dx * mx + dy * my
    minor_coord = dx * nx + dy * ny
    rho2 = (major_coord / a) ** 2 + (minor_coord / b) ** 2
    return rho2, a, b


def _paint_one_shell(
    posxy: np.ndarray,
    v_axis: np.ndarray,
    shell: Dict,
    opts: Dict,
) -> tuple[np.ndarray, dict | None, list[dict]]:
    """Create one shell's PV mask and metadata for one grid slice."""
    warnings: list[dict] = []
    shell_type = shell.get("type")
    shell_id = shell.get("shell_id")

    if not np.isfinite(shell.get("vel_center", np.nan)):
        return np.zeros((len(v_axis), posxy.shape[0]), dtype=bool), None, [
            {"shell_id": shell_id, "type": shell_type, "reason": "missing_velocity_center"}
        ]

    rho2, a_pix, b_pix = _ellipse_rho2_for_slice(
        posxy,
        shell,
        pa_convention=str(opts.get("catalog_pa_convention", "astro")),
        min_axis_ratio_circular=float(opts.get("min_axis_ratio_circular", 0.8)),
        spatial_pad_fraction=float(opts.get("spatial_pad_fraction", 0.0)),
    )
    inside = rho2 <= 1.0
    if not np.any(inside):
        return np.zeros((len(v_axis), posxy.shape[0]), dtype=bool), None, []

    dv_native = abs(float(np.nanmedian(np.diff(v_axis)))) if len(v_axis) > 1 else 1.0
    vhalf = np.zeros(posxy.shape[0], dtype=float)
    label_mode = "type1_cavity" if shell_type == 1 else "expanding"

    if shell_type == 1:
        # Type 1 holes do not require a resolved expansion signature. We mark
        # the spatial cavity/gap across the cut, using a local velocity band
        # around the catalog HV. The width is configurable in catalog km/s and
        # scaled into cube units by hv_scale.
        hv_scale = float(opts.get("hv_scale", 1.0))
        half_kms = float(opts.get("type1_velocity_half_width_kms", 10.0)) * hv_scale
        half_channels = opts.get("type1_velocity_half_width_channels")
        if half_channels is not None:
            half_kms = max(half_kms, float(half_channels) * dv_native)
        vhalf[inside] = half_kms
        dilate_ch = int(opts.get("type1_dilate_vel_channels", opts.get("dilate_vel_channels", 1)))
    else:
        vexp = shell.get("vexp", np.nan)
        pad = float(opts.get("expansion_velocity_padding_kms", 0.0)) * float(opts.get("hv_scale", 1.0))
        if np.isfinite(vexp) and float(vexp) > 0:
            vhalf[inside] = (float(vexp) + pad) * np.sqrt(np.maximum(0.0, 1.0 - rho2[inside]))
        else:
            fallback = int(opts.get("vexp_fallback_channels", 3)) * dv_native
            vhalf[inside] = fallback * np.sqrt(np.maximum(0.0, 1.0 - rho2[inside]))
            warnings.append({"shell_id": shell_id, "type": shell_type, "reason": "expanding_shell_used_vexp_fallback"})

        min_channels = int(opts.get("min_vband_channels", 1))
        if min_channels > 0:
            vhalf[inside] = np.maximum(vhalf[inside], 0.5 * min_channels * dv_native)
        dilate_ch = int(opts.get("dilate_vel_channels", 1))

    shell_mask = _velocity_band_mask(
        v_axis,
        float(shell["vel_center"]),
        vhalf,
        dilate_ch=dilate_ch,
    )
    shell_mask[:, ~inside] = False
    box = _bbox(shell_mask)
    if box is None:
        return shell_mask, None, warnings

    center_col = int(np.argmin((posxy[:, 0] - shell["xc"]) ** 2 + (posxy[:, 1] - shell["yc"]) ** 2))
    center_v_idx = int(np.argmin(np.abs(v_axis - float(shell["vel_center"]))))
    inside_cols = np.where(inside)[0]
    obj = {
        "shell_id": None if shell_id is None else int(shell_id),
        "type": None if shell_type is None else int(shell_type),
        "label_mode": label_mode,
        "bbox_vpos": box,
        "center_col": center_col,
        "center_v_idx": center_v_idx,
        "center_velocity": float(shell["vel_center"]),
        "vexp": float(shell["vexp"]) if np.isfinite(shell.get("vexp", np.nan)) else None,
        "spatial_cols": [int(inside_cols.min()), int(inside_cols.max())],
        "projected_radius_cols": int(inside_cols.max() - inside_cols.min() + 1),
        "a_pix_used": float(a_pix),
        "b_pix_used": float(b_pix),
        "mask_pixels": int(shell_mask.sum()),
    }
    return shell_mask, obj, warnings


def build_labels_for_grid_pv(
    posxy: np.ndarray,
    v_axis: np.ndarray,
    shells_pix: List[Dict],
    opts: Dict,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Build binary/type masks plus per-object metadata for one PV slice."""
    lab = np.zeros((len(v_axis), posxy.shape[0]), dtype=np.uint8)
    type_mask = np.zeros_like(lab)
    objects: list[dict] = []
    warnings: list[dict] = []

    for shell in shells_pix:
        shell_mask, obj, shell_warnings = _paint_one_shell(posxy, v_axis, shell, opts)
        warnings.extend(shell_warnings)
        if obj is None:
            continue
        lab[shell_mask] = 1
        if shell.get("type") in (1, 2, 3):
            type_mask[shell_mask] = int(shell["type"])
        objects.append(obj)

    return lab, type_mask, objects, warnings


def _qa_overlay(
    pv: np.ndarray,
    type_mask: np.ndarray,
    objects: list[dict],
    out_path: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.imshow(pv, origin="lower", aspect="auto", cmap="gray")

    colors = {1: "deepskyblue", 2: "orange", 3: "red"}
    for shell_type, color in colors.items():
        mm = np.ma.masked_where(type_mask != shell_type, type_mask)
        if np.ma.count(mm) > 0:
            ax.imshow(mm, origin="lower", aspect="auto", alpha=0.32, cmap=ListedColormap([color]))

    for obj in objects[:20]:
        v0, p0, v1, p1 = obj["bbox_vpos"]
        color = colors.get(obj.get("type"), "white")
        ax.plot([p0, p1, p1, p0, p0], [v0, v0, v1, v1, v0], color=color, lw=0.8)
        ax.plot(
            obj["spatial_cols"],
            [obj["center_v_idx"], obj["center_v_idx"]],
            color=color,
            lw=1.2,
            alpha=0.85,
        )
        ax.scatter([obj["center_col"]], [obj["center_v_idx"]], s=14, color=color, edgecolors="black", linewidths=0.3)
        label = f"T{obj.get('type')}:{obj.get('shell_id')}"
        ax.text(p0, max(0, v1 + 1), label, color=color, fontsize=7)

    ax.set_title(f"{stem} : PV catalog labels")
    ax.set_xlabel("position sample along grid cut")
    ax.set_ylabel("velocity channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(cfg: Dict):
    if cfg["galaxy"].get("distance_mpc") is None:
        raise SystemExit("Please add galaxy.distance_mpc to YAML; Bagetakos d_pc needs distance for arcsec radii.")

    out_root = Path(cfg["output_root"])
    out_lab = out_root / "labels"
    out_type = out_root / "label_types"
    pv_dir = out_root / "pv"
    qa_dir = out_root / "qa_labels"
    for d in (out_lab, out_type, qa_dir):
        d.mkdir(parents=True, exist_ok=True)

    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
    v_axis = velocity_axis_kms(hdr)
    hv_scale, hv_offset = _velocity_scale(v_axis, cfg)

    label_opts = cfg.get("pv", {}).get("label", {}) if "pv" in cfg else {}
    keep_types = label_opts.get("keep_types", [1, 2, 3])

    holes_dat = cfg.get("catalogs", {}).get("holes_dat")
    if not holes_dat or not Path(holes_dat).exists():
        raise SystemExit("catalogs.holes_dat missing or not found; needed for labeling.")
    target_gal = cfg.get("catalogs", {}).get("target_galaxy") or cfg.get("galaxy", {}).get("name")

    catalog = load_bagetakos_table7(holes_dat, target_galaxy=target_gal, keep_types=keep_types)
    shells_pix, cat_diag = catalog_to_pixel_shells(
        catalog,
        wcs=wcs,
        distance_mpc=float(cfg["galaxy"]["distance_mpc"]),
        hv_scale=hv_scale,
        hv_offset=hv_offset,
    )

    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    outside = [
        {"shell_id": sh["shell_id"], "type": sh["type"], "reason": "shell_center_outside_cube"}
        for sh in shells_pix
        if not (0 <= sh["xc"] < nx and 0 <= sh["yc"] < ny)
    ]

    opts = dict(label_opts)
    opts["hv_scale"] = hv_scale

    print(f"[label_pv] v_axis ~ [{float(v_axis.min()):.1f}, {float(v_axis.max()):.1f}]")
    print(f"[label_pv] hv_scale={hv_scale:g}, hv_offset={hv_offset:+.1f}, keep_types={keep_types}")
    print(f"[label_pv] catalog rows kept={len(catalog)}, usable_shells={len(shells_pix)}, by_type={dict(Counter(sh['type'] for sh in shells_pix))}")
    if outside:
        print(f"[label_pv] WARNING: {len(outside)} shell centers are outside cube WCS.")

    pv_files = sorted([p for p in pv_dir.glob("*.npy") if not p.name.endswith("_posxy.npy")])
    written = 0
    positive_slices = 0
    objects_by_type: Counter = Counter()
    unique_shells_by_type: dict[int, set] = defaultdict(set)
    all_warnings = list(cat_diag["warnings"]) + outside

    for pv_path in pv_files:
        stem = pv_path.stem
        meta_path = pv_dir / f"{stem}.json"
        pos_path = pv_dir / f"{stem}_posxy.npy"
        if not meta_path.exists() or not pos_path.exists():
            all_warnings.append({"pv": stem, "reason": "missing_meta_or_posxy"})
            continue

        meta = json.loads(meta_path.read_text())
        if meta.get("type") != "grid":
            continue

        pv = np.load(pv_path)
        posxy = np.load(pos_path)
        lab, type_mask, objects, warnings = build_labels_for_grid_pv(posxy, v_axis, shells_pix, opts)
        all_warnings.extend({"pv": stem, **w} for w in warnings)

        np.save(out_lab / f"{stem}.npy", lab.astype(np.uint8))
        np.save(out_type / f"{stem}.npy", type_mask.astype(np.uint8))

        for obj in objects:
            objects_by_type[obj["type"]] += 1
            unique_shells_by_type[obj["type"]].add(obj["shell_id"])
        if objects:
            positive_slices += 1

        sidecar = {
            "script": __VERSION__,
            "cfg_hash": cfg["_meta"]["_hash"],
            "pv_file": pv_path.name,
            "label_shape": list(lab.shape),
            "n_objects": len(objects),
            "objects": objects,
            "params": {
                "hv_scale": float(hv_scale),
                "hv_offset": float(hv_offset),
                "keep_types": list(keep_types),
                "catalog_pa_convention": opts.get("catalog_pa_convention", "astro"),
                "min_axis_ratio_circular": float(opts.get("min_axis_ratio_circular", 0.8)),
                "type1_velocity_half_width_kms": float(opts.get("type1_velocity_half_width_kms", 10.0)),
                "vexp_fallback_channels": int(opts.get("vexp_fallback_channels", 3)),
            },
            "assumptions": cat_diag["assumptions"],
        }
        dumps_json(sidecar, out_lab / f"{stem}.json")
        _qa_overlay(pv, type_mask, objects, qa_dir / f"{stem}_label_overlay.png", stem)
        written += 1

    summary = {
        "script": __VERSION__,
        "pv_slices_seen": len(pv_files),
        "label_masks_written": written,
        "positive_slices": positive_slices,
        "negative_slices": max(0, written - positive_slices),
        "catalog_rows": len(catalog),
        "usable_shells": len(shells_pix),
        "catalog_shells_by_type": dict(Counter(str(sh["type"]) for sh in shells_pix)),
        "label_objects_by_type": {str(k): int(v) for k, v in objects_by_type.items()},
        "unique_shells_labeled_by_type": {str(k): len(v) for k, v in unique_shells_by_type.items()},
        "warnings": all_warnings,
        "assumptions": cat_diag["assumptions"],
    }
    dumps_json(summary, out_root / "label_summary.json")

    print(f"[label_pv] wrote {written} label masks to {out_lab}")
    print(f"[label_pv] positive_slices={positive_slices}, negative_slices={written - positive_slices}")
    print(f"[label_pv] label objects by type={dict(objects_by_type)}")
    print(f"[label_pv] unique shells labeled by type={ {k: len(v) for k, v in unique_shells_by_type.items()} }")
    print(f"[label_pv] warnings={len(all_warnings)} -> {out_root / 'label_summary.json'}")
    print(f"[label_pv] overlays -> {qa_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    main(cfg)
