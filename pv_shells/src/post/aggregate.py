# src/post/aggregate.py
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, center_of_mass, maximum_filter

import tensorflow as tf
from tensorflow import keras

from astropy.io import fits
from astropy.wcs import WCS

# Project utils
from src.utils.config import resolve_config
from src.utils.io import dumps_json
from src.utils.wcs_tools import open_cube, pixel_scales_arcsec

# ---------------------------
# Helpers
# ---------------------------

def _load_cfg(cfg_path: str) -> Dict:
    # Prefer resolved copy if present
    res = Path("data/_resolved_config.yaml")
    return resolve_config(cfg_path, write_resolved=True) if not res.exists() else json.loads(res.read_text()).copy() if res.suffix==".json" else __import__("yaml").safe_load(res.read_text())

def _safe_load_yaml(path: Path) -> Dict:
    import yaml
    return yaml.safe_load(path.read_text())

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _iter_manifest_items(root: Path, split: str) -> List[str]:
    man = root / "splits" / f"{split}_manifest.txt"
    if not man.exists():
        raise FileNotFoundError(f"Missing manifest: {man}")
    return [ln.strip() for ln in man.read_text().splitlines() if ln.strip() and not ln.strip().endswith("_posxy.npy")]

def _tile_predict_full_pv(
    model: keras.Model,
    pv: np.ndarray,                 # (V,S)
    ph: int, pw: int,               # patch size (vel x pos)
    stride_v: int = None,
    stride_s: int = None,
    batch: int = 16,
) -> np.ndarray:
    """
    Predict a full PV by tiling with overlap and averaging overlaps.
    Returns probability map in [0,1], shape (V,S).
    """
    V, S = pv.shape
    # pad to at least patch size
    pad_v = max(0, ph - V)
    pad_s = max(0, pw - S)
    pv_pad = np.pad(pv, ((pad_v//2, pad_v - pad_v//2), (pad_s//2, pad_s - pad_s//2)), mode="edge")
    Vp, Sp = pv_pad.shape

    if stride_v is None: stride_v = max(1, ph // 2)
    if stride_s is None: stride_s = max(1, pw // 2)

    ys = list(range(0, max(1, Vp - ph + 1), stride_v))
    xs = list(range(0, max(1, Sp - pw + 1), stride_s))
    if ys[-1] != Vp - ph: ys.append(Vp - ph)
    if xs[-1] != Sp - pw: xs.append(Sp - pw)

    out   = np.zeros((Vp, Sp), dtype=np.float32)
    count = np.zeros((Vp, Sp), dtype=np.float32)

    batch_buf = []
    coords = []
    for y0 in ys:
        for x0 in xs:
            patch = pv_pad[y0:y0+ph, x0:x0+pw][..., None]   # (ph,pw,1)
            batch_buf.append(patch.astype(np.float32))
            coords.append((y0, x0))
            if len(batch_buf) == batch:
                probs = model.predict(np.stack(batch_buf, axis=0), verbose=0)  # (B,ph,pw,1)
                probs = probs[..., 0]
                for (y, x), pr in zip(coords, probs):
                    out[y:y+ph, x:x+pw] += pr
                    count[y:y+ph, x:x+pw] += 1.0
                batch_buf.clear(); coords.clear()

    if batch_buf:
        probs = model.predict(np.stack(batch_buf, axis=0), verbose=0)
        probs = probs[..., 0]
        for (y, x), pr in zip(coords, probs):
            out[y:y+ph, x:x+pw] += pr
            count[y:y+ph, x:x+pw] += 1.0

    # avoid divide by zero
    count[count == 0] = 1.0
    prob_pad = out / count

    # crop back to original
    y0 = pad_v//2; y1 = y0 + V
    x0 = pad_s//2; x1 = x0 + S
    return np.clip(prob_pad[y0:y1, x0:x1], 0.0, 1.0)

def _binary_clean(mask: np.ndarray, min_area_pix: int) -> np.ndarray:
    """Remove tiny components in PV space."""
    if min_area_pix <= 1:
        return mask
    lab, n = label(mask > 0)
    if n == 0:
        return mask*0
    counts = np.bincount(lab.ravel())
    kill = set(np.where(counts < min_area_pix)[0])
    kill.discard(0)
    keep = np.isin(lab, list(kill), invert=True)
    return (keep & (lab > 0)).astype(np.uint8)

def _splat_votes(vote_map: np.ndarray, xs: np.ndarray, ys: np.ndarray, weights: np.ndarray, sigma_pix: float):
    """Add weighted impulses at (x,y), then Gaussian blur in calling code (or here for speed)."""
    h, w = vote_map.shape
    for x, y, wgt in zip(xs, ys, weights):
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            vote_map[yi, xi] += float(wgt)

def _peak_nms(vmap: np.ndarray, nms_radius_pix: int, min_peak: float = 0.0) -> List[Tuple[int,int,float]]:
    """
    Simple max-filter NMS over vote_map.
    Returns list of (y, x, score) peaks.
    """
    if nms_radius_pix < 1:
        nms_radius_pix = 1
    size = 2 * nms_radius_pix + 1
    mf = maximum_filter(vmap, size=size, mode="nearest")
    is_peak = (vmap == mf) & (vmap > min_peak)
    ys, xs = np.where(is_peak)
    scores = vmap[ys, xs]
    order = np.argsort(scores)[::-1]
    return [(int(ys[i]), int(xs[i]), float(scores[i])) for i in order]

def _estimate_radius_from_votes(vmap: np.ndarray, y: int, x: int, frac: float = 0.5, max_r: int = 30) -> float:
    """
    Crude radius: radius at which circular mean vote falls below a fraction of peak.
    """
    peak = vmap[y, x]
    if peak <= 0:
        return 0.0
    h, w = vmap.shape
    r = 1
    while r < max_r:
        # ring pixels (approx): square annulus
        y0 = max(0, y - r); y1 = min(h, y + r + 1)
        x0 = max(0, x - r); x1 = min(w, x + r + 1)
        ring = vmap[y0:y1, x0:x1]
        mean_ring = float(np.mean(ring))
        if mean_ring < frac * peak:
            break
        r += 1
    return float(r)

def _write_regions(peaks: List[Tuple[int,int,float]], wcs: WCS, path: Path, color="cyan"):
    """
    Write DS9/CARTA region file with points at peak positions.
    """
    lines = [
        "# Region file format: DS9 version 4.1",
        "global color={} dashlist=8 3 width=2 font=\"helvetica 10 bold\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0".format(color),
        "fk5"
    ]
    for (y, x, s) in peaks:
        ra, dec = wcs.pixel_to_world_values(x, y)[:2]
        lines.append(f"point({ra:.8f},{dec:.8f}) # point=circle text={{score={s:.3f}}}")
    path.write_text("\n".join(lines))

# ---------------------------
# Core pipeline
# ---------------------------

def aggregate(
    cfg_path: str,
    run_dir: str,
    split: str = "test",
    thresh: float = 0.5,
    vote_sigma_pix: float = 5.0,
    nms_radius_pix: int = 12,
    min_component_area_pv: Tuple[int,int] = (6,5),
    min_radius_beams: float = 2.5,
    max_thickness_frac: float = 0.4,
    write_regions: bool = True,
):
    """
    Post-aggregation:
      1) For each PV: predict probability, threshold+clean, project to sky plane via *_posxy.npy.
      2) Build vote map, blur, find peaks, filter by beam/geometry.
      3) Save FITS/PNG/JSON (and .reg) under <run_dir>/aggregate_<split>.
    """
    cfg = resolve_config(cfg_path, write_resolved=False)
    root = Path(cfg["output_root"])
    pv_dir = root / "pv"
    if not pv_dir.exists():
        raise FileNotFoundError(f"PV directory not found: {pv_dir}")

    # Model + patch sizes from training cfg
    model_path = Path(run_dir) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    ph = int(cfg["train"]["patch_vel"])
    pw = int(cfg["train"]["patch_pos"])

    # Sky geometry
    cube, hdr, wcs, _ = open_cube(cfg["cube_path"])
    H, W = hdr["NAXIS2"], hdr["NAXIS1"]   # im plane
    ax_arcsec, ay_arcsec = pixel_scales_arcsec(wcs)
    beam_as = cfg["galaxy"].get("beam_fwhm_arcsec", None)
    if beam_as is None:
        print("[aggregate] WARNING: beam_fwhm_arcsec unknown; size-based filters may be lenient.")
        beam_pix = None
    else:
        # Use geometric mean axis scale (close to square pixels)
        pix_as = 0.5*(ax_arcsec + ay_arcsec)
        beam_pix = float(beam_as / pix_as)

    # Output dirs
    out_dir = _ensure_dir(Path(run_dir) / f"aggregate_{split}")
    png_dir = _ensure_dir(out_dir / "png")
    meta_dir = _ensure_dir(out_dir / "meta")

    # Vote map (sky plane)
    vote = np.zeros((H, W), dtype=np.float32)

    # Figure out PV items to process
    items = _iter_manifest_items(root, split)
    print(f"[aggregate] split={split}, items={len(items)}")

    min_area_pix = int(max(1, min_component_area_pv[0] * min_component_area_pv[1]))

    processed = 0
    for fname in items:
        pv_path = pv_dir / fname
        posxy_path = pv_dir / fname.replace(".npy", "_posxy.npy")
        meta_path = pv_dir / fname.replace(".npy", ".json")
        if not pv_path.exists() or not posxy_path.exists():
            print(f"[aggregate] skip (missing sidecars): {fname}")
            continue

        pv = np.load(pv_path)        # (V,S)
        posxy = np.load(posxy_path)  # (S,2) -> x,y per position sample
        meta = json.loads(Path(meta_path).read_text())

        # Predict full PV probability (tile if needed)
        prob = _tile_predict_full_pv(model, pv, ph, pw, stride_v=ph//2, stride_s=pw//2, batch=16)  # (V,S)

        # Threshold & clean in PV space
        mask = (prob >= float(thresh)).astype(np.uint8)
        mask = _binary_clean(mask, min_area_pix=min_area_pix)

        # Collapse along velocity to get per-position weight.
        # Use max prob across V at each S (robust to shell thickness in V).
        weights = np.max(prob * mask, axis=0)  # (S,)

        # Project votes back to sky plane
        xs = posxy[:, 0]; ys = posxy[:, 1]
        _splat_votes(vote, xs, ys, weights, sigma_pix=vote_sigma_pix)

        processed += 1
        if processed % 25 == 0:
            print(f"[aggregate] processed {processed}/{len(items)}")

    # Smooth (Gaussian) to accumulate nearby support
    if vote_sigma_pix > 0:
        vote = gaussian_filter(vote, sigma=float(vote_sigma_pix))

    # Simple NMS to get candidate peaks
    peaks = _peak_nms(vote, nms_radius_pix=int(nms_radius_pix), min_peak=float(np.percentile(vote, 90)))
    print(f"[aggregate] raw peaks: {len(peaks)}")

    # Radius estimate + filters
    dets = []
    for (y, x, s) in peaks:
        r_pix = _estimate_radius_from_votes(vote, y, x, frac=0.5, max_r=40)

        # beam-based size floor
        if beam_pix is not None:
            if r_pix < float(min_radius_beams) * beam_pix:
                continue

        # Optional thickness sanity (heuristic placeholder):
        # we interpret thickness as ratio of sigma (vote spread) to radius
        # using a very rough local variance estimate
        y0 = max(0, y - int(r_pix)); y1 = min(vote.shape[0], y + int(r_pix) + 1)
        x0 = max(0, x - int(r_pix)); x1 = min(vote.shape[1], x + int(r_pix) + 1)
        patch = vote[y0:y1, x0:x1]
        if patch.size > 0:
            var = float(np.var(patch))
            thickness = min(1.0, math.sqrt(max(var, 1e-8)) / (r_pix + 1e-3))
            if thickness > float(max_thickness_frac):
                continue

        dets.append({"x_pix": int(x), "y_pix": int(y), "score": float(s), "r_pix": float(r_pix)})

    print(f"[aggregate] kept {len(dets)} detections after filters")

    # ---------------------------
    # Save outputs
    # ---------------------------

    # Vote map FITS (CARTA-friendly)
    hdu = fits.PrimaryHDU(data=vote.astype(np.float32), header=WCS(hdr).to_header())
    fits_path = out_dir / f"vote_map_{split}.fits"
    hdu.writeto(fits_path, overwrite=True)

    # PNG preview
    plt.figure(figsize=(6, 6))
    plt.imshow(vote, origin="lower", cmap="magma")
    plt.colorbar(label="vote")
    if dets:
        xs = [d["x_pix"] for d in dets]; ys = [d["y_pix"] for d in dets]
        plt.scatter(xs, ys, s=30, facecolors='none', edgecolors='cyan', linewidths=1.5, label="detections")
        plt.legend(loc="upper right")
    plt.title(f"Vote map ({split})")
    plt.tight_layout()
    png_path = png_dir / f"vote_map_{split}.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    # Detections JSON
    out_json = {
        "split": split,
        "vote_fits": str(fits_path),
        "detections": dets,
        "params": {
            "thresh": float(thresh),
            "vote_sigma_pix": float(vote_sigma_pix),
            "nms_radius_pix": int(nms_radius_pix),
            "min_component_area_pv": list(min_component_area_pv),
            "min_radius_beams": float(min_radius_beams),
            "max_thickness_frac": float(max_thickness_frac),
        }
    }
    json_path = out_dir / f"detections_{split}.json"
    dumps_json(out_json, json_path)

    # Optional regions (FK5) for CARTA/DS9
    if write_regions:
        try:
            _write_regions([(d["y_pix"], d["x_pix"], d["score"]) for d in dets], WCS(hdr), out_dir / f"detections_{split}.reg", color="cyan")
        except Exception as e:
            print(f"[aggregate] region write failed: {e}")

    print(f"[aggregate] wrote:\n  - {fits_path}\n  - {png_path}\n  - {json_path}")
    if write_regions:
        print(f"  - {out_dir / f'detections_{split}.reg'}")


# ---------------------------
# CLI
# ---------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pv_config.yaml")
    ap.add_argument("--run_dir", required=True, help="Run dir with best_model.keras")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--vote_sigma_pix", type=float, default=None, help="Override cfg.aggregate.vote_sigma_pix")
    ap.add_argument("--nms_radius_pix", type=int, default=None, help="Override cfg.aggregate.nms_radius_pix")
    ap.add_argument("--min_radius_beams", type=float, default=None)
    ap.add_argument("--max_thickness_frac", type=float, default=None)
    ap.add_argument("--no_regions", action="store_true", help="Disable writing .reg file")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    cfg = resolve_config(args.config, write_resolved=False)

    post = cfg.get("post", {})
    agg  = cfg.get("aggregate", {})

    # PV component clean-up area (height x width pixels in PV)
    min_area = tuple(post.get("min_component_area_pv", [6,5]))
    # Aggregation knobs (with CLI override)
    vote_sigma = args.vote_sigma_pix if args.vote_sigma_pix is not None else float(agg.get("vote_sigma_pix", 5))
    nms_r      = args.nms_radius_pix if args.nms_radius_pix is not None else int(agg.get("nms_radius_pix", 12))
    min_r_beam = args.min_radius_beams if args.min_radius_beams is not None else float(agg.get("min_radius_beams", 2.5))
    thick_max  = args.max_thickness_frac if args.max_thickness_frac is not None else float(agg.get("max_thickness_frac", 0.4))

    aggregate(
        cfg_path=args.config,
        run_dir=args.run_dir,
        split=args.split,
        thresh=float(args.thresh),
        vote_sigma_pix=float(vote_sigma),
        nms_radius_pix=int(nms_r),
        min_component_area_pv=min_area,
        min_radius_beams=float(min_r_beam),
        max_thickness_frac=float(thick_max),
        write_regions=(not args.no_regions),
    )