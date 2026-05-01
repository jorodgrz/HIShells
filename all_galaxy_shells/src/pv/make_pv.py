# src/pv/make_pv.py
# Shell-axis PV slicer + empty-region negatives (no grid) with robust catalog parsers.
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import path hygiene: make sure repo root and src/ are importable
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.io import load_yaml, dumps_json
from src.utils.config import resolve_config
from src.utils.wcs_tools import (
    open_cube, pixel_scales_arcsec, radec_to_xy,
    velocity_axis_kms, unit_vectors_for_pa
)

__VERSION__ = "make_pv.py@shell_axes+neg_empty-1.4"

# -----------------------------------------------------------------------------
# Config loader (writes a resolved copy next to the provided YAML)
# -----------------------------------------------------------------------------
def load_cfg(cfg_path: str | Path):
    return resolve_config(cfg_path, write_resolved=True)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _moment0(cube: np.ndarray) -> np.ndarray:
    m0 = np.nansum(cube, axis=0).astype(np.float32)
    mx = np.nanmax(m0) if np.isfinite(m0).any() else 0.0
    if mx > 0:
        m0 = m0 / mx
    return np.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, perp_vec):
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
    ax.plot([x1, y1], [x2, y2], lw=lw)

def _save_line_pv(stem_base, x0, y0, dir_x, dir_y, perp_x, perp_y, half_len_pix,
                  cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path,
                  slit_half: int, pos_step_pix: float, frame: str, conv: str,
                  pa_eff: float | None, m0: np.ndarray, v_grid):
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])

    # positions along the line
    npos = int(2 * half_len_pix / pos_step_pix) + 1
    if npos < 9:
        return False
    ts = np.linspace(-half_len_pix, +half_len_pix, npos, dtype=np.float32)
    xs = x0 + ts * dir_x
    ys = y0 + ts * dir_y

    # clip to in-bounds
    mask = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 8:
        return False

    # PV
    pv = np.zeros((len(v_grid), xs.size), dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        pv[:, i] = _safe_sample_spectrum(cube, x, y, slit_half, nx, ny, (perp_x, perp_y))

    # saves
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{stem_base}.npy", pv)
    np.save(out_dir / f"{stem_base}_posxy.npy", np.stack([xs, ys], axis=1).astype(np.float32))
    cfg_hash = (cfg.get("_meta") or {}).get("_hash") or "unknown"
    meta = {
        "script": __VERSION__, "cfg_hash": cfg_hash,
        "type": "line",
        "frame": frame,
        "pa_convention": conv,
        "pa_eff_deg": None if pa_eff is None else float(pa_eff),
        "center_pix": [float(x0), float(y0)],
        "dir_pix": [float(dir_x), float(dir_y)],
        "perp_pix": [float(perp_x), float(perp_y)],
        "slit_width_pix": int(2*slit_half+1),
        "npos": int(pv.shape[1]),
        "nv": int(pv.shape[0]),
        "vel_kms": [float(v_grid[0]), float(v_grid[-1]), float(v_grid[1]-v_grid[0])]
    }
    dumps_json(meta, out_dir / f"{stem_base}.json")

    # spatial overlay
    plt.figure(figsize=(6, 5))
    plt.imshow(m0, origin="lower", cmap="gray")
    plt.plot(xs, ys, lw=1.0)
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

# -----------------------------------------------------------------------------
# Catalog loaders
# -----------------------------------------------------------------------------
def _hms_to_deg(parts3) -> float | None:
    try:
        h, m, s = float(parts3[0]), float(parts3[1]), float(parts3[2])
        return (h + m/60.0 + s/3600.0) * 15.0
    except Exception:
        return None

def _dms_to_deg(parts3) -> float | None:
    try:
        d0 = str(parts3[0])
        sign = -1.0 if d0.startswith("-") else 1.0
        d = float(d0.lstrip("+-"))
        m, s = float(parts3[1]), float(parts3[2])
        return sign * (d + m/60.0 + s/3600.0)
    except Exception:
        return None

def _load_shell_catalog_bagetakos_like(path: Path, galaxy_filter: str | None):
    """
    Parse Bagetakos-style per-hole rows without headers, e.g.:
    'Holmberg I    2  9 40 22.1  71 12 03.5  146 2  ...   1.3  0.7'
         name     id   RA(h m s)  Dec(d m s)  PA  T             maj  min

    Returns list of dicts:
      {galaxy, ra_deg, dec_deg, pa_deg, type, major_val, minor_val, size_unit_hint}
    """
    if not path.exists() or not path.is_file():
        return []
    lines = [
        ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    out = []
    for ln in lines:
        # split into tokens; name may be multi-word (stop when tokens become numeric)
        toks = re.split(r"\s+", ln)
        i = 0
        while i < len(toks) and not re.match(r"^[+-]?\d+(\.\d+)?$", toks[i]):
            i += 1
        name = " ".join(toks[:i]).strip()
        rest = toks[i:]
        if galaxy_filter and name and galaxy_filter.lower() not in name.lower():
            continue
        if len(rest) < 3 + 3 + 2:  # id + RA3 + Dec3 + PA/Type + tail
            continue
        # id
        try:
            _ = int(float(rest[0]))
        except Exception:
            continue
        # RA/Dec
        ra_deg = _hms_to_deg(rest[1:4])
        dec_deg = _dms_to_deg(rest[4:7])
        # PA, Type
        pa_deg = None
        shell_type = None
        idx = 7
        if idx < len(rest):
            try:
                x = float(rest[idx])
                if -5.0 <= x <= 365.0:  # tolerant
                    pa_deg = x; idx += 1
            except Exception:
                pass
        if idx < len(rest):
            try:
                t = int(float(rest[idx]))
                if t in (1, 2, 3):
                    shell_type = t; idx += 1
            except Exception:
                pass
        # Take last two numeric tokens as maj/min
        nums = []
        for tok in rest[::-1]:
            try:
                nums.append(float(tok))
            except Exception:
                if len(nums) >= 2:
                    break
        nums = nums[:2][::-1] if len(nums) >= 2 else []
        maj_val = nums[0] if len(nums) >= 1 else None
        min_val = nums[1] if len(nums) >= 2 else None
        unit_hint = "kpc" if (maj_val is not None and min_val is not None and 0.01 <= maj_val <= 10.0 and 0.01 <= min_val <= 10.0) else "arcsec"
        out.append({
            "galaxy": name,
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "pa_deg": pa_deg, "type": shell_type,
            "major_val": maj_val, "minor_val": min_val, "size_unit_hint": unit_hint
        })
    return out

def _load_galaxy_meta_fixedwidth(path: Path):
    """
    Parse a fixed-width galaxy meta table with columns like:
      Name, OName, RAh, RAm, RAs, DE-, DEd, DEm, DEs, Type, Dist, Incl, PA, ...
    Returns dict by name -> meta fields (deg, distance_mpc, inc_deg, pa_deg).
    """
    if not path.exists() or not path.is_file():
        return {}
    metas = {}
    lines = [
        ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip() and not ln.startswith("-") and not ln.strip().startswith("1-")
    ]
    for ln in lines:
        try:
            name   = ln[0:11].strip()
            # OName = ln[12:21]  # not used
            RAh    = ln[22:24].strip()
            RAm    = ln[25:27].strip()
            RAs    = ln[28:32].strip()
            DEsign = ln[33:34].strip() or "+"
            DEd    = ln[34:36].strip()
            DEm    = ln[37:39].strip()
            DEs    = ln[40:44].strip()
            Type   = ln[45:55].strip()
            Dist   = ln[56:60].strip()
            Incl   = ln[61:63].strip()
            PA     = ln[64:67].strip()
        except Exception:
            continue

        ra_deg = _hms_to_deg([RAh or "0", RAm or "0", RAs or "0"])
        dsign  = "-" if (DEsign == "-") else "+"
        dec_deg = _dms_to_deg([dsign + (DEd or "0"), DEm or "0", DEs or "0"])

        def f(x):
            try: return float(x)
            except: return None

        metas[name] = {
            "name": name,
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "distance_mpc": f(Dist),
            "inc_deg": f(Incl), "pa_deg": f(PA), "type": Type
        }
    return metas

# -----------------------------------------------------------------------------
# Shell-axis PV slicer (two cuts per shell; filter by type)
# -----------------------------------------------------------------------------
def make_shell_axis_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path):
    v_grid = velocity_axis_kms(hdr)
    ax_as, ay_as = pixel_scales_arcsec(wcs)

    gal = (cfg.get("galaxy") or {})
    gal_name_for_filter = gal.get("name", gal.get("id", "")) or ""
    sh = (cfg.get("pv") or {}).get("shell_axes", {})
    if not sh.get("enabled", False):
        return {"n_shells": 0, "n_cuts": 0}

    # label table path (robust to None)
    label_val = sh.get("label_table_path") or (cfg.get("catalogs", {}) or {}).get("holes_dat", "")
    if not isinstance(label_val, (str, os.PathLike)) or not label_val:
        print("[make_pv] WARNING: no label table path provided.")
        return {"n_shells": 0, "n_cuts": 0}
    label_path = Path(label_val)

    pa_conv   = str(sh.get("pa_convention", "astro")).lower()
    allowed   = set(sh.get("allowed_types", [2, 3]))
    fallback_gal_pa = bool(sh.get("fallback_to_gal_pa_if_missing", True))

    length_scale_major = float(sh.get("length_scale_major", 1.2))
    length_scale_minor = float(sh.get("length_scale_minor", 1.2))
    slit_w      = int(sh.get("slit_width_pix", 3))
    slit_half   = max(0, slit_w // 2)
    pos_step_pix = float(sh.get("pos_step_pix", 1.0))
    name_prefix = str(sh.get("name_prefix", "shell"))

    # Load shells
    shells = _load_shell_catalog_bagetakos_like(label_path, galaxy_filter=gal_name_for_filter)
    print(f"[make_pv] loaded {len(shells)} shell rows from {label_path} for galaxy '{gal_name_for_filter}'")
    if not shells:
        return {"n_shells": 0, "n_cuts": 0}

    # Galaxy geometry (from YAML first)
    gal_pa_deg = gal.get("pa_deg")
    dist_mpc   = gal.get("distance_mpc")
    gal.setdefault("ra_deg", gal.get("ra_deg"))
    gal.setdefault("dec_deg", gal.get("dec_deg"))

    # Optional galaxy meta table (None/dir-safe)
    meta = {}
    meta_val = (cfg.get("catalogs") or {}).get("galaxy_meta", None)
    if isinstance(meta_val, (str, os.PathLike)) and meta_val:
        meta_path = Path(meta_val)
        if meta_path.is_file():
            meta = _load_galaxy_meta_fixedwidth(meta_path)

    # If YAML missing some fields, fill from meta by name
    if (not gal_pa_deg or not dist_mpc or not gal.get("ra_deg") or not gal.get("dec_deg")) and gal_name_for_filter:
        m = meta.get(gal_name_for_filter)
        if m:
            gal_pa_deg = gal_pa_deg or m.get("pa_deg")
            dist_mpc   = dist_mpc   or m.get("distance_mpc")
            gal.setdefault("ra_deg", m.get("ra_deg"))
            gal.setdefault("dec_deg", m.get("dec_deg"))

    m0 = _moment0(cube)

    def kpc_to_arcsec(x_kpc):
        if dist_mpc and dist_mpc > 0:
            return float(x_kpc) * 206.265 / float(dist_mpc)
        return None

    def as2pix_x(as_val): return as_val / ax_as if as_val is not None and ax_as > 0 else None
    def as2pix_y(as_val): return as_val / ay_as if as_val is not None and ay_as > 0 else None

    n_cuts = 0
    used   = 0
    skipped_type = 0
    skipped_coord = 0

    for idx, s in enumerate(shells):
        stype = s.get("type")
        if stype is None or stype not in allowed:
            skipped_type += 1
            continue
        if s.get("ra_deg") is None or s.get("dec_deg") is None:
            skipped_coord += 1
            continue

        cx, cy = radec_to_xy(wcs, s["ra_deg"], s["dec_deg"])

        maj_val = s.get("major_val")
        min_val = s.get("minor_val") or maj_val
        if maj_val is None:
            continue

        if s.get("size_unit_hint") == "kpc":
            maj_as = kpc_to_arcsec(maj_val)
            min_as = kpc_to_arcsec(min_val) if min_val is not None else maj_as
        else:
            maj_as = maj_val
            min_as = min_val

        if maj_as is None:
            # fallback to negatives length if sizes unknown
            maj_as = float((cfg.get("pv") or {}).get("negatives", {}).get("length_arcsec", 600.0))
            min_as = maj_as

        pa_deg = s.get("pa_deg")
        if pa_deg is None:
            if fallback_gal_pa and gal_pa_deg is not None:
                pa_deg = float(gal_pa_deg)
            else:
                continue

        (mx, my), (nxv, nyv) = unit_vectors_for_pa(pa_deg, convention=pa_conv)

        half_len_major_pix = max(as2pix_x(maj_as), as2pix_y(maj_as)) * 0.5 * length_scale_major
        half_len_minor_pix = max(as2pix_x(min_as), as2pix_y(min_as)) * 0.5 * length_scale_minor

        # Major-axis cut
        stem1 = f"{name_prefix}{idx:04d}_type{stype}_major_pa{int(round(pa_deg))}"
        ok1 = _save_line_pv(
            stem1, cx, cy,
            dir_x=mx, dir_y=my, perp_x=nxv, perp_y=nyv,
            half_len_pix=half_len_major_pix,
            cube=cube, hdr=hdr, wcs=wcs, cfg=cfg, out_dir=out_dir, overlays_dir=overlays_dir,
            slit_half=slit_half, pos_step_pix=pos_step_pix,
            frame="shell_axes", conv=pa_conv, pa_eff=pa_deg,
            m0=m0, v_grid=v_grid
        )

        # Minor-axis cut
        stem2 = f"{name_prefix}{idx:04d}_type{stype}_minor_pa{int(round((pa_deg+90)%180))}"
        ok2 = _save_line_pv(
            stem2, cx, cy,
            dir_x=nxv, dir_y=nyv, perp_x=mx, perp_y=my,
            half_len_pix=half_len_minor_pix,
            cube=cube, hdr=hdr, wcs=wcs, cfg=cfg, out_dir=out_dir, overlays_dir=overlays_dir,
            slit_half=slit_half, pos_step_pix=pos_step_pix,
            frame="shell_axes", conv=pa_conv, pa_eff=pa_deg,
            m0=m0, v_grid=v_grid
        )

        if ok1 or ok2:
            used += 1
            n_cuts += int(ok1) + int(ok2)

    print(f"[make_pv] shell_axes: kept {used} shells (types {sorted(allowed)}), "
          f"skipped type={skipped_type}, coord-missing={skipped_coord} → {n_cuts} cuts")
    return {"n_shells": used, "n_cuts": n_cuts}

# -----------------------------------------------------------------------------
# Negative cuts: "empty" regions within the galaxy mask
# -----------------------------------------------------------------------------
def _build_masks(m0, galaxy_pct=50.0, empty_pct=15.0, smooth=False):
    if smooth:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m0, sigma=1.0)
    else:
        m = m0
    m_flat = m[np.isfinite(m)]
    if m_flat.size == 0:
        z = np.zeros_like(m, dtype=bool)
        return z, z
    g_thr = np.percentile(m_flat, float(galaxy_pct))
    e_thr = np.percentile(m_flat, float(empty_pct))
    galaxy_mask = (m >= g_thr)
    empty_mask = (m <= e_thr) & galaxy_mask
    return galaxy_mask, empty_mask

def _min_pix_dist(pix_pt, centers_pix):
    if not centers_pix:
        return np.inf
    dx = np.array([pix_pt[0]-c[0] for c in centers_pix], dtype=np.float32)
    dy = np.array([pix_pt[1]-c[1] for c in centers_pix], dtype=np.float32)
    return float(np.sqrt((dx*dx + dy*dy).min()))

def _line_mask_fraction(x0, y0, dir_x, dir_y, half_len_pix, pos_step_pix, mask, min_frac=0.7):
    npos = int(2 * half_len_pix / pos_step_pix) + 1
    if npos < 9:
        return False
    ts = np.linspace(-half_len_pix, +half_len_pix, npos, dtype=np.float32)
    xs = x0 + ts * dir_x
    ys = y0 + ts * dir_y
    h, w = mask.shape
    ok, total = 0, 0
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            total += 1
            if mask[yi, xi]:
                ok += 1
    return total >= 8 and (ok / max(1, total)) >= min_frac

def make_negative_empty_pv(cube, hdr, wcs, cfg, out_dir: Path, overlays_dir: Path,
                           shell_centers_pix):
    v_grid = velocity_axis_kms(hdr)
    ax_as, ay_as = pixel_scales_arcsec(wcs)
    m0 = _moment0(cube)

    neg = (cfg.get("pv") or {}).get("negatives", {})
    if not neg.get("enabled", True):
        return {"n_negs": 0}

    n_per_shell = int(neg.get("n_per_shell", 2))
    galaxy_pct = float(neg.get("galaxy_mask_percentile", 50.0))
    empty_pct  = float(neg.get("empty_percentile", 15.0))
    min_sep_as = float(neg.get("min_sep_from_shell_arcsec", 30.0))
    orient     = str(neg.get("orientation", "random")).lower()  # 'random'|'galaxy_axes'
    length_as  = float(neg.get("length_arcsec", 600.0))         # total length in arcsec
    slit_w     = int(neg.get("slit_width_pix", 3))
    slit_half  = max(0, slit_w // 2)
    pos_step   = float(neg.get("pos_step_pix", 1.0))
    pa_conv    = "astro"
    gal_pa     = (cfg.get("galaxy") or {}).get("pa_deg")

    galaxy_mask, empty_mask = _build_masks(m0, galaxy_pct=galaxy_pct, empty_pct=empty_pct, smooth=False)

    x_pix_per_as = 1.0 / ax_as if ax_as > 0 else 1.0
    min_sep_pix = max(1.0, min_sep_as * x_pix_per_as)

    def as2pix(as_val):
        if as_val is None:
            return None
        return max(as_val / ax_as, as_val / ay_as)
    half_len_pix = 0.5 * as2pix(length_as)

    rng = np.random.default_rng(int(neg.get("seed", 42)))
    empty_ys, empty_xs = np.where(empty_mask)
    candidates = list(zip(empty_xs.tolist(), empty_ys.tolist()))
    rng.shuffle(candidates)

    if orient == "galaxy_axes" and gal_pa is not None:
        (mx, my), (nxv, nyv) = unit_vectors_for_pa(float(gal_pa), convention=pa_conv)
        dir_choices = [(mx, my), (nxv, nyv)]
    else:
        dir_choices = None  # random

    needed = n_per_shell * max(1, len(shell_centers_pix))
    taken = 0

    while candidates and taken < needed:
        x0, y0 = candidates.pop()
        if _min_pix_dist((x0, y0), shell_centers_pix) < min_sep_pix:
            continue

        if dir_choices is None:
            theta = rng.uniform(0, np.pi)
            dx, dy = np.cos(theta), np.sin(theta)
        else:
            dx, dy = dir_choices[rng.integers(0, len(dir_choices))]

        if not _line_mask_fraction(x0, y0, dx, dy, half_len_pix, pos_step, galaxy_mask, min_frac=0.7):
            continue

        px, py = -dy, dx
        stem = f"neg_empty_{taken:05d}"
        ok = _save_line_pv(
            stem, x0, y0,
            dir_x=dx, dir_y=dy, perp_x=px, perp_y=py,
            half_len_pix=half_len_pix,
            cube=cube, hdr=hdr, wcs=wcs, cfg=cfg, out_dir=out_dir, overlays_dir=overlays_dir,
            slit_half=slit_half, pos_step_pix=pos_step,
            frame="neg_empty", conv=pa_conv, pa_eff=None,
            m0=m0, v_grid=v_grid
        )
        if ok:
            taken += 1

    print(f"[make_pv] negatives: placed {taken} empty-region cuts (goal={needed})")
    return {"n_negs": taken}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(cfg):
    # accept either top-level cube_path or nested cube.path
    cube_path = cfg.get("cube_path") or (cfg.get("cube", {}) or {}).get("path")
    if not cube_path:
        raise SystemExit("cube_path (or cube.path) not found in config.")

    outdir   = Path(cfg["output_root"]) / "pv"
    overlays = Path(cfg["output_root"]) / "qa_pv"
    outdir.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    cube, hdr, wcs, _ = open_cube(cube_path)

    # 1) Shell-axis positives
    pos_counts = make_shell_axis_pv(cube, hdr, wcs, cfg, outdir, overlays)

    # Rebuild shell centers for negatives avoidance
    sh_cfg = (cfg.get("pv") or {}).get("shell_axes", {})
    label_val = sh_cfg.get("label_table_path") or (cfg.get("catalogs", {}) or {}).get("holes_dat", "")
    shell_centers_pix = []
    if isinstance(label_val, (str, os.PathLike)) and label_val:
        label_path = Path(label_val)
        allowed = set(sh_cfg.get("allowed_types", [2, 3]))
        gal_name_for_filter = (cfg.get("galaxy") or {}).get("name", (cfg.get("galaxy") or {}).get("id", "")) or ""
        shells = _load_shell_catalog_bagetakos_like(label_path, galaxy_filter=gal_name_for_filter) if label_path.is_file() else []
        for s in shells or []:
            if s.get("type") in allowed and s.get("ra_deg") is not None and s.get("dec_deg") is not None:
                x, y = radec_to_xy(wcs, s["ra_deg"], s["dec_deg"])
                shell_centers_pix.append((float(x), float(y)))

    # 2) Empty-region negatives
    neg_counts = make_negative_empty_pv(cube, hdr, wcs, cfg, outdir, overlays, shell_centers_pix)

    total = pos_counts.get("n_cuts", 0) + neg_counts.get("n_negs", 0)
    print(f"[make_pv] wrote positives: {pos_counts.get('n_cuts',0)}, negatives: {neg_counts.get('n_negs',0)}; total={total} → {outdir}")
    print(f"[make_pv] overlays saved to {overlays}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if "pv" not in cfg or "shell_axes" not in cfg["pv"]:
        raise SystemExit("pv.shell_axes block missing in YAML.")
    main(cfg)