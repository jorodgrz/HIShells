# src/utils/config.py
import os, json, copy, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from jsonschema import validate

from src.utils.wcs_tools import velocity_axis_kms  # robust spectral axis helper

SCHEMA = {
  "type": "object",
  "properties": {
    "cube_path": {"type":"string"},
    "output_root": {"type":"string"},
    "catalog_csv": {"type":["string","null"]},   # legacy, optional
    "catalogs": {"type":["object","null"]},      # optional
    "galaxy": {
      "type":"object",
      "properties": {
        "ra_deg": {"type":["number","null"]},
        "dec_deg": {"type":["number","null"]},
        "pa_deg": {"type":["number","null"]},
        "inc_deg": {"type":["number","null"]},
        "vsys_kms": {"type":["number","null"]},
        "beam_fwhm_arcsec": {"type":["number","null"]}
      }
    },
    # All other sections (pv/train/model/optim/aggregate/vis) are accepted leniently.
  },
  "required": ["cube_path","output_root","galaxy"]
}


def _read_yaml(path:str)->Dict[str,Any]:
    with open(path,"r") as f: return yaml.safe_load(f)

def _save_yaml(obj:Dict[str,Any], path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f: yaml.safe_dump(obj, f, sort_keys=False)

def _hash_cfg(obj:Dict[str,Any])->str:
    s = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(s).hexdigest()[:12]

def _env_overrides(prefix="PV_")->Dict[str,Any]:
    """
    Convert env like PV_model.base_filters=48, PV_optim.lr=3e-4 to nested dict.
    """
    out: Dict[str,Any] = {}
    for k,v in os.environ.items():
        if not k.startswith(prefix): continue
        path = k[len(prefix):]  # e.g., "model.base_filters"
        cursor = out
        keys = path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        # try to coerce number/bool
        if v.lower() in ("true","false"):
            val = v.lower()=="true"
        else:
            try:
                val = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                val = v
        cursor[keys[-1]] = val
    return out

def _cli_overrides(pairs:List[str])->Dict[str,Any]:
    out: Dict[str,Any] = {}
    for p in pairs:
        if "=" not in p: continue
        k, v = p.split("=",1)
        cursor = out
        keys = k.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        # type infer
        vv = v
        if v.lower() in ("true","false"): vv = v.lower()=="true"
        else:
            try:
                vv = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                pass
        cursor[keys[-1]] = vv
    return out

def _deep_update(a:Dict[str,Any], b:Dict[str,Any])->Dict[str,Any]:
    for k,v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _deep_update(a[k], v)
        else:
            a[k] = v
    return a

# ---------- FITS-driven inference ----------

def _pixel_scales_from_header(hdr) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (pix_arcsec, dv_kms). pix_arcsec is the average of |CDELT1|/|CDELT2| in arcsec.
    dv_kms is estimated via velocity_axis_kms() spacing if possible; else None.
    """
    pix_arcsec = None
    try:
        cdelt1 = abs(hdr.get("CDELT1") or hdr.get("CD1_1") or 0.0) * 3600.0  # deg->arcsec
        cdelt2 = abs(hdr.get("CDELT2") or hdr.get("CD2_2") or 0.0) * 3600.0
        if cdelt1 and cdelt2:
            pix_arcsec = 0.5*(cdelt1 + cdelt2)
        else:
            pix_arcsec = cdelt1 or cdelt2 or None
    except Exception:
        pix_arcsec = None

    dv_kms = None
    try:
        v = velocity_axis_kms(hdr)
        if len(v) > 1:
            dv_kms = float(np.median(np.diff(v)))
    except Exception:
        dv_kms = None

    return pix_arcsec, dv_kms

def _infer_from_fits(cfg: dict) -> dict:
    path = cfg["cube_path"]
    hdr = fits.getheader(path)
    wcs = WCS(hdr)
    g = copy.deepcopy(cfg["galaxy"])

    # center RA/Dec from 2D celestial WCS at image center
    if g.get("ra_deg") is None or g.get("dec_deg") is None:
        nx, ny = hdr.get("NAXIS1", 0), hdr.get("NAXIS2", 0)
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        try:
            cel = wcs.celestial
            ra, dec = cel.pixel_to_world_values(cx, cy)
            g["ra_deg"] = float(ra)
            g["dec_deg"] = float(dec)
        except Exception as e:
            print(f"[config] WARNING: could not infer RA/Dec from WCS.celestial: {e}")

    # systemic velocity guess from spectral axis (center channel)
    if g.get("vsys_kms") is None:
        try:
            v = velocity_axis_kms(hdr)
            g["vsys_kms"] = float(v[len(v)//2])
        except Exception as e:
            print(f"[config] WARNING: could not infer vsys from header: {e}")

    # beam FWHM from BMAJ/BMIN (deg -> arcsec, geometric mean)
    if g.get("beam_fwhm_arcsec") is None:
        bmaj = hdr.get("BMAJ"); bmin = hdr.get("BMIN")
        if bmaj and bmin:
            f_arcsec = float(np.sqrt(bmaj * bmin) * 3600.0)
            g["beam_fwhm_arcsec"] = f_arcsec

    # attach meta scales for downstream (pix + dv)
    pix_arcsec, dv_kms = _pixel_scales_from_header(hdr)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["pix_arcsec"] = pix_arcsec
    cfg["_meta"]["dv_kms"] = dv_kms

    cfg["galaxy"] = g
    return cfg

# ---------- beam/pixel conversions & defaults ----------

def _beam_to_pix(beam_arcsec: Optional[float], pix_arcsec: Optional[float], val_beam: Optional[float]) -> Optional[float]:
    if val_beam is None: return None
    if beam_arcsec and pix_arcsec and pix_arcsec > 0:
        return float(val_beam) * (beam_arcsec / pix_arcsec)
    return None

def _apply_pv_defaults_and_conversions(cfg: Dict[str, Any]) -> None:
    """Mutates cfg in place: set PV defaults, convert beam-based params, align unwrap bins."""
    pv = cfg.setdefault("pv", {})
    spoke = pv.setdefault("spoke", {})
    ring  = pv.setdefault("ring", {})
    axes  = pv.setdefault("axes", {})
    sampling = pv.setdefault("sampling", {})

    # sensible defaults (in case YAML omitted them)
    axes.setdefault("include_major_minor", True)

    sampling.setdefault("pos_samples_per_beam", 3)  # density along POS when extracting slits
    sampling.setdefault("vel_smooth", 0)            # 0 = no smoothing
    sampling.setdefault("vel_subsample", 1)         # 1 = no subsample

    # gather scales
    beam_arcsec = cfg["galaxy"].get("beam_fwhm_arcsec")
    pix_arcsec  = cfg.get("_meta", {}).get("pix_arcsec")

    # spoke slit width: allow slit_width_beam OR slit_width_pix (beam wins if convertible)
    if "slit_width_beam" in spoke and spoke["slit_width_beam"] is not None:
        slit_pix = _beam_to_pix(beam_arcsec, pix_arcsec, spoke["slit_width_beam"])
        if slit_pix is not None:
            spoke["slit_width_pix"] = max(1, int(round(slit_pix)))
        else:
            print("[config] WARNING: pv.spoke.slit_width_beam provided but beam/pixel scales unavailable; please set slit_width_pix directly.")
    # fallback: ensure at least 1 pixel
    if "slit_width_pix" not in spoke or spoke["slit_width_pix"] in (None, 0):
        spoke["slit_width_pix"] = 3  # safe default

    # ring radial step: allow r_step_beam OR r_step_arcsec (beam wins if convertible)
    if "r_step_beam" in ring and ring["r_step_beam"] is not None:
        if beam_arcsec:
            ring["r_step_arcsec"] = float(ring["r_step_beam"]) * float(beam_arcsec)
        else:
            print("[config] WARNING: pv.ring.r_step_beam provided but galaxy.beam_fwhm_arcsec is unknown; please set r_step_arcsec.")
    # keep start/stop as arcsec; defaults if missing
    ring.setdefault("r_start_arcsec", 10.0)
    ring.setdefault("r_stop_arcsec",  300.0)
    ring.setdefault("r_step_arcsec",  10.0)
    # unwrap bins: enforce match with train.patch_pos (avoid silent resampling later)
    train = cfg.get("train", {})
    patch_pos = int(train.get("patch_pos", ring.get("unwrap_bins", 512) or 512))
    if ring.get("unwrap_bins") not in (None, patch_pos):
        print(f"[config] INFO: overriding pv.ring.unwrap_bins={ring['unwrap_bins']} -> train.patch_pos={patch_pos} to avoid resampling.")
    ring["unwrap_bins"] = int(patch_pos)

def _stamp_warnings_if_missing(cfg: Dict[str, Any]) -> None:
    g = cfg["galaxy"]
    for key in ("pa_deg", "inc_deg", "vsys_kms", "beam_fwhm_arcsec"):
        if g.get(key) is None:
            print(f"[config] WARNING: galaxy.{key} not set and not inferrable from FITS.")
    if cfg.get("_meta", {}).get("pix_arcsec") is None:
        print("[config] WARNING: could not infer sky pixel scale (arcsec/pix) from FITS WCS; beam→pixel conversions may be skipped.")

# ---------- public API ----------

def resolve_config(cfg_path:str, set_pairs:List[str]=None, write_resolved:bool=True)->Dict[str,Any]:
    base = _read_yaml(cfg_path)
    # apply env overrides, then CLI overrides
    env = _env_overrides()
    cli = _cli_overrides(set_pairs or [])
    merged = _deep_update(copy.deepcopy(base), env)
    merged = _deep_update(merged, cli)

    # FITS inference (RA/Dec, vsys, beam, plus meta scales)
    merged = _infer_from_fits(merged)

    # PV defaults & unit conversions
    _apply_pv_defaults_and_conversions(merged)

    # validate (lenient)
    try:
        validate(instance=merged, schema=SCHEMA)
    except Exception as e:
        print("[config] WARNING: schema validation warning:", e)

    # stamp meta hash + echo scales
    meta = merged.setdefault("_meta", {})
    meta["_resolved"] = True
    meta["_hash"] = _hash_cfg(merged)

    # final warnings for any missing geometry/scales
    _stamp_warnings_if_missing(merged)

    if write_resolved:
        out = Path(cfg_path).resolve().with_name(Path(cfg_path).stem + "._resolved.yaml")
        _save_yaml(merged, out)
        print(f"[config] wrote resolved config -> {out} (hash={meta['_hash']})")

    return merged
