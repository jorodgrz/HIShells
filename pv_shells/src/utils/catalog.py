"""
Robust loader for Bagetakos-style catalogs from two .dat tables.

Primary target:
  - table7.dat : per-hole catalog (headerless, whitespace-tokenized)
  - table2.dat or table8.dat : optional; may contain v_exp or galaxy-level params

Returns a unified pandas.DataFrame with columns:
  ra_deg, dec_deg, major_arcsec, minor_arcsec, pa_deg, vexp_kms, id, galaxy
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import Angle
import astropy.units as u


# -----------------------------
# Low-level helpers
# -----------------------------

def _read_dat_any(path: str) -> pd.DataFrame:
    """Try Astropy ASCII readers; fall back to pandas."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    # Try Astropy readers (best for CDS/VizieR .dat)
    for fmt in ("ascii.cds", "ascii.fixed_width", "ascii.basic", "ascii"):
        try:
            t = Table.read(path, format=fmt, guess=False)
            return t.to_pandas()
        except Exception:
            pass
    # Fallbacks
    try:
        return pd.read_fwf(path)
    except Exception:
        return pd.read_csv(path, delim_whitespace=True)


def _parse_ra_tokens(h: str, m: str, s: str) -> float:
    """RA from tokens h m s to degrees."""
    txt = f"{int(h)}h{int(m)}m{float(s)}s"
    return Angle(txt, unit=u.hourangle).degree


def _parse_dec_tokens(d: str, m: str, s: str) -> float:
    """Dec from tokens d m s to degrees. Handles signed degrees."""
    # Normalize unicode minus if present
    d = str(d).replace("−", "-")
    sign = "-" if str(d).strip().startswith("-") else "+"
    dd = abs(int(float(d)))  # handle strings like "-65"
    txt = f"{sign}{dd}d{int(m)}m{float(s)}s"
    return Angle(txt, unit=u.deg).degree


def _safe_float(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except Exception:
        return None


def _tokenize_lines(path: str) -> List[List[str]]:
    toks: List[List[str]] = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        # collapse multiple spaces
        parts = s.split()
        toks.append(parts)
    return toks


# -----------------------------
# Parsers for known layouts
# -----------------------------

def _looks_like_table7_tokens(tokens: List[str]) -> bool:
    """
    Heuristic: Bagetakos table-7 rows typically have >= 18 tokens and start with a galaxy name like 'NGC 2403'.
    We just require len>=12 and that the first token is alpha (NGC, IC, DDO, UGC, M, etc.).
    """
    if len(tokens) < 12:
        return False
    return tokens[0].isalpha()  # 'NGC','IC','DDO','UGC','M', etc.


def _parse_table7_tokenized(path: str, target_galaxy: str | None = None) -> pd.DataFrame:
    """
    Parse headerless, whitespace-tokenized per-hole catalog (Bagetakos Table 7).
    Expected token positions (0-based):
      0,1 : GAL name parts (e.g., 'NGC','2403') -> 'NGC 2403'
      2   : Hole ID (int)
      3,4,5 : RA h m s
      6,7,8 : Dec d m s  (d may include sign)
      12  : (optional) local PA of the hole ellipse, deg (if present)
      13  : Diameter arcsec (major)
      14  : axial ratio b/a
      17  : v_exp km/s  (often near the end)
    If some fields aren’t present/convertible, we set NaN and keep going.
    """
    rows = _tokenize_lines(path)
    out = {
        "galaxy": [], "id": [], "ra_deg": [], "dec_deg": [],
        "major_arcsec": [], "minor_arcsec": [], "pa_deg": [],
        "vexp_kms": []
    }

    for parts in rows:
        if not _looks_like_table7_tokens(parts):
            continue

        gal = " ".join(parts[0:2])
        if target_galaxy and gal.lower() != target_galaxy.lower():
            # Keep anyway; filtering can be done later by caller.
            pass

        # ID
        hid = None
        if len(parts) > 2:
            try:
                hid = int(float(parts[2]))
            except Exception:
                hid = None

        # RA/Dec
        ra_deg = dec_deg = np.nan
        if len(parts) > 8:
            try:
                ra_deg = _parse_ra_tokens(parts[3], parts[4], parts[5])
                dec_deg = _parse_dec_tokens(parts[6], parts[7], parts[8])
            except Exception:
                pass

        # PA (optional, token 12)
        pa_deg = np.nan
        if len(parts) > 12:
            val = _safe_float(parts[12])
            if val is not None and -360.0 <= val <= 360.0:
                pa_deg = float(val)

        # Diameter arcsec (token 13), axial ratio b/a (token 14)
        major_arcsec = minor_arcsec = np.nan
        if len(parts) > 14:
            diam_as = _safe_float(parts[13])
            b_over_a = _safe_float(parts[14])
            if diam_as is not None and diam_as > 0:
                major_arcsec = float(diam_as)
                if b_over_a is not None and 0 < b_over_a <= 1.5:
                    minor_arcsec = float(diam_as) * float(b_over_a)
                else:
                    minor_arcsec = major_arcsec

        # v_exp (often token 17)
        vexp_kms = np.nan
        if len(parts) > 17:
            v = _safe_float(parts[17])
            if v is not None:
                vexp_kms = float(v)

        out["galaxy"].append(gal)
        out["id"].append(hid)
        out["ra_deg"].append(ra_deg)
        out["dec_deg"].append(dec_deg)
        out["major_arcsec"].append(major_arcsec)
        out["minor_arcsec"].append(minor_arcsec)
        out["pa_deg"].append(pa_deg)
        out["vexp_kms"].append(vexp_kms)

    df = pd.DataFrame(out)
    # drop rows without RA/Dec
    df = df[pd.notnull(df["ra_deg"]) & pd.notnull(df["dec_deg"])].reset_index(drop=True)
    return df


def _norm_geom_df(df: pd.DataFrame, hints: Dict) -> pd.DataFrame:
    """
    Normalize a geometry DataFrame that already has named columns.
    (Used for CDS/CSV fallback.)
    """
    def _guess_col(df: pd.DataFrame, keys) -> Optional[str]:
        low = {c.lower(): c for c in df.columns}
        for k in keys:
            if k in low: return low[k]
        for lk, orig in low.items():
            if any(lk.startswith(x) for x in keys): return orig
        return None

    ra  = hints.get("ra_col")  or _guess_col(df, ["ra(deg)","ra_deg","raj2000","radeg","ra","_ra"])
    dec = hints.get("dec_col") or _guess_col(df, ["dec(deg)","dec_deg","dej2000","dedeg","dec","_de"])
    maj = hints.get("maj_col") or _guess_col(df, ["major","major_arcsec","dmaj","a","maj"])
    mii = hints.get("min_col") or _guess_col(df, ["minor","minor_arcsec","dmin","b","min"])
    pa  = hints.get("pa_col")  or _guess_col(df, ["pa","pa_deg"])
    idc = hints.get("id_col")  or _guess_col(df, ["id","hole","name","index","no"])
    galc= _guess_col(df, ["galaxy","name","ngc","object"])

    if ra is None or dec is None:
        raise ValueError(f"Could not find RA/Dec columns. Columns: {list(df.columns)}")

    # RA/Dec may be sexagesimal strings or degrees; let Angle handle strings
    def parse_ra_any(x):
        if pd.isna(x): return np.nan
        sx = str(x)
        if any(ch in sx for ch in (":","h","m","s")):
            return Angle(sx, unit=u.hourangle).degree
        return float(sx)

    def parse_dec_any(x):
        if pd.isna(x): return np.nan
        sx = str(x).replace("−","-")
        if any(ch in sx for ch in (":","d","m","s","+","-")):
            return Angle(sx, unit=u.deg).degree
        return float(sx)

    ra_deg  = df[ra].apply(parse_ra_any)
    dec_deg = df[dec].apply(parse_dec_any)

    major_arcsec = pd.to_numeric(df[maj], errors="coerce") if maj else pd.Series(np.nan, index=df.index)
    minor_arcsec = pd.to_numeric(df[mii], errors="coerce") if mii else major_arcsec.copy()
    pa_deg = pd.to_numeric(df[pa], errors="coerce") if pa else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "galaxy": df[galc] if galc else pd.Series([None]*len(df)),
        "id": df[idc] if idc in df else pd.Series([None]*len(df)),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "major_arcsec": major_arcsec,
        "minor_arcsec": minor_arcsec,
        "pa_deg": pa_deg
    })
    return out


def _attach_vexp(base: pd.DataFrame, kin_df: Optional[pd.DataFrame], hints: Dict) -> pd.DataFrame:
    """Attach expansion velocities from a separate table if provided."""
    base = base.copy()
    base["vexp_kms"] = base.get("vexp_kms", pd.Series(np.nan, index=base.index))
    if kin_df is None or kin_df.empty:
        return base

    # Try to find ID and Vexp columns
    def _guess_col(df: pd.DataFrame, keys) -> Optional[str]:
        low = {c.lower(): c for c in df.columns}
        for k in keys:
            if k in low: return low[k]
        for lk, orig in low.items():
            if any(lk.startswith(x) for x in keys): return orig
        return None

    vcol = hints.get("vexp_col") or _guess_col(kin_df, ["vexp","v_exp","vexp_kms","vexp(km/s)","vexpkms"])
    idc  = hints.get("id_col")   or _guess_col(kin_df, ["id","hole","name","index","no"])

    if vcol and idc and ("id" in base.columns):
        tmp = kin_df[[idc, vcol]].copy()
        tmp.columns = ["id", "vexp_kms"]
        # Ensure numeric
        tmp["vexp_kms"] = pd.to_numeric(tmp["vexp_kms"], errors="coerce")
        return base.merge(tmp, on="id", how="left")

    # Fallback: do nothing if we cannot match
    warnings.warn("[catalog] Could not merge kinematics: missing id or vexp column hints.")
    return base


# -----------------------------
# Public entrypoint
# -----------------------------

def load_catalogs(cfg: Dict) -> pd.DataFrame:
    """
    Preferred path: cfg['catalogs'] with holes_dat & kin_dat.
    Fallback: cfg['catalog_csv'] if present (legacy).
    """
    cats = cfg.get("catalogs", {}) or {}

    if cats and cats.get("holes_dat"):
        holes_path = cats["holes_dat"]
        kin_path   = cats.get("kin_dat")
        hints      = cats.get("hints", {})

        # Try to parse as tokenized table-7 first
        try:
            df_holes = _parse_table7_tokenized(holes_path)
            # Attach velocities if needed/available
            if df_holes is not None:
                if "vexp_kms" not in df_holes.columns or df_holes["vexp_kms"].isna().all():
                    kin_df = _read_dat_any(kin_path) if kin_path else None
                    df_holes = _attach_vexp(df_holes, kin_df, hints)

            # NEW: filter to target galaxy (speeds everything up)
            target = cats.get("target_galaxy")
            if target:
                before = len(df_holes)
                df_holes = df_holes[df_holes["galaxy"].str.fullmatch(target, case=False, na=False)]
                print(f"[catalog] filtered to galaxy={target!r}: {before} -> {len(df_holes)} rows")
            parsed_mode = "table7_tokenized"
        except Exception as e:
            df_holes = None
            parsed_mode = None

        # If tokenized parse failed or produced nothing, try generic readers
        if df_holes is None or df_holes.empty:
            df_generic = _read_dat_any(holes_path)
            df_holes = _norm_geom_df(df_generic, hints)
            parsed_mode = parsed_mode or "generic_reader"

        # Attach velocities if needed/available
        if df_holes is not None:
            if "vexp_kms" not in df_holes.columns or df_holes["vexp_kms"].isna().all():
                kin_df = _read_dat_any(kin_path) if kin_path else None
                df_holes = _attach_vexp(df_holes, kin_df, hints)

        # Final normalization of expected columns
        for col in ("ra_deg","dec_deg","major_arcsec","minor_arcsec","pa_deg","vexp_kms","id","galaxy"):
            if col not in df_holes.columns:
                df_holes[col] = np.nan

        # Coerce types
        df_holes["ra_deg"] = pd.to_numeric(df_holes["ra_deg"], errors="coerce")
        df_holes["dec_deg"] = pd.to_numeric(df_holes["dec_deg"], errors="coerce")
        df_holes["major_arcsec"] = pd.to_numeric(df_holes["major_arcsec"], errors="coerce")
        df_holes["minor_arcsec"] = pd.to_numeric(df_holes["minor_arcsec"], errors="coerce")
        df_holes["pa_deg"] = pd.to_numeric(df_holes["pa_deg"], errors="coerce")
        df_holes["vexp_kms"] = pd.to_numeric(df_holes["vexp_kms"], errors="coerce")

        # Report a short summary (useful for verify.sh logs)
        print(f"[catalog] holes_dat parsed via: {parsed_mode}; rows={len(df_holes)}; "
              f"with vexp non-null={df_holes['vexp_kms'].notna().sum()}")

        return df_holes.reset_index(drop=True)

    # Fallback: single CSV (backward-compatible)
    cat_csv = cfg.get("catalog_csv")
    if cat_csv:
        df = pd.read_csv(cat_csv)
        return _norm_geom_df(df, hints={})

    raise ValueError("No catalogs provided. Specify `catalogs.holes_dat` (and optional `catalogs.kin_dat`) or `catalog_csv`.")