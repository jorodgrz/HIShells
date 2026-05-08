"""Catalog loading and normalization for PV-shell labels.

The Bagetakos et al. HI-hole table (CDS J/AJ/141/23/table7.dat) gives:

- ``d``: shell diameter in pc. We convert this to an angular *diameter*
  with the configured galaxy distance, then use half of it as the major
  semi-axis for PV-cut intersection tests.
- ``Ratio``: minor/major axis ratio. We assume ``minor_radius =
  major_radius * Ratio``; values missing or invalid fall back to circular.
- ``HV``: heliocentric velocity of the shell center. It must be in the same
  velocity frame as the cube after the caller applies any configured scale or
  offset.
- ``Vexp``: expansion velocity. For type-2/3 shells it is treated as the
  velocity half-width of the expanding-shell label around ``HV``. Type-1 shells
  often have no reliable expansion signature, so they are labeled separately by
  the caller using a local velocity band around ``HV``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from io import StringIO

import numpy as np
import pandas as pd

from src.utils.wcs_tools import pixel_scales_arcsec, radec_to_xy


TABLE7_COLS = [
    "Name", "Seq", "RAh", "RAm", "RAs", "DEsign", "DEd", "DEm", "DEs",
    "HV", "Type", "d_pc", "Vexp", "PA", "Ratio", "R_kpc", "nHI",
    "tkin", "logE", "logMHI",
]

TABLE7_COLSPECS = [
    (0, 11), (12, 15), (16, 18), (19, 21), (22, 26),
    (27, 28), (28, 30), (31, 33), (34, 38),
    (39, 43), (44, 45), (46, 50), (51, 53), (54, 57),
    (58, 61), (62, 66), (67, 71), (72, 75), (76, 80), (81, 85),
]


def _hms_to_deg(h, m, s) -> float:
    return 15.0 * (float(h) + float(m) / 60.0 + float(s) / 3600.0)


def _dms_to_deg(sign_char, d, m, s) -> float:
    sign = -1.0 if str(sign_char).strip() == "-" else 1.0
    return sign * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)


def _data_lines(path: str | Path) -> list[str]:
    """Return only data-like table rows, dropping CDS headers/separators."""
    rows: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("-"):
            continue
        if stripped.startswith("Name") or stripped.startswith("|"):
            continue
        rows.append(line)
    return rows


def _parse_pipe_table7(rows: list[str]) -> pd.DataFrame:
    """Parse CDS text rows whose columns are separated by ``|`` characters."""
    parsed: list[dict] = []
    for line in rows:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 14:
            continue
        coord = parts[2].split()
        if len(coord) != 6:
            continue
        try:
            dec_sign = "-" if coord[3].startswith("-") else "+"
            dec_deg = coord[3].lstrip("+-")
            parsed.append(
                {
                    "Name": parts[0],
                    "shell_id": int(float(parts[1])),
                    "RAh": float(coord[0]),
                    "RAm": float(coord[1]),
                    "RAs": float(coord[2]),
                    "DEsign": dec_sign,
                    "DEd": float(dec_deg),
                    "DEm": float(coord[4]),
                    "DEs": float(coord[5]),
                    "vel_center_kms": float(parts[3]) if parts[3] else np.nan,
                    "shell_type": int(float(parts[4])) if parts[4] else pd.NA,
                    "d_pc": float(parts[5]) if parts[5] else np.nan,
                    "vexp_kms": float(parts[6]) if parts[6] else np.nan,
                    "pa_deg": float(parts[7]) if parts[7] else np.nan,
                    "axis_ratio": float(parts[8]) if parts[8] else np.nan,
                    "R_kpc": float(parts[9]) if parts[9] else np.nan,
                    "nHI": float(parts[10]) if parts[10] else np.nan,
                    "tkin": float(parts[11]) if parts[11] else np.nan,
                    "logE": float(parts[12]) if parts[12] else np.nan,
                    "logMHI": float(parts[13]) if parts[13] else np.nan,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(parsed)


def load_bagetakos_table7(
    path: str | Path,
    target_galaxy: str | None = None,
    keep_types: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load Bagetakos-style shell rows into a normalized DataFrame.

    ``keep_types=None`` keeps all shell types. The returned size fields are
    still physical/catalog quantities; pixel conversion is handled by
    :func:`catalog_to_pixel_shells` once a cube WCS is available.
    """
    rows = _data_lines(path)
    if any("|" in row for row in rows):
        df = _parse_pipe_table7(rows)
    else:
        df = pd.read_fwf(
            StringIO("\n".join(rows)),
            colspecs=TABLE7_COLSPECS,
            names=TABLE7_COLS,
            header=None,
        )

    df = df.rename(
        columns={
            "Seq": "shell_id",
            "Type": "shell_type",
            "HV": "vel_center_kms",
            "Vexp": "vexp_kms",
            "PA": "pa_deg",
            "Ratio": "axis_ratio",
        }
    )

    df["Name"] = df["Name"].astype(str).str.strip()
    if target_galaxy:
        df = df[df["Name"].str.casefold() == str(target_galaxy).casefold()].copy()

    for col in ("shell_id", "RAh", "RAm", "DEd", "DEm", "vel_center_kms", "shell_type", "d_pc", "vexp_kms", "pa_deg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("RAs", "DEs", "axis_ratio", "R_kpc", "nHI", "tkin", "logE", "logMHI"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if keep_types is not None:
        keep = {int(t) for t in keep_types}
        df = df[df["shell_type"].isin(keep)].copy()

    df = df.dropna(subset=["RAh", "RAm", "RAs", "DEd", "DEm", "DEs"]).copy()
    df["ra_deg"] = [
        _hms_to_deg(h, m, s) for h, m, s in zip(df["RAh"], df["RAm"], df["RAs"])
    ]
    df["dec_deg"] = [
        _dms_to_deg(sign, d, m, s)
        for sign, d, m, s in zip(df["DEsign"], df["DEd"], df["DEm"], df["DEs"])
    ]

    df["axis_ratio"] = df["axis_ratio"].where(
        (df["axis_ratio"] > 0) & (df["axis_ratio"] <= 1.5), 1.0
    )
    df["shell_type"] = df["shell_type"].astype("Int64")
    df["shell_id"] = df["shell_id"].astype("Int64")

    cols = [
        "Name", "shell_id", "shell_type", "ra_deg", "dec_deg",
        "vel_center_kms", "d_pc", "vexp_kms", "pa_deg", "axis_ratio",
        "R_kpc", "nHI", "tkin", "logE", "logMHI",
    ]
    return df[cols].reset_index(drop=True)


def catalog_to_pixel_shells(
    catalog: pd.DataFrame,
    *,
    wcs,
    distance_mpc: float,
    hv_scale: float = 1.0,
    hv_offset: float = 0.0,
) -> tuple[list[dict], dict]:
    """Convert catalog shell rows to pixel and cube-velocity coordinates."""
    ax_as, ay_as = pixel_scales_arcsec(wcs)
    pix_per_as = 1.0 / max(ax_as, ay_as, 1e-9)
    warnings: list[dict] = []
    shells: list[dict] = []

    for _, row in catalog.iterrows():
        shell_id = None if pd.isna(row["shell_id"]) else int(row["shell_id"])
        shell_type = None if pd.isna(row["shell_type"]) else int(row["shell_type"])

        if not np.isfinite(row["ra_deg"]) or not np.isfinite(row["dec_deg"]):
            warnings.append({"shell_id": shell_id, "reason": "missing_ra_dec"})
            continue
        if not np.isfinite(row["d_pc"]) or float(row["d_pc"]) <= 0:
            warnings.append({"shell_id": shell_id, "type": shell_type, "reason": "invalid_diameter_pc"})
            continue
        if not distance_mpc or not np.isfinite(distance_mpc) or distance_mpc <= 0:
            warnings.append({"shell_id": shell_id, "type": shell_type, "reason": "invalid_distance_mpc"})
            continue

        x, y = radec_to_xy(wcs, float(row["ra_deg"]), float(row["dec_deg"]))
        diameter_arcsec = float(row["d_pc"]) / float(distance_mpc) * 0.206265
        major_radius_arcsec = 0.5 * diameter_arcsec
        axis_ratio = float(row["axis_ratio"]) if np.isfinite(row["axis_ratio"]) else 1.0
        minor_radius_arcsec = major_radius_arcsec * axis_ratio

        vel_raw = float(row["vel_center_kms"]) if np.isfinite(row["vel_center_kms"]) else np.nan
        vel_center = vel_raw * hv_scale + hv_offset if np.isfinite(vel_raw) else np.nan
        vexp = float(row["vexp_kms"]) if np.isfinite(row["vexp_kms"]) else np.nan
        if np.isfinite(vexp):
            vexp *= hv_scale

        if not np.isfinite(vel_center):
            warnings.append({"shell_id": shell_id, "type": shell_type, "reason": "missing_velocity_center"})
        if shell_type in (2, 3) and (not np.isfinite(vexp) or vexp <= 0):
            warnings.append({"shell_id": shell_id, "type": shell_type, "reason": "missing_expansion_velocity"})

        shells.append(
            {
                "galaxy": row["Name"],
                "shell_id": shell_id,
                "type": shell_type,
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "xc": float(x),
                "yc": float(y),
                "diameter_pc": float(row["d_pc"]),
                "diameter_arcsec": float(diameter_arcsec),
                "a_arcsec": float(major_radius_arcsec),
                "b_arcsec": float(minor_radius_arcsec),
                "a_pix": float(major_radius_arcsec * pix_per_as),
                "b_pix": float(minor_radius_arcsec * pix_per_as),
                "pa_deg": float(row["pa_deg"]) if np.isfinite(row["pa_deg"]) else 0.0,
                "axis_ratio": axis_ratio,
                "vel_center": float(vel_center) if np.isfinite(vel_center) else np.nan,
                "vel_center_catalog_kms": float(vel_raw) if np.isfinite(vel_raw) else np.nan,
                "vexp": float(vexp) if np.isfinite(vexp) else np.nan,
                "vexp_catalog_kms": float(row["vexp_kms"]) if np.isfinite(row["vexp_kms"]) else np.nan,
            }
        )

    diagnostics = {
        "catalog_rows": int(len(catalog)),
        "usable_shells": int(len(shells)),
        "warnings": warnings,
        "assumptions": {
            "d_pc": "Bagetakos d is treated as full shell diameter in pc; labels use d/2 as major radius.",
            "axis_ratio": "Bagetakos Ratio is treated as minor/major axis ratio.",
            "vel_center": "Bagetakos HV is treated as the label velocity center after hv_scale and hv_offset.",
            "vexp": "Bagetakos Vexp is treated as velocity half-width for type 2/3 expanding-shell labels.",
        },
    }
    return shells, diagnostics
