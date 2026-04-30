"""Bagetakos+ 2011 (B11) HI hole catalog loader.

The CDS bundle for J/AJ/141/23 ships two fixed-width tables under
``Data/J_AJ_141_23/``:

* ``table2.dat`` -- 20 rows of per-galaxy properties (distance,
  inclination, position angle, HI mass, SFR, etc.).
* ``table7.dat`` -- 1046 rows of per-hole properties (RA/Dec, helio
  velocity, hole type 1/2/3, diameter, expansion velocity, PA, axial
  ratio, galactocentric radius, density, kinetic age, energy, missing
  HI mass).

This module parses both into pandas DataFrames keyed by
``galaxy_id`` (the THINGS filename stem, e.g. ``"NGC_2403"``,
``"HO_II"``) so the rest of the pipeline can join hole rows to the
correct ``Data/THINGS/<galaxy_id>_NA_CUBE_THINGS.FITS``.

Byte specs are taken verbatim from ``Data/J_AJ_141_23/ReadMe``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Map B11 catalog name (with internal whitespace, exactly as it appears
# in tableN.dat columns 1-11) to the THINGS file stem used by
# ``scripts/fetch_things.py`` and the rest of the pipeline.
NAME_TO_THINGS_STEM: dict[str, str] = {
    "NGC 628": "NGC_628",
    "NGC 2366": "NGC_2366",
    "NGC 2403": "NGC_2403",
    "Holmberg II": "HO_II",
    "DDO 53": "DDO53",
    "NGC 2841": "NGC_2841",
    "Holmberg I": "HO_I",
    "NGC 2976": "NGC_2976",
    "NGC 3031": "NGC_3031",
    "NGC 3184": "NGC_3184",
    "IC 2574": "IC_2574",
    "NGC 3521": "NGC_3521",
    "NGC 3627": "NGC_3627",
    "NGC 4214": "NGC_4214",
    "NGC 4449": "NGC_4449",
    "NGC 4736": "NGC_4736",
    "DDO 154": "DDO154",
    "NGC 5194": "NGC_5194",
    "NGC 6946": "NGC_6946",
    "NGC 7793": "NGC_7793",
}

# Inverse lookup, kept as a plain dict so callers can
# ``THINGS_STEM_TO_NAME[stem]`` for plot labels etc.
THINGS_STEM_TO_NAME: dict[str, str] = {v: k for k, v in NAME_TO_THINGS_STEM.items()}

# IC 2574 is in the catalog but the THINGS public mirror does not host
# its cube (see ``scripts/fetch_things.py`` and plan §6 / §12).
LOGO_GALAXIES_19: tuple[str, ...] = tuple(
    sorted(stem for stem in NAME_TO_THINGS_STEM.values() if stem != "IC_2574")
)

# B11 reports a per-galaxy gas velocity dispersion in their Table 3
# ("6-14 km/s per galaxy" -- plan §2.1). Table 3 is not in the CDS
# bundle, so we hard-code typical values; they are used only as the
# Type-1 (no measurable Vexp) fallback for ``vel_extent`` in
# ``hishells.pvcut.window_extent_for_hole``. The default 10 km/s for
# missing entries is the midpoint of the B11-quoted range.
SIGMA_GAS_KMS_BY_STEM: dict[str, float] = {
    "NGC_628": 10.0,
    "NGC_2366": 9.0,
    "NGC_2403": 10.0,
    "HO_II": 9.0,
    "DDO53": 9.0,
    "NGC_2841": 12.0,
    "HO_I": 8.0,
    "NGC_2976": 10.0,
    "NGC_3031": 12.0,
    "NGC_3184": 11.0,
    "IC_2574": 9.0,
    "NGC_3521": 12.0,
    "NGC_3627": 12.0,
    "NGC_4214": 9.0,
    "NGC_4449": 10.0,
    "NGC_4736": 11.0,
    "DDO154": 8.0,
    "NGC_5194": 12.0,
    "NGC_6946": 12.0,
    "NGC_7793": 10.0,
}


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------


def _slice(line: str, start: int, end: int) -> str:
    """Extract bytes ``[start, end]`` (1-indexed, inclusive) from ``line``.

    The CDS byte spec uses 1-indexed inclusive ranges; Python slicing is
    0-indexed half-open, so ``line[start - 1:end]`` is the right
    translation.
    """

    return line[start - 1 : end]


def _f(line: str, start: int, end: int) -> float | None:
    raw = _slice(line, start, end).strip()
    if not raw:
        return None
    return float(raw)


def _i(line: str, start: int, end: int) -> int | None:
    raw = _slice(line, start, end).strip()
    if not raw:
        return None
    return int(raw)


def _hms_to_deg(rah: int, ram: int, ras: float) -> float:
    """Sexagesimal hours/minutes/seconds -> decimal degrees of RA."""

    return (rah + ram / 60.0 + ras / 3600.0) * 15.0


def _dms_to_deg(sign: str, ded: int, dem: int, des: float) -> float:
    """Sexagesimal degrees/minutes/seconds -> decimal degrees of Dec.

    ``sign`` is ``'-'`` for southern declinations and any of ``''``,
    ``' '``, or ``'+'`` for northern declinations (the CDS table uses a
    blank sign for ``+``).
    """

    deg = ded + dem / 60.0 + des / 3600.0
    return -deg if sign.strip() == "-" else deg


# ---------------------------------------------------------------------------
# table7.dat -- HI hole catalog (1046 rows)
# ---------------------------------------------------------------------------


_TABLE7_COLUMNS: tuple[str, ...] = (
    "name_b11",
    "galaxy_id",
    "hole_idx",
    "ra_deg",
    "dec_deg",
    "vel_helio_kms",
    "hole_type",
    "diameter_pc",
    "vexp_kms",
    "pa_deg",
    "axial_ratio",
    "gc_radius_kpc",
    "n_HI_raw",
    "t_kin_myr",
    "log_E_J43",
    "log_MHI_1e4Msun",
)


def _parse_table7_line(line: str) -> dict | None:
    """Parse one row of ``table7.dat`` per the CDS byte spec.

    Returns ``None`` for lines shorter than 85 chars (e.g. trailing
    whitespace-only lines). Otherwise returns a dict with one key per
    entry in ``_TABLE7_COLUMNS``.
    """

    # The byte spec runs to column 85. Pad short lines so ``_slice``
    # never reads past the end of an unexpectedly-trimmed line.
    if len(line.rstrip("\n")) < 11:
        return None
    line = line.rstrip("\n").ljust(85)

    name_b11 = _slice(line, 1, 11).strip()
    if not name_b11:
        return None

    seq = _i(line, 13, 15)
    rah = _i(line, 17, 18)
    ram = _i(line, 20, 21)
    ras = _f(line, 23, 26)
    de_sign = _slice(line, 28, 28)
    ded = _i(line, 29, 30)
    dem = _i(line, 32, 33)
    des = _f(line, 35, 38)
    hv = _i(line, 40, 43)
    htype = _i(line, 45, 45)
    diam = _i(line, 47, 50)
    vexp = _i(line, 52, 53)
    pa = _i(line, 55, 57)
    ratio = _f(line, 59, 61)
    r_gc = _f(line, 63, 66)
    n_hi = _f(line, 68, 71)
    tkin = _i(line, 73, 75)
    log_e = _f(line, 77, 80)
    log_mhi = _f(line, 82, 85)

    if seq is None or rah is None or ded is None:
        return None

    return {
        "name_b11": name_b11,
        "galaxy_id": NAME_TO_THINGS_STEM.get(name_b11, name_b11.replace(" ", "_")),
        "hole_idx": seq,
        "ra_deg": _hms_to_deg(rah, ram or 0, ras or 0.0),
        "dec_deg": _dms_to_deg(de_sign, ded, dem or 0, des or 0.0),
        "vel_helio_kms": float(hv) if hv is not None else np.nan,
        "hole_type": htype if htype is not None else 0,
        "diameter_pc": float(diam) if diam is not None else np.nan,
        "vexp_kms": float(vexp) if vexp is not None else np.nan,
        "pa_deg": float(pa) if pa is not None else np.nan,
        "axial_ratio": ratio if ratio is not None else np.nan,
        "gc_radius_kpc": r_gc if r_gc is not None else np.nan,
        # B11 lists this column as "cm-3" but the values are clearly
        # log10(n_HI/cm^-3) (range -1.5..0.5). Preserve the raw value
        # and let downstream code decide.
        "n_HI_raw": n_hi if n_hi is not None else np.nan,
        "t_kin_myr": float(tkin) if tkin is not None else np.nan,
        "log_E_J43": log_e if log_e is not None else np.nan,
        "log_MHI_1e4Msun": log_mhi if log_mhi is not None else np.nan,
    }


def load_holes(table7_path: str | Path) -> pd.DataFrame:
    """Parse ``table7.dat`` and return the 1046-row hole catalog.

    Columns are ``_TABLE7_COLUMNS`` plus ``diameter_arcsec`` derived
    from ``diameter_pc`` and the per-galaxy distance in ``table2.dat``
    if it sits next to ``table7.dat`` (it normally does). If the
    ``table2.dat`` sibling is missing, ``diameter_arcsec`` is filled
    with NaN and downstream code must handle that.
    """

    path = Path(table7_path)
    rows: list[dict] = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            row = _parse_table7_line(line)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=list(_TABLE7_COLUMNS))

    # Derive diameter in arcsec from physical diameter + per-galaxy
    # distance. The conversion is angle [rad] = size [pc] / (1e6 *
    # distance [Mpc] * pc/m); in arcsec it's
    # (d_pc / D_Mpc) * (206265 / 1e6) ≈ d_pc / D_Mpc * 0.206265.
    table2_path = path.parent / "table2.dat"
    if table2_path.exists():
        gal = load_galaxies(table2_path).set_index("galaxy_id")
        dist_map = gal["distance_mpc"].to_dict()
        df["distance_mpc"] = df["galaxy_id"].map(dist_map).astype(float)
        df["diameter_arcsec"] = df["diameter_pc"] / df["distance_mpc"] * 0.206265
    else:
        df["distance_mpc"] = np.nan
        df["diameter_arcsec"] = np.nan

    df["sigma_gas_kms"] = df["galaxy_id"].map(SIGMA_GAS_KMS_BY_STEM).astype(float)
    df["sigma_gas_kms"] = df["sigma_gas_kms"].fillna(10.0)

    return df


# ---------------------------------------------------------------------------
# table2.dat -- per-galaxy properties (20 rows)
# ---------------------------------------------------------------------------


_TABLE2_COLUMNS: tuple[str, ...] = (
    "name_b11",
    "galaxy_id",
    "other_name",
    "ra_deg",
    "dec_deg",
    "morph_type",
    "distance_mpc",
    "inclination_deg",
    "pa_deg",
    "MHI_1e8Msun",
    "log_sfr",
    "log_d25",
    "resolution_pc",
)


def _parse_table2_line(line: str) -> dict | None:
    if len(line.rstrip("\n")) < 11:
        return None
    line = line.rstrip("\n").ljust(88)

    name_b11 = _slice(line, 1, 11).strip()
    if not name_b11:
        return None

    other = _slice(line, 13, 21).strip()
    rah = _i(line, 23, 24)
    ram = _i(line, 26, 27)
    ras = _f(line, 29, 32)
    de_sign = _slice(line, 34, 34)
    ded = _i(line, 35, 36)
    dem = _i(line, 38, 39)
    des = _f(line, 41, 44)
    morph = _slice(line, 46, 55).strip()
    dist = _f(line, 57, 60)
    incl = _i(line, 62, 63)
    pa = _i(line, 65, 67)
    mhi = _f(line, 69, 72)
    logsfr = _f(line, 74, 78)
    logd25 = _f(line, 81, 84)
    res = _i(line, 86, 88)

    if rah is None or ded is None:
        return None

    return {
        "name_b11": name_b11,
        "galaxy_id": NAME_TO_THINGS_STEM.get(name_b11, name_b11.replace(" ", "_")),
        "other_name": other,
        "ra_deg": _hms_to_deg(rah, ram or 0, ras or 0.0),
        "dec_deg": _dms_to_deg(de_sign, ded, dem or 0, des or 0.0),
        "morph_type": morph,
        "distance_mpc": dist if dist is not None else np.nan,
        "inclination_deg": float(incl) if incl is not None else np.nan,
        "pa_deg": float(pa) if pa is not None else np.nan,
        "MHI_1e8Msun": mhi if mhi is not None else np.nan,
        "log_sfr": logsfr if logsfr is not None else np.nan,
        "log_d25": logd25 if logd25 is not None else np.nan,
        "resolution_pc": float(res) if res is not None else np.nan,
    }


def load_galaxies(table2_path: str | Path) -> pd.DataFrame:
    """Parse ``table2.dat`` and return the 20-row galaxy table."""

    path = Path(table2_path)
    rows: list[dict] = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            row = _parse_table2_line(line)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows, columns=list(_TABLE2_COLUMNS))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B11Catalog:
    """Bundle of the parsed per-hole and per-galaxy tables.

    Attributes
    ----------
    holes : pd.DataFrame
        One row per HI hole. Always 1046 rows for the CDS bundle.
    galaxies : pd.DataFrame
        One row per galaxy. Always 20 rows for the CDS bundle.
    """

    holes: pd.DataFrame
    galaxies: pd.DataFrame

    def filter(
        self,
        *,
        galaxy_ids: Iterable[str] | None = None,
        hole_types: Iterable[int] | None = None,
        downloaded_in: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return ``holes`` filtered by galaxy stem, hole type, and disk presence.

        Parameters
        ----------
        galaxy_ids
            Restrict to these THINGS stems (e.g. ``["NGC_2403"]``).
        hole_types
            Restrict to these B11 hole types (default: all of {1,2,3}).
            v1 default per plan §2.4 is ``(2, 3)``.
        downloaded_in
            If supplied, also drop rows whose
            ``Data/THINGS/<galaxy_id>_NA_CUBE_THINGS.FITS`` is missing
            from this directory. Useful so notebooks only see holes for
            cubes that ``fetch_things.py`` actually delivered.
        """

        df = self.holes
        if galaxy_ids is not None:
            df = df[df["galaxy_id"].isin(set(galaxy_ids))]
        if hole_types is not None:
            df = df[df["hole_type"].isin(set(hole_types))]
        if downloaded_in is not None:
            d = Path(downloaded_in)
            present = {
                p.name.replace("_NA_CUBE_THINGS.FITS", "")
                for p in d.glob("*_NA_CUBE_THINGS.FITS")
            }
            df = df[df["galaxy_id"].isin(present)]
        return df.reset_index(drop=True)


def load_catalog(catalog_dir: str | Path = "Data/J_AJ_141_23") -> B11Catalog:
    """Load both B11 tables from ``catalog_dir`` (default
    ``Data/J_AJ_141_23``) and return them in a :class:`B11Catalog`.
    """

    d = Path(catalog_dir)
    return B11Catalog(
        holes=load_holes(d / "table7.dat"),
        galaxies=load_galaxies(d / "table2.dat"),
    )
