"""Unit tests for ``hishells.catalog`` against the on-disk B11 tables."""

from __future__ import annotations

import math

import pytest

from hishells.catalog import (
    LOGO_GALAXIES_19,
    NAME_TO_THINGS_STEM,
    THINGS_STEM_TO_NAME,
    load_catalog,
    load_galaxies,
    load_holes,
)


def test_holes_row_count_and_galaxies(b11_dir):
    holes = load_holes(b11_dir / "table7.dat")
    assert len(holes) == 1046
    assert holes["galaxy_id"].nunique() == 20
    assert set(holes["hole_type"].unique()) <= {1, 2, 3}


def test_galaxies_row_count(b11_dir):
    gals = load_galaxies(b11_dir / "table2.dat")
    assert len(gals) == 20
    assert set(gals["galaxy_id"]) == set(NAME_TO_THINGS_STEM.values())


def test_first_row_spot_check(b11_dir):
    """First row of table7.dat:

    ``NGC 628       1  1 36 20.1  15 43 45.2  645 1  520  7 109 0.7 13.1 -1.1  36  1.2  1.2``

    Verify the byte parser converts each field correctly.
    """

    holes = load_holes(b11_dir / "table7.dat")
    row = holes.iloc[0]
    assert row["name_b11"] == "NGC 628"
    assert row["galaxy_id"] == "NGC_628"
    assert row["hole_idx"] == 1
    # 1h36m20.1s = 1.6055833... h * 15 = 24.0838 deg
    assert row["ra_deg"] == pytest.approx(
        (1 + 36 / 60 + 20.1 / 3600) * 15.0, abs=1e-6
    )
    assert row["dec_deg"] == pytest.approx(
        15 + 43 / 60 + 45.2 / 3600, abs=1e-6
    )
    assert row["vel_helio_kms"] == 645.0
    assert row["hole_type"] == 1
    assert row["diameter_pc"] == 520.0
    assert row["vexp_kms"] == 7.0
    assert row["pa_deg"] == 109.0
    assert row["axial_ratio"] == pytest.approx(0.7)
    assert row["gc_radius_kpc"] == pytest.approx(13.1)
    assert row["t_kin_myr"] == 36
    assert row["log_E_J43"] == pytest.approx(1.2)
    assert row["log_MHI_1e4Msun"] == pytest.approx(1.2)


def test_last_row_spot_check(b11_dir):
    """Last row of table7.dat (NGC 7793 #27)."""

    holes = load_holes(b11_dir / "table7.dat")
    row = holes.iloc[-1]
    assert row["name_b11"] == "NGC 7793"
    assert row["galaxy_id"] == "NGC_7793"
    assert row["hole_idx"] == 27
    # Negative declination
    assert row["dec_deg"] < 0
    assert row["dec_deg"] == pytest.approx(
        -(32 + 37 / 60 + 11.8 / 3600), abs=1e-6
    )


def test_name_normalisation_round_trip():
    for stem in NAME_TO_THINGS_STEM.values():
        assert THINGS_STEM_TO_NAME[stem] in NAME_TO_THINGS_STEM
        assert NAME_TO_THINGS_STEM[THINGS_STEM_TO_NAME[stem]] == stem


def test_logo_galaxies_19_excludes_ic2574():
    assert "IC_2574" not in LOGO_GALAXIES_19
    assert len(LOGO_GALAXIES_19) == 19


def test_load_catalog_filter_by_hole_type(b11_dir):
    cat = load_catalog(b11_dir)
    df_23 = cat.filter(hole_types=(2, 3))
    df_all = cat.filter()
    assert len(df_23) < len(df_all)
    assert set(df_23["hole_type"].unique()) <= {2, 3}


def test_diameter_arcsec_present_after_load(b11_dir):
    cat = load_catalog(b11_dir)
    assert "diameter_arcsec" in cat.holes.columns
    assert math.isfinite(cat.holes["diameter_arcsec"].iloc[0])
