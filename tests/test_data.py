"""Tests for ``hishells.data`` and ``hishells.windows``.

The dataset tests use a tiny synthetic in-memory cube + a fake catalog
DataFrame so they don't depend on the THINGS download.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from hishells.augment import AugmentConfig
from hishells.cubes import Cube
from hishells.data import (
    CubeStore,
    DatasetConfig,
    LOGOSplitter,
    ShellWindowDataset,
    make_subset,
)
from hishells.windows import (
    NegSampleConfig,
    build_window_table,
    emission_free_mask,
    label_positives,
    normalize_window,
    sample_negatives,
)


# ---------------------------------------------------------------------------
# Synthetic cube helpers
# ---------------------------------------------------------------------------


def _synth_cube(galaxy_id: str = "FAKE", n_pix: int = 96, n_chan: int = 31) -> Cube:
    rng = np.random.default_rng(123)
    data = rng.normal(0, 0.001, size=(n_chan, n_pix, n_pix)).astype(np.float32)
    # Add some bright emission in the central half so the
    # emission-free mask is non-trivial.
    data[:, 24:72, 24:72] += 0.01
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-1.5 / 3600.0, 1.5 / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return Cube(
        data=data,
        wcs2d=w,
        velocity_kms=np.linspace(60, -60, n_chan),
        beam_bmaj_arcsec=6.0,
        beam_bmin_arcsec=6.0,
        beam_bpa_deg=0.0,
        pixel_scale_arcsec=1.5,
        galaxy_id=galaxy_id,
        path=Path("synth"),
    )


def _fake_positives(galaxy_id: str = "FAKE", n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append(
            {
                "galaxy_id": galaxy_id,
                "hole_idx": i + 1,
                "ra_deg": 180.0 + rng.uniform(-0.005, 0.005),
                "dec_deg": 30.0 + rng.uniform(-0.005, 0.005),
                "vel_helio_kms": float(rng.uniform(-30, 30)),
                "pa_deg": float(rng.uniform(0, 180)),
                "diameter_arcsec": 12.0,
                "diameter_pc": 200.0,
                "vexp_kms": 12.0,
                "sigma_gas_kms": 10.0,
                "hole_type": 3,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# windows.py
# ---------------------------------------------------------------------------


def test_normalize_window_zero_median_unit_scale():
    rng = np.random.default_rng(0)
    win = rng.normal(5.0, 2.0, size=(64, 64)).astype(np.float32)
    out = normalize_window(win, cube_sigma=2.0)
    assert abs(np.median(out)) < 1e-5
    assert out.dtype == np.float32


def test_emission_free_mask_excludes_bright_center():
    cube = _synth_cube()
    mask = emission_free_mask(cube, sigma_threshold=3.0)
    # Bright square at [24:72, 24:72] should be largely masked-out.
    assert mask[40, 40] == False  # noqa: E712
    # Corner should be emission-free.
    assert mask[5, 5] == True  # noqa: E712


def test_sample_negatives_count_and_ratio():
    cube = _synth_cube()
    pos = _fake_positives(n=5)
    cfg = NegSampleConfig(ratio=4.0, hard_frac=0.75, rng_seed=0)
    negs = sample_negatives(cube, pos, cfg)
    # 4 negatives per positive, 5 positives => 20 negatives.
    assert len(negs) == 20
    n_hard = (negs["neg_kind"] == "hard").sum()
    n_easy = (negs["neg_kind"] == "easy").sum()
    # Allow rounding slack on hard_frac.
    assert n_hard >= 14 and n_easy >= 4
    assert (negs["label"] == 0).all()


def test_build_window_table_concatenates():
    cube = _synth_cube()
    pos = _fake_positives(n=4)
    cfg = NegSampleConfig(ratio=3.0, rng_seed=1)
    table = build_window_table(pos, {"FAKE": cube}, cfg)
    assert (table["label"] == 1).sum() == 4
    assert (table["label"] == 0).sum() == 12


# ---------------------------------------------------------------------------
# data.py -- Dataset
# ---------------------------------------------------------------------------


class _FakeStore:
    """CubeStore stand-in that returns a fixed in-memory cube."""

    def __init__(self, cube: Cube):
        self._cube = cube

    def __call__(self, gid: str) -> Cube:
        assert gid == self._cube.galaxy_id
        return self._cube


def test_shell_window_dataset_shapes():
    cube = _synth_cube()
    pos = _fake_positives(n=3)
    cfg = NegSampleConfig(ratio=2.0, rng_seed=0)
    table = build_window_table(pos, {"FAKE": cube}, cfg)

    ds = ShellWindowDataset(
        table=table,
        cubes=_FakeStore(cube),  # type: ignore[arg-type]
        config=DatasetConfig(window_pix=64, augment=AugmentConfig()),
    )
    assert len(ds) == len(table)
    x, y, gid = ds[0]
    assert tuple(x.shape) == (1, 64, 64)
    assert y.dtype.is_floating_point
    assert gid == "FAKE"


def test_logo_splitter_no_galaxy_leak():
    # Tiny multi-galaxy table with 3 galaxies x 4 positives + negatives.
    rows = []
    for gid in ("A", "B", "C"):
        for i in range(4):
            rows.append({"galaxy_id": gid, "label": 1, "hole_idx": i})
        for i in range(8):
            rows.append({"galaxy_id": gid, "label": 0, "hole_idx": -i})
    table = pd.DataFrame(rows)

    splitter = LOGOSplitter(table, galaxies=("A", "B", "C"), val_frac=0.25, rng_seed=0)
    folds = list(splitter)
    assert len(folds) == 3
    seen_test = set()
    for fold in folds:
        seen_test.add(fold.test_galaxy)
        # No row appears in both train and test.
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))
        assert set(fold.val_idx).isdisjoint(set(fold.test_idx))
        # Test set is exactly the held-out galaxy.
        assert (table.iloc[fold.test_idx]["galaxy_id"] == fold.test_galaxy).all()
        # Train + val galaxies exclude the held-out one.
        assert (table.iloc[fold.train_idx]["galaxy_id"] != fold.test_galaxy).all()
        assert (table.iloc[fold.val_idx]["galaxy_id"] != fold.test_galaxy).all()
        # Val rows are positives only.
        assert (table.iloc[fold.val_idx]["label"] == 1).all()
    assert seen_test == {"A", "B", "C"}


def test_make_subset_preserves_cube_cache():
    cube = _synth_cube()
    pos = _fake_positives(n=3)
    cfg = NegSampleConfig(ratio=1.0, rng_seed=0)
    table = build_window_table(pos, {"FAKE": cube}, cfg)
    parent = ShellWindowDataset(
        table=table,
        cubes=_FakeStore(cube),  # type: ignore[arg-type]
        sigma_rms_by_galaxy={"FAKE": 0.001},
    )
    sub = make_subset(parent, np.arange(2))
    assert len(sub) == 2
    assert sub._sigma_rms is parent._sigma_rms
