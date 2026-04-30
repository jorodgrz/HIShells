"""Shared pytest fixtures.

Tests run from the ``HIShells/`` repo root (``pytest`` from there picks
up ``pyproject.toml``); the fixtures here resolve the on-disk data
products relative to that root so individual tests don't have to.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
B11_DIR = REPO_ROOT / "Data" / "J_AJ_141_23"
THINGS_DIR = REPO_ROOT / "Data" / "THINGS"


@pytest.fixture(scope="session")
def b11_dir() -> Path:
    """Directory containing ``table2.dat`` / ``table7.dat`` / ``ReadMe``."""

    if not (B11_DIR / "table7.dat").exists():
        pytest.skip(f"B11 catalog missing under {B11_DIR}")
    return B11_DIR


@pytest.fixture(scope="session")
def things_dir() -> Path:
    """Directory containing the THINGS NA cubes."""

    if not THINGS_DIR.exists():
        pytest.skip(f"THINGS data dir missing: {THINGS_DIR}")
    return THINGS_DIR
