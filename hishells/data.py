"""PyTorch Dataset + LOGO splitter wiring everything together.

Pipeline (per plan \u00a72 + \u00a76):

1. :class:`ShellWindowDataset` is constructed with a *window table*
   (positives + negatives, schema from
   :func:`hishells.windows.build_window_table`) and a directory of
   THINGS cubes.
2. ``__getitem__`` lazily loads the relevant cube (LRU-cached by
   ``galaxy_id``), extracts the p-v window via
   :func:`hishells.pvcut.extract_window_for_hole`, normalises with
   :func:`hishells.windows.normalize_window`, and (in train mode)
   passes it through :class:`hishells.augment.Augmenter`.
3. :class:`LOGOSplitter` yields 19 ``(train_idx, val_idx, test_idx)``
   triples over the 19 LOGO galaxies; the val split is 10% of the
   train *positives* (negatives are resampled per epoch via the
   sampler in ``hishells.windows`` -- the splitter just keeps the
   train/test partition honest).

This module does not import ``torch`` at module load time; it does
inside the dataset class so that catalog-only users (notebook 01)
don't have to install PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .augment import AugmentConfig, Augmenter, no_augment
from .catalog import LOGO_GALAXIES_19
from .cubes import Cube, channel_width_kms, load_cube, sigma_rms
from .pvcut import extract_window_for_hole
from .windows import normalize_window


# ---------------------------------------------------------------------------
# Cube cache
# ---------------------------------------------------------------------------


class CubeStore:
    """Lazy LRU cache of THINGS cubes keyed by galaxy_id.

    Loading a single THINGS cube is ~1-3 GB; we only keep the two most
    recently used in memory by default, which is enough for sequential
    per-fold training (positives + negatives for one galaxy at a time)
    while leaving headroom on a 16-GB workstation.
    """

    def __init__(
        self,
        cube_dir: str | Path,
        *,
        weighting: str = "NA",
        max_cubes: int = 2,
    ):
        self.cube_dir = Path(cube_dir)
        self.weighting = weighting
        self._loader = lru_cache(maxsize=max_cubes)(self._load_uncached)

    def _load_uncached(self, galaxy_id: str) -> Cube:
        path = self.cube_dir / f"{galaxy_id}_{self.weighting}_CUBE_THINGS.FITS"
        return load_cube(path)

    def __call__(self, galaxy_id: str) -> Cube:
        return self._loader(galaxy_id)

    def cache_clear(self) -> None:
        self._loader.cache_clear()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Knobs that tune :class:`ShellWindowDataset`."""

    window_pix: int = 96
    pos_factor: float = 2.0
    vel_factor: float = 2.0
    vel_floor_kms: float = 20.0
    augment: AugmentConfig | None = None
    rng_seed: int | None = None


class ShellWindowDataset:
    """``torch.utils.data.Dataset`` of ``(window, label, galaxy_id)`` tuples.

    Parameters
    ----------
    table
        Window table from :func:`hishells.windows.build_window_table`
        (or any DataFrame following the same schema). Must include
        columns ``galaxy_id``, ``ra_deg``, ``dec_deg``,
        ``vel_helio_kms``, ``pa_deg``, ``diameter_arcsec``,
        ``vexp_kms``, ``sigma_gas_kms``, ``label``.
    cubes
        :class:`CubeStore` providing on-demand cubes.
    sigma_rms_by_galaxy
        Optional precomputed per-galaxy ``sigma_rms``. If not supplied
        the dataset computes it on first access and caches per cube.
    config
        :class:`DatasetConfig`; defaults match plan \u00a72.1 / \u00a72.3.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        cubes: CubeStore,
        *,
        sigma_rms_by_galaxy: dict[str, float] | None = None,
        config: DatasetConfig | None = None,
    ):
        self.table = table.reset_index(drop=True)
        self.cubes = cubes
        self.config = config or DatasetConfig()
        self._sigma_rms = dict(sigma_rms_by_galaxy or {})
        self._channel_width: dict[str, float] = {}
        self._augmenter = (
            Augmenter(self.config.augment)
            if self.config.augment is not None
            else None
        )
        # Per-worker RNG -- seeded on first use so DataLoader workers
        # diverge if the user wraps with worker_init_fn that re-seeds.
        self._rng: np.random.Generator | None = None

    # -- torch.utils.data.Dataset API -----------------------------------------

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        import torch  # local import keeps catalog-only users light

        row = self.table.iloc[idx]
        gid = str(row["galaxy_id"])
        cube = self.cubes(gid)

        sigma = self._sigma_rms.get(gid)
        if sigma is None:
            sigma = sigma_rms(cube)
            self._sigma_rms[gid] = sigma
        chw = self._channel_width.get(gid)
        if chw is None:
            chw = channel_width_kms(cube)
            self._channel_width[gid] = chw

        win = extract_window_for_hole(
            cube,
            row.to_dict(),
            window_pix=self.config.window_pix,
            pos_factor=self.config.pos_factor,
            vel_factor=self.config.vel_factor,
            vel_floor_kms=self.config.vel_floor_kms,
        )
        win = normalize_window(win, sigma)

        if self._augmenter is not None:
            if self._rng is None:
                self._rng = np.random.default_rng(self.config.rng_seed)
            win = self._augmenter(win, self._rng, channel_width_kms=chw)

        x = torch.from_numpy(win).unsqueeze(0).float()  # (1, H, W)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y, gid


# ---------------------------------------------------------------------------
# LOGO splitter
# ---------------------------------------------------------------------------


@dataclass
class LOGOSplit:
    """One fold's train / val / test indices into the window table.

    ``test_galaxy`` is the held-out galaxy stem. ``val_idx`` is a
    held-out subset of the *training* positives' rows used for early
    stopping and operating-point selection per plan \u00a76.1.
    """

    test_galaxy: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class LOGOSplitter:
    """Leave-one-galaxy-out splitter over the 19 LOGO galaxies.

    The splitter yields one :class:`LOGOSplit` per galaxy in
    :data:`hishells.catalog.LOGO_GALAXIES_19` (or a user-supplied
    subset). The val split is ``val_frac`` of the training *positives*,
    sampled deterministically by ``rng_seed``.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        galaxies: tuple[str, ...] = LOGO_GALAXIES_19,
        val_frac: float = 0.1,
        rng_seed: int = 0,
    ):
        self.table = table.reset_index(drop=True)
        self.galaxies = tuple(galaxies)
        self.val_frac = val_frac
        self.rng_seed = rng_seed

    def __len__(self) -> int:
        return len(self.galaxies)

    def __iter__(self) -> Iterator[LOGOSplit]:
        rng = np.random.default_rng(self.rng_seed)
        all_galaxies = self.galaxies
        for held_out in all_galaxies:
            test_idx = np.flatnonzero(self.table["galaxy_id"].values == held_out)
            train_pool = np.flatnonzero(
                (self.table["galaxy_id"].values != held_out)
                & np.isin(self.table["galaxy_id"].values, all_galaxies)
            )
            # Pull val from training positives only.
            train_pos = train_pool[self.table["label"].values[train_pool] == 1]
            train_neg = train_pool[self.table["label"].values[train_pool] == 0]
            n_val = max(1, int(round(self.val_frac * len(train_pos))))
            val_pos = rng.choice(train_pos, size=n_val, replace=False)
            train_pos_remaining = np.setdiff1d(train_pos, val_pos)
            train_idx = np.concatenate([train_pos_remaining, train_neg])
            yield LOGOSplit(
                test_galaxy=held_out,
                train_idx=np.sort(train_idx),
                val_idx=np.sort(val_pos),
                test_idx=np.sort(test_idx),
            )


# ---------------------------------------------------------------------------
# Convenience: subset a dataset by index array
# ---------------------------------------------------------------------------


def make_subset(dataset: ShellWindowDataset, indices: np.ndarray) -> ShellWindowDataset:
    """Return a new :class:`ShellWindowDataset` over ``dataset.table.iloc[indices]``.

    Reuses the parent's :class:`CubeStore` *and* the same per-galaxy
    sigma / channel-width caches by reference so any sigma computed by
    the subset is visible to the parent (and vice versa).
    """

    sub = ShellWindowDataset(
        table=dataset.table.iloc[indices].reset_index(drop=True),
        cubes=dataset.cubes,
        config=dataset.config,
    )
    # Replace the freshly-allocated caches with the parent's so updates
    # propagate both ways.
    sub._sigma_rms = dataset._sigma_rms
    sub._channel_width = dataset._channel_width
    return sub
