"""Trivial baseline: window-integrated flux deficit (plan \u00a79 Row 3).

For every (positive + negative) window in the table, score it by

::

    deficit = -mean(window_normalised)

i.e. an integrated flux deficit relative to the per-cube background.
At the operating point this is just "if the window is darker than the
median by more than ``tau`` sigma, predict shell". The point of this
row is to set the floor that any downstream classifier has to clear.

The function returns ``(scores, labels)`` arrays compatible with
:func:`hishells.eval.compute_metrics`; the operating-point selection
is done downstream so this module stays threshold-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..cubes import sigma_rms
from ..data import CubeStore
from ..pvcut import extract_window_for_hole
from ..windows import normalize_window


def score_table(
    table: pd.DataFrame,
    cubes: CubeStore,
    *,
    sigma_rms_by_galaxy: dict[str, float] | None = None,
    window_pix: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, labels)`` over every row in ``table``.

    ``scores`` is the negative mean of the per-cube-normalised window
    (so larger = more shell-like). ``labels`` is ``table["label"]`` as
    a 1-D array.
    """

    sig = dict(sigma_rms_by_galaxy or {})
    scores = np.empty(len(table), dtype=np.float64)
    labels = table["label"].values.astype(np.int64)

    for i, (_, row) in enumerate(table.iterrows()):
        gid = str(row["galaxy_id"])
        cube = cubes(gid)
        if gid not in sig:
            sig[gid] = sigma_rms(cube)
        win = extract_window_for_hole(cube, row.to_dict(), window_pix=window_pix)
        win = normalize_window(win, sig[gid])
        scores[i] = -float(np.nanmean(win))

    return scores, labels
