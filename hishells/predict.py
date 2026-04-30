"""MC-dropout inference + temperature scaling.

Per ``plan.md`` \u00a77 / \u00a78. The two public entry points are:

* :func:`predict_with_uncertainty` -- run ``T`` stochastic forward
  passes with dropout active (BN frozen, see
  :func:`hishells.model.mc_dropout_eval`) and return per-sample mean,
  std, and the 5/95 score quantiles. The arrays returned match the
  output FITS table schema in plan \u00a77.
* :func:`fit_temperature` / :func:`apply_temperature` -- post-hoc
  calibration via temperature scaling (Guo et al. 2017). Fit the
  scalar ``T`` on within-fold validation logits with LBFGS and divide
  test logits by it before sigmoid.

Catalog-driven inference (score every B11 row through the trained
model) lives in :func:`predict_catalog`; cube-driven inference uses
:mod:`hishells.candidates` to enumerate sightlines first and is
wrapped by ``scripts/predict_galaxy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import mc_dropout_eval
from .train import _window_collate, select_device


# ---------------------------------------------------------------------------
# Per-batch / per-dataset MC dropout
# ---------------------------------------------------------------------------


@dataclass
class MCResult:
    """Per-sample MC-dropout summary stats.

    Each field is a 1-D array of length ``N`` (samples).
    """

    score_mean: np.ndarray
    score_std: np.ndarray
    score_q05: np.ndarray
    score_q95: np.ndarray
    n_passes: int


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    *,
    T: int = 50,
    device: torch.device | None = None,
) -> MCResult:
    """Run ``T`` MC-dropout forward passes on a single batch.

    ``x`` is shaped ``(B, 1, H, W)``; the model is *not* required to
    be in any particular mode at call time (we always switch to
    :func:`hishells.model.mc_dropout_eval`).
    """

    device = device or select_device(None)
    model = model.to(device)
    mc_dropout_eval(model)
    x = x.to(device)
    samples = []
    for _ in range(T):
        samples.append(torch.sigmoid(model(x)).flatten().cpu().numpy())
    arr = np.stack(samples, axis=0)  # (T, B)
    return MCResult(
        score_mean=arr.mean(axis=0),
        score_std=arr.std(axis=0),
        score_q05=np.quantile(arr, 0.05, axis=0),
        score_q95=np.quantile(arr, 0.95, axis=0),
        n_passes=T,
    )


@torch.no_grad()
def predict_dataset_mc(
    model: nn.Module,
    dataset: Dataset,
    *,
    T: int = 50,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[MCResult, np.ndarray, list[str]]:
    """Run MC-dropout inference over an entire ``Dataset``.

    Returns ``(MCResult, labels, galaxy_ids)`` aligned by sample.
    """

    device = device or select_device(None)
    model = model.to(device)
    mc_dropout_eval(model)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_window_collate,
    )
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    q05s: list[np.ndarray] = []
    q95s: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    galaxy_ids: list[str] = []
    for xs, ys, gids in loader:
        xs = xs.to(device, non_blocking=True)
        # Re-enter MC mode each batch in case a wrapping
        # ``model.train()`` somewhere else flipped it.
        mc_dropout_eval(model)
        samples = []
        for _ in range(T):
            samples.append(torch.sigmoid(model(xs)).flatten().cpu().numpy())
        arr = np.stack(samples, axis=0)
        means.append(arr.mean(axis=0))
        stds.append(arr.std(axis=0))
        q05s.append(np.quantile(arr, 0.05, axis=0))
        q95s.append(np.quantile(arr, 0.95, axis=0))
        labels.append(ys.flatten().numpy())
        galaxy_ids.extend(gids)
    if not means:
        empty = np.empty(0)
        return MCResult(empty, empty, empty, empty, T), empty, []
    res = MCResult(
        score_mean=np.concatenate(means),
        score_std=np.concatenate(stds),
        score_q05=np.concatenate(q05s),
        score_q95=np.concatenate(q95s),
        n_passes=T,
    )
    return res, np.concatenate(labels), galaxy_ids


# ---------------------------------------------------------------------------
# Catalog-driven prediction
# ---------------------------------------------------------------------------


def predict_catalog(
    model: nn.Module,
    rows,  # pd.DataFrame with the same columns as the window table
    cube_store,
    *,
    T: int = 50,
    sigma_rms_by_galaxy: dict[str, float] | None = None,
    window_pix: int = 96,
    batch_size: int = 64,
    device: torch.device | None = None,
):
    """Score every row in ``rows`` with MC dropout.

    Convenience wrapper that builds a :class:`hishells.data.ShellWindowDataset`
    on the fly so callers don't have to. Returns the same tuple as
    :func:`predict_dataset_mc`.
    """

    from .data import DatasetConfig, ShellWindowDataset

    ds = ShellWindowDataset(
        table=rows,
        cubes=cube_store,
        sigma_rms_by_galaxy=sigma_rms_by_galaxy,
        config=DatasetConfig(window_pix=window_pix),
    )
    return predict_dataset_mc(
        model, ds, T=T, batch_size=batch_size, device=device
    )


# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------


def fit_temperature(
    val_logits: np.ndarray | torch.Tensor,
    val_labels: np.ndarray | torch.Tensor,
    *,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Fit a single scalar ``T`` minimising NLL on ``(val_logits, val_labels)``.

    Returns the fitted temperature (always > 0). Use
    :func:`apply_temperature` to apply it to test logits before
    sigmoid.
    """

    logits = torch.as_tensor(val_logits, dtype=torch.float32).flatten()
    labels = torch.as_tensor(val_labels, dtype=torch.float32).flatten()
    if logits.numel() == 0 or labels.unique().numel() < 2:
        return 1.0
    log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T) > 0
    optim = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optim.zero_grad()
        T = torch.exp(log_T)
        loss = nn.functional.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(torch.exp(log_T).detach().item())


def apply_temperature(
    logits: np.ndarray | torch.Tensor, T: float
) -> np.ndarray:
    """Return ``sigmoid(logits / T)`` as a numpy array."""

    arr = torch.as_tensor(logits, dtype=torch.float32) / float(T)
    return torch.sigmoid(arr).cpu().numpy()


# ---------------------------------------------------------------------------
# FITS-table writer (output schema in plan \u00a77)
# ---------------------------------------------------------------------------


def write_candidates_fits(
    path: str | Path,
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    vel_kms: np.ndarray,
    mc: MCResult,
    galaxy_id: str,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> None:
    """Write the per-galaxy candidate FITS table per plan \u00a77.

    Columns: ``ra``, ``dec``, ``vel``, ``score_mean``, ``score_std``,
    ``score_q05``, ``score_q95``, ``n_passes``, ``galaxy``.
    """

    from astropy.io import fits

    n = len(ra_deg)
    cols = [
        fits.Column(name="ra", format="D", unit="deg", array=ra_deg),
        fits.Column(name="dec", format="D", unit="deg", array=dec_deg),
        fits.Column(name="vel", format="D", unit="km/s", array=vel_kms),
        fits.Column(name="score_mean", format="E", array=mc.score_mean),
        fits.Column(name="score_std", format="E", array=mc.score_std),
        fits.Column(name="score_q05", format="E", array=mc.score_q05),
        fits.Column(name="score_q95", format="E", array=mc.score_q95),
        fits.Column(name="n_passes", format="J", array=np.full(n, mc.n_passes, dtype=np.int32)),
        fits.Column(name="galaxy", format="20A", array=np.array([galaxy_id] * n)),
    ]
    if extra_cols:
        for name, arr in extra_cols.items():
            arr_np = np.asarray(arr)
            if arr_np.dtype.kind in {"S", "U"}:
                # Fixed-length string column. ``itemsize`` for a unicode
                # array is bytes-per-char; convert to byte strings first
                # so the FITS column width matches the actual encoding.
                if arr_np.dtype.kind == "U":
                    arr_np = np.array([str(s).encode() for s in arr_np])
                width = max(int(arr_np.dtype.itemsize), 1)
                fmt = f"{width}A"
            elif np.issubdtype(arr_np.dtype, np.floating):
                fmt = "D"
            else:
                fmt = "J"
            cols.append(fits.Column(name=name, format=fmt, array=arr_np))
    hdu = fits.BinTableHDU.from_columns(cols, name="CANDIDATES")
    hdu.header["GALAXY"] = galaxy_id
    hdu.header["NPASSES"] = mc.n_passes
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(p, overwrite=True)
