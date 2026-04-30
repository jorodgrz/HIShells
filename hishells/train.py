"""Single-fold training loop.

Per ``plan.md`` \u00a75:

* Optimizer: ``AdamW`` with default ``lr=1e-3`` (custom CNN) /
  ``1e-4`` (transfer-learning stage 2), weight decay ``1e-4``.
* Schedule: :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.
* Mixed precision when CUDA is available (``torch.cuda.amp``); MPS
  and CPU run in fp32.
* Gradient clipping at norm 1.0.
* Early stopping with ``patience=10`` on within-fold validation
  PR-AUC.

The loop returns ``(model, history, val_scores, val_labels)``;
``history`` is a list of per-epoch dicts so the training-diagnostics
notebook can plot it. Negatives are typically resampled per epoch by
the caller (``scripts/train_logo.py``); we don't bake that in here.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import average_precision_score
except Exception:  # pragma: no cover
    average_precision_score = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    num_workers: int = 0
    pin_memory: bool = False
    device: str | None = None
    mixed_precision: bool = True
    log_every: int = 0  # 0 = silent, N = print every N epochs
    seed: int = 0


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_pr_auc: float
    lr: float


@dataclass
class FoldResult:
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_pr_auc: float = float("-inf")
    val_scores: np.ndarray | None = None
    val_labels: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# One-fold training loop
# ---------------------------------------------------------------------------


def _autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    cfg: TrainConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=_window_collate,
    )


def _window_collate(batch):
    """Strip the ``galaxy_id`` string out of the default collate."""

    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    gids = [b[2] for b in batch]
    return xs, ys, gids


@torch.no_grad()
def _eval_pr_auc(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for xs, ys, _ in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True).view(-1, 1).float()
        logits = model(xs)
        loss = loss_fn(logits, ys)
        losses.append(float(loss.detach().cpu()))
        all_scores.append(torch.sigmoid(logits).flatten().cpu().numpy())
        all_labels.append(ys.flatten().cpu().numpy())
    if not all_scores:
        return 0.0, 0.0, np.empty(0), np.empty(0)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if average_precision_score is None or labels.sum() == 0 or labels.sum() == labels.size:
        pr_auc = 0.0
    else:
        pr_auc = float(average_precision_score(labels.astype(int), scores))
    return float(np.mean(losses)), float(pr_auc), scores, labels


def train_one_fold(
    model: nn.Module,
    train_set: Dataset,
    val_set: Dataset,
    loss_fn: nn.Module,
    *,
    config: TrainConfig | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
) -> tuple[nn.Module, FoldResult]:
    """Train ``model`` for one LOGO fold; return best-checkpoint state.

    The function restores the model to its best-PR-AUC weights before
    returning so that downstream code (predict / eval) doesn't need to
    re-load a checkpoint just to use the trained weights.
    """

    cfg = config or TrainConfig()
    device = select_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    train_loader = _make_loader(train_set, batch_size=cfg.batch_size, shuffle=True, cfg=cfg)
    val_loader = _make_loader(val_set, batch_size=cfg.batch_size, shuffle=False, cfg=cfg)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    result = FoldResult()
    best_state: dict[str, Any] | None = None
    epochs_since_best = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xs, ys, _ in train_loader:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).view(-1, 1).float()
            optim.zero_grad(set_to_none=True)
            with _autocast_ctx(device, use_amp):
                logits = model(xs)
                loss = loss_fn(logits, ys)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        val_loss, val_pr_auc, val_scores, val_labels = _eval_pr_auc(
            model, val_loader, loss_fn, device
        )
        scheduler.step()

        rec = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_pr_auc=val_pr_auc,
            lr=float(scheduler.get_last_lr()[0]),
        )
        result.history.append(rec)
        if on_epoch_end is not None:
            on_epoch_end(rec)
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"epoch {epoch:3d}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_PR_AUC={val_pr_auc:.4f} lr={rec.lr:.2e}"
            )

        if val_pr_auc > result.best_val_pr_auc:
            result.best_val_pr_auc = val_pr_auc
            result.best_epoch = epoch
            result.val_scores = val_scores
            result.val_labels = val_labels
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, result


# ---------------------------------------------------------------------------
# Test-set scoring (deterministic, no MC dropout)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(scores, labels, galaxy_ids)`` over ``dataset``.

    Deterministic: no MC dropout. For uncertainty estimates use
    :func:`hishells.predict.predict_with_uncertainty` instead.
    """

    device = device or select_device(None)
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_window_collate
    )
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    for xs, ys, gids in loader:
        xs = xs.to(device, non_blocking=True)
        logits = model(xs)
        all_scores.append(torch.sigmoid(logits).flatten().cpu().numpy())
        all_labels.append(ys.flatten().numpy())
        all_ids.extend(gids)
    if not all_scores:
        return np.empty(0), np.empty(0), []
    return np.concatenate(all_scores), np.concatenate(all_labels), all_ids


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: TrainConfig,
    fold: str,
    seed: int,
    extras: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg_to_dict(config),
            "fold": fold,
            "seed": seed,
            "extras": extras or {},
        },
        p,
    )


def cfg_to_dict(cfg: TrainConfig) -> dict:
    return {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}


def load_checkpoint(path: str | Path, model: nn.Module) -> dict:
    p = Path(path)
    obj = torch.load(p, map_location="cpu")
    model.load_state_dict(obj["state_dict"])
    return obj
