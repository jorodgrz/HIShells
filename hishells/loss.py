"""Loss functions for shell-vs-not binary classification.

Per ``plan.md`` \u00a74:

* :class:`WeightedBCEWithLogits` -- ``BCEWithLogitsLoss`` with
  ``pos_weight = N_neg / N_pos`` and optional symmetric label
  smoothing (target ``\u2208 {label_smoothing, 1 - label_smoothing}``).
* :class:`FocalLoss` -- Lin et al. 2017, with ``alpha = 0.25`` and
  ``gamma = 2.0``. Plan \u00a74 explicitly says do *not* combine focal +
  label smoothing.

Both losses return a scalar and accept ``(logits, targets)`` of shape
``(B, 1)`` or ``(B,)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pos_weight_from_counts(n_pos: int, n_neg: int) -> torch.Tensor:
    """Return ``pos_weight = N_neg / N_pos`` as a 1-element tensor.

    Falls back to ``1.0`` if ``n_pos`` is zero so the loss is still
    well-defined for an empty batch.
    """

    if n_pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(float(n_neg) / float(n_pos))


def smooth_targets(targets: torch.Tensor, eps: float) -> torch.Tensor:
    """``y \u2192 eps + (1 - 2*eps) * y`` (symmetric label smoothing)."""

    if eps <= 0:
        return targets
    return eps + (1.0 - 2.0 * eps) * targets


# ---------------------------------------------------------------------------
# Weighted BCE + label smoothing
# ---------------------------------------------------------------------------


class WeightedBCEWithLogits(nn.Module):
    """Class-imbalance-aware BCE with optional label smoothing.

    Parameters
    ----------
    pos_weight
        Scalar tensor for ``BCEWithLogitsLoss(pos_weight=...)``. Use
        :func:`pos_weight_from_counts` to derive from training-fold
        counts.
    label_smoothing
        Symmetric epsilon. ``0.05`` -> targets become ``\u2208 {0.05,
        0.95}``. Plan default is ``0.05``.
    reduction
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | float | None = None,
        *,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight))
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None
        )
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            targets = targets.view_as(logits)
        targets = smooth_targets(targets.float(), self.label_smoothing)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
            reduction=self.reduction,
        )


# ---------------------------------------------------------------------------
# Focal loss (Lin et al. 2017)
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017).

    ``L(p) = -alpha_t * (1 - p_t)**gamma * log(p_t)`` where
    ``p_t = p`` for ``y=1`` and ``1-p`` for ``y=0``, and
    ``alpha_t = alpha`` for ``y=1`` and ``1-alpha`` for ``y=0``.
    Defaults match plan \u00a74: ``alpha=0.25, gamma=2.0``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            targets = targets.view_as(logits)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = p for y=1 else 1-p; using the BCE-derived form:
        # exp(-bce) = p_t, so (1 - p_t)**gamma = (1 - exp(-bce))**gamma
        p_t = torch.exp(-bce)
        focal_factor = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_loss(
    name: str,
    *,
    n_pos: int | None = None,
    n_neg: int | None = None,
    label_smoothing: float = 0.05,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Instantiate a loss by name. Used by ``scripts/train_logo.py``.

    Supported names: ``"bce"`` (weighted BCE + label smoothing),
    ``"focal"`` (focal loss).
    """

    name = name.lower()
    if name in ("bce", "weighted_bce", "wbce"):
        pw = pos_weight_from_counts(n_pos or 1, n_neg or 1)
        return WeightedBCEWithLogits(pos_weight=pw, label_smoothing=label_smoothing)
    if name in ("focal", "focal_loss"):
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    raise ValueError(f"unknown loss name: {name!r}")
