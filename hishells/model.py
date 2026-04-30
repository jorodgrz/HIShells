"""Classifier architectures.

Two models per ``plan.md`` \u00a73:

* :class:`SmallShellCNN` -- 4-block conv stack with BN, Dropout2d,
  GAP head, Dropout, Linear -- the v1 default (~241k params).
* :class:`ResNetShellHead` -- ResNet-18 transfer-learning head with a
  1-channel ``conv1`` and a ``Dropout(0.4) + Linear(512, 1)`` head.

Both expose :func:`mc_dropout_eval` which is the BN-safe variant of
the snippet in plan \u00a77: dropout layers stay in *train* mode (so they
keep sampling masks) while every other module -- including
:class:`torch.nn.BatchNorm2d` and the un-dropout convs -- stays in
*eval* mode (so running batch statistics are used). The plan's
``model.train()`` snippet inadvertently flips BN back into training
mode and corrupts its running statistics; this fix avoids that.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MC-dropout helper
# ---------------------------------------------------------------------------


_DROPOUT_TYPES: tuple[type, ...] = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def mc_dropout_eval(model: nn.Module) -> nn.Module:
    """Switch ``model`` to a state suitable for MC-dropout inference.

    Sets the whole model to ``eval`` first (freezes BN running stats,
    disables training-mode behaviour everywhere), then flips only the
    Dropout layers back to ``train`` so they keep sampling new masks.
    Returns the same model for chaining.
    """

    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
    return model


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# SmallShellCNN -- v1 default
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        return x


class SmallShellCNN(nn.Module):
    """4-block CNN per plan \u00a73.1.

    Input  : ``(B, 1, H, W)`` -- the v1 default is ``96\u00d796`` per the
             ``window_pix`` retirement in plan \u00a72.1, but the head uses
             :class:`nn.AdaptiveAvgPool2d`, so any spatial size that
             survives 4 max-pools (i.e. \u2265 16 and divisible enough to
             leave a non-empty feature map) works without code changes.
    Output : ``(B, 1)`` raw logit (apply ``torch.sigmoid`` for probability).

    The dropout schedule (0.1, 0.1, 0.15, 0.15, 0.4) is load-bearing
    for the MC-dropout uncertainty estimates in
    :mod:`hishells.predict`.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        block_channels: tuple[int, int, int, int] = (32, 64, 128, 128),
        block_dropout: tuple[float, float, float, float] = (0.1, 0.1, 0.15, 0.15),
        head_dropout: float = 0.4,
    ):
        super().__init__()
        c1, c2, c3, c4 = block_channels
        d1, d2, d3, d4 = block_dropout
        self.block1 = _ConvBlock(in_channels, c1, d1)
        self.block2 = _ConvBlock(c1, c2, d2)
        self.block3 = _ConvBlock(c2, c3, d3)
        self.block4 = _ConvBlock(c3, c4, d4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)  # (B, C)
        x = self.head_drop(x)
        x = self.head(x)  # (B, 1) raw logit
        return x


# ---------------------------------------------------------------------------
# ResNetShellHead -- transfer-learning fallback
# ---------------------------------------------------------------------------


class ResNetShellHead(nn.Module):
    """ResNet-18 with a 1-channel conv1 and a Dropout+Linear head.

    Per plan \u00a73.2: stage-1 freezes everything but ``conv1`` and
    ``head``; stage-2 unfreezes the rest. We expose the freeze
    convenience as :meth:`freeze_backbone` / :meth:`unfreeze_all`.
    Pretrained weights are loaded if available (best-effort; falls
    back to random init in offline / sandboxed contexts).
    """

    def __init__(
        self,
        *,
        head_dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision import models  # local import to keep CLI light

        try:
            weights = (
                models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone = models.resnet18(weights=weights)
        except Exception:  # offline / network-blocked
            backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # -- freeze helpers per \u00a73.2 ------------------------------------------

    def _trainable_modules(self) -> Iterable[nn.Module]:
        yield self.backbone.conv1
        yield self.backbone.fc

    def freeze_backbone(self) -> "ResNetShellHead":
        """Stage-1: freeze everything except ``conv1`` and ``fc``."""

        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self._trainable_modules():
            for p in m.parameters():
                p.requires_grad = True
        return self

    def unfreeze_all(self) -> "ResNetShellHead":
        """Stage-2: unfreeze the entire backbone."""

        for p in self.backbone.parameters():
            p.requires_grad = True
        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name. Used by ``scripts/train_logo.py``.

    Supported names: ``"small"`` (default), ``"resnet18"``.
    """

    name = name.lower()
    if name in ("small", "smallshellcnn", "cnn"):
        return SmallShellCNN(**kwargs)
    if name in ("resnet", "resnet18", "rn18"):
        return ResNetShellHead(**kwargs)
    raise ValueError(f"unknown model name: {name!r}")
