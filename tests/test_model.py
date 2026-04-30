"""Tests for ``hishells.model`` and ``hishells.loss``."""

from __future__ import annotations

import importlib.util

import pytest
import torch

torchvision_available = importlib.util.find_spec("torchvision") is not None
needs_torchvision = pytest.mark.skipif(
    not torchvision_available, reason="torchvision not installed in this env"
)

from hishells.loss import (
    FocalLoss,
    WeightedBCEWithLogits,
    build_loss,
    pos_weight_from_counts,
    smooth_targets,
)
from hishells.model import (
    ResNetShellHead,
    SmallShellCNN,
    build_model,
    count_parameters,
    mc_dropout_eval,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_small_shell_cnn_forward_shape():
    model = SmallShellCNN()
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    assert y.shape == (4, 1)


def test_small_shell_cnn_param_count_in_range():
    model = SmallShellCNN()
    n = count_parameters(model)
    # Plan \u00a73.1 says ~250k; we count 241k. Allow a generous range
    # so legitimate hyperparam tweaks don't blow up the test.
    assert 100_000 < n < 600_000


@needs_torchvision
def test_resnet_shell_head_forward_shape():
    model = ResNetShellHead(pretrained=False)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1)


@needs_torchvision
def test_resnet_freeze_unfreeze():
    model = ResNetShellHead(pretrained=False).freeze_backbone()
    n_trainable_frozen = count_parameters(model, trainable_only=True)
    model.unfreeze_all()
    n_trainable_unfrozen = count_parameters(model, trainable_only=True)
    assert n_trainable_unfrozen > n_trainable_frozen


def test_build_model_dispatch_small():
    assert isinstance(build_model("small"), SmallShellCNN)


@needs_torchvision
def test_build_model_dispatch_resnet():
    assert isinstance(build_model("resnet18", pretrained=False), ResNetShellHead)


# ---------------------------------------------------------------------------
# MC dropout
# ---------------------------------------------------------------------------


def test_mc_dropout_eval_keeps_dropout_active():
    model = SmallShellCNN()
    mc_dropout_eval(model)
    # All Dropout / Dropout2d layers should be in train mode.
    drop_modes = [
        m.training
        for m in model.modules()
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))
    ]
    assert all(drop_modes), drop_modes
    # All BN layers should be in eval mode.
    bn_modes = [
        m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    ]
    assert not any(bn_modes), bn_modes


def test_mc_dropout_produces_variance():
    torch.manual_seed(0)
    model = SmallShellCNN()
    mc_dropout_eval(model)
    x = torch.randn(8, 1, 64, 64)
    with torch.no_grad():
        passes = torch.stack([torch.sigmoid(model(x)) for _ in range(10)])
    # Variance across the T axis must be strictly positive somewhere.
    assert passes.std(dim=0).max().item() > 1e-4


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_pos_weight_basic():
    pw = pos_weight_from_counts(n_pos=10, n_neg=50)
    assert pw.item() == 5.0


def test_smooth_targets_zero_eps_identity():
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    out = smooth_targets(y, eps=0.0)
    assert torch.equal(out, y)


def test_smooth_targets_clips_endpoints():
    y = torch.tensor([0.0, 1.0])
    out = smooth_targets(y, eps=0.05)
    assert torch.allclose(out, torch.tensor([0.05, 0.95]))


def test_weighted_bce_decreases_with_correct_logits():
    loss = WeightedBCEWithLogits(pos_weight=torch.tensor(2.0), label_smoothing=0.0)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    bad = torch.tensor([-3.0, -3.0, 3.0, 3.0])
    good = torch.tensor([3.0, 3.0, -3.0, -3.0])
    assert loss(good, targets) < loss(bad, targets)


def test_focal_loss_sane_at_easy_examples():
    loss = FocalLoss()
    targets = torch.tensor([1.0, 0.0])
    easy = torch.tensor([5.0, -5.0])
    hard = torch.tensor([0.5, -0.5])
    # Focal loss strongly down-weights easy examples relative to hard.
    assert loss(easy, targets) < 0.1 * loss(hard, targets)


def test_build_loss_dispatch():
    assert isinstance(build_loss("bce", n_pos=1, n_neg=5), WeightedBCEWithLogits)
    assert isinstance(build_loss("focal"), FocalLoss)
