"""Tests for ``hishells.augment``."""

from __future__ import annotations

import numpy as np
import pytest

from hishells.augment import (
    AugmentConfig,
    Augmenter,
    add_noise,
    beam_perturb,
    maybe_flip,
    no_augment,
    random_zoom,
    roll_axis,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def window():
    rng = np.random.default_rng(7)
    return rng.standard_normal((64, 64)).astype(np.float32)


def test_add_noise_increases_variance(window, rng):
    out = add_noise(window, rng, low=1.0, high=1.0)
    assert out.shape == window.shape
    assert out.dtype == window.dtype
    # Adding unit Gaussian noise to a unit-variance window should
    # roughly double the variance.
    assert out.var() > 1.5 * window.var()


def test_roll_axis_shape(window, rng):
    out = roll_axis(window, rng, axis=0, max_shift=4)
    assert out.shape == window.shape


def test_maybe_flip_deterministic(window):
    rng = np.random.default_rng(0)
    out = maybe_flip(window, rng, axis=0, p=1.0)
    assert np.allclose(out, window[::-1])
    out2 = maybe_flip(window, rng, axis=0, p=0.0)
    assert np.allclose(out2, window)


def test_random_zoom_shape(window, rng):
    out = random_zoom(window, rng, low=0.8, high=1.2)
    assert out.shape == window.shape


def test_beam_perturb_smooths(window, rng):
    out = beam_perturb(window, rng, low=1.0, high=1.0, beam_sigma_pix=2.0)
    # Smoothing should reduce the per-pixel std along the position axis.
    assert out.std(axis=0).mean() < window.std(axis=0).mean()


def test_no_augment_is_identity(window):
    rng = np.random.default_rng(0)
    aug = Augmenter(no_augment())
    out = aug(window, rng, channel_width_kms=5.0)
    assert np.allclose(out, window)


def test_full_augmenter_preserves_shape(window):
    rng = np.random.default_rng(0)
    aug = Augmenter(AugmentConfig())
    out = aug(window, rng, channel_width_kms=5.0)
    assert out.shape == window.shape
    assert out.dtype == np.float32
