"""Per-window augmentations.

All augmentations are *physically valid* for an HI p-v cut (per plan
\u00a72.3): adding noise, sub-channel velocity rolls, sub-pixel position
rolls, mirror flips, and small zoom. We deliberately exclude Mixup /
CutMix because pixel-level brightness blending has no physical
interpretation in K or Jy/beam.

API: each augmentation is a callable ``aug(window, rng) -> window``.
:class:`AugmentConfig` toggles each one and supplies its hyperparams;
:class:`Augmenter` composes them in a fixed order. The augmenter is
*stateless* with respect to its input (returns a new array) and uses
the supplied :class:`numpy.random.Generator`, so it composes cleanly
inside a ``Dataset.__getitem__``.

Inputs are normalised windows (output of
:func:`hishells.windows.normalize_window`), so the noise sigma is
specified in units of per-cube ``sigma_rms`` (i.e. the windows have
unit noise to begin with).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    noise_enabled: bool = True
    noise_sigma_low: float = 0.5
    noise_sigma_high: float = 2.0

    pos_roll_enabled: bool = True
    pos_roll_max_frac: float = 0.2  # +/- 0.2 * window_pix

    vel_roll_enabled: bool = True
    vel_roll_max_kms: float = 5.0

    flip_pos_enabled: bool = True
    flip_pos_p: float = 0.5

    flip_vel_enabled: bool = True
    flip_vel_p: float = 0.5

    zoom_enabled: bool = True
    zoom_low: float = 0.85
    zoom_high: float = 1.15

    beam_perturb_enabled: bool = False  # off by default per \u00a72.3
    beam_perturb_low: float = 0.9
    beam_perturb_high: float = 1.1
    beam_sigma_pix: float = 1.5  # default beam in pixels


def no_augment() -> "AugmentConfig":
    """Identity-augmentation config (used for val / test)."""

    return AugmentConfig(
        noise_enabled=False,
        pos_roll_enabled=False,
        vel_roll_enabled=False,
        flip_pos_enabled=False,
        flip_vel_enabled=False,
        zoom_enabled=False,
        beam_perturb_enabled=False,
    )


# ---------------------------------------------------------------------------
# Individual augmentation primitives
# ---------------------------------------------------------------------------


def add_noise(window: np.ndarray, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    sigma = float(rng.uniform(low, high))
    return window + rng.normal(0.0, sigma, size=window.shape).astype(window.dtype)


def roll_axis(window: np.ndarray, rng: np.random.Generator, axis: int, max_shift: int) -> np.ndarray:
    if max_shift <= 0:
        return window
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(window, shift, axis=axis)


def maybe_flip(window: np.ndarray, rng: np.random.Generator, axis: int, p: float) -> np.ndarray:
    if rng.random() < p:
        return np.flip(window, axis=axis).copy()
    return window


def random_zoom(window: np.ndarray, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    factor = float(rng.uniform(low, high))
    if abs(factor - 1.0) < 1e-3:
        return window
    h, w = window.shape
    zoomed = zoom(window, factor, order=1, mode="reflect")
    zh, zw = zoomed.shape
    if factor > 1.0:
        # Center-crop to (h, w)
        oy = (zh - h) // 2
        ox = (zw - w) // 2
        return np.ascontiguousarray(zoomed[oy : oy + h, ox : ox + w])
    # Center-pad with zeros to (h, w)
    out = np.zeros_like(window)
    oy = (h - zh) // 2
    ox = (w - zw) // 2
    out[oy : oy + zh, ox : ox + zw] = zoomed
    return out


def beam_perturb(
    window: np.ndarray, rng: np.random.Generator, low: float, high: float, beam_sigma_pix: float
) -> np.ndarray:
    factor = float(rng.uniform(low, high))
    sigma = max(1e-3, factor * beam_sigma_pix)
    # Smooth only along the position axis; the velocity axis is not
    # convolved with the synthesized beam in a real cube.
    return gaussian_filter(window, sigma=(sigma, 0.0), mode="reflect")


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class Augmenter:
    """Compose a fixed sequence of per-window augmentations.

    Use as ``Augmenter(cfg)(window, rng)``. Augmentations are applied
    in this order: zoom, flip-pos, flip-vel, position-roll, vel-roll,
    beam-perturb, noise.
    """

    def __init__(self, config: AugmentConfig):
        self.cfg = config

    def __call__(
        self,
        window: np.ndarray,
        rng: np.random.Generator,
        *,
        channel_width_kms: float | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        x = window
        if cfg.zoom_enabled:
            x = random_zoom(x, rng, cfg.zoom_low, cfg.zoom_high)
        if cfg.flip_pos_enabled:
            x = maybe_flip(x, rng, axis=0, p=cfg.flip_pos_p)
        if cfg.flip_vel_enabled:
            x = maybe_flip(x, rng, axis=1, p=cfg.flip_vel_p)
        if cfg.pos_roll_enabled:
            max_shift = int(round(cfg.pos_roll_max_frac * x.shape[0]))
            x = roll_axis(x, rng, axis=0, max_shift=max_shift)
        if cfg.vel_roll_enabled:
            if channel_width_kms is None or channel_width_kms <= 0:
                max_shift = 1
            else:
                max_shift = max(1, int(round(cfg.vel_roll_max_kms / channel_width_kms)))
            x = roll_axis(x, rng, axis=1, max_shift=max_shift)
        if cfg.beam_perturb_enabled:
            x = beam_perturb(
                x, rng, cfg.beam_perturb_low, cfg.beam_perturb_high, cfg.beam_sigma_pix
            )
        if cfg.noise_enabled:
            x = add_noise(x, rng, cfg.noise_sigma_low, cfg.noise_sigma_high)
        return x.astype(np.float32, copy=False)
