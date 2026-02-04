from __future__ import annotations

import numpy as np

from .geometry import relative_pitch


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel."""
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


def smooth_line_field(
    target_r: np.ndarray,
    sample_r: np.ndarray,
    sample_theta: np.ndarray,
    sample_phi_spatial: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Kernel regression for line fields using the double-angle trick.

    Returns the smoothed relative pitch angle psi in radians.
    """
    psi_obs = relative_pitch(sample_theta, sample_phi_spatial)
    cos_2psi = np.cos(2 * psi_obs)
    sin_2psi = np.sin(2 * psi_obs)

    u = (target_r[:, None] - sample_r[None, :]) / bandwidth
    weights = gaussian_kernel(u)
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0

    avg_cos = np.sum(weights * cos_2psi[None, :], axis=1) / sum_weights
    avg_sin = np.sum(weights * sin_2psi[None, :], axis=1) / sum_weights

    return 0.5 * np.arctan2(avg_sin, avg_cos)


def smooth_spiral_pitch(
    target_r: np.ndarray,
    sample_r: np.ndarray,
    sample_alpha_rad: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Kernel smoothing of spiral pitch using the double-angle embedding."""
    cos_2alpha = np.cos(2 * sample_alpha_rad)
    sin_2alpha = np.sin(2 * sample_alpha_rad)

    u = (target_r[:, None] - sample_r[None, :]) / bandwidth
    weights = gaussian_kernel(u)
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0

    avg_cos = np.sum(weights * cos_2alpha[None, :], axis=1) / sum_weights
    avg_sin = np.sum(weights * sin_2alpha[None, :], axis=1) / sum_weights

    return 0.5 * np.arctan2(avg_sin, avg_cos)
