from __future__ import annotations

import numpy as np
from typing import Literal  # <- add (optional but good)

from .geometry import relative_pitch


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel."""
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform kernel."""
    return 0.5 * (np.abs(u) <= 1.0).astype(float)

def gaussian_rbf(dr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel on raw radial distances."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return np.exp(-(dr**2) / (2.0 * sigma**2))

def wrap_angle(dtheta: np.ndarray) -> np.ndarray:
    """Wrap angular differences to (-pi, pi]."""
    return np.angle(np.exp(1j * dtheta))

def von_mises_kernel(dtheta: np.ndarray, kappa: float, normalize: bool = False) -> np.ndarray:
    """Von Mises kernel exp(kappa*cos(dtheta)) on angular differences."""
    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    dtheta = wrap_angle(dtheta)
    w = np.exp(kappa * np.cos(dtheta))
    if normalize:
        # Normalizing constant for von Mises on circle: 2π I0(kappa)
        w = w / (2.0 * np.pi * np.i0(kappa))
    return w

def multiplicative_radial_angular_kernel(
    target_r: np.ndarray,
    sample_r: np.ndarray,
    target_theta: np.ndarray,
    sample_theta: np.ndarray,
    sigma: float,
    kappa: float,
    normalize_angular: bool = False,
) -> np.ndarray:
    """Product kernel: K = K_radial * K_angular, returns (n_target, n_sample) weights."""
    dr = target_r[:, None] - sample_r[None, :]
    dtheta = target_theta[:, None] - sample_theta[None, :]
    return gaussian_rbf(dr, sigma) * von_mises_kernel(dtheta, kappa, normalize=normalize_angular)


def smooth_line_field(
    target_r: np.ndarray,
    sample_r: np.ndarray,
    sample_theta: np.ndarray,
    sample_phi_spatial: np.ndarray,
    bandwidth: float,
    *,
    kernel: Literal["gaussian", "uniform", "multiplicative"] = "gaussian",
    target_theta: np.ndarray | None = None,
    angular_kappa: float | None = None,
    normalize_angular: bool = False,
) -> np.ndarray:
    """Kernel regression for line fields using the double-angle trick.

    Args:
        sample_theta: observed line direction angles (e.g. data.phi_rad)
        sample_phi_spatial: spatial angles at sample points (e.g. data.theta)
        kernel:
          - "gaussian": Gaussian kernel on normalized radial distance
          - "uniform": Uniform (box) kernel on normalized radial distance
          - "multiplicative": Gaussian radial × von Mises angular (on *spatial angle*)
        target_theta: required for kernel="multiplicative" (target spatial angles)
        angular_kappa: required for kernel="multiplicative" (von Mises concentration)

    Returns:
        Smoothed relative pitch angle psi (radians).
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    # observed pitch at samples
    psi_obs = relative_pitch(sample_theta, sample_phi_spatial)
    c2 = np.cos(2.0 * psi_obs)
    s2 = np.sin(2.0 * psi_obs)

    if kernel == "multiplicative":
        if target_theta is None or angular_kappa is None:
            raise ValueError("kernel='multiplicative' requires target_theta and angular_kappa")
        weights = multiplicative_radial_angular_kernel(
            target_r=target_r,
            sample_r=sample_r,
            target_theta=target_theta,
            sample_theta=sample_phi_spatial,  # angular coordinate is spatial angle
            sigma=float(bandwidth),
            kappa=float(angular_kappa),
            normalize_angular=normalize_angular,
        )
    else:
        u = (target_r[:, None] - sample_r[None, :]) / float(bandwidth)
        if kernel == "gaussian":
            weights = gaussian_kernel(u)
        elif kernel == "uniform":
            weights = uniform_kernel(u)
        else:
            raise ValueError(f"Unknown kernel: {kernel!r}")

    denom = np.sum(weights, axis=1)
    zero_mask = denom < 1e-12
    denom[zero_mask] = 1.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        avg_c2 = (weights @ c2) / denom
        avg_s2 = (weights @ s2) / denom

    # For points with effectively zero support, fall back to 0 pitch
    avg_c2[zero_mask] = 0.0
    avg_s2[zero_mask] = 0.0

    return 0.5 * np.arctan2(avg_s2, avg_c2)



def smooth_spiral_pitch(
    target_r: np.ndarray,
    sample_r: np.ndarray,
    sample_alpha_rad: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    
    """Kernel smoothing of spiral pitch using the double-angle embedding."""
    cos_2alpha = np.cos(2 * sample_alpha_rad)
    sin_2alpha = np.sin(2 * sample_alpha_rad)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    u = (target_r[:, None] - sample_r[None, :]) / bandwidth
    weights = gaussian_kernel(u)
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0

    avg_cos = np.sum(weights * cos_2alpha[None, :], axis=1) / sum_weights
    avg_sin = np.sum(weights * sin_2alpha[None, :], axis=1) / sum_weights

    return 0.5 * np.arctan2(avg_sin, avg_cos)
