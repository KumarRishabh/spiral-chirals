from __future__ import annotations

from typing import Tuple
import numpy as np


def to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar (r, theta)."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def wrap_angle_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def wrap_angle_half_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap line-field angles to (-pi/2, pi/2]."""
    return (angle_rad + np.pi / 2) % np.pi - np.pi / 2


def angle_residual_line_field(observed: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Compute line-field residuals wrapped to (-pi/2, pi/2]."""
    return wrap_angle_half_pi(observed - fitted)


def relative_pitch(sample_theta: np.ndarray, sample_phi_spatial: np.ndarray) -> np.ndarray:
    """Compute relative pitch angle (psi) by removing spatial orientation."""
    return sample_theta - sample_phi_spatial


def vector_from_angle(theta_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit vectors (u, v) for angles in radians."""
    return np.cos(theta_rad), np.sin(theta_rad)
