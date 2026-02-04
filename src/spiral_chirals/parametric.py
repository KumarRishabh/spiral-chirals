from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    value: float
    success: bool
    message: str
    objective_value: float


def log_spiral_pitch(r: np.ndarray, k: float, eps: float = 1e-8) -> np.ndarray:
    """Log-spiral pitch model used in the notebooks: alpha'(r) = k * log(r)."""
    return k * np.log(r + eps)


def fermat_pitch(r: np.ndarray, a: float) -> np.ndarray:
    """Fermat spiral pitch: alpha'(r) = arctan(2 r^2 / a^2)."""
    return np.arctan(2 * r**2 / (a**2))


def archimedean_pitch(r: np.ndarray, b: float) -> np.ndarray:
    """Archimedean spiral pitch: alpha'(r) = arctan(r / b)."""
    return np.arctan(r / b)


def _weighted_residuals(angle_rad: np.ndarray, predicted: np.ndarray, r: np.ndarray, scaling: bool) -> np.ndarray:
    if scaling:
        r_safe = r.copy()
        r_safe[r_safe < 1e-8] = 1e-8
        return (angle_rad - predicted) / r_safe
    return angle_rad - predicted


def _fit_scalar_param(
    objective_fn: Callable[[float], float],
    x0: float,
) -> FitResult:
    result = minimize(lambda z: objective_fn(float(np.atleast_1d(z)[0])), x0)
    return FitResult(
        value=float(result.x[0]),
        success=bool(result.success),
        message=str(result.message),
        objective_value=float(result.fun),
    )


def fit_log_spiral(
    angle_rad: np.ndarray,
    r: np.ndarray,
    k0: float = 1.0,
    scaling: bool = True,
    use_sin: bool = False,
) -> FitResult:
    """Fit the log-spiral model using least squares or sine loss."""
    def objective(k: float) -> float:
        predicted = log_spiral_pitch(r, k)
        if use_sin:
            return float(np.sum(np.sin(angle_rad - predicted) ** 2))
        residuals = _weighted_residuals(angle_rad, predicted, r, scaling)
        return float(np.sum(residuals**2))

    return _fit_scalar_param(objective, k0)


def fit_fermat_spiral(
    angle_rad: np.ndarray,
    r: np.ndarray,
    a0: float = 0.1,
    scaling: bool = True,
    use_sin: bool = False,
) -> FitResult:
    """Fit the Fermat spiral model using least squares or sine loss."""
    def objective(a: float) -> float:
        predicted = fermat_pitch(r, a)
        if use_sin:
            return float(np.sum(np.sin(angle_rad - predicted) ** 2))
        residuals = _weighted_residuals(angle_rad, predicted, r, scaling)
        return float(np.sum(residuals**2))

    return _fit_scalar_param(objective, a0)


def fit_archimedean_spiral(
    angle_rad: np.ndarray,
    r: np.ndarray,
    b0: float = 1.0,
    scaling: bool = True,
    use_sin: bool = False,
) -> FitResult:
    """Fit the Archimedean spiral model using least squares or sine loss."""
    def objective(b: float) -> float:
        predicted = archimedean_pitch(r, b)
        if use_sin:
            return float(np.sum(np.sin(angle_rad - predicted) ** 2))
        residuals = _weighted_residuals(angle_rad, predicted, r, scaling)
        return float(np.sum(residuals**2))

    return _fit_scalar_param(objective, b0)


def predict_phi(theta: np.ndarray, pitch_rad: np.ndarray) -> np.ndarray:
    """Combine spatial angle and pitch to form the global direction angle."""
    return theta + pitch_rad
