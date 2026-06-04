from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FourierFit:
    coeff_u: np.ndarray
    coeff_v: np.ndarray
    kmax: int


def make_grid(n: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, XY


def smooth_radial(X: np.ndarray, Y: np.ndarray, cx: float = 0.5, cy: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + 1e-12
    ux = (X - cx) * np.exp(-(r / 0.35) ** 2)
    uy = (Y - cy) * np.exp(-(r / 0.35) ** 2)
    return ux, uy


def smooth_spiral(
    X: np.ndarray, Y: np.ndarray, cx: float = 0.5, cy: float = 0.5, twist: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    dx = X - cx
    dy = Y - cy
    r2 = dx * dx + dy * dy
    ux = -dy * np.exp(-3 * r2) * (1 + twist * r2)
    uy = dx * np.exp(-3 * r2) * (1 + twist * r2)
    return ux, uy


def add_noise(U: np.ndarray, V: np.ndarray, sigma: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    return U + sigma * np.random.randn(*U.shape), V + sigma * np.random.randn(*V.shape)


def fourier_basis_2d(X: np.ndarray, Y: np.ndarray, kmax: int = 3) -> np.ndarray:
    """Return basis matrix Phi of shape [n_points, n_features]."""
    x = X.ravel()
    y = Y.ravel()
    feats = [np.ones_like(x)]
    for k in range(1, kmax + 1):
        feats.append(np.cos(2 * np.pi * k * x))
        feats.append(np.sin(2 * np.pi * k * x))
        feats.append(np.cos(2 * np.pi * k * y))
        feats.append(np.sin(2 * np.pi * k * y))
    for kx in range(1, kmax + 1):
        for ky in range(1, kmax + 1):
            cx = np.cos(2 * np.pi * kx * x)
            sx = np.sin(2 * np.pi * kx * x)
            cy = np.cos(2 * np.pi * ky * y)
            sy = np.sin(2 * np.pi * ky * y)
            feats += [cx * cy, cx * sy, sx * cy, sx * sy]
    return np.stack(feats, axis=1)


def fourier_penalty(kmax: int) -> np.ndarray:
    """Construct diagonal penalty weights for fourier_basis_2d."""
    weights = [0.0]
    for k in range(1, kmax + 1):
        w = k**2
        weights += [w, w, w, w]
    for kx in range(1, kmax + 1):
        for ky in range(1, kmax + 1):
            w = kx**2 + ky**2
            weights += [w, w, w, w]
    return np.array(weights, dtype=float)


def ridge_fit(Phi: np.ndarray, y: np.ndarray, Rdiag: np.ndarray, lam: float) -> np.ndarray:
    """Solve ridge regression: (Phi^T Phi + lam R) c = Phi^T y."""
    PtP = Phi.T @ Phi
    R = np.diag(Rdiag)
    return np.linalg.solve(PtP + lam * R, Phi.T @ y)


def fit_vector_field_fourier(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    kmax: int = 3,
    lam: float = 1e-2,
) -> FourierFit:
    """Fit Fourier basis coefficients to a vector field."""
    Phi = fourier_basis_2d(X, Y, kmax=kmax)
    Rdiag = fourier_penalty(kmax)
    coeff_u = ridge_fit(Phi, U.ravel(), Rdiag, lam)
    coeff_v = ridge_fit(Phi, V.ravel(), Rdiag, lam)
    return FourierFit(coeff_u=coeff_u, coeff_v=coeff_v, kmax=kmax)
