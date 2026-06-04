from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_line_field_quiver(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color: Optional[np.ndarray] = None,
    scale: float = 1.0,
    title: str | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a line field using headless quivers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    q = ax.quiver(
        x,
        y,
        u,
        v,
        color,
        angles="xy",
        scale_units="xy",
        scale=scale,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="mid",
        width=0.004,
    )
    ax.scatter(x, y, s=8, c="k", alpha=0.3)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if color is not None:
        plt.colorbar(q, ax=ax)
    return ax


def plot_streamlines(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U_grid: np.ndarray,
    V_grid: np.ndarray,
    density: float = 1.5,
    title: str | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.streamplot(X_grid, Y_grid, U_grid, V_grid, density=density, linewidth=1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_structure_function(
    r: np.ndarray,
    pitch_rad: np.ndarray,
    r_grid: np.ndarray,
    pitch_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r, np.degrees(pitch_rad), s=10, alpha=0.4, color="gray")
    ax.plot(r_grid, np.degrees(pitch_grid), color="red", linewidth=2)
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Pitch (deg)")
    ax.set_ylim(-90, 90)
    ax.grid(alpha=0.3)
    return ax


def plot_residual_hist(residuals_rad: np.ndarray, bins: int = 30, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.degrees(residuals_rad), bins=bins, color="purple", alpha=0.7, density=True)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("Residual (deg)")
    ax.set_ylabel("Density")
    return ax
