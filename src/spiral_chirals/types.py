from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SpiralDataset:
    """Container for spiral line-field observations."""

    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    angle_deg: np.ndarray
    angle_rad: np.ndarray
    phi_rad: np.ndarray
    u: np.ndarray
    v: np.ndarray
