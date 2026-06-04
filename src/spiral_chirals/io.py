from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import ast

import numpy as np
import pandas as pd

from .types import SpiralDataset
from .geometry import to_polar, vector_from_angle


def parse_coord(value: object) -> Tuple[float, float]:
    """Parse a coordinate value from string/list/tuple to (x, y).

    Supports formats like "(7.00, 0.00)" or "[7.00, 0.00]".
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) >= 2:
            return float(value[0]), float(value[1])
        return np.nan, np.nan
    if not isinstance(value, str):
        return np.nan, np.nan

    text = value.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass

    parts = [p.strip() for p in text.strip("()[]").split(",")]
    if len(parts) < 2:
        return np.nan, np.nan
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return np.nan, np.nan


def load_angle_coordinate_csv(
    csv_path: str | Path,
    angle_col: str = "Angle (α′)",
    coord_col: str = "Coordinate",
    drop_last: bool = True,
) -> pd.DataFrame:
    """Load spiral coordinate data from a CSV file."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if drop_last and len(df) > 0:
        df = df.iloc[:-1]
    if angle_col not in df.columns:
        raise KeyError(f"Angle column '{angle_col}' not found in {csv_path}.")
    if coord_col not in df.columns:
        raise KeyError(f"Coordinate column '{coord_col}' not found in {csv_path}.")
    return df


def build_spiral_dataset(
    df: pd.DataFrame,
    angle_col: str = "Angle (α′)",
    coord_col: str = "Coordinate",
) -> SpiralDataset:
    """Build a SpiralDataset from a dataframe with angle + coordinate columns."""
    coordinates: Iterable[Tuple[float, float]] = df[coord_col].apply(parse_coord).tolist()
    x = np.array([c[0] for c in coordinates], dtype=float)
    y = np.array([c[1] for c in coordinates], dtype=float)
    r, theta = to_polar(x, y)

    angle_deg = df[angle_col].to_numpy(dtype=float)
    angle_rad = np.deg2rad(angle_deg)

    phi_rad = angle_rad + theta
    u, v = vector_from_angle(phi_rad)

    return SpiralDataset(
        x=x,
        y=y,
        r=r,
        theta=theta,
        angle_deg=angle_deg,
        angle_rad=angle_rad,
        phi_rad=phi_rad,
        u=u,
        v=v,
    )
