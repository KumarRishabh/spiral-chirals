"""
Experiment tracking and management for spiral vector field fitting.

Provides:
- ExperimentConfig: YAML/JSON config for reproducible runs
- ExperimentRunner: Execute and log experiments with data variants
- Results aggregation and comparison
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .io import load_angle_coordinate_csv, build_spiral_dataset
from .kernels import smooth_line_field
from .parametric import fit_log_spiral, fit_fermat_spiral, fit_archimedean_spiral
from .geometry import angle_residual_line_field
from .bayes import SpiralBayes
from .types import SpiralDataset


@dataclass
class KernelFitConfig:
    """Configuration for kernel regression."""
    bandwidth: float = 1.0
    kernel: str = "gaussian"


@dataclass
class ParametricFitConfig:
    """Configuration for parametric models."""
    fit_log_spiral: bool = True
    fit_fermat: bool = True
    fit_archimedean: bool = True
    scaling: bool = True
    use_sin_loss: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    csv_file: str
    angle_col: str = "Angle (α′)"
    coord_col: str = "Coordinate"
    data_transforms: List[Dict[str, Any]] = field(default_factory=list)
    kernel_config: Optional[KernelFitConfig] = None
    parametric_config: Optional[ParametricFitConfig] = None
    run_bayes: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Note: asdict() already converts nested dataclasses to dicts
        # so kernel_config and parametric_config are already dicts
        return d


class DataTransformRegistry:
    """Registry of standard data transformations."""

    @staticmethod
    def subsample(data: SpiralDataset, frac: float = 0.8) -> SpiralDataset:
        """Subsample data by a fraction."""
        n = len(data.x)
        idx = np.random.choice(n, size=int(n * frac), replace=False)
        return SpiralDataset(
            x=data.x[idx], y=data.y[idx], r=data.r[idx], theta=data.theta[idx],
            angle_deg=data.angle_deg[idx], angle_rad=data.angle_rad[idx],
            phi_rad=data.phi_rad[idx], u=data.u[idx], v=data.v[idx],
        )

    @staticmethod
    def add_noise(data: SpiralDataset, sigma: float = 0.1) -> SpiralDataset:
        """Add Gaussian noise to angles (in degrees)."""
        angle_noisy = data.angle_rad + np.random.normal(0, np.radians(sigma), len(data.angle_rad))
        u_noisy = np.cos(data.theta + angle_noisy)
        v_noisy = np.sin(data.theta + angle_noisy)
        return SpiralDataset(
            x=data.x, y=data.y, r=data.r, theta=data.theta,
            angle_deg=np.degrees(angle_noisy), angle_rad=angle_noisy,
            phi_rad=data.theta + angle_noisy, u=u_noisy, v=v_noisy,
        )

    @staticmethod
    def outlier_remove(data: SpiralDataset, percentile: float = 95.0) -> SpiralDataset:
        """Remove outlier angles."""
        threshold = np.percentile(np.abs(data.angle_rad), percentile)
        mask = np.abs(data.angle_rad) <= threshold
        return SpiralDataset(
            x=data.x[mask], y=data.y[mask], r=data.r[mask], theta=data.theta[mask],
            angle_deg=data.angle_deg[mask], angle_rad=data.angle_rad[mask],
            phi_rad=data.phi_rad[mask], u=data.u[mask], v=data.v[mask],
        )

    @classmethod
    def apply(cls, data: SpiralDataset, transforms: List[Dict[str, Any]]) -> SpiralDataset:
        """Apply a list of transforms sequentially."""
        for t in transforms:
            name = t.get("name")
            params = t.get("params", {})
            if hasattr(cls, name):
                method = getattr(cls, name)
                data = method(data, **params)
        return data


class ExperimentRunner:
    """Run and log spiral fitting experiments."""

    def __init__(self, output_dir: str | Path = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment and save results."""
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Load and transform data
        df = load_angle_coordinate_csv(
            config.csv_file, angle_col=config.angle_col, coord_col=config.coord_col
        )
        data = build_spiral_dataset(df, angle_col=config.angle_col, coord_col=config.coord_col)
        data = DataTransformRegistry.apply(data, config.data_transforms)

        result = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data.x),
        }

        # Run kernel fit
        if config.kernel_config:
            result["kernel_results"] = self._run_kernel_fit(data, config.kernel_config, exp_dir)

        # Run parametric fits
        if config.parametric_config:
            result["parametric_results"] = self._run_parametric_fit(data, config.parametric_config, exp_dir)

        # Run Bayesian comparison
        if config.run_bayes:
            result["bayes_results"] = self._run_bayes(data, exp_dir)

        # Save results
        self._save_result(result, exp_dir)
        return result

    def _run_kernel_fit(self, data: SpiralDataset, cfg: KernelFitConfig, exp_dir: Path) -> Dict[str, Any]:
        psi_fitted = smooth_line_field(
            target_r=data.r, sample_r=data.r, sample_theta=data.phi_rad,
            sample_phi_spatial=data.theta, bandwidth=cfg.bandwidth,
        )
        theta_fitted = data.theta + psi_fitted
        residuals = angle_residual_line_field(data.phi_rad, theta_fitted)
        res_deg = np.degrees(residuals)

        np.save(exp_dir / "kernel_residuals.npy", residuals)
        return {
            "bandwidth": cfg.bandwidth,
            "mae_deg": float(np.mean(np.abs(res_deg))),
            "rmse_deg": float(np.sqrt(np.mean(res_deg**2))),
            "mean_residual_deg": float(np.mean(res_deg)),
            "std_residual_deg": float(np.std(res_deg)),
        }

    def _run_parametric_fit(self, data: SpiralDataset, cfg: ParametricFitConfig, exp_dir: Path) -> Dict[str, Any]:
        results = {}
        if cfg.fit_log_spiral:
            res = fit_log_spiral(data.angle_rad, data.r, k0=1.0, scaling=cfg.scaling, use_sin=cfg.use_sin_loss)
            results["log_spiral"] = {"k": res.value, "rss": res.objective_value, "success": res.success}
        if cfg.fit_fermat:
            res = fit_fermat_spiral(data.angle_rad, data.r, a0=0.1, scaling=cfg.scaling, use_sin=cfg.use_sin_loss)
            results["fermat"] = {"a": res.value, "rss": res.objective_value, "success": res.success}
        if cfg.fit_archimedean:
            res = fit_archimedean_spiral(data.angle_rad, data.r, b0=1.0, scaling=cfg.scaling, use_sin=cfg.use_sin_loss)
            results["archimedean"] = {"b": res.value, "rss": res.objective_value, "success": res.success}
        return results

    def _run_bayes(self, data: SpiralDataset, exp_dir: Path) -> List[Dict[str, Any]]:
        analysis = SpiralBayes(coordinates=np.column_stack([data.x, data.y]), vectors=np.column_stack([data.u, data.v]))
        bayes_df = analysis.compare_spirals()
        bayes_df.to_csv(exp_dir / "bayes_comparison.csv", index=False)
        return bayes_df.to_dict(orient="records")

    def _save_result(self, result: Dict[str, Any], exp_dir: Path) -> None:
        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(result["config"], f, indent=2)

        result_path = exp_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    def run_batch(self, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """Run multiple experiments."""
        results = []
        for config in configs:
            print(f"Running experiment: {config.name}")
            result = self.run(config)
            results.append(result)
        return results

    def summarize(self) -> pd.DataFrame:
        """Load all results and create a summary table."""
        rows = []
        for exp_dir in sorted(self.output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            result_path = exp_dir / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                    best_model = (data.get("bayes_results", [{}])[0].get("Model") if data.get("bayes_results") else None)
                    rows.append({
                        "experiment": exp_dir.name,
                        "timestamp": data["timestamp"],
                        "n_samples": data["n_samples"],
                        "kernel_mae": data.get("kernel_results", {}).get("mae_deg"),
                        "best_model": best_model,
                    })
        return pd.DataFrame(rows)