from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SpiralModelResult:
    p: float
    A: float
    b: float
    RSS: float
    BIC: float


class SpiralBayes:
    """Parametric spiral model comparison using linear least squares."""

    def __init__(self, coordinates: np.ndarray, vectors: np.ndarray):
        self.coords = np.array(coordinates, dtype=float)
        self.vectors = np.array(vectors, dtype=float)
        self.n = len(self.coords)
        self.n_obs = 2 * self.n

        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)

        self.Y = self.vectors.flatten()

    def fit_model(self, p: float) -> SpiralModelResult:
        r_safe = self.r.copy()
        r_safe[r_safe < 1e-6] = 1e-6
        gp = r_safe ** (-1 - p)

        X = np.zeros((self.n_obs, 2))
        X[0::2, 0] = gp * np.cos(self.phi)
        X[0::2, 1] = (1 / r_safe) * np.sin(self.phi)
        X[1::2, 0] = gp * np.sin(self.phi)
        X[1::2, 1] = -(1 / r_safe) * np.cos(self.phi)

        theta, residuals, _, _ = np.linalg.lstsq(X, self.Y, rcond=None)
        A, b = float(theta[0]), float(theta[1])
        RSS = float(residuals[0]) if len(residuals) > 0 else float(np.sum((self.Y - X @ theta) ** 2))

        k = 2
        bic = self.n_obs * np.log(RSS / self.n_obs) + k * np.log(self.n_obs)
        return SpiralModelResult(p=p, A=A, b=b, RSS=RSS, BIC=bic)

    def compare_spirals(self) -> pd.DataFrame:
        models = {
            "Logarithmic": 0,
            "Archimedean": 1,
            "Fermat": 2,
        }
        results: List[Dict[str, float | str]] = []
        for name, p in models.items():
            res = self.fit_model(p)
            results.append({
                "Model": name,
                "p": res.p,
                "A": res.A,
                "b": res.b,
                "RSS": res.RSS,
                "BIC": res.BIC,
            })
        df_res = pd.DataFrame(results)
        min_bic = df_res["BIC"].min()
        df_res["Delta_BIC"] = df_res["BIC"] - min_bic
        df_res["BayesFactor"] = np.exp(-0.5 * df_res["Delta_BIC"])
        return df_res.sort_values("BIC")


def kernel_matrix(r: np.ndarray, bandwidth: float, sigma_n: float = 1e-2) -> np.ndarray:
    dist_sq = (r[:, None] - r[None, :]) ** 2
    K = np.exp(-0.5 * dist_sq / bandwidth**2)
    return K + (sigma_n**2) * np.eye(len(r))


def calculate_gp_lml(r: np.ndarray, y: np.ndarray, bandwidth: float, sigma_n: float) -> float:
    n = len(y)
    K = kernel_matrix(r, bandwidth, sigma_n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        data_fit = -0.5 * y.T @ alpha
        complexity = -np.sum(np.log(np.diag(L)))
        constant = -(n / 2) * np.log(2 * np.pi)
        return float(data_fit + complexity + constant)
    except np.linalg.LinAlgError:
        return float("-inf")


def parametric_log_evidence(rss: float, n: int, k: int = 2) -> float:
    bic = n * np.log(rss / n) + k * np.log(n)
    return -0.5 * bic


def perform_bayesian_comparison(
    analysis_obj: SpiralBayes,
    y_r_obs: np.ndarray,
    y_phi_obs: np.ndarray,
    bandwidth: float = 1.0,
    sigma_n: float = 0.1,
) -> pd.DataFrame:
    comparison: List[Dict[str, float | str]] = []

    for name, p_val in {"Log": 0, "Archimedean": 1, "Fermat": 2}.items():
        res = analysis_obj.fit_model(p_val)
        ln_E = parametric_log_evidence(res.RSS, analysis_obj.n_obs)
        comparison.append({"Model": name, "Type": "Parametric", "LogEvidence": ln_E})

    lml_r = calculate_gp_lml(analysis_obj.r, y_r_obs, bandwidth, sigma_n)
    lml_phi = calculate_gp_lml(analysis_obj.r, y_phi_obs, bandwidth, sigma_n)
    lml_total = lml_r + lml_phi
    comparison.append({"Model": "RKHS (RBF)", "Type": "Non-Parametric", "LogEvidence": lml_total})

    df_comp = pd.DataFrame(comparison)
    max_lml = df_comp["LogEvidence"].max()
    df_comp["BayesFactor"] = np.exp(df_comp["LogEvidence"] - max_lml)
    return df_comp.sort_values("LogEvidence", ascending=False)
