from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import griddata
from sklearn.model_selection import KFold

from cleaned_linefield_comparison import (
    FIXED_PS,
    axial_mae_deg,
    axial_residual,
    axial_rss,
    fit_continuous_p,
    fit_gamma_for_p,
    parametric_predict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VF_DIR = ROOT / "vf_exports"
DEFAULT_OUT_DIR = ROOT / "data" / "gabor_nw_gp_kernel_regression_comparison"

# Matched Gabor reporting protocol:
# h_j = exp(log(1.0) + j * (log(1500.0) - log(1.0)) / 13), j=0,...,13.
# RBF and uniform sweep h only; RBF-von-Mises sweeps h crossed with DEFAULT_KAPPAS.
# GP/KRR additionally crosses every kernel candidate with DEFAULT_SIGMAS; parametric
# baselines are refit and scored on the same cross-validation folds.
DEFAULT_BANDWIDTHS = np.exp(np.linspace(np.log(1.0), np.log(1500.0), 14))
DEFAULT_KAPPAS = np.array(
    [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    dtype=float,
)
DEFAULT_SIGMAS = np.array([0.03, 0.06, 0.10, 0.18, 0.32, 0.56, 1.0], dtype=float)
STREAM_DENSITY = 1.15
STREAM_LINEWIDTH = 0.75
STREAM_ARROWSIZE = 0.85


@dataclass(frozen=True)
class KernelCfg:
    kernel: str
    bandwidth: float
    kappa: float | None = None


def parse_coord(value: object) -> tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass
    parts = [part.strip() for part in text.strip("()[]").split(",")]
    if len(parts) < 2:
        return np.nan, np.nan
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return np.nan, np.nan


def load_gabor_vf(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, float | int | str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"Coordinate", "Angle (α′)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    coords = np.asarray(df["Coordinate"].apply(parse_coord).tolist(), dtype=float)
    alpha_deg = pd.to_numeric(df["Angle (α′)"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(alpha_deg)
    X = coords[valid].astype(float)
    alpha = np.deg2rad(alpha_deg[valid])
    theta = np.arctan2(X[:, 1], X[:, 0])
    phi = theta + alpha
    stats: dict[str, float | int | str] = {
        "dataset": csv_path.name,
        "n_raw": int(len(df)),
        "n_used": int(len(phi)),
        "n_dropped": int(len(df) - len(phi)),
    }
    return X, alpha.astype(float), phi.astype(float), stats


def pitch_embedding(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([np.cos(2.0 * alpha), np.sin(2.0 * alpha)]).astype(float)


def embedding_to_pitch(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.arctan2(Y[:, 1], Y[:, 0])


def pairwise_radial_delta(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    r1 = np.linalg.norm(X1, axis=1)
    r2 = np.linalg.norm(X2, axis=1)
    return r1[:, None] - r2[None, :]


def kernel_matrix(cfg: KernelCfg, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    dr = pairwise_radial_delta(X1, X2)
    if cfg.kernel == "rbf_pitch":
        return np.exp(-0.5 * (dr / cfg.bandwidth) ** 2)
    if cfg.kernel == "uniform_pitch":
        return (np.abs(dr) <= cfg.bandwidth).astype(float)
    if cfg.kernel == "multiplicative_pitch":
        if cfg.kappa is None:
            raise ValueError("multiplicative_pitch requires kappa")
        th1 = np.arctan2(X1[:, 1], X1[:, 0])
        th2 = np.arctan2(X2[:, 1], X2[:, 0])
        radial = np.exp(-0.5 * (dr / cfg.bandwidth) ** 2)
        angular = np.exp(cfg.kappa * (np.cos(th1[:, None] - th2[None, :]) - 1.0))
        return radial * angular
    raise ValueError(cfg.kernel)


def model_label(estimator: str, kernel: str) -> str:
    prefix = "NW" if estimator == "nw" else "GP/KRR"
    if kernel == "rbf_pitch":
        return f"{prefix} RBF pitch"
    if kernel == "uniform_pitch":
        return f"{prefix} uniform pitch"
    if kernel == "multiplicative_pitch":
        return f"{prefix} RBF-von-Mises pitch"
    raise ValueError(kernel)


def fixed_grid(kernel: str, bandwidths: NDArray[np.float64], kappas: NDArray[np.float64]) -> list[KernelCfg]:
    if kernel in {"rbf_pitch", "uniform_pitch"}:
        return [KernelCfg(kernel, float(h)) for h in bandwidths]
    if kernel == "multiplicative_pitch":
        return [KernelCfg(kernel, float(h), float(kappa)) for h in bandwidths for kappa in kappas]
    raise ValueError(kernel)


def predict_nw_alpha(cfg: KernelCfg, Xtrain: NDArray[np.float64], alpha_train: NDArray[np.float64], Xtarget: NDArray[np.float64]) -> NDArray[np.float64]:
    K = kernel_matrix(cfg, Xtarget, Xtrain)
    denom = np.sum(K, axis=1, keepdims=True)
    empty = denom[:, 0] <= 1e-14
    if np.any(empty):
        d2 = np.sum((Xtarget[empty, None, :] - Xtrain[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        K[empty, :] = 0.0
        K[np.where(empty)[0], nearest] = 1.0
        denom = np.sum(K, axis=1, keepdims=True)
    weights = K / np.maximum(denom, 1e-14)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Yhat = weights @ pitch_embedding(alpha_train)
    if not np.isfinite(Yhat).all():
        Yhat = np.nan_to_num(Yhat, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding_to_pitch(Yhat)


def predict_gp_alpha(
    cfg: KernelCfg,
    sigma_n: float,
    Xtrain: NDArray[np.float64],
    alpha_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    Y = pitch_embedding(alpha_train)
    K = kernel_matrix(cfg, Xtrain, Xtrain)
    K = 0.5 * (K + K.T)
    Kte = kernel_matrix(cfg, Xtarget, Xtrain)
    try:
        evals, evecs = np.linalg.eigh(K)
    except np.linalg.LinAlgError:
        return None
    denom = evals + sigma_n * sigma_n
    abs_denom = np.abs(denom)
    if np.any(abs_denom < 1e-6):
        return None
    if float(np.max(abs_denom) / np.min(abs_denom)) > 1e12:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coeff = (evecs.T @ Y) / denom[:, None]
        Yhat = Kte @ evecs @ coeff
    if not np.isfinite(Yhat).all():
        return None
    return embedding_to_pitch(Yhat)


def predict_phi_from_alpha(alpha_hat: NDArray[np.float64], Xtarget: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.arctan2(Xtarget[:, 1], Xtarget[:, 0]) + alpha_hat


def cv_splits(n: int, n_folds: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    cv = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=42)
    index = np.arange(n)
    return [(tr, te) for tr, te in cv.split(index)]


def score_nw_cfg(
    dataset: str,
    X: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: KernelCfg,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> dict[str, float | int | str]:
    fold_rss = []
    fold_mae = []
    for train_idx, test_idx in splits:
        alpha_hat = predict_nw_alpha(cfg, X[train_idx], alpha[train_idx], X[test_idx])
        pred = predict_phi_from_alpha(alpha_hat, X[test_idx])
        fold_rss.append(axial_rss(phi[test_idx], pred) / len(test_idx))
        fold_mae.append(axial_mae_deg(phi[test_idx], pred))
    return {
        "dataset": dataset,
        "n": int(len(phi)),
        "estimator": "Nadaraya-Watson",
        "model": model_label("nw", cfg.kernel),
        "kernel": cfg.kernel,
        "bandwidth": float(cfg.bandwidth),
        "kappa": np.nan if cfg.kappa is None else float(cfg.kappa),
        "sigma_n": np.nan,
        "mean_test_rss": float(np.mean(fold_rss)),
        "sd_test_rss": float(np.std(fold_rss, ddof=1)) if len(fold_rss) > 1 else 0.0,
        "mean_test_mae_deg": float(np.mean(fold_mae)),
        "n_folds": int(len(splits)),
    }


def score_gp_cfg(
    dataset: str,
    X: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: KernelCfg,
    sigmas: NDArray[np.float64],
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> list[dict[str, float | int | str]]:
    rss_by_sigma: list[list[float]] = [[] for _ in sigmas]
    mae_by_sigma: list[list[float]] = [[] for _ in sigmas]
    failed = np.zeros(len(sigmas), dtype=bool)

    for train_idx, test_idx in splits:
        for idx, sigma_n in enumerate(sigmas):
            if failed[idx]:
                continue
            alpha_hat = predict_gp_alpha(cfg, float(sigma_n), X[train_idx], alpha[train_idx], X[test_idx])
            if alpha_hat is None:
                failed[idx] = True
                continue
            pred = predict_phi_from_alpha(alpha_hat, X[test_idx])
            rss_by_sigma[idx].append(axial_rss(phi[test_idx], pred) / len(test_idx))
            mae_by_sigma[idx].append(axial_mae_deg(phi[test_idx], pred))

    rows: list[dict[str, float | int | str]] = []
    for idx, sigma_n in enumerate(sigmas):
        if failed[idx] or len(rss_by_sigma[idx]) != len(splits):
            continue
        rows.append(
            {
                "dataset": dataset,
                "n": int(len(phi)),
                "estimator": "GP/KRR",
                "model": model_label("gp", cfg.kernel),
                "kernel": cfg.kernel,
                "bandwidth": float(cfg.bandwidth),
                "kappa": np.nan if cfg.kappa is None else float(cfg.kappa),
                "sigma_n": float(sigma_n),
                "mean_test_rss": float(np.mean(rss_by_sigma[idx])),
                "sd_test_rss": float(np.std(rss_by_sigma[idx], ddof=1)) if len(rss_by_sigma[idx]) > 1 else 0.0,
                "mean_test_mae_deg": float(np.mean(mae_by_sigma[idx])),
                "n_folds": int(len(splits)),
            }
        )
    return rows


def score_parametric_cfgs(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    specs = [(p, model) for p, model in FIXED_PS] + [(None, "Parametric continuous p")]
    for p_fixed, model in specs:
        fold_rss = []
        fold_mae = []
        fit_ps = []
        fit_gammas = []
        for train_idx, test_idx in splits:
            if p_fixed is None:
                p, gamma, _ = fit_continuous_p(X[train_idx], phi[train_idx])
            else:
                p = float(p_fixed)
                gamma, _ = fit_gamma_for_p(X[train_idx], phi[train_idx], p)
            pred = parametric_predict(X[test_idx], p, gamma)
            fold_rss.append(axial_rss(phi[test_idx], pred) / len(test_idx))
            fold_mae.append(axial_mae_deg(phi[test_idx], pred))
            fit_ps.append(p)
            fit_gammas.append(gamma)
        rows.append(
            {
                "dataset": dataset,
                "n": int(len(phi)),
                "estimator": "Parametric",
                "model": model,
                "kernel": "parametric",
                "bandwidth": np.nan,
                "kappa": np.nan,
                "sigma_n": np.nan,
                "mean_test_rss": float(np.mean(fold_rss)),
                "sd_test_rss": float(np.std(fold_rss, ddof=1)) if len(fold_rss) > 1 else 0.0,
                "mean_test_mae_deg": float(np.mean(fold_mae)),
                "n_folds": int(len(splits)),
                "p_median": float(np.median(fit_ps)),
                "gamma_median": float(np.median(fit_gammas)),
            }
        )
    return rows


def best_by_kernel(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    df = pd.DataFrame(rows)
    out = []
    for _, group in df.groupby(["dataset", "estimator", "kernel"], dropna=False):
        out.append(group.loc[group["mean_test_rss"].astype(float).idxmin()].to_dict())
    return out


def orient_director_grid(U: NDArray[np.float64], V: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    Uo = U.copy()
    Vo = V.copy()
    rows, cols = Uo.shape
    for i in range(rows):
        for j in range(1, cols):
            if np.isnan(Uo[i, j]) or np.isnan(Uo[i, j - 1]):
                continue
            if Uo[i, j] * Uo[i, j - 1] + Vo[i, j] * Vo[i, j - 1] < 0:
                Uo[i, j] *= -1
                Vo[i, j] *= -1
    for j in range(cols):
        for i in range(1, rows):
            if np.isnan(Uo[i, j]) or np.isnan(Uo[i - 1, j]):
                continue
            if Uo[i, j] * Uo[i - 1, j] + Vo[i, j] * Vo[i - 1, j] < 0:
                Uo[i, j] *= -1
                Vo[i, j] *= -1
    return Uo, Vo


def notebook_style_stream_grid(
    X: NDArray[np.float64],
    phi_fit_obs: NDArray[np.float64],
    grid_points: int = 220,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    xi = np.linspace(float(np.min(X[:, 0])), float(np.max(X[:, 0])), grid_points)
    yi = np.linspace(float(np.min(X[:, 1])), float(np.max(X[:, 1])), grid_points)
    Xi, Yi = np.meshgrid(xi, yi)
    U = np.cos(phi_fit_obs)
    V = np.sin(phi_fit_obs)
    Ui = griddata((X[:, 0], X[:, 1]), U, (Xi, Yi), method="linear")
    Vi = griddata((X[:, 0], X[:, 1]), V, (Xi, Yi), method="linear")
    Ui, Vi = orient_director_grid(Ui, Vi)
    mask_nan = np.isnan(Ui) | np.isnan(Vi)
    Ui[mask_nan] = 0.0
    Vi[mask_nan] = 0.0
    return Xi, Yi, Ui, Vi


def add_observed_arrows(ax: plt.Axes, X: NDArray[np.float64], phi: NDArray[np.float64]) -> None:
    span = max(float(np.max(np.ptp(X, axis=0))), 1.0)
    length = span / 34.0
    ax.quiver(
        X[:, 0],
        X[:, 1],
        length * np.cos(phi),
        length * np.sin(phi),
        color="#4F5B66",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        headwidth=4.5,
        headlength=5.5,
        headaxislength=4.5,
        alpha=0.55,
        zorder=2,
    )
    ax.scatter(X[:, 0], X[:, 1], s=8, color="black", alpha=0.25, zorder=3)


def cfg_from_row(row: pd.Series) -> KernelCfg:
    kappa = None if pd.isna(row.get("kappa", np.nan)) else float(row["kappa"])
    return KernelCfg(str(row["kernel"]), float(row["bandwidth"]), kappa)


def full_fit_phi(row: pd.Series, X: NDArray[np.float64], alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    if row["estimator"] == "Parametric":
        return parametric_predict(X, float(row["p_median"]), float(row["gamma_median"]))
    cfg = cfg_from_row(row)
    if row["estimator"] == "Nadaraya-Watson":
        alpha_hat = predict_nw_alpha(cfg, X, alpha, X)
    else:
        alpha_hat = predict_gp_alpha(cfg, float(row["sigma_n"]), X, alpha, X)
        if alpha_hat is None:
            alpha_hat = np.zeros(len(X), dtype=float)
    return predict_phi_from_alpha(alpha_hat, X)


OVERLAY_ORDER = [
    ("Nadaraya-Watson", "rbf_pitch"),
    ("Nadaraya-Watson", "uniform_pitch"),
    ("Nadaraya-Watson", "multiplicative_pitch"),
    ("GP/KRR", "rbf_pitch"),
    ("GP/KRR", "uniform_pitch"),
    ("GP/KRR", "multiplicative_pitch"),
    ("Parametric", "Parametric continuous p"),
    ("Parametric", "Parametric p=0 (Logarithmic)"),
    ("Parametric", "Parametric p=1 (Archimedean)"),
    ("Parametric", "Parametric p=2 (Fermat)"),
]

OVERLAY_COLORS = {
    ("Nadaraya-Watson", "rbf_pitch"): "#1F5B99",
    ("Nadaraya-Watson", "uniform_pitch"): "#8A5A00",
    ("Nadaraya-Watson", "multiplicative_pitch"): "#13795B",
    ("GP/KRR", "rbf_pitch"): "#5E6AD2",
    ("GP/KRR", "uniform_pitch"): "#B45309",
    ("GP/KRR", "multiplicative_pitch"): "#7C3AED",
    ("Parametric", "Parametric continuous p"): "#D55E00",
    ("Parametric", "Parametric p=0 (Logarithmic)"): "#D55E00",
    ("Parametric", "Parametric p=1 (Archimedean)"): "#D55E00",
    ("Parametric", "Parametric p=2 (Fermat)"): "#D55E00",
}


def selected_overlay_row(selected_rows: pd.DataFrame, key: tuple[str, str]) -> pd.Series:
    estimator, selector = key
    if estimator == "Parametric":
        rows = selected_rows[(selected_rows["estimator"] == estimator) & (selected_rows["model"] == selector)]
    else:
        rows = selected_rows[(selected_rows["estimator"] == estimator) & (selected_rows["kernel"] == selector)]
    if rows.empty:
        raise ValueError(f"No selected row found for {key}")
    return rows.iloc[0]


def overlay_title(row: pd.Series) -> str:
    title = f"{row['model']}\nRSS={float(row['mean_test_rss']):.3f}"
    if row["estimator"] == "Parametric":
        return title + f", p={float(row['p_median']):.3g}, gamma={float(row['gamma_median']):.2f}"
    title += f", h={float(row['bandwidth']):.2g}"
    if not pd.isna(row["kappa"]):
        title += f", kappa={float(row['kappa']):g}"
    if not pd.isna(row["sigma_n"]):
        title += f", sigma={float(row['sigma_n']):g}"
    return title


def plot_fitted_streamlines(
    ax: plt.Axes,
    row: pd.Series,
    key: tuple[str, str],
    X: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> tuple[str, NDArray[np.float64]]:
    phi_fit_obs = full_fit_phi(row, X, alpha)
    Xi, Yi, Ui, Vi = notebook_style_stream_grid(X, phi_fit_obs)
    add_observed_arrows(ax, X, phi)
    ax.streamplot(
        Xi,
        Yi,
        Ui,
        Vi,
        density=STREAM_DENSITY,
        color=OVERLAY_COLORS[key],
        linewidth=STREAM_LINEWIDTH,
        arrowsize=STREAM_ARROWSIZE,
    )
    ax.set_title(overlay_title(row), fontsize=9)
    return str(row["model"]), np.degrees(axial_residual(phi, phi_fit_obs))


def make_overlay_plot(dataset: str, X: NDArray[np.float64], alpha: NDArray[np.float64], phi: NDArray[np.float64], selected_rows: pd.DataFrame, overlay_dir: Path) -> str:
    fig, axes = plt.subplots(3, 4, figsize=(20.0, 13.6))
    axes_flat = axes.ravel()
    add_observed_arrows(axes_flat[0], X, phi)
    axes_flat[0].set_title("Observed Gabor vector field")

    hist_rows = []
    for ax, key in zip(axes_flat[1:11], OVERLAY_ORDER):
        row = selected_overlay_row(selected_rows, key)
        hist_rows.append(plot_fitted_streamlines(ax, row, key, X, alpha, phi))

    ax_hist = axes_flat[11]
    bins = np.linspace(-90, 90, 31)
    for label, vals in hist_rows:
        ax_hist.hist(vals, bins=bins, alpha=0.24, density=True, label=label)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_hist.set_title("Full-data residual diagnostic")
    ax_hist.set_xlabel("axial residual (degrees)")
    ax_hist.set_ylabel("density")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend(fontsize=7)

    for ax in axes_flat[:11]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.suptitle(dataset, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = dataset.replace(".csv", "_nw_gp_streamlines.png")
    out_path = overlay_dir / out_name
    fig.savefig(out_path, dpi=175)
    plt.close(fig)
    return f"plots/streamline_overlays/{out_name}"


def compact_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.3g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(compact_float(value) if isinstance(value, (int, float, np.number)) and not isinstance(value, str) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    stats: pd.DataFrame,
    overlay_rows: list[dict[str, str]],
) -> None:
    plot_dir = out_dir / "plots"
    selected.to_csv(out_dir / "gabor_nw_gp_cv_selected.csv", index=False)
    candidates.to_csv(out_dir / "gabor_nw_gp_cv_all_candidates.csv", index=False)
    stats.to_csv(out_dir / "dataset_summary.csv", index=False)

    summary = (
        selected.groupby(["estimator", "model", "kernel"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
        )
        .sort_values("mean_test_rss_mean")
    )
    summary.to_csv(out_dir / "gabor_nw_gp_summary.csv", index=False)

    pivot = selected.pivot_table(index=["dataset", "kernel"], columns="estimator", values="mean_test_rss").reset_index()
    if {"GP/KRR", "Nadaraya-Watson"}.issubset(pivot.columns):
        pivot["gp_minus_nw_rss"] = pivot["GP/KRR"] - pivot["Nadaraya-Watson"]
    pivot.to_csv(out_dir / "gabor_nw_gp_paired_rss_by_dataset_kernel.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    colors = [
        "#7C3AED" if estimator == "GP/KRR" else "#13795B" if estimator == "Nadaraya-Watson" else "#B45309"
        for estimator in summary["estimator"]
    ]
    ax.barh(summary["model"], summary["mean_test_rss_mean"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Gabor matched model comparison")
    fig.tight_layout()
    fig.savefig(plot_dir / "gabor_nw_gp_mean_rss.png", dpi=180)
    plt.close(fig)

    order = summary["model"].tolist()
    vals = [selected.loc[selected["model"] == model, "mean_test_rss"].to_numpy() for model in order]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Held-out RSS distribution by selected model")
    fig.tight_layout()
    fig.savefig(plot_dir / "gabor_nw_gp_rss_boxplot.png", dpi=180)
    plt.close(fig)

    if "gp_minus_nw_rss" in pivot.columns:
        paired_summary = (
            pivot.dropna(subset=["gp_minus_nw_rss"])
            .groupby("kernel", as_index=False)["gp_minus_nw_rss"]
            .mean()
        )
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        ax.barh(paired_summary["kernel"], paired_summary["gp_minus_nw_rss"], color="#5E6AD2")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("mean RSS difference (GP/KRR - NW)")
        ax.set_title("Paired held-out RSS difference by kernel")
        fig.tight_layout()
        fig.savefig(plot_dir / "gabor_gp_minus_nw_rss.png", dpi=180)
        plt.close(fig)

    winners = selected.loc[selected.groupby("dataset")["mean_test_rss"].idxmin(), "model"].value_counts().reset_index()
    winners.columns = ["model", "wins"]
    winners.to_csv(out_dir / "gabor_nw_gp_winner_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.barh(winners["model"], winners["wins"], color="#13795B")
    ax.invert_yaxis()
    ax.set_xlabel("datasets")
    ax.set_title("Dataset-level winners")
    fig.tight_layout()
    fig.savefig(plot_dir / "gabor_nw_gp_winner_counts.png", dpi=180)
    plt.close(fig)

    md = [
        "# Gabor Matched Model Comparison",
        "",
        "The parametric, Nadaraya-Watson, and GP/KRR rows are scored on the same shuffled held-out folds using a fixed random seed. The kernel estimators smooth the local Gabor pitch angle on the doubled-angle embedding and reconstruct the streamline direction as `atan2(y, x) + alpha-prime`.",
        "",
        "The GP/KRR rows use the posterior-mean/kernel-ridge form `K_* (K + sigma_n^2 I)^{-1} U`. For the uniform kernel this should be read as regularized kernel regression rather than a strictly valid GP covariance model.",
        "",
        "## Selected Model Summary",
        "",
        markdown_table(summary),
        "",
        "## Paired RSS Difference",
        "",
        markdown_table(pivot),
        "",
        "## Plots",
        "",
        "![Mean RSS](plots/gabor_nw_gp_mean_rss.png)",
        "",
        "![RSS boxplot](plots/gabor_nw_gp_rss_boxplot.png)",
        "",
        "![GP minus NW RSS](plots/gabor_gp_minus_nw_rss.png)",
        "",
        "![Winners](plots/gabor_nw_gp_winner_counts.png)",
        "",
    ]
    for row in overlay_rows:
        md.extend([f"### {row['dataset']}", "", f"![{row['dataset']}]({row['image']})", ""])
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def run(vf_dir: Path, out_dir: Path, max_overlays: int, n_folds: int) -> None:
    plot_dir = out_dir / "plots"
    overlay_dir = plot_dir / "streamline_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(vf_dir.glob("*_rings_coords.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_rings_coords.csv files found in {vf_dir}")

    all_candidate_rows: list[dict[str, float | int | str]] = []
    all_selected_rows: list[dict[str, float | int | str]] = []
    stats_rows: list[dict[str, float | int | str]] = []
    overlay_rows: list[dict[str, str]] = []

    print(f"Comparing NW and GP/KRR kernel regression for {len(csvs)} Gabor vf_exports datasets", flush=True)
    kernels = ["rbf_pitch", "uniform_pitch", "multiplicative_pitch"]
    for csv_idx, csv_path in enumerate(csvs, start=1):
        dataset = csv_path.name
        X, alpha, phi, stats = load_gabor_vf(csv_path)
        stats_rows.append(stats)
        splits = cv_splits(len(phi), n_folds)
        print(f"[{csv_idx}/{len(csvs)}] {dataset}: n={len(phi)}", flush=True)

        dataset_kernel_rows: list[dict[str, float | int | str]] = []
        for kernel in kernels:
            cfgs = fixed_grid(kernel, DEFAULT_BANDWIDTHS, DEFAULT_KAPPAS)
            nw_rows = [score_nw_cfg(dataset, X, alpha, phi, cfg, splits) for cfg in cfgs]
            gp_rows: list[dict[str, float | int | str]] = []
            for cfg in cfgs:
                gp_rows.extend(score_gp_cfg(dataset, X, alpha, phi, cfg, DEFAULT_SIGMAS, splits))
            dataset_kernel_rows.extend(nw_rows)
            dataset_kernel_rows.extend(gp_rows)
            selected_now = best_by_kernel(nw_rows + gp_rows)
            for row in selected_now:
                if row["kernel"] == kernel:
                    sigma = "" if pd.isna(row["sigma_n"]) else f", sigma={float(row['sigma_n']):g}"
                    kappa = "" if pd.isna(row["kappa"]) else f", kappa={float(row['kappa']):g}"
                    print(f"  {row['model']}: RSS={float(row['mean_test_rss']):.4f}, h={float(row['bandwidth']):.3g}{kappa}{sigma}", flush=True)

        kernel_selected_rows = best_by_kernel(dataset_kernel_rows)
        param_rows = score_parametric_cfgs(dataset, X, phi, splits)
        dataset_rows = dataset_kernel_rows + param_rows
        selected_rows = kernel_selected_rows + param_rows
        all_candidate_rows.extend(dataset_rows)
        all_selected_rows.extend(selected_rows)
        selected_df = pd.DataFrame(selected_rows)
        if len(overlay_rows) < max_overlays:
            image = make_overlay_plot(dataset, X, alpha, phi, selected_df, overlay_dir)
            overlay_rows.append({"dataset": dataset, "image": image})

    selected = pd.DataFrame(all_selected_rows)
    candidates = pd.DataFrame(all_candidate_rows)
    stats = pd.DataFrame(stats_rows)
    write_outputs(out_dir, selected, candidates, stats, overlay_rows)
    print(f"\nWrote outputs to {out_dir}", flush=True)
    print(pd.read_csv(out_dir / "gabor_nw_gp_summary.csv").to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gabor Nadaraya-Watson and GP/KRR kernel regression with RBF, uniform, and RBF-von-Mises kernels.")
    parser.add_argument("--vf-dir", type=Path, default=DEFAULT_VF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-overlays", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=20)
    args = parser.parse_args()
    run(args.vf_dir.resolve(), args.out_dir.resolve(), args.max_overlays, args.n_folds)


if __name__ == "__main__":
    main()
