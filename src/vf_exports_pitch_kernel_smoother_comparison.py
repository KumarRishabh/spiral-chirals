from __future__ import annotations

import ast
import os
import subprocess
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
    axial_mae_deg,
    axial_residual,
    axial_rss,
    fit_parametric_cv,
    fit_parametric_full,
    parametric_predict,
)


ROOT = Path(__file__).resolve().parents[1]
VF_DIR = ROOT / "vf_exports"
OUT_DIR = ROOT / "data" / "vf_exports_pitch_kernel_smoother_comparison"
PLOT_DIR = OUT_DIR / "plots"
OVERLAY_DIR = PLOT_DIR / "streamline_overlays"

# Table 1 / full NW Gabor sweep:
# h_j = exp(log(1.0) + j * (log(1500.0) - log(1.0)) / 71), j=0,...,71.
# RBF sweeps h only; RBF-von-Mises sweeps h crossed with FIXED_KAPPAS.
FIXED_BANDWIDTHS = np.exp(np.linspace(np.log(1.0), np.log(1500.0), 72))
FIXED_KAPPAS = np.array(
    [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    dtype=float,
)


@dataclass(frozen=True)
class PitchSmootherCfg:
    name: str
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


def fixed_grid(name: str) -> list[PitchSmootherCfg]:
    if name == "rbf_pitch":
        return [PitchSmootherCfg(name, float(h)) for h in FIXED_BANDWIDTHS]
    if name == "multiplicative_pitch":
        return [PitchSmootherCfg(name, float(h), float(kappa)) for h in FIXED_BANDWIDTHS for kappa in FIXED_KAPPAS]
    raise ValueError(name)


def model_label(cfg: PitchSmootherCfg) -> str:
    if cfg.name == "rbf_pitch":
        return "RBF pitch kernel smoother"
    if cfg.name == "multiplicative_pitch":
        return "RBF-von-Mises pitch kernel smoother"
    raise ValueError(cfg.name)


def pitch_kernel_log_weights(
    cfg: PitchSmootherCfg,
    target_r: NDArray[np.float64],
    sample_r: NDArray[np.float64],
    target_theta: NDArray[np.float64],
    sample_theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    u = (target_r[:, None] - sample_r[None, :]) / cfg.bandwidth
    logw = -0.5 * u * u
    if cfg.name == "multiplicative_pitch":
        if cfg.kappa is None:
            raise ValueError("multiplicative_pitch needs kappa")
        logw = logw + cfg.kappa * np.cos(target_theta[:, None] - sample_theta[None, :])
    elif cfg.name != "rbf_pitch":
        raise ValueError(cfg.name)
    return logw


def smooth_alpha_prime(
    cfg: PitchSmootherCfg,
    Xtrain: NDArray[np.float64],
    alpha_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64]:
    sample_r = np.linalg.norm(Xtrain, axis=1)
    target_r = np.linalg.norm(Xtarget, axis=1)
    sample_theta = np.arctan2(Xtrain[:, 1], Xtrain[:, 0])
    target_theta = np.arctan2(Xtarget[:, 1], Xtarget[:, 0])

    y = np.column_stack([np.cos(2.0 * alpha_train), np.sin(2.0 * alpha_train)])
    logw = pitch_kernel_log_weights(cfg, target_r, sample_r, target_theta, sample_theta)
    logw = logw - np.max(logw, axis=1, keepdims=True)
    weights = np.exp(logw)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    yhat = weights @ y
    return 0.5 * np.arctan2(yhat[:, 1], yhat[:, 0])


def predict_phi(cfg: PitchSmootherCfg, Xtrain: NDArray[np.float64], alpha_train: NDArray[np.float64], Xtarget: NDArray[np.float64]) -> NDArray[np.float64]:
    alpha_hat = smooth_alpha_prime(cfg, Xtrain, alpha_train, Xtarget)
    theta_target = np.arctan2(Xtarget[:, 1], Xtarget[:, 0])
    return theta_target + alpha_hat


def cv_score(dataset: str, X: NDArray[np.float64], alpha: NDArray[np.float64], phi: NDArray[np.float64], cfg: PitchSmootherCfg) -> dict[str, float | str | int]:
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    fold_rss = []
    fold_mae = []
    for train_idx, test_idx in cv.split(X):
        pred = predict_phi(cfg, X[train_idx], alpha[train_idx], X[test_idx])
        fold_rss.append(axial_rss(phi[test_idx], pred) / len(test_idx))
        fold_mae.append(axial_mae_deg(phi[test_idx], pred))
    return {
        "dataset": dataset,
        "n": int(len(phi)),
        "model": model_label(cfg),
        "kernel": cfg.name,
        "bandwidth": float(cfg.bandwidth),
        "kappa": np.nan if cfg.kappa is None else float(cfg.kappa),
        "mean_test_rss": float(np.mean(fold_rss)),
        "sd_test_rss": float(np.std(fold_rss, ddof=1)),
        "mean_test_mae_deg": float(np.mean(fold_mae)),
    }


def heldout_residuals(X: NDArray[np.float64], alpha: NDArray[np.float64], phi: NDArray[np.float64], cfg: PitchSmootherCfg) -> NDArray[np.float64]:
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    residuals = []
    for train_idx, test_idx in cv.split(X):
        pred = predict_phi(cfg, X[train_idx], alpha[train_idx], X[test_idx])
        residuals.append(axial_residual(phi[test_idx], pred))
    return np.concatenate(residuals)


def best_by_cv(dataset: str, X: NDArray[np.float64], alpha: NDArray[np.float64], phi: NDArray[np.float64], cfgs: list[PitchSmootherCfg]) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    rows = [cv_score(dataset, X, alpha, phi, cfg) for cfg in cfgs]
    best = min(rows, key=lambda row: float(row["mean_test_rss"]))
    return best, rows


def cfg_from_row(row: pd.Series) -> PitchSmootherCfg:
    kappa = None if pd.isna(row.get("kappa", np.nan)) else float(row["kappa"])
    return PitchSmootherCfg(str(row["kernel"]), float(row["bandwidth"]), kappa)


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
        alpha=0.7,
        zorder=2,
    )
    ax.scatter(X[:, 0], X[:, 1], s=8, color="black", alpha=0.32, zorder=3)


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


def make_overlay_plot(
    dataset: str,
    X: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
    selected_rows: pd.DataFrame,
    param_cv_rows: pd.DataFrame,
    param_full_rows: pd.DataFrame,
) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.8))
    axes_flat = axes.ravel()

    add_observed_arrows(axes_flat[0], X, phi)
    axes_flat[0].set_title("Observed Gabor vector field")

    residuals = []
    colors = {"multiplicative_pitch": "#13795B", "rbf_pitch": "#1F5B99"}
    for ax, (_, row) in zip(axes_flat[1:3], selected_rows.sort_values("kernel").iterrows()):
        cfg = cfg_from_row(row)
        phi_fit_obs = predict_phi(cfg, X, alpha, X)
        Xi, Yi, Ui, Vi = notebook_style_stream_grid(X, phi_fit_obs)
        add_observed_arrows(ax, X, phi)
        ax.streamplot(Xi, Yi, Ui, Vi, density=2.0, color=colors[cfg.name], linewidth=0.8, arrowsize=1.0)
        title = f"{model_label(cfg)}\nCV RSS={float(row['mean_test_rss']):.3f}, h={cfg.bandwidth:.2f}"
        if cfg.kappa is not None:
            title += f", kappa={cfg.kappa:g}"
        ax.set_title(title)
        residuals.append((model_label(cfg), np.rad2deg(heldout_residuals(X, alpha, phi, cfg)), float(row["mean_test_rss"])))

    param_color = "#B45309"
    param_panel_order = [
        "Parametric continuous p",
        "Parametric p=0 (Logarithmic)",
        "Parametric p=1 (Archimedean)",
        "Parametric p=2 (Fermat)",
    ]
    for ax, model in zip(axes_flat[3:7], param_panel_order):
        cv_row = param_cv_rows[param_cv_rows["model"] == model].iloc[0]
        full_row = param_full_rows[param_full_rows["model"] == model].iloc[0]
        phi_fit_obs = parametric_predict(X, float(full_row["p"]), float(full_row["gamma"]))
        Xi, Yi, Ui, Vi = notebook_style_stream_grid(X, phi_fit_obs)
        add_observed_arrows(ax, X, phi)
        ax.streamplot(Xi, Yi, Ui, Vi, density=2.0, color=param_color, linewidth=0.8, arrowsize=1.0)
        ax.set_title(
            f"{model}\nCV RSS={float(cv_row['mean_test_rss']):.3f}, "
            f"p={float(full_row['p']):.3f}, gamma={float(full_row['gamma']):.2f}"
        )

    ax_hist = axes_flat[7]
    bins = np.linspace(-90, 90, 31)
    for label, vals, rss in residuals:
        ax_hist.hist(vals, bins=bins, alpha=0.5, density=True, label=f"{label}, CV RSS={rss:.3f}")
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_hist.set_title("Held-out residuals for selected pitch smoothers")
    ax_hist.set_xlabel("axial residual (degrees)")
    ax_hist.set_ylabel("density")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend(fontsize=8)

    for ax in axes_flat[:7]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.suptitle(dataset, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_name = dataset.replace(".csv", "_pitch_kernel_streamlines.png")
    out_path = OVERLAY_DIR / out_name
    fig.savefig(out_path, dpi=190)
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


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(compact_float(value) if isinstance(value, (int, float, np.number)) and not isinstance(value, str) else latex_escape(value))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def plot_summary(summary: pd.DataFrame, combined_summary: pd.DataFrame, combined: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(summary["model"], summary["mean_test_rss_mean"], color=["#13795B", "#1F5B99"][: len(summary)])
    ax.invert_yaxis()
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Pitch kernel smoother held-out reconstruction error")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pitch_kernel_smoother_mean_rss.png", dpi=180)
    plt.close(fig)

    order = combined_summary["model"].tolist()
    vals = [combined.loc[combined["model"] == model, "mean_test_rss"].to_numpy() for model in order]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Pitch kernel smoother versus parametric families")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pitch_kernel_vs_parametric_boxplot.png", dpi=180)
    plt.close(fig)

    winners = selected.loc[selected.groupby("dataset")["mean_test_rss"].idxmin(), "model"].value_counts().reset_index()
    winners.columns = ["model", "wins"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(winners["model"], winners["wins"], color="#13795B")
    ax.invert_yaxis()
    ax.set_xlabel("datasets")
    ax.set_title("Pitch kernel smoother winners")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pitch_kernel_winner_counts.png", dpi=180)
    plt.close(fig)


def write_reports(summary: pd.DataFrame, combined_summary: pd.DataFrame, overlay_rows: list[dict[str, str]]) -> None:
    md = [
        "# vf_exports Pitch Kernel Smoother Comparison",
        "",
        "This analysis follows `python_notebooks/streamlines.ipynb`: the model smooths local pitch angle `alpha-prime`, reconstructs `phi = atan2(y, x) + alpha-prime`, and renders streamlines by interpolating fitted sample vectors onto a dense grid with a director-field continuity correction.",
        "",
        "Bandwidth and kappa are selected by shuffled 10-fold held-out axial reconstruction RSS over fixed, non-data-adaptive grids.",
        "",
        "## Mean held-out RSS",
        "",
        markdown_table(summary),
        "",
        "## Parametric comparison",
        "",
        markdown_table(combined_summary),
        "",
        "## Plots",
        "",
        "![Mean RSS](plots/pitch_kernel_smoother_mean_rss.png)",
        "",
        "![Pitch smoother versus parametric](plots/pitch_kernel_vs_parametric_boxplot.png)",
        "",
        "![Pitch smoother winners](plots/pitch_kernel_winner_counts.png)",
        "",
    ]
    for row in overlay_rows:
        md.extend([f"### {row['dataset']}", "", f"![{row['dataset']}]({row['image']})", ""])
    (OUT_DIR / "report.md").write_text("\n".join(md))

    figures = []
    for row in overlay_rows:
        figures.extend(
            [
                r"\begin{figure}[p]",
                r"\centering",
                rf"\includegraphics[width=0.96\linewidth]{{{latex_escape(row['image'])}}}",
                rf"\caption{{Notebook-style pitch-kernel and parametric streamline overlays for \texttt{{{latex_escape(row['dataset'])}}}. Observed Gabor arrows are shown at opacity $\alpha=0.7$.}}",
                r"\end{figure}",
            ]
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=0.75in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{placeins}}
\graphicspath{{{{./}}}}
\title{{Gabor Vector Fields: Pitch Kernel Smoother Comparison}}
\author{{Rishabh Kumar}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Model}}
Each Gabor observation supplies $z_i=(x_i,y_i)$ and local pitch $\alpha'_i$. The global line-field angle is
\[
  \phi_i=\operatorname{{atan2}}(y_i,x_i)+\alpha'_i.
\]
The smoother is applied to pitch, not directly to the global angle:
\[
  a_i=(\cos 2\alpha'_i,\sin 2\alpha'_i).
\]
For target $z$ with radius $r$ and polar angle $\vartheta$, the fitted pitch embedding is
\[
  \hat a(z)=\frac{{\sum_i K(z,z_i)a_i}}{{\sum_i K(z,z_i)}},
  \qquad
  \hat\alpha'(z)=\frac12\operatorname{{atan2}}(\hat a_2(z),\hat a_1(z)).
\]
The fitted streamline direction is then
\[
  \hat\phi(z)=\operatorname{{atan2}}(y,x)+\hat\alpha'(z).
\]
The RBF pitch kernel is
\[
  K_{{\mathrm{{RBF}}}}(z,z';h)=
  \exp\left[-\frac{{(r-r')^2}}{{2h^2}}\right],
\]
and the multiplicative pitch kernel is
\[
  K_{{\mathrm{{mult}}}}(z,z';h,\kappa)=
  \exp\left[-\frac{{(r-r')^2}}{{2h^2}}\right]
  \exp\left[\kappa\cos\{{\vartheta-\vartheta'\}}\right].
\]

\section{{Hyperparameter Selection}}
The fixed bandwidth grid contains 72 positive values from 1 to 1500, with a constant multiplicative ratio between adjacent values. The multiplicative kernel additionally sweeps
\[
  \kappa\in
  \{{0,0.125,0.25,0.5,0.75,1,1.25,1.5,2,3,4,6,8,12,16,24,32,48,64\}}.
\]
Each candidate is selected by shuffled 10-fold held-out axial reconstruction RSS. This is used as an anti-overfitting check while keeping the scientific target as streamline description.

\section{{Results}}
\begin{{table}}[h!]
\centering
\small
{latex_table(summary, ["model", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean", "bandwidth_median", "kappa_median"], ["Model", "Mean RSS", "Median RSS", "Mean MAE deg", "$h$ med", "$\\kappa$ med"])}
\caption{{Selected pitch-kernel smoother performance across the Gabor vector-field exports.}}
\end{{table}}

\begin{{table}}[h!]
\centering
\small
{latex_table(combined_summary, ["model", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean"], ["Model", "Mean RSS", "Median RSS", "Mean MAE deg"])}
\caption{{Held-out axial reconstruction comparison including the parametric spiral families.}}
\end{{table}}

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.78\linewidth]{{plots/pitch_kernel_smoother_mean_rss.png}}
\caption{{Mean held-out axial RSS for the two pitch-kernel smoothers.}}
\end{{figure}}

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.86\linewidth]{{plots/pitch_kernel_vs_parametric_boxplot.png}}
\caption{{Held-out axial RSS distribution comparing selected pitch-kernel smoothers with parametric families.}}
\end{{figure}}

\FloatBarrier
\section{{Streamline Overlays}}
{"\n".join(figures)}

\end{{document}}
"""
    (OUT_DIR / "model.tex").write_text(tex)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "model.tex"], cwd=OUT_DIR, check=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, float | str | int]] = []
    candidate_rows: list[dict[str, float | str | int]] = []
    param_rows: list[dict[str, float | str | int]] = []
    param_full_rows: list[dict[str, float | str | int]] = []
    load_rows: list[dict[str, float | int | str]] = []
    overlay_rows: list[dict[str, str]] = []

    csvs = sorted(VF_DIR.glob("*_rings_coords.csv"))
    print(f"Pitch-kernel smoother comparison for {len(csvs)} vf_exports datasets")
    for idx, csv_path in enumerate(csvs, start=1):
        dataset = csv_path.name
        X, alpha, phi, stats = load_gabor_vf(csv_path)
        load_rows.append(stats)
        print(f"[{idx}/{len(csvs)}] {dataset}: n={len(phi)}")

        dataset_param_cv = pd.DataFrame(fit_parametric_cv(dataset, X, phi))
        dataset_param_full = pd.DataFrame(fit_parametric_full(dataset, X, phi))
        param_rows.extend(dataset_param_cv.to_dict("records"))
        param_full_rows.extend(dataset_param_full.to_dict("records"))
        dataset_selected = []
        for name in ["rbf_pitch", "multiplicative_pitch"]:
            best, rows = best_by_cv(dataset, X, alpha, phi, fixed_grid(name))
            selected_rows.append(best)
            candidate_rows.extend(rows)
            dataset_selected.append(best)
            msg = f"  {best['model']}: RSS={float(best['mean_test_rss']):.4f}, h={float(best['bandwidth']):.3g}"
            if not pd.isna(best["kappa"]):
                msg += f", kappa={float(best['kappa']):g}"
            print(msg)

        image = make_overlay_plot(dataset, X, alpha, phi, pd.DataFrame(dataset_selected), dataset_param_cv, dataset_param_full)
        overlay_rows.append({"dataset": dataset, "image": image})

    selected = pd.DataFrame(selected_rows)
    candidates = pd.DataFrame(candidate_rows)
    params = pd.DataFrame(param_rows)
    params_full = pd.DataFrame(param_full_rows)
    combined = pd.concat([selected, params], ignore_index=True, sort=False)

    selected.to_csv(OUT_DIR / "pitch_kernel_smoother_cv_selected.csv", index=False)
    candidates.to_csv(OUT_DIR / "pitch_kernel_smoother_cv_all_candidates.csv", index=False)
    selected.sort_values(["dataset", "kernel"])[["dataset", "model", "bandwidth", "kappa", "mean_test_rss", "mean_test_mae_deg"]].to_csv(OUT_DIR / "selected_pitch_kernel_smoothers_by_dataset.csv", index=False)
    params.to_csv(OUT_DIR / "parametric_cv_for_reference.csv", index=False)
    params_full.to_csv(OUT_DIR / "parametric_full_for_streamlines.csv", index=False)
    combined.to_csv(OUT_DIR / "combined_cv_pitch_kernel_smoother_parametric.csv", index=False)
    pd.DataFrame(load_rows).to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    pd.DataFrame({"bandwidth": FIXED_BANDWIDTHS}).to_csv(OUT_DIR / "fixed_bandwidth_grid.csv", index=False)
    pd.DataFrame({"kappa": FIXED_KAPPAS}).to_csv(OUT_DIR / "fixed_kappa_grid.csv", index=False)

    summary = (
        selected.groupby("model", as_index=False)
        .agg(
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
            bandwidth_median=("bandwidth", "median"),
            kappa_median=("kappa", "median"),
        )
        .sort_values("mean_test_rss_mean")
    )
    combined_summary = (
        combined.groupby("model", as_index=False)
        .agg(
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
        )
        .sort_values("mean_test_rss_mean")
    )
    summary.to_csv(OUT_DIR / "pitch_kernel_smoother_summary.csv", index=False)
    combined_summary.to_csv(OUT_DIR / "combined_summary.csv", index=False)

    plot_summary(summary, combined_summary, combined, selected)
    write_reports(summary, combined_summary, overlay_rows)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
