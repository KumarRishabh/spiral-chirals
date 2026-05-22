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
from sklearn.model_selection import KFold

from cleaned_linefield_comparison import (
    FIXED_PS,
    axial_mae_deg,
    axial_residual,
    axial_rss,
    embedding_to_angle,
    fit_parametric_cv,
    fit_parametric_full,
    line_embedding,
    parametric_predict,
)


ROOT = Path(__file__).resolve().parents[1]
VF_DIR = ROOT / "vf_exports"
OUT_DIR = ROOT / "data" / "vf_exports_kernel_smoother_comparison"
PLOT_DIR = OUT_DIR / "plots"
OVERLAY_DIR = PLOT_DIR / "streamline_overlays"

# Fixed, non-data-adaptive grids in the vf_exports pixel coordinate scale.
FIXED_BANDWIDTHS = np.geomspace(1.0, 1500.0, 72)
FIXED_KAPPAS = np.array(
    [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    dtype=float,
)


@dataclass(frozen=True)
class SmootherCfg:
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


def load_gabor_vf(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float | int | str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"Coordinate", "Angle (α′)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    coords = np.asarray(df["Coordinate"].apply(parse_coord).tolist(), dtype=float)
    alpha_deg = pd.to_numeric(df["Angle (α′)"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(alpha_deg)
    X = coords[valid].astype(float)
    alpha_rad = np.deg2rad(alpha_deg[valid])
    theta = np.arctan2(X[:, 1], X[:, 0])
    phi = theta + alpha_rad
    stats: dict[str, float | int | str] = {
        "dataset": csv_path.name,
        "n_raw": int(len(df)),
        "n_used": int(len(phi)),
        "n_dropped": int(len(df) - len(phi)),
        "x_min": float(np.min(X[:, 0])) if len(X) else np.nan,
        "x_max": float(np.max(X[:, 0])) if len(X) else np.nan,
        "y_min": float(np.min(X[:, 1])) if len(X) else np.nan,
        "y_max": float(np.max(X[:, 1])) if len(X) else np.nan,
    }
    return X, phi.astype(float), stats


def fixed_grid(name: str) -> list[SmootherCfg]:
    if name == "rbf":
        return [SmootherCfg(name, float(bw)) for bw in FIXED_BANDWIDTHS]
    if name == "multiplicative_rbf_vm":
        return [SmootherCfg(name, float(bw), float(kappa)) for bw in FIXED_BANDWIDTHS for kappa in FIXED_KAPPAS]
    raise ValueError(name)


def model_label(cfg: SmootherCfg) -> str:
    if cfg.name == "rbf":
        return "RBF kernel smoother"
    if cfg.name == "multiplicative_rbf_vm":
        return "RBF-von-Mises kernel smoother"
    raise ValueError(cfg.name)


def pairwise_sqdist(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = X1[:, None, :] - X2[None, :, :]
    return np.sum(diff * diff, axis=2)


def kernel_log_weights(cfg: SmootherCfg, Xtarget: NDArray[np.float64], Xsample: NDArray[np.float64]) -> NDArray[np.float64]:
    h2 = cfg.bandwidth * cfg.bandwidth
    logw = -0.5 * pairwise_sqdist(Xtarget, Xsample) / h2
    if cfg.name == "multiplicative_rbf_vm":
        if cfg.kappa is None:
            raise ValueError("multiplicative_rbf_vm needs kappa")
        target_theta = np.arctan2(Xtarget[:, 1], Xtarget[:, 0])
        sample_theta = np.arctan2(Xsample[:, 1], Xsample[:, 0])
        logw = logw + cfg.kappa * np.cos(target_theta[:, None] - sample_theta[None, :])
    elif cfg.name != "rbf":
        raise ValueError(cfg.name)
    return logw


def kernel_smooth_embedding(
    cfg: SmootherCfg,
    Xtrain: NDArray[np.float64],
    phi_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64]:
    Ytrain = line_embedding(phi_train)
    logw = kernel_log_weights(cfg, Xtarget, Xtrain)
    logw = logw - np.max(logw, axis=1, keepdims=True)
    weights = np.exp(logw)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ Ytrain


def kernel_smooth_phi(
    cfg: SmootherCfg,
    Xtrain: NDArray[np.float64],
    phi_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64]:
    return embedding_to_angle(kernel_smooth_embedding(cfg, Xtrain, phi_train, Xtarget))


def cv_score_smoother(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: SmootherCfg,
) -> dict[str, float | str | int]:
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    fold_rss = []
    fold_mae = []
    for train_idx, test_idx in cv.split(X):
        pred = kernel_smooth_phi(cfg, X[train_idx], phi[train_idx], X[test_idx])
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


def cv_residuals_for_smoother(
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: SmootherCfg,
) -> NDArray[np.float64]:
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    residuals = []
    for train_idx, test_idx in cv.split(X):
        pred = kernel_smooth_phi(cfg, X[train_idx], phi[train_idx], X[test_idx])
        residuals.append(axial_residual(phi[test_idx], pred))
    return np.concatenate(residuals)


def best_smoother_by_cv(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfgs: list[SmootherCfg],
) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    rows = [cv_score_smoother(dataset, X, phi, cfg) for cfg in cfgs]
    best = min(rows, key=lambda row: float(row["mean_test_rss"]))
    return best, rows


def cfg_from_row(row: pd.Series) -> SmootherCfg:
    kappa = None if pd.isna(row.get("kappa", np.nan)) else float(row["kappa"])
    return SmootherCfg(str(row["kernel"]), float(row["bandwidth"]), kappa)


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
    ax.scatter(X[:, 0], X[:, 1], s=9, color="black", alpha=0.42, zorder=3)


def stream_grid(X: NDArray[np.float64], n: int = 110) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    radius = float(np.quantile(np.linalg.norm(X, axis=1), 0.985))
    pad = max(0.08 * radius, 8.0)
    xs = np.linspace(-radius - pad, radius + pad, n)
    ys = np.linspace(-radius - pad, radius + pad, n)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx * xx + yy * yy) <= (radius + 0.02 * pad) ** 2
    return xx, yy, inside


def add_streamlines(
    ax: plt.Axes,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    inside: NDArray[np.bool_],
    phi_grid: NDArray[np.float64],
    color: str,
) -> None:
    U = np.ma.array(np.cos(phi_grid).reshape(xx.shape), mask=~inside)
    V = np.ma.array(np.sin(phi_grid).reshape(xx.shape), mask=~inside)
    ax.streamplot(xx, yy, U, V, color=color, density=1.35, linewidth=0.9, arrowsize=0.5, minlength=0.08, zorder=4)


def make_overlay_plot(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    selected_rows: pd.DataFrame,
) -> str:
    xx, yy, inside = stream_grid(X)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    axes_flat = axes.ravel()

    add_observed_arrows(axes_flat[0], X, phi)
    axes_flat[0].set_title("Observed Gabor vector field")

    residuals: list[tuple[str, NDArray[np.float64], float]] = []
    colors = {"rbf": "#1F5B99", "multiplicative_rbf_vm": "#13795B"}
    for ax, (_, row) in zip(axes_flat[1:3], selected_rows.sort_values("kernel").iterrows()):
        cfg = cfg_from_row(row)
        phi_fit_grid = kernel_smooth_phi(cfg, X, phi, grid)
        resid_deg = np.rad2deg(cv_residuals_for_smoother(X, phi, cfg))
        residuals.append((model_label(cfg), resid_deg, float(row["mean_test_rss"])))
        add_observed_arrows(ax, X, phi)
        add_streamlines(ax, xx, yy, inside, phi_fit_grid, colors[cfg.name])
        title = f"{model_label(cfg)}\nCV RSS={float(row['mean_test_rss']):.3f}, h={cfg.bandwidth:.2f}"
        if cfg.kappa is not None:
            title += f", kappa={cfg.kappa:g}"
        ax.set_title(title)

    ax_hist = axes_flat[3]
    bins = np.linspace(-90, 90, 31)
    for label, resid_deg, cv_rss in residuals:
        ax_hist.hist(resid_deg, bins=bins, alpha=0.5, density=True, label=f"{label}, CV RSS={cv_rss:.3f}")
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax_hist.set_title("Held-out residuals for selected smoothers")
    ax_hist.set_xlabel("axial residual (degrees)")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.25)

    for ax in axes_flat[:3]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(dataset, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_name = dataset.replace(".csv", "_kernel_smoother_streamlines.png")
    out_path = OVERLAY_DIR / out_name
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return f"plots/streamline_overlays/{out_name}"


def plot_summary(summary: pd.DataFrame, selected: pd.DataFrame, combined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(summary["model"], summary["mean_test_rss_mean"], color=["#1F5B99", "#13795B"][: len(summary)])
    ax.invert_yaxis()
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Kernel smoother held-out reconstruction error")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kernel_smoother_mean_rss.png", dpi=180)
    plt.close(fig)

    order = combined.groupby("model")["mean_test_rss"].mean().sort_values().index.tolist()
    vals = [combined.loc[combined["model"] == model, "mean_test_rss"].to_numpy() for model in order]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Kernel smoother versus parametric families")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kernel_smoother_vs_parametric_boxplot.png", dpi=180)
    plt.close(fig)

    winner_counts = selected.loc[selected.groupby("dataset")["mean_test_rss"].idxmin(), "model"].value_counts().reset_index()
    winner_counts.columns = ["model", "wins"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(winner_counts["model"], winner_counts["wins"], color="#13795B")
    ax.invert_yaxis()
    ax.set_xlabel("datasets")
    ax.set_title("Kernel smoother winners")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kernel_smoother_winner_counts.png", dpi=180)
    plt.close(fig)


def tex_escape(value: object) -> str:
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


def compact_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.3g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def latex_table(df: pd.DataFrame, columns: list[str], headers: list[str], digits: int = 4) -> str:
    align = "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(compact_float(value, digits) if isinstance(value, (int, float, np.number)) and not isinstance(value, str) else tex_escape(value))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (int, float, np.number)) and not isinstance(value, str):
                vals.append(compact_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_reports(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    combined_summary: pd.DataFrame,
    overlay_rows: list[dict[str, str]],
) -> None:
    selected_display = selected.sort_values(["dataset", "kernel"])[["dataset", "model", "bandwidth", "kappa", "mean_test_rss", "mean_test_mae_deg"]]
    selected_display.to_csv(OUT_DIR / "selected_kernel_smoothers_by_dataset.csv", index=False)

    md = [
        "# vf_exports Kernel Smoother Comparison",
        "",
        "This analysis treats the Gabor CSV files as line-field observations. Each row gives a coordinate and local pitch angle alpha-prime; the global direction is reconstructed as `phi = atan2(y, x) + alpha-prime`.",
        "",
        "The non-parametric models are Nadaraya-Watson kernel smoothers on the doubled-angle embedding `(cos 2phi, sin 2phi)`. The two kernels are Gaussian RBF and multiplicative RBF-von-Mises. Bandwidth and kappa are selected by shuffled 10-fold held-out axial reconstruction RSS over fixed, non-data-adaptive grids.",
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
        "![Mean RSS](plots/kernel_smoother_mean_rss.png)",
        "",
        "![Smoother versus parametric](plots/kernel_smoother_vs_parametric_boxplot.png)",
        "",
        "![Smoother winners](plots/kernel_smoother_winner_counts.png)",
        "",
    ]
    for row in overlay_rows:
        md.extend([f"### {row['dataset']}", "", f"![{row['dataset']}]({row['image']})", ""])
    (OUT_DIR / "report.md").write_text("\n".join(md))

    overlay_blocks = []
    for row in overlay_rows:
        overlay_blocks.extend(
            [
            r"\begin{figure}[p]",
            r"\centering",
            rf"\includegraphics[width=0.96\linewidth]{{{tex_escape(row['image'])}}}",
            rf"\caption{{Kernel-smoother streamline overlays for \texttt{{{tex_escape(row['dataset'])}}}. Observed Gabor arrows are shown at opacity $\alpha=0.7$.}}",
            r"\end{figure}",
            ]
        )
    overlay_figs = "\n".join(overlay_blocks)
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=0.75in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{placeins}}
\graphicspath{{{{./}}}}
\title{{Gabor Vector Fields: Kernel Smoother Model Comparison}}
\author{{Rishabh Kumar}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Model}}
Each Gabor observation is a coordinate $z_i=(x_i,y_i)$ and a local pitch angle $\alpha'_i$. The global line-field angle used for fitting is
\[
  \phi_i=\operatorname{{atan2}}(y_i,x_i)+\alpha'_i.
\]
Because the data are axial line fields, the smoother is applied to the doubled-angle embedding
\[
  u_i=(\cos 2\phi_i,\sin 2\phi_i).
\]
For a target point $z$, the Nadaraya--Watson kernel smoother is
\[
  \hat u(z)=\frac{{\sum_i K(z,z_i)u_i}}{{\sum_i K(z,z_i)}},
  \qquad
  \hat\phi(z)=\frac12\operatorname{{atan2}}(\hat u_2(z),\hat u_1(z)).
\]
The RBF kernel is
\[
  K_{{\mathrm{{RBF}}}}(z,z';h)=
  \exp\left[-\frac{{\|z-z'\|^2}}{{2h^2}}\right],
\]
and the multiplicative RBF--von-Mises kernel is
\[
  K_{{\mathrm{{mult}}}}(z,z';h,\kappa)=
  \exp\left[-\frac{{\|z-z'\|^2}}{{2h^2}}\right]
  \exp\left[\kappa\cos\{{\vartheta(z)-\vartheta(z')\}}\right],
  \quad
  \vartheta(z)=\operatorname{{atan2}}(y,x).
\]

\section{{Hyperparameter Selection}}
The sweep is fixed across all datasets:
\[
  h\in \operatorname{{geomspace}}(1,1500,72),
\]
and for the multiplicative kernel
\[
  \kappa\in
  \{{0,0.125,0.25,0.5,0.75,1,1.25,1.5,2,3,4,6,8,12,16,24,32,48,64\}}.
\]
Each candidate is scored by shuffled 10-fold held-out axial reconstruction RSS using random seed 42. This is used only as an anti-overfitting check for selecting the smoothness of the streamline description.

\section{{Results}}
\begin{{table}}[h!]
\centering
\small
{latex_table(summary, ["model", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean", "bandwidth_median", "kappa_median"], ["Model", "Mean RSS", "Median RSS", "Mean MAE deg", "$h$ med", "$\\kappa$ med"])}
\caption{{Selected kernel smoother performance across the Gabor vector-field exports.}}
\end{{table}}

\begin{{table}}[h!]
\centering
\small
{latex_table(combined_summary, ["model", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean"], ["Model", "Mean RSS", "Median RSS", "Mean MAE deg"])}
\caption{{Held-out axial reconstruction comparison including the parametric spiral families.}}
\end{{table}}

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.78\linewidth]{{plots/kernel_smoother_mean_rss.png}}
\caption{{Mean held-out axial RSS for the two kernel smoothers.}}
\end{{figure}}

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.86\linewidth]{{plots/kernel_smoother_vs_parametric_boxplot.png}}
\caption{{Held-out axial RSS distribution comparing selected kernel smoothers with the parametric spiral families.}}
\end{{figure}}

\FloatBarrier
\section{{Streamline Overlays}}
{overlay_figs}

\end{{document}}
"""
    (OUT_DIR / "model.tex").write_text(tex)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "model.tex"], cwd=OUT_DIR, check=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(VF_DIR.glob("*_rings_coords.csv"))
    selected_rows: list[dict[str, float | str | int]] = []
    candidate_rows: list[dict[str, float | str | int]] = []
    param_rows: list[dict[str, float | str | int]] = []
    overlay_rows: list[dict[str, str]] = []
    load_rows: list[dict[str, float | int | str]] = []

    print(f"Kernel smoother comparison for {len(csvs)} vf_exports datasets")
    for idx, csv_path in enumerate(csvs, start=1):
        dataset = csv_path.name
        X, phi, stats = load_gabor_vf(csv_path)
        load_rows.append(stats)
        print(f"[{idx}/{len(csvs)}] {dataset}: n={len(phi)}")

        param_rows.extend(fit_parametric_cv(dataset, X, phi))

        dataset_selected = []
        for kernel_name in ["rbf", "multiplicative_rbf_vm"]:
            best, rows = best_smoother_by_cv(dataset, X, phi, fixed_grid(kernel_name))
            selected_rows.append(best)
            candidate_rows.extend(rows)
            dataset_selected.append(best)
            msg = f"  {best['model']}: RSS={float(best['mean_test_rss']):.4f}, h={float(best['bandwidth']):.3g}"
            if not pd.isna(best["kappa"]):
                msg += f", kappa={float(best['kappa']):g}"
            print(msg)

        selected_df_for_plot = pd.DataFrame(dataset_selected)
        image = make_overlay_plot(dataset, X, phi, selected_df_for_plot)
        overlay_rows.append({"dataset": dataset, "image": image})

    selected = pd.DataFrame(selected_rows)
    candidates = pd.DataFrame(candidate_rows)
    params = pd.DataFrame(param_rows)
    combined = pd.concat([selected, params], ignore_index=True, sort=False)

    selected.to_csv(OUT_DIR / "kernel_smoother_cv_selected.csv", index=False)
    candidates.to_csv(OUT_DIR / "kernel_smoother_cv_all_candidates.csv", index=False)
    params.to_csv(OUT_DIR / "parametric_cv_for_reference.csv", index=False)
    combined.to_csv(OUT_DIR / "combined_cv_kernel_smoother_parametric.csv", index=False)
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
    summary.to_csv(OUT_DIR / "kernel_smoother_summary.csv", index=False)
    combined_summary.to_csv(OUT_DIR / "combined_summary.csv", index=False)

    plot_summary(summary, selected, combined)
    write_reports(summary, selected, combined_summary, overlay_rows)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
