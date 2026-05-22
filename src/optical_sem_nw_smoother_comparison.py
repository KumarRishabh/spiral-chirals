from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold, KFold

from cleaned_linefield_comparison import (
    axial_mae_deg,
    axial_residual,
    axial_rss,
    embedding_to_angle,
    line_embedding,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "data" / "optical_sem_nw_smoother_comparison"

FIXED_BANDWIDTHS = np.geomspace(1.0, 220.0, 72)
FIXED_KAPPAS = np.array(
    [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    dtype=float,
)


@dataclass(frozen=True)
class SmootherCfg:
    name: str
    bandwidth: float
    kappa: float | None = None


def wrap_angle(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.arctan2(np.sin(a), np.cos(a))


def find_mats(data_dir: Path) -> list[Path]:
    mats = sorted(data_dir.glob("**/arrow_segments.mat"))
    mats.extend(sorted(data_dir.glob("**/arrow segments.mat")))
    seen: set[Path] = set()
    unique = []
    for mat in mats:
        resolved = mat.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(mat)
    return unique


def extract_raw_xy_phi(mat_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    mat = loadmat(mat_path)
    if "perpendicular_segments" not in mat:
        raise ValueError(f"{mat_path} does not contain perpendicular_segments")
    segs = np.asarray(mat["perpendicular_segments"], dtype=float)
    segs = segs[np.isfinite(segs).all(axis=1)]
    if len(segs) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float), np.array([np.nan, np.nan]), np.nan

    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    raw_x = 0.5 * (x1 + x2)
    raw_y = 0.5 * (y1 + y2)
    dx = x2 - x1
    dy = -(y2 - y1)
    phi = np.arctan2(dy, dx)
    raw_X = np.column_stack([raw_x, raw_y]).astype(float)

    valid = np.isfinite(raw_X).all(axis=1) & np.isfinite(phi)
    raw_X = raw_X[valid]
    phi = phi[valid]

    map_path = mat_path.with_name("arrow_maps.mat")
    if map_path.exists():
        maps = loadmat(map_path)
        key = "LBm" if "LBm" in maps else next((k for k in maps if not k.startswith("__")), None)
        if key is None:
            center = np.median(raw_X, axis=0)
            radius = float(np.quantile(np.linalg.norm(raw_X - center, axis=1), 0.9))
        else:
            h, w = np.asarray(maps[key]).shape[:2]
            center = np.array([w / 2.0, h / 2.0], dtype=float)
            radius = 0.39 * float(min(h, w))
    else:
        center = np.median(raw_X, axis=0)
        radius = float(np.quantile(np.linalg.norm(raw_X - center, axis=1), 0.9))
    return raw_X, phi.astype(float), center.astype(float), radius


def model_coordinates(raw_X: NDArray[np.float64]) -> NDArray[np.float64]:
    X = raw_X.copy().astype(float)
    if len(X) == 0:
        return X
    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = -(X[:, 1] - np.mean(X[:, 1]))
    return np.clip(X, -1e4, 1e4)


def clean_dataset(mat_path: Path, data_dir: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float | int | str]]:
    raw_X, phi, center, radius = extract_raw_xy_phi(mat_path)
    keep = np.linalg.norm(raw_X - center, axis=1) <= radius
    X = model_coordinates(raw_X[keep])
    y = phi[keep]
    rel = str(mat_path.relative_to(data_dir))
    stats: dict[str, float | int | str] = {
        "dataset": rel,
        "n_raw": int(len(phi)),
        "n_clean": int(len(y)),
        "retention": float(len(y) / len(phi)) if len(phi) else 0.0,
        "roi_center_x": float(center[0]),
        "roi_center_y": float(center[1]),
        "roi_radius": float(radius),
    }
    return X, y, stats


def pairwise_sqdist(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = X1[:, None, :] - X2[None, :, :]
    return np.sum(diff * diff, axis=2)


def kernel_weights(cfg: SmootherCfg, Xtarget: NDArray[np.float64], Xsample: NDArray[np.float64]) -> NDArray[np.float64]:
    d2 = pairwise_sqdist(Xtarget, Xsample)
    if cfg.name == "rbf":
        logw = -0.5 * d2 / (cfg.bandwidth * cfg.bandwidth)
        logw = logw - np.max(logw, axis=1, keepdims=True)
        return np.exp(logw)
    if cfg.name == "uniform":
        return (d2 <= cfg.bandwidth * cfg.bandwidth).astype(float)
    if cfg.name == "multiplicative_rbf_vm":
        if cfg.kappa is None:
            raise ValueError("multiplicative_rbf_vm requires kappa")
        logw = -0.5 * d2 / (cfg.bandwidth * cfg.bandwidth)
        target_theta = np.arctan2(Xtarget[:, 1], Xtarget[:, 0])
        sample_theta = np.arctan2(Xsample[:, 1], Xsample[:, 0])
        logw = logw + cfg.kappa * np.cos(target_theta[:, None] - sample_theta[None, :])
        logw = logw - np.max(logw, axis=1, keepdims=True)
        return np.exp(logw)
    raise ValueError(cfg.name)


def kernel_smooth_embedding(
    cfg: SmootherCfg,
    Xtrain: NDArray[np.float64],
    phi_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64]:
    weights = kernel_weights(cfg, Xtarget, Xtrain)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sum(weights, axis=1, keepdims=True)
    empty = denom[:, 0] <= 1e-14
    if np.any(empty):
        d2 = pairwise_sqdist(Xtarget[empty], Xtrain)
        nearest = np.argmin(d2, axis=1)
        weights[empty, :] = 0.0
        weights[np.where(empty)[0], nearest] = 1.0
        denom = np.sum(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(denom, 1e-14)
    Ytrain = np.nan_to_num(line_embedding(phi_train), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.einsum("ij,jk->ik", weights, Ytrain, optimize=True)
    return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)


def kernel_smooth_phi(
    cfg: SmootherCfg,
    Xtrain: NDArray[np.float64],
    phi_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64]:
    return embedding_to_angle(kernel_smooth_embedding(cfg, Xtrain, phi_train, Xtarget))


def fixed_grid(kernel_name: str) -> list[SmootherCfg]:
    if kernel_name in {"rbf", "uniform"}:
        return [SmootherCfg(kernel_name, float(h)) for h in FIXED_BANDWIDTHS]
    if kernel_name == "multiplicative_rbf_vm":
        return [SmootherCfg(kernel_name, float(h), float(k)) for h in FIXED_BANDWIDTHS for k in FIXED_KAPPAS]
    raise ValueError(kernel_name)


def model_label(cfg: SmootherCfg) -> str:
    if cfg.name == "rbf":
        return "RBF NW smoother"
    if cfg.name == "uniform":
        return "Uniform NW smoother"
    if cfg.name == "multiplicative_rbf_vm":
        return "RBF-von-Mises NW smoother"
    raise ValueError(cfg.name)


def spatial_groups(X: NDArray[np.float64], bins: int = 4) -> NDArray[np.int64]:
    x_edges = np.quantile(X[:, 0], np.linspace(0, 1, bins + 1))
    y_edges = np.quantile(X[:, 1], np.linspace(0, 1, bins + 1))
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)
    if len(x_edges) <= 2 or len(y_edges) <= 2:
        return np.zeros(len(X), dtype=np.int64)
    bx = np.clip(np.searchsorted(x_edges[1:-1], X[:, 0], side="right"), 0, len(x_edges) - 2)
    by = np.clip(np.searchsorted(y_edges[1:-1], X[:, 1], side="right"), 0, len(y_edges) - 2)
    return (bx * (len(y_edges) - 1) + by).astype(np.int64)


def cv_splits(X: NDArray[np.float64], y: NDArray[np.float64]) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    groups = spatial_groups(X)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 5:
        splitter = GroupKFold(n_splits=5)
        return [(tr, te) for tr, te in splitter.split(X, y, groups)]
    splitter = KFold(n_splits=min(10, len(y)), shuffle=True, random_state=42)
    return [(tr, te) for tr, te in splitter.split(X)]


def cv_score_smoother(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: SmootherCfg,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> dict[str, float | int | str]:
    fold_rss = []
    fold_mae = []
    for train_idx, test_idx in splits:
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
        "sd_test_rss": float(np.std(fold_rss, ddof=1)) if len(fold_rss) > 1 else 0.0,
        "mean_test_mae_deg": float(np.mean(fold_mae)),
        "n_folds": int(len(splits)),
    }


def best_smoother_by_cv(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfgs: list[SmootherCfg],
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rows = [cv_score_smoother(dataset, X, phi, cfg, splits) for cfg in cfgs]
    return min(rows, key=lambda row: float(row["mean_test_rss"])), rows


def cfg_from_row(row: pd.Series) -> SmootherCfg:
    kappa = None if pd.isna(row.get("kappa", np.nan)) else float(row["kappa"])
    return SmootherCfg(str(row["kernel"]), float(row["bandwidth"]), kappa)


def plot_linefield(ax: plt.Axes, X: NDArray[np.float64], phi: NDArray[np.float64], title: str, color: str) -> None:
    span = max(float(np.max(np.ptp(X, axis=0))), 1.0)
    seg_len = span / 42.0
    U = np.column_stack([np.cos(phi), np.sin(phi)]) * seg_len
    segments = np.stack([X - U, X + U], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.9, alpha=0.72))
    ax.scatter(X[:, 0], X[:, 1], s=7, color="black", alpha=0.35, zorder=3)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def stream_grid(X: NDArray[np.float64], n: int = 105) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    radius = float(np.quantile(np.linalg.norm(X, axis=1), 0.985))
    pad = max(0.08 * radius, 7.0)
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
    ax.streamplot(xx, yy, U, V, color=color, density=1.35, linewidth=0.9, arrowsize=0.45, minlength=0.08, zorder=4)


def cv_residuals(
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: SmootherCfg,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> NDArray[np.float64]:
    residuals = []
    for train_idx, test_idx in splits:
        pred = kernel_smooth_phi(cfg, X[train_idx], phi[train_idx], X[test_idx])
        residuals.append(axial_residual(phi[test_idx], pred))
    return np.concatenate(residuals)


def make_overlay_plot(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    selected_rows: pd.DataFrame,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    overlay_dir: Path,
) -> str:
    xx, yy, inside = stream_grid(X)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    rows = selected_rows.sort_values("kernel").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 11))
    axes_flat = axes.ravel()
    plot_linefield(axes_flat[0], X, phi, "Observed cleaned SEM line field", "#4F5B66")

    colors = {"multiplicative_rbf_vm": "#13795B", "rbf": "#1F5B99", "uniform": "#8A5A00"}
    for ax, (_, row) in zip(axes_flat[1:], rows.iterrows()):
        cfg = cfg_from_row(row)
        phi_grid = kernel_smooth_phi(cfg, X, phi, grid)
        plot_linefield(ax, X, phi, "", "#B8B8B8")
        add_streamlines(ax, xx, yy, inside, phi_grid, colors[cfg.name])
        title = f"{model_label(cfg)}\nRSS={float(row['mean_test_rss']):.3f}, h={cfg.bandwidth:.2f}"
        if cfg.kappa is not None:
            title += f", kappa={cfg.kappa:g}"
        title += f", MAE={float(row['mean_test_mae_deg']):.1f} deg"
        ax.set_title(title, fontsize=9)

    fig.suptitle(dataset, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    safe_name = dataset.replace("/", "__").replace(" ", "_").replace(".mat", "_nw_streamlines.png")
    out_path = overlay_dir / safe_name
    fig.savefig(out_path, dpi=185)
    plt.close(fig)
    return f"plots/streamline_overlays/{safe_name}"


def plot_summary(out_dir: Path, selected: pd.DataFrame, combined: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    summary = (
        selected.groupby(["model", "kernel"], dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
            bandwidth_median=("bandwidth", "median"),
            kappa_median=("kappa", "median"),
        )
        .reset_index()
        .sort_values("mean_test_rss_mean")
    )
    summary.to_csv(out_dir / "nw_smoother_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(summary["model"], summary["mean_test_rss_mean"], color="#2F6B5E")
    ax.invert_yaxis()
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Optical SEM NW smoother held-out reconstruction error")
    fig.tight_layout()
    fig.savefig(plot_dir / "nw_smoother_mean_rss.png", dpi=180)
    plt.close(fig)

    order = combined.groupby("model")["mean_test_rss"].mean().sort_values().index.tolist()
    vals = [combined.loc[combined["model"] == model, "mean_test_rss"].to_numpy() for model in order]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Optical SEM NW smoothers distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "nw_smoother_boxplot.png", dpi=180)
    plt.close(fig)

    winners = selected.loc[selected.groupby("dataset")["mean_test_rss"].idxmin(), "model"].value_counts().reset_index()
    winners.columns = ["model", "wins"]
    winners.to_csv(out_dir / "nw_smoother_winner_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.barh(winners["model"], winners["wins"], color="#13795B")
    ax.invert_yaxis()
    ax.set_xlabel("datasets")
    ax.set_title("Optical SEM NW smoother winners")
    fig.tight_layout()
    fig.savefig(plot_dir / "nw_smoother_winner_counts.png", dpi=180)
    plt.close(fig)


def compact_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(str(col) for col in columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(compact_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, mats: list[Path], stats: pd.DataFrame, selected: pd.DataFrame, combined: pd.DataFrame, overlay_rows: list[dict[str, str]]) -> None:
    summary = pd.read_csv(out_dir / "nw_smoother_summary.csv")
    combined_summary = (
        combined.groupby("model", dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_rss_mean")
    )

    lines = [
        "# Optical SEM Local Nadaraya-Watson Smoother Comparison",
        "",
        f"Datasets found: **{len(mats)}**",
        f"Datasets processed: **{selected['dataset'].nunique() if len(selected) else 0}**",
        "",
        "The local smoothing models are Nadaraya-Watson smoothers on the doubled-angle line embedding `(cos 2 phi, sin 2 phi)`. The three kernels are Gaussian RBF, uniform/local-neighbourhood, and multiplicative RBF-von-Mises. Hyperparameters are selected by spatially blocked held-out axial RSS.",
        "",
        "## Kernel Summary",
        "",
        markdown_table(summary),
        "",
        "## Selected NW Smoother Summary",
        "",
        markdown_table(combined_summary),
        "",
        "## Plots",
        "",
        "![Mean RSS](plots/nw_smoother_mean_rss.png)",
        "",
        "![NW test distribution](plots/nw_smoother_boxplot.png)",
        "",
        "![NW winners](plots/nw_smoother_winner_counts.png)",
    ]
    for row in overlay_rows:
        lines.extend(["", f"![{row['dataset']}]({row['image']})"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_compile_quick_tex(out_dir: Path) -> None:
    report = out_dir / "model.tex"
    summary = pd.read_csv(out_dir / "nw_smoother_summary.csv")
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{row['model']} & {compact_float(row['mean_test_rss_mean'])} & "
            f"{compact_float(row['mean_test_rss_median'])} & "
            f"{compact_float(row['mean_test_mae_deg_mean'], 2)} & "
            f"{compact_float(row['bandwidth_median'], 2)} & "
            f"{compact_float(row['kappa_median'], 3)} \\\\"
        )
    tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=0.75in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{plots/}}
\begin{document}
\section*{Optical SEM Local Nadaraya--Watson Smoother Comparison}
The optical SEM line fields are fitted with local Nadaraya--Watson smoothers on the doubled-angle embedding. Hyperparameters are selected by spatially blocked held-out axial RSS.
\begin{table}[h]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Model & Mean RSS & Median RSS & Mean MAE & Median $h$ & Median $\kappa$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Selected local NW smoother performance across optical SEM datasets.}
\end{table}
\begin{figure}[h]
\centering
\includegraphics[width=0.78\linewidth]{nw_smoother_mean_rss.png}
\caption{Mean held-out axial RSS for optical SEM local NW smoothers.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=0.86\linewidth]{nw_smoother_boxplot.png}
\caption{Held-out axial RSS distribution for the selected local NW smoothers.}
\end{figure}
\end{document}
"""
    report.write_text(tex, encoding="utf-8")
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "model.tex"], cwd=out_dir, check=False)


def run(data_dir: Path, out_dir: Path, max_overlays: int) -> None:
    plot_dir = out_dir / "plots"
    overlay_dir = plot_dir / "streamline_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mats = find_mats(data_dir)
    if not mats:
        raise FileNotFoundError(f"No arrow_segments.mat or 'arrow segments.mat' files found under {data_dir}")

    stats_rows = []
    selected_rows = []
    candidate_rows = []
    skipped_rows = []
    overlay_rows: list[dict[str, str]] = []

    for index, mat in enumerate(mats, start=1):
        dataset = str(mat.relative_to(data_dir))
        X, phi, stats = clean_dataset(mat, data_dir)
        stats_rows.append(stats)
        print(f"[{index}/{len(mats)}] {dataset}: n_clean={len(phi)}", flush=True)
        if len(phi) < 20:
            skipped_rows.append({"dataset": dataset, "n_clean": len(phi), "reason": "fewer than 20 cleaned segments"})
            continue

        splits = cv_splits(X, phi)
        for kernel_name in ["rbf", "uniform", "multiplicative_rbf_vm"]:
            best, rows = best_smoother_by_cv(dataset, X, phi, fixed_grid(kernel_name), splits)
            selected_rows.append(best)
            candidate_rows.extend(rows)

        if len(overlay_rows) < max_overlays:
            selected_df = pd.DataFrame([row for row in selected_rows if row["dataset"] == dataset])
            image = make_overlay_plot(dataset, X, phi, selected_df, splits, overlay_dir)
            overlay_rows.append({"dataset": dataset, "image": image})

    stats_df = pd.DataFrame(stats_rows)
    selected = pd.DataFrame(selected_rows)
    candidates = pd.DataFrame(candidate_rows)
    skipped = pd.DataFrame(skipped_rows, columns=["dataset", "n_clean", "reason"])

    stats_df.to_csv(out_dir / "cleaning_stats.csv", index=False)
    selected.to_csv(out_dir / "nw_smoother_cv_selected.csv", index=False)
    candidates.to_csv(out_dir / "nw_smoother_cv_all_candidates.csv", index=False)
    skipped.to_csv(out_dir / "skipped_datasets.csv", index=False)
    pd.DataFrame(overlay_rows).to_csv(out_dir / "fitted_overlay_index.csv", index=False)

    selected_display = selected.sort_values(["dataset", "kernel"])[
        ["dataset", "model", "bandwidth", "kappa", "mean_test_rss", "mean_test_mae_deg", "n_folds"]
    ]
    selected_display.to_csv(out_dir / "selected_nw_smoothers_by_dataset.csv", index=False)

    nw_cmp = selected.copy()
    nw_cmp["family"] = "local_nadaraya_watson"
    combined = nw_cmp
    combined.to_csv(out_dir / "nw_smoother_scores.csv", index=False)

    plot_summary(out_dir, selected, combined)
    write_report(out_dir, mats, stats_df, selected, combined, overlay_rows)
    maybe_compile_quick_tex(out_dir)

    summary = pd.read_csv(out_dir / "nw_smoother_summary.csv")
    print("\nSelected NW smoother summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local NW smoother comparison for optical SEM arrow_segments.mat files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-overlays", type=int, default=6)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.out_dir.resolve(), args.max_overlays)


if __name__ == "__main__":
    main()
