from __future__ import annotations

import argparse
import os
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

from cleaned_linefield_comparison import axial_mae_deg, axial_rss, embedding_to_angle, line_embedding


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "data" / "optical_sem_nw_gp_kernel_regression_comparison"

DEFAULT_BANDWIDTHS = np.exp(np.linspace(np.log(1.0), np.log(220.0), 8))
DEFAULT_KAPPAS = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
DEFAULT_SIGMAS = np.array([0.03, 0.06, 0.10, 0.18, 0.32, 0.56, 1.0], dtype=float)


@dataclass(frozen=True)
class KernelCfg:
    kernel: str
    bandwidth: float
    kappa: float | None = None


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


def kernel_matrix(cfg: KernelCfg, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    d2 = pairwise_sqdist(X1, X2)
    if cfg.kernel == "rbf":
        return np.exp(-0.5 * d2 / (cfg.bandwidth * cfg.bandwidth))
    if cfg.kernel == "uniform":
        return (d2 <= cfg.bandwidth * cfg.bandwidth).astype(float)
    if cfg.kernel == "multiplicative_rbf_vm":
        if cfg.kappa is None:
            raise ValueError("multiplicative_rbf_vm requires kappa")
        th1 = np.arctan2(X1[:, 1], X1[:, 0])
        th2 = np.arctan2(X2[:, 1], X2[:, 0])
        radial = np.exp(-0.5 * d2 / (cfg.bandwidth * cfg.bandwidth))
        angular = np.exp(cfg.kappa * (np.cos(th1[:, None] - th2[None, :]) - 1.0))
        return radial * angular
    raise ValueError(cfg.kernel)


def model_label(estimator: str, kernel: str) -> str:
    prefix = "NW" if estimator == "nw" else "GP/KRR"
    if kernel == "rbf":
        return f"{prefix} RBF line field"
    if kernel == "uniform":
        return f"{prefix} uniform line field"
    if kernel == "multiplicative_rbf_vm":
        return f"{prefix} RBF-von-Mises line field"
    raise ValueError(kernel)


def fixed_grid(kernel: str, bandwidths: NDArray[np.float64], kappas: NDArray[np.float64]) -> list[KernelCfg]:
    if kernel in {"rbf", "uniform"}:
        return [KernelCfg(kernel, float(h)) for h in bandwidths]
    if kernel == "multiplicative_rbf_vm":
        return [KernelCfg(kernel, float(h), float(kappa)) for h in bandwidths for kappa in kappas]
    raise ValueError(kernel)


def predict_nw_phi(cfg: KernelCfg, Xtrain: NDArray[np.float64], phi_train: NDArray[np.float64], Xtarget: NDArray[np.float64]) -> NDArray[np.float64]:
    K = kernel_matrix(cfg, Xtarget, Xtrain)
    denom = np.sum(K, axis=1, keepdims=True)
    empty = denom[:, 0] <= 1e-14
    if np.any(empty):
        d2 = pairwise_sqdist(Xtarget[empty], Xtrain)
        nearest = np.argmin(d2, axis=1)
        K[empty, :] = 0.0
        K[np.where(empty)[0], nearest] = 1.0
        denom = np.sum(K, axis=1, keepdims=True)
    W = K / np.maximum(denom, 1e-14)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Yhat = W @ line_embedding(phi_train)
    if not np.isfinite(Yhat).all():
        Yhat = np.nan_to_num(Yhat, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding_to_angle(Yhat)


def predict_gp_phi(
    cfg: KernelCfg,
    sigma_n: float,
    Xtrain: NDArray[np.float64],
    phi_train: NDArray[np.float64],
    Xtarget: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    Y = line_embedding(phi_train)
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
    return embedding_to_angle(Yhat)


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
        splitter = GroupKFold(n_splits=min(10, len(unique_groups)))
        return [(tr, te) for tr, te in splitter.split(X, y, groups)]
    splitter = KFold(n_splits=min(10, len(y)), shuffle=True, random_state=42)
    return [(tr, te) for tr, te in splitter.split(X)]


def score_nw_cfg(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfg: KernelCfg,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> dict[str, float | int | str]:
    fold_rss = []
    fold_mae = []
    for train_idx, test_idx in splits:
        pred = predict_nw_phi(cfg, X[train_idx], phi[train_idx], X[test_idx])
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
            pred = predict_gp_phi(cfg, float(sigma_n), X[train_idx], phi[train_idx], X[test_idx])
            if pred is None:
                failed[idx] = True
                continue
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


def best_by_kernel(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    df = pd.DataFrame(rows)
    out = []
    for _, group in df.groupby(["dataset", "estimator", "kernel"], dropna=False):
        out.append(group.loc[group["mean_test_rss"].astype(float).idxmin()].to_dict())
    return out


def cfg_from_row(row: pd.Series) -> KernelCfg:
    kappa = None if pd.isna(row.get("kappa", np.nan)) else float(row["kappa"])
    return KernelCfg(str(row["kernel"]), float(row["bandwidth"]), kappa)


def full_fit_phi(row: pd.Series, X: NDArray[np.float64], phi: NDArray[np.float64]) -> NDArray[np.float64]:
    cfg = cfg_from_row(row)
    if row["estimator"] == "Nadaraya-Watson":
        return predict_nw_phi(cfg, X, phi, X)
    out = predict_gp_phi(cfg, float(row["sigma_n"]), X, phi, X)
    if out is None:
        return np.zeros(len(X), dtype=float)
    return out


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


def add_streamlines(ax: plt.Axes, xx: NDArray[np.float64], yy: NDArray[np.float64], inside: NDArray[np.bool_], phi_grid: NDArray[np.float64], color: str) -> None:
    U = np.ma.array(np.cos(phi_grid).reshape(xx.shape), mask=~inside)
    V = np.ma.array(np.sin(phi_grid).reshape(xx.shape), mask=~inside)
    ax.streamplot(xx, yy, U, V, color=color, density=1.25, linewidth=0.85, arrowsize=0.45, minlength=0.08, zorder=4)


def make_overlay_plot(dataset: str, X: NDArray[np.float64], phi: NDArray[np.float64], selected_rows: pd.DataFrame, overlay_dir: Path) -> str:
    xx, yy, inside = stream_grid(X)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    order = [
        ("Nadaraya-Watson", "multiplicative_rbf_vm"),
        ("GP/KRR", "multiplicative_rbf_vm"),
        ("Nadaraya-Watson", "rbf"),
        ("GP/KRR", "rbf"),
    ]
    colors = {
        ("Nadaraya-Watson", "multiplicative_rbf_vm"): "#13795B",
        ("GP/KRR", "multiplicative_rbf_vm"): "#7C3AED",
        ("Nadaraya-Watson", "rbf"): "#1F5B99",
        ("GP/KRR", "rbf"): "#5E6AD2",
    }

    fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.3))
    plot_linefield(axes[0], X, phi, "Observed cleaned SEM line field", "#4F5B66")
    for ax, key in zip(axes[1:], order):
        row = selected_rows[(selected_rows["estimator"] == key[0]) & (selected_rows["kernel"] == key[1])].iloc[0]
        cfg = cfg_from_row(row)
        if row["estimator"] == "Nadaraya-Watson":
            phi_grid = predict_nw_phi(cfg, X, phi, grid)
        else:
            phi_grid = predict_gp_phi(cfg, float(row["sigma_n"]), X, phi, grid)
            if phi_grid is None:
                phi_grid = np.zeros(len(grid), dtype=float)
        plot_linefield(ax, X, phi, "", "#B8B8B8")
        add_streamlines(ax, xx, yy, inside, phi_grid, colors[key])
        title = f"{row['model']}\nRSS={float(row['mean_test_rss']):.3f}, h={float(row['bandwidth']):.2g}"
        if not pd.isna(row["kappa"]):
            title += f", kappa={float(row['kappa']):g}"
        if not pd.isna(row["sigma_n"]):
            title += f", sigma={float(row['sigma_n']):g}"
        ax.set_title(title, fontsize=8)
    fig.suptitle(dataset, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe_name = dataset.replace("/", "__").replace(" ", "_").replace(".mat", "_nw_gp_streamlines.png")
    out_path = overlay_dir / safe_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return f"plots/streamline_overlays/{safe_name}"


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


def write_outputs(out_dir: Path, selected: pd.DataFrame, candidates: pd.DataFrame, stats: pd.DataFrame, overlay_rows: list[dict[str, str]]) -> None:
    plot_dir = out_dir / "plots"
    selected.to_csv(out_dir / "optical_nw_gp_cv_selected.csv", index=False)
    candidates.to_csv(out_dir / "optical_nw_gp_cv_all_candidates.csv", index=False)
    stats.to_csv(out_dir / "cleaning_stats.csv", index=False)

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
    summary.to_csv(out_dir / "optical_nw_gp_summary.csv", index=False)

    pivot = selected.pivot_table(index=["dataset", "kernel"], columns="estimator", values="mean_test_rss").reset_index()
    if {"GP/KRR", "Nadaraya-Watson"}.issubset(pivot.columns):
        pivot["gp_minus_nw_rss"] = pivot["GP/KRR"] - pivot["Nadaraya-Watson"]
    pivot.to_csv(out_dir / "optical_nw_gp_paired_rss_by_dataset_kernel.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#7C3AED" if estimator == "GP/KRR" else "#13795B" for estimator in summary["estimator"]]
    ax.barh(summary["model"], summary["mean_test_rss_mean"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Optical SEM NW versus GP/KRR kernel regression")
    fig.tight_layout()
    fig.savefig(plot_dir / "optical_nw_gp_mean_rss.png", dpi=180)
    plt.close(fig)

    order = summary["model"].tolist()
    vals = [selected.loc[selected["model"] == model, "mean_test_rss"].to_numpy() for model in order]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_xlabel("mean held-out axial RSS")
    ax.set_title("Optical SEM held-out RSS distribution by selected model")
    fig.tight_layout()
    fig.savefig(plot_dir / "optical_nw_gp_rss_boxplot.png", dpi=180)
    plt.close(fig)

    if "gp_minus_nw_rss" in pivot.columns:
        paired_summary = pivot.groupby("kernel", as_index=False)["gp_minus_nw_rss"].mean()
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        ax.barh(paired_summary["kernel"], paired_summary["gp_minus_nw_rss"], color="#5E6AD2")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("mean RSS difference (GP/KRR - NW)")
        ax.set_title("Optical SEM paired RSS difference by kernel")
        fig.tight_layout()
        fig.savefig(plot_dir / "optical_gp_minus_nw_rss.png", dpi=180)
        plt.close(fig)

    winners = selected.loc[selected.groupby("dataset")["mean_test_rss"].idxmin(), "model"].value_counts().reset_index()
    winners.columns = ["model", "wins"]
    winners.to_csv(out_dir / "optical_nw_gp_winner_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.barh(winners["model"], winners["wins"], color="#13795B")
    ax.invert_yaxis()
    ax.set_xlabel("datasets")
    ax.set_title("Optical SEM dataset-level winners")
    fig.tight_layout()
    fig.savefig(plot_dir / "optical_nw_gp_winner_counts.png", dpi=180)
    plt.close(fig)

    md = [
        "# Optical SEM NW versus GP/KRR Kernel Regression",
        "",
        "Both estimator classes regress the doubled-angle line-field embedding `(cos 2 phi, sin 2 phi)` and are scored by spatially blocked held-out axial RSS.",
        "",
        "The GP/KRR rows use `K_* (K + sigma_n^2 I)^{-1} U`. The uniform-kernel row is regularized kernel regression rather than a strictly valid GP covariance model.",
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
        "![Mean RSS](plots/optical_nw_gp_mean_rss.png)",
        "",
        "![RSS boxplot](plots/optical_nw_gp_rss_boxplot.png)",
        "",
        "![GP minus NW RSS](plots/optical_gp_minus_nw_rss.png)",
        "",
        "![Winners](plots/optical_nw_gp_winner_counts.png)",
        "",
    ]
    for row in overlay_rows:
        md.extend([f"### {row['dataset']}", "", f"![{row['dataset']}]({row['image']})", ""])
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def run(data_dir: Path, out_dir: Path, max_overlays: int) -> None:
    plot_dir = out_dir / "plots"
    overlay_dir = plot_dir / "streamline_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mats = find_mats(data_dir)
    if not mats:
        raise FileNotFoundError(f"No arrow_segments.mat or 'arrow segments.mat' files found under {data_dir}")

    all_candidate_rows: list[dict[str, float | int | str]] = []
    all_selected_rows: list[dict[str, float | int | str]] = []
    stats_rows: list[dict[str, float | int | str]] = []
    overlay_rows: list[dict[str, str]] = []
    kernels = ["rbf", "uniform", "multiplicative_rbf_vm"]

    print(f"Comparing NW and GP/KRR kernel regression for {len(mats)} optical SEM datasets", flush=True)
    for index, mat in enumerate(mats, start=1):
        dataset = str(mat.relative_to(data_dir))
        X, phi, stats = clean_dataset(mat, data_dir)
        stats_rows.append(stats)
        print(f"[{index}/{len(mats)}] {dataset}: n_clean={len(phi)}", flush=True)
        if len(phi) < 20:
            continue

        splits = cv_splits(X, phi)
        dataset_rows: list[dict[str, float | int | str]] = []
        for kernel in kernels:
            cfgs = fixed_grid(kernel, DEFAULT_BANDWIDTHS, DEFAULT_KAPPAS)
            nw_rows = [score_nw_cfg(dataset, X, phi, cfg, splits) for cfg in cfgs]
            gp_rows: list[dict[str, float | int | str]] = []
            for cfg in cfgs:
                gp_rows.extend(score_gp_cfg(dataset, X, phi, cfg, DEFAULT_SIGMAS, splits))
            dataset_rows.extend(nw_rows)
            dataset_rows.extend(gp_rows)
            selected_now = best_by_kernel(nw_rows + gp_rows)
            for row in selected_now:
                if row["kernel"] == kernel:
                    sigma = "" if pd.isna(row["sigma_n"]) else f", sigma={float(row['sigma_n']):g}"
                    kappa = "" if pd.isna(row["kappa"]) else f", kappa={float(row['kappa']):g}"
                    print(f"  {row['model']}: RSS={float(row['mean_test_rss']):.4f}, h={float(row['bandwidth']):.3g}{kappa}{sigma}", flush=True)

        selected_rows = best_by_kernel(dataset_rows)
        all_candidate_rows.extend(dataset_rows)
        all_selected_rows.extend(selected_rows)
        if len(overlay_rows) < max_overlays:
            image = make_overlay_plot(dataset, X, phi, pd.DataFrame(selected_rows), overlay_dir)
            overlay_rows.append({"dataset": dataset, "image": image})

    selected = pd.DataFrame(all_selected_rows)
    candidates = pd.DataFrame(all_candidate_rows)
    stats = pd.DataFrame(stats_rows)
    write_outputs(out_dir, selected, candidates, stats, overlay_rows)
    print(f"\nWrote outputs to {out_dir}", flush=True)
    print(pd.read_csv(out_dir / "optical_nw_gp_summary.csv").to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optical SEM Nadaraya-Watson and GP/KRR kernel regression with RBF, uniform, and RBF-von-Mises kernels.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-overlays", type=int, default=3)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.out_dir.resolve(), args.max_overlays)


if __name__ == "__main__":
    main()
