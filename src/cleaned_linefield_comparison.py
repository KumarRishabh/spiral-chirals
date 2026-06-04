from __future__ import annotations

from dataclasses import dataclass
from html import escape
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "cleaned_linefield_comparison"
PLOT_DIR = OUT_DIR / "plots"

FIXED_PS = [(0.0, "Parametric p=0 (Logarithmic)"), (1.0, "Parametric p=1 (Archimedean)"), (2.0, "Parametric p=2 (Fermat)")]
P_BOUNDS = (-0.999, 2.999)
GAMMA_BOUNDS = (-np.pi, np.pi)


@dataclass(frozen=True)
class KernelCfg:
    name: str
    bandwidth: float
    kappa: float | None = None


def wrap_angle(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.arctan2(np.sin(a), np.cos(a))


def axial_residual(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.arctan2(np.sin(2.0 * (y_true - y_pred)), np.cos(2.0 * (y_true - y_pred)))


def axial_rss(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    d = axial_residual(y_true, y_pred)
    return float(np.sum(d * d))


def axial_mae_deg(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    d = np.abs(axial_residual(y_true, y_pred))
    return float(np.degrees(np.mean(d)))


def line_embedding(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([np.cos(2.0 * phi), np.sin(2.0 * phi)]).astype(float)


def embedding_to_angle(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.arctan2(Y[:, 1], Y[:, 0])


def extract_raw_xy_phi(mat_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    m = loadmat(mat_path)
    segs = np.asarray(m["perpendicular_segments"], dtype=float)
    segs = segs[np.isfinite(segs).all(axis=1)]
    if len(segs) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float), np.array([np.nan, np.nan]), np.nan
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    dx = x2 - x1
    dy = -(y2 - y1)
    phi = np.arctan2(dy, dx)
    X = np.column_stack([xc, yc]).astype(float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(phi)
    X = X[mask].astype(float)
    phi = phi[mask].astype(float)
    map_path = mat_path.with_name("arrow_maps.mat")
    if map_path.exists():
        maps = loadmat(map_path)
        h, w = np.asarray(maps["LBm"]).shape
        center = np.array([w / 2.0, h / 2.0], dtype=float)
        radius = 0.39 * float(min(h, w))
    else:
        center = np.median(X, axis=0)
        radius = float(np.quantile(np.linalg.norm(X - center, axis=1), 0.9))
    return X, phi, center, radius


def central_roi_filter(X: NDArray[np.float64], center: NDArray[np.float64], radius: float) -> NDArray[np.bool_]:
    return np.linalg.norm(X - center, axis=1) <= radius


def model_coordinates(raw_X: NDArray[np.float64]) -> NDArray[np.float64]:
    X = raw_X.copy().astype(float)
    if len(X) == 0:
        return X
    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = -(X[:, 1] - np.mean(X[:, 1]))
    return np.clip(X, -1e4, 1e4)


def clean_dataset(mat_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float | int | str]]:
    raw_X, phi, center, radius = extract_raw_xy_phi(mat_path)
    keep = central_roi_filter(raw_X, center, radius)
    Xc, phic = model_coordinates(raw_X[keep]), phi[keep]
    rel = str(mat_path.relative_to(DATA_DIR))
    stats: dict[str, float | int | str] = {
        "dataset": rel,
        "n_raw": int(len(phi)),
        "n_clean": int(len(phic)),
        "retention": float(len(phic) / len(phi)) if len(phi) else 0.0,
        "roi_center_x": float(center[0]),
        "roi_center_y": float(center[1]),
        "roi_radius": float(radius),
    }
    return Xc, phic, stats


def pairwise_dist(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if not np.isfinite(X1).all() or not np.isfinite(X2).all():
        raise ValueError("kernel inputs contain non-finite coordinates")
    return cdist(X1, X2, metric="euclidean")


def kernel_matrix(cfg: KernelCfg, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
    D = pairwise_dist(X1, X2)
    if cfg.name == "gaussian_rbf":
        return np.exp(-(D * D) / (2.0 * cfg.bandwidth * cfg.bandwidth))
    if cfg.name == "uniform":
        return (D <= cfg.bandwidth).astype(float)
    if cfg.name == "multiplicative_rbf_vm":
        if cfg.kappa is None:
            raise ValueError("multiplicative kernel needs kappa")
        Kx = np.exp(-(D * D) / (2.0 * cfg.bandwidth * cfg.bandwidth))
        th1 = np.arctan2(X1[:, 1], X1[:, 0])
        th2 = np.arctan2(X2[:, 1], X2[:, 0])
        return Kx * np.exp(cfg.kappa * np.cos(th1[:, None] - th2[None, :]))
    raise ValueError(cfg.name)


def bandwidth_grid(X: NDArray[np.float64]) -> NDArray[np.float64]:
    span = max(float(np.ptp(X[:, 0])), float(np.ptp(X[:, 1])), 1.0)
    if len(X) >= 2:
        D = pairwise_dist(X, X)
        D[D == 0.0] = np.nan
        nn = float(np.nanmedian(np.nanmin(D, axis=1)))
    else:
        nn = span / 20.0
    low = max(0.5 * nn, span / 80.0, 1e-3)
    high = max(1.5 * span, 4.0 * low)
    return np.geomspace(low, high, 14)


def sweep_grid(X: NDArray[np.float64]) -> list[KernelCfg]:
    bandwidths = bandwidth_grid(X)
    kappas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    out: list[KernelCfg] = []
    for bw in bandwidths:
        out.append(KernelCfg("gaussian_rbf", float(bw)))
        out.append(KernelCfg("uniform", float(bw)))
    for bw in bandwidths:
        for k in kappas:
            out.append(KernelCfg("multiplicative_rbf_vm", float(bw), float(k)))
    return out


def sigma_grid(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    base = max(float(np.mean(np.std(Y, axis=0))), 0.05)
    return np.geomspace(base * 0.05, base * 0.9, 8)


def gp_vector_lml_and_predict(
    cfg: KernelCfg,
    sigma_n: float,
    Xtr: NDArray[np.float64],
    Ytr: NDArray[np.float64],
    Xte: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64] | None]:
    K = kernel_matrix(cfg, Xtr, Xtr)
    n, m = Ytr.shape
    base_jitter = sigma_n * sigma_n + 1e-8
    c = low = None
    for scale in [1.0, 10.0, 100.0, 1000.0]:
        try:
            c, low = cho_factor(K + (base_jitter * scale) * np.eye(n), lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            continue
    if c is None or low is None:
        return -np.inf, None
    alpha = cho_solve((c, low), Ytr, check_finite=False)
    if not np.isfinite(alpha).all():
        return -np.inf, None
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    lml = -0.5 * float(np.sum(Ytr * alpha)) - 0.5 * m * logdet - 0.5 * n * m * np.log(2.0 * np.pi)
    if not np.isfinite(lml):
        return -np.inf, None
    pred = None
    if Xte is not None:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred = kernel_matrix(cfg, Xte, Xtr) @ alpha
        if not np.isfinite(pred).all():
            return lml, None
    return float(lml), pred


def model_label(cfg: KernelCfg) -> str:
    if cfg.name == "gaussian_rbf":
        return "Gaussian RBF RKHS"
    if cfg.name == "uniform":
        return "Uniform RKHS"
    return "Multiplicative RBF-VM RKHS"


def best_kernel_by_lml(X: NDArray[np.float64], phi: NDArray[np.float64], cfgs: list[KernelCfg]) -> tuple[KernelCfg, float, float]:
    Y = line_embedding(phi)
    best = None
    for cfg in cfgs:
        for sig in sigma_grid(Y):
            lml, _ = gp_vector_lml_and_predict(cfg, float(sig), X, Y, None)
            if best is None or lml > best[2]:
                best = (cfg, float(sig), float(lml))
    assert best is not None
    return best


def best_kernel_by_cv_rss(X: NDArray[np.float64], phi: NDArray[np.float64], cfgs: list[KernelCfg]) -> tuple[KernelCfg, float, float, float]:
    Y = line_embedding(phi)
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    best = None
    for cfg in cfgs:
        for sig in sigma_grid(Y):
            fold_rss = []
            fold_mae = []
            for tr_idx, te_idx in cv.split(X):
                _, Yhat = gp_vector_lml_and_predict(cfg, float(sig), X[tr_idx], Y[tr_idx], X[te_idx])
                if Yhat is None or not np.isfinite(Yhat).all():
                    fold_rss.append(np.inf)
                    fold_mae.append(np.inf)
                    continue
                pred = embedding_to_angle(Yhat)
                fold_rss.append(axial_rss(phi[te_idx], pred) / len(te_idx))
                fold_mae.append(axial_mae_deg(phi[te_idx], pred))
            m_rss = float(np.mean(fold_rss))
            m_mae = float(np.mean(fold_mae))
            if best is None or m_rss < best[2]:
                best = (cfg, float(sig), m_rss, m_mae)
    assert best is not None
    return best


def parametric_predict(X: NDArray[np.float64], p: float, gamma: float) -> NDArray[np.float64]:
    x = X[:, 0]
    y = X[:, 1]
    r = np.maximum(np.hypot(x, y), 1e-6)
    theta = np.arctan2(y, x)
    radial = np.cos(gamma) * np.power(r, p)
    tangential = -np.sin(gamma)
    return wrap_angle(theta + np.arctan2(tangential, radial))


def parametric_rss(X: NDArray[np.float64], phi: NDArray[np.float64], p: float, gamma: float) -> float:
    return axial_rss(phi, parametric_predict(X, p, gamma))


def fit_gamma_for_p(X: NDArray[np.float64], phi: NDArray[np.float64], p: float) -> tuple[float, float]:
    grid = np.linspace(-np.pi, np.pi, 361)
    rss_vals = np.array([parametric_rss(X, phi, p, g) for g in grid])
    starts = grid[np.argsort(rss_vals)[:6]]
    best_gamma = float(starts[0])
    best_rss = float(rss_vals.min())
    for start in starts:
        res = minimize(
            lambda z: parametric_rss(X, phi, p, float(z[0])),
            x0=np.array([start], dtype=float),
            method="L-BFGS-B",
            bounds=[GAMMA_BOUNDS],
        )
        if res.success and float(res.fun) < best_rss:
            best_gamma = float(res.x[0])
            best_rss = float(res.fun)
    return best_gamma, best_rss


def fit_continuous_p(X: NDArray[np.float64], phi: NDArray[np.float64]) -> tuple[float, float, float]:
    p_grid = np.linspace(-0.95, 2.95, 79)
    starts = []
    for p in p_grid:
        gamma, rss = fit_gamma_for_p(X, phi, float(p))
        starts.append((rss, float(p), gamma))
    best_rss, best_p, best_gamma = sorted(starts, key=lambda t: t[0])[0]
    for _, p0, g0 in sorted(starts, key=lambda t: t[0])[:8]:
        res = minimize(
            lambda z: parametric_rss(X, phi, float(z[0]), float(z[1])),
            x0=np.array([p0, g0], dtype=float),
            method="L-BFGS-B",
            bounds=[P_BOUNDS, GAMMA_BOUNDS],
        )
        if res.success and float(res.fun) < best_rss:
            best_p = float(res.x[0])
            best_gamma = float(res.x[1])
            best_rss = float(res.fun)
    return best_p, best_gamma, best_rss


def bic_log_evidence(rss: float, n: int, k: int) -> tuple[float, float, float]:
    sigma2 = max(rss / n, 1e-12)
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    bic = k * np.log(n) - 2.0 * loglik
    return float(loglik), float(bic), float(-0.5 * bic)


def fit_parametric_full(dataset: str, X: NDArray[np.float64], phi: NDArray[np.float64]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for p, model in FIXED_PS:
        gamma, rss = fit_gamma_for_p(X, phi, p)
        loglik, bic, logev = bic_log_evidence(rss, len(phi), 1)
        pred = parametric_predict(X, p, gamma)
        rows.append({"dataset": dataset, "n": len(phi), "model": model, "family": "parametric_fixed_p", "p": p, "gamma": gamma, "rss": rss, "rss_per_point": rss / len(phi), "mae_deg": axial_mae_deg(phi, pred), "log_likelihood": loglik, "bic": bic, "bic_log_evidence": logev, "k_params": 1})
    p, gamma, rss = fit_continuous_p(X, phi)
    loglik, bic, logev = bic_log_evidence(rss, len(phi), 2)
    pred = parametric_predict(X, p, gamma)
    rows.append({"dataset": dataset, "n": len(phi), "model": "Parametric continuous p", "family": "parametric_continuous_p", "p": p, "gamma": gamma, "rss": rss, "rss_per_point": rss / len(phi), "mae_deg": axial_mae_deg(phi, pred), "log_likelihood": loglik, "bic": bic, "bic_log_evidence": logev, "k_params": 2})
    return rows


def fit_parametric_cv(dataset: str, X: NDArray[np.float64], phi: NDArray[np.float64]) -> list[dict[str, float | str | int]]:
    rows = []
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    specs = [(p, model, "parametric_fixed_p") for p, model in FIXED_PS] + [(None, "Parametric continuous p", "parametric_continuous_p")]
    for p_fixed, model, family in specs:
        fold_rss = []
        fold_mae = []
        fit_ps = []
        fit_gammas = []
        for tr_idx, te_idx in cv.split(X):
            if p_fixed is None:
                p, gamma, _ = fit_continuous_p(X[tr_idx], phi[tr_idx])
            else:
                p = float(p_fixed)
                gamma, _ = fit_gamma_for_p(X[tr_idx], phi[tr_idx], p)
            pred = parametric_predict(X[te_idx], p, gamma)
            fold_rss.append(axial_rss(phi[te_idx], pred) / len(te_idx))
            fold_mae.append(axial_mae_deg(phi[te_idx], pred))
            fit_ps.append(p)
            fit_gammas.append(gamma)
        rows.append({"dataset": dataset, "n": len(phi), "model": model, "family": family, "mean_test_rss": float(np.mean(fold_rss)), "sd_test_rss": float(np.std(fold_rss, ddof=1)), "mean_test_mae_deg": float(np.mean(fold_mae)), "p_median": float(np.median(fit_ps)), "gamma_median": float(np.median(fit_gammas))})
    return rows


def df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def md_to_html(md_text: str) -> str:
    def inline(s: str) -> str:
        s = escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return s

    parts = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            parts.append(f"<h1>{inline(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("## "):
            parts.append(f"<h2>{inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("### "):
            parts.append(f"<h3>{inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                parts.append(f'<p><img alt="{escape(m.group(1))}" src="{escape(m.group(2))}"></p>')
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            table = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>"
            for row in rows:
                table += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
            table += "</tbody></table>"
            parts.append(table)
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            parts.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>")
            continue
        parts.append(f"<p>{inline(line)}</p>")
        i += 1

    css = (
        "body{font-family:Arial,sans-serif;max-width:1160px;margin:24px auto;line-height:1.48}"
        "table{border-collapse:collapse;margin:12px 0 24px 0}th,td{border:1px solid #ccc;padding:6px 8px}"
        "code{background:#f2f2f2;padding:1px 4px}img{max-width:100%;border:1px solid #ddd}"
    )
    return "<!doctype html><html><head><meta charset='utf-8'><title>Cleaned Line-Field Comparison</title><style>" + css + "</style></head><body>" + "\n".join(parts) + "</body></html>"


def plot_linefield(ax, X: NDArray[np.float64], phi: NDArray[np.float64], title: str, color: str) -> None:
    if len(phi) == 0:
        return
    seg_len = max(float(np.percentile(np.ptp(X, axis=0), 40)) / 18.0, 2.0)
    u = np.column_stack([np.cos(phi), np.sin(phi)]) * seg_len
    segments = np.stack([X - u, X + u], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.9, alpha=0.75))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_cleaning_plot(mats: list[Path], stats_df: pd.DataFrame) -> None:
    examples = stats_df.sort_values("retention").head(2)["dataset"].tolist() + stats_df.sort_values("retention").tail(2)["dataset"].tolist()
    fig, axes = plt.subplots(len(examples), 2, figsize=(8, 2.3 * len(examples)))
    for row, rel in enumerate(examples):
        mat = DATA_DIR / rel
        raw_X, phi, center, radius = extract_raw_xy_phi(mat)
        keep = central_roi_filter(raw_X, center, radius)
        plot_linefield(axes[row, 0], raw_X, phi, f"Raw: {Path(rel).parts[0]}", "#999999")
        plot_linefield(axes[row, 1], raw_X[keep], phi[keep], f"Cleaned: retained {keep.mean():.0%}", "#2F6B5E")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cleaning_examples.png", dpi=180)
    plt.close(fig)


def plot_summary(evidence_counts: pd.DataFrame, rss_counts: pd.DataFrame, rss_cmp: pd.DataFrame, evidence_summary: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(evidence_counts["model"], evidence_counts["wins"], color="#4C72B0")
    axes[0].invert_yaxis()
    axes[0].set_title("Evidence-style winners")
    axes[0].set_xlabel("datasets")
    axes[1].barh(rss_counts["model"], rss_counts["wins"], color="#55A868")
    axes[1].invert_yaxis()
    axes[1].set_title("Held-out axial RSS winners")
    axes[1].set_xlabel("datasets")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "winner_counts_cleaned.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = rss_cmp.groupby("model")["mean_test_rss"].mean().sort_values().index.tolist()
    vals = [rss_cmp.loc[rss_cmp["model"] == model, "mean_test_rss"].dropna().to_numpy() for model in order]
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_title("Held-out axial RSS by model on cleaned data")
    ax.set_xlabel("mean axial test RSS per held-out point")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cv_axial_rss_boxplot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(stats_df["retention"], bins=np.linspace(0.65, 1.0, 15), color="#8172B2", alpha=0.85)
    ax.set_title("Cleaning retention by dataset")
    ax.set_xlabel("fraction of segments retained")
    ax.set_ylabel("datasets")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cleaning_retention.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = evidence_summary.sort_values("evidence_score_mean", ascending=True)
    ax.barh(top["model"], top["evidence_score_mean"], color="#4C72B0")
    ax.set_title("Mean evidence-style score by model")
    ax.set_xlabel("mean optimized score (higher is better)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mean_evidence_cleaned.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    kernel_names = ["gaussian_rbf", "uniform", "multiplicative_rbf_vm"]
    mats = sorted(DATA_DIR.glob("**/arrow_segments.mat"))

    cleaning_rows = []
    bayes_kernel_rows = []
    rss_kernel_rows = []
    param_full_rows = []
    param_cv_rows = []
    skipped_rows = []

    for mat in mats:
        X, phi, stats = clean_dataset(mat)
        cleaning_rows.append(stats)
        dataset = str(mat.relative_to(DATA_DIR))
        if len(phi) < 20:
            skipped_rows.append({"dataset": dataset, "n_clean": len(phi), "reason": "fewer than 20 cleaned segments"})
            continue
        cfgs = sweep_grid(X)
        param_full_rows.extend(fit_parametric_full(dataset, X, phi))
        param_cv_rows.extend(fit_parametric_cv(dataset, X, phi))
        for kernel_name in kernel_names:
            kernel_cfgs = [cfg for cfg in cfgs if cfg.name == kernel_name]
            b_cfg, b_sig, b_lml = best_kernel_by_lml(X, phi, kernel_cfgs)
            r_cfg, r_sig, r_rss, r_mae = best_kernel_by_cv_rss(X, phi, kernel_cfgs)
            bayes_kernel_rows.append({"dataset": dataset, "n": len(phi), "model": model_label(b_cfg), "kernel": b_cfg.name, "bandwidth": b_cfg.bandwidth, "kappa": b_cfg.kappa, "sigma_n": b_sig, "evidence_score": b_lml})
            rss_kernel_rows.append({"dataset": dataset, "n": len(phi), "model": model_label(r_cfg), "kernel": r_cfg.name, "bandwidth": r_cfg.bandwidth, "kappa": r_cfg.kappa, "sigma_n": r_sig, "mean_test_rss": r_rss, "mean_test_mae_deg": r_mae})

    stats_df = pd.DataFrame(cleaning_rows)
    stats_df.to_csv(OUT_DIR / "cleaning_stats.csv", index=False)
    pd.DataFrame(skipped_rows, columns=["dataset", "n_clean", "reason"]).to_csv(OUT_DIR / "skipped_datasets.csv", index=False)

    kernel_bayes = pd.DataFrame(bayes_kernel_rows)
    kernel_rss = pd.DataFrame(rss_kernel_rows)
    param_full = pd.DataFrame(param_full_rows)
    param_cv = pd.DataFrame(param_cv_rows)
    kernel_bayes.to_csv(OUT_DIR / "kernel_evidence_by_dataset.csv", index=False)
    kernel_rss.to_csv(OUT_DIR / "kernel_cv_rss_by_dataset.csv", index=False)
    param_full.to_csv(OUT_DIR / "parametric_full_data_scores.csv", index=False)
    param_cv.to_csv(OUT_DIR / "parametric_cv_scores.csv", index=False)

    kernel_bayes_cmp = kernel_bayes.copy()
    kernel_bayes_cmp["family"] = "kernel_gp_line_embedding_lml"
    kernel_bayes_cmp["p"] = np.nan
    param_bayes_cmp = param_full.rename(columns={"bic_log_evidence": "evidence_score"})[
        ["dataset", "n", "model", "family", "p", "evidence_score", "rss_per_point", "mae_deg", "bic"]
    ]
    evidence_cmp = pd.concat([kernel_bayes_cmp, param_bayes_cmp], ignore_index=True, sort=False)
    evidence_cmp.to_csv(OUT_DIR / "combined_evidence_scores.csv", index=False)

    kernel_rss_cmp = kernel_rss.copy()
    kernel_rss_cmp["family"] = "kernel_gp_line_embedding_cv"
    kernel_rss_cmp["p_median"] = np.nan
    rss_cmp = pd.concat([kernel_rss_cmp, param_cv], ignore_index=True, sort=False)
    rss_cmp.to_csv(OUT_DIR / "combined_cv_rss_scores.csv", index=False)

    evidence_winners = evidence_cmp.loc[evidence_cmp.groupby("dataset")["evidence_score"].idxmax()].reset_index(drop=True)
    rss_winners = rss_cmp.loc[rss_cmp.groupby("dataset")["mean_test_rss"].idxmin()].reset_index(drop=True)
    evidence_winners.to_csv(OUT_DIR / "evidence_winners.csv", index=False)
    rss_winners.to_csv(OUT_DIR / "rss_winners.csv", index=False)
    evidence_counts = evidence_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    rss_counts = rss_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    evidence_counts.to_csv(OUT_DIR / "evidence_winner_counts.csv", index=False)
    rss_counts.to_csv(OUT_DIR / "rss_winner_counts.csv", index=False)

    evidence_summary = (
        evidence_cmp.groupby(["model", "family"], dropna=False)
        .agg(datasets=("dataset", "nunique"), evidence_score_mean=("evidence_score", "mean"), evidence_score_median=("evidence_score", "median"), p_median=("p", "median"))
        .reset_index()
        .sort_values("evidence_score_mean", ascending=False)
    )
    rss_summary = (
        rss_cmp.groupby(["model", "family"], dropna=False)
        .agg(datasets=("dataset", "nunique"), mean_test_rss_mean=("mean_test_rss", "mean"), mean_test_rss_median=("mean_test_rss", "median"), mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"), p_median=("p_median", "median"))
        .reset_index()
        .sort_values("mean_test_rss_mean", ascending=True)
    )
    evidence_summary.to_csv(OUT_DIR / "evidence_summary.csv", index=False)
    rss_summary.to_csv(OUT_DIR / "rss_summary.csv", index=False)

    param_evidence_winners = param_full.loc[param_full.groupby("dataset")["bic_log_evidence"].idxmax()].reset_index(drop=True)
    param_rss_winners = param_cv.loc[param_cv.groupby("dataset")["mean_test_rss"].idxmin()].reset_index(drop=True)
    param_evidence_counts = param_evidence_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    param_rss_counts = param_rss_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    param_evidence_counts.to_csv(OUT_DIR / "parametric_internal_evidence_winner_counts.csv", index=False)
    param_rss_counts.to_csv(OUT_DIR / "parametric_internal_rss_winner_counts.csv", index=False)

    make_cleaning_plot(mats, stats_df)
    plot_summary(evidence_counts, rss_counts, rss_cmp, evidence_summary, stats_df)

    processed = int(evidence_cmp["dataset"].nunique())
    retention = stats_df["retention"].describe()
    md = []
    md.append("# Cleaned Axial Line-Field Comparison")
    md.append("")
    md.append(f"Datasets found: **{len(mats)}**")
    md.append(f"Datasets processed after cleaning: **{processed}**")
    md.append(f"Datasets skipped: **{len(skipped_rows)}**")
    md.append("")
    md.append("## Cleaning Rule")
    md.append("")
    md.append(
        "The cleaned experiment starts from the same `data/**/arrow_segments.mat` files but treats each segment as an axial line-field observation, so `phi` and `phi+pi` represent the same biological orientation. A segment is retained only if its midpoint lies inside the central circular ROI of the corresponding `arrow_maps.mat` grid. For the available `150 x 150` arrow maps this uses center `(75,75)` and radius `0.39 * 150 = 58.5`. This rule intentionally preserves the central circular line field and removes peripheral/extraneous line segments outside the circle."
    )
    md.append("")
    md.append(
        f"Across the datasets, the median retention was `{retention['50%']:.3f}`, with range `{retention['min']:.3f}` to `{retention['max']:.3f}`. The median cleaned sample size was `{stats_df['n_clean'].median():.0f}` segments. The report therefore does not switch to a manually selected subset; it defines a cleaned version of every dataset by the same central-ROI rule."
    )
    md.append("")
    md.append("## Model and Metric Changes")
    md.append("")
    md.append(
        "Because the cleaned data are line fields, the primary loss is now axial RSS: `r_i = 0.5 atan2(sin(2(phi_i-phi_hat_i)), cos(2(phi_i-phi_hat_i)))`. This resolves the sign ambiguity that previously made `phi` and `phi+pi` look maximally different under signed-angle RSS. The parametric models are unchanged geometrically: fixed `p in {0,1,2}` and continuous `p in (-1,3)` are fit by minimizing axial residuals. The kernel regressions are also made axial by fitting the doubled-angle embedding `(cos 2phi, sin 2phi)` with independent GP outputs under the same covariance matrix, then mapping predictions back to an orientation by `0.5 atan2(sin_hat, cos_hat)`."
    )
    md.append("")
    md.append(
        "The kernel bandwidth sweep is dataset-adaptive. For each cleaned dataset, let `s_X=max(range(x), range(y), 1)` and let `d_nn` be the median nearest-neighbor spacing. The 14 bandwidth candidates are `ell in geomspace(max(0.5 d_nn, s_X/80, 1e-3), max(1.5 s_X, 4 max(0.5 d_nn, s_X/80, 1e-3)), 14)`. This replaces the earlier absolute grid `geomspace(0.2, 220, 14)`, which mixed sub-pixel scales with scales larger than the cleaned field of view."
    )
    md.append("")
    md.append(
        "The RSS comparison is therefore the fairest comparison in this report: every model is trained only on the training folds and scored by held-out axial RSS. The evidence-style comparison is useful but should be read more carefully, because the kernel evidence is the optimized GP marginal likelihood of the two-dimensional doubled-angle embedding, while the parametric evidence is a BIC approximation from axial residuals. Both are reasonable evidence-style summaries of line-field fit, but they are not an exact common-prior Bayes factor calculation."
    )
    md.append("")
    md.append("## Evidence-Style Winner Counts")
    md.append(df_to_md_table(evidence_counts))
    md.append("")
    md.append("## Held-Out Axial RSS Winner Counts")
    md.append(df_to_md_table(rss_counts))
    md.append("")
    md.append("## Parametric-Only Evidence Winner Counts")
    md.append(df_to_md_table(param_evidence_counts))
    md.append("")
    md.append("## Parametric-Only RSS Winner Counts")
    md.append(df_to_md_table(param_rss_counts))
    md.append("")
    md.append("## Evidence Summary")
    md.append(df_to_md_table(evidence_summary.round(4)))
    md.append("")
    md.append("## RSS Summary")
    md.append(df_to_md_table(rss_summary.round(4)))
    md.append("")
    md.append("## Plots")
    md.append("![Cleaning examples](plots/cleaning_examples.png)")
    md.append("")
    md.append("![Cleaning retention](plots/cleaning_retention.png)")
    md.append("")
    md.append("![Winner counts](plots/winner_counts_cleaned.png)")
    md.append("")
    md.append("![Held-out axial RSS boxplot](plots/cv_axial_rss_boxplot.png)")
    md.append("")
    md.append("![Mean evidence](plots/mean_evidence_cleaned.png)")
    md.append("")
    md.append("## Artifacts")
    for name in [
        "cleaning_stats.csv",
        "kernel_evidence_by_dataset.csv",
        "kernel_cv_rss_by_dataset.csv",
        "parametric_full_data_scores.csv",
        "parametric_cv_scores.csv",
        "combined_evidence_scores.csv",
        "combined_cv_rss_scores.csv",
        "evidence_summary.csv",
        "rss_summary.csv",
        "evidence_winners.csv",
        "rss_winners.csv",
    ]:
        md.append(f"- `{name}`")

    md_text = "\n".join(md)
    (OUT_DIR / "report.md").write_text(md_text)
    (OUT_DIR / "report.html").write_text(md_to_html(md_text))


if __name__ == "__main__":
    main()
