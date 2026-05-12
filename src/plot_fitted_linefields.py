from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from cleaned_linefield_comparison import (
    DATA_DIR,
    OUT_DIR,
    KernelCfg,
    axial_residual,
    central_roi_filter,
    clean_dataset,
    embedding_to_angle,
    extract_raw_xy_phi,
    fit_continuous_p,
    fit_gamma_for_p,
    gp_vector_lml_and_predict,
    line_embedding,
    model_coordinates,
    parametric_predict,
)


PLOT_DIR = OUT_DIR / "fitted_streamlines"


def add_line_glyphs(ax: plt.Axes, X: np.ndarray, phi: np.ndarray, color: str = "#2F6B5E") -> None:
    if len(phi) == 0:
        return
    span = max(float(np.max(np.ptp(X, axis=0))), 1.0)
    seg_len = span / 55.0
    u = np.column_stack([np.cos(phi), np.sin(phi)]) * seg_len
    segments = np.stack([X - u, X + u], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.85, alpha=0.75))


def stream_grid(center: np.ndarray, radius: float, n: int = 75) -> tuple[np.ndarray, np.ndarray]:
    pad = 0.04 * radius
    xs = np.linspace(center[0] - radius - pad, center[0] + radius + pad, n)
    ys = np.linspace(center[1] - radius - pad, center[1] + radius + pad, n)
    return np.meshgrid(xs, ys)


def add_streamlines(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, phi: np.ndarray, center: np.ndarray, radius: float, color: str) -> None:
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius
    U = np.ma.array(np.cos(phi).reshape(xx.shape), mask=~inside)
    V = np.ma.array(np.sin(phi).reshape(xx.shape), mask=~inside)
    ax.streamplot(xx, yy, U, V, color=color, density=1.25, linewidth=0.75, arrowsize=0.45, minlength=0.1)


def best_parametric_from_csv(dataset: str, X: np.ndarray, phi: np.ndarray) -> tuple[str, float, float]:
    path = OUT_DIR / "parametric_full_data_scores.csv"
    if path.exists():
        df = pd.read_csv(path)
        rows = df[df["dataset"] == dataset].sort_values("rss_per_point")
        if len(rows):
            row = rows.iloc[0]
            return str(row["model"]), float(row["p"]), float(row["gamma"])
    p, gamma, _ = fit_continuous_p(X, phi)
    return "Parametric continuous p", p, gamma


def best_kernel_from_csv(dataset: str) -> tuple[str, KernelCfg, float]:
    path = OUT_DIR / "kernel_cv_rss_by_dataset.csv"
    df = pd.read_csv(path)
    rows = df[df["dataset"] == dataset].sort_values("mean_test_rss")
    if not len(rows):
        raise ValueError(f"No kernel RSS row for {dataset}")
    row = rows.iloc[0]
    cfg = KernelCfg(str(row["kernel"]), float(row["bandwidth"]), None if pd.isna(row["kappa"]) else float(row["kappa"]))
    return str(row["model"]), cfg, float(row["sigma_n"])


def choose_examples() -> list[str]:
    stats = pd.read_csv(OUT_DIR / "cleaning_stats.csv").sort_values("dataset").reset_index(drop=True)
    picks = [
        stats.iloc[0]["dataset"],
        stats.sort_values("retention").iloc[0]["dataset"],
        stats.iloc[len(stats) // 2]["dataset"],
        stats.sort_values("n_clean", ascending=False).iloc[0]["dataset"],
    ]
    out: list[str] = []
    for item in picks:
        if item not in out:
            out.append(str(item))
    return out


def make_plot(dataset: str, index: int) -> dict[str, str | float | int]:
    mat = DATA_DIR / dataset
    raw_X, raw_phi, raw_center, radius = extract_raw_xy_phi(mat)
    keep = central_roi_filter(raw_X, raw_center, radius)
    X = model_coordinates(raw_X[keep])
    phi = raw_phi[keep]
    center = np.array([raw_center[0] - np.mean(raw_X[keep, 0]), -(raw_center[1] - np.mean(raw_X[keep, 1]))])
    kernel_model, cfg, sigma = best_kernel_from_csv(dataset)
    param_model, p, gamma = best_parametric_from_csv(dataset, X, phi)

    xx, yy = stream_grid(center, radius)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    _, Yhat = gp_vector_lml_and_predict(cfg, sigma, X, line_embedding(phi), grid)
    if Yhat is None or not np.isfinite(Yhat).all():
        raise ValueError(f"Non-finite GP prediction for {dataset}")
    phi_gp = embedding_to_angle(Yhat)
    phi_param = parametric_predict(grid, p, gamma)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    add_line_glyphs(axes[0], X, phi, "#2F6B5E")
    axes[0].set_title(f"Cleaned observations\nn={len(phi)}")
    add_streamlines(axes[1], xx, yy, phi_gp, center, radius, "#335C99")
    add_line_glyphs(axes[1], X, phi, "#BBBBBB")
    axes[1].set_title(f"{kernel_model}\nell={cfg.bandwidth:.2f}, sigma={sigma:.3f}" + ("" if cfg.kappa is None else f", kappa={cfg.kappa:g}"))
    add_streamlines(axes[2], xx, yy, phi_param, center, radius, "#9A3412")
    add_line_glyphs(axes[2], X, phi, "#BBBBBB")
    axes[2].set_title(f"{param_model}\np={p:.3f}, gamma={gamma:.3f}")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(center[0] - 1.08 * radius, center[0] + 1.08 * radius)
        ax.set_ylim(center[1] - 1.08 * radius, center[1] + 1.08 * radius)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(dataset, fontsize=9, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    name = f"{index:02d}_{dataset.replace('/', '__').replace('.mat', '')}_streamlines.png"
    out_path = PLOT_DIR / name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "dataset": dataset,
        "image": f"fitted_streamlines/{name}",
        "n_clean": int(len(phi)),
        "kernel_model": kernel_model,
        "kernel_bandwidth": float(cfg.bandwidth),
        "kernel_sigma_n": float(sigma),
        "kernel_kappa": np.nan if cfg.kappa is None else float(cfg.kappa),
        "parametric_model": param_model,
        "parametric_p": float(p),
        "parametric_gamma": float(gamma),
    }


def write_index(rows: list[dict[str, str | float | int]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article>"
            f"<img src='{row['image']}' alt='{row['dataset']}'>"
            f"<h2>{row['dataset']}</h2>"
            f"<p>Kernel: {row['kernel_model']} (ell={float(row['kernel_bandwidth']):.2f}, "
            f"sigma={float(row['kernel_sigma_n']):.3f}). Parametric: {row['parametric_model']} "
            f"(p={float(row['parametric_p']):.3f}).</p>"
            "</article>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Fitted Line-Field Streamlines</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1280px;margin:24px auto;line-height:1.45}"
        ".grid{display:grid;grid-template-columns:1fr;gap:22px}article{border:1px solid #d8dde3;padding:12px}"
        "img{width:100%;display:block;border:1px solid #edf0f2}h2{font-size:14px;word-break:break-word}"
        "p{color:#52606d}</style></head><body>"
        "<h1>Fitted Line-Field Streamline Examples</h1>"
        "<p>Each example shows the cleaned central-ROI observations, the best held-out-RSS kernel fit, and the best full-data parametric spiral fit.</p>"
        "<section class='grid'>"
        + "\n".join(cards)
        + "</section></body></html>"
    )
    (OUT_DIR / "fitted_streamlines.html").write_text(html)
    pd.DataFrame(rows).to_csv(OUT_DIR / "fitted_streamline_examples.csv", index=False)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [make_plot(dataset, i) for i, dataset in enumerate(choose_examples(), 1)]
    write_index(rows)


if __name__ == "__main__":
    main()
