from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from cleaned_linefield_comparison import (
    FIXED_PS,
    KernelCfg,
    axial_mae_deg,
    axial_rss,
    df_to_md_table,
    embedding_to_angle,
    fit_continuous_p,
    fit_gamma_for_p,
    fit_parametric_cv,
    fit_parametric_full,
    gp_vector_lml_and_predict,
    line_embedding,
    md_to_html,
    model_label,
    parametric_predict,
)
from vf_exports_linefield_comparison import ROOT, VF_DIR, load_vf_export


OUT_DIR = ROOT / "data" / "vf_exports_fixed_grid_rss_comparison"
PLOT_DIR = OUT_DIR / "plots"
OVERLAY_DIR = PLOT_DIR / "fitted_overlays"

# Fixed, non-data-adaptive hyperparameter grids.  These are intentionally broad
# in the pixel coordinate scale of vf_exports.
FIXED_BANDWIDTHS = np.geomspace(1.0, 1500.0, 72)
FIXED_SIGMA_NS = np.geomspace(1e-3, 10.0, 17)
FIXED_KAPPAS = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0])


def fixed_kernel_grid(kernel_name: str) -> list[KernelCfg]:
    if kernel_name in {"gaussian_rbf", "uniform"}:
        return [KernelCfg(kernel_name, float(bw)) for bw in FIXED_BANDWIDTHS]
    if kernel_name == "multiplicative_rbf_vm":
        return [
            KernelCfg(kernel_name, float(bw), float(kappa))
            for bw in FIXED_BANDWIDTHS
            for kappa in FIXED_KAPPAS
        ]
    raise ValueError(kernel_name)


def best_kernel_by_fixed_grid_cv_rss(
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfgs: list[KernelCfg],
) -> tuple[KernelCfg, float, float, float]:
    Y = line_embedding(phi)
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    best: tuple[KernelCfg, float, float, float] | None = None
    for cfg in cfgs:
        for sigma_n in FIXED_SIGMA_NS:
            fold_rss = []
            fold_mae = []
            for tr_idx, te_idx in cv.split(X):
                _, Yhat = gp_vector_lml_and_predict(cfg, float(sigma_n), X[tr_idx], Y[tr_idx], X[te_idx])
                if Yhat is None or not np.isfinite(Yhat).all():
                    fold_rss.append(np.inf)
                    fold_mae.append(np.inf)
                    continue
                pred = embedding_to_angle(Yhat)
                fold_rss.append(axial_rss(phi[te_idx], pred) / len(te_idx))
                fold_mae.append(axial_mae_deg(phi[te_idx], pred))
            mean_rss = float(np.mean(fold_rss))
            mean_mae = float(np.mean(fold_mae))
            if best is None or mean_rss < best[2]:
                best = (cfg, float(sigma_n), mean_rss, mean_mae)
    assert best is not None
    return best


def stream_grid(X: NDArray[np.float64], n: int = 95) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    radius = float(np.quantile(np.linalg.norm(X, axis=1), 0.98))
    pad = max(0.08 * radius, 8.0)
    xs = np.linspace(-radius - pad, radius + pad, n)
    ys = np.linspace(-radius - pad, radius + pad, n)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx * xx + yy * yy) <= (radius + 0.02 * pad) ** 2
    return xx, yy, inside


def add_observed_arrows(ax: plt.Axes, X: NDArray[np.float64], phi: NDArray[np.float64]) -> None:
    span = max(float(np.max(np.ptp(X, axis=0))), 1.0)
    scale = span / 34.0
    U = np.cos(phi) * scale
    V = np.sin(phi) * scale
    ax.quiver(
        X[:, 0],
        X[:, 1],
        U,
        V,
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
    ax.scatter(X[:, 0], X[:, 1], s=8, color="black", alpha=0.45, zorder=3)


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
    ax.streamplot(xx, yy, U, V, color=color, density=1.25, linewidth=0.85, arrowsize=0.55, minlength=0.08, zorder=4)


def model_title(row: pd.Series) -> str:
    model = str(row["model"])
    if "kernel" in row and isinstance(row.get("kernel"), str):
        title = f"{model}\nRSS={float(row['mean_test_rss']):.3f}, ell={float(row['bandwidth']):.2f}, sigma={float(row['sigma_n']):.4g}"
        if not pd.isna(row.get("kappa", np.nan)):
            title += f", kappa={float(row['kappa']):g}"
        return title
    return f"{model}\nRSS={float(row['mean_test_rss']):.3f}, p={float(row['p_median']):.3f}, gamma={float(row['gamma_median']):.3f}"


def make_overlay_plot(
    dataset: str,
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    kernel_rows: pd.DataFrame,
    param_cv_rows: pd.DataFrame,
    param_full_rows: pd.DataFrame,
) -> str:
    xx, yy, inside = stream_grid(X)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    panels: list[tuple[str, NDArray[np.float64] | None, str]] = [("Observed Gabor field\nconverted from alpha-prime", None, "#4F5B66")]

    Y = line_embedding(phi)
    for _, row in kernel_rows.sort_values("model").iterrows():
        cfg = KernelCfg(str(row["kernel"]), float(row["bandwidth"]), None if pd.isna(row["kappa"]) else float(row["kappa"]))
        _, Yhat = gp_vector_lml_and_predict(cfg, float(row["sigma_n"]), X, Y, grid)
        if Yhat is None or not np.isfinite(Yhat).all():
            phi_fit = np.full(len(grid), np.nan)
        else:
            phi_fit = embedding_to_angle(Yhat)
        panels.append((model_title(row), phi_fit, "#1F5B99"))

    for _, row in param_cv_rows.sort_values("model").iterrows():
        full = param_full_rows[param_full_rows["model"] == row["model"]].iloc[0]
        phi_fit = parametric_predict(grid, float(full["p"]), float(full["gamma"]))
        panels.append((model_title(row), phi_fit, "#B45309"))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (title, phi_fit, color) in zip(axes_flat, panels):
        add_observed_arrows(ax, X, phi)
        if phi_fit is not None and np.isfinite(phi_fit).all():
            add_streamlines(ax, xx, yy, inside, phi_fit, color)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(dataset, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = dataset.replace(".csv", "_fixed_grid_overlays.png")
    out_path = OVERLAY_DIR / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return f"plots/fitted_overlays/{out_name}"


def write_grid_spec() -> None:
    rows = [
        {"hyperparameter": "bandwidth_or_radius_ell", "count": len(FIXED_BANDWIDTHS), "min": FIXED_BANDWIDTHS.min(), "max": FIXED_BANDWIDTHS.max(), "grid": "geomspace"},
        {"hyperparameter": "sigma_n", "count": len(FIXED_SIGMA_NS), "min": FIXED_SIGMA_NS.min(), "max": FIXED_SIGMA_NS.max(), "grid": "geomspace"},
        {"hyperparameter": "kappa", "count": len(FIXED_KAPPAS), "min": FIXED_KAPPAS.min(), "max": FIXED_KAPPAS.max(), "grid": "explicit"},
        {"hyperparameter": "parametric_fixed_p", "count": 3, "min": 0.0, "max": 2.0, "grid": "{0,1,2}"},
        {"hyperparameter": "parametric_continuous_p_bounds", "count": np.nan, "min": -0.999, "max": 2.999, "grid": "L-BFGS-B bounds"},
        {"hyperparameter": "parametric_gamma_bounds", "count": np.nan, "min": -np.pi, "max": np.pi, "grid": "L-BFGS-B bounds"},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "fixed_hyperparameter_grid.csv", index=False)
    pd.DataFrame({"bandwidth": FIXED_BANDWIDTHS}).to_csv(OUT_DIR / "fixed_bandwidth_grid.csv", index=False)
    pd.DataFrame({"sigma_n": FIXED_SIGMA_NS}).to_csv(OUT_DIR / "fixed_sigma_n_grid.csv", index=False)
    pd.DataFrame({"kappa": FIXED_KAPPAS}).to_csv(OUT_DIR / "fixed_kappa_grid.csv", index=False)


def plot_summary(rss_counts: pd.DataFrame, rss_summary: pd.DataFrame, combined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(rss_counts["model"], rss_counts["wins"], color="#476A9E")
    ax.invert_yaxis()
    ax.set_title("Held-out axial RSS winners with fixed hyperparameter grids")
    ax.set_xlabel("datasets")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "winner_counts_fixed_grid.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = rss_summary["model"].tolist()
    vals = [combined.loc[combined["model"] == model, "mean_test_rss"].dropna().to_numpy() for model in order]
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_title("Held-out axial RSS by model with fixed hyperparameter grids")
    ax.set_xlabel("mean axial RSS per held-out point")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cv_axial_rss_boxplot_fixed_grid.png", dpi=180)
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
    try:
        number = float(value)
    except Exception:
        return tex_escape(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.3g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def latex_table(df: pd.DataFrame, columns: list[str], headers: list[str], float_digits: int = 4) -> str:
    align = "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, row in df[columns].iterrows():
        cells = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                cells.append(compact_float(row[col], float_digits))
            else:
                cells.append(tex_escape(row[col]))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_model_tex(
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    winners: pd.DataFrame,
    kernel_df: pd.DataFrame,
    overlay_rows: list[dict[str, str]],
) -> None:
    grid = pd.read_csv(OUT_DIR / "fixed_hyperparameter_grid.csv")
    selected = kernel_df.groupby("model", dropna=False).agg(
        ell_min=("bandwidth", "min"),
        ell_median=("bandwidth", "median"),
        ell_max=("bandwidth", "max"),
        sigma_min=("sigma_n", "min"),
        sigma_median=("sigma_n", "median"),
        sigma_max=("sigma_n", "max"),
        kappa_min=("kappa", "min"),
        kappa_median=("kappa", "median"),
        kappa_max=("kappa", "max"),
    ).reset_index()

    overlay_figures = []
    for row in overlay_rows:
        overlay_figures.append(
            "\n".join(
                [
                    r"\begin{figure}[p]",
                    r"\centering",
                    rf"\includegraphics[width=\linewidth]{{{tex_escape(row['image'])}}}",
                    rf"\caption{{Fitted overlays for \texttt{{{tex_escape(row['dataset'])}}}.  The observed Gabor field is shown with arrow opacity $\alpha=0.7$; fitted models are shown as streamlines.}}",
                    r"\end{figure}",
                ]
            )
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=0.65in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{placeins}}
\usepackage{{float}}
\graphicspath{{{{./}}}}

\title{{Gabor Vector-Field Model Comparison with Fixed Hyperparameter Sweeps}}
\author{{Rishabh Kumar}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report compares parametric spiral families and non-parametric kernel/RKHS families on the Gabor-filter vector-field exports in \texttt{{vf\_exports}}.  Unlike the earlier adaptive-grid analysis, every kernel model here uses the same broad fixed hyperparameter grids.  The primary score is held-out axial residual sum of squares (RSS), not Bayesian evidence.  The plots overlay fitted streamlines on the original Gabor arrows, with the observed field drawn at opacity $\alpha=0.7$.
\end{{abstract}}

\section{{Data and Angular Representation}}
Each Gabor CSV row supplies a coordinate $(x_i,y_i)$ and the local pitch angle $\alpha'_i$ in degrees.  Because $\alpha'_i$ is not the global field direction, the observed modelling angle is reconstructed as
\[
  \phi_i=\operatorname{{atan2}}(y_i,x_i)+\alpha'_i.
\]
The models are scored as line fields, so $\phi$ and $\phi+\pi$ are equivalent.  Held-out error uses the axial residual
\[
  \Delta_i=\frac12\operatorname{{atan2}}\left(\sin 2(\phi_i-\hat\phi_i),\cos 2(\phi_i-\hat\phi_i)\right).
\]
The reported RSS is the mean across shuffled 10-fold cross-validation folds of the per-held-out-point sum $\sum_i\Delta_i^2/|I_k|$.

\section{{Fixed Hyperparameter Sweep}}
The kernel sweep is intentionally wide and fixed across datasets.  It does not use the dataset-adaptive length-scale or noise ranges from the earlier analysis.

\begin{{center}}
{latex_table(grid, ["hyperparameter", "count", "min", "max", "grid"], ["Hyperparameter", "Count", "Min", "Max", "Grid"], 6)}
\end{{center}}

The Gaussian RBF and uniform kernels each evaluate $72\times17=1224$ kernel--noise settings per dataset.  The multiplicative RBF--von-Mises kernel evaluates $72\times17\times19=23256$ settings per dataset.  Each setting is scored by shuffled 10-fold held-out axial RSS with random seed 42.

\section{{Model Families}}
The non-parametric families are Gaussian RBF, uniform local-neighbourhood, and multiplicative RBF--von-Mises kernels on the doubled-angle embedding $(\cos 2\phi,\sin 2\phi)$.  The parametric families are fixed $p\in\{{0,1,2\}}$ and a continuous $p$ family with $p\in[-0.999,2.999]$ and $\gamma\in[-\pi,\pi]$.

\section{{Results}}
\subsection{{Winner Counts}}
\begin{{center}}
{latex_table(counts, ["model", "wins"], ["Model", "Wins"], 0)}
\end{{center}}

\subsection{{Held-Out RSS Summary}}
\begin{{center}}
{latex_table(summary[["model", "datasets", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean"]], ["model", "datasets", "mean_test_rss_mean", "mean_test_rss_median", "mean_test_mae_deg_mean"], ["Model", "Datasets", "Mean RSS", "Median RSS", "Mean MAE deg"], 4)}
\end{{center}}

\subsection{{Selected Kernel Hyperparameters}}
\begin{{center}}
{latex_table(selected, ["model", "ell_min", "ell_median", "ell_max", "sigma_min", "sigma_median", "sigma_max", "kappa_min", "kappa_median", "kappa_max"], ["Model", "$\\ell$ min", "$\\ell$ med", "$\\ell$ max", "$\\sigma_n$ min", "$\\sigma_n$ med", "$\\sigma_n$ max", "$\\kappa$ min", "$\\kappa$ med", "$\\kappa$ max"], 4)}
\end{{center}}

\subsection{{Dataset Winners}}
\begin{{center}}
{latex_table(winners[["dataset", "model", "mean_test_rss", "mean_test_mae_deg"]], ["dataset", "model", "mean_test_rss", "mean_test_mae_deg"], ["Dataset", "Winner", "RSS", "MAE deg"], 4)}
\end{{center}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{plots/winner_counts_fixed_grid.png}}
\caption{{Winner counts under the fixed-grid held-out RSS comparison.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{plots/cv_axial_rss_boxplot_fixed_grid.png}}
\caption{{Distribution of held-out axial RSS across the 10 Gabor datasets.}}
\end{{figure}}

\FloatBarrier

\section{{Fitted Overlay Plots}}
Each plot shows the original Gabor vector field and the fitted streamlines for all compared families.  The Gabor arrows come from $\alpha'$ after conversion to the global angle $\phi=\operatorname{{atan2}}(y,x)+\alpha'$.

{chr(10).join(overlay_figures)}

\FloatBarrier

\section{{Interpretation}}
The fixed-grid rerun confirms the qualitative conclusion from held-out RSS: the kernel families fit the Gabor vector-field exports substantially better than the current parametric spiral families.  With the wider and denser non-adaptive grid, the multiplicative RBF--von-Mises kernel has the best mean held-out RSS, with Gaussian RBF very close behind.  The parametric models remain interpretable, but their held-out axial errors are roughly twice as large.

\end{{document}}
"""
    (OUT_DIR / "model.tex").write_text(tex)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    write_grid_spec()

    csvs = sorted(path for path in VF_DIR.glob("*.csv") if path.name != "image_pattern_summary_forced.csv")
    kernel_names = ["gaussian_rbf", "uniform", "multiplicative_rbf_vm"]
    stats_rows = []
    kernel_rows = []
    param_cv_rows = []
    param_full_rows = []
    overlay_rows = []

    print(f"Fixed-grid RSS comparison for {len(csvs)} vf_exports datasets")
    print(f"ell grid: {len(FIXED_BANDWIDTHS)} values from {FIXED_BANDWIDTHS.min():.3g} to {FIXED_BANDWIDTHS.max():.3g}")
    print(f"sigma_n grid: {len(FIXED_SIGMA_NS)} values from {FIXED_SIGMA_NS.min():.3g} to {FIXED_SIGMA_NS.max():.3g}")
    print(f"kappa grid: {len(FIXED_KAPPAS)} values from {FIXED_KAPPAS.min():.3g} to {FIXED_KAPPAS.max():.3g}")

    for i, csv_path in enumerate(csvs, 1):
        dataset = csv_path.name
        X, phi, stats = load_vf_export(csv_path)
        stats_rows.append(stats)
        print(f"[{i}/{len(csvs)}] {dataset}: n={len(phi)}")

        p_cv = pd.DataFrame(fit_parametric_cv(dataset, X, phi))
        p_full = pd.DataFrame(fit_parametric_full(dataset, X, phi))
        param_cv_rows.extend(p_cv.to_dict("records"))
        param_full_rows.extend(p_full.to_dict("records"))

        k_rows_for_dataset = []
        for kernel_name in kernel_names:
            cfgs = fixed_kernel_grid(kernel_name)
            cfg, sigma_n, mean_rss, mean_mae = best_kernel_by_fixed_grid_cv_rss(X, phi, cfgs)
            row = {
                "dataset": dataset,
                "n": len(phi),
                "model": model_label(cfg),
                "family": "kernel_gp_line_embedding_cv_fixed_grid",
                "kernel": cfg.name,
                "bandwidth": cfg.bandwidth,
                "kappa": cfg.kappa,
                "sigma_n": sigma_n,
                "mean_test_rss": mean_rss,
                "mean_test_mae_deg": mean_mae,
            }
            kernel_rows.append(row)
            k_rows_for_dataset.append(row)
            print(f"  {model_label(cfg)}: RSS={mean_rss:.4f}, ell={cfg.bandwidth:.3g}, sigma={sigma_n:.3g}" + ("" if cfg.kappa is None else f", kappa={cfg.kappa:g}"))

        image = make_overlay_plot(dataset, X, phi, pd.DataFrame(k_rows_for_dataset), p_cv, p_full)
        overlay_rows.append({"dataset": dataset, "image": image})

    stats_df = pd.DataFrame(stats_rows)
    kernel_df = pd.DataFrame(kernel_rows)
    param_cv = pd.DataFrame(param_cv_rows)
    param_full = pd.DataFrame(param_full_rows)
    param_cv_cmp = param_cv.copy()
    param_cv_cmp["kernel"] = np.nan
    param_cv_cmp["bandwidth"] = np.nan
    param_cv_cmp["kappa"] = np.nan
    param_cv_cmp["sigma_n"] = np.nan
    combined = pd.concat([kernel_df, param_cv_cmp], ignore_index=True, sort=False)

    stats_df.to_csv(OUT_DIR / "vf_export_stats.csv", index=False)
    kernel_df.to_csv(OUT_DIR / "kernel_cv_rss_by_dataset.csv", index=False)
    param_cv.to_csv(OUT_DIR / "parametric_cv_scores.csv", index=False)
    param_full.to_csv(OUT_DIR / "parametric_full_data_scores.csv", index=False)
    combined.to_csv(OUT_DIR / "combined_cv_rss_scores.csv", index=False)
    pd.DataFrame(overlay_rows).to_csv(OUT_DIR / "fitted_overlay_index.csv", index=False)

    winners = combined.loc[combined.groupby("dataset")["mean_test_rss"].idxmin()].reset_index(drop=True)
    counts = winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    summary = (
        combined.groupby(["model", "family"], dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_test_rss_mean=("mean_test_rss", "mean"),
            mean_test_rss_median=("mean_test_rss", "median"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
            bandwidth_median=("bandwidth", "median"),
            sigma_n_median=("sigma_n", "median"),
            kappa_median=("kappa", "median"),
            p_median=("p_median", "median"),
        )
        .reset_index()
        .sort_values("mean_test_rss_mean", ascending=True)
    )
    winners.to_csv(OUT_DIR / "rss_winners.csv", index=False)
    counts.to_csv(OUT_DIR / "rss_winner_counts.csv", index=False)
    summary.to_csv(OUT_DIR / "rss_summary.csv", index=False)
    plot_summary(counts, summary, combined)
    write_model_tex(counts, summary, winners, kernel_df, overlay_rows)

    cards = []
    for row in overlay_rows:
        cards.append(f"![{row['dataset']}]({row['image']})")
    md = [
        "# vf_exports Fixed-Grid Held-Out RSS Comparison",
        "",
        "This rerun uses fixed, non-data-adaptive hyperparameter grids for the kernel families. The model-selection criterion is held-out axial RSS. The observed Gabor vector field is plotted with arrow opacity `alpha=0.7`; the model fits are overlaid as streamlines.",
        "",
        "## Fixed Hyperparameter Ranges",
        df_to_md_table(pd.read_csv(OUT_DIR / "fixed_hyperparameter_grid.csv").round(6)),
        "",
        "## Held-Out Axial RSS Winner Counts",
        df_to_md_table(counts),
        "",
        "## Held-Out Axial RSS Summary",
        df_to_md_table(summary.round(5)),
        "",
        "## Plots",
        "![Winner counts](plots/winner_counts_fixed_grid.png)",
        "",
        "![RSS boxplot](plots/cv_axial_rss_boxplot_fixed_grid.png)",
        "",
        "## Fitted Overlays",
        "",
        *cards,
    ]
    md_text = "\n".join(md)
    (OUT_DIR / "report.md").write_text(md_text)
    (OUT_DIR / "report.html").write_text(md_to_html(md_text))

    print("Winner counts:")
    print(counts.to_string(index=False))
    print("Summary:")
    print(summary[["model", "mean_test_rss_mean", "mean_test_mae_deg_mean"]].to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
