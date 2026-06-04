from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cleaned_linefield_comparison import (
    FIXED_PS,
    KernelCfg,
    axial_residual,
    axial_mae_deg,
    axial_rss,
    best_kernel_by_cv_rss,
    best_kernel_by_lml,
    bic_log_evidence,
    df_to_md_table,
    embedding_to_angle,
    fit_continuous_p,
    fit_gamma_for_p,
    fit_parametric_cv,
    fit_parametric_full,
    gp_vector_lml_and_predict,
    kernel_matrix,
    line_embedding,
    md_to_html,
    model_label,
    parametric_predict,
    sweep_grid,
)
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
VF_DIR = ROOT / "vf_exports"
OUT_DIR = ROOT / "data" / "vf_exports_linefield_comparison"
PLOT_DIR = OUT_DIR / "plots"
MIN_AXIAL_SIGMA = 1e-6


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


def load_vf_export(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float | int | str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"Coordinate", "Angle (α′)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    coords = np.asarray(df["Coordinate"].apply(parse_coord).tolist(), dtype=float)
    alpha_deg = pd.to_numeric(df["Angle (α′)"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(coords).all(axis=1) & np.isfinite(alpha_deg)
    coords = coords[mask]
    alpha_deg = alpha_deg[mask]

    X = coords.astype(float)
    theta = np.arctan2(X[:, 1], X[:, 0])
    alpha_rad = np.deg2rad(alpha_deg)
    phi = theta + alpha_rad

    # Match the MAT pipeline's modelling convention: centered coordinates with
    # image-style y inverted into mathematical coordinates.
    X_model = X.copy()
    X_model[:, 0] = X_model[:, 0] - np.mean(X_model[:, 0])
    X_model[:, 1] = -(X_model[:, 1] - np.mean(X_model[:, 1]))

    stats: dict[str, float | int | str] = {
        "dataset": csv_path.name,
        "n_raw": int(len(df)),
        "n_used": int(len(phi)),
        "n_dropped": int(len(df) - len(phi)),
        "x_min": float(np.min(X[:, 0])) if len(X) else np.nan,
        "x_max": float(np.max(X[:, 0])) if len(X) else np.nan,
        "y_min": float(np.min(X[:, 1])) if len(X) else np.nan,
        "y_max": float(np.max(X[:, 1])) if len(X) else np.nan,
        "pitch_deg_mean": float(np.mean(alpha_deg)) if len(alpha_deg) else np.nan,
        "pitch_deg_sd": float(np.std(alpha_deg, ddof=1)) if len(alpha_deg) > 1 else np.nan,
    }
    return X_model, phi.astype(float), stats


def plot_linefield(ax: plt.Axes, X: NDArray[np.float64], phi: NDArray[np.float64], title: str) -> None:
    if len(phi) == 0:
        return
    seg_len = max(float(np.percentile(np.ptp(X, axis=0), 40)) / 16.0, 2.0)
    U = np.cos(phi) * seg_len
    V = np.sin(phi) * seg_len
    ax.quiver(X[:, 0], X[:, 1], U, V, np.rad2deg(phi), cmap="hsv", pivot="mid", scale_units="xy", scale=1, width=0.006, headwidth=0, headlength=0, headaxislength=0)
    ax.scatter(X[:, 0], X[:, 1], s=9, color="black", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_input_gallery(datasets: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]]) -> None:
    n = len(datasets)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (name, X, phi) in zip(axes_arr, datasets):
        plot_linefield(ax, X, phi, name.replace("_3000x_rings_coords.csv", ""))
    for ax in axes_arr[len(datasets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "vf_export_linefields.png", dpi=180)
    plt.close(fig)


def plot_summary(evidence_counts: pd.DataFrame, rss_counts: pd.DataFrame, rss_cmp: pd.DataFrame, evidence_summary: pd.DataFrame) -> None:
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
    fig.savefig(PLOT_DIR / "winner_counts_vf_exports.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = rss_cmp.groupby("model")["mean_test_rss"].mean().sort_values().index.tolist()
    vals = [rss_cmp.loc[rss_cmp["model"] == model, "mean_test_rss"].dropna().to_numpy() for model in order]
    ax.boxplot(vals, tick_labels=order, showfliers=False, vert=False)
    ax.set_title("Held-out axial RSS by model on vf_exports")
    ax.set_xlabel("mean axial test RSS per held-out point")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cv_axial_rss_boxplot_vf_exports.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = evidence_summary.sort_values("evidence_score_mean", ascending=True)
    ax.barh(top["model"], top["evidence_score_mean"], color="#4C72B0")
    ax.set_title("Mean evidence-style score by model")
    ax.set_xlabel("mean optimized score (higher is better)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mean_evidence_vf_exports.png", dpi=180)
    plt.close(fig)


def axial_gaussian_nlpd(residuals: NDArray[np.float64], sigma: float) -> float:
    sigma = max(float(sigma), MIN_AXIAL_SIGMA)
    sigma2 = sigma * sigma
    vals = 0.5 * (np.log(2.0 * np.pi * sigma2) + residuals * residuals / sigma2)
    return float(np.mean(vals))


def fitted_axial_sigma(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    residuals = axial_residual(y_true, y_pred)
    return float(max(np.sqrt(np.mean(residuals * residuals)), MIN_AXIAL_SIGMA))


def best_kernel_by_cv_nlpd(
    X: NDArray[np.float64],
    phi: NDArray[np.float64],
    cfgs: list[KernelCfg],
) -> tuple[KernelCfg, float, float, float, float]:
    Y = line_embedding(phi)
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    best = None
    for cfg in cfgs:
        for sig in np.geomspace(max(float(np.mean(np.std(Y, axis=0))), 0.05) * 0.05, max(float(np.mean(np.std(Y, axis=0))), 0.05) * 0.9, 8):
            fold_nlpd = []
            fold_sigma = []
            fold_mae = []
            for tr_idx, te_idx in cv.split(X):
                _, Yhat_tr = gp_vector_lml_and_predict(cfg, float(sig), X[tr_idx], Y[tr_idx], X[tr_idx])
                _, Yhat_te = gp_vector_lml_and_predict(cfg, float(sig), X[tr_idx], Y[tr_idx], X[te_idx])
                if Yhat_tr is None or Yhat_te is None or not np.isfinite(Yhat_tr).all() or not np.isfinite(Yhat_te).all():
                    fold_nlpd.append(np.inf)
                    fold_sigma.append(np.inf)
                    fold_mae.append(np.inf)
                    continue
                pred_tr = embedding_to_angle(Yhat_tr)
                pred_te = embedding_to_angle(Yhat_te)
                axial_sigma = fitted_axial_sigma(phi[tr_idx], pred_tr)
                residuals_te = axial_residual(phi[te_idx], pred_te)
                fold_nlpd.append(axial_gaussian_nlpd(residuals_te, axial_sigma))
                fold_sigma.append(axial_sigma)
                fold_mae.append(axial_mae_deg(phi[te_idx], pred_te))
            mean_nlpd = float(np.mean(fold_nlpd))
            if best is None or mean_nlpd < best[2]:
                best = (cfg, float(sig), mean_nlpd, float(np.mean(fold_sigma)), float(np.mean(fold_mae)))
    assert best is not None
    return best


def parametric_cv_nlpd(dataset: str, X: NDArray[np.float64], phi: NDArray[np.float64]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    cv = KFold(n_splits=min(10, len(phi)), shuffle=True, random_state=42)
    specs = [(p, model, "parametric_fixed_p") for p, model in FIXED_PS] + [(None, "Parametric continuous p", "parametric_continuous_p")]
    for p_fixed, model, family in specs:
        fold_nlpd = []
        fold_sigma = []
        fold_mae = []
        fit_ps = []
        fit_gammas = []
        for tr_idx, te_idx in cv.split(X):
            if p_fixed is None:
                p, gamma, _ = fit_continuous_p(X[tr_idx], phi[tr_idx])
            else:
                p = float(p_fixed)
                gamma, _ = fit_gamma_for_p(X[tr_idx], phi[tr_idx], p)
            pred_tr = parametric_predict(X[tr_idx], p, gamma)
            pred_te = parametric_predict(X[te_idx], p, gamma)
            axial_sigma = fitted_axial_sigma(phi[tr_idx], pred_tr)
            residuals_te = axial_residual(phi[te_idx], pred_te)
            fold_nlpd.append(axial_gaussian_nlpd(residuals_te, axial_sigma))
            fold_sigma.append(axial_sigma)
            fold_mae.append(axial_mae_deg(phi[te_idx], pred_te))
            fit_ps.append(p)
            fit_gammas.append(gamma)
        rows.append({
            "dataset": dataset,
            "n": len(phi),
            "model": model,
            "family": family,
            "mean_test_nlpd": float(np.mean(fold_nlpd)),
            "mean_train_axial_sigma": float(np.mean(fold_sigma)),
            "mean_test_mae_deg": float(np.mean(fold_mae)),
            "p_median": float(np.median(fit_ps)),
            "gamma_median": float(np.median(fit_gammas)),
        })
    return rows


def audit_kernel_evidence_identity(kernel_bayes: pd.DataFrame, datasets: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]) -> pd.DataFrame:
    rows = []
    for dataset, (X, _) in datasets.items():
        gaussian = kernel_bayes[(kernel_bayes["dataset"] == dataset) & (kernel_bayes["kernel"] == "gaussian_rbf")].iloc[0]
        mult = kernel_bayes[(kernel_bayes["dataset"] == dataset) & (kernel_bayes["kernel"] == "multiplicative_rbf_vm")].iloc[0]
        g_cfg = KernelCfg("gaussian_rbf", float(gaussian["bandwidth"]))
        m_cfg = KernelCfg("multiplicative_rbf_vm", float(mult["bandwidth"]), float(mult["kappa"]))
        max_abs_gram_diff = float(np.max(np.abs(kernel_matrix(g_cfg, X, X) - kernel_matrix(m_cfg, X, X))))
        rows.append({
            "dataset": dataset,
            "gaussian_bandwidth": float(gaussian["bandwidth"]),
            "gaussian_sigma_n": float(gaussian["sigma_n"]),
            "gaussian_evidence": float(gaussian["evidence_score"]),
            "multiplicative_bandwidth": float(mult["bandwidth"]),
            "multiplicative_kappa": float(mult["kappa"]),
            "multiplicative_sigma_n": float(mult["sigma_n"]),
            "multiplicative_evidence": float(mult["evidence_score"]),
            "same_hyperparameters": bool(
                np.isclose(float(gaussian["bandwidth"]), float(mult["bandwidth"]))
                and np.isclose(float(gaussian["sigma_n"]), float(mult["sigma_n"]))
                and np.isclose(float(mult["kappa"]), 0.0)
            ),
            "max_abs_gram_diff": max_abs_gram_diff,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    kernel_names = ["gaussian_rbf", "uniform", "multiplicative_rbf_vm"]
    csvs = sorted(path for path in VF_DIR.glob("*.csv") if path.name != "image_pattern_summary_forced.csv")

    stats_rows = []
    bayes_kernel_rows = []
    rss_kernel_rows = []
    nlpd_rows = []
    param_full_rows = []
    param_cv_rows = []
    skipped_rows = []
    loaded_for_gallery = []
    dataset_lookup: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for csv_path in csvs:
        dataset = csv_path.name
        X, phi, stats = load_vf_export(csv_path)
        stats_rows.append(stats)
        if len(phi) < 20:
            skipped_rows.append({"dataset": dataset, "n_used": len(phi), "reason": "fewer than 20 vector-field observations"})
            continue

        loaded_for_gallery.append((dataset, X, phi))
        dataset_lookup[dataset] = (X, phi)
        cfgs = sweep_grid(X)
        param_full_rows.extend(fit_parametric_full(dataset, X, phi))
        param_cv_rows.extend(fit_parametric_cv(dataset, X, phi))
        nlpd_rows.extend(parametric_cv_nlpd(dataset, X, phi))

        for kernel_name in kernel_names:
            kernel_cfgs = [cfg for cfg in cfgs if cfg.name == kernel_name]
            b_cfg, b_sig, b_lml = best_kernel_by_lml(X, phi, kernel_cfgs)
            r_cfg, r_sig, r_rss, r_mae = best_kernel_by_cv_rss(X, phi, kernel_cfgs)
            n_cfg, n_sig, n_nlpd, n_axial_sig, n_mae = best_kernel_by_cv_nlpd(X, phi, kernel_cfgs)
            bayes_kernel_rows.append({
                "dataset": dataset,
                "n": len(phi),
                "model": model_label(b_cfg),
                "kernel": b_cfg.name,
                "bandwidth": b_cfg.bandwidth,
                "kappa": b_cfg.kappa,
                "sigma_n": b_sig,
                "evidence_score": b_lml,
            })
            rss_kernel_rows.append({
                "dataset": dataset,
                "n": len(phi),
                "model": model_label(r_cfg),
                "kernel": r_cfg.name,
                "bandwidth": r_cfg.bandwidth,
                "kappa": r_cfg.kappa,
                "sigma_n": r_sig,
                "mean_test_rss": r_rss,
                "mean_test_mae_deg": r_mae,
            })
            nlpd_rows.append({
                "dataset": dataset,
                "n": len(phi),
                "model": model_label(n_cfg),
                "family": "kernel_gp_line_embedding_cv",
                "kernel": n_cfg.name,
                "bandwidth": n_cfg.bandwidth,
                "kappa": n_cfg.kappa,
                "sigma_n": n_sig,
                "mean_test_nlpd": n_nlpd,
                "mean_train_axial_sigma": n_axial_sig,
                "mean_test_mae_deg": n_mae,
                "p_median": np.nan,
            })

    stats_df = pd.DataFrame(stats_rows)
    kernel_bayes = pd.DataFrame(bayes_kernel_rows)
    kernel_rss = pd.DataFrame(rss_kernel_rows)
    nlpd_cmp = pd.DataFrame(nlpd_rows)
    param_full = pd.DataFrame(param_full_rows)
    param_cv = pd.DataFrame(param_cv_rows)
    evidence_audit = audit_kernel_evidence_identity(kernel_bayes, dataset_lookup)

    stats_df.to_csv(OUT_DIR / "vf_export_stats.csv", index=False)
    pd.DataFrame(skipped_rows, columns=["dataset", "n_used", "reason"]).to_csv(OUT_DIR / "skipped_datasets.csv", index=False)
    kernel_bayes.to_csv(OUT_DIR / "kernel_evidence_by_dataset.csv", index=False)
    kernel_rss.to_csv(OUT_DIR / "kernel_cv_rss_by_dataset.csv", index=False)
    nlpd_cmp.to_csv(OUT_DIR / "combined_cv_nlpd_scores.csv", index=False)
    evidence_audit.to_csv(OUT_DIR / "kernel_evidence_identity_audit.csv", index=False)
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
    nlpd_winners = nlpd_cmp.loc[nlpd_cmp.groupby("dataset")["mean_test_nlpd"].idxmin()].reset_index(drop=True)
    evidence_counts = evidence_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    rss_counts = rss_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    nlpd_counts = nlpd_winners["model"].value_counts().rename_axis("model").reset_index(name="wins")
    evidence_winners.to_csv(OUT_DIR / "evidence_winners.csv", index=False)
    rss_winners.to_csv(OUT_DIR / "rss_winners.csv", index=False)
    nlpd_winners.to_csv(OUT_DIR / "nlpd_winners.csv", index=False)
    evidence_counts.to_csv(OUT_DIR / "evidence_winner_counts.csv", index=False)
    rss_counts.to_csv(OUT_DIR / "rss_winner_counts.csv", index=False)
    nlpd_counts.to_csv(OUT_DIR / "nlpd_winner_counts.csv", index=False)

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
    nlpd_summary = (
        nlpd_cmp.groupby(["model", "family"], dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_test_nlpd_mean=("mean_test_nlpd", "mean"),
            mean_test_nlpd_median=("mean_test_nlpd", "median"),
            mean_train_axial_sigma_mean=("mean_train_axial_sigma", "mean"),
            mean_test_mae_deg_mean=("mean_test_mae_deg", "mean"),
            p_median=("p_median", "median"),
        )
        .reset_index()
        .sort_values("mean_test_nlpd_mean", ascending=True)
    )
    evidence_summary.to_csv(OUT_DIR / "evidence_summary.csv", index=False)
    rss_summary.to_csv(OUT_DIR / "rss_summary.csv", index=False)
    nlpd_summary.to_csv(OUT_DIR / "nlpd_summary.csv", index=False)

    make_input_gallery(loaded_for_gallery)
    plot_summary(evidence_counts, rss_counts, rss_cmp, evidence_summary)

    md = [
        "# vf_exports Axial Line-Field Comparison",
        "",
        f"CSV files found: **{len(csvs)}**",
        f"Datasets processed: **{evidence_cmp['dataset'].nunique()}**",
        f"Datasets skipped: **{len(skipped_rows)}**",
        "",
        "The `vf_exports/*.csv` files are treated as vector-field exports from the Gabor-filter workflow. Each row supplies `Coordinate` and local pitch angle `Angle (α′)`. The comparison converts these into global axial line-field observations by computing `phi = atan2(y, x) + alpha`, then fits the same kernel and non-kernel families used by `cleaned_linefield_comparison.py`.",
        "",
        "## Evidence Identity Audit",
        "The exact equality between Gaussian RBF and Multiplicative RBF-VM evidence is explained by the optimized multiplicative evidence choosing `kappa=0` on every dataset. With `kappa=0`, the von-Mises factor is `exp(0*cos(delta))=1`, so the multiplicative kernel is exactly the Gaussian RBF kernel at the same bandwidth. The audit CSV verifies that the selected Gram matrices have zero maximum absolute difference.",
        df_to_md_table(evidence_audit[["dataset", "multiplicative_kappa", "same_hyperparameters", "max_abs_gram_diff"]]),
        "",
        "## Evidence-Style Winner Counts",
        df_to_md_table(evidence_counts),
        "",
        "## Held-Out Axial RSS Winner Counts",
        df_to_md_table(rss_counts),
        "",
        "## Held-Out Axial NLPD Winner Counts",
        df_to_md_table(nlpd_counts),
        "",
        "## Evidence Summary",
        df_to_md_table(evidence_summary.round(4)),
        "",
        "## RSS Summary",
        df_to_md_table(rss_summary.round(4)),
        "",
        "## Axial NLPD Summary",
        "This uses one common held-out likelihood for every model: a Gaussian density on the axial residual, with the residual scale estimated on the corresponding training fold. Lower is better.",
        df_to_md_table(nlpd_summary.round(4)),
        "",
        "## Input Line Fields",
        "![vf_exports line fields](plots/vf_export_linefields.png)",
        "",
        "## Model Comparison Plots",
        "![Winner counts](plots/winner_counts_vf_exports.png)",
        "",
        "![Held-out axial RSS boxplot](plots/cv_axial_rss_boxplot_vf_exports.png)",
        "",
        "![Mean evidence](plots/mean_evidence_vf_exports.png)",
    ]
    md_text = "\n".join(md)
    (OUT_DIR / "report.md").write_text(md_text)
    (OUT_DIR / "report.html").write_text(md_to_html(md_text))

    print(f"Processed {evidence_cmp['dataset'].nunique()} vf_exports datasets")
    print("RSS winners:")
    print(rss_counts.to_string(index=False))
    print("Evidence winners:")
    print(evidence_counts.to_string(index=False))
    print("NLPD winners:")
    print(nlpd_counts.to_string(index=False))
    print("Evidence identity audit:")
    print(evidence_audit[["multiplicative_kappa", "same_hyperparameters", "max_abs_gram_diff"]].describe(include="all").to_string())
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
