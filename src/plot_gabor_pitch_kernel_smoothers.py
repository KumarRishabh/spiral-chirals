from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VF_DIR = ROOT / "vf_exports"
OUT_DIR = ROOT / "data" / "gabor_pitch_kernel_plots"
PLOT_DIR = OUT_DIR / "plots"


def parse_coord(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    if not isinstance(s, str):
        return [np.nan, np.nan]
    s = s.strip()
    try:
        val = ast.literal_eval(s)
        return list(val)
    except Exception:
        parts = [p.strip() for p in s.strip("()[]").split(",")]
        try:
            return [float(parts[0]), float(parts[1])]
        except Exception:
            return [np.nan, np.nan]


def load_gabor_csv(csv_file: Path) -> pd.DataFrame:
    with open(csv_file, "r") as f:
        df = pd.read_csv(f, encoding="utf-8-sig")
    # The vf_exports files end with a SUMMARY row, matching the notebook code.
    df = df.iloc[:-1].copy()
    return df


def prepare_field(df: pd.DataFrame):
    angles_deg = df["Angle (α′)"].values.astype(float)
    coordinates = df["Coordinate"].apply(parse_coord).tolist()
    X = np.array([coord[0] for coord in coordinates], dtype=float)
    Y = np.array([coord[1] for coord in coordinates], dtype=float)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    alpha_rad = np.radians(angles_deg)
    phi_rad = theta + alpha_rad
    U = np.cos(phi_rad)
    V = np.sin(phi_rad)
    return X, Y, r, theta, angles_deg, alpha_rad, phi_rad, U, V


def gaussian_kernel(u):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


def uniform_kernel(u):
    return (np.abs(u) <= 1.0).astype(float)


def smooth_spiral_pitch(
    target_r,
    sample_r,
    sample_alpha_rad,
    bandwidth,
    kernel="gaussian",
    target_theta=None,
    sample_theta=None,
    kappa=1.0,
):
    """Smooth relative pitch alpha-prime with double-angle line-field symmetry."""
    cos_2alpha = np.cos(2 * sample_alpha_rad)
    sin_2alpha = np.sin(2 * sample_alpha_rad)

    u = (target_r[:, None] - sample_r[None, :]) / bandwidth
    if kernel == "gaussian":
        weights = gaussian_kernel(u)
    elif kernel == "uniform":
        weights = uniform_kernel(u)
    elif kernel == "multiplicative":
        if target_theta is None or sample_theta is None:
            raise ValueError("multiplicative smoothing needs target_theta and sample_theta")
        radial = gaussian_kernel(u)
        angular = np.exp(kappa * np.cos(target_theta[:, None] - sample_theta[None, :]))
        weights = radial * angular
    else:
        raise ValueError(kernel)

    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0
    avg_cos = np.sum(weights * cos_2alpha[None, :], axis=1) / sum_weights
    avg_sin = np.sum(weights * sin_2alpha[None, :], axis=1) / sum_weights
    return 0.5 * np.arctan2(avg_sin, avg_cos)


def axial_residual(alpha_obs_rad, alpha_fit_rad):
    error = alpha_obs_rad - alpha_fit_rad
    return np.mod(error + np.pi / 2, np.pi) - np.pi / 2


def plot_original_vector_field(csv_file: Path, df: pd.DataFrame) -> str:
    X, Y, _r, _theta, angles_deg, _alpha_rad, phi_rad, U, V = prepare_field(df)
    scale_factor = np.round(np.percentile(np.sqrt(U**2 + V**2), 75), 1)
    if scale_factor == 0:
        scale_factor = 1.0

    fig, ax = plt.subplots(figsize=(9, 9))
    magnitude_scale = 25.0
    q = ax.quiver(
        X,
        Y,
        magnitude_scale * U,
        magnitude_scale * V,
        np.degrees(phi_rad),
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="hsv",
        width=0.015,
        headwidth=12,
        headlength=16,
        headaxislength=8,
        alpha=0.95,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.scatter(X, Y, c="k", s=20, zorder=3)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Vector field from coordinates and Angle (α′)")
    cbar = fig.colorbar(q, ax=ax, label="Global angle φ = atan2(y,x)+α′ (deg)")
    cbar.ax.tick_params(labelsize=9)
    ax.quiverkey(
        q,
        X=0.92,
        Y=1.02,
        U=scale_factor,
        label=f"{scale_factor} data-units",
        labelpos="E",
        fontproperties={"size": 10},
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_name = csv_file.stem + "_original_vector_field.svg"
    out_path = PLOT_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return f"plots/{out_name}"


def add_fit_panel(ax, X, Y, theta, alpha_fit_rad, title, color_values=None):
    phi_fit = theta + alpha_fit_rad
    magnitude_scale = 25.0
    U_fit = magnitude_scale * np.cos(phi_fit)
    V_fit = magnitude_scale * np.sin(phi_fit)
    q = ax.quiver(
        X,
        Y,
        U_fit,
        V_fit,
        np.degrees(alpha_fit_rad) if color_values is None else color_values,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="hsv",
        width=0.01,
        headwidth=0,
        headaxislength=0,
        pivot="mid",
        alpha=0.95,
    )
    ax.scatter(X, Y, c="k", s=5, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    return q


def plot_kernel_pitch_smoothers(csv_file: Path, df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    X, Y, r_obs, theta, angles_deg, alpha_obs_rad, phi_obs_rad, U_raw, V_raw = prepare_field(df)
    r_grid = np.linspace(0, np.max(r_obs), 160)
    bandwidth = np.percentile(r_obs, 99) / 10.0
    kappa = 1.0

    alpha_rbf_grid = smooth_spiral_pitch(r_grid, r_obs, alpha_obs_rad, bandwidth, kernel="gaussian")
    alpha_uniform_grid = smooth_spiral_pitch(r_grid, r_obs, alpha_obs_rad, bandwidth, kernel="uniform")
    alpha_rbf = smooth_spiral_pitch(r_obs, r_obs, alpha_obs_rad, bandwidth, kernel="gaussian")
    alpha_uniform = smooth_spiral_pitch(r_obs, r_obs, alpha_obs_rad, bandwidth, kernel="uniform")
    alpha_mult = smooth_spiral_pitch(
        r_obs,
        r_obs,
        alpha_obs_rad,
        bandwidth,
        kernel="multiplicative",
        target_theta=theta,
        sample_theta=theta,
        kappa=kappa,
    )

    residuals = {
        "RBF": np.degrees(axial_residual(alpha_obs_rad, alpha_rbf)),
        "Uniform": np.degrees(axial_residual(alpha_obs_rad, alpha_uniform)),
        "Multiplicative": np.degrees(axial_residual(alpha_obs_rad, alpha_mult)),
    }
    rss = {name: float(np.mean(np.radians(vals) ** 2)) for name, vals in residuals.items()}

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2)

    ax_profile = fig.add_subplot(gs[0, 0])
    alpha_obs_wrapped = np.degrees(np.arctan(np.tan(alpha_obs_rad)))
    ax_profile.scatter(r_obs, alpha_obs_wrapped, c="gray", alpha=0.4, s=20, label="Raw data α′")
    ax_profile.plot(r_grid, np.degrees(alpha_rbf_grid), color="#C62828", linewidth=2.4, label="RBF profile")
    ax_profile.plot(r_grid, np.degrees(alpha_uniform_grid), color="#1565C0", linewidth=2.0, label="Uniform profile")
    ax_profile.axhline(0, color="k", linestyle="--", alpha=0.5, label="Radial (0°)")
    ax_profile.set_xlabel("Distance from center (r)")
    ax_profile.set_ylabel("Pitch angle α′ (deg)")
    ax_profile.set_title("Non-parametric pitch structure")
    ax_profile.legend()
    ax_profile.grid(alpha=0.3)

    ax_raw = fig.add_subplot(gs[0, 1])
    ax_raw.quiver(X, Y, U_raw, V_raw, color="gray", alpha=0.7, scale=25, width=0.003, headwidth=3)
    ax_raw.scatter(X, Y, c="k", s=5, zorder=3)
    ax_raw.set_aspect("equal")
    ax_raw.set_title("Original Gabor field (full data)")
    ax_raw.set_xlabel("X")
    ax_raw.set_ylabel("Y")

    ax_rbf = fig.add_subplot(gs[1, 0])
    q = add_fit_panel(ax_rbf, X, Y, theta, alpha_rbf, f"RBF pitch smoother (h={bandwidth:.2f})")
    fig.colorbar(q, ax=ax_rbf, label="Inferred pitch α′ (deg)")

    ax_uniform = fig.add_subplot(gs[1, 1])
    q = add_fit_panel(ax_uniform, X, Y, theta, alpha_uniform, f"Uniform pitch smoother (h={bandwidth:.2f})")
    fig.colorbar(q, ax=ax_uniform, label="Inferred pitch α′ (deg)")

    ax_mult = fig.add_subplot(gs[2, 0])
    q = add_fit_panel(ax_mult, X, Y, theta, alpha_mult, f"Multiplicative pitch smoother (h={bandwidth:.2f}, κ={kappa:g})")
    fig.colorbar(q, ax=ax_mult, label="Inferred pitch α′ (deg)")

    ax_hist = fig.add_subplot(gs[2, 1])
    bins = np.linspace(-90, 90, 31)
    for name, vals in residuals.items():
        ax_hist.hist(vals, bins=bins, alpha=0.45, density=True, label=f"{name} RSS={rss[name]:.3f}")
    ax_hist.axvline(0, color="k", linestyle="--")
    ax_hist.set_xlabel("Residual error in α′ (degrees, mod 180°)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Full-data residual distributions")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    fig.suptitle(csv_file.name, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_name = csv_file.stem + "_pitch_kernel_smoothers.png"
    out_path = PLOT_DIR / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    stats = {"bandwidth": float(bandwidth), "kappa": float(kappa), **{f"full_data_rss_{k.lower()}": v for k, v in rss.items()}}
    return f"plots/{out_name}", stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(path for path in VF_DIR.glob("*_rings_coords.csv"))
    rows = []
    for csv_file in csvs:
        df = load_gabor_csv(csv_file)
        original = plot_original_vector_field(csv_file, df)
        smoothers, stats = plot_kernel_pitch_smoothers(csv_file, df)
        rows.append({"dataset": csv_file.name, "original_plot": original, "smoother_plot": smoothers, **stats})
        print(f"Wrote plots for {csv_file.name}")

    index = pd.DataFrame(rows)
    index.to_csv(OUT_DIR / "plot_index.csv", index=False)
    md = [
        "# Gabor Pitch-Kernel Plots",
        "",
        "These plots follow the notebook-style code: the Gabor CSV supplies local pitch `Angle (α′)`, and the global plotted direction is reconstructed as `phi = atan2(y, x) + α′`.",
        "",
        "The kernel smoother plots are full-data visual fits, not held-out test-set scores. Held-out RSS is computed separately in the fixed-grid comparison.",
        "",
    ]
    for row in rows:
        md.extend([
            f"## {row['dataset']}",
            "",
            f"![Original vector field]({row['original_plot']})",
            "",
            f"![Pitch kernel smoothers]({row['smoother_plot']})",
            "",
        ])
    (OUT_DIR / "report.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
