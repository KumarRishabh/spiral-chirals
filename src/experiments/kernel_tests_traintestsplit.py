from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from spiral_chirals.geometry import angle_residual_line_field
from spiral_chirals.io import build_spiral_dataset, load_angle_coordinate_csv
from spiral_chirals.kernels import smooth_line_field
from spiral_chirals.types import SpiralDataset


DEFAULT_CSV = Path("vf_exports/Front_EE-1_1_3000x_rings_coords.csv")
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUT = Path(f"src/experiments/multiplicative_kernel_traintest_{timestamp}")


@dataclass(frozen=True)
class SweepResult:
    bandwidth: float
    kappa: float
    train_mae_mean: float
    train_mae_sd: float
    test_mae_mean: float
    test_mae_sd: float
    gap_mean: float
    gap_sd: float
    train_rss_mean: float
    test_rss_mean: float
    n_folds: int


def subset_dataset(data: SpiralDataset, idx: np.ndarray) -> SpiralDataset:
    return SpiralDataset(
        x=data.x[idx],
        y=data.y[idx],
        r=data.r[idx],
        theta=data.theta[idx],
        angle_deg=data.angle_deg[idx],
        angle_rad=data.angle_rad[idx],
        phi_rad=data.phi_rad[idx],
        u=data.u[idx],
        v=data.v[idx],
    )


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def line_field_scores(
    target: SpiralDataset,
    train: SpiralDataset,
    bandwidth: float,
    kappa: float,
) -> tuple[float, float]:
    psi = smooth_line_field(
        target_r=target.r,
        sample_r=train.r,
        sample_theta=train.phi_rad,
        sample_phi_spatial=train.theta,
        bandwidth=bandwidth,
        kernel="multiplicative",
        target_theta=target.theta,
        angular_kappa=kappa,
    )
    fitted = target.theta + psi
    residual = angle_residual_line_field(target.phi_rad, fitted)
    mae_deg = float(np.mean(np.abs(np.degrees(residual))))
    rss = float(np.mean(residual * residual))
    return mae_deg, rss


def run_cv_sweep(
    data: SpiralDataset,
    bandwidths: np.ndarray,
    kappas: list[float],
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(data.x))
    rows: list[SweepResult] = []
    total_configs = len(bandwidths) * len(kappas)

    config_bar = tqdm(total=total_configs, desc="multiplicative kernel configs", unit="cfg")
    for kappa in tqdm(kappas, desc="kappa sweep", unit="kappa"):
        for bandwidth in tqdm(bandwidths, desc=f"bandwidth sweep (kappa={kappa:g})", unit="bw", leave=False):
            train_mae: list[float] = []
            test_mae: list[float] = []
            train_rss: list[float] = []
            test_rss: list[float] = []

            for train_idx, test_idx in folds:
                train = subset_dataset(data, train_idx)
                test = subset_dataset(data, test_idx)

                tr_mae, tr_rss = line_field_scores(train, train, float(bandwidth), float(kappa))
                te_mae, te_rss = line_field_scores(test, train, float(bandwidth), float(kappa))
                train_mae.append(tr_mae)
                test_mae.append(te_mae)
                train_rss.append(tr_rss)
                test_rss.append(te_rss)

            gap = np.asarray(test_mae) - np.asarray(train_mae)
            rows.append(
                SweepResult(
                    bandwidth=float(bandwidth),
                    kappa=float(kappa),
                    train_mae_mean=float(np.mean(train_mae)),
                    train_mae_sd=float(np.std(train_mae, ddof=1)),
                    test_mae_mean=float(np.mean(test_mae)),
                    test_mae_sd=float(np.std(test_mae, ddof=1)),
                    gap_mean=float(np.mean(gap)),
                    gap_sd=float(np.std(gap, ddof=1)),
                    train_rss_mean=float(np.mean(train_rss)),
                    test_rss_mean=float(np.mean(test_rss)),
                    n_folds=n_splits,
                )
            )
            config_bar.update(1)

    config_bar.close()
    return pd.DataFrame([asdict(row) for row in rows])


def choose_models(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    min_test = results.loc[results["test_mae_mean"].idxmin()]

    nonnegative_gap = results.loc[results["gap_mean"] >= 0].copy()
    if nonnegative_gap.empty:
        min_gap = results.iloc[(results["gap_mean"].abs()).argmin()]
    else:
        # Tie-break toward lower held-out error and then smaller bandwidth.
        min_gap = nonnegative_gap.sort_values(
            ["gap_mean", "test_mae_mean", "bandwidth", "kappa"],
            ascending=[True, True, True, True],
        ).iloc[0]
    return min_test, min_gap


def plot_cv_curves(results: pd.DataFrame, min_test: pd.Series, min_gap: pd.Series, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for kappa, group in results.groupby("kappa"):
        ordered = group.sort_values("bandwidth")
        label = f"kappa={kappa:g}"
        axes[0].plot(ordered["bandwidth"], ordered["test_mae_mean"], lw=1.4, alpha=0.8, label=label)
        axes[1].plot(ordered["bandwidth"], ordered["gap_mean"], lw=1.4, alpha=0.8, label=label)

    axes[0].scatter([min_test["bandwidth"]], [min_test["test_mae_mean"]], s=70, c="red", zorder=5)
    axes[0].set_title("Held-out MAE by bandwidth and angular kappa")
    axes[0].set_xlabel("Bandwidth")
    axes[0].set_ylabel("Test MAE (degrees)")
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color="#555555", lw=1, ls="--")
    axes[1].scatter([min_gap["bandwidth"]], [min_gap["gap_mean"]], s=70, c="green", zorder=5)
    axes[1].set_title("Overfitting gap: test MAE - train MAE")
    axes[1].set_xlabel("Bandwidth")
    axes[1].set_ylabel("Gap (degrees)")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) <= 12:
        axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "multiplicative_train_test_curves.png", dpi=180)
    plt.close(fig)


def plot_heatmap(results: pd.DataFrame, min_test: pd.Series, out_dir: Path) -> None:
    pivot = results.pivot(index="kappa", columns="bandwidth", values="test_mae_mean").sort_index()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Multiplicative kernel held-out MAE surface")
    ax.set_xlabel("Bandwidth grid index")
    ax.set_ylabel("Angular kappa")
    ax.set_yticks(np.arange(len(pivot.index)), [f"{v:g}" for v in pivot.index])

    bw_values = pivot.columns.to_numpy()
    tick_idx = np.linspace(0, len(bw_values) - 1, min(8, len(bw_values)), dtype=int)
    ax.set_xticks(tick_idx, [f"{bw_values[i]:.2g}" for i in tick_idx])

    kappa_idx = int(np.where(pivot.index.to_numpy() == min_test["kappa"])[0][0])
    bandwidth_idx = int(np.argmin(np.abs(bw_values - min_test["bandwidth"])))
    ax.scatter([bandwidth_idx], [kappa_idx], s=90, c="red", marker="x", linewidths=2)
    fig.colorbar(im, ax=ax, label="Test MAE (degrees)")
    fig.tight_layout()
    fig.savefig(out_dir / "multiplicative_test_mae_heatmap.png", dpi=180)
    plt.close(fig)


def plot_residuals(data: SpiralDataset, bandwidth: float, kappa: float, out_dir: Path) -> None:
    psi = smooth_line_field(
        target_r=data.r,
        sample_r=data.r,
        sample_theta=data.phi_rad,
        sample_phi_spatial=data.theta,
        bandwidth=bandwidth,
        kernel="multiplicative",
        target_theta=data.theta,
        angular_kappa=kappa,
    )
    fitted = data.theta + psi
    residuals_deg = np.degrees(angle_residual_line_field(data.phi_rad, fitted))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(data.r, residuals_deg, s=12, alpha=0.55, color="#4C72B0")
    ax.axhline(0, color="#C44E52", linestyle="--", lw=1.2)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Axial residual (degrees)")
    ax.set_title(f"Residuals for min-test multiplicative fit (bw={bandwidth:.3g}, kappa={kappa:g})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "multiplicative_residuals_min_test.png", dpi=180)
    plt.close(fig)


def plot_streamlines(data: SpiralDataset, bandwidth: float, kappa: float, out_dir: Path) -> None:
    psi = smooth_line_field(
        target_r=data.r,
        sample_r=data.r,
        sample_theta=data.phi_rad,
        sample_phi_spatial=data.theta,
        bandwidth=bandwidth,
        kernel="multiplicative",
        target_theta=data.theta,
        angular_kappa=kappa,
    )
    phi_fitted = data.theta + psi
    u_fit = np.cos(phi_fitted)
    v_fit = np.sin(phi_fitted)

    xi = np.linspace(data.x.min(), data.x.max(), 120)
    yi = np.linspace(data.y.min(), data.y.max(), 120)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    ui = griddata((data.x, data.y), u_fit, (xi_grid, yi_grid), method="linear")
    vi = griddata((data.x, data.y), v_fit, (xi_grid, yi_grid), method="linear")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.quiver(
        data.x,
        data.y,
        u_fit,
        v_fit,
        pivot="mid",
        headwidth=0,
        headlength=0,
        headaxislength=0,
        scale=25,
        width=0.004,
        color="purple",
        alpha=0.7,
    )
    ax1.scatter(data.x, data.y, s=8, c="k", alpha=0.28)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Fitted line field (bw={bandwidth:.3g}, kappa={kappa:g})")
    ax1.grid(alpha=0.25)

    ax2.streamplot(xi_grid, yi_grid, ui, vi, density=2.0, color="teal", linewidth=1)
    ax2.scatter(data.x, data.y, s=5, c="red", alpha=0.2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Streamlines from multiplicative smoothed field")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "streamlines_multiplicative_min_test_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test sweep for multiplicative line-field kernel.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Angle-coordinate CSV file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--folds", type=int, default=50, help="Number of KFold splits.")
    parser.add_argument("--bandwidth-min", type=float, default=0.1)
    parser.add_argument("--bandwidth-max", type=float, default=100.0)
    parser.add_argument("--bandwidth-count", type=int, default=2000)
    parser.add_argument(
        "--bandwidth-spacing",
        choices=["linear", "geom"],
        default="linear",
        help="Use linear spacing to match the earlier snippet, or geom for log-scale sweeps.",
    )
    parser.add_argument("--kappas", default="0,0.5,1,2,4,8,16", help="Comma-separated kappa grid.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.bandwidth_spacing == "geom":
        bandwidths = np.geomspace(args.bandwidth_min, args.bandwidth_max, args.bandwidth_count)
    else:
        bandwidths = np.linspace(args.bandwidth_min, args.bandwidth_max, args.bandwidth_count)
    kappas = parse_float_list(args.kappas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_angle_coordinate_csv(args.csv)
    data = build_spiral_dataset(df)

    print(f"Loaded {len(data.x)} observations from {args.csv}")
    print(
        "Sweeping multiplicative kernel over "
        f"{len(bandwidths)} bandwidths x {len(kappas)} kappas x {args.folds} folds "
        f"= {len(bandwidths) * len(kappas) * args.folds:,} fold fits."
    )

    results = run_cv_sweep(data, bandwidths, kappas, args.folds, args.random_state)
    results_path = args.out_dir / "multiplicative_bandwidth_kappa_cv.csv"
    results.to_csv(results_path, index=False)

    min_test, min_gap = choose_models(results)
    selected = pd.DataFrame(
        [
            {"criterion": "minimum_test_mae", **min_test.to_dict()},
            {"criterion": "minimum_nonnegative_gap", **min_gap.to_dict()},
        ]
    )
    selected_path = args.out_dir / "selected_multiplicative_configs.csv"
    selected.to_csv(selected_path, index=False)

    plot_cv_curves(results, min_test, min_gap, args.out_dir)
    plot_heatmap(results, min_test, args.out_dir)
    plot_residuals(data, float(min_test["bandwidth"]), float(min_test["kappa"]), args.out_dir)
    plot_streamlines(data, float(min_test["bandwidth"]), float(min_test["kappa"]), args.out_dir)

    print("\nSelected multiplicative kernel configurations")
    print(selected[["criterion", "bandwidth", "kappa", "train_mae_mean", "test_mae_mean", "gap_mean"]].to_string(index=False))
    print(f"\nSaved CV table: {results_path}")
    print(f"Saved selected configs: {selected_path}")
    print(f"Saved plots in: {args.out_dir}")


if __name__ == "__main__":
    main()
