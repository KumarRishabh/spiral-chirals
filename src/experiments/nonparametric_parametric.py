from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import griddata  # <-- add this
from spiral_chirals.io import build_spiral_dataset, load_angle_coordinate_csv
from spiral_chirals.kernels import smooth_line_field
from spiral_chirals.geometry import angle_residual_line_field, relative_pitch
from spiral_chirals.parametric import (
    fit_archimedean_spiral,
    fit_fermat_spiral,
    fit_log_spiral as fit_log_spiral_pitch,  # <-- alias to avoid collision
    archimedean_pitch,
    fermat_pitch,
    log_spiral_pitch,
    predict_phi,
)

from spiral_chirals.io import build_spiral_dataset, load_angle_coordinate_csv
from spiral_chirals.kernels import smooth_line_field
from spiral_chirals.geometry import angle_residual_line_field
from spiral_chirals.types import SpiralDataset


# ----------------------------
# Synthetic 1D regression part
# ----------------------------
def generate_spiral_data(n_points: int = 200, noise: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)
    x = t * np.cos(t)
    y = t * np.sin(t) + rng.normal(0, noise, n_points)
    return x, y


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(u) <= 1)


def _unwrap_phi(phi: np.ndarray) -> np.ndarray:
    """Unwrap phi robustly by sorting, unwrapping, then unsorting."""
    idx = np.argsort(phi)
    phi_sorted = phi[idx]
    phi_unw_sorted = np.unwrap(phi_sorted)
    phi_unw = np.empty_like(phi_sorted)
    phi_unw[:] = phi_unw_sorted
    out = np.empty_like(phi)
    out[idx] = phi_unw
    return out


def _tangent_direction_from_rphi(phi_wrapped: np.ndarray, r: np.ndarray, dr_dphi: np.ndarray) -> np.ndarray:
    """Return tangent direction angle atan2(dy/dphi, dx/dphi) for polar curve r(phi)."""
    c = np.cos(phi_wrapped)
    s = np.sin(phi_wrapped)
    dx = dr_dphi * c - r * s
    dy = dr_dphi * s + r * c
    return np.arctan2(dy, dx)


def fit_archimedean(phi_unw: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """r = a + b*phi"""
    reg = LinearRegression().fit(phi_unw.reshape(-1, 1), r)
    a = float(reg.intercept_)
    b = float(reg.coef_[0])
    return a, b


def fit_log_spiral(phi_unw: np.ndarray, r: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """r = a*exp(b*phi)  <=>  log(r) = log(a) + b*phi"""
    mask = r > eps
    reg = LinearRegression().fit(phi_unw[mask].reshape(-1, 1), np.log(r[mask]))
    log_a = float(reg.intercept_)
    b = float(reg.coef_[0])
    a = float(np.exp(log_a))
    return a, b


def fit_fermat(phi_unw: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """r^2 = A + B*phi"""
    reg = LinearRegression().fit(phi_unw.reshape(-1, 1), (r ** 2))
    A = float(reg.intercept_)
    B = float(reg.coef_[0])
    return A, B


def predict_theta_parametric(phi_wrapped: np.ndarray, phi_unw: np.ndarray, model: str, params: tuple[float, float]) -> np.ndarray:
    """Predict line direction theta(phi) as the tangent direction implied by the spiral family."""
    eps = 1e-12

    if model == "archimedean":
        a, b = params
        r_hat = a + b * phi_unw
        r_hat = np.maximum(r_hat, eps)
        dr = np.full_like(phi_unw, b, dtype=float)

    elif model == "log":
        a, b = params
        r_hat = a * np.exp(b * phi_unw)
        r_hat = np.maximum(r_hat, eps)
        dr = b * r_hat

    elif model == "fermat":
        A, B = params
        inside = A + B * phi_unw
        inside = np.maximum(inside, eps)
        r_hat = np.sqrt(inside)
        dr = B / (2.0 * r_hat)

    else:
        raise ValueError(f"Unknown model: {model}")

    return _tangent_direction_from_rphi(phi_wrapped=phi_wrapped, r=r_hat, dr_dphi=dr)


def kernel_regression_1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel_func,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Nadaraya–Watson 1D kernel regression, vectorized."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    u = (x_test[:, None] - x_train[None, :]) / bandwidth
    w = kernel_func(u)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1.0
    w = w / w_sum
    return (w @ y_train).ravel()


def run_synthetic_regression(bandwidth: float = 1.0, poly_degree: int = 2) -> None:
    # Generate synthetic spiral in Cartesian
    x, y = generate_spiral_data(n_points=250, noise=0.35, seed=0)

    # Convert to polar for spiral model comparisons: r(theta)
    theta = np.unwrap(np.arctan2(y, x))
    r = np.sqrt(x**2 + y**2)

    # simple split by index (kept from your draft)
    split_idx = int(0.8 * len(theta))
    theta_train, theta_test = theta[:split_idx], theta[split_idx:]
    r_train, r_test = r[:split_idx], r[split_idx:]

    kernels = {
        "Gaussian": gaussian_kernel,
        "Epanechnikov": epanechnikov_kernel,
        "Uniform": uniform_kernel,
    }

    rss_results: dict[str, float] = {}

    # ----------------------------
    # Nonparametric: kernel regression r(theta)
    # ----------------------------
    for name, kernel in kernels.items():
        r_pred = kernel_regression_1d(theta_train, r_train, theta_test, kernel, bandwidth=bandwidth)
        rss = float(np.sum((r_test - r_pred) ** 2))
        rss_results[f"{name} kernel (bw={bandwidth:g})"] = rss
        print(f"{name} kernel RSS (r vs theta): {rss:.4f}")

    # ----------------------------
    # Parametric baselines on r(theta)
    # ----------------------------

    # 1) Linear (same form as Archimedean: r = a + b*theta)
    lin_reg = LinearRegression().fit(theta_train.reshape(-1, 1), r_train)
    r_pred_lin = lin_reg.predict(theta_test.reshape(-1, 1))
    rss_lin = float(np.sum((r_test - r_pred_lin) ** 2))
    rss_results["Linear / Archimedean (r=a+bθ)"] = rss_lin
    print(f"Linear/Archimedean RSS: {rss_lin:.4f}")

    # 2) Polynomial regression: r = poly(theta)
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    th_train_poly = poly.fit_transform(theta_train.reshape(-1, 1))
    th_test_poly = poly.transform(theta_test.reshape(-1, 1))
    poly_reg = LinearRegression().fit(th_train_poly, r_train)
    r_pred_poly = poly_reg.predict(th_test_poly)
    rss_poly = float(np.sum((r_test - r_pred_poly) ** 2))
    rss_results[f"Polynomial (deg {poly_degree})"] = rss_poly
    print(f"Polynomial RSS: {rss_poly:.4f}")

    # 3) Log spiral: r = a * exp(b*theta)  <=>  log(r) = log(a) + b*theta
    eps = 1e-12
    mask_pos = r_train > eps
    log_reg = LinearRegression().fit(theta_train[mask_pos].reshape(-1, 1), np.log(r_train[mask_pos]))
    log_r_pred = log_reg.predict(theta_test.reshape(-1, 1))
    r_pred_log = np.exp(log_r_pred)
    rss_log = float(np.sum((r_test - r_pred_log) ** 2))
    rss_results["Log spiral (r=a·exp(bθ))"] = rss_log
    print(f"Log spiral RSS: {rss_log:.4f}")

    # 4) Fermat spiral: r^2 = A + B*theta  (allows shift/intercept)
    fermat_reg = LinearRegression().fit(theta_train.reshape(-1, 1), r_train**2)
    r2_pred = fermat_reg.predict(theta_test.reshape(-1, 1))
    r2_pred = np.maximum(r2_pred, 0.0)
    r_pred_fermat = np.sqrt(r2_pred)
    rss_fermat = float(np.sum((r_test - r_pred_fermat) ** 2))
    rss_results["Fermat spiral (r²=A+Bθ)"] = rss_fermat
    print(f"Fermat spiral RSS: {rss_fermat:.4f}")

    # Plot RSS comparison
    plt.figure(figsize=(12, 4))
    plt.bar(list(rss_results.keys()), list(rss_results.values()))
    plt.ylabel("RSS on r(theta)")
    plt.title("Parametric vs Nonparametric (synthetic spiral; polar regression)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Line-field CV bandwidth part
# ----------------------------
@dataclass(frozen=True)
class KernelCVResult:
    name: str
    best_params_mae: dict
    best_params_rss: dict
    train_mae: np.ndarray
    test_mae: np.ndarray
    train_rss: np.ndarray
    test_rss: np.ndarray
    x1: np.ndarray
    x2: np.ndarray | None = None


def _mae_deg_linefield(obs_dir: np.ndarray, pred_dir: np.ndarray) -> float:
    res = angle_residual_line_field(obs_dir, pred_dir)
    return float(np.mean(np.abs(np.degrees(res))))


def _rss_linefield(obs_dir: np.ndarray, pred_dir: np.ndarray) -> float:
    """RSS = sum of squared angular residuals (radians^2)."""
    res = angle_residual_line_field(obs_dir, pred_dir)
    return float(np.sum(res**2))

def run_linefield_cv_kernels(
    csv_file: str,
    *,
    bandwidths: np.ndarray,
    n_splits: int = 20,
    seed: int = 42,
    kappas: np.ndarray | None = None,
) -> list[KernelCVResult]:
    df = load_angle_coordinate_csv(csv_file)
    data = build_spiral_dataset(df)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ---- MAE accumulators (mean over folds) ----
    g_train_mae = np.zeros(len(bandwidths))
    g_test_mae = np.zeros(len(bandwidths))
    u_train_mae = np.zeros(len(bandwidths))
    u_test_mae = np.zeros(len(bandwidths))

    # ---- RSS accumulators (mean over folds) ----
    g_train_rss = np.zeros(len(bandwidths))
    g_test_rss = np.zeros(len(bandwidths))
    u_train_rss = np.zeros(len(bandwidths))
    u_test_rss = np.zeros(len(bandwidths))

    if kappas is not None:
        m_train_mae = np.zeros((len(bandwidths), len(kappas)))
        m_test_mae = np.zeros((len(bandwidths), len(kappas)))
        m_train_rss = np.zeros((len(bandwidths), len(kappas)))
        m_test_rss = np.zeros((len(bandwidths), len(kappas)))
    else:
        m_train_mae = m_test_mae = None
        m_train_rss = m_test_rss = None

    folds = 0
    for train_idx, test_idx in kf.split(data.x):
        train = SpiralDataset(
            x=data.x[train_idx], y=data.y[train_idx], r=data.r[train_idx],
            theta=data.theta[train_idx], angle_deg=data.angle_deg[train_idx],
            angle_rad=data.angle_rad[train_idx], phi_rad=data.phi_rad[train_idx],
            u=data.u[train_idx], v=data.v[train_idx],
        )
        test = SpiralDataset(
            x=data.x[test_idx], y=data.y[test_idx], r=data.r[test_idx],
            theta=data.theta[test_idx], angle_deg=data.angle_deg[test_idx],
            angle_rad=data.angle_rad[test_idx], phi_rad=data.phi_rad[test_idx],
            u=data.u[test_idx], v=data.v[test_idx],
        )

        for i, bw in enumerate(bandwidths):
            bw = float(bw)

            # ---- Gaussian ----
            psi_tr = smooth_line_field(
                target_r=train.r, sample_r=train.r,
                sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                bandwidth=bw, kernel="gaussian",
            )
            pred_tr = train.theta + psi_tr
            g_train_mae[i] += _mae_deg_linefield(train.phi_rad, pred_tr)
            g_train_rss[i] += _rss_linefield(train.phi_rad, pred_tr)

            psi_te = smooth_line_field(
                target_r=test.r, sample_r=train.r,
                sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                bandwidth=bw, kernel="gaussian",
            )
            pred_te = test.theta + psi_te
            g_test_mae[i] += _mae_deg_linefield(test.phi_rad, pred_te)
            g_test_rss[i] += _rss_linefield(test.phi_rad, pred_te)

            # ---- Uniform ----
            psi_tr = smooth_line_field(
                target_r=train.r, sample_r=train.r,
                sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                bandwidth=bw, kernel="uniform",
            )
            pred_tr = train.theta + psi_tr
            u_train_mae[i] += _mae_deg_linefield(train.phi_rad, pred_tr)
            u_train_rss[i] += _rss_linefield(train.phi_rad, pred_tr)

            psi_te = smooth_line_field(
                target_r=test.r, sample_r=train.r,
                sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                bandwidth=bw, kernel="uniform",
            )
            pred_te = test.theta + psi_te
            u_test_mae[i] += _mae_deg_linefield(test.phi_rad, pred_te)
            u_test_rss[i] += _rss_linefield(test.phi_rad, pred_te)

            # ---- Multiplicative (if enabled) ----
            if kappas is not None:
                assert m_train_mae is not None and m_test_mae is not None
                assert m_train_rss is not None and m_test_rss is not None
                for j, kappa in enumerate(kappas):
                    kappa = float(kappa)

                    psi_tr = smooth_line_field(
                        target_r=train.r, sample_r=train.r,
                        sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                        bandwidth=bw, kernel="multiplicative",
                        target_theta=train.theta, angular_kappa=kappa,
                    )
                    pred_tr = train.theta + psi_tr
                    m_train_mae[i, j] += _mae_deg_linefield(train.phi_rad, pred_tr)
                    m_train_rss[i, j] += _rss_linefield(train.phi_rad, pred_tr)

                    psi_te = smooth_line_field(
                        target_r=test.r, sample_r=train.r,
                        sample_theta=train.phi_rad, sample_phi_spatial=train.theta,
                        bandwidth=bw, kernel="multiplicative",
                        target_theta=test.theta, angular_kappa=kappa,
                    )
                    pred_te = test.theta + psi_te
                    m_test_mae[i, j] += _mae_deg_linefield(test.phi_rad, pred_te)
                    m_test_rss[i, j] += _rss_linefield(test.phi_rad, pred_te)

        folds += 1

    # Average over folds
    denom = max(folds, 1)
    g_train_mae /= denom
    g_test_mae /= denom
    u_train_mae /= denom
    u_test_mae /= denom
    g_train_rss /= denom
    g_test_rss /= denom
    u_train_rss /= denom
    u_test_rss /= denom

    results: list[KernelCVResult] = []

    # Gaussian best by MAE and by RSS
    gi_mae = int(np.argmin(g_test_mae))
    gi_rss = int(np.argmin(g_test_rss))
    results.append(
        KernelCVResult(
            name="gaussian",
            best_params_mae={"bandwidth": float(bandwidths[gi_mae])},
            best_params_rss={"bandwidth": float(bandwidths[gi_rss])},
            train_mae=g_train_mae,
            test_mae=g_test_mae,
            train_rss=g_train_rss,
            test_rss=g_test_rss,
            x1=bandwidths,
        )
    )

    # Uniform best
    ui_mae = int(np.argmin(u_test_mae))
    ui_rss = int(np.argmin(u_test_rss))
    results.append(
        KernelCVResult(
            name="uniform",
            best_params_mae={"bandwidth": float(bandwidths[ui_mae])},
            best_params_rss={"bandwidth": float(bandwidths[ui_rss])},
            train_mae=u_train_mae,
            test_mae=u_test_mae,
            train_rss=u_train_rss,
            test_rss=u_test_rss,
            x1=bandwidths,
        )
    )

    # Multiplicative best (2D)
    if kappas is not None and m_test_mae is not None and m_test_rss is not None and m_train_mae is not None and m_train_rss is not None:
        m_train_mae /= denom
        m_test_mae /= denom
        m_train_rss /= denom
        m_test_rss /= denom

        flat_mae = int(np.argmin(m_test_mae))
        bi_mae, bj_mae = np.unravel_index(flat_mae, m_test_mae.shape)

        flat_rss = int(np.argmin(m_test_rss))
        bi_rss, bj_rss = np.unravel_index(flat_rss, m_test_rss.shape)

        results.append(
            KernelCVResult(
                name="multiplicative",
                best_params_mae={"sigma": float(bandwidths[bi_mae]), "kappa": float(kappas[bj_mae])},
                best_params_rss={"sigma": float(bandwidths[bi_rss]), "kappa": float(kappas[bj_rss])},
                train_mae=m_train_mae,
                test_mae=m_test_mae,
                train_rss=m_train_rss,
                test_rss=m_test_rss,
                x1=bandwidths,
                x2=kappas,
            )
        )

    return results


def plot_kernel_cv_results_rss(results: list[KernelCVResult]) -> None:
    # 1D RSS plots
    for res in results:
        if res.name in ("gaussian", "uniform"):
            best_bw = res.best_params_rss["bandwidth"]
            plt.figure(figsize=(10, 6))
            plt.plot(res.x1, res.train_rss, "-", label="Train RSS", linewidth=2)
            plt.plot(res.x1, res.test_rss, "-", label="Test RSS", linewidth=2)
            plt.axvline(best_bw, color="black", linestyle="--", label=f"best={best_bw:g}")
            plt.xlabel("Bandwidth")
            plt.ylabel("RSS (rad²)")
            plt.title(f"Line-field CV: {res.name} kernel (RSS)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 2D RSS heatmap for multiplicative
    for res in results:
        if res.name == "multiplicative":
            sigma_grid = res.x1
            kappa_grid = res.x2
            assert kappa_grid is not None

            test_rss = res.test_rss  # (n_sigma, n_kappa)
            plt.figure(figsize=(9, 6))
            plt.imshow(
                test_rss,
                origin="lower",
                aspect="auto",
                extent=[float(kappa_grid.min()), float(kappa_grid.max()), float(sigma_grid.min()), float(sigma_grid.max())],
            )
            plt.colorbar(label="Mean Test RSS (rad²)")
            plt.xlabel("kappa")
            plt.ylabel("sigma (bandwidth)")
            bp = res.best_params_rss
            plt.scatter([bp["kappa"]], [bp["sigma"]], c="red")
            plt.title(f"Multiplicative kernel CV (RSS best sigma={bp['sigma']:.3g}, kappa={bp['kappa']:.3g})")
            plt.tight_layout()
            plt.show()


def print_kernel_rss_leaderboard(results: list[KernelCVResult]) -> None:
    rows: list[tuple[str, dict, float]] = []
    for res in results:
        if res.name in ("gaussian", "uniform"):
            best_i = int(np.argmin(res.test_rss))
            rows.append((res.name, {"bandwidth": float(res.x1[best_i])}, float(res.test_rss[best_i])))
        else:
            # multiplicative
            flat = int(np.argmin(res.test_rss))
            bi, bj = np.unravel_index(flat, res.test_rss.shape)
            assert res.x2 is not None
            rows.append((res.name, {"sigma": float(res.x1[bi]), "kappa": float(res.x2[bj])}, float(res.test_rss[bi, bj])))

    print("\nKernel RSS leaderboard (mean over folds; lower is better):")
    for name, params, val in sorted(rows, key=lambda t: t[2]):
        print(f"  {name:14s} params={params}  test_RSS={val:.6g} (rad²)")

def plot_kernel_cv_results(results: list[KernelCVResult]) -> None:
    # 1D plots: gaussian + uniform
    for res in results:
        if res.name in ("gaussian", "uniform"):
            best_bw = res.best_params_mae["bandwidth"]
            plt.figure(figsize=(10, 6))
            plt.plot(res.x1, res.train_means, "-", label="Train MAE", linewidth=2)
            plt.plot(res.x1, res.test_means, "-", label="Test MAE", linewidth=2)
            plt.axvline(best_bw, color="black", linestyle="--", label=f"best={best_bw:g}")
            plt.xlabel("Bandwidth")
            plt.ylabel("MAE (degrees)")
            plt.title(f"Line-field CV: {res.name} kernel")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 2D plot: multiplicative (heatmap over sigma x kappa)
    for res in results:
        if res.name == "multiplicative":
            sigma_grid = res.x1
            kappa_grid = res.x2
            assert kappa_grid is not None
            test_mae = res.test_means  # shape (n_sigma, n_kappa)

            plt.figure(figsize=(9, 6))
            # imshow expects [rows, cols] => sigma as rows, kappa as cols
            plt.imshow(
                test_mae,
                origin="lower",
                aspect="auto",
                extent=[float(kappa_grid.min()), float(kappa_grid.max()), float(sigma_grid.min()), float(sigma_grid.max())],
            )
            plt.colorbar(label="Mean Test MAE (deg)")
            plt.xlabel("kappa")
            plt.ylabel("sigma (bandwidth)")
            bp = res.best_params
            plt.scatter([bp["kappa"]], [bp["sigma"]], c="red")
            plt.title(f"Multiplicative kernel CV (best sigma={bp['sigma']:.3g}, kappa={bp['kappa']:.3g})")
            plt.tight_layout()
            plt.show()


def compare_linefield_models_rss(
    csv_file: str,
    *,
    kernel_bw: float,
    scaling: bool = True,
    use_sin: bool = False,
) -> dict[str, float]:
    """
    Compare RSS on line-field residuals for:
      - Kernel smooth_line_field (bandwidth=kernel_bw)
      - Archimedean / Fermat / Log parametric pitch families (from parametric.py)

    Returns dict model_name -> RSS (lower is better).
    """
    df = load_angle_coordinate_csv(csv_file)
    data = build_spiral_dataset(df)

    # Conventions consistent with your smooth_line_field call:
    #   observed line direction = data.phi_rad
    #   spatial angle (position) = data.theta
    psi_obs = relative_pitch(sample_theta=data.phi_rad, sample_phi_spatial=data.theta)

    rss: dict[str, float] = {}

    # --- Kernel (nonparametric) ---
    psi_kernel = smooth_line_field(
        target_r=data.r,
        sample_r=data.r,
        sample_theta=data.phi_rad,
        sample_phi_spatial=data.theta,
        bandwidth=float(kernel_bw),
    )
    dir_kernel = predict_phi(theta=data.theta, pitch_rad=psi_kernel)
    resid_kernel = angle_residual_line_field(data.phi_rad, dir_kernel)
    rss["Kernel (smooth_line_field)"] = float(np.sum(resid_kernel**2))

    # --- Parametric fits (pitch is function of r) ---
    fit_arch = fit_archimedean_spiral(psi_obs, data.r, scaling=scaling, use_sin=use_sin)
    psi_arch = archimedean_pitch(data.r, b=fit_arch.value)
    dir_arch = predict_phi(theta=data.theta, pitch_rad=psi_arch)
    resid_arch = angle_residual_line_field(data.phi_rad, dir_arch)
    rss["Archimedean (parametric.py)"] = float(np.sum(resid_arch**2))

    fit_fer = fit_fermat_spiral(psi_obs, data.r, scaling=scaling, use_sin=use_sin)
    psi_fer = fermat_pitch(data.r, a=fit_fer.value)
    dir_fer = predict_phi(theta=data.theta, pitch_rad=psi_fer)
    resid_fer = angle_residual_line_field(data.phi_rad, dir_fer)
    rss["Fermat (parametric.py)"] = float(np.sum(resid_fer**2))

    fit_log = fit_log_spiral_pitch(psi_obs, data.r, scaling=scaling, use_sin=use_sin)  # <-- use aliased import
    psi_log = log_spiral_pitch(data.r, k=fit_log.value)
    dir_log = predict_phi(theta=data.theta, pitch_rad=psi_log)
    resid_log = angle_residual_line_field(data.phi_rad, dir_log)
    rss["Log (parametric.py)"] = float(np.sum(resid_log**2))

    # Print + bar plot
    print("\nRSS comparison (lower is better):")
    for name, val in sorted(rss.items(), key=lambda kv: kv[1]):
        print(f"  {name:28s} RSS={val:.6g}")

    plt.figure(figsize=(10, 4))
    names = list(rss.keys())
    vals = [rss[n] for n in names]
    plt.bar(names, vals)
    plt.ylabel("RSS (sum of squared angular residuals)")
    plt.title(f"Line-field RSS: kernel vs parametric (kernel bw={kernel_bw:g})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    return rss

def plot_kernel_cv_results(results: list[KernelCVResult]) -> None:
    # 1D plots: gaussian + uniform (MAE)
    for res in results:
        if res.name in ("gaussian", "uniform"):
            best_bw = res.best_params_mae["bandwidth"]
            plt.figure(figsize=(10, 6))
            plt.plot(res.x1, res.train_mae, "-", label="Train MAE", linewidth=2)
            plt.plot(res.x1, res.test_mae, "-", label="Test MAE", linewidth=2)
            plt.axvline(best_bw, color="black", linestyle="--", label=f"best={best_bw:g}")
            plt.xlabel("Bandwidth")
            plt.ylabel("MAE (degrees)")
            plt.title(f"Line-field CV: {res.name} kernel (MAE)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 2D plot: multiplicative (MAE heatmap over sigma x kappa)
    for res in results:
        if res.name == "multiplicative":
            sigma_grid = res.x1
            kappa_grid = res.x2
            assert kappa_grid is not None
            test_mae = res.test_mae  # (n_sigma, n_kappa)

            plt.figure(figsize=(9, 6))
            plt.imshow(
                test_mae,
                origin="lower",
                aspect="auto",
                extent=[
                    float(kappa_grid.min()),
                    float(kappa_grid.max()),
                    float(sigma_grid.min()),
                    float(sigma_grid.max()),
                ],
            )
            plt.colorbar(label="Mean Test MAE (deg)")
            plt.xlabel("kappa")
            plt.ylabel("sigma (bandwidth)")
            bp = res.best_params_mae
            plt.scatter([bp["kappa"]], [bp["sigma"]], c="red")
            plt.title(f"Multiplicative kernel CV (MAE best sigma={bp['sigma']:.3g}, kappa={bp['kappa']:.3g})")
            plt.tight_layout()
            plt.show()


def _fit_predict_parametric_pitch(
    model: str,
    psi_train: np.ndarray,
    r_train: np.ndarray,
    theta_train: np.ndarray,
    phi_train: np.ndarray,
    psi_test: np.ndarray,
    r_test: np.ndarray,
    theta_test: np.ndarray,
    phi_test: np.ndarray,
    *,
    scaling: bool,
    use_sin: bool,
) -> tuple[float, float, float, float]:
    """
    Returns (train_mae_deg, test_mae_deg, train_rss, test_rss) for one parametric family.
    """
    if model == "archimedean":
        fit = fit_archimedean_spiral(psi_train, r_train, scaling=scaling, use_sin=use_sin)
        pitch_tr = archimedean_pitch(r_train, b=fit.value)
        pitch_te = archimedean_pitch(r_test, b=fit.value)
    elif model == "fermat":
        fit = fit_fermat_spiral(psi_train, r_train, scaling=scaling, use_sin=use_sin)
        pitch_tr = fermat_pitch(r_train, a=fit.value)
        pitch_te = fermat_pitch(r_test, a=fit.value)
    elif model == "log":
        fit = fit_log_spiral_pitch(psi_train, r_train, use_sin=use_sin)  # no 'scaling' in this implementation
        pitch_tr = log_spiral_pitch(r_train, k=fit.value)
        pitch_te = log_spiral_pitch(r_test, k=fit.value)
    else:
        raise ValueError(model)

    pred_tr = predict_phi(theta=theta_train, pitch_rad=pitch_tr)
    pred_te = predict_phi(theta=theta_test, pitch_rad=pitch_te)

    tr_mae = _mae_deg_linefield(phi_train, pred_tr)
    te_mae = _mae_deg_linefield(phi_test, pred_te)
    tr_rss = _rss_linefield(phi_train, pred_tr)
    te_rss = _rss_linefield(phi_test, pred_te)
    return tr_mae, te_mae, tr_rss, te_rss


def run_linefield_cv_parametric(
    csv_file: str,
    *,
    n_splits: int = 20,
    seed: int = 42,
    scaling: bool = True,
    use_sin: bool = False,
) -> dict[str, dict[str, float]]:
    """
    CV (train/test) for parametric pitch families from parametric.py.
    Returns dict:
      model -> {train_mae, test_mae, train_rss, test_rss}
    """
    df = load_angle_coordinate_csv(csv_file)
    data = build_spiral_dataset(df)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = ["archimedean", "fermat", "log"]
    acc = {
        m: {"train_mae": 0.0, "test_mae": 0.0, "train_rss": 0.0, "test_rss": 0.0}
        for m in models
    }
    folds = 0

    for train_idx, test_idx in kf.split(data.x):
        r_tr = data.r[train_idx]
        r_te = data.r[test_idx]
        theta_tr = data.theta[train_idx]
        theta_te = data.theta[test_idx]
        phi_tr = data.phi_rad[train_idx]
        phi_te = data.phi_rad[test_idx]

        psi_tr = relative_pitch(sample_theta=phi_tr, sample_phi_spatial=theta_tr)
        psi_te = relative_pitch(sample_theta=phi_te, sample_phi_spatial=theta_te)

        for m in models:
            tr_mae, te_mae, tr_rss, te_rss = _fit_predict_parametric_pitch(
                m,
                psi_train=psi_tr,
                r_train=r_tr,
                theta_train=theta_tr,
                phi_train=phi_tr,
                psi_test=psi_te,
                r_test=r_te,
                theta_test=theta_te,
                phi_test=phi_te,
                scaling=scaling,
                use_sin=use_sin,
            )
            acc[m]["train_mae"] += tr_mae
            acc[m]["test_mae"] += te_mae
            acc[m]["train_rss"] += tr_rss
            acc[m]["test_rss"] += te_rss

        folds += 1

    denom = max(folds, 1)
    for m in models:
        for k in acc[m]:
            acc[m][k] /= denom

    print("\nParametric CV summary (mean over folds; lower is better):")
    for m in sorted(models, key=lambda mm: acc[mm]["test_rss"]):
        print(f"  {m:11s} test_RSS={acc[m]['test_rss']:.6g}  test_MAE={acc[m]['test_mae']:.4f}°")

    # Plot test RSS bars
    plt.figure(figsize=(8, 4))
    names = models
    vals = [acc[m]["test_rss"] for m in names]
    plt.bar(names, vals)
    plt.ylabel("Mean Test RSS (rad²)")
    plt.title("Parametric families (CV): Test RSS")
    plt.tight_layout()
    plt.show()

    return acc

def plot_streamplot_overlay_from_bw(
    csv_file: str,
    bandwidth: float,
    *,
    grid_n: int = 140,
    density_bg: float = 1.6,
    density_fit: float = 2.2,
    title: str | None = None,
) -> None:
    """Streamplot overlay: df/original field (background) + fitted field (overlay)."""
    df = load_angle_coordinate_csv(csv_file)
    data = build_spiral_dataset(df)

    # Fit most-generalized field using the chosen bandwidth
    psi_fitted = smooth_line_field(
        target_r=data.r,
        sample_r=data.r,
        sample_theta=data.phi_rad,
        sample_phi_spatial=data.theta,
        bandwidth=float(bandwidth),
    )
    phi_fitted = data.theta + psi_fitted
    u_fit = np.cos(phi_fitted)
    v_fit = np.sin(phi_fitted)

    # Grid for streamplot
    xi = np.linspace(float(data.x.min()), float(data.x.max()), grid_n)
    yi = np.linspace(float(data.y.min()), float(data.y.max()), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate BOTH fields onto the grid
    u_bg = griddata((data.x, data.y), data.u, (Xi, Yi), method="linear")
    v_bg = griddata((data.x, data.y), data.v, (Xi, Yi), method="linear")
    u_fg = griddata((data.x, data.y), u_fit, (Xi, Yi), method="linear")
    v_fg = griddata((data.x, data.y), v_fit, (Xi, Yi), method="linear")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))

    # Background: df/original streamlines
    ax.streamplot(
        Xi, Yi, u_bg, v_bg,
        density=density_bg,
        color="lightgray",
        linewidth=1.0,
        arrowsize=0.8,
        zorder=1,
    )

    # Overlay: fitted streamlines
    ax.streamplot(
        Xi, Yi, u_fg, v_fg,
        density=density_fit,
        color="teal",
        linewidth=1.2,
        arrowsize=0.9,
        zorder=2,
    )

    ax.scatter(data.x, data.y, s=6, c="k", alpha=0.12, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Streamplot overlay (df bg + fitted overlay), bw={bandwidth:g}")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "linefield"], default="synthetic")
    ap.add_argument("--csv", type=str, default="/Users/rishabhkumar/spiral-chirals/src/experiments/experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv")

    ap.add_argument("--bw-min", type=float, default=0.1)
    ap.add_argument("--bw-max", type=float, default=100.0)
    ap.add_argument("--bw-n", type=int, default=80)
    ap.add_argument("--cv-splits", type=int, default=20)

    ap.add_argument("--test-kernels", action="store_true", help="Run CV for gaussian+uniform (+multiplicative if --kappas provided).")
    ap.add_argument("--kappas", type=str, default="0,1,2,4,8,16", help="Comma-separated kappa grid for multiplicative kernel.")
    ap.add_argument("--no-multiplicative", action="store_true", help="Skip multiplicative kernel even if kappas given.")

    ap.add_argument("--plot-overlay", action="store_true")
    ap.add_argument("--grid-n", type=int, default=140)

    args = ap.parse_args()

    if args.mode == "synthetic":
        run_synthetic_regression(bandwidth=1.0, poly_degree=2)
        return

    bandwidths = np.linspace(args.bw_min, args.bw_max, args.bw_n)
    kappas = None
    if not args.no_multiplicative:
        kappas = np.array([float(s.strip()) for s in args.kappas.split(",") if s.strip()])

    if args.test_kernels:
        results = run_linefield_cv_kernels(
            args.csv,
            bandwidths=bandwidths,
            n_splits=args.cv_splits,
            kappas=kappas,
        )

        for r in results:
            print(f"✓ {r.name} best_by_MAE: {r.best_params_mae} | best_by_RSS: {r.best_params_rss}")

        plot_kernel_cv_results(results)
        print_kernel_rss_leaderboard(results)
        plot_kernel_cv_results_rss(results)

        run_linefield_cv_parametric(
            args.csv,
            n_splits=args.cv_splits,
            seed=42,
            scaling=True,
            use_sin=False,
        )
        # If you want overlay of the best model: use gaussian best by default
        if args.plot_overlay:
            # pick best overall (by min mean test MAE)
            best_name = None
            best_score = float("inf")
            best_params: dict | None = None
            for r in results:
                if r.name in ("gaussian", "uniform"):
                    score = float(np.min(r.test_means))
                    if score < best_score:
                        best_score, best_name, best_params = score, r.name, r.best_params
                elif r.name == "multiplicative":
                    score = float(np.min(r.test_means))
                    if score < best_score:
                        best_score, best_name, best_params = score, r.name, r.best_params

            print(f"✓ Best overall by mean test MAE: {best_name} params={best_params} (MAE={best_score:.4f}°)")

            # For overlay: currently uses gaussian kernel internally.
            # If you want overlay to respect kernel type, tell me and I’ll extend plot_streamplot_overlay_from_bw()
            plot_streamplot_overlay_from_bw(
                csv_file=args.csv,
                bandwidth=float(best_params.get("bandwidth", best_params.get("sigma"))),
                grid_n=args.grid_n,
                title=f"Best overall (by test MAE): {best_name} {best_params}",
            )
    else:
        # fallback to your previous gaussian-only CV
        summary = run_linefield_cv(args.csv, bandwidths=bandwidths, n_splits=args.cv_splits)
        if args.plot_overlay:
            plot_streamplot_overlay_from_bw(
                csv_file=args.csv,
                bandwidth=summary.min_test_mae_bw,
                grid_n=args.grid_n,
                title="Most-generalized vector field (min test MAE bw): df bg + fitted overlay",
            )

    best_kernel_name = None
    best_kernel_rss = float("inf")
    best_kernel_bw = None
    for r in results:
        if r.name in ("gaussian", "uniform"):
            i = int(np.argmin(r.test_rss))
            val = float(r.test_rss[i])
            if val < best_kernel_rss:
                best_kernel_rss = val
                best_kernel_name = r.name
                best_kernel_bw = float(r.x1[i])
        else:
            flat = int(np.argmin(r.test_rss))
            bi, bj = np.unravel_index(flat, r.test_rss.shape)
            val = float(r.test_rss[bi, bj])
            if val < best_kernel_rss:
                best_kernel_rss = val
                best_kernel_name = r.name
                best_kernel_bw = float(r.x1[bi])  # sigma acts like bandwidth here

    print(f"\n✓ Best kernel by mean test RSS: {best_kernel_name} bw/sigma={best_kernel_bw:g}  (RSS={best_kernel_rss:.6g})")
    compare_linefield_models_rss(args.csv, kernel_bw=float(best_kernel_bw))

            
if __name__ == "__main__":
    main()