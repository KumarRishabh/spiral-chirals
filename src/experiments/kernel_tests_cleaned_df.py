import json
from pathlib import Path
from spiral_chirals.experiments import ExperimentConfig, ExperimentRunner, KernelFitConfig
from spiral_chirals.io import build_spiral_dataset, load_angle_coordinate_csv
from spiral_chirals.kernels import smooth_line_field
from spiral_chirals.geometry import angle_residual_line_field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split, KFold
from spiral_chirals.types import SpiralDataset
# Load cleaned data once
# csv_file = "/Users/rishabhkumar/spiral-chirals/vf_exports/Front_EE-1_1_3000x_rings_coords.csv"
# df = load_angle_coordinate_csv(csv_file)
# data = build_spiral_dataset(df)
csv_file = "experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv"
df = pd.read_csv("experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv")
data = build_spiral_dataset(df)
# Test multiple bandwidths
bandwidths = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
results = []

print("Testing kernel fits with multiple bandwidths...\n")

for bw in bandwidths:
    # Fit smoothed line field
    psi_fitted = smooth_line_field(
        target_r=data.r,
        sample_r=data.r,
        sample_theta=data.phi_rad,
        sample_phi_spatial=data.theta,
        bandwidth=bw,
    )
    theta_fitted = data.theta + psi_fitted
    residuals = angle_residual_line_field(data.phi_rad, theta_fitted)
    res_deg = np.degrees(residuals)
    
    # Compute metrics
    mae = np.mean(np.abs(res_deg))
    rmse = np.sqrt(np.mean(res_deg**2))
    std_res = np.std(res_deg)
    
    # BIC for model selection (lower is better)
    # BIC = n*ln(RSS/n) + k*ln(n), where k is number of parameters
    rss = np.sum(residuals**2)
    n = len(residuals)
    k = 1  # bandwidth is the only parameter
    bic = n * np.log(rss / n) + k * np.log(n)
    
    results.append({
        'bandwidth': bw,
        'mae_deg': mae,
        'rmse_deg': rmse,
        'std_residual_deg': std_res,
        'bic': bic,
        'rss': rss,
    })
    
    print(f"Bandwidth: {bw:4.1f} | MAE: {mae:6.3f}° | RMSE: {rmse:6.3f}° | BIC: {bic:10.1f}")

# Create results dataframe
results_df = pd.DataFrame(results)
print(f"\n{'='*70}")

# Find best model by BIC
best_idx = results_df['bic'].idxmin()
best_bw = results_df.loc[best_idx, 'bandwidth']
print(f"\n✓ Best bandwidth by BIC: {best_bw} (BIC = {results_df.loc[best_idx, 'bic']:.1f})")
print(f"  MAE: {results_df.loc[best_idx, 'mae_deg']:.4f}°")
print(f"  RMSE: {results_df.loc[best_idx, 'rmse_deg']:.4f}°")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MAE vs Bandwidth
axes[0, 0].plot(results_df['bandwidth'], results_df['mae_deg'], 'o-', color='blue', linewidth=2, markersize=8)
axes[0, 0].axvline(best_bw, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_bw}')
axes[0, 0].set_xlabel('Bandwidth')
axes[0, 0].set_ylabel('MAE (degrees)')
axes[0, 0].set_title('Mean Absolute Error vs Bandwidth')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# RMSE vs Bandwidth
axes[0, 1].plot(results_df['bandwidth'], results_df['rmse_deg'], 'o-', color='green', linewidth=2, markersize=8)
axes[0, 1].axvline(best_bw, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_bw}')
axes[0, 1].set_xlabel('Bandwidth')
axes[0, 1].set_ylabel('RMSE (degrees)')
axes[0, 1].set_title('Root Mean Squared Error vs Bandwidth')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

# BIC vs Bandwidth (model selection criterion)
axes[1, 0].plot(results_df['bandwidth'], results_df['bic'], 'o-', color='purple', linewidth=2, markersize=8)
axes[1, 0].axvline(best_bw, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_bw}')
axes[1, 0].set_xlabel('Bandwidth')
axes[1, 0].set_ylabel('BIC')
axes[1, 0].set_title('BIC vs Bandwidth (Lower is Better)')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

# Std Residual vs Bandwidth
axes[1, 1].plot(results_df['bandwidth'], results_df['std_residual_deg'], 'o-', color='orange', linewidth=2, markersize=8)
axes[1, 1].axvline(best_bw, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_bw}')
axes[1, 1].set_xlabel('Bandwidth')
axes[1, 1].set_ylabel('Std Dev of Residuals (degrees)')
axes[1, 1].set_title('Residual Std Dev vs Bandwidth')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('bandwidth_selection.png', dpi=150)
print("\n✓ Model selection plots saved to bandwidth_selection.png")
plt.show()

# Save results to CSV
results_df.to_csv('bandwidth_comparison.csv', index=False)
print("✓ Results saved to bandwidth_comparison.csv")

# Now run baseline experiment with best bandwidth
print(f"\n{'='*70}")
print(f"Running baseline experiment with best bandwidth: {best_bw}")

config = ExperimentConfig(
    name="baseline_optimized_bandwidth",
    csv_file=csv_file,
    kernel_config=KernelFitConfig(bandwidth=best_bw),
    run_bayes=True,
    notes=f"Optimized: bandwidth={best_bw} selected by BIC",
)

runner = ExperimentRunner(output_dir="experiments")
result = runner.run(config)
print("✓ Experiment completed")
print(f"  Samples: {result['n_samples']}")
print(f"  Kernel MAE: {result['kernel_results']['mae_deg']:.4f}°")
print(f"  Best model: {result['bayes_results'][0]['Model']}")

# Visualize best fit
df = load_angle_coordinate_csv(csv_file)
data = build_spiral_dataset(df)

psi_fitted = smooth_line_field(
    target_r=data.r,
    sample_r=data.r,
    sample_theta=data.phi_rad,
    sample_phi_spatial=data.theta,
    bandwidth=best_bw,
)
theta_fitted = data.theta + psi_fitted
phi_fitted = theta_fitted

U_fit = np.cos(phi_fitted)
V_fit = np.sin(phi_fitted)

xi_smooth = np.linspace(data.x.min(), data.x.max(), 100)
yi_smooth = np.linspace(data.y.min(), data.y.max(), 100)
Xi_smooth, Yi_smooth = np.meshgrid(xi_smooth, yi_smooth)

Ui_smooth = griddata((data.x, data.y), U_fit, (Xi_smooth, Yi_smooth), method='linear')
Vi_smooth = griddata((data.x, data.y), V_fit, (Xi_smooth, Yi_smooth), method='linear')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.quiver(data.x, data.y, U_fit, V_fit,
           pivot='mid', headwidth=0, headlength=0, headaxislength=0,
           scale=25, width=0.004, color='purple', alpha=0.7)
ax1.scatter(data.x, data.y, s=8, c='k', alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Fitted Line Field (Bandwidth={best_bw})')
ax1.grid(alpha=0.3)

ax2.streamplot(Xi_smooth, Yi_smooth, Ui_smooth, Vi_smooth,
               density=2.0, color='teal', linewidth=1)
ax2.scatter(data.x, data.y, s=5, c='red', alpha=0.2)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Streamlines from Smoothed Field')
ax2.grid(alpha=0.3)

plt.tight_layout()

exp_dir = Path(runner.output_dir) / config.name
plot_path = exp_dir / "streamlines.png"
plt.savefig(plot_path, dpi=150)
print(f"✓ Streamlines plot saved to {plot_path}")
plt.show()


### Model selection using Train/Test Split ###

kf = KFold(n_splits=50, shuffle=True, random_state=42)

cv_data = {bw: {'train_mae': [], 'test_mae': []} for bw in bandwidths}

for train_idx, test_idx in kf.split(data.x):
    train_set = SpiralDataset(
        x=data.x[train_idx], y=data.y[train_idx], r=data.r[train_idx],
        theta=data.theta[train_idx], angle_deg=data.angle_deg[train_idx],
        angle_rad=data.angle_rad[train_idx], phi_rad=data.phi_rad[train_idx],
        u=data.u[train_idx], v=data.v[train_idx]
    )
    test_set = SpiralDataset(
        x=data.x[test_idx], y=data.y[test_idx], r=data.r[test_idx],
        theta=data.theta[test_idx], angle_deg=data.angle_deg[test_idx],
        angle_rad=data.angle_rad[test_idx], phi_rad=data.phi_rad[test_idx],
        u=data.u[test_idx], v=data.v[test_idx]
    )
    
    for bw in bandwidths:
        # Fit on training
        psi_train = smooth_line_field(train_set.r, train_set.r, train_set.phi_rad, train_set.theta, bw)
        train_mae = np.mean(np.abs(np.degrees(angle_residual_line_field(train_set.phi_rad, train_set.theta + psi_train))))
        
        # Test on test set
        psi_test = smooth_line_field(test_set.r, train_set.r, train_set.phi_rad, train_set.theta, bw)
        test_mae = np.mean(np.abs(np.degrees(angle_residual_line_field(test_set.phi_rad, test_set.theta + psi_test))))
        
        cv_data[bw]['train_mae'].append(train_mae)
        cv_data[bw]['test_mae'].append(test_mae)

# Plot train vs test
train_means = [np.mean(cv_data[bw]['train_mae']) for bw in bandwidths]
test_means = [np.mean(cv_data[bw]['test_mae']) for bw in bandwidths]

plt.figure(figsize=(10, 6))
plt.plot(bandwidths, train_means, 'o-', label='Train MAE', linewidth=2)
plt.plot(bandwidths, test_means, 's-', label='Test MAE', linewidth=2)
plt.fill_between(bandwidths, train_means, test_means, alpha=0.2, color='red')
plt.xlabel('Bandwidth')
plt.ylabel('MAE (degrees)')
plt.title('Overfitting Detection: Train vs Test Error')
plt.legend()
plt.grid(alpha=0.3)
plt.show()