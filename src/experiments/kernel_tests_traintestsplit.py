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
# csv_file = "experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv"
# df = pd.read_csv("experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv")
# Use the uncleaned data for overfitting detection
csv_file = "/Users/rishabhkumar/spiral-chirals/vf_exports/Front_EE-1_1_3000x_rings_coords.csv"
df = load_angle_coordinate_csv(csv_file)
data = build_spiral_dataset(df)
# Test multiple bandwidths
# bandwidths = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
# Choose a finer grid of bandwidths for better resolution
bandwidths = np.linspace(0.1, 100, 2000)
results = []
gaps = []
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
gaps = [test_means[i] - train_means[i] for i in range(len(bandwidths))]

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
### Print the results ###   
print("\n✓ Train/Test Split Results:"
      "\nBandwidth\tTrain MAE\tTest MAE\tGap (Test - Train)")
for i, bw in enumerate(bandwidths):
    print(f"{bw:.2f}\t\t{train_means[i]:.4f}\t\t{test_means[i]:.4f}\t\t{gaps[i]:.4f}")


### After fitting the model with the most general bandwidth on the full dataset ###
# Select the best bandwith based on minimum gap between train and test, while keeping the bandwith small
gaps = [test_means[i] - train_means[i] for i in range(len(bandwidths))]
best_idx = np.argmin([gap if gap >= 0 else np.inf for gap in gaps])
best_bw = bandwidths[best_idx]
print(f"\n✓ Selected best bandwidth based on Train/Test split: {best_bw}")


### Select and plot the minimal test MAE bandwidth by passing a line through the gaps ###
min_test_mae_idx = np.argmin(test_means)
min_test_mae_bw = bandwidths[min_test_mae_idx]
print(f"✓ Bandwidth with minimal Test MAE: {min_test_mae_bw}")
plt.figure(figsize=(10, 6))
plt.plot(bandwidths, test_means, 'o-', label='Test MAE', linewidth=2)
plt.axvline(best_bw, color='green', linestyle='--', label='Selected Best BW')
plt.axvline(min_test_mae_bw, color='red', linestyle='--', label='Min Test MAE BW')
plt.xlabel('Bandwidth')    

### Plot the fit with the min_test_mae_bw ###
psi_fit = smooth_line_field(data.r, data.r, data.phi_rad, data.theta, min_test_mae_bw)
phi_fit = data.theta + psi_fit
residuals = np.degrees(angle_residual_line_field(data.phi_rad, phi_fit))
plt.figure(figsize=(10, 6))
plt.scatter(data.r, residuals, s=10, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Radius')
# plt.ylabel('Residual (degrees)')
plt.title(f'Residuals of Kernel Fit with Bandwidth {min_test_mae_bw:.2f}')
# plt.grid(alpha=0.3)
# plt.show()
plt.title('Test MAE vs Bandwidth')
plt.ylabel('Test MAE (degrees)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Adapted code to fit the min_test_mae_bw bandwidth
df = load_angle_coordinate_csv(csv_file)
data = build_spiral_dataset(df)

psi_fitted = smooth_line_field(
    target_r=data.r,
    sample_r=data.r,
    sample_theta=data.phi_rad,
    sample_phi_spatial=data.theta,
    bandwidth=min_test_mae_bw,
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
ax1.set_title(f'Fitted Line Field (Bandwidth={min_test_mae_bw})')
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

plot_path = Path("streamlines_min_test_mae.png")
plt.savefig(plot_path, dpi=150)
print(f"✓ Streamlines plot saved to {plot_path}")
plt.show()
