import json
from pathlib import Path
from spiral_chirals.experiments import ExperimentConfig, ExperimentRunner, KernelFitConfig
from spiral_chirals.io import build_spiral_dataset, load_angle_coordinate_csv
from spiral_chirals.kernels import smooth_line_field
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

config = ExperimentConfig(
    name="baseline_full_data",
    csv_file="/Users/rishabhkumar/spiral-chirals/vf_exports/Front_EE-1_1_3000x_rings_coords.csv",
    kernel_config=KernelFitConfig(bandwidth=1.0),
    run_bayes=True,
    notes="Baseline: no transforms",
)

runner = ExperimentRunner(output_dir="experiments")
result = runner.run(config)
print("✓ Experiment completed")
print(f"  Samples: {result['n_samples']}")
print(f"  Kernel MAE: {result['kernel_results']['mae_deg']:.4f}°")
print(f"  Best model: {result['bayes_results'][0]['Model']}")
print("Experiment completed. Results:")
print(json.dumps(result, indent=2))

df = load_angle_coordinate_csv(config.csv_file)
data = build_spiral_dataset(df)

# Fit smoothed line field
psi_fitted = smooth_line_field(
    target_r=data.r,
    sample_r=data.r,
    sample_theta=data.phi_rad,
    sample_phi_spatial=data.theta,
    bandwidth=1.0,
)
theta_fitted = data.theta + psi_fitted
phi_fitted = theta_fitted

# Get unit vectors from fitted angles
U_fit = np.cos(phi_fitted)
V_fit = np.sin(phi_fitted)

# Create interpolated grid for streamplot
xi_smooth = np.linspace(data.x.min(), data.x.max(), 100)
yi_smooth = np.linspace(data.y.min(), data.y.max(), 100)
Xi_smooth, Yi_smooth = np.meshgrid(xi_smooth, yi_smooth)

# Interpolate fitted vectors to grid
Ui_smooth = griddata((data.x, data.y), U_fit, (Xi_smooth, Yi_smooth), method='linear')
Vi_smooth = griddata((data.x, data.y), V_fit, (Xi_smooth, Yi_smooth), method='linear')

# Plot streamlines
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Quiver plot of fitted field
ax1.quiver(data.x, data.y, U_fit, V_fit,
           pivot='mid', headwidth=0, headlength=0, headaxislength=0,
           scale=25, width=0.004, color='purple', alpha=0.7)
ax1.scatter(data.x, data.y, s=8, c='k', alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Fitted Line Field (Quiver)')
ax1.grid(alpha=0.3)

# Right: Streamlines from interpolated field
ax2.streamplot(Xi_smooth, Yi_smooth, Ui_smooth, Vi_smooth,
               density=2.0, color='teal', linewidth=1)
ax2.scatter(data.x, data.y, s=5, c='red', alpha=0.2)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Streamlines from Smoothed Field')
ax2.grid(alpha=0.3)

plt.tight_layout()

# Save plot to the experiment's output directory
exp_dir = Path(runner.output_dir) / config.name
plot_path = exp_dir / "streamlines.png"
plt.savefig(plot_path, dpi=150)
print(f"\n✓ Streamlines plot saved to {plot_path}")
plt.show()

print("\nResults saved in: experiments/baseline_full_data/")
print("  - config.json")
print("  - result.json")
print("  - kernel_residuals.npy")
print("  - bayes_comparison.csv")
print("  - streamlines.png")