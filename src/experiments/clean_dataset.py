## Clean up the outer and inner circles in the dataset and then do parametric and kernel fits
import json
from pathlib import Path
import pandas as pd
import numpy as np
from spiral_chirals.io import load_angle_coordinate_csv, build_spiral_dataset
from spiral_chirals.experiments import ExperimentConfig, ExperimentRunner, KernelFitConfig, ParametricFitConfig
import matplotlib.pyplot as plt

# Load raw data
csv_file = "/Users/rishabhkumar/spiral-chirals/vf_exports/Front_EE-1_1_3000x_rings_coords.csv"
df = load_angle_coordinate_csv(csv_file)
# Clean dataset by removing outer and inner circles
def clean_spiral_dataset(df: pd.DataFrame, inner_radius: float, outer_radius: float) -> pd.DataFrame:
    data = build_spiral_dataset(df)
    mask = (data.r >= inner_radius) & (data.r <= outer_radius)
    cleaned_df = df[mask].reset_index(drop=True)
    return cleaned_df
cleaned_df = clean_spiral_dataset(df, inner_radius=10.0, outer_radius=290.0)
# Save cleaned dataset to a new CSV
cleaned_csv_file = Path("experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv")
cleaned_csv_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
cleaned_df.to_csv(cleaned_csv_file, index=False)
print(f"Cleaned dataset saved to {cleaned_csv_file}")

data_original = build_spiral_dataset(df)

# Cleaned data
data_cleaned = build_spiral_dataset(cleaned_df)
data_original = build_spiral_dataset(df)

# Create mask for removed points
mask_kept = (data_original.r >= 10.0) & (data_original.r <= 290.0)
mask_removed = ~mask_kept

# Extract removed and kept data
x_removed = data_original.x[mask_removed]
y_removed = data_original.y[mask_removed]
u_removed = data_original.u[mask_removed]
v_removed = data_original.v[mask_removed]

x_kept = data_original.x[mask_kept]
y_kept = data_original.y[mask_kept]
u_kept = data_original.u[mask_kept]
v_kept = data_original.v[mask_kept]

# Plot in same figure
fig, ax = plt.subplots(figsize=(10, 10))

# Removed points (outer/inner circles) in red
ax.quiver(x_removed, y_removed, u_removed, v_removed,
          pivot='mid', headwidth=0, headlength=0, headaxislength=0,
          scale=25, width=0.004, color='red', alpha=0.5, label='Removed')
ax.scatter(x_removed, y_removed, s=8, c='red', alpha=0.2)

# Kept points (cleaned dataset) in blue
ax.quiver(x_kept, y_kept, u_kept, v_kept,
          pivot='mid', headwidth=0, headlength=0, headaxislength=0,
          scale=25, width=0.004, color='blue', alpha=0.6, label='Kept')
ax.scatter(x_kept, y_kept, s=8, c='blue', alpha=0.3)

ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Cleaned Dataset: Removed vs Kept Points')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

config = ExperimentConfig(
    name="baseline_clean_data",
    csv_file=str(cleaned_csv_file.resolve()),
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