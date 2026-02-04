# spiral-chirals

> [!NOTE]
> This is a consulting project, currently in progress. 

Research codebase for **spiral vector/line-field analysis**, with utilities for:
- Loading coordinate + angle datasets
- Kernel regression on line fields (double-angle trick)
- Parametric spiral fitting (log, Archimedean, Fermat)
- Bayesian model comparison (BIC and GP/RKHS evidence)
- Smooth vector field fitting via Fourier bases

The reusable software package lives under src/spiral_chirals.

## Quick start

1) Install runtime dependencies:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

2) Load data and fit a smoothed line field:

```python
from pathlib import Path
from spiral_chirals import load_angle_coordinate_csv, build_spiral_dataset
from spiral_chirals import smooth_line_field, angle_residual_line_field

csv_path = Path("vf_exports/Front_EE-1_1_3000x_rings_coords.csv")
df = load_angle_coordinate_csv(csv_path)
data = build_spiral_dataset(df)

psi_fitted = smooth_line_field(
	target_r=data.r,
	sample_r=data.r,
	sample_theta=data.phi_rad,
	sample_phi_spatial=data.theta,
	bandwidth=1.0,
)

theta_fitted = data.theta + psi_fitted
residuals = angle_residual_line_field(data.phi_rad, theta_fitted)
```

3) Fit a parametric spiral model:

```python
from spiral_chirals import fit_log_spiral, fit_fermat_spiral, fit_archimedean_spiral

res_log = fit_log_spiral(data.angle_rad, data.r, k0=1.0, scaling=True)
res_fermat = fit_fermat_spiral(data.angle_rad, data.r, a0=0.1, scaling=True)
res_arch = fit_archimedean_spiral(data.angle_rad, data.r, b0=1.0, scaling=True)
```

4) Compare parametric models:

```python
from spiral_chirals import SpiralBayes

analysis = SpiralBayes(
	coordinates=list(zip(data.x, data.y)),
	vectors=list(zip(data.u, data.v)),
)
results = analysis.compare_spirals()
```

## Package structure

The reusable code is organized as a small research package.

```
src/
  spiral_chirals/
	__init__.py
	types.py
	io.py
	geometry.py
	kernels.py
	parametric.py
	basis.py
	bayes.py
	visualization.py
```

### Module overview

- types.py: `SpiralDataset` dataclass for standardized data objects.
- io.py: CSV loading + coordinate parsing + dataset construction.
- geometry.py: Polar conversion, angle wrapping, residuals.
- kernels.py: Gaussian kernel and double-angle smoothing utilities.
- parametric.py: Log/Fermat/Archimedean pitch models and fit wrappers.
- basis.py: Smooth vector field fitting via low-frequency Fourier bases.
- bayes.py: Parametric model fitting and Bayesian model comparison.
- visualization.py: Quiver/streamline plotting helpers.

## Data format

Expected CSV columns (default):
- Coordinate: a string like "(x, y)" or "[x, y]"
- Angle (α′): relative pitch angle in degrees

You can override column names in load_angle_coordinate_csv and build_spiral_dataset.

## Notes

- The line-field kernels use the **double-angle embedding**, appropriate for axial data.
- The parametric fits implement the same objective forms used in the notebooks.
- Bayesian comparison uses a BIC approximation for parametric models and an RBF GP marginal likelihood for the non-parametric baseline.

## License

Internal research code. Add a license if you plan to share publicly.

