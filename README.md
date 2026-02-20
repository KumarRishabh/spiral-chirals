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
pip install numpy pandas scipy matplotlib scikit-learn pyplot anywidgets
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

## Experiment script — `src/experiments/nonparametric_parametric.py`

This is the main CLI entry point for running kernel CV, parametric CV, model comparison, and streamplot visualisation. All modes are invoked via `--mode linefield`.

### 1. Multi-file K-Fold CV (model selection across all datasets)

Runs K-Fold cross-validation for every kernel and every parametric family over **all `*.csv` files** in a directory, averages the test RSS across files, and prints a ranked leaderboard + bar chart.

```bash
python src/experiments/nonparametric_parametric.py \
  --mode linefield \
  --vf-exports-dir vf_exports \
  --no-multiplicative \
  --bw-min 0.1 --bw-max 100 --bw-n 80 \
  --cv-splits 20
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--vf-exports-dir DIR` | — | Directory of `*.csv` files to evaluate over |
| `--bw-min` / `--bw-max` / `--bw-n` | 0.1 / 100 / 80 | Bandwidth search grid |
| `--cv-splits` | 20 | Number of K-Fold splits per file |
| `--no-multiplicative` | off | Skip the multiplicative kernel (faster) |
| `--kappas` | `"0,1,2,4,8,16"` | Von Mises κ grid for the multiplicative kernel |

**What it produces:**
- Per-file best test RSS for each model
- A global leaderboard sorted by mean test RSS across all files
- A bar chart: blue bars = nonparametric kernels, red bars = parametric families

**Example leaderboard output (10 files, 20-fold CV):**

```
========================================================================
MULTI-FILE LEADERBOARD  (mean test RSS across files; lower is better)
========================================================================
  kernel:uniform                       mean_RSS=4.53   (n_files=10)
  kernel:gaussian                      mean_RSS=4.55   (n_files=10)
  parametric:fermat                    mean_RSS=6.20   (n_files=10)
  parametric:log                       mean_RSS=6.37   (n_files=10)
  parametric:archimedean               mean_RSS=6.50   (n_files=10)
```

To include the multiplicative (radial × von Mises angular) kernel in the competition:

```bash
python src/experiments/nonparametric_parametric.py \
  --mode linefield \
  --vf-exports-dir vf_exports \
  --bw-min 0.1 --bw-max 100 --bw-n 80 \
  --cv-splits 20 \
  --kappas "0,1,2,4,8,16,32,64"
```

### 2. Single-file kernel CV + streamplot of the best estimator

Runs CV on a single CSV, identifies the best nonparametric kernel (by mean test MAE or RSS), and renders a streamplot overlay: the **observed field** as grey background streamlines and the **fitted field** in teal.

```bash
python src/experiments/nonparametric_parametric.py \
  --mode linefield \
  --csv "src/experiments/experiments/cleaned_front_ee_1_1_3000x_rings_coords.csv" \
  --no-multiplicative \
  --bw-min 0.1 --bw-max 100 --bw-n 80 \
  --cv-splits 20 \
  --test-kernels \
  --plot-overlay \
  --best-by mae
```

To select the best model by RSS instead of MAE, change `--best-by mae` → `--best-by rss`.

**What it produces:**
- CV curves (train vs test MAE and RSS) for each kernel
- A printed RSS leaderboard for the single file
- A parametric family CV comparison (Archimedean / Fermat / Log)
- A **streamplot overlay** for the winning kernel + bandwidth

#### How the streamplot works

The fitted field is evaluated directly on a fine grid (not interpolated from sample predictions):

```
r_grid, θ_grid  ←  polar coordinates of each grid point
ψ_grid          ←  smooth_line_field(target_r=r_grid, sample_r=data.r, ...)
φ_grid          =  θ_grid + ψ_grid
(U, V)          =  (cos φ_grid, sin φ_grid)
```

This means the streamplot respects the kernel type and bandwidth exactly, including the multiplicative kernel's angular concentration parameter κ.

### 3. Without the multiplicative kernel

Pass `--no-multiplicative` to restrict CV to only the **Gaussian** and **Uniform** radial kernels. This roughly halves runtime and is the recommended starting point:

```bash
# Multi-file, no multiplicative
python src/experiments/nonparametric_parametric.py \
  --mode linefield \
  --vf-exports-dir vf_exports \
  --no-multiplicative \
  --bw-min 0.1 --bw-max 100 --bw-n 80 \
  --cv-splits 20

# Single-file with streamplot, no multiplicative
python src/experiments/nonparametric_parametric.py \
  --mode linefield \
  --csv "vf_exports/Front_EE-1_1_3000x_rings_coords.csv" \
  --no-multiplicative \
  --bw-min 0.1 --bw-max 100 --bw-n 80 \
  --cv-splits 20 \
  --test-kernels \
  --plot-overlay \
  --best-by rss
```

### Kernel descriptions

#### Gaussian radial kernel
Nadaraya–Watson kernel regression using a Gaussian weight decaying with radial distance:

$$w_i = \exp\!\left(-\frac{(r - r_i)^2}{2\,h^2}\right)$$

Smoothly downweights distant samples. `--bw-min/max/n` controls the bandwidth $h$ search grid.

#### Uniform radial kernel
Box kernel — equal weight for all samples within bandwidth $h$, zero outside:

$$w_i = \mathbf{1}\!\left[\,|r - r_i| \le h\,\right]$$

Equivalent to a local average over a radial window.

#### Multiplicative kernel (radial × angular)
Product of a Gaussian radial term and a von Mises angular term on the **spatial angle** $\varphi$:

$$w_i = \exp\!\left(-\frac{(r - r_i)^2}{2\,\sigma^2}\right) \cdot \exp\!\left(\kappa \cos(\varphi - \varphi_i)\right)$$

Requires two hyperparameters: radial bandwidth $\sigma$ and angular concentration $\kappa$. CV is performed on a 2D grid ($\sigma$ × $\kappa$). Use `--kappas` to set the $\kappa$ candidates.

#### Double-angle embedding (all kernels)
All kernels regress the **line-field pitch** $\psi$ via the double-angle trick to handle the $\pi$-periodicity of line directions:

$$\hat{\psi} = \tfrac{1}{2}\,\mathrm{atan2}\!\left(\sum_i w_i \sin 2\psi_i,\; \sum_i w_i \cos 2\psi_i\right)$$

### Parametric families

| Family | Pitch model | Fitted parameter |
|---|---|---|
| Archimedean | $\psi(r) = b \cdot r$ | $b$ |
| Fermat | $\psi(r) = a / \sqrt{r}$ | $a$ |
| Log spiral | $\psi(r) = k$ (constant) | $k$ |

Fits minimise the sum of squared angular residuals (RSS) via `scipy.optimize`. CV uses the same K-Fold splits as the kernel models for a fair comparison.

### CV metric: RSS vs MAE

- **RSS** (sum of squared angular residuals, rad²): more sensitive to outliers; use for model selection when large errors are costly.
- **MAE** (mean absolute error, degrees): more robust; use when the field has occasional noisy samples.

The `--best-by {mae,rss}` flag controls which metric selects the winning bandwidth for the streamplot.

---

## Notes

- The line-field kernels use the **double-angle embedding**, appropriate for axial data.
- The parametric fits implement the same objective forms used in the notebooks.
- Bayesian comparison uses a BIC approximation for parametric models and an RBF GP marginal likelihood for the non-parametric baseline.
- Files in `vf_exports/` that do not contain the expected `Angle (α′)` column are skipped gracefully with a warning.

## Research workflow & TODOs

Use this checklist to keep the prediction pipeline reproducible and accurate.

### Data & QC
- [ ] Add a data inventory table (file name, sample count, missing rows).
- [ ] Standardize coordinate parsing errors into a report.
- [ ] Validate angle range and wrap to $(-\pi/2, \pi/2]$ for line-field use.

### Baselines
- [ ] Reproduce parametric fits (log, Archimedean, Fermat) on all datasets.
- [ ] Save residual histograms and Q-Q plots for each model.
- [ ] Track best model via BIC and Bayes factors.

### Non‑parametric models
- [x] Cross‑validate kernel bandwidths for line‑field smoothing (multi-file K-Fold CV, `--vf-exports-dir`).
- [x] Compare Gaussian vs Uniform kernels (Epanechnikov can be added via `kernels.py`).
- [ ] Add uncertainty bands to $\hat{\psi}(r)$ using bootstrap resampling.

### Vector field reconstruction
- [ ] Compare headless quiver vs streamline rendering for stability.
- [ ] Add a grid‑based interpolation strategy (linear vs RBF).
- [ ] Quantify vector field error using angular residual metrics.

### Evaluation
- [ ] Define a standard metric suite (MAE in degrees, circular RMSE, BIC, LML).
- [x] Multi-file CV leaderboard (`--vf-exports-dir`) prints ranked RSS summary; extend with `--save-csv` if needed.

### Packaging
- [ ] Add pyproject.toml with versioning and dependencies.
- [ ] Add a minimal CLI for loading data and running a selected model.
- [ ] Add tests for `parse_coord`, wrapping, and kernel smoothing.

## License

Internal research code. Add a license if you plan to share publicly.

