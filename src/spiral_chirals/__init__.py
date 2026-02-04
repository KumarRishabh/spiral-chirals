"""Spiral vector-field analysis and fitting."""

from .types import SpiralDataset
from .io import load_angle_coordinate_csv, build_spiral_dataset
from .geometry import (
    to_polar,
    wrap_angle_pi,
    wrap_angle_half_pi,
    angle_residual_line_field,
    relative_pitch,
    vector_from_angle,
)
from .kernels import gaussian_kernel, smooth_line_field, smooth_spiral_pitch
from .parametric import (
    log_spiral_pitch,
    fermat_pitch,
    archimedean_pitch,
    fit_log_spiral,
    fit_fermat_spiral,
    fit_archimedean_spiral,
    predict_phi,
)
from .basis import (
    make_grid,
    smooth_radial,
    smooth_spiral,
    add_noise,
    fourier_basis_2d,
    fourier_penalty,
    ridge_fit,
    fit_vector_field_fourier,
)
from .bayes import (
    SpiralBayes,
    kernel_matrix,
    calculate_gp_lml,
    parametric_log_evidence,
    perform_bayesian_comparison,
)
from .visualization import (
    plot_line_field_quiver,
    plot_streamlines,
    plot_structure_function,
    plot_residual_hist,
)

__all__ = [
    "SpiralDataset",
    "load_angle_coordinate_csv",
    "build_spiral_dataset",
    "to_polar",
    "wrap_angle_pi",
    "wrap_angle_half_pi",
    "angle_residual_line_field",
    "relative_pitch",
    "vector_from_angle",
    "gaussian_kernel",
    "smooth_line_field",
    "smooth_spiral_pitch",
    "log_spiral_pitch",
    "fermat_pitch",
    "archimedean_pitch",
    "fit_log_spiral",
    "fit_fermat_spiral",
    "fit_archimedean_spiral",
    "predict_phi",
    "make_grid",
    "smooth_radial",
    "smooth_spiral",
    "add_noise",
    "fourier_basis_2d",
    "fourier_penalty",
    "ridge_fit",
    "fit_vector_field_fourier",
    "SpiralBayes",
    "kernel_matrix",
    "calculate_gp_lml",
    "parametric_log_evidence",
    "perform_bayesian_comparison",
    "plot_line_field_quiver",
    "plot_streamlines",
    "plot_structure_function",
    "plot_residual_hist",
]
