"""
===============================================================================
DyeFit: Spectroscopic Baseline Correction & Deconvolution Library
===============================================================================
This package provides modular tools for preprocessing and analyzing 
spectroscopic absorption profiles of scientific dyes.

Modules exposed:
  - `dyes`: Registry profiles for standard dye configurations.
  - `baseline`: Polynomial, Linear, and Asymmetric Least Squares (ALS) baselines.
  - `models`: Peak math profiles (Gaussian, Lorentzian, Voigt, Pseudo-Voigt, etc.).
  - `fitting`: Automated peak detection and non-linear regression solvers.
  - `reporting`: Statistical evaluators, numerical integration, and plot outputs.
"""

from .dyes import get_dye_config, DYE_REGISTRY, DyeConfig, reload_dye_registry, save_dye_to_db
from .baseline import polynomial_baseline, linear_baseline, als_baseline, constant_offset_single, constant_offset_region
from .models import gaussian, lorentzian, voigt, pseudo_voigt, asym_gaussian
from .fitting import (
    fit_spectrum, 
    get_initial_guesses, 
    get_bounds, 
    auto_detect_peaks, 
    get_num_params, 
    combined_model_systematic,
    curve_fit_wrapper
)
from .reporting import (
    calculate_r_squared, 
    calculate_rmse, 
    calculate_aic, 
    calculate_bic, 
    calculate_auc,
    generate_matplotlib_fig,
    generate_plotly_fig
)

__version__ = "0.4.0"
__author__ = "DyeFit Developers"
