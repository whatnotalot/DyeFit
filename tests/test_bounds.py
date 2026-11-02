"""
Unit and regression tests verifying physicality constraints, parameter bounds,
and model selection logic in DyeFit.
"""

import numpy as np
import pytest
from dyefit.fitting import fit_spectrum, get_bounds, get_num_params
from dyefit.reporting import calculate_aic, calculate_bic, calculate_r_squared, calculate_rmse
from dyefit.kinetics import generate_demo_kinetic_data, batch_deconvolve_series, fit_pseudo_first_order
from dyefit.baseline import als_baseline


def test_parameter_bounds_structure():
    """Verify that get_bounds enforces strict non-negativity and positive widths."""
    x = np.linspace(500, 750, 251)
    y = np.exp(-((x - 664.0) / 20.0) ** 2)
    
    models = ['Gaussian', 'Lorentzian', 'Voigt', 'Pseudo-Voigt', 'Asymmetric Gaussian']
    for m in models:
        lower, upper = get_bounds(m, x, y, num_peaks=2)
        n_params = get_num_params(m) * 2
        assert len(lower) == n_params
        assert len(upper) == n_params
        
        # Check amplitudes (index 0 and n_params//2)
        assert lower[0] >= 1e-4, f"Amplitude lower bound must be >= 1e-4 for {m}"
        assert lower[get_num_params(m)] >= 1e-4
        
        # Check widths (index 2)
        assert lower[2] >= 0.1, f"Width lower bound must be >= 0.1 nm for {m}"
        
        # Check shape parameters
        if m == 'Pseudo-Voigt':
            # fraction parameter in [0, 1]
            frac_idx = 3
            assert lower[frac_idx] == 0.0
            assert upper[frac_idx] == 1.0
        elif m == 'Asymmetric Gaussian':
            # skew parameter in [-1, 1]
            skew_idx = 3
            assert lower[skew_idx] == -1.0
            assert upper[skew_idx] == 1.0


def test_fit_spectrum_respects_physical_bounds():
    """Verify that curve fitting never produces negative amplitudes or sub-0.1 nm widths."""
    x = np.linspace(500, 750, 251)
    # Synthetic spectrum with noise
    np.random.seed(42)
    y = 0.5 * np.exp(-((x - 664.0) / 18.0) ** 2) + 0.2 * np.exp(-((x - 612.0) / 22.0) ** 2)
    y += np.random.normal(0, 0.01, len(x))
    
    for m in ['Gaussian', 'Lorentzian', 'Voigt', 'Pseudo-Voigt', 'Asymmetric Gaussian']:
        popt, pcov, y_fit, individual_peaks = fit_spectrum(x, y, m, num_peaks=2, custom_guesses=[612.0, 664.0])
        n_p = get_num_params(m)
        for i in range(2):
            amp = popt[i * n_p]
            cen = popt[i * n_p + 1]
            wid = popt[i * n_p + 2]
            
            assert amp > 0.0, f"Peak {i+1} amplitude for {m} must be positive, got {amp}"
            assert wid >= 0.1, f"Peak {i+1} width for {m} must be >= 0.1, got {wid}"
            assert 450.0 <= cen <= 800.0, f"Peak {i+1} center for {m} must be inside bounded window, got {cen}"


def test_delta_aic_and_bic_parsimony():
    """Verify that delta AIC and BIC correctly penalize additional parameters on identical residuals."""
    y_true = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    y_pred = np.array([1.01, 1.99, 3.02, 1.98, 1.01])
    
    aic_3 = calculate_aic(y_true, y_pred, num_params=3)
    aic_4 = calculate_aic(y_true, y_pred, num_params=4)
    
    # 4 params should have higher AIC by exactly 2 * (4 - 3) = 2.0
    assert np.isclose(aic_4 - aic_3, 2.0)
    
    bic_3 = calculate_bic(y_true, y_pred, num_params=3)
    bic_4 = calculate_bic(y_true, y_pred, num_params=4)
    # BIC penalty is ln(5) = 1.609
    assert np.isclose(bic_4 - bic_3, np.log(5))


def test_kinetic_batch_deconvolution_pipeline():
    """Verify that batch deconvolution recovers pseudo-first-order kinetics without errors."""
    df = generate_demo_kinetic_data()
    wvl = df['Wavelength'].values
    time_points = [0, 10, 20, 30, 40, 50, 60]
    spectra_dict = {t: df[f"{t} min"].values for t in time_points}
    
    from dyefit.dyes import get_dye_config
    mb_config = get_dye_config("Methylene Blue")
    
    res = batch_deconvolve_series(
        wvl,
        spectra_dict,
        dye_config=mb_config,
        baseline_type="ALS",
        model_type="Gaussian",
        als_lam=1e5,
        als_p=0.01
    )
    
    assert "summary_df" in res
    assert "first_order_monomer" in res
    assert "first_order_raw" in res
    sum_df = res["summary_df"]
    assert len(sum_df) == len(time_points)
    
    # Check that monomer and raw first-order rate constants are positive and reasonable
    fo_mono = res["first_order_monomer"]
    fo_raw = res["first_order_raw"]
    assert fo_mono["k"] > 0.0
    assert fo_raw["k"] > 0.0
    assert fo_mono["r_squared"] > 0.95
