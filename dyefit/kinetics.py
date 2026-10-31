"""
===============================================================================
Chemical Kinetics & Time-Series Reaction Modeling Module
===============================================================================
This module processes spectroscopic time-series datasets to extract reaction 
kinetics from deconvolved peak areas (AUC) and nominal peak absorbances.

Key capabilities:
  1. Pseudo-first-order, pseudo-second-order, and zero-order kinetic solvers.
  2. Sequential batch deconvolution across time-series spectra.
  3. Calculation of relative concentration degradation curves (C_t / C_0).
  4. Publication-grade visualization comparing raw A_max vs. deconvolved AUC kinetics.
  5. Synthetic kinetic dataset generation reproducing baseline scattering and dimer shifts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .baseline import linear_baseline, polynomial_baseline, als_baseline
from .fitting import fit_spectrum
from .reporting import calculate_auc, calculate_r_squared, calculate_rmse


# =============================================================================
# 1. KINETIC REGRESSION SOLVERS
# =============================================================================

def fit_pseudo_first_order(times, concentrations, c0=None):
    """
    Fits reaction data to pseudo-first-order kinetics:
        -ln(C_t / C_0) = k_app * t

    Args:
        times (array-like): Reaction time points (e.g., minutes).
        concentrations (array-like): Observed concentration or AUC proxy at each time.
        c0 (float, optional): Initial concentration at t=0. Defaults to concentrations[0].

    Returns:
        dict: Fitted kinetic parameters including:
            - 'k': Apparent rate constant (min^-1 or matching time unit).
            - 'r_squared': Coefficient of determination.
            - 'half_life': Reaction half-life t_1/2 = ln(2) / k.
            - 'intercept': Linear regression intercept (b in y = k*t + b).
            - 'y_actual': -ln(C_t / C_0) array.
            - 'y_pred': Fitted -ln(C_t / C_0) values.
            - 'c_pred': Fitted C_t / C_0 values (exp(-k*t)).
            - 'c_ratio': Actual C_t / C_0 array.
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(concentrations, dtype=float)

    if c0 is None or c0 <= 0:
        c0 = conc[0] if len(conc) > 0 and conc[0] > 0 else 1.0

    # Guard against zero or negative values in logarithm
    safe_conc = np.clip(conc, 1e-12, None)
    c_ratio = safe_conc / c0
    y_actual = -np.log(c_ratio)

    # Standard linear regression: y = k * t + b
    # Also evaluate slope constrained through origin for standard chemical kinetic reporting
    if len(times) >= 2:
        # Unconstrained fit for diagnostic check of induction period / adsorption
        poly = np.polyfit(times, y_actual, 1)
        k_unconstrained = float(poly[0])
        intercept = float(poly[1])

        # Constrained through origin: y = k * t
        sum_t2 = np.sum(times ** 2)
        if sum_t2 > 0:
            k = float(np.sum(times * y_actual) / sum_t2)
        else:
            k = k_unconstrained

        y_pred = k * times
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        r_squared = max(0.0, min(1.0, r_squared))

        half_life = float(np.log(2.0) / k) if k > 0 else np.nan
        c_pred = np.exp(-k * times)
    else:
        k = 0.0
        intercept = 0.0
        r_squared = 0.0
        half_life = np.nan
        y_pred = y_actual.copy()
        c_pred = np.ones_like(times)

    return {
        'model': 'Pseudo-first-order',
        'k': k,
        'k_unconstrained': k_unconstrained if len(times) >= 2 else 0.0,
        'intercept': intercept if len(times) >= 2 else 0.0,
        'r_squared': r_squared,
        'half_life': half_life,
        'times': times,
        'c_ratio': c_ratio,
        'y_actual': y_actual,
        'y_pred': y_pred,
        'c_pred': c_pred
    }


def fit_pseudo_second_order(times, concentrations, c0=None):
    """
    Fits reaction data to pseudo-second-order kinetics:
        1 / C_t - 1 / C_0 = k_2 * t
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(concentrations, dtype=float)

    if c0 is None or c0 <= 0:
        c0 = conc[0] if len(conc) > 0 and conc[0] > 0 else 1.0

    safe_conc = np.clip(conc, 1e-12, None)
    c_ratio = safe_conc / c0
    y_actual = (1.0 / c_ratio) - 1.0

    if len(times) >= 2:
        sum_t2 = np.sum(times ** 2)
        k2 = float(np.sum(times * y_actual) / sum_t2) if sum_t2 > 0 else 0.0
        y_pred = k2 * times
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        r_squared = max(0.0, min(1.0, r_squared))
        half_life = float(1.0 / (k2 * c0)) if k2 > 0 and c0 > 0 else np.nan
    else:
        k2 = 0.0
        r_squared = 0.0
        half_life = np.nan
        y_pred = y_actual.copy()

    return {
        'model': 'Pseudo-second-order',
        'k2': k2,
        'r_squared': r_squared,
        'half_life': half_life,
        'times': times,
        'y_actual': y_actual,
        'y_pred': y_pred
    }


def fit_zero_order(times, concentrations, c0=None):
    """
    Fits reaction data to zero-order kinetics:
        C_0 - C_t = k_0 * t
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(concentrations, dtype=float)

    if c0 is None or c0 <= 0:
        c0 = conc[0] if len(conc) > 0 and conc[0] > 0 else 1.0

    c_ratio = conc / c0
    y_actual = 1.0 - c_ratio

    if len(times) >= 2:
        sum_t2 = np.sum(times ** 2)
        k0 = float(np.sum(times * y_actual) / sum_t2) if sum_t2 > 0 else 0.0
        y_pred = k0 * times
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
        r_squared = max(0.0, min(1.0, r_squared))
        half_life = float(0.5 / k0) if k0 > 0 else np.nan
    else:
        k0 = 0.0
        r_squared = 0.0
        half_life = np.nan
        y_pred = y_actual.copy()

    return {
        'model': 'Zero-order',
        'k0': k0,
        'r_squared': r_squared,
        'half_life': half_life,
        'times': times,
        'y_actual': y_actual,
        'y_pred': y_pred
    }


# =============================================================================
# 2. BATCH DECONVOLUTION ENGINE
# =============================================================================

def batch_deconvolve_series(
    wavelengths,
    spectra_dict,
    dye_config=None,
    baseline_type="Linear",
    model_type="Gaussian",
    num_peaks=2,
    custom_guesses=None,
    wl_min=None,
    wl_max=None,
    poly_deg=2,
    als_lam=1e5,
    als_p=0.01
):
    """
    Deconvolves a series of UV-Vis spectra recorded over reaction time.

    Args:
        wavelengths (array-like): Wavelength vector (nm).
        spectra_dict (dict): Dictionary mapping time (float) to absorbance array (np.array).
        dye_config (DyeConfig, optional): Preset configuration with default bounds and peak guesses.
        baseline_type (str): 'None', 'Linear', 'Polynomial', or 'ALS'.
        model_type (str): 'Gaussian', 'Lorentzian', 'Voigt', 'Pseudo-Voigt', or 'Asymmetric Gaussian'.
        num_peaks (int): Number of sub-peaks to deconvolve (e.g., 2 for Monomer + Dimer).
        custom_guesses (list of float, optional): Custom initial center wavelengths.
        wl_min, wl_max (float, optional): Wavelength cropping bounds.
        poly_deg (int): Polynomial baseline degree.
        als_lam (float): ALS smoothing parameter lambda.
        als_p (float): ALS asymmetry parameter p.

    Returns:
        dict: Complete kinetic results package:
            - 'summary_df': DataFrame of time, raw A_max, individual peak AUCs, and kinetic ratios.
            - 'first_order_raw': Kinetic fit of raw single-wavelength A_max.
            - 'first_order_monomer': Kinetic fit of deconvolved monomer peak AUC.
            - 'first_order_total': Kinetic fit of total deconvolved AUC.
            - 'spectral_fits': Detailed fit parameters for each time point.
    """
    wl = np.asarray(wavelengths, dtype=float)

    # Apply wavelength cropping
    if wl_min is None:
        wl_min = getattr(dye_config, 'min_wavelength', None) if dye_config else None
        if wl_min is None:
            wl_min = float(np.min(wl))
    if wl_max is None:
        wl_max = getattr(dye_config, 'max_wavelength', None) if dye_config else None
        if wl_max is None:
            wl_max = float(np.max(wl))

    mask = (wl >= wl_min) & (wl <= wl_max)
    x_crop = wl[mask]

    sorted_times = sorted(spectra_dict.keys())
    results_rows = []
    spectral_fits = {}

    for t in sorted_times:
        y_raw_full = np.asarray(spectra_dict[t], dtype=float)
        y_raw_crop = y_raw_full[mask]

        # Raw nominal absorbance at peak max
        raw_a_max = float(np.max(y_raw_crop))
        raw_max_wl = float(x_crop[np.argmax(y_raw_crop)])

        # Apply baseline correction
        if baseline_type == "Linear":
            baseline = linear_baseline(x_crop, y_raw_crop)
        elif baseline_type == "Polynomial":
            baseline = polynomial_baseline(x_crop, y_raw_crop, degree=poly_deg)
        elif baseline_type == "ALS":
            baseline = als_baseline(y_raw_crop, lam=als_lam, p=als_p)
        else:
            baseline = np.zeros_like(y_raw_crop)

        y_corrected = np.maximum(y_raw_crop - baseline, 0.0)

        # Deconvolution curve fitting
        guesses = custom_guesses if custom_guesses else (getattr(dye_config, 'peak_guesses', None) if dye_config else None)
        try:
            popt, pcov, y_fitted, individual_peaks = fit_spectrum(
                x_crop, y_corrected, model_type, num_peaks=num_peaks, custom_guesses=guesses
            )
            fit_success = True
        except Exception:
            # Fallback if optimization encounters numerical failure
            fit_success = False
            popt = []
            individual_peaks = [np.zeros_like(x_crop) for _ in range(num_peaks)]
            y_fitted = y_corrected.copy()

        # Calculate AUC for each peak
        peak_aucs = [calculate_auc(x_crop, p) for p in individual_peaks]
        total_auc = calculate_auc(x_crop, y_fitted)

        # In Methylene Blue, Peak 2 (higher wavelength ~664 nm) is monomer, Peak 1 (~612 nm) is dimer
        # Generally, identify the monomer as the peak with the maximum center wavelength or largest AUC
        if len(peak_aucs) >= 2:
            monomer_auc = peak_aucs[-1]  # Monomer is typically the red-shifted band in cyanines/phenothiazines
            dimer_auc = peak_aucs[0]
        else:
            monomer_auc = peak_aucs[0] if len(peak_aucs) > 0 else total_auc
            dimer_auc = 0.0

        r_squared = calculate_r_squared(y_corrected, y_fitted) if fit_success else 0.0
        rmse = calculate_rmse(y_corrected, y_fitted) if fit_success else 0.0

        results_rows.append({
            'Time': float(t),
            'Raw_A_max': raw_a_max,
            'Raw_Max_Wavelength': raw_max_wl,
            'Monomer_AUC': monomer_auc,
            'Dimer_AUC': dimer_auc,
            'Total_AUC': total_auc,
            'Fit_R2': r_squared,
            'Fit_RMSE': rmse
        })

        spectral_fits[t] = {
            'x': x_crop,
            'y_raw': y_raw_crop,
            'baseline': baseline,
            'y_corrected': y_corrected,
            'y_fitted': y_fitted,
            'individual_peaks': individual_peaks,
            'peak_aucs': peak_aucs,
            'popt': popt,
            'fit_success': fit_success
        }

    summary_df = pd.DataFrame(results_rows)

    # Compute kinetic degradation ratios
    t0_row = summary_df.iloc[0]
    summary_df['C_C0_Raw'] = summary_df['Raw_A_max'] / max(t0_row['Raw_A_max'], 1e-12)
    summary_df['C_C0_Monomer'] = summary_df['Monomer_AUC'] / max(t0_row['Monomer_AUC'], 1e-12)
    summary_df['C_C0_Total'] = summary_df['Total_AUC'] / max(t0_row['Total_AUC'], 1e-12)

    summary_df['Minus_ln_C_C0_Raw'] = -np.log(np.clip(summary_df['C_C0_Raw'], 1e-12, None))
    summary_df['Minus_ln_C_C0_Monomer'] = -np.log(np.clip(summary_df['C_C0_Monomer'], 1e-12, None))
    summary_df['Minus_ln_C_C0_Total'] = -np.log(np.clip(summary_df['C_C0_Total'], 1e-12, None))

    # Kinetic regressions
    first_order_raw = fit_pseudo_first_order(summary_df['Time'], summary_df['Raw_A_max'])
    first_order_monomer = fit_pseudo_first_order(summary_df['Time'], summary_df['Monomer_AUC'])
    first_order_total = fit_pseudo_first_order(summary_df['Time'], summary_df['Total_AUC'])

    second_order_monomer = fit_pseudo_second_order(summary_df['Time'], summary_df['Monomer_AUC'])
    zero_order_monomer = fit_zero_order(summary_df['Time'], summary_df['Monomer_AUC'])

    return {
        'summary_df': summary_df,
        'first_order_raw': first_order_raw,
        'first_order_monomer': first_order_monomer,
        'first_order_total': first_order_total,
        'second_order_monomer': second_order_monomer,
        'zero_order_monomer': zero_order_monomer,
        'spectral_fits': spectral_fits,
        'wavelength_crop': x_crop
    }


# =============================================================================
# 3. SYNTHETIC KINETIC DEMO GENERATOR
# =============================================================================

def generate_demo_kinetic_data(dye_name="Methylene Blue"):
    """
    Generates a realistic 7-point kinetic photocatalytic degradation time-series
    reproducing the physical phenomenon:
      - Exponential degradation of intrinsic monomer (k = 0.025 min^-1)
      - Monomer-dimer dynamic equilibrium re-equilibration
      - Nanoparticle catalyst colloidal scattering baseline drift
    """
    times = [0, 10, 20, 30, 40, 50, 60]  # minutes
    wl = np.linspace(500, 750, 251)  # 500 to 750 nm with 1 nm step

    k_true = 0.025
    data_dict = {"Wavelength": wl}

    for t in times:
        decay = np.exp(-k_true * t)

        # Monomer peak (~664 nm) decays cleanly with true first-order kinetics
        a_monomer = 0.85 * decay
        peak_monomer = a_monomer * np.exp(-((wl - 664.0) / 22.0) ** 2)

        # Dimer peak (~612 nm) exhibits non-linear shift as equilibrium shifts
        a_dimer = 0.42 * (decay ** 1.35)
        peak_dimer = a_dimer * np.exp(-((wl - 612.0) / 28.0) ** 2)

        # Nanoparticle scattering baseline (increases slightly or persists as catalyst remains suspended)
        # Rayleigh-like 1/lambda^2 baseline slope + background offset
        base = 0.08 + 0.05 * decay + 0.04 * ((750.0 - wl) / 250.0)

        # Instrument noise
        noise = np.random.normal(0, 0.002, len(wl))

        y = base + peak_monomer + peak_dimer + noise
        data_dict[f"{t} min"] = y

    return pd.DataFrame(data_dict)


# =============================================================================
# 4. PUBLICATION & INTERACTIVE VISUALIZATIONS
# =============================================================================

def generate_multi_spectrum_plotly_fig(wavelengths, spectra_dict, dye_name="Methylene Blue", theme="dark"):
    """
    Creates an interactive Plotly figure displaying the overlaid UV-Vis absorption spectra
    across the complete time series (t_0 -> t_final), with a smooth color gradient.
    """
    fig = go.Figure()
    sorted_times = sorted(spectra_dict.keys())
    n_times = len(sorted_times)

    # Styling based on theme
    is_pub = (theme == "publication_light")
    bg_color = "#FFFFFF" if is_pub else "rgba(0,0,0,0)"
    paper_bg = "#FFFFFF" if is_pub else "rgba(0,0,0,0)"
    text_color = "#000000" if is_pub else "#FFFFFF"
    grid_color = "#E2E8F0" if is_pub else "rgba(255,255,255,0.18)"
    border_color = "#000000" if is_pub else "rgba(255,255,255,0.45)"

    # Generate a smooth blue-to-amber or spectral color progression
    for i, t in enumerate(sorted_times):
        fraction = i / max(n_times - 1, 1)
        # Gradient from deep violet-blue (t=0) to bright cyan-teal, then golden amber (final)
        r = int(41 + fraction * (230 - 41))
        g = int(128 - fraction * (128 - 76))
        b = int(185 - fraction * (185 - 60))
        color_str = f"rgb({r},{g},{b})"

        y = spectra_dict[t]
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=y,
            mode='lines',
            name=f'{t} min',
            line=dict(color=color_str, width=2.0 if (i == 0 or i == n_times - 1) else 1.3)
        ))

    fig.update_layout(
        title=f"<b>Time-Resolved UV-Vis Spectra: {dye_name} (0 → {sorted_times[-1]} min)</b>",
        title_font=dict(size=16, family="Google Sans Flex, Inter, sans-serif", color=text_color),
        xaxis=dict(
            title="Wavelength (nm)",
            gridcolor=grid_color,
            color=text_color,
            linecolor=border_color,
            linewidth=1.5 if is_pub else 1.0,
            mirror=True,
            ticks="inside"
        ),
        yaxis=dict(
            title="Absorbance",
            gridcolor=grid_color,
            color=text_color,
            linecolor=border_color,
            linewidth=1.5 if is_pub else 1.0,
            mirror=True,
            ticks="inside",
            rangemode="tozero"
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        template="plotly_white" if is_pub else "plotly_dark",
        height=450,
        legend=dict(
            title=dict(text="<b>Time</b>", font=dict(color=text_color, size=11)),
            font=dict(color=text_color, size=10),
            bgcolor="rgba(255,255,255,0.9)" if is_pub else "rgba(30,41,59,0.85)",
            bordercolor=border_color,
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig


def generate_kinetic_plotly_fig(kinetic_result, dye_name="Methylene Blue", theme="dark"):
    """
    Creates an interactive 2-panel Plotly figure matching scientific publication standards:
        Panel 1: Degradation curves (C_t / C_0 vs. Time)
        Panel 2: Pseudo-first-order kinetic plot (-ln(C_t / C_0) vs. Time)
    """
    df = kinetic_result['summary_df']
    fo_raw = kinetic_result['first_order_raw']
    fo_monomer = kinetic_result['first_order_monomer']

    is_pub = (theme == "publication_light")
    bg_color = "#FFFFFF" if is_pub else "rgba(0,0,0,0)"
    paper_bg = "#FFFFFF" if is_pub else "rgba(0,0,0,0)"
    text_color = "#000000" if is_pub else "#FFFFFF"
    grid_color = "#E2E8F0" if is_pub else "rgba(255,255,255,0.18)"
    border_color = "#000000" if is_pub else "rgba(255,255,255,0.45)"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>(a) Photodegradation Curves (C<sub>t</sub> / C<sub>0</sub>)</b>",
            "<b>(b) Pseudo-First-Order Kinetics (-ln(C<sub>t</sub> / C<sub>0</sub>))</b>"
        ),
        horizontal_spacing=0.12
    )

    t = df['Time']

    # Colors
    c_raw = "#d9534f" if is_pub else "#FF6B6B"
    c_monomer = "#0275d8" if is_pub else "#00D2FF"

    # --- Panel 1: C_t / C_0 vs Time ---
    fig.add_trace(
        go.Scatter(
            x=t, y=df['C_C0_Raw'],
            mode='markers+lines',
            name='Raw A<sub>max</sub> (Uncorrected)',
            marker=dict(size=8, color=c_raw, symbol='square'),
            line=dict(color=c_raw, width=1.8, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=df['C_C0_Monomer'],
            mode='markers+lines',
            name='DyeFit Monomer AUC',
            marker=dict(size=9, color=c_monomer, symbol='circle'),
            line=dict(color=c_monomer, width=2.5)
        ),
        row=1, col=1
    )

    # --- Panel 2: -ln(C_t / C_0) vs Time ---
    fig.add_trace(
        go.Scatter(
            x=t, y=df['Minus_ln_C_C0_Raw'],
            mode='markers',
            name='Raw A<sub>max</sub> Points',
            marker=dict(size=8, color=c_raw, symbol='square'),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=fo_raw['y_pred'],
            mode='lines',
            name=f'Raw: k = {fo_raw["k"]:.4f} min⁻¹ (R² = {fo_raw["r_squared"]:.3f})',
            line=dict(color=c_raw, width=1.8, dash='dot')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=df['Minus_ln_C_C0_Monomer'],
            mode='markers',
            name='Monomer AUC Points',
            marker=dict(size=9, color=c_monomer, symbol='circle'),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=fo_monomer['y_pred'],
            mode='lines',
            name=f'DyeFit: k = {fo_monomer["k"]:.4f} min⁻¹ (R² = {fo_monomer["r_squared"]:.3f})',
            line=dict(color=c_monomer, width=2.5)
        ),
        row=1, col=2
    )

    # Axis updates
    for col_idx in [1, 2]:
        fig.update_xaxes(
            title_text="Irradiation Time (min)",
            row=1, col=col_idx,
            gridcolor=grid_color,
            color=text_color,
            linecolor=border_color,
            linewidth=1.2,
            mirror=True,
            ticks="inside"
        )

    fig.update_yaxes(
        title_text="Relative Concentration (C<sub>t</sub> / C<sub>0</sub>)",
        row=1, col=1,
        range=[-0.05, 1.15],
        gridcolor=grid_color,
        color=text_color,
        linecolor=border_color,
        linewidth=1.2,
        mirror=True,
        ticks="inside"
    )

    fig.update_yaxes(
        title_text="-ln(C<sub>t</sub> / C<sub>0</sub>)",
        row=1, col=2,
        gridcolor=grid_color,
        color=text_color,
        linecolor=border_color,
        linewidth=1.2,
        mirror=True,
        ticks="inside"
    )

    fig.update_layout(
        title=f"<b>Photocatalytic Reaction Kinetics: {dye_name}</b>",
        title_font=dict(size=17, family="Google Sans Flex, Inter, sans-serif", color=text_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        template="plotly_white" if is_pub else "plotly_dark",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(color=text_color, size=10),
            bgcolor="rgba(255,255,255,0.9)" if is_pub else "rgba(30,41,59,0.85)",
            bordercolor=border_color,
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=100)
    )

    return fig


def generate_kinetic_matplotlib_fig(kinetic_result, dye_name="Methylene Blue"):
    """
    Creates a publication-grade 300 DPI Matplotlib figure (matching Figure 3 in manuscript):
        Panel A: Degradation curves (C_t / C_0 vs. Time)
        Panel B: Pseudo-first-order kinetic plot (-ln(C_t / C_0) vs. Time)
    """
    df = kinetic_result['summary_df']
    fo_raw = kinetic_result['first_order_raw']
    fo_monomer = kinetic_result['first_order_monomer']

    t = df['Time'].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

    # --- Panel A ---
    ax1.set_title('(a) Photocatalytic Degradation', loc='left', fontweight='bold', fontsize=10, pad=7)
    ax1.plot(t, df['C_C0_Raw'], 's--', color='#d9534f', label='Raw $A_{664}$ (Nominal)', markersize=5, linewidth=1.2)
    ax1.plot(t, df['C_C0_Monomer'], 'o-', color='#0275d8', label='DyeFit Monomer AUC', markersize=5, linewidth=1.5)
    ax1.set_xlabel('Irradiation Time (min)', fontsize=9, fontweight='medium')
    ax1.set_ylabel(r'Apparent Concentration ($C_t / C_0$)', fontsize=9, fontweight='medium')
    ax1.set_ylim(-0.05, 1.30)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#e0e0e0', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax1.tick_params(direction='in', top=True, right=True, labelsize=8)

    # --- Panel B ---
    ax2.set_title('(b) Pseudo-First-Order Kinetics', loc='left', fontweight='bold', fontsize=10, pad=7)
    ax2.plot(t, df['Minus_ln_C_C0_Raw'], 's', color='#d9534f', markersize=5, alpha=0.9)
    ax2.plot(t, fo_raw['y_pred'], '--', color='#d9534f',
             label=f'Raw: $k = {fo_raw["k"]:.3f}$ min$^{{-1}}$\n($R^2 = {fo_raw["r_squared"]:.3f}$)', linewidth=1.2)

    ax2.plot(t, df['Minus_ln_C_C0_Monomer'], 'o', color='#0275d8', markersize=5, alpha=0.9)
    ax2.plot(t, fo_monomer['y_pred'], '-', color='#0275d8',
             label=f'DyeFit: $k = {fo_monomer["k"]:.3f}$ min$^{{-1}}$\n($R^2 = {fo_monomer["r_squared"]:.3f}$)', linewidth=1.5)

    ax2.set_xlabel('Irradiation Time (min)', fontsize=9, fontweight='medium')
    ax2.set_ylabel(r'$-\ln(C_t / C_0)$', fontsize=9, fontweight='medium')
    ax2.set_ylim(-0.15, max(3.5, np.max(df['Minus_ln_C_C0_Monomer']) * 1.35))
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#e0e0e0', fontsize=7.5)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax2.tick_params(direction='in', top=True, right=True, labelsize=8)

    plt.tight_layout()
    return fig
