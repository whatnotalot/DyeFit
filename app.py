"""
===============================================================================
DyeFit Interactive Dashboard (app.py) - Version 1.1.1
===============================================================================
This is the main application file that implements the interactive web dashboard
using Streamlit. 

Features:
  1. Automated spectrometer CSV/Excel file loading (and metadata stripping).
  2. Built-in synthetic datasets for training/testing.
  3. Interactive cropping and baseline correction (Polynomial, Linear, ALS, or Offset).
  4. Real-time peak deconvolution and statistical evaluation (R^2, RMSE, AIC/BIC, AUC).
  5. Multi-model comparison leaderboard.
  6. Scientific journal-standard downloads (Excel, 300 DPI PNG, Vector PDF).
  7. Time-series reaction kinetics & batch deconvolution (Pseudo-first/second-order).
  8. Iframe embedding snippet helper for WordPress pages.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import inspect
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def st_plotly_chart(fig, **kwargs):
    """Render Plotly figure with forward/backward compatibility across Streamlit versions."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    return st.plotly_chart(fig, use_container_width=True, **kwargs)

def st_dataframe(data, **kwargs):
    """Render DataFrame with forward/backward compatibility across Streamlit versions."""
    if "width" in inspect.signature(st.dataframe).parameters:
        return st.dataframe(data, width="stretch", **kwargs)
    return st.dataframe(data, use_container_width=True, **kwargs)

# Import core mathematical algorithms from the local dyefit package
from dyefit import (
    get_dye_config,
    DYE_REGISTRY,
    polynomial_baseline,
    linear_baseline,
    als_baseline,
    fit_spectrum,
    combined_model_systematic,
    calculate_auc,
    calculate_r_squared,
    calculate_rmse,
    calculate_aic,
    calculate_bic,
    generate_plotly_fig,
    reload_dye_registry,
    save_dye_to_db,
    constant_offset_single,
    constant_offset_region,
    fit_pseudo_first_order,
    fit_pseudo_second_order,
    fit_zero_order,
    batch_deconvolve_series,
    generate_demo_kinetic_data,
    generate_kinetic_plotly_fig,
    generate_multi_spectrum_plotly_fig,
    generate_kinetic_matplotlib_fig
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "session_dyes" not in st.session_state:
    import copy
    # Deep copy presets from DYE_REGISTRY so each visitor has a private, clean template
    st.session_state.session_dyes = copy.deepcopy(DYE_REGISTRY)
if "show_warning" not in st.session_state:
    st.session_state.show_warning = True

# =============================================================================
# 1. PAGE SETUP & CSS INJECTION
# =============================================================================
st.set_page_config(
    page_title="DyeFit - Dye Deconvolution & Kinetics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styles for cards and typography (premium dark-theme aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:wght@400;500;700&family=Outfit:wght@400;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: rgba(31, 41, 55, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        background: rgba(31, 41, 55, 0.6);
        border-color: rgba(0, 210, 255, 0.6);
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.18);
    }
    .custom-title {
        color: #00D2FF;
        font-family: 'Google Sans Flex', 'Google Sans', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .custom-subtitle {
        color: #9CA3AF;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.15rem;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    
    /* Button micro-animations */
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        filter: brightness(1.1);
    }
    
    /* Tab micro-animations */
    .stTabs [data-baseweb="tab-list"] button {
        transition: color 0.2s ease, background-color 0.2s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #00D2FF;
    }
</style>
""", unsafe_allow_html=True)

# Detect if the app is embedded inside an iframe (e.g. in a WordPress page)
is_embedded = False
try:
    if hasattr(st, "query_params"):
        is_embedded = "embed" in st.query_params
    elif hasattr(st, "experimental_get_query_params"):
        is_embedded = "embed" in st.experimental_get_query_params()
except Exception:
    pass

if not is_embedded:
    st.markdown("<h1 class='custom-title'>DyeFit <span style='font-size:1.1rem; color:#00D2FF; font-weight:400;'>v1.1.0</span></h1>", unsafe_allow_html=True)
    st.markdown("<p class='custom-subtitle'>Spectroscopic Fitting & Kinetic Quantification Agent</p>", unsafe_allow_html=True)
    
    with st.expander("📖 How to Use & Scientific Guide", expanded=False):
        st.markdown(r"""
        **Automated baseline correction, multi-model curve fitting, peak deconvolution, and reaction kinetics for UV-Vis spectra of scientific dyes.**
        
        ### Single Spectrum Mode:
        1. **Load Spectrum**: Upload your raw CSV or Excel file containing wavelength and absorbance data from your spectrometer, or use a built-in demo dataset.
        2. **Dye & Baseline Config**: Select the appropriate dye template to automatically set wavelength bounds and peak guesses. Adjust baseline removal to flatten background scattering.
        3. **Curve Fitting**: Choose a mathematical model (e.g., Gaussian, Voigt) and adjust the number of expected peaks. The algorithm deconvolutes overlapping absorption bands.
        4. **Model Comparison & Export**: Rank models via AIC/BIC and download 300 DPI PNG, Vector PDF, or Multi-tab Excel reports.
        
        ### Kinetic Time-Series Mode:
        1. **Load Time-Series**: Upload a multi-column spreadsheet (e.g., Wavelength, 0 min, 10 min...), multiple CSV files, or use the built-in 7-point photocatalytic degradation dataset.
        2. **Batch Deconvolution**: Sequentially deconvolve all time points, separating the true monomer peak from dimer aggregates and scattering baselines.
        3. **Kinetic Fitting**: Extract apparent rate constants ($k$), half-lives ($t_{1/2}$), and evaluate pseudo-first-order kinetics ($-\ln(C_t/C_0) = k t$).
        """)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def auto_detect_columns(df):
    """
    Intelligently identifies Wavelength and Absorbance columns.
    """
    wavelength_col = None
    absorbance_col = None
    
    wl_aliases = ['wavelength', 'nm', 'wvl', 'lambda', 'wave', 'nanometer', 'x']
    abs_aliases = ['absorbance', 'abs', 'od', 'optical density', 'intensity', 'a', 'y', 'raw data']
    
    # 1. Alias Matching
    for col in df.columns:
        c_clean = str(col).strip().lower()
        if not wavelength_col and any(alias == c_clean or alias in c_clean for alias in wl_aliases):
            wavelength_col = col
        elif not absorbance_col and any(alias == c_clean or alias in c_clean for alias in abs_aliases):
            absorbance_col = col
            
    # 2. Numeric Content Scanning Heuristic
    if not wavelength_col or not absorbance_col:
        numeric_cols = []
        for col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) > 10:
                numeric_cols.append((col, series))
                
        if len(numeric_cols) >= 2:
            for col, series in numeric_cols:
                median_val = np.median(series)
                if 190.0 <= median_val <= 5000.0:
                    wavelength_col = col
                    break
            
            for col, series in numeric_cols:
                if col != wavelength_col:
                    absorbance_col = col
                    break

    if wavelength_col is None and len(df.columns) > 0:
        wavelength_col = df.columns[0]
    if absorbance_col is None and len(df.columns) > 1:
        absorbance_col = df.columns[1]
    elif absorbance_col is None and len(df.columns) > 0:
        absorbance_col = df.columns[0]
        
    return wavelength_col, absorbance_col


def generate_demo_data(dye_name):
    """
    Generates synthetic absorption profiles with realistic polynomial baselines.
    """
    x = np.arange(400, 800, 1.0)
    base = 0.03 + 0.0001 * (x - 400) + 0.000002 * (x - 500)**2
    
    if dye_name == "Methylene Blue":
        peak1 = 0.16 * np.exp(-((x - 664.0) / 18.0) ** 2)
        peak2 = 0.08 * np.exp(-((x - 612.0) / 25.0) ** 2)
        y = base + peak1 + peak2 + np.random.normal(0, 0.001, len(x))
    elif dye_name == "Methyl Orange":
        peak1 = 0.22 * np.exp(-((x - 464.0) / 45.0) ** 2)
        peak2 = 0.06 * np.exp(-((x - 380.0) / 30.0) ** 2)
        y = base + peak1 + peak2 + np.random.normal(0, 0.001, len(x))
    else:
        peak1 = 0.18 * np.exp(-((x - 550.0) / 35.0) ** 2)
        y = base + peak1 + np.random.normal(0, 0.001, len(x))
        
    return pd.DataFrame({"Wavelength": x, "Absorbance": y})


# =============================================================================
# 3. SIDEBAR - ANALYSIS MODE SELECTION
# =============================================================================
st.sidebar.markdown("<h2 style='color:#00D2FF; margin-bottom:0px;'>🔬 DyeFit Controls</h2>", unsafe_allow_html=True)
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Spectrum Deconvolution", "⏱️ Kinetic Time-Series Batch"],
    index=0
)


# =============================================================================
# MODE 1: SINGLE SPECTRUM DECONVOLUTION (ORIGINAL WORKFLOW)
# =============================================================================
def render_single_spectrum_mode():
    st.sidebar.markdown("---")
    st.sidebar.header("📁 1. Load Spectrum")
    data_source = st.sidebar.radio("Data Source", ["Upload Excel/CSV", "Use Demo Data"])

    uploaded_df = None
    if data_source == "Upload Excel/CSV":
        uploaded_file = st.sidebar.file_uploader("Upload UV-Vis Spectrum File", type=["xlsx", "xls", "csv"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    uploaded_df = pd.read_csv(uploaded_file)
                else:
                    uploaded_df = pd.read_excel(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    else:
        demo_dye = st.sidebar.selectbox("Demo Dye Template", ["Methylene Blue", "Methyl Orange", "Generic Dye"])
        uploaded_df = generate_demo_data(demo_dye)

    if uploaded_df is not None:
        uploaded_df.columns = [str(col).strip() for col in uploaded_df.columns]
        col_names = list(uploaded_df.columns)
        
        detected_wl, detected_abs = auto_detect_columns(uploaded_df)
        wl_index = col_names.index(detected_wl) if detected_wl in col_names else 0
        abs_index = col_names.index(detected_abs) if detected_abs in col_names else (1 if len(col_names) > 1 else 0)
        
        wavelength_col = st.sidebar.selectbox("Wavelength Column", col_names, index=wl_index, key=f"wl_{data_source}")
        absorbance_col = st.sidebar.selectbox("Absorbance Column", col_names, index=abs_index, key=f"abs_{data_source}")
        
        if wavelength_col in uploaded_df.columns and absorbance_col in uploaded_df.columns:
            df_clean = uploaded_df[[wavelength_col, absorbance_col]].copy()
            df_clean[wavelength_col] = pd.to_numeric(df_clean[wavelength_col], errors='coerce')
            df_clean[absorbance_col] = pd.to_numeric(df_clean[absorbance_col], errors='coerce')
            df_clean = df_clean.dropna().sort_values(by=wavelength_col)
            
            if len(df_clean) == 0:
                st.error("No numeric data found in the selected columns. Please check your data sheet.")
                st.stop()
                
            wavelength = df_clean[wavelength_col].to_numpy()
            absorbance = df_clean[absorbance_col].to_numpy()
        else:
            st.info("Syncing columns...")
            st.stop()
        
        st.sidebar.markdown("---")
        st.sidebar.header("🧪 2. Dye & Baseline Configuration")
        
        dye_options = list(st.session_state.session_dyes.keys()) + ["Custom", "➕ Create Custom Preset"]
        selected_dye = st.sidebar.selectbox("Select Dye Config Preset", dye_options)
        
        if selected_dye == "➕ Create Custom Preset":
            st.sidebar.markdown("### 🛠️ Custom Preset Builder")
            custom_name = st.sidebar.text_input("Dye Name", "My Custom Dye")
            custom_min_w = st.sidebar.number_input("Min Wavelength (nm)", 190.0, 1100.0, float(np.min(wavelength)))
            custom_max_w = st.sidebar.number_input("Max Wavelength (nm)", 190.0, 1100.0, float(np.max(wavelength)))
            custom_base_left = st.sidebar.number_input("Baseline Left Bound (nm)", 190.0, 1100.0, float(custom_min_w + (custom_max_w - custom_min_w)*0.15))
            custom_base_right = st.sidebar.number_input("Baseline Right Bound (nm)", 190.0, 1100.0, float(custom_max_w - (custom_max_w - custom_min_w)*0.15))
            custom_peaks_count = st.sidebar.number_input("Default Fit Peaks Count", 1, 5, 2)
            
            default_guesses = f"{custom_min_w + (custom_max_w - custom_min_w)*0.3:.1f}, {custom_min_w + (custom_max_w - custom_min_w)*0.7:.1f}"
            custom_centers = st.sidebar.text_input("Expected Peak Centers (comma-separated)", default_guesses)
            
            if st.sidebar.button("💾 Save Preset to Session"):
                try:
                    guesses_list = [float(x.strip()) for x in custom_centers.split(",") if x.strip() != ""] if custom_centers else None
                    config_new = get_dye_config(
                        custom_name,
                        min_wavelength=custom_min_w,
                        max_wavelength=custom_max_w,
                        num_peaks=custom_peaks_count,
                        baseline_left=custom_base_left,
                        baseline_right=custom_base_right,
                        peak_guesses=guesses_list
                    )
                    st.session_state.session_dyes[custom_name] = config_new
                    st.sidebar.success(f"Preset '{custom_name}' added to your session!")
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to save preset: {e}")
                    
            guesses_list = [float(x.strip()) for x in custom_centers.split(",")] if custom_centers else None
            config = get_dye_config(
                custom_name,
                min_wavelength=custom_min_w,
                max_wavelength=custom_max_w,
                num_peaks=custom_peaks_count,
                baseline_left=custom_base_left,
                baseline_right=custom_base_right,
                peak_guesses=guesses_list
            )
            min_w_val = custom_min_w
            max_w_val = custom_max_w
            num_peaks_default = custom_peaks_count
        
        elif selected_dye != "Custom":
            config = st.session_state.session_dyes[selected_dye]
            min_w_val = float(config.min_wavelength)
            max_w_val = float(config.max_wavelength)
            num_peaks_default = config.num_peaks
        else:
            min_w_val = float(np.min(wavelength))
            max_w_val = float(np.max(wavelength))
            num_peaks_default = 2
            
        if selected_dye == "➕ Create Custom Preset":
            min_w = st.sidebar.slider("Min Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), float(custom_min_w))
            max_w = st.sidebar.slider("Max Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), float(custom_max_w))
        else:
            default_min = max(float(np.min(wavelength)), min_w_val)
            default_max = min(float(np.max(wavelength)), max_w_val)
            if default_min >= default_max:
                default_min = float(np.min(wavelength))
                default_max = float(np.max(wavelength))
            min_w, max_w = st.sidebar.slider("Analysis Range (nm)", float(np.min(wavelength)), float(np.max(wavelength)), (default_min, default_max))
        
        mask = (wavelength >= min_w) & (wavelength <= max_w)
        x_cropped = wavelength[mask]
        y_cropped = absorbance[mask]
        
        if len(x_cropped) < 5:
            st.warning("Selected wavelength range contains too few data points. Please expand your bounds.")
            st.stop()
            
        baseline_method = st.sidebar.selectbox("Baseline Method", ["Polynomial", "Linear", "ALS", "Constant Offset", "None"])
        
        if baseline_method == "Polynomial":
            poly_deg = st.sidebar.slider("Polynomial Degree", 1, 5, 2)
            st.sidebar.markdown("**Baseline Fitting Regions (nm)**")
            if selected_dye not in ["Custom", "➕ Create Custom Preset"] and hasattr(config, 'baseline_mask_fn'):
                st.sidebar.info(f"Default preset baseline ranges are active for {selected_dye}.")
                baseline_mask = config.baseline_mask_fn(x_cropped)
            elif selected_dye == "➕ Create Custom Preset" and hasattr(config, 'baseline_mask_fn'):
                st.sidebar.info(f"Custom preset baseline values: <{custom_base_left} nm or >{custom_base_right} nm.")
                baseline_mask = config.baseline_mask_fn(x_cropped)
            else:
                left_bound = st.sidebar.slider("Left Baseline Threshold", float(min_w), float(max_w), float(min_w + (max_w - min_w)*0.15))
                right_bound = st.sidebar.slider("Right Baseline Threshold", float(min_w), float(max_w), float(max_w - (max_w - min_w)*0.15))
                baseline_mask = (x_cropped < left_bound) | (x_cropped > right_bound)
            
            if not np.any(baseline_mask):
                baseline_mask = np.ones_like(x_cropped, dtype=bool)

            baseline = polynomial_baseline(x_cropped, y_cropped, mask=baseline_mask, degree=poly_deg)
        elif baseline_method == "Linear":
            baseline = linear_baseline(x_cropped, y_cropped)
        elif baseline_method == "ALS":
            als_lam = st.sidebar.select_slider("Smoothness (λ)", options=[1e2, 1e3, 1e4, 1e5, 1e6], value=1e5)
            als_p = st.sidebar.slider("Asymmetry (p)", 0.001, 0.1, 0.01, step=0.005)
            baseline = als_baseline(y_cropped, lam=als_lam, p=als_p)
        elif baseline_method == "Constant Offset":
            offset_mode = st.sidebar.radio("Offset Mode", ["Single Wavelength Point", "Averaged Region"])
            if offset_mode == "Single Wavelength Point":
                target_w = st.sidebar.number_input("Target Wavelength (nm)", float(np.min(x_cropped)), float(np.max(x_cropped)), float(np.min(x_cropped)))
                baseline = constant_offset_single(x_cropped, y_cropped, target_w)
            else:
                region_min = st.sidebar.number_input("Region Min (nm)", float(np.min(x_cropped)), float(np.max(x_cropped)), float(np.min(x_cropped)))
                region_max = st.sidebar.number_input("Region Max (nm)", float(np.min(x_cropped)), float(np.max(x_cropped)), float(np.min(x_cropped)) + 20.0)
                baseline = constant_offset_region(x_cropped, y_cropped, (region_min, region_max))
        else:
            baseline = np.zeros_like(y_cropped)
            
        abs_corrected = np.maximum(y_cropped - baseline, 0.0)
        
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 3. Peak Fitting Configuration")
        num_peaks = st.sidebar.number_input("Number of Peaks", min_value=1, max_value=6, value=int(num_peaks_default), step=1)
        
        custom_guesses = None
        use_custom_guesses = st.sidebar.checkbox("Manually Specify Initial Center Guesses", value=False)
        if use_custom_guesses:
            default_val = ", ".join([str(x) for x in config.peak_guesses]) if (selected_dye != "Custom" and hasattr(config, 'peak_guesses') and config.peak_guesses) else ""
            custom_input = st.sidebar.text_input("Initial Peak Centers (comma-separated)", value=default_val)
            if custom_input:
                try:
                    custom_guesses = [float(x.strip()) for x in custom_input.split(",") if x.strip() != ""]
                except ValueError:
                    st.sidebar.warning("Please enter valid numbers (separated by commas).")
                    
        model_type = st.sidebar.selectbox("Curve Fitting Model", ["Gaussian", "Lorentzian", "Voigt", "Pseudo-Voigt", "Asymmetric Gaussian"])
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 4. Chart Style Settings")
        theme_choice = st.sidebar.radio("Plotly Chart Style", ["Web Dark", "Publication Light"])
        plotly_theme = "publication_light" if theme_choice == "Publication Light" else "dark"
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Baseline Correction", "🔍 Peak Fitting & Deconvolution", "🧪 Model Comparison", "💾 Export Results"])
        
        with tab1:
            st.markdown("### Raw vs. Baseline-Corrected Spectrum")
            st.markdown("Adjust parameters in the sidebar to flat-correct the baseline before fitting peaks.")
            
            if plotly_theme == "publication_light":
                raw_color = "#4B5563"
                base_color = "#D55E00"
                corr_color = "#000000"
                bg_color = "#FFFFFF"
                paper_bg = "#FFFFFF"
                grid_color = "#E2E8F0"
                text_color = "#000000"
                border_color = "#000000"
                axis_line_width = 1.5
                legend_bg = "rgba(255,255,255,0.95)"
                legend_border = border_color
            else:
                raw_color = "#E5E7EB"
                base_color = "#FFC700"
                corr_color = "#00D2FF"
                bg_color = "rgba(0,0,0,0)"
                paper_bg = "rgba(0,0,0,0)"
                grid_color = "rgba(255,255,255,0.18)"
                text_color = "#FFFFFF"
                border_color = "rgba(255,255,255,0.45)"
                axis_line_width = 1.2
                legend_bg = "rgba(30, 41, 59, 0.85)"
                legend_border = "rgba(255,255,255,0.3)"
                
            fig_base = go.Figure()
            fig_base.add_trace(go.Scatter(x=x_cropped, y=y_cropped, name="Raw Absorbance", line=dict(color=raw_color, width=1.5)))
            fig_base.add_trace(go.Scatter(x=x_cropped, y=baseline, name="Fitted Baseline", line=dict(color=base_color, width=1.8, dash='dash')))
            fig_base.add_trace(go.Scatter(x=x_cropped, y=abs_corrected, name="Corrected Absorbance", line=dict(color=corr_color, width=2)))
            
            fig_base.update_layout(
                xaxis=dict(title="Wavelength (nm)", gridcolor=grid_color, color=text_color, linecolor=border_color, linewidth=axis_line_width, mirror=True, ticks="inside"),
                yaxis=dict(title="Absorbance", gridcolor=grid_color, color=text_color, linecolor=border_color, linewidth=axis_line_width, mirror=True, ticks="inside"),
                plot_bgcolor=bg_color,
                paper_bgcolor=paper_bg,
                template="plotly_white" if plotly_theme == "publication_light" else "plotly_dark",
                height=450,
                legend=dict(bgcolor=legend_bg, bordercolor=legend_border, borderwidth=1, font=dict(color=text_color)),
                margin=dict(l=50, r=30, t=30, b=50)
            )
            st_plotly_chart(fig_base)
            
        with tab2:
            st.markdown(f"### Deconvoluted Spectral Profile ({model_type})")
            guesses = custom_guesses if custom_guesses else (getattr(config, 'peak_guesses', None) if selected_dye != "Custom" else None)
            
            try:
                popt, pcov, y_fitted, individual_peaks = fit_spectrum(
                    x_cropped, abs_corrected, model_type, num_peaks=num_peaks, custom_guesses=guesses
                )
                
                r2 = calculate_r_squared(abs_corrected, y_fitted)
                rmse = calculate_rmse(abs_corrected, y_fitted)
                from dyefit.fitting import get_num_params
                num_p = get_num_params(model_type) * num_peaks
                aic_val = calculate_aic(abs_corrected, y_fitted, num_p)
                bic_val = calculate_bic(abs_corrected, y_fitted, num_p)
                total_auc = calculate_auc(x_cropped, y_fitted)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(f"<div class='metric-card'><h4>R² Value</h4><h2>{r2:.4f}</h2></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><h4>RMSE</h4><h2>{rmse:.4f}</h2></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><h4>AIC</h4><h2>{aic_val:.1f}</h2></div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='metric-card'><h4>BIC</h4><h2>{bic_val:.1f}</h2></div>", unsafe_allow_html=True)
                c5.markdown(f"<div class='metric-card'><h4>Total AUC</h4><h2>{total_auc:.2f}</h2></div>", unsafe_allow_html=True)
                
                fig_fit = generate_plotly_fig(x_cropped, abs_corrected, y_fitted, individual_peaks, model_type, theme=plotly_theme)
                st_plotly_chart(fig_fit)
                
                # Deconvoluted peaks breakdown table
                st.markdown("#### Resolved Sub-Peak Characteristics")
                peak_records = []
                for i in range(num_peaks):
                    p_peak = popt[i * get_num_params(model_type) : (i + 1) * get_num_params(model_type)]
                    peak_records.append({
                        "Peak": f"Peak {i+1}",
                        "Center (nm)": f"{p_peak[1]:.2f}",
                        "Height (Amp)": f"{p_peak[0]:.4f}",
                        "Width Parameter": f"{p_peak[2]:.2f}",
                        "Peak Area (AUC)": f"{calculate_auc(x_cropped, individual_peaks[i]):.3f}",
                        "% of Total AUC": f"{(calculate_auc(x_cropped, individual_peaks[i]) / max(total_auc, 1e-12) * 100):.1f}%"
                    })
                st_dataframe(pd.DataFrame(peak_records))
                
            except Exception as e:
                st.error(f"Curve fitting failed to converge: {e}")
                
        with tab3:
            st.markdown("### Objective Model Comparison Leaderboard")
            st.markdown("Compare all peak profiles on this spectrum to select the mathematically superior model.")
            
            models_to_test = ["Gaussian", "Lorentzian", "Voigt", "Pseudo-Voigt", "Asymmetric Gaussian"]
            leaderboard = []
            
            with st.spinner("Evaluating all mathematical models..."):
                for m in models_to_test:
                    try:
                        popt_m, _, y_fit_m, _ = fit_spectrum(x_cropped, abs_corrected, m, num_peaks=num_peaks, custom_guesses=guesses)
                        num_params_total = get_num_params(m) * num_peaks
                        r2_m = calculate_r_squared(abs_corrected, y_fit_m)
                        rmse_m = calculate_rmse(abs_corrected, y_fit_m)
                        aic_m = calculate_aic(abs_corrected, y_fit_m, num_params_total)
                        bic_m = calculate_bic(abs_corrected, y_fit_m, num_params_total)
                        leaderboard.append({
                            "Model Profile": m,
                            "R² (Variance Explained)": r2_m,
                            "RMSE": rmse_m,
                            "AIC": aic_m,
                            "BIC": bic_m
                        })
                    except Exception:
                        pass
                        
            if leaderboard:
                ld_df = pd.DataFrame(leaderboard).sort_values(by="AIC")
                st_dataframe(
                    ld_df.style.format({
                        "R² (Variance Explained)": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "AIC": "{:.1f}",
                        "BIC": "{:.1f}"
                    }).highlight_min(subset=["AIC", "BIC", "RMSE"], color="#005f73")
                      .highlight_max(subset=["R² (Variance Explained)"], color="#005f73")
                )
                
        with tab4:
            st.markdown("### Publication & Archival Export")
            if 'popt' in locals():
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_fit_df = pd.DataFrame({
                        "Wavelength (nm)": x_cropped,
                        "Raw Absorbance": y_cropped,
                        "Baseline Fitted": baseline,
                        "Corrected Absorbance": abs_corrected,
                        "Overall Model Fit": y_fitted
                    })
                    for i, peak_y in enumerate(individual_peaks):
                        export_fit_df[f"Deconvoluted Peak {i+1}"] = peak_y
                    export_fit_df.to_excel(writer, sheet_name="Fitted Spectrum", index=False)
                    
                    param_export = []
                    for i in range(num_peaks):
                        p_peak = popt[i * get_num_params(model_type) : (i + 1) * get_num_params(model_type)]
                        peak_y = individual_peaks[i]
                        param_export.append({
                            "Peak Number": f"Peak {i+1}",
                            "Center Wavelength (nm)": p_peak[1],
                            "Amplitude (Height)": p_peak[0],
                            "Width Parameter": p_peak[2],
                            "Area Under Curve (AUC)": calculate_auc(x_cropped, peak_y)
                        })
                    pd.DataFrame(param_export).to_excel(writer, sheet_name="Peak Parameters", index=False)
                    
                st.download_button(
                    label="📥 Download Fit Report (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"{selected_dye.lower().replace(' ', '_')}_fit_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                from dyefit import generate_matplotlib_fig
                fig_matplotlib = generate_matplotlib_fig(
                    x_cropped, abs_corrected, y_fitted, individual_peaks, model_type
                )
                png_buffer = io.BytesIO()
                fig_matplotlib.savefig(png_buffer, format='png', dpi=300)
                st.download_button(
                    label="🖼️ Download Journal Figure (PNG - 300 DPI)",
                    data=png_buffer.getvalue(),
                    file_name=f"{selected_dye.lower().replace(' ', '_')}_figure.png",
                    mime="image/png"
                )
                
                pdf_buffer = io.BytesIO()
                fig_matplotlib.savefig(pdf_buffer, format='pdf', dpi=300)
                st.download_button(
                    label="📄 Download Journal Figure (PDF - Vector)",
                    data=pdf_buffer.getvalue(),
                    file_name=f"{selected_dye.lower().replace(' ', '_')}_figure.pdf",
                    mime="application/pdf"
                )
                plt.close(fig_matplotlib)
    else:
        st.info("Please upload an Excel/CSV file or select 'Use Demo Data' in the sidebar to start spectroscopic curve deconvolution.")


# =============================================================================
# MODE 2: KINETIC TIME-SERIES BATCH ANALYSIS (NEW V1.1.0 FEATURE)
# =============================================================================
def render_kinetic_mode():
    st.sidebar.markdown("---")
    st.sidebar.header("📁 1. Load Kinetic Time-Series")
    kinetic_source = st.sidebar.radio(
        "Time-Series Source",
        ["Use Demo Kinetic Dataset", "Upload Multi-Column Excel/CSV", "Upload Multiple Spectrum Files"]
    )

    wavelengths = None
    spectra_dict = None
    dataset_name = "Kinetic Series"

    # --- Option A: Built-in 7-Point Synthetic Photocatalysis Dataset ---
    if kinetic_source == "Use Demo Kinetic Dataset":
        demo_k_df = generate_demo_kinetic_data("Methylene Blue")
        wavelengths = demo_k_df["Wavelength"].to_numpy()
        time_cols = [c for c in demo_k_df.columns if c != "Wavelength"]
        times = []
        for c in time_cols:
            m = re.search(r"(\d+(?:\.\d+)?)", c)
            times.append(float(m.group(1)) if m else float(len(times)))
        spectra_dict = {t: demo_k_df[c].to_numpy() for t, c in zip(times, time_cols)}
        dataset_name = "Methylene Blue (Photocatalysis 0–60 min)"
        st.sidebar.success(f"Loaded demo dataset: {len(times)} time points ({min(times):.0f} to {max(times):.0f} min).")

    # --- Option B: Multi-Column Spreadsheet (Wavelength, t=0, t=10, t=20...) ---
    elif kinetic_source == "Upload Multi-Column Excel/CSV":
        uploaded_k_file = st.sidebar.file_uploader("Upload Time-Series Spreadsheet", type=["xlsx", "xls", "csv"])
        if uploaded_k_file is not None:
            try:
                if uploaded_k_file.name.endswith(".csv"):
                    k_df = pd.read_csv(uploaded_k_file)
                else:
                    k_df = pd.read_excel(uploaded_k_file)
                k_df.columns = [str(c).strip() for c in k_df.columns]
                
                detected_wl, _ = auto_detect_columns(k_df)
                wl_index = list(k_df.columns).index(detected_wl) if detected_wl in k_df.columns else 0
                wl_col = st.sidebar.selectbox("Wavelength Column", k_df.columns, index=wl_index)
                
                pot_cols = [c for c in k_df.columns if c != wl_col]
                sel_time_cols = st.sidebar.multiselect("Time-Series Columns", pot_cols, default=pot_cols)
                
                if len(sel_time_cols) < 2:
                    st.warning("Please select at least 2 time-series absorbance columns for kinetic analysis.")
                    st.stop()
                    
                clean_k_df = k_df[[wl_col] + sel_time_cols].apply(pd.to_numeric, errors='coerce').dropna().sort_values(by=wl_col)
                wavelengths = clean_k_df[wl_col].to_numpy()
                
                times = []
                for c in sel_time_cols:
                    m = re.search(r"(\d+(?:\.\d+)?)", c)
                    times.append(float(m.group(1)) if m else None)
                    
                if any(t is None for t in times) or len(set(times)) != len(times):
                    st.sidebar.info("Specify numerical reaction times for each column:")
                    default_time_str = ", ".join([f"{i*10:.0f}" for i in range(len(sel_time_cols))])
                    t_str = st.sidebar.text_input("Reaction Times (min, comma-separated)", default_time_str)
                    try:
                        times = [float(x.strip()) for x in t_str.split(",")]
                    except Exception:
                        times = [float(i) for i in range(len(sel_time_cols))]
                        
                spectra_dict = {t: clean_k_df[c].to_numpy() for t, c in zip(times, sel_time_cols)}
                dataset_name = uploaded_k_file.name
                st.sidebar.success(f"Parsed {len(times)} time points.")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")
                st.stop()

    # --- Option C: Multiple Separate Single-Spectrum Files ---
    else:
        multi_files = st.sidebar.file_uploader("Upload Spectrum Files (One per time point)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
        if multi_files and len(multi_files) >= 2:
            try:
                guessed_times = []
                for f in multi_files:
                    m = re.search(r"(\d+(?:\.\d+)?)", f.name)
                    guessed_times.append(float(m.group(1)) if m else 0.0)
                    
                with st.sidebar.expander("⏱️ Review & Confirm Time Points (min)", expanded=False):
                    confirmed_times = []
                    for i, (f, t_guess) in enumerate(zip(multi_files, guessed_times)):
                        t_val = st.number_input(f"{f.name}", value=float(t_guess), step=5.0, key=f"file_t_{i}")
                        confirmed_times.append(t_val)
                        
                spectra_dict = {}
                common_wl = None
                for f, t_val in zip(multi_files, confirmed_times):
                    if f.name.endswith(".csv"):
                        f_df = pd.read_csv(f)
                    else:
                        f_df = pd.read_excel(f)
                    f_df.columns = [str(c).strip() for c in f_df.columns]
                    wl_col, abs_col = auto_detect_columns(f_df)
                    f_clean = f_df[[wl_col, abs_col]].apply(pd.to_numeric, errors='coerce').dropna().sort_values(by=wl_col)
                    
                    if common_wl is None:
                        common_wl = f_clean[wl_col].to_numpy()
                        spectra_dict[t_val] = f_clean[abs_col].to_numpy()
                    else:
                        spectra_dict[t_val] = np.interp(common_wl, f_clean[wl_col].to_numpy(), f_clean[abs_col].to_numpy())
                        
                wavelengths = common_wl
                dataset_name = f"{len(multi_files)} Uploaded Files"
                st.sidebar.success(f"Loaded {len(multi_files)} spectrum files.")
            except Exception as e:
                st.sidebar.error(f"Error loading files: {e}")
                st.stop()
        elif multi_files and len(multi_files) < 2:
            st.sidebar.warning("Please upload at least 2 files to analyze reaction kinetics.")
            st.stop()

    # If data is ready, configure kinetics and render main view
    if wavelengths is not None and spectra_dict is not None and len(spectra_dict) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.header("🧪 2. Dye Preset & Baseline")
        
        dye_options = list(st.session_state.session_dyes.keys())
        selected_dye = st.sidebar.selectbox("Select Dye Config Preset", dye_options, key="k_dye_preset")
        config = st.session_state.session_dyes[selected_dye]
        
        default_min = max(float(np.min(wavelengths)), float(config.min_wavelength))
        default_max = min(float(np.max(wavelengths)), float(config.max_wavelength))
        if default_min >= default_max:
            default_min, default_max = float(np.min(wavelengths)), float(np.max(wavelengths))
            
        min_w, max_w = st.sidebar.slider(
            "Kinetic Analysis Wavelength Window (nm)",
            float(np.min(wavelengths)), float(np.max(wavelengths)),
            (default_min, default_max)
        )
        
        baseline_method = st.sidebar.selectbox("Baseline Correction Method", ["Linear", "Polynomial", "ALS", "None"], index=0, key="k_baseline_method")
        poly_deg = 2
        als_lam = 1e5
        als_p = 0.01
        if baseline_method == "Polynomial":
            poly_deg = st.sidebar.slider("Polynomial Degree", 1, 5, 2, key="k_poly_deg")
        elif baseline_method == "ALS":
            als_lam = st.sidebar.select_slider("ALS Smoothness (λ)", options=[1e2, 1e3, 1e4, 1e5, 1e6], value=1e5, key="k_als_lam")
            als_p = st.sidebar.slider("ALS Asymmetry (p)", 0.001, 0.1, 0.01, step=0.005, key="k_als_p")
            
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 3. Peak Deconvolution Settings")
        model_type = st.sidebar.selectbox(
            "Deconvolution Peak Profile",
            ["Gaussian", "Lorentzian", "Voigt", "Pseudo-Voigt", "Asymmetric Gaussian"],
            index=0, key="k_model_type"
        )
        num_peaks = st.sidebar.number_input("Number of Sub-Peaks", min_value=1, max_value=6, value=int(config.num_peaks), step=1, key="k_num_peaks")
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 4. Chart Style")
        theme_choice = st.sidebar.radio("Plot Theme", ["Web Dark", "Publication Light"], key="k_theme")
        plotly_theme = "publication_light" if theme_choice == "Publication Light" else "dark"
        
        # --- Batch Solver Execution ---
        with st.spinner("Deconvolving time-series spectra & solving chemical kinetics..."):
            k_result = batch_deconvolve_series(
                wavelengths=wavelengths,
                spectra_dict=spectra_dict,
                dye_config=config,
                baseline_type=baseline_method,
                model_type=model_type,
                num_peaks=num_peaks,
                wl_min=min_w,
                wl_max=max_w,
                poly_deg=poly_deg,
                als_lam=als_lam,
                als_p=als_p
            )
            
        df_sum = k_result['summary_df']
        fo_monomer = k_result['first_order_monomer']
        fo_raw = k_result['first_order_raw']
        
        k_mono = fo_monomer['k']
        r2_mono = fo_monomer['r_squared']
        t_half_mono = fo_monomer['half_life']
        
        k_raw = fo_raw['k']
        r2_raw = fo_raw['r_squared']
        t_half_raw = fo_raw['half_life']
        
        rate_diff_pct = ((k_mono - k_raw) / max(k_raw, 1e-12)) * 100.0
        
        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""
        <div class='metric-card'>
            <h4>DyeFit Intrinsic Rate (k)</h4>
            <h2>{k_mono:.4f} <span style='font-size:1rem; color:#9CA3AF;'>min⁻¹</span></h2>
            <p style='color:#00D2FF; margin:0;'>t<sub>1/2</sub> = {t_half_mono:.1f} min (R² = {r2_mono:.4f})</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
        <div class='metric-card'>
            <h4>Raw Nominal Rate (A<sub>max</sub>)</h4>
            <h2>{k_raw:.4f} <span style='font-size:1rem; color:#9CA3AF;'>min⁻¹</span></h2>
            <p style='color:#E5E7EB; margin:0;'>t<sub>1/2</sub> = {t_half_raw:.1f} min (R² = {r2_raw:.4f})</p>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
        <div class='metric-card'>
            <h4>Kinetic Linearity (R²)</h4>
            <h2>{r2_mono:.4f}</h2>
            <p style='color:#10B981; margin:0;'>DyeFit vs. {r2_raw:.4f} Raw</p>
        </div>
        """, unsafe_allow_html=True)
        
        col4.markdown(f"""
        <div class='metric-card'>
            <h4>Intrinsic Bias Recovered</h4>
            <h2>{rate_diff_pct:+.1f}%</h2>
            <p style='color:#F59E0B; margin:0;'>Baseline & aggregate unmasking</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Kinetic Tabs
        ktab1, ktab2, ktab3, ktab4, ktab5 = st.tabs([
            "📈 Kinetic Reaction Plots",
            "🌈 Spectral Evolution",
            "🔍 Time Inspector",
            "📋 Kinetic Data Table",
            "💾 Export Package"
        ])
        
        # TAB 1: 2-Panel Kinetic Reaction Plots
        with ktab1:
            st.markdown("### Deconvolved Monomer AUC vs. Raw Absorbance Reaction Kinetics")
            fig_kin = generate_kinetic_plotly_fig(k_result, dye_name=selected_dye, theme=plotly_theme)
            st_plotly_chart(fig_kin)
            
            st.info("""
            **Scientific Physical Chemistry Insight**:
            Notice how tracking raw nominal absorbance ($A_{664}$) underestimates the intrinsic reaction rate constant 
            due to persisting nanoparticle scattering baselines and dynamic monomer–dimer re-equilibration. 
            By isolating the pure deconvolved monomer absorption area, DyeFit restores genuine, artifact-free first-order kinetics.
            """)
            
        # TAB 2: Multi-Spectrum Overlay
        with ktab2:
            st.markdown("### Time-Resolved Spectral Overlay")
            st.markdown("Evolution of the absorption profile from initial reaction ($t=0$) through degradation.")
            fig_multi = generate_multi_spectrum_plotly_fig(wavelengths, spectra_dict, dye_name=selected_dye, theme=plotly_theme)
            st_plotly_chart(fig_multi)
            
        # TAB 3: Single-Spectrum Time Inspector
        with ktab3:
            st.markdown("### Deconvolution Inspector at Selected Reaction Time")
            sorted_t = sorted(spectra_dict.keys())
            t_choice = st.select_slider("Select Reaction Time Point (min)", options=sorted_t, value=sorted_t[0])
            
            fit_t = k_result['spectral_fits'][t_choice]
            x_t = fit_t['x']
            y_corr_t = fit_t['y_corrected']
            y_fit_t = fit_t['y_fitted']
            ind_peaks_t = fit_t['individual_peaks']
            
            fig_single_t = generate_plotly_fig(x_t, y_corr_t, y_fit_t, ind_peaks_t, model_type, theme=plotly_theme)
            st_plotly_chart(fig_single_t)
            
            st.markdown(f"**Parameters at $t = {t_choice}$ min**: Monomer AUC = `{fit_t['peak_aucs'][-1]:.3f}`, Dimer AUC = `{fit_t['peak_aucs'][0]:.3f}` (R² = `{calculate_r_squared(y_corr_t, y_fit_t):.4f}`)")
            
        # TAB 4: Tabular Breakdown
        with ktab4:
            st.markdown("### Tabular Kinetic Quantitation Summary")
            st_dataframe(
                df_sum.style.format({
                    "Time": "{:.1f}",
                    "Raw_A_max": "{:.4f}",
                    "Raw_Max_Wavelength": "{:.1f}",
                    "Monomer_AUC": "{:.3f}",
                    "Dimer_AUC": "{:.3f}",
                    "Total_AUC": "{:.3f}",
                    "C_C0_Raw": "{:.4f}",
                    "C_C0_Monomer": "{:.4f}",
                    "Minus_ln_C_C0_Raw": "{:.4f}",
                    "Minus_ln_C_C0_Monomer": "{:.4f}",
                    "Fit_R2": "{:.4f}",
                    "Fit_RMSE": "{:.4f}"
                })
            )
            
        # TAB 5: Export Kinetic Package
        with ktab5:
            st.markdown("### Download Complete Kinetic Package")
            st.markdown("Export comprehensive multi-tab spreadsheets and publication-grade vector graphics matching ChemRxiv standards.")
            
            # Excel Workbook
            k_buffer = io.BytesIO()
            with pd.ExcelWriter(k_buffer, engine='openpyxl') as writer:
                # Sheet 1: Summary Table
                df_sum.to_excel(writer, sheet_name="Kinetic Summary", index=False)
                
                # Sheet 2: Rate Constants & Model Specs
                k_specs = pd.DataFrame([{
                    "Reaction Dye System": selected_dye,
                    "Deconvolution Model": model_type,
                    "Baseline Correction": baseline_method,
                    "Wavelength Window (nm)": f"{min_w:.1f} - {max_w:.1f}",
                    "DyeFit Monomer Rate k (min^-1)": k_mono,
                    "DyeFit Monomer R^2": r2_mono,
                    "DyeFit Half-Life (min)": t_half_mono,
                    "Raw Nominal Rate k (min^-1)": k_raw,
                    "Raw Nominal R^2": r2_raw,
                    "Raw Half-Life (min)": t_half_raw,
                    "Intrinsic Activity Bias (%)": rate_diff_pct
                }])
                k_specs.to_excel(writer, sheet_name="Kinetic Parameters", index=False)
                
                # Sheet 3: Multi-Spectrum Spectra
                spec_export_df = pd.DataFrame({"Wavelength (nm)": wavelengths})
                for t in sorted_t:
                    spec_export_df[f"{t} min"] = spectra_dict[t]
                spec_export_df.to_excel(writer, sheet_name="Raw Time-Series Spectra", index=False)
                
            st.download_button(
                label="📥 Download Complete Kinetic Workbook (Excel)",
                data=k_buffer.getvalue(),
                file_name=f"{selected_dye.lower().replace(' ', '_')}_kinetics_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Matplotlib 2-panel figure matching Figure 3
            fig_mat_k = generate_kinetic_matplotlib_fig(k_result, dye_name=selected_dye)
            
            png_k_buf = io.BytesIO()
            fig_mat_k.savefig(png_k_buf, format='png', dpi=300)
            st.download_button(
                label="🖼️ Download Kinetic Publication Figure (PNG - 300 DPI)",
                data=png_k_buf.getvalue(),
                file_name=f"{selected_dye.lower().replace(' ', '_')}_kinetic_figure.png",
                mime="image/png"
            )
            
            pdf_k_buf = io.BytesIO()
            fig_mat_k.savefig(pdf_k_buf, format='pdf', dpi=300)
            st.download_button(
                label="📄 Download Kinetic Publication Figure (PDF - Vector)",
                data=pdf_k_buf.getvalue(),
                file_name=f"{selected_dye.lower().replace(' ', '_')}_kinetic_figure.pdf",
                mime="application/pdf"
            )
            plt.close(fig_mat_k)
            
    else:
        st.info("Please select or upload time-series data in the sidebar to run batch reaction kinetics.")


# =============================================================================
# MAIN CONTROLLER DISPATCH
# =============================================================================
if analysis_mode == "Single Spectrum Deconvolution":
    render_single_spectrum_mode()
else:
    render_kinetic_mode()
