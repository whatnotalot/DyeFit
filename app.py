"""
===============================================================================
DyeFit Interactive Dashboard (app.py)
===============================================================================
This is the main application file that implements the interactive web dashboard
using Streamlit. 

Features:
  1. Automated spectrometer CSV/Excel file loading (and metadata stripping).
  2. Built-in synthetic datasets for training/testing.
  3. Interactive cropping and baseline correction (Polynomial, Linear, or ALS).
  4. Real-time peak deconvolution and statistical evaluation (R^2, RMSE, AUC).
  5. Multi-model comparison leaderboard.
  6. Scientific journal-standard downloads (Excel, 300 DPI PNG, Vector PDF).
  7. Iframe embedding snippet helper for WordPress pages.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    constant_offset_region
)

def safe_rerun():
    """Reruns the Streamlit app safely across all Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "session_dyes" not in st.session_state:
    import copy
    # Deep copy presets from DYE_REGISTRY so each visitor has a private, clean template
    st.session_state.session_dyes = copy.deepcopy(DYE_REGISTRY)
if "show_warning" not in st.session_state:
    st.session_state.show_warning = True
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "light"
if "layout_wide" not in st.session_state:
    st.session_state.layout_wide = True

import os
import base64

# Load project logo image asset as Base64
logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
logo_b64 = ""
if os.path.exists(logo_path):
    try:
        with open(logo_path, "rb") as img_file:
            logo_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception:
        logo_b64 = ""

# =============================================================================
# 1. PAGE SETUP & CSS INJECTION
# =============================================================================
# Set page configuration for professional design and iframe responsiveness
st.set_page_config(
    page_title="DyeFit v0.4 - Dye Deconvolution & Fitting",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS styles for Dyefit v0.4 (Modern Light & Dark Theme aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #2563EB;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
    }

    .custom-title {
        color: #1E293B;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .version-badge {
        background: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        vertical-align: middle;
    }
    .custom-subtitle {
        color: #64748B;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 2px;
        margin-bottom: 22px;
    }
    
    /* Button micro-animations */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #64748B;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #2563EB;
        background-color: #EFF6FF;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2563EB !important;
        background-color: #EFF6FF !important;
    }

    /* Top header & 3-dots Main Menu bar formatting */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99999 !important;
    }
    [data-testid="stToolbar"] {
        right: 1rem !important;
        visibility: visible !important;
        display: flex !important;
    }
    #MainMenu, [data-testid="stMainMenu"], button[aria-label="Main menu"] {
        visibility: visible !important;
        display: inline-flex !important;
    }

    /* Hide ONLY Fork button, GitHub link, and Deploy button */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    [data-testid="stHeader"] a[href*="github.com"],
    [data-testid="stHeader"] a[href*="github"],
    [data-testid="stHeader"] a[href*="fork"],
    [data-testid="stHeader"] button[title*="GitHub"],
    [data-testid="stHeader"] button[title*="git"],
    [data-testid="stHeader"] button[title*="Fork"],
    [data-testid="stHeader"] button[title*="fork"],
    [data-testid="stToolbar"] a[href*="github"],
    [data-testid="stToolbar"] a[href*="fork"],
    [data-testid="stToolbar"] button[title*="Fork"],
    [data-testid="stToolbar"] button[title*="fork"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Dynamic layout width CSS injection
if not st.session_state.get("layout_wide", True):
    st.markdown("""
    <style>
        .block-container {
            max-width: 1050px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin: 0 auto !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Dynamic theme CSS injection
curr_theme = st.session_state.get("app_theme", "light")
if curr_theme == "dark":
    st.markdown("""
    <style>
        .stApp { background-color: #0F172A !important; color: #F8FAFC !important; }
        .metric-card { background: #1E293B !important; border-color: #334155 !important; color: #F8FAFC !important; }
        .metric-card div { color: #F8FAFC !important; }
        .custom-title { color: #F8FAFC !important; }
        .custom-subtitle { color: #94A3B8 !important; }
    </style>
    """, unsafe_allow_html=True)
elif curr_theme == "system":
    st.markdown("""
    <style>
        @media (prefers-color-scheme: dark) {
            .stApp { background-color: #0F172A !important; color: #F8FAFC !important; }
            .metric-card { background: #1E293B !important; border-color: #334155 !important; color: #F8FAFC !important; }
            .metric-card div { color: #F8FAFC !important; }
            .custom-title { color: #F8FAFC !important; }
            .custom-subtitle { color: #94A3B8 !important; }
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
    if logo_b64:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
            <img src="data:image/jpeg;base64,{logo_b64}" style="width: 76px; height: 76px; border-radius: 16px; object-fit: cover; box-shadow: 0 4px 16px rgba(37,99,235,0.15); border: 2px solid #E2E8F0;">
            <div>
                <h1 class="custom-title" style="margin: 0; padding: 0;">DyeFit <span class="version-badge">v0.4</span></h1>
                <p class="custom-subtitle" style="margin: 2px 0 0 0;">Spectroscopic Baseline Correction & Peak Deconvolution Workbench</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='custom-title'>🧪 DyeFit <span class='version-badge'>v0.4</span></h1>", unsafe_allow_html=True)
        st.markdown("<p class='custom-subtitle'>Spectroscopic Baseline Correction & Peak Deconvolution Workbench</p>", unsafe_allow_html=True)
    
    with st.expander("📖 How to Use & Scientific Guide", expanded=False):
        st.markdown("""
        **Automated baseline correction, multi-model curve fitting, and peak deconvolution for UV spectra of widely used dyes in scientific studies.**
        
        ### Step-by-Step Instructions:
        1. **Load Spectrum**: Upload your raw CSV or Excel file containing wavelength and absorbance data from your spectrometer, or use a built-in demo dataset.
        2. **Dye & Baseline Config**: Select the appropriate dye template to automatically set wavelength bounds and peak guesses. Adjust the baseline removal method to flatten the background scattering.
        3. **Curve Fitting**: Choose a mathematical model (e.g., Gaussian, Voigt) and adjust the number of expected peaks. The algorithm will automatically deconvolute overlapping absorption bands.
        4. **Model Comparison**: Use the Model Comparison tab to mathematically evaluate which fitting function (based on R² and RMSE) best describes your experimental data.
        5. **Export Results**: Head to the Export tab to download your fitted data as an Excel file and generate publication-ready figures (PNG/PDF). Customize the export style via the Configuration Panel.
        """)
else:
    # Minimalist header for clean iframe rendering
    st.markdown("<h3 class='custom-title'>🧪 Dyefit Analyzer <span class='version-badge'>v0.4</span></h3>", unsafe_allow_html=True)

# Render dismissible session notice
if st.session_state.get("show_warning", True) and not is_embedded:
    col_warn, col_close = st.columns([8, 1])
    with col_warn:
        st.warning("⚠️ **Session Notice**: Custom presets created during this session are stored temporarily in your browser's memory. They are completely private and will be cleared once you refresh the page or close the tab.")
    with col_close:
        st.write("") # Spacer
        if st.button("Got it! ✅"):
            st.session_state.show_warning = False
            safe_rerun()


# =============================================================================
# 2. COLUMN AUTO-DETECTOR
# =============================================================================
def auto_detect_columns(df):
    """
    Heuristically identifies which spreadsheet column represents Wavelength (x)
    and which represents Absorbance (y).
    
    Strategy:
      1. Performs case-insensitive matching on column name strings.
      2. If string match fails, scans numerical data: classifies Wavelength 
         as the column whose median value lies between 190 and 5000 nm.
      3. Defaults to column 1 as Wavelength and column 2 as Absorbance.
    """
    wavelength_col = None
    absorbance_col = None
    
    # 1. String-based name matching
    for col in df.columns:
        col_str = str(col).lower()
        if any(w in col_str for w in ["wavelength", "wvl", "wave", "nm", "lambda"]):
            wavelength_col = col
        elif any(a in col_str for a in ["absorbance", "abs", "intensity", "optical density", "od"]):
            absorbance_col = col

    # 2. Range-based numerical heuristic (median in 190 - 5000 nm)
    if wavelength_col is None or absorbance_col is None:
        numeric_cols = []
        for col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) > 10:
                numeric_cols.append((col, series))
                
        if len(numeric_cols) >= 2:
            # Wavelength has values in UV-Vis-NIR scale (190 to 5000 nm)
            for col, series in numeric_cols:
                median_val = np.median(series)
                if 190.0 <= median_val <= 5000.0:
                    wavelength_col = col
                    break
            
            # The remaining numeric column is assigned as Absorbance
            for col, series in numeric_cols:
                if col != wavelength_col:
                    absorbance_col = col
                    break

    # 3. Last resort fallbacks
    if wavelength_col is None and len(df.columns) > 0:
        wavelength_col = df.columns[0]
    if absorbance_col is None and len(df.columns) > 1:
        absorbance_col = df.columns[1]
    elif absorbance_col is None and len(df.columns) > 0:
        absorbance_col = df.columns[0]
        
    return wavelength_col, absorbance_col


# =============================================================================
# 3. SYNTHETIC DEMO DATA GENERATION
# =============================================================================
def generate_demo_data(dye_name):
    """
    Generates synthetic absorption profiles with realistic polynomial baselines 
    and random measurement noise for testing/training purposes.
    """
    x = np.arange(400, 800, 1.0)
    # Background slope simulating solvent/scatter baseline
    base = 0.03 + 0.0001 * (x - 400) + 0.000002 * (x - 500)**2
    
    if dye_name == "Methylene Blue":
        # Two overlapping Gaussian peaks (monomer at 664nm, dimer at 612nm)
        peak1 = 0.16 * np.exp(-((x - 664.0) / 18.0) ** 2)
        peak2 = 0.08 * np.exp(-((x - 612.0) / 25.0) ** 2)
        y = base + peak1 + peak2 + np.random.normal(0, 0.001, len(x))
    elif dye_name == "Methyl Orange":
        # Broad overlapping peaks at lower wavelengths (380nm and 464nm)
        peak1 = 0.22 * np.exp(-((x - 464.0) / 45.0) ** 2)
        peak2 = 0.06 * np.exp(-((x - 380.0) / 30.0) ** 2)
        y = base + peak1 + peak2 + np.random.normal(0, 0.001, len(x))
    else:
        # Standard generic dye single peak
        peak1 = 0.18 * np.exp(-((x - 550.0) / 35.0) ** 2)
        y = base + peak1 + np.random.normal(0, 0.001, len(x))
        
    return pd.DataFrame({"Wavelength": x, "Absorbance": y})


# =============================================================================
# 4. SIDEBAR - FILE INGESTION
# =============================================================================
if logo_b64:
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 16px; margin-top: -10px;">
        <img src="data:image/jpeg;base64,{logo_b64}" style="width: 140px; height: 140px; border-radius: 18px; object-fit: cover; box-shadow: 0 8px 24px rgba(0,0,0,0.12); border: 3px solid #FFFFFF;">
    </div>
    """, unsafe_allow_html=True)
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

# Proceed only if data is successfully loaded into the DataFrame
if uploaded_df is not None:
    # Clean whitespace and cast headers to strings
    uploaded_df.columns = [str(col).strip() for col in uploaded_df.columns]
    col_names = list(uploaded_df.columns)
    
    # Run auto-detector and set default index selections
    detected_wl, detected_abs = auto_detect_columns(uploaded_df)
    wl_index = col_names.index(detected_wl) if detected_wl in col_names else 0
    abs_index = col_names.index(detected_abs) if detected_abs in col_names else (1 if len(col_names) > 1 else 0)
    
    wavelength_col = st.sidebar.selectbox("Wavelength Column", col_names, index=wl_index, key=f"wl_{data_source}")
    absorbance_col = st.sidebar.selectbox("Absorbance Column", col_names, index=abs_index, key=f"abs_{data_source}")
    
    # Ensure selected columns exist and extract numeric arrays (coercing text/headers to NaN)
    if wavelength_col in uploaded_df.columns and absorbance_col in uploaded_df.columns:
        df_clean = uploaded_df[[wavelength_col, absorbance_col]].copy()
        df_clean[wavelength_col] = pd.to_numeric(df_clean[wavelength_col], errors='coerce')
        df_clean[absorbance_col] = pd.to_numeric(df_clean[absorbance_col], errors='coerce')
        
        # Drop all rows containing NaN values (automatically strips header text blocks)
        df_clean = df_clean.dropna().sort_values(by=wavelength_col)
        
        if len(df_clean) == 0:
            st.error("No numeric data found in the selected columns. Please check your data sheet.")
            st.stop()
            
        wavelength = df_clean[wavelength_col].to_numpy()
        absorbance = df_clean[absorbance_col].to_numpy()
    else:
        st.info("Syncing columns...")
        st.stop()
    
    # =============================================================================
    # 5. SIDEBAR - DYE & BASELINE SETTINGS
    # =============================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🧪 2. Dye & Baseline Configuration")
    
    dye_options = list(st.session_state.session_dyes.keys()) + ["Custom", "➕ Create Custom Preset"]
    selected_dye = st.sidebar.selectbox("Select Dye Config Preset", dye_options)
    
    # ----------------- CUSTOM PRESET BUILDER -----------------
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
                # Store in session state dictionary (completely in-memory, thread-safe, temporary)
                st.session_state.session_dyes[custom_name] = config_new
                st.sidebar.success(f"Preset '{custom_name}' added to your session!")
                safe_rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to save preset: {e}")
                
        # Setup temporary config for interactive preview
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
        
    # Crop spectrum to region of interest
    if selected_dye == "➕ Create Custom Preset":
        min_w = st.sidebar.slider("Min Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), float(custom_min_w))
        max_w = st.sidebar.slider("Max Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), float(custom_max_w))
    else:
        min_w = st.sidebar.slider("Min Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), min_w_val)
        max_w = st.sidebar.slider("Max Wavelength (nm)", float(np.min(wavelength)), float(np.max(wavelength)), max_w_val)
    
    crop_mask = (wavelength >= min_w) & (wavelength <= max_w)
    x_cropped = wavelength[crop_mask]
    y_cropped = absorbance[crop_mask]
    
    # Select baseline method
    baseline_method = st.sidebar.selectbox(
        "Baseline Subtraction Method", 
        [
            "Polynomial Fit", 
            "Linear Subtraction", 
            "Asymmetric Least Squares (ALS)",
            "Constant Offset (Single Wavelength)",
            "Constant Offset (Region Average)"
        ]
    )
    
    if baseline_method == "Polynomial Fit":
        poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)
        
        st.sidebar.markdown("**Baseline Fitting Regions (nm)**")
        if selected_dye not in ["Custom", "➕ Create Custom Preset"]:
            st.sidebar.info(f"Default preset baseline ranges are active for {selected_dye}.")
            baseline_mask = config.baseline_mask_fn(x_cropped)
        elif selected_dye == "➕ Create Custom Preset":
            st.sidebar.info(f"Custom preset baseline values: <{custom_base_left} nm or >{custom_base_right} nm.")
            baseline_mask = config.baseline_mask_fn(x_cropped)
        else:
            left_bound = st.sidebar.slider("Left Baseline Threshold", float(min_w), float(max_w), float(min_w + (max_w - min_w)*0.15))
            right_bound = st.sidebar.slider("Right Baseline Threshold", float(min_w), float(max_w), float(max_w - (max_w - min_w)*0.15))
            baseline_mask = (x_cropped < left_bound) | (x_cropped > right_bound)
            
        baseline = polynomial_baseline(x_cropped, y_cropped, baseline_mask, degree=poly_degree)
        
    elif baseline_method == "Linear Subtraction":
        baseline = linear_baseline(x_cropped, y_cropped)
        
    elif baseline_method == "Asymmetric Least Squares (ALS)":
        st.sidebar.markdown("**ALS Parameters**")
        lam_exp = st.sidebar.slider("Smoothness (Log10 Lambda)", min_value=2.0, max_value=9.0, value=5.0, step=0.1)
        p_val = st.sidebar.slider("Asymmetry (p)", 0.001, 0.100, 0.010, step=0.001, format="%.3f")
        baseline = als_baseline(y_cropped, lam=10**lam_exp, p=p_val)
        
    elif baseline_method == "Constant Offset (Single Wavelength)":
        st.sidebar.markdown("**Single Wavelength Offset**")
        # Default reference: 800 nm, clamped to current crop range
        ref_val_default = 800.0 if 800.0 >= float(min_w) and 800.0 <= float(max_w) else float(max_w)
        ref_wave = st.sidebar.slider("Reference Wavelength (nm)", float(min_w), float(max_w), float(ref_val_default), step=1.0)
        baseline = constant_offset_single(x_cropped, y_cropped, ref_wavelength=ref_wave)
        
    else: # Constant Offset (Region Average)
        st.sidebar.markdown("**Region Average Offset**")
        # Default range: last 20 nm or last 5% of spectrum
        default_min = float(max_w - 20.0) if (max_w - 20.0) >= float(min_w) else float(min_w + (max_w - min_w)*0.9)
        default_max = float(max_w)
        ref_range = st.sidebar.slider("Wavelength Region (nm)", float(min_w), float(max_w), (default_min, default_max), step=1.0)
        baseline = constant_offset_region(x_cropped, y_cropped, range_min=ref_range[0], range_max=ref_range[1])
        
    # Subtract baseline
    abs_corrected = y_cropped - baseline
    
    # =============================================================================
    # 6. SIDEBAR - CURVE FITTING SETTINGS
    # =============================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("📈 3. Curve Fitting Settings")
    num_peaks = st.sidebar.number_input("Number of Peaks to Deconvolute", min_value=1, max_value=5, value=num_peaks_default)
    
    auto_detect = st.sidebar.checkbox("Auto-Detect Peak Positions", value=True)
    custom_guesses = None
    if not auto_detect:
        custom_input = st.sidebar.text_input("Custom Peak Centers (comma-separated)", "")
        if custom_input:
            try:
                # Split by commas and filter out empty values (handles spaces and trailing commas gracefully)
                custom_guesses = [float(x.strip()) for x in custom_input.split(",") if x.strip() != ""]
            except ValueError:
                st.sidebar.warning("Please enter valid numbers (separated by commas).")
                
    model_type = st.sidebar.selectbox("Curve Fitting Model", ["Gaussian", "Lorentzian", "Voigt", "Pseudo-Voigt", "Asymmetric Gaussian"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎨 4. App View & Theme Settings")
    theme_choice = st.sidebar.radio("Site & Chart Theme", ["Clean Light (Default)", "Publication Light", "Web Dark Mode"])
    if theme_choice == "Publication Light":
        plotly_theme = "publication_light"
    elif theme_choice == "Web Dark Mode":
        plotly_theme = "dark"
        st.markdown("""
        <style>
            .stApp { background-color: #0F172A !important; color: #F8FAFC !important; }
            .metric-card { background: #1E293B !important; border-color: #334155 !important; color: #F8FAFC !important; }
            .metric-card div { color: #F8FAFC !important; }
            .custom-title { color: #F8FAFC !important; }
            .custom-subtitle { color: #94A3B8 !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        plotly_theme = "light"
    
    # =============================================================================
    # 7. MAIN INTERFACE TABS
    # =============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Baseline Correction", "🔍 Peak Fitting & Deconvolution", "🧪 Model Comparison", "💾 Export Results"])
    
    # -------------------------------------------------------------------------
    # TAB 1: Baseline Correction View
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### Raw vs. Baseline-Corrected Spectrum")
        st.markdown("Adjust parameters in the sidebar to flat-correct the baseline before fitting peaks.")
        
        # Setup colors based on dynamic Plotly theme
        if plotly_theme == "publication_light":
            raw_color = "#64748B"  
            base_color = "#D55E00" 
            corr_color = "#000000" 
            bg_color = "#FFFFFF"
            paper_bg = "#FFFFFF"
            grid_color = "#E5E7EB"
            text_color = "#111827"
            border_color = "#111827"
            layout_w_h = dict(width=580, height=435) 
        elif plotly_theme == "light":
            raw_color = "#94A3B8"  
            base_color = "#D97706" 
            corr_color = "#2563EB" 
            bg_color = "#FFFFFF"
            paper_bg = "#FFFFFF"
            grid_color = "#E2E8F0"
            text_color = "#0F172A"
            border_color = "#94A3B8"
            layout_w_h = dict()
        else: # dark
            raw_color = "#CCCCCC"
            base_color = "#FFC700"
            corr_color = "#00D2FF"
            bg_color = "rgba(0,0,0,0)"
            paper_bg = "rgba(0,0,0,0)"
            grid_color = "rgba(255,255,255,0.1)"
            text_color = "#F3F4F6"
            border_color = "rgba(255,255,255,0.2)"
            layout_w_h = dict()
            
        fig_base = go.Figure()
        fig_base.add_trace(go.Scatter(x=x_cropped, y=y_cropped, name="Raw Absorbance", line=dict(color=raw_color, width=1.5)))
        fig_base.add_trace(go.Scatter(x=x_cropped, y=baseline, name="Fitted Baseline", line=dict(color=base_color, width=1.5, dash='dash')))
        fig_base.add_trace(go.Scatter(x=x_cropped, y=abs_corrected, name="Corrected Absorbance", line=dict(color=corr_color, width=2)))
        
        layout_opts = dict(
            title="Baseline Fitting & Correction",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance",
            paper_bgcolor=paper_bg,
            plot_bgcolor=bg_color,
            font=dict(family="Inter, Arial, sans-serif", size=11, color=text_color),
            margin=dict(l=50, r=40, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        layout_opts.update(layout_w_h)
        fig_base.update_layout(**layout_opts)
        
        fig_base.update_xaxes(showgrid=True, gridwidth=0.8, gridcolor=grid_color, linecolor=border_color, mirror=True, ticks="inside")
        fig_base.update_yaxes(showgrid=True, gridwidth=0.8, gridcolor=grid_color, linecolor=border_color, mirror=True, ticks="inside")
        
        st.plotly_chart(fig_base, use_container_width=plotly_theme != "publication_light")
        
    # -------------------------------------------------------------------------
    # TAB 2: Peak Fitting & Deconvolution View
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### Peak Deconvolution Fitting")
        
        try:
            # Execute bounded non-linear regression using the dyefit solver
            popt, pcov, y_fitted, individual_peaks = fit_spectrum(
                x_cropped, abs_corrected, model_type, num_peaks=num_peaks, custom_guesses=custom_guesses
            )
            
            col_chart, col_stats = st.columns([2, 1])
            
            with col_chart:
                # Render interactive deconvolution plot
                fig_fit = generate_plotly_fig(x_cropped, abs_corrected, y_fitted, individual_peaks, model_type, theme=plotly_theme)
                st.plotly_chart(fig_fit, use_container_width=plotly_theme != "publication_light")
                
            with col_stats:
                st.markdown("#### Goodness of Fit")
                r2 = calculate_r_squared(abs_corrected, y_fitted)
                rmse = calculate_rmse(abs_corrected, y_fitted)
                
                # Metric cards styled for Light Theme
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:12px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;'>R² (Fit Accuracy)</div>
                    <div style='font-size:26px;font-weight:700;color:#059669;margin-top:2px;'>{r2:.5f}</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size:12px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;'>RMSE (Residual Error)</div>
                    <div style='font-size:26px;font-weight:700;color:#E11D48;margin-top:2px;'>{rmse:.5f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Peak Properties")
                
                from dyefit.fitting import get_num_params
                num_p = get_num_params(model_type)
                
                peaks_data = []
                for i in range(num_peaks):
                    p_peak = popt[i * num_p : (i + 1) * num_p]
                    peak_y = individual_peaks[i]
                    auc_val = calculate_auc(x_cropped, peak_y)
                    
                    amp_val = p_peak[0]
                    cen_val = p_peak[1]
                    wid_val = p_peak[2]
                    
                    peaks_data.append({
                        "Peak #": f"Peak {i+1}",
                        "Center Wavelength (nm)": f"{cen_val:.2f}",
                        "Absorbance (Height)": f"{amp_val:.4f}",
                        "Width Parameter": f"{wid_val:.2f}",
                        "Area Under Curve (AUC)": f"{auc_val:.4f}"
                    })
                
                st.dataframe(pd.DataFrame(peaks_data), hide_index=True)
                
        except Exception as e:
            st.error(f"Curve fitting failed. Try modifying the number of peaks or custom guesses. Details: {e}")
            
    # -------------------------------------------------------------------------
    # TAB 3: Model Comparison View
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### Model Selection Analysis")
        st.markdown("This tab fits all 5 profiles simultaneously, helping you choose the mathematically superior model based on R² and RMSE.")
        
        if st.button("🚀 Run Comparative Model Fit"):
            comparison_results = []
            
            with st.spinner("Fitting models..."):
                for m in ["Gaussian", "Lorentzian", "Voigt", "Pseudo-Voigt", "Asymmetric Gaussian"]:
                    try:
                        popt_m, _, y_fitted_m, individual_peaks_m = fit_spectrum(
                            x_cropped, abs_corrected, m, num_peaks=num_peaks, custom_guesses=custom_guesses
                        )
                        r2_m = calculate_r_squared(abs_corrected, y_fitted_m)
                        rmse_m = calculate_rmse(abs_corrected, y_fitted_m)
                        
                        # Generate lists of peak centers and AUCs for comparison
                        aucs_str = ", ".join([f"P{i+1}: {calculate_auc(x_cropped, py):.3f}" for i, py in enumerate(individual_peaks_m)])
                        num_p = get_num_params(m)
                        centers_str = ", ".join([f"P{i+1}: {popt_m[i*num_p + 1]:.1f}nm" for i in range(num_peaks)])
                        
                        comparison_results.append({
                            "Mathematical Model": m,
                            "R² (higher is better)": r2_m,
                            "RMSE (lower is better)": rmse_m,
                            "Peak Centers (nm)": centers_str,
                            "Peak AUCs": aucs_str
                        })
                    except Exception:
                        comparison_results.append({
                            "Mathematical Model": m,
                            "R² (higher is better)": np.nan,
                            "RMSE (lower is better)": np.nan,
                            "Peak Centers (nm)": "Fit failed",
                            "Peak AUCs": "N/A"
                        })
                        
            comp_df = pd.DataFrame(comparison_results).sort_values(by="R² (higher is better)", ascending=False)
            st.dataframe(comp_df, hide_index=True)
            
    # -------------------------------------------------------------------------
    # TAB 4: Export Results
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### Export Results")
        
        st.markdown("#### Publication Style Configurator")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            export_font = st.selectbox("Font Family", ["Arial", "Helvetica", "Times New Roman", "Computer Modern"])
            export_dim = st.radio("Layout Dimensions", ["Single-Column 3.3\"", "Double-Column 6.8\""])
            export_grid = st.checkbox("Show Gridlines", value=False)
        with col_c2:
            export_linewidth = st.slider("Line Thickness (pt)", 1.0, 3.0, 1.2, step=0.1)
            export_fontsize = st.slider("Font Size (pt)", 8, 14, 8)
        st.markdown("---")
        
        # Only render export buttons if curve fitting was successful
        if 'popt' in locals():
            # Build the Excel Workbook in memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Sheet 1: Export fitted dataset values
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
                
                # Sheet 2: Export peak parameters
                num_p = get_num_params(model_type)
                param_export = []
                for i in range(num_peaks):
                    p_peak = popt[i * num_p : (i + 1) * num_p]
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
            
            # Generate static Matplotlib figure (strictly formatted publication style)
            from dyefit import generate_matplotlib_fig
            fig_matplotlib = generate_matplotlib_fig(
                x_cropped, abs_corrected, y_fitted, individual_peaks, model_type,
                font_family=export_font,
                layout_dim=export_dim,
                line_width=export_linewidth,
                font_size=export_fontsize,
                show_grid=export_grid
            )
            
            # Export to high-res PNG
            png_buffer = io.BytesIO()
            fig_matplotlib.savefig(png_buffer, format='png', dpi=300)
            st.download_button(
                label="🖼️ Download Journal Figure (PNG - 300 DPI)",
                data=png_buffer.getvalue(),
                file_name=f"{selected_dye.lower().replace(' ', '_')}_figure.png",
                mime="image/png"
            )
            
            # Export to vector PDF
            pdf_buffer = io.BytesIO()
            fig_matplotlib.savefig(pdf_buffer, format='pdf', dpi=300)
            st.download_button(
                label="📄 Download Journal Figure (PDF - Vector)",
                data=pdf_buffer.getvalue(),
                file_name=f"{selected_dye.lower().replace(' ', '_')}_figure.pdf",
                mime="application/pdf"
            )
            
            # Close the Matplotlib figure to free server memory resources
            plt.close(fig_matplotlib)

else:
    st.info("Please upload an Excel/CSV file or select 'Use Demo Data' in the sidebar to start spectroscopic curve deconvolution.")

# =============================================================================
# BOTTOM QUICK VIEW TUNING TOOLBAR
# =============================================================================
st.markdown("<br><hr style='border: 0; height: 1px; background: #E2E8F0; margin-top: 30px; margin-bottom: 20px;'>", unsafe_allow_html=True)
st.markdown("#### ⚙️ Quick View Tuning & Appearance Shortcuts")

col_v1, col_v2, col_v3, col_vw = st.columns([1, 1, 1, 1.4])

with col_v1:
    btn_l_label = "☼ Light Mode ✅" if st.session_state.get("app_theme") == "light" else "☼ Light Mode"
    if st.button(btn_l_label, key="b_theme_light", use_container_width=True):
        st.session_state.app_theme = "light"
        safe_rerun()

with col_v2:
    btn_d_label = "☾ Dark Mode ✅" if st.session_state.get("app_theme") == "dark" else "☾ Dark Mode"
    if st.button(btn_d_label, key="b_theme_dark", use_container_width=True):
        st.session_state.app_theme = "dark"
        safe_rerun()

with col_v3:
    btn_s_label = "◐ System Theme ✅" if st.session_state.get("app_theme") == "system" else "◐ System Theme"
    if st.button(btn_s_label, key="b_theme_system", use_container_width=True):
        st.session_state.app_theme = "system"
        safe_rerun()

with col_vw:
    layout_label = "↔️ Wide View (Active)" if st.session_state.get("layout_wide", True) else "↕️ Centered / Standard View"
    if st.button(layout_label, key="b_layout_toggle", use_container_width=True):
        st.session_state.layout_wide = not st.session_state.get("layout_wide", True)
        safe_rerun()
