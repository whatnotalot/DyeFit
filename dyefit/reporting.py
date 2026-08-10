"""
===============================================================================
Statistical Calculations & Figure Reporting Module
===============================================================================
This module calculates fit quality indicators, performs trapezoidal integration
to calculate integrated peak areas (AUC), and generates publication-grade 
visualizations for both on-screen dashboards and scientific journals.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =============================================================================
# 1. STATISTICAL EVALUATORS
# =============================================================================
def calculate_r_squared(y_true, y_pred):
    """
    Calculates the Coefficient of Determination (R^2).
    
    R^2 measures the proportion of variance in the experimental data that is 
    explained by the model fit. Values close to 1.0 indicate excellent fits.
    
    Mathematical formula:
        R^2 = 1 - (SS_res / SS_tot)
        
        Where:
          - SS_res = sum( (y_true - y_pred)^2 ) [residual sum of squares]
          - SS_tot = sum( (y_true - mean(y_true))^2 ) [total sum of squares]
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).
    
    RMSE represents the standard deviation of the residuals (prediction errors),
    indicating the absolute average distance between data points and the fit line.
    
    Mathematical formula:
        RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_aic(y_true, y_pred, num_params):
    """
    Calculates the Akaike Information Criterion (AIC).
    
    Used to compare different models (e.g. Gaussian vs. Voigt). It rewards 
    fitting accuracy but penalizes model complexity (number of parameters, k) 
    to prevent overfitting. Lower values are preferred.
    
    Mathematical formula:
        AIC = n * ln(RSS / n) + 2 * k
        
        Where:
          - n: number of data points
          - RSS: residual sum of squares
          - k (num_params): total fitted parameters
    """
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss == 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * num_params


def calculate_bic(y_true, y_pred, num_params):
    """
    Calculates the Bayesian Information Criterion (BIC).
    
    Similar to AIC, but introduces a larger penalty for parameters based on 
    the sample size (n). Highly recommended for large spectroscopic datasets.
    
    Mathematical formula:
        BIC = n * ln(RSS / n) + k * ln(n)
    """
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss == 0:
        return -np.inf
    return n * np.log(rss / n) + num_params * np.log(n)


# =============================================================================
# 2. NUMERICAL AREA INTEGRATION (AUC)
# =============================================================================
def calculate_auc(x, y):
    """
    Calculates the Area Under the Curve (AUC) using the composite Trapezoidal Rule.
    
    This custom implementation avoids deprecation warnings associated with 
    `numpy.trapz` under NumPy 2.0+ by explicitly calculating intervals.
    
    Mathematical formula:
        Integral ≈ sum_{i=0}^{N-2} [ (y_i + y_{i+1}) / 2 ] * (x_{i+1} - x_i)
        
        This divides the area beneath the curve into individual trapezoids, 
        calculates the area of each, and sums them.
    """
    # Ensure data is sorted by wavelength (x) to prevent negative area segments
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Calculate spacing (width) of each trapezoid (dx)
    dx = np.diff(x_sorted)
    
    # Calculate average height of each trapezoid (y_mean)
    y_mean = (y_sorted[:-1] + y_sorted[1:]) / 2.0
    
    # Total area is the sum of (height * width) for all intervals
    return np.sum(y_mean * dx)


# =============================================================================
# 3. SCIENTIFIC MATPLOTLIB FIGURE GENERATION (PNG/PDF EXPORT)
# =============================================================================
def generate_matplotlib_fig(x_data, y_data, y_fitted, individual_peaks, model_type,
                            font_family='Arial', layout_dim='Single-Column 3.3"', 
                            line_width=1.0, font_size=8, show_grid=False):
    """
    Generates a publication-ready Matplotlib figure matching ACS/RSC/Nature layouts.
    
    Formatting specifications:
      - Size: Dynamic based on layout_dim (Single-Column or Double-Column).
      - Resolution: 300 DPI (high-definition vector-like rendering).
      - Borders: Closed bounding box with inward-facing ticks on all four sides.
      - Contrast: Solid black raw data, dashed red fit, and colorblind-safe peak fills.
      - Typography: Dynamic font family and size based on user configuration.
    """
    # Define journal standards in Matplotlib config overrides
    if font_family in ['Times New Roman', 'Computer Modern']:
        plt.rcParams['font.family'] = 'serif'
        if font_family == 'Computer Modern':
            plt.rcParams['font.serif'] = ['Computer Modern Roman']
        else:
            plt.rcParams['font.serif'] = ['Times New Roman']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_family, 'Helvetica', 'Arial', 'DejaVu Sans']
        
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig_width = 6.8 if "Double-Column" in layout_dim else 3.3
    fig_height = fig_width * (2.5 / 3.3)
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    
    # Plot experimental data (solid black)
    ax.plot(x_data, y_data, color='black', linestyle='-', label='Data', linewidth=line_width)
    
    # Plot overall fit (dashed red)
    ax.plot(x_data, y_fitted, color='red', linestyle='--', label='Fit', linewidth=line_width * 1.2)
    
    # Colorblind-safe palette (Okabe-Ito) and distinct line dashes for deconvoluted peaks
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']
    styles = [':', '-.', ':', '-.']
    
    for i, peak_y in enumerate(individual_peaks):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        ax.plot(x_data, peak_y, color=color, linestyle=style, label=f'Peak {i+1}', linewidth=line_width)
        
    ax.set_xlabel('Wavelength (nm)', fontsize=font_size+1, fontweight='normal')
    ax.set_ylabel('Absorbance', fontsize=font_size+1, fontweight='normal')
    
    # Ticks pointing inwards on all sides
    ax.tick_params(direction='in', top=True, right=True, labelsize=font_size-1, width=0.8)
    
    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6)
        
    # Remove outer legend frame to save space
    ax.legend(loc='best', frameon=False, fontsize=font_size-1)
    
    ax.set_xlim(np.min(x_data), np.max(x_data))
    ax.set_ylim(bottom=0.0)
    
    fig.tight_layout()
    return fig


# =============================================================================
# 4. INTERACTIVE PLOTLY FIGURE GENERATION (DASHBOARD PREVIEW)
# =============================================================================
def generate_plotly_fig(x_data, y_data, y_fitted, individual_peaks, model_type, theme="dark"):
    """
    Generates a Plotly figure with dynamic styling (Web Dark or Publication Light).
    
    Features:
      - Web Dark: Transparent background, glowing neon traces, optimized for monitors.
      - Publication Light: White background, black axis lines, inward ticks, 
        and a locked 4:3 aspect ratio (580x435 px) to match journal layout.
    """
    fig = go.Figure()
    
    # Define styling colors based on the chosen theme
    if theme == "publication_light":
        data_color = "#000000"       # Black for raw data
        fit_color = "#FF2E93"        # Pink/red for fit
        bg_color = "#FFFFFF"         # White background
        paper_bg = "#FFFFFF"
        grid_color = "#E5E7EB"       # Light gray grid
        text_color = "#111827"       # Dark text
        border_color = "#111827"
        peak_colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00']
    else: # Default Web Dark
        data_color = "#00D2FF"       # Cyan for data
        fit_color = "#FF2E93"        # Pink for fit
        bg_color = "rgba(0,0,0,0)"   # Transparent plot
        paper_bg = "rgba(0,0,0,0)"   # Transparent paper
        grid_color = "rgba(255,255,255,0.1)"
        text_color = "#F3F4F6"       # Light gray text
        border_color = "rgba(255,255,255,0.2)"
        peak_colors = ['#FFC700', '#00FF66', '#A100FF', '#FF7A00', '#00E1FF']

    # 1. Plot Corrected Experimental Data
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=y_data,
        mode='lines',
        name='Data',
        line=dict(color=data_color, width=1.8)
    ))
    
    # 2. Plot Overall Fit Model
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=y_fitted,
        mode='lines',
        name=f'Fit ({model_type})',
        line=dict(color=fit_color, width=2.2, dash='dash')
    ))
    
    # 3. Plot Individual Deconvoluted Peaks (with transparent fills)
    for i, peak_y in enumerate(individual_peaks):
        color = peak_colors[i % len(peak_colors)]
        dash_style = 'dot' if i % 2 == 0 else 'dashdot'
        
        # Convert hex color to semi-transparent RGBA color
        if color.startswith("#"):
            rgba_color = f'rgba{tuple(list(int(color[j:j+2], 16) for j in (1, 3, 5)) + [0.08])}'
        else:
            rgba_color = 'rgba(0,0,0,0.05)'
            
        fig.add_trace(go.Scatter(
            x=x_data, 
            y=peak_y,
            mode='lines',
            name=f'Peak {i+1}',
            line=dict(color=color, width=1.5, dash=dash_style),
            fill='tozeroy',  # Fill down to the baseline
            fillcolor=rgba_color
        ))
        
    layout_opts = dict(
        title=f'Deconvoluted {model_type} Fit',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Absorbance',
        paper_bgcolor=paper_bg,
        plot_bgcolor=bg_color,
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=11,
            color=text_color
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=40, t=80, b=50),
    )
    
    # Set a fixed, clean aspect ratio for publication format (e.g. 4:3)
    if theme == "publication_light":
        layout_opts.update(dict(
            width=580,
            height=435  # Exactly 4:3 ratio
        ))
        
    fig.update_layout(**layout_opts)
    
    # Configure axes borders and inward ticks
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=0.8, 
        gridcolor=grid_color,
        linecolor=border_color,
        mirror=True,
        ticks="inside"
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.8, 
        gridcolor=grid_color,
        linecolor=border_color,
        mirror=True,
        ticks="inside"
    )
    
    return fig
