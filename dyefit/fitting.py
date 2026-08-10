"""
===============================================================================
Spectroscopic Fitting & Deconvolution Engine
===============================================================================
This module contains the curve-fitting and peak deconvolution algorithms. 
It uses non-linear least-squares optimization to fit multiple peaks to a 
baseline-corrected spectrum. 

Key workflow steps:
  1. Count degrees of freedom (parameters per peak model).
  2. Estimate initial parameter guesses using automated signal peak-finding.
  3. Establish parameter boundaries to prevent non-physical fitting.
  4. Perform optimization using SciPy's bounded Levenberg-Marquardt / Trust-Region.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from .models import gaussian, lorentzian, voigt, pseudo_voigt, asym_gaussian

# =============================================================================
# 1. PARAMETER COUNTING & DECODING
# =============================================================================
def get_num_params(model_type):
    """
    Returns the number of parameters per peak for the specified model type.
    
    This dictates how we unpack the flat 1D parameter array for the optimizer.
    """
    if model_type in ['Gaussian', 'Lorentzian']:
        return 3  # [amplitude, center, width]
    elif model_type in ['Voigt', 'Pseudo-Voigt', 'Asymmetric Gaussian']:
        return 4  # [amplitude, center, width, shape_parameter]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def combined_model_systematic(x, model_type, *params):
    """
    Combines multiple individual peak shapes into a single cumulative curve.
    
    This function:
      1. Determines how many peaks are being fitted by dividing the total
         parameter list length by the number of parameters per peak.
      2. Unpacks the parameters for each peak.
      3. Evaluates each peak individually.
      4. Sums all peak curves to get the cumulative fit curve.
      
    Args:
        x (np.array): Wavelength array.
        model_type (str): Name of peak model ('Gaussian', 'Voigt', etc.).
        *params: Flat list of parameters for all peaks (e.g. [A1, xc1, w1, A2, xc2, w2...]).
        
    Returns:
        tuple: (cumulative_absorbance_array, list_of_individual_peak_arrays)
    """
    num_params_per_peak = get_num_params(model_type)
    num_peaks = len(params) // num_params_per_peak
    
    y = np.zeros_like(x, dtype=float)
    individual_peaks = []
    
    # Unpack and evaluate each peak sequentially
    for i in range(num_peaks):
        p = params[i * num_params_per_peak:(i + 1) * num_params_per_peak]
        try:
            if model_type == 'Gaussian':
                peak_y = gaussian(x, *p)
            elif model_type == 'Lorentzian':
                peak_y = lorentzian(x, *p)
            elif model_type == 'Voigt':
                peak_y = voigt(x, *p)
            elif model_type == 'Pseudo-Voigt':
                peak_y = pseudo_voigt(x, *p)
            elif model_type == 'Asymmetric Gaussian':
                peak_y = asym_gaussian(x, *p)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            y += peak_y
            individual_peaks.append(peak_y)
        except Exception:
            # Fallback in case parameter edge-cases trigger evaluation errors
            individual_peaks.append(np.zeros_like(x, dtype=float))
            
    return y, individual_peaks


def curve_fit_wrapper(x, model_type, *params):
    """
    A helper wrapper function that returns only the cumulative fitted spectrum.
    
    This is required because scipy.optimize.curve_fit expects a function that 
    returns a single 1D array representing the total model curve.
    """
    y, _ = combined_model_systematic(x, model_type, *params)
    return y


# =============================================================================
# 2. AUTOMATIC PEAK DETECTION (INITIAL GUESSING)
# =============================================================================
def auto_detect_peaks(x_data, y_data, num_peaks=2):
    """
    Uses signal processing (prominence and distance checks) to locate major peaks.
    
    Mathematical concept:
      - Uses `scipy.signal.find_peaks` to identify local maxima.
      - Prominence threshold: Only identifies peaks that rise at least 2% above 
        local background noise.
      - Minimum Distance constraint: Restricts peaks from being located within 
        15 nm of each other to prevent fitting overlapping noise.
        
    Args:
        x_data (np.array): Wavelength vector.
        y_data (np.array): Absorbance vector.
        num_peaks (int): The target number of peaks to locate.
        
    Returns:
        tuple: (detected_centers_array, detected_amplitudes_array)
    """
    # 2% prominence threshold prevents noise fitting
    prominence = 0.02 * np.max(y_data)
    
    # Calculate spacing (in data index units) corresponding to 15 nm separation
    step = abs(x_data[1] - x_data[0]) if len(x_data) > 1 else 1.0
    distance = int(15 / step) if step > 0 else 10
    
    # Find peaks using SciPy's topological peak-finding
    peaks, _ = find_peaks(y_data, prominence=prominence, distance=max(1, distance))
    
    if len(peaks) == 0:
        # Fallback: Segment the wavelength space equally if no clear peak is found
        centers = np.linspace(np.min(x_data) + 20, np.max(x_data) - 20, num_peaks)
        amplitudes = [np.max(y_data) * 0.5] * num_peaks
    else:
        # Sort detected peaks by absorbance value (descending)
        peak_heights = y_data[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sorted_indices[:num_peaks]]
        
        # Sort them by wavelength (ascending) so Peak 1 is left and Peak 2 is right
        top_peaks = sorted(top_peaks)
        
        centers = x_data[top_peaks]
        amplitudes = y_data[top_peaks]
        
        # If we found fewer peaks than requested, fill the remaining with center values
        while len(centers) < num_peaks:
            centers = np.append(centers, np.mean(x_data))
            amplitudes = np.append(amplitudes, np.max(y_data) * 0.5)
            
    return centers, amplitudes


def get_initial_guesses(model_type, x_data, y_data, num_peaks=2, custom_guesses=None):
    """
    Assembles the initial parameters list (p0) for the curve fitter.
    
    Initial guesses are formatted as: [Amp1, Cen1, Width1, ShapeParam1, Amp2, Cen2...]
    """
    centers = []
    amplitudes = []
    
    # 1. First, load all valid custom guesses up to the requested number of peaks
    if custom_guesses is not None and len(custom_guesses) > 0:
        for c in custom_guesses[:num_peaks]:
            try:
                # Find the index of the closest wavelength in the actual data
                idx = np.argmin(np.abs(x_data - c))
                centers.append(float(c))
                amplitudes.append(float(y_data[idx]))
            except Exception:
                pass
                
    # 2. If we need more peaks than custom guesses provided, fill in using auto-detection
    if len(centers) < num_peaks:
        detected_centers, detected_amplitudes = auto_detect_peaks(x_data, y_data, num_peaks)
        
        # Append auto-detected peaks that are not already close to the user's custom guesses
        for dc, da in zip(detected_centers, detected_amplitudes):
            if len(centers) >= num_peaks:
                break
            # Avoid adding duplicate peaks within 5nm of already selected ones
            if not any(abs(dc - cc) < 5.0 for cc in centers):
                centers.append(float(dc))
                amplitudes.append(float(da))
                
        # If still not enough (due to duplicates/overlaps), just append the remaining detected peaks
        for dc, da in zip(detected_centers, detected_amplitudes):
            if len(centers) >= num_peaks:
                break
            if float(dc) not in centers:
                centers.append(float(dc))
                amplitudes.append(float(da))
                
        # Ultimate fallback: pad with the mean wavelength of the cropped area
        while len(centers) < num_peaks:
            centers.append(float(np.mean(x_data)))
            amplitudes.append(float(np.max(y_data) * 0.5))
            
    # 3. Sort the centers and amplitudes in ascending order of wavelength
    sorted_indices = np.argsort(centers)
    centers = [centers[i] for i in sorted_indices]
    amplitudes = [amplitudes[i] for i in sorted_indices]
        
    # Default width guess: 5% of the cropped spectrum's wavelength range
    x_range = np.max(x_data) - np.min(x_data)
    wid_guess = x_range * 0.05
    
    guesses = []
    for i in range(num_peaks):
        amp = amplitudes[i]
        cen = centers[i]
        
        if model_type in ['Gaussian', 'Lorentzian']:
            guesses.extend([amp, cen, wid_guess])
        elif model_type == 'Voigt':
            # Gauss-sigma guess, Lorentz-gamma guess
            guesses.extend([amp, cen, wid_guess * 0.7, wid_guess * 0.3])
        elif model_type == 'Pseudo-Voigt':
            # Width guess, Lorentzian fraction guess (default to 0.5)
            guesses.extend([amp, cen, wid_guess, 0.5])
        elif model_type == 'Asymmetric Gaussian':
            # Width guess, skew guess (default to 0.0 symmetric)
            guesses.extend([amp, cen, wid_guess, 0.0])
            
    return guesses


# =============================================================================
# 3. FIT BOUNDS CONSTRAINTS
# =============================================================================
def get_bounds(model_type, x_data, y_data, num_peaks):
    """
    Constructs boundaries for all variables to ensure physical consistency.
    
    This prevents:
      - Negative peak heights or widths.
      - Peaks migrating completely outside the measured spectrum.
      - Shape parameters escaping their logical domains (e.g. Voigt fraction > 1.0).
      
    Returns:
        tuple: (lower_bounds_list, upper_bounds_list)
    """
    max_y = np.max(y_data)
    min_x = np.min(x_data)
    max_x = np.max(x_data)
    x_range = max_x - min_x
    
    if model_type in ['Gaussian', 'Lorentzian']:
        # [amp, cen, wid]
        lower_bound = [0.0001, min_x - 50.0, 0.1]
        upper_bound = [max_y * 3.0, max_x + 50.0, x_range * 0.5]
    elif model_type == 'Voigt':
        # [amp, cen, sigma, gamma]
        lower_bound = [0.0001, min_x - 50.0, 0.1, 0.0]
        upper_bound = [max_y * 3.0, max_x + 50.0, x_range * 0.5, x_range * 0.5]
    elif model_type == 'Pseudo-Voigt':
        # [amp, cen, sigma, frac] (frac must be between 0.0 and 1.0)
        lower_bound = [0.0001, min_x - 50.0, 0.1, 0.0]
        upper_bound = [max_y * 3.0, max_x + 50.0, x_range * 0.5, 1.0]
    elif model_type == 'Asymmetric Gaussian':
        # [amp, cen, wid, skew] (skew locked between -1.0 and 1.0)
        lower_bound = [0.0001, min_x - 50.0, 0.1, -1.0]
        upper_bound = [max_y * 3.0, max_x + 50.0, x_range * 0.5, 1.0]
    else:
        lower_bound = [-np.inf] * get_num_params(model_type)
        upper_bound = [np.inf] * get_num_params(model_type)
        
    return lower_bound * num_peaks, upper_bound * num_peaks


# =============================================================================
# 4. LEAST-SQUARES CRITICAL OPTIMIZER
# =============================================================================
def fit_spectrum(x_data, y_data, model_type, num_peaks=2, custom_guesses=None):
    """
    Fits the spectroscopic peaks using SciPy's bounded non-linear curve optimizer.
    
    This function resolves the parameters of overlapping peaks by minimizing 
    the sum of squared residuals:
        min sum_{i} (y_data_i - combined_model(x_data_i, *popt))^2
        
    Because bounds are defined, SciPy utilizes the Trust-Region Reflective (trf) 
    algorithm, which is highly robust against local minima.
    
    Args:
        x_data (np.array): Corrected wavelength vector.
        y_data (np.array): Corrected absorbance vector.
        model_type (str): Fitting model type ('Gaussian', 'Lorentzian', etc.).
        num_peaks (int): Number of curves/peaks to deconvolute.
        custom_guesses (list of float, optional): Custom initial center wavelengths.
        
    Returns:
        tuple: (popt, pcov, y_fitted, list_of_individual_peak_curves)
            - popt: Optimized parameters list.
            - pcov: Covariance matrix (uncertainty estimator).
            - y_fitted: Evaluated overall model fit.
            - individual_peaks: Evaluated single curves.
    """
    # 1. Gather initial guesses and bounds
    initial_guesses = get_initial_guesses(model_type, x_data, y_data, num_peaks, custom_guesses)
    lower_bounds, upper_bounds = get_bounds(model_type, x_data, y_data, num_peaks)
    
    # 2. Bind the objective wrapper for curve_fit
    objective = lambda x, *params: curve_fit_wrapper(x, model_type, *params)
    
    # 3. Perform optimization
    popt, pcov = curve_fit(
        objective, 
        x_data, 
        y_data, 
        p0=initial_guesses, 
        bounds=(lower_bounds, upper_bounds), 
        maxfev=10000  # Give the solver up to 10,000 steps to converge
    )
    
    # 4. Generate final curves from optimized parameters
    y_fitted, individual_peaks = combined_model_systematic(x_data, model_type, *popt)
    
    return popt, pcov, y_fitted, individual_peaks
