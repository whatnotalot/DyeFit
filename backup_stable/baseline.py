"""
===============================================================================
Baseline Subtraction Module
===============================================================================
This module contains algorithms to estimate and subtract the background signal
(baseline) from raw UV-Vis spectra. Three methods are provided:
  1. Polynomial Baseline (fits a polynomial to specified solvent zones)
  2. Linear Baseline (2-point start-to-end interpolation)
  3. Asymmetric Least Squares (ALS) Smoothing (automatic, mask-free estimation)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# =============================================================================
# 1. POLYNOMIAL BASELINE METHOD
# =============================================================================
def polynomial_baseline(wavelength, absorbance, mask, degree=2):
    """
    Fits a polynomial of specified degree to baseline-only regions (indicated by 
    a boolean mask) and evaluates it across the entire wavelength range.
    
    Mathematical concept:
        P(x) = c_n * x^n + c_{n-1} * x^{n-1} + ... + c_0
        We minimize the sum of squared residuals over the masked indices:
        min sum_{i in mask} (y_i - P(x_i))^2
        
    Args:
        wavelength (np.array): Wavelength vector (x-axis, nm).
        absorbance (np.array): Raw absorbance vector (y-axis).
        mask (np.array of bool): True for background regions, False for peak absorption regions.
        degree (int): Polynomial order (default is 2).
        
    Returns:
        np.array: Evaluated baseline curve across all wavelengths.
    """
    # Filter out only the wavelengths and absorbance points belonging to the baseline
    x_base = wavelength[mask]
    y_base = absorbance[mask]
    
    # Fallback safety checks
    if len(x_base) < degree + 1:
        # If not enough data points exist for the requested degree, lower the degree
        degree = max(1, len(x_base) - 1)
        
    if len(x_base) == 0:
        # If the mask is empty, return a flat baseline of zeros
        return np.zeros_like(wavelength)
        
    # Fit coefficients using least-squares polynomial fit
    coeffs = np.polyfit(x_base, y_base, degree)
    
    # Evaluate the fitted polynomial across all wavelengths
    baseline = np.polyval(coeffs, wavelength)
    return baseline


# =============================================================================
# 2. LINEAR BASELINE METHOD
# =============================================================================
def linear_baseline(wavelength, absorbance):
    """
    Performs a simple linear baseline subtraction using the start and end 
    points of the cropped spectrum. Useful for flat baselines with basic tilt.
    
    Mathematical concept:
        Draws a line y = m * (x - x1) + y1 connecting (wavelength[0], absorbance[0])
        and (wavelength[-1], absorbance[-1]).
        
    Args:
        wavelength (np.array): Wavelength vector.
        absorbance (np.array): Raw absorbance vector.
        
    Returns:
        np.array: Linear baseline curve.
    """
    # Identify boundary coordinate points
    x1, y1 = wavelength[0], absorbance[0]
    x2, y2 = wavelength[-1], absorbance[-1]
    
    # Calculate slope (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Interpolate baseline at all wavelengths
    baseline = m * (wavelength - x1) + y1
    return baseline


# =============================================================================
# 3. ASYMMETRIC LEAST SQUARES (ALS) METHOD
# =============================================================================
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (ALS) baseline estimation (Eilers & Boelens, 2005).
    
    This is an advanced, automated method that estimates a baseline WITHOUT 
    requiring a manual wavelength mask. It works by balancing two goals:
      1. Finding a baseline curve (z) that is smooth.
      2. Making the baseline curve run along the bottom of the data peaks.
      
    Mathematical concept:
        We minimize the objective function:
        S = sum_{i} w_i * (y_i - z_i)^2 + lam * sum_{i} (Delta^2 z_i)^2
        
        Where:
          - Delta^2 is the second-difference operator (penalizes curvature/roughness).
          - lam (Lambda) is the smoothness scaling factor (higher = flatter baseline).
          - w_i is an asymmetric weight:
                w_i = p     if y_i > z_i  (under a peak; give low weight so baseline ignores it)
                w_i = 1-p   if y_i <= z_i (baseline region; give high weight to stick to data)
                
    Args:
        y (np.array): 1D array of raw absorbance values.
        lam (float): Smoothness penalty (typical range: 10^2 to 10^9).
        p (float): Asymmetry parameter (typical range: 0.001 to 0.1). 
            Smaller p forces the baseline to sit lower beneath the absorption peaks.
        niter (int): Number of fitting iterations (usually 10 is enough for convergence).
        
    Returns:
        np.array: The estimated baseline (same length as y).
    """
    L = len(y)
    
    # D is the second-order finite difference matrix.
    # It calculates the second derivative (curvature) of the baseline curve.
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    
    # Initialize all weights to 1.0
    w = np.ones(L)
    
    # Iterative reweighted least squares solver
    for _ in range(niter):
        # Create a diagonal matrix of current weights
        W = sparse.spdiags(w, 0, L, L)
        
        # Formulate the system of equations (W + lam * D^T * D) * z = W * y
        # We use sparse matrices to solve this extremely fast (O(N) time).
        A = W + lam * D.T @ D
        z = spsolve(A, w * y)
        
        # Update weights: if the data is above the baseline (y > z), it's a peak, 
        # so set weight to p (very small). Otherwise, set weight to 1-p (large).
        w = p * (y > z) + (1 - p) * (y < z)
        
    return z
