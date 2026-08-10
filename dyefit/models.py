"""
===============================================================================
Peak Shape Models
===============================================================================
This module defines the mathematical profiles used to model individual 
spectroscopic absorption peaks. All profiles are normalized so that the 
amplitude parameter (amp) represents the true physical peak height (absorbance).
"""

import numpy as np
from scipy.special import voigt_profile

# =============================================================================
# 1. GAUSSIAN PROFILE
# =============================================================================
def gaussian(x, amp, cen, wid):
    """
    Gaussian (Normal) Distribution Profile.
    
    In spectroscopy, Gaussian broadening occurs due to thermal motion (Doppler 
    broadening) or random instrumental fluctuations.
    
    Mathematical formula:
        f(x) = A * exp( -(x - x0)^2 / (2 * w^2) )
        
        Where:
          - A (amp): peak height (absorbance).
          - x0 (cen): peak center (wavelength, nm).
          - w (wid): standard deviation (sigma).
                     Full Width at Half Maximum (FWHM) = 2 * sqrt(2 * ln(2)) * w ≈ 2.3548 * w.
    """
    # Safeguard against division-by-zero or negative widths
    wid = max(wid, 1e-9)
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


# =============================================================================
# 2. LORENTZIAN PROFILE
# =============================================================================
def lorentzian(x, amp, cen, wid):
    """
    Lorentzian (Cauchy) Distribution Profile.
    
    In spectroscopy, Lorentzian broadening represents homogeneous lifetime 
    broadening (Heisenberg Uncertainty principle) or collisional broadening.
    
    Mathematical formula:
        f(x) = A * ( w^2 / ((x - x0)^2 + w^2) )
        
        Where:
          - A (amp): peak height (absorbance).
          - x0 (cen): peak center (wavelength, nm).
          - w (wid): Half-Width at Half-Maximum (HWHM).
                     Full Width at Half Maximum (FWHM) = 2 * w.
    """
    # Safeguard against division-by-zero or negative widths
    wid = max(wid, 1e-9)
    return amp * (wid ** 2 / ((x - cen) ** 2 + wid ** 2))


# =============================================================================
# 3. VOIGT PROFILE
# =============================================================================
def voigt(x, amp, cen, sigma, gamma):
    """
    Voigt Profile (Convolution of Gaussian and Lorentzian).
    
    Used when both Gaussian (Doppler/instrumental) and Lorentzian (lifetime/pressure)
    broadening are active. It provides the most physically accurate description 
    for gas-phase or solution-phase molecular spectra.
    
    Mathematical formula:
        f(x) = A * ( V(x - x0; sigma, gamma) / V(0; sigma, gamma) )
        
        Where:
          - V(x; sigma, gamma) is the convolution of Gaussian (sigma) and Lorentzian (gamma) curves.
          - We scale by V(0; sigma, gamma) so that the peak height equals the physical amplitude (amp).
          - We compute this using SciPy's complex Faddeeva function wrapper.
    """
    sigma = max(sigma, 1e-9)
    gamma = max(gamma, 1e-9)
    
    # Calculate the un-normalized Voigt profile (area is normalized to 1)
    normalized_voigt = voigt_profile(x - cen, sigma, gamma)
    
    # Calculate the peak height value at the center (x = cen)
    peak_height_normalized = voigt_profile(0, sigma, gamma)
    
    if peak_height_normalized == 0:
        return np.zeros_like(x, dtype=float)
        
    # Scale so that the peak maximum equals the amplitude parameter
    return amp * (normalized_voigt / peak_height_normalized)


# =============================================================================
# 4. PSEUDO-VOIGT PROFILE
# =============================================================================
def pseudo_voigt(x, amp, cen, sigma, frac):
    """
    Pseudo-Voigt Profile.
    
    A linear combination (approximation) of a Gaussian and a Lorentzian profile.
    It is computationally faster than the true Voigt profile while maintaining
    excellent fit quality for spectroscopic peaks.
    
    Mathematical formula:
        f(x) = A * ( (frac * L(x; cen, sigma) + (1 - frac) * G(x; cen, sigma)) / Max )
        
        Where:
          - G(x) is a Gaussian with height 1.0.
          - L(x) is a Lorentzian with height 1.0.
          - frac (Lorentzian Fraction): Between 0.0 (pure Gaussian) and 1.0 (pure Lorentzian).
    """
    sigma = max(sigma, 1e-9)
    frac = np.clip(frac, 0.0, 1.0)
    
    # Linear combination of height-normalized components
    normalized_pv = frac * lorentzian(x, 1, cen, sigma) + (1 - frac) * gaussian(x, 1, cen, sigma)
    peak_height_normalized = frac * lorentzian(cen, 1, cen, sigma) + (1 - frac) * gaussian(cen, 1, cen, sigma)
    
    if peak_height_normalized == 0:
        return np.zeros_like(x, dtype=float)
        
    return amp * (normalized_pv / peak_height_normalized)


# =============================================================================
# 5. ASYMMETRIC GAUSSIAN PROFILE
# =============================================================================
def asym_gaussian(x, amp, cen, wid, skew):
    """
    Asymmetric Gaussian Profile.
    
    Commonly used to fit skewed peaks caused by solvent interaction, 
    instrumental lag, or unresolved vibronic shoulder bands.
    
    Mathematical formula:
        f(x) = A * exp( -(x - x0)^2 / [ 2 * w^2 * (1 + s * (x - x0))^2 ] )
        
        Where:
          - A (amp): peak height.
          - x0 (cen): peak center.
          - w (wid): base width.
          - s (skew): skewness parameter (s > 0 skews to the right, s < 0 to the left).
    """
    wid = max(wid, 1e-9)
    x_arr = np.asarray(x, dtype=float)
    
    # The width varies linearly with the distance from the center, controlled by the skew factor
    denominator = (2 * wid**2 * (1 + skew * (x_arr - cen))**2)
    
    # Prevent division by zero if the denominator falls to zero
    if isinstance(denominator, np.ndarray):
        denominator[denominator == 0] = 1e-9
    elif denominator == 0:
        denominator = 1e-9
        
    return amp * np.exp(-(x_arr - cen)**2 / denominator)
