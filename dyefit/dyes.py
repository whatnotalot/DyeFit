"""
===============================================================================
DyeConfig & Presets Registry (JSON Powered)
===============================================================================
This module defines the DyeConfig class and manages loading/saving presets 
from the external `dyes.json` database.

These configurations dictate:
  1. The region of interest (cropping bounds) for analysis.
  2. The baseline mask (defining which wavelengths represent the flat baseline 
     free from dye absorption).
  3. The expected number of peaks and their initial center guesses.
"""

import numpy as np
import os
import json

# Locate dyes.json in the project root directory (one level up from this package folder)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dyes.json"))

class DyeConfig:
    """
    Configuration template for baseline correction and curve fitting of a specific dye.
    
    Attributes:
        name (str): Human-readable name of the dye.
        min_wavelength (float): The lower boundary (nm) for cropping the spectrum.
        max_wavelength (float): The upper boundary (nm) for cropping the spectrum.
        num_peaks (int): The default number of peaks to search for/fit.
        baseline_left (float, optional): Wavelength below which is pure baseline.
        baseline_right (float, optional): Wavelength above which is pure baseline.
        peak_guesses (list of float, optional): Initial guesses for the peak wavelengths (nm).
        baseline_mask_fn (callable): A function (lambda w: boolean array) that returns True 
            for wavelengths that contain no dye absorption (i.e., pure solvent background).
    """
    def __init__(self, name, min_wavelength, max_wavelength, num_peaks, baseline_left=None, baseline_right=None, peak_guesses=None, baseline_mask_fn=None):
        self.name = name
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.num_peaks = num_peaks
        self.baseline_left = baseline_left
        self.baseline_right = baseline_right
        self.peak_guesses = peak_guesses  
        
        # If a direct lambda function is provided, use it.
        # Otherwise, dynamically build it from left and right baseline boundary limits.
        if baseline_mask_fn is not None:
            self.baseline_mask_fn = baseline_mask_fn
        elif baseline_left is not None and baseline_right is not None:
            self.baseline_mask_fn = lambda w: (w < baseline_left) | (w > baseline_right)
        else:
            # Default to selecting all wavelengths if no baseline parameters are supplied
            self.baseline_mask_fn = lambda w: np.ones_like(w, dtype=bool)

# =============================================================================
# JSON DATABASE LOADING & SAVING UTILITIES
# =============================================================================
def load_dyes_db():
    """
    Loads the dye configurations from the dyes.json file.
    If the file does not exist, it automatically regenerates a default one.
    """
    if not os.path.exists(DB_PATH):
        # Default presets database schema
        default_db = {
            "Methylene Blue": {
                "name": "Methylene Blue",
                "min_wavelength": 500.0,
                "max_wavelength": 750.0,
                "baseline_left": 540.0,
                "baseline_right": 725.0,
                "num_peaks": 2,
                "peak_guesses": [612.0, 664.0]
            },
            "Methyl Orange": {
                "name": "Methyl Orange",
                "min_wavelength": 320.0,
                "max_wavelength": 600.0,
                "baseline_left": 350.0,
                "baseline_right": 550.0,
                "num_peaks": 2,
                "peak_guesses": [380.0, 464.0]
            },
            "Rhodamine B": {
                "name": "Rhodamine B",
                "min_wavelength": 450.0,
                "max_wavelength": 650.0,
                "baseline_left": 480.0,
                "baseline_right": 620.0,
                "num_peaks": 2,
                "peak_guesses": [515.0, 554.0]
            },
            "Fluorescein": {
                "name": "Fluorescein",
                "min_wavelength": 400.0,
                "max_wavelength": 600.0,
                "baseline_left": 430.0,
                "baseline_right": 560.0,
                "num_peaks": 2,
                "peak_guesses": [455.0, 490.0]
            },
            "Crystal Violet": {
                "name": "Crystal Violet",
                "min_wavelength": 450.0,
                "max_wavelength": 700.0,
                "baseline_left": 490.0,
                "baseline_right": 660.0,
                "num_peaks": 2,
                "peak_guesses": [540.0, 590.0]
            },
            "Congo Red": {
                "name": "Congo Red",
                "min_wavelength": 350.0,
                "max_wavelength": 650.0,
                "baseline_left": 380.0,
                "baseline_right": 590.0,
                "num_peaks": 2,
                "peak_guesses": [450.0, 497.0]
            }
        }
        try:
            with open(DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_db, f, indent=2)
        except Exception:
            pass
            
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = {}
        
    # Convert JSON profiles to runtime DyeConfig objects
    registry = {}
    for name, item in data.items():
        registry[name] = DyeConfig(
            name=item.get("name", name),
            min_wavelength=float(item.get("min_wavelength", 300.0)),
            max_wavelength=float(item.get("max_wavelength", 800.0)),
            num_peaks=int(item.get("num_peaks", 1)),
            baseline_left=float(item.get("baseline_left")) if item.get("baseline_left") is not None else None,
            baseline_right=float(item.get("baseline_right")) if item.get("baseline_right") is not None else None,
            peak_guesses=item.get("peak_guesses", None)
        )
    return registry


def save_dye_to_db(name, data_dict):
    """
    Saves or updates a custom dye configuration in the dyes.json file.
    
    Args:
        name (str): Name of the dye.
        data_dict (dict): Configuration variables containing min_wavelength, 
                          max_wavelength, baseline_left, baseline_right, 
                          num_peaks, and peak_guesses.
    """
    data = {}
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}
            
    data[name] = {
        "name": name,
        "min_wavelength": float(data_dict.get("min_wavelength", 300.0)),
        "max_wavelength": float(data_dict.get("max_wavelength", 800.0)),
        "baseline_left": float(data_dict["baseline_left"]) if data_dict.get("baseline_left") is not None else None,
        "baseline_right": float(data_dict["baseline_right"]) if data_dict.get("baseline_right") is not None else None,
        "num_peaks": int(data_dict.get("num_peaks", 1)),
        "peak_guesses": data_dict.get("peak_guesses", None)
    }
    
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# INITIALIZATION
# =============================================================================
# Instantiate registry at startup
DYE_REGISTRY = load_dyes_db()

def reload_dye_registry():
    """
    Forces the package to reload dyes.json, updating the in-memory registry.
    """
    global DYE_REGISTRY
    DYE_REGISTRY = load_dyes_db()
    return DYE_REGISTRY


def get_dye_config(name, **kwargs):
    """
    Retrieves a dye configuration from the active registry, or returns a custom instance.
    """
    if name in DYE_REGISTRY:
        return DYE_REGISTRY[name]
    else:
        # User defined dynamic fallback (unsaved config)
        return DyeConfig(
            name=name,
            min_wavelength=kwargs.get("min_wavelength", 300.0),
            max_wavelength=kwargs.get("max_wavelength", 800.0),
            num_peaks=kwargs.get("num_peaks", 1),
            baseline_left=kwargs.get("baseline_left", None),
            baseline_right=kwargs.get("baseline_right", None),
            peak_guesses=kwargs.get("peak_guesses", None)
        )
