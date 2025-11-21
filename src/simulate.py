"""
SAR data simulation module.
Creates synthetic radar returns with targets, clutter, and noise.
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def simulate_raw_sar_data(chirp: np.ndarray, config: Dict) -> np.ndarray:
    """
    Simulate raw SAR data with realistic effects.
    
    Parameters:
        chirp: Reference chirp signal
        config: Configuration dictionary
        
    Returns:
        raw_data: Simulated 2D complex SAR data (range x azimuth)
    """
    np.random.seed(config['simulation']['random_seed'])
    
    N_rg = config['scene']['n_range']
    N_az = config['scene']['n_azimuth']
    noise_sigma = config['simulation']['thermal_noise_sigma']
    
    # Thermal noise baseline
    raw_data = _generate_thermal_noise(N_rg, N_az, noise_sigma)
    
    # Point targets
    raw_data = _add_point_targets(raw_data, chirp)
    
    # Extended clutter
    raw_data = _add_clutter(raw_data)
    
    # Multiplicative speckle
    speckle = _generate_correlated_speckle(
        raw_data.shape, 
        config['simulation']['speckle_corr_kernel']
    )
    raw_data *= speckle
    
    # Platform motion errors
    raw_data = _apply_motion_errors(raw_data, config)
    
    logger.info(f"Raw SAR data simulated: {N_rg}×{N_az}")
    return raw_data


def _generate_thermal_noise(n_rg: int, n_az: int, sigma: float) -> np.ndarray:
    """Generate complex Gaussian noise."""
    return (np.random.randn(n_rg, n_az) + 
            1j * np.random.randn(n_rg, n_az)) * sigma


def _add_point_targets(data: np.ndarray, chirp: np.ndarray) -> np.ndarray:
    """Add point targets to scene."""
    chirp_len = len(chirp)
    data[40:40+chirp_len, 65] += chirp * 5.0  # Strong target
    data[80:80+chirp_len, 25] += chirp * 2.0  # Weak target
    return data


def _add_clutter(data: np.ndarray) -> np.ndarray:
    """Add extended clutter patches."""
    def add_patch(top, left, h, w, rcs):
        patch = (np.random.randn(h, w) + 1j * np.random.randn(h, w)) * rcs
        data[top:top+h, left:left+w] += patch
    
    add_patch(10, 10, 20, 30, 0.4)   # Vegetation
    add_patch(60, 90, 30, 20, 0.8)   # Urban
    return data


def _generate_correlated_speckle(shape: Tuple, kernel_size: int) -> np.ndarray:
    """Generate spatially correlated speckle."""
    L = 1.0
    speckle = np.random.gamma(L, 1.0/L, size=shape)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    speckle = fftconvolve(speckle, kernel, mode='same')
    return speckle / speckle.mean()


def _apply_motion_errors(data: np.ndarray, config: Dict) -> np.ndarray:
    """Apply platform motion phase errors."""
    N_az = config['scene']['n_azimuth']
    jitter = config['simulation']['motion_phase_jitter']
    
    phase_ramp = np.exp(1j * np.linspace(0, 0.6, N_az))
    phase_jitter = np.exp(1j * np.random.randn(N_az) * jitter)
    phase_error = phase_ramp * phase_jitter
    
    return data * phase_error[np.newaxis, :]
