"""
Speckle reduction filters for SAR imagery.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Apply Lee adaptive filter for speckle reduction.
    
    Parameters:
        img: Input amplitude image
        window_size: Local window size (must be odd)
        
    Returns:
        filtered: Despeckled amplitude image
    """
    intensity = img**2
    pad = window_size // 2
    padded = np.pad(intensity, pad, mode='reflect')
    filtered_int = np.zeros_like(intensity)
    
    for i in range(intensity.shape[0]):
        for j in range(intensity.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            local_mean = window.mean()
            local_var = window.var()
            
            # Adaptive weighting
            noise_var = local_mean**2
            weight = max(0, (local_var - noise_var) / (local_var + 1e-12))
            
            filtered_int[i, j] = local_mean + (1 - weight) * (intensity[i, j] - local_mean)
    
    logger.info(f"Lee filter applied (window={window_size}x{window_size})")
    return np.sqrt(filtered_int)


def frost_filter(img: np.ndarray, window_size: int = 7, 
                damping: float = 2.0) -> np.ndarray:
    """
    Apply Frost adaptive filter (edge-preserving alternative to Lee).
    
    Parameters:
        img: Input amplitude image
        window_size: Local window size
        damping: Damping factor (higher = more smoothing)
        
    Returns:
        filtered: Despeckled image
    """
    # Implementation placeholder
    logger.warning("Frost filter not yet implemented, using Lee filter")
    return lee_filter(img, window_size)
