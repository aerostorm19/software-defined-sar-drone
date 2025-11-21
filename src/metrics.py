"""
SAR image quality metrics computation.
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_pslr_islr(slice_db: np.ndarray) -> Tuple[float, float]:
    """
    Compute PSLR and ISLR from impulse response.
    
    Parameters:
        slice_db: 1D impulse response in dB
        
    Returns:
        pslr: Peak Sidelobe Ratio (dB)
        islr: Integrated Sidelobe Ratio (dB)
    """
    peak = np.max(slice_db)
    peak_idx = np.argmax(slice_db)
    
    # PSLR
    sidelobes = np.copy(slice_db)
    sidelobes[max(0, peak_idx-1):min(len(slice_db), peak_idx+2)] = -999
    pslr = peak - np.max(sidelobes)
    
    # ISLR
    linear = 10**(slice_db / 10.0)
    mainlobe_idx = np.arange(max(0, peak_idx-1), min(len(slice_db), peak_idx+2))
    mainlobe_energy = np.sum(linear[mainlobe_idx])
    sidelobe_energy = np.sum(linear) - mainlobe_energy
    islr = 10 * np.log10(mainlobe_energy / (sidelobe_energy + 1e-12))
    
    return pslr, islr


def compute_enl(img: np.ndarray, patch_coords: Tuple[int, int, int, int]) -> float:
    """
    Compute Equivalent Number of Looks (ENL).
    
    Parameters:
        img: Amplitude image
        patch_coords: (top, left, height, width)
        
    Returns:
        enl: Equivalent number of looks
    """
    top, left, h, w = patch_coords
    patch = img[top:top+h, left:left+w]
    mean_val = patch.mean()
    var_val = patch.var()
    return (mean_val**2) / (var_val + 1e-12)


def compute_all_metrics(focused_img: np.ndarray) -> Dict:
    """
    Compute all quality metrics for focused SAR image.
    
    Parameters:
        focused_img: Focused SAR amplitude image
        
    Returns:
        metrics: Dictionary of all computed metrics
    """
    # Find brightest target
    rg_idx, az_idx = np.unravel_index(np.argmax(focused_img), focused_img.shape)
    
    # Extract impulse responses
    range_slice = 20 * np.log10(focused_img[:, az_idx] + 1e-9)
    azimuth_slice = 20 * np.log10(focused_img[rg_idx, :] + 1e-9)
    
    # Compute metrics
    pslr_r, islr_r = compute_pslr_islr(range_slice)
    pslr_a, islr_a = compute_pslr_islr(azimuth_slice)
    enl = compute_enl(focused_img, (12, 12, 20, 20))
    
    metrics = {
        'pslr_range': pslr_r,
        'islr_range': islr_r,
        'pslr_azimuth': pslr_a,
        'islr_azimuth': islr_a,
        'enl': enl,
        'target_location': (rg_idx, az_idx)
    }
    
    logger.info(f"Metrics computed: PSLR_r={pslr_r:.2f}dB, ENL={enl:.2f}")
    return metrics
