"""
Range compression module using matched filtering.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def range_compression(raw_data: np.ndarray, chirp: np.ndarray, 
                     config: Dict) -> np.ndarray:
    """
    Perform matched filtering for range compression.
    
    Parameters:
        raw_data: Raw SAR data (range x azimuth)
        chirp: Reference chirp signal
        config: Configuration dictionary
        
    Returns:
        rc_data: Range-compressed data
    """
    N_rg = config['scene']['n_range']
    N_az = config['scene']['n_azimuth']
    
    # Matched filter in frequency domain
    ref_fft = np.conj(np.fft.fft(chirp, N_rg))
    rc_data = np.zeros_like(raw_data, dtype=complex)
    
    for az in range(N_az):
        signal_fft = np.fft.fft(raw_data[:, az])
        compressed_fft = signal_fft * ref_fft
        rc_data[:, az] = np.fft.ifft(compressed_fft)
    
    logger.info("Range compression complete")
    return rc_data
