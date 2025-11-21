"""
Azimuth compression module using Doppler processing.
"""

import numpy as np
from scipy.signal import windows
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def azimuth_compression(rc_data: np.ndarray, config: Dict) -> np.ndarray:
    """
    Perform azimuth compression via Doppler matched filtering.
    
    Parameters:
        rc_data: Range-compressed data
        config: Configuration dictionary
        
    Returns:
        focused_data: Fully focused SAR image
    """
    N_az = config['scene']['n_azimuth']
    N_rg = config['scene']['n_range']
    az_bw = config['processing']['azimuth_bandwidth']
    
    # Azimuth reference function
    az_time = np.arange(N_az) - N_az/2
    az_ref_td = np.sinc(az_bw * az_time / N_az) * windows.hann(N_az)
    az_ref_fd = np.conj(np.fft.fft(az_ref_td, N_az))
    
    # Matched filtering
    focused_data = np.zeros_like(rc_data, dtype=complex)
    for rg in range(N_rg):
        signal_fft = np.fft.fft(rc_data[rg, :])
        focused_fft = signal_fft * az_ref_fd
        focused_data[rg, :] = np.fft.ifft(focused_fft)
    
    # Add residual noise
    focused_data += (np.random.randn(*focused_data.shape) + 
                     1j * np.random.randn(*focused_data.shape)) * 0.01
    
    logger.info("Azimuth compression complete")
    return focused_data
