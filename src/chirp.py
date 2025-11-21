"""
Chirp signal generation module.
Generates Linear Frequency Modulated (LFM) radar pulses.
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def generate_chirp(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Linear Frequency Modulated (LFM) chirp signal.
    
    Parameters:
        config: Configuration dictionary containing radar parameters
        
    Returns:
        t: Time vector (seconds)
        chirp: Complex chirp signal
    """
    B = config['radar']['bandwidth']
    T = config['radar']['pulse_duration']
    N_rg = config['scene']['n_range']
    
    fs = 2 * B  # Nyquist sampling rate
    K = B / T   # Chirp rate (Hz/s)
    
    t = np.arange(N_rg) / fs
    chirp_len = N_rg // 4
    chirp = np.exp(1j * np.pi * K * t[:chirp_len]**2)
    
    logger.info(f"Chirp generated: B={B/1e6:.1f} MHz, T={T*1e6:.1f} μs")
    return t[:chirp_len], chirp


def apply_window(chirp: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Apply window function to chirp for sidelobe reduction.
    
    Parameters:
        chirp: Input chirp signal
        window_type: Type of window ('hamming', 'hann', 'blackman')
        
    Returns:
        windowed_chirp: Windowed chirp signal
    """
    from scipy.signal import get_window
    window = get_window(window_type, len(chirp))
    return chirp * window
