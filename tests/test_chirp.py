"""
Unit tests for chirp generation module.
"""

import pytest
import numpy as np
import sys
sys.path.append('../src')

from chirp import generate_chirp
from config import CONFIG


def test_chirp_generation():
    """Test chirp signal generation."""
    t, chirp = generate_chirp(CONFIG)
    
    # Check output types
    assert isinstance(t, np.ndarray)
    assert isinstance(chirp, np.ndarray)
    assert chirp.dtype == complex
    
    # Check lengths
    assert len(t) == len(chirp)
    assert len(chirp) == CONFIG['scene']['n_range'] // 4


def test_chirp_amplitude():
    """Test chirp has unit amplitude."""
    _, chirp = generate_chirp(CONFIG)
    assert np.allclose(np.abs(chirp), 1.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
