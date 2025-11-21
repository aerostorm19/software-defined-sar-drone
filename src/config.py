"""
Configuration parameters for SAR processing pipeline.
Modify these values to tune algorithm performance.
"""

CONFIG = {
    'radar': {
        'bandwidth': 100e6,          # Hz - Chirp bandwidth
        'pulse_duration': 3e-6,      # seconds - Chirp duration
        'speed_of_light': 3e8,       # m/s
        'carrier_frequency': 5.4e9,  # Hz - X-band radar
    },
    
    'scene': {
        'n_azimuth': 128,            # Number of azimuth pulses
        'n_range': 128,              # Number of range bins
        'pixel_spacing_range': 1.5,  # meters
        'pixel_spacing_azimuth': 1.5,# meters
    },
    
    'simulation': {
        'thermal_noise_sigma': 0.15,
        'speckle_corr_kernel': 9,
        'motion_phase_jitter': 0.05,
        'random_seed': 42,           # For reproducibility
    },
    
    'processing': {
        'lee_window_size': 7,
        'azimuth_bandwidth': 0.5,
        'rcmc_enabled': False,       # Range cell migration correction
    },
    
    'visualization': {
        'dpi': 150,
        'cmap_amplitude': 'gray',
        'cmap_phase': 'hsv',
        'figsize_default': (10, 6),
    },
    
    'output': {
        'save_intermediates': True,
        'output_format': 'png',
        'results_dir': '../results/',
    }
}
