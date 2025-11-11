import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt

class SARProcessor:
    def __init__(self, raw_data, params):
        """
        Initialize SAR processor
        
        Parameters:
        -----------
        raw_data : complex numpy array
            Raw SAR echo data (range x azimuth)
        params : dict
            System parameters (wavelength, PRF, bandwidth, etc.)
        """
        self.raw_data = raw_data
        self.params = params
        
    def range_compression(self):
        """
        Step 1: Range compression using matched filtering
        """
        # Generate reference chirp signal
        bandwidth = self.params['bandwidth']
        pulse_duration = self.params['pulse_duration']
        sample_rate = self.params['sample_rate']
        
        # Chirp rate
        K = bandwidth / pulse_duration
        
        # Time vector
        t = np.arange(0, pulse_duration, 1/sample_rate)
        
        # Reference chirp (linear FM signal)
        ref_chirp = np.exp(1j * np.pi * K * t**2)
        
        # Matched filter in frequency domain
        compressed = np.zeros_like(self.raw_data)
        ref_fft = np.conj(fft(ref_chirp))
        
        for i in range(self.raw_data.shape[1]):  # For each azimuth line
            range_fft = fft(self.raw_data[:, i])
            compressed[:, i] = ifft(range_fft * ref_fft)
            
        return compressed
    
    def range_cell_migration_correction(self, compressed_data):
        """
        Step 2: Correct range cell migration (RCMC)
        """
        # Simplified RCMC using interpolation
        # In production, use Stolt interpolation or similar
        
        velocity = self.params['platform_velocity']
        wavelength = self.params['wavelength']
        range_resolution = self.params['range_resolution']
        
        corrected = np.zeros_like(compressed_data)
        
        # Apply range-dependent phase correction
        for i in range(compressed_data.shape[1]):
            # Calculate migration for this azimuth position
            migration = self._calculate_migration(i, velocity, wavelength)
            
            # Shift range bins accordingly (simplified)
            corrected[:, i] = np.roll(compressed_data[:, i], 
                                      int(migration/range_resolution))
        
        return corrected
    
    def azimuth_compression(self, rcmc_data):
        """
        Step 3: Azimuth compression (Doppler processing)
        """
        # Transform to range-Doppler domain
        rd_domain = fft(rcmc_data, axis=1)
        
        # Generate azimuth matched filter
        azimuth_ref = self._generate_azimuth_reference()
        
        # Apply matched filter
        compressed = rd_domain * np.conj(azimuth_ref)
        
        # Transform back to image domain
        sar_image = ifft(compressed, axis=1)
        
        return sar_image
    
    def process(self):
        """
        Complete SAR processing pipeline
        """
        print("Step 1: Range Compression...")
        range_compressed = self.range_compression()
        
        print("Step 2: Range Cell Migration Correction...")
        rcmc_data = self.range_cell_migration_correction(range_compressed)
        
        print("Step 3: Azimuth Compression...")
        sar_image = self.azimuth_compression(rcmc_data)
        
        return np.abs(sar_image)  # Return magnitude
    
    def _calculate_migration(self, azimuth_idx, velocity, wavelength):
        """Helper: Calculate range migration"""
        # Simplified calculation
        return (velocity**2 * azimuth_idx) / (wavelength * self.params['center_frequency'])
    
    def _generate_azimuth_reference(self):
        """Helper: Generate azimuth reference function"""
        # Simplified azimuth reference
        prf = self.params['PRF']
        wavelength = self.params['wavelength']
        
        doppler_freq = np.fft.fftfreq(self.raw_data.shape[1], 1/prf)
        ref = np.exp(-1j * np.pi * wavelength * doppler_freq**2)
        
        return ref[:, np.newaxis]

# Example usage with simulated data
def simulate_sar_raw_data():
    """
    Simulate raw SAR data for testing
    """
    # Create point targets
    range_bins = 512
    azimuth_bins = 512
    
    raw_data = np.random.randn(range_bins, azimuth_bins) + \
               1j * np.random.randn(range_bins, azimuth_bins)
    
    # Add strong point targets
    raw_data[256, 256] += 100 + 100j
    raw_data[200, 300] += 80 + 80j
    
    return raw_data

# System parameters (typical X-band SAR)
params = {
    'wavelength': 0.03,  # 3 cm (X-band)
    'bandwidth': 100e6,  # 100 MHz
    'pulse_duration': 10e-6,  # 10 microseconds
    'sample_rate': 200e6,  # 200 MHz
    'PRF': 1000,  # Pulse Repetition Frequency
    'platform_velocity': 100,  # m/s
    'center_frequency': 10e9,  # 10 GHz
    'range_resolution': 1.5  # meters
}

# Process
raw_data = simulate_sar_raw_data()
processor = SARProcessor(raw_data, params)
sar_image = processor.process()

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(20*np.log10(sar_image + 1e-10), cmap='gray', aspect='auto')
plt.colorbar(label='Magnitude (dB)')
plt.title('SAR Image - Range-Doppler Algorithm')
plt.xlabel('Azimuth')
plt.ylabel('Range')
plt.savefig('sar_image_rda.png', dpi=300)
plt.show()
