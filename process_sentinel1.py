import rasterio
import numpy as np
from sar_processor import SARProcessor
import matplotlib.pyplot as plt

def load_sentinel1_data(filepath):
    """
    Load Sentinel-1 GRD data
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read first band
        metadata = src.meta
    
    return data, metadata

def preprocess_sentinel1(data):
    """
    Basic preprocessing: calibration and filtering
    """
    # Convert to linear scale (if in dB)
    linear_data = 10**(data/10.0)
    
    # Speckle filtering (Lee filter)
    from scipy.ndimage import uniform_filter, variance
    
    img_mean = uniform_filter(linear_data, size=5)
    img_sqr_mean = uniform_filter(linear_data**2, size=5)
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = variance(linear_data)
    
    img_weights = img_variance / (img_variance + overall_variance)
    filtered = img_mean + img_weights * (linear_data - img_mean)
    
    return filtered

# Load and process
data, meta = load_sentinel1_data('path/to/sentinel1.tif')
filtered_data = preprocess_sentinel1(data)

# Display
plt.figure(figsize=(12, 10))
plt.subplot(121)
plt.imshow(data, cmap='gray')
plt.title('Original SAR Image')
plt.colorbar()

plt.subplot(122)
plt.imshow(filtered_data, cmap='gray')
plt.title('Speckle Filtered SAR')
plt.colorbar()

plt.tight_layout()
plt.savefig('sentinel1_processing.png', dpi=300)
plt.show()
