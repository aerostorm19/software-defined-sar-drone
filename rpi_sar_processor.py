import numpy as np
import cv2
import time
import psutil
from scipy import signal

class RPiSARProcessor:
    """
    Lightweight SAR image processor optimized for Raspberry Pi
    Focus: Post-processing and enhancement rather than raw signal processing
    """
    
    def __init__(self):
        self.process_start_time = None
        
    def load_sar_image(self, filepath):
        """Load pre-processed SAR image"""
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    def adaptive_despeckling(self, sar_img, window_size=5):
        """
        Adaptive Lee filter for speckle reduction
        Optimized for Raspberry Pi
        """
        self.process_start_time = time.time()
        
        # Convert to float
        img_float = sar_img.astype(np.float32)
        
        # Calculate local statistics
        mean = cv2.boxFilter(img_float, -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter(img_float**2, -1, (window_size, window_size))
        variance = sqr_mean - mean**2
        
        # Global variance
        global_variance = np.var(img_float)
        
        # Adaptive weights
        weights = variance / (variance + global_variance + 1e-10)
        
        # Filtered output
        filtered = mean + weights * (img_float - mean)
        
        elapsed = time.time() - self.process_start_time
        print(f"Despeckling completed in {elapsed:.2f}s")
        
        return filtered.astype(np.uint8)
    
    def edge_enhancement(self, sar_img):
        """
        Enhance edges in SAR imagery
        """
        # Sobel edge detection
        sobelx = cv2.Sobel(sar_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sar_img, cv2.CV_64F, 0, 1, ksize=3)
        
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        
        # Combine with original
        enhanced = cv2.addWeighted(sar_img, 0.7, edges, 0.3, 0)
        
        return enhanced
    
    def feature_extraction(self, sar_img):
        """
        Extract features for target detection
        """
        # SIFT features (lightweight alternative: ORB)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(sar_img, None)
        
        # Draw keypoints
        img_keypoints = cv2.drawKeypoints(sar_img, keypoints, None, 
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return keypoints, descriptors, img_keypoints
    
    def monitor_performance(self):
        """Monitor CPU and memory usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory.percent}%")
        print(f"Available Memory: {memory.available / 1024**2:.1f} MB")
    
    def real_time_simulation(self, image_folder, fps=10):
        """
        Simulate real-time SAR processing from image sequence
        """
        import glob
        import os
        
        images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        
        for img_path in images:
            frame_start = time.time()
            
            # Load
            img = self.load_sar_image(img_path)
            
            # Process
            despeckled = self.adaptive_despeckling(img)
            enhanced = self.edge_enhancement(despeckled)
            
            # Display (if running with display)
            cv2.imshow('SAR Processing', enhanced)
            
            # Maintain frame rate
            processing_time = time.time() - frame_start
            wait_time = max(1, int((1/fps - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
            print(f"Frame processed in {processing_time:.3f}s")
        
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    processor = RPiSARProcessor()
    
    # Load sample SAR image
    sar_img = processor.load_sar_image('sar_sample.png')
    
    # Process
    despeckled = processor.adaptive_despeckling(sar_img)
    enhanced = processor.edge_enhancement(despeckled)
    keypoints, desc, kp_img = processor.feature_extraction(enhanced)
    
    # Save results
    cv2.imwrite('despeckled.png', despeckled)
    cv2.imwrite('enhanced.png', enhanced)
    cv2.imwrite('features.png', kp_img)
    
    # Performance monitoring
    processor.monitor_performance()
    
    print(f"Detected {len(keypoints)} features")
