import cv2
import numpy as np
import urllib.request

class ESP32SAROverlay:
    """
    Simulate SAR-Optical fusion by overlaying
    SAR data on ESP32-CAM video stream
    """
    
    def __init__(self, esp32_ip, sar_image_path):
        self.stream_url = f'http://{esp32_ip}/cam-hi.jpg'
        self.sar_overlay = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
        
    def get_esp32_frame(self):
        """Fetch frame from ESP32-CAM"""
        try:
            img_resp = urllib.request.urlopen(self.stream_url, timeout=5)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            return frame
        except:
            return None
    
    def create_fusion(self, optical_frame):
        """Create fused SAR-Optical visualization"""
        # Resize SAR to match optical
        h, w = optical_frame.shape[:2]
        sar_resized = cv2.resize(self.sar_overlay, (w, h))
        
        # Apply colormap to SAR
        sar_colored = cv2.applyColorMap(sar_resized, cv2.COLORMAP_HOT)
        
        # Blend
        fusion = cv2.addWeighted(optical_frame, 0.6, sar_colored, 0.4, 0)
        
        # Add split view
        split = np.hstack([optical_frame, sar_colored, fusion])
        
        return split
    
    def run(self):
        """Run fusion visualization"""
        print("Starting ESP32-CAM SAR fusion demo...")
        print("Press 'q' to quit")
        
        while True:
            frame = self.get_esp32_frame()
            
            if frame is not None:
                fusion_view = self.create_fusion(frame)
                
                cv2.putText(fusion_view, 'Optical', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(fusion_view, 'SAR', (frame.shape[1]+10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(fusion_view, 'Fused', (frame.shape[1]*2+10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('SAR-Optical Fusion Demo', fusion_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

# Usage
overlay = ESP32SAROverlay('192.168.1.100', 'sar_sample.png')
overlay.run()
