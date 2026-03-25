"""
Image Processing Module
Kuchambua picha za ECG
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class ECGImageProcessor:
    """Class ya kuchambua picha za ECG"""
    
    def __init__(self):
        self.image = None
        self.processed_image = None
        
    def load_image(self, image_path):
        """
        Load ECG image from file (PNG, JPG, etc.)
        """
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError("Could not load image")
            
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            print(f"✅ Image loaded: {image_path}")
            print(f"   Dimensions: {self.image.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
            
    def extract_ecg_trace(self):
        """
        Extract ECG trace from image (basic implementation)
        """
        if self.image is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to get the waveform
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find the waveform (assuming it's the darkest part)
        waveform_y = np.where(binary > 0)[0]
        
        if len(waveform_y) == 0:
            return None
            
        # Extract signal
        signal = []
        for x in range(self.image.shape[1]):
            y_values = np.where(binary[:, x] > 0)[0]
            if len(y_values) > 0:
                # Average of detected points
                y_mean = np.mean(y_values)
                signal.append(y_mean)
            else:
                signal.append(signal[-1] if signal else 0)
                
        # Normalize signal
        signal = np.array(signal)
        if len(signal) > 0:
            signal = (signal - np.mean(signal)) / np.std(signal)
            
        return signal
        
    def display_image(self):
        """
        Display loaded image
        """
        if self.image is not None:
            plt.figure(figsize=(12, 6))
            plt.imshow(self.image)
            plt.title("ECG Image")
            plt.axis('off')
            plt.show()
            
    def extract_metadata(self):
        """
        Extract metadata from image (size, format, etc.)
        """
        if self.image is None:
            return None
            
        return {
            'width': self.image.shape[1],
            'height': self.image.shape[0],
            'channels': self.image.shape[2] if len(self.image.shape) > 2 else 1,
            'total_pixels': self.image.shape[0] * self.image.shape[1]
        }
