"""
ECG Signal Preprocessing Module
Kusafisha na kuandaa data ya ECG kwa ajili ya AI
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt

class ECGPreprocessor:
    """Class ya kusafisha ishara za ECG"""
    
    def __init__(self, sampling_rate=360):
        """
        Initialize preprocessor
        sampling_rate: frequency ya ECG signal (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def remove_noise(self, ecg_signal):
        """
        Remove noise kutoka ECG signal
        Filter: bandpass filter (0.5 - 50 Hz)
        """
        # Design bandpass filter
        nyquist = 0.5 * self.sampling_rate
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def detect_r_peaks(self, ecg_signal):
        """
        Detect R peaks katika ECG signal
        Hii ni muhimu kwa kuhesabu heart rate
        """
        # Simple peak detection using scipy
        from scipy.signal import find_peaks
        
        # Find peaks with minimum height and distance
        peaks, properties = find_peaks(
            ecg_signal,
            height=np.mean(ecg_signal) + 0.5 * np.std(ecg_signal),
            distance=self.sampling_rate * 0.4  # Minimum 400ms between peaks
        )
        
        return peaks
    
    def calculate_heart_rate(self, r_peaks):
        """
        Calculate heart rate kutoka R peaks
        Returns: heart rate in BPM (beats per minute)
        """
        if len(r_peaks) < 2:
            return 0
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        
        # Heart rate = 60 / RR interval
        heart_rates = 60 / rr_intervals
        
        return np.mean(heart_rates)
    
    def segment_ecg(self, ecg_signal, window_size=5):
        """
        Segment ECG signal into windows
        window_size: duration in seconds
        """
        window_samples = window_size * self.sampling_rate
        segments = []
        
        for i in range(0, len(ecg_signal), window_samples):
            segment = ecg_signal[i:i + window_samples]
            if len(segment) == window_samples:
                segments.append(segment)
                
        return np.array(segments)
