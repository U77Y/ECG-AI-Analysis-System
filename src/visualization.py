"""
ECG Visualization Module
Kuplot na kuonyesha ECG waves
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image

class ECGVisualizer:
    """Class ya kuvisualize ECG signals"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_ecg_wave(self, ecg_signal, sampling_rate=360, title="ECG Signal"):
        """
        Plot ECG waveform
        """
        # Create time axis
        duration = len(ecg_signal) / sampling_rate
        time = np.linspace(0, duration, len(ecg_signal))
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Plot signal
        self.ax.plot(time, ecg_signal, color='blue', linewidth=1.5)
        self.ax.set_xlabel('Time (seconds)', fontsize=12)
        self.ax.set_ylabel('Amplitude (mV)', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim([0, min(duration, 10)])  # Show first 10 seconds
        
        # Add baseline
        self.ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        return self.fig
        
    def plot_with_r_peaks(self, ecg_signal, r_peaks, sampling_rate=360):
        """
        Plot ECG signal with R peaks marked
        """
        duration = len(ecg_signal) / sampling_rate
        time = np.linspace(0, duration, len(ecg_signal))
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Plot ECG signal
        self.ax.plot(time, ecg_signal, color='blue', linewidth=1.5, label='ECG Signal')
        
        # Mark R peaks
        peak_times = time[r_peaks]
        peak_values = ecg_signal[r_peaks]
        self.ax.scatter(peak_times, peak_values, color='red', s=50, 
                       marker='^', zorder=5, label=f'R Peaks ({len(r_peaks)})')
        
        self.ax.set_xlabel('Time (seconds)', fontsize=12)
        self.ax.set_ylabel('Amplitude (mV)', fontsize=12)
        self.ax.set_title('ECG Signal with Detected R Peaks', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        return self.fig
        
    def plot_12_leads(self, ecg_data_12leads, sampling_rate=360):
        """
        Plot 12-lead ECG (Lead I, II, III, aVR, aVL, aVF, V1-V6)
        """
        fig, axes = plt.subplots(6, 2, figsize=(15, 12))
        fig.suptitle('12-Lead ECG Analysis', fontsize=16, fontweight='bold')
        
        lead_names = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        duration = len(ecg_data_12leads[0]) / sampling_rate
        time = np.linspace(0, duration, len(ecg_data_12leads[0]))
        
        idx = 0
        for i in range(6):
            for j in range(2):
                if idx < len(lead_names):
                    axes[i, j].plot(time, ecg_data_12leads[idx], color='blue', linewidth=1)
                    axes[i, j].set_title(lead_names[idx], fontsize=10)
                    axes[i, j].set_xlim([0, min(duration, 5)])
                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].axhline(y=0, color='black', linewidth=0.5)
                    idx += 1
                    
        plt.tight_layout()
        self.fig = fig
        return fig
        
    def save_to_image(self, filename="ecg_plot.png"):
        """
        Save current plot to image file
        """
        if self.fig:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to {filename}")
            return filename
        return None
        
    def get_image_bytes(self):
        """
        Get plot as bytes (for PDF/Word export)
        """
        if self.fig:
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            return buf.getvalue()
        return None
