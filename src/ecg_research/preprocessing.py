"""ECG filtering and per-recording normalization."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def preprocess_ecg(
    signal: np.ndarray,
    sampling_rate: int = 500,
    filter_enabled: bool = True,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 40.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Filter and normalize a (samples, leads) ECG array."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("ECG signal must have shape (samples, leads)")
    if x.shape[0] < 32 or x.shape[1] == 0:
        raise ValueError("ECG signal is too short or has no leads")
    if not np.isfinite(x).all():
        raise ValueError("ECG signal contains NaN or infinite values")

    if filter_enabled:
        nyquist = sampling_rate / 2.0
        if not 0 < lowcut_hz < highcut_hz < nyquist:
            raise ValueError("Require 0 < lowcut_hz < highcut_hz < Nyquist")
        sos = butter(
            filter_order,
            [lowcut_hz, highcut_hz],
            btype="bandpass",
            fs=sampling_rate,
            output="sos",
        )
        x = sosfiltfilt(sos, x, axis=0)

    # Normalize each lead independently within this record.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    return x.astype(np.float32)
