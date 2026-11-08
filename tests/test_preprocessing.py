import numpy as np
import pytest

from src.ecg_research.preprocessing import preprocess_ecg


def test_preprocess_returns_finite_float32():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(5000, 12))
    result = preprocess_ecg(signal)
    assert result.shape == signal.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.allclose(result.mean(axis=0), 0, atol=1e-5)


def test_rejects_invalid_shape():
    with pytest.raises(ValueError):
        preprocess_ecg(np.ones(100))
