import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.normalization import frequency_resampler, signal_scaling


# --- signal_scaling ---------------------------------------------------

def test_signal_scaling_converts_uv_to_mv():
    signal_uv = np.array([0.0, 1000.0, -2500.0, 500.0])
    expected = np.array([0.0, 1.0, -2.5, 0.5])
    assert np.allclose(signal_scaling(signal_uv), expected)


def test_signal_scaling_preserves_shape_and_dtype():
    signal_uv = np.random.RandomState(0).randn(2, 100).astype(np.float64)
    scaled = signal_scaling(signal_uv)
    assert scaled.shape == signal_uv.shape
    assert scaled.dtype == signal_uv.dtype


def test_signal_scaling_is_linear():
    signal_uv = np.random.RandomState(1).randn(1000) * 1000
    scaled = signal_scaling(signal_uv)
    assert np.allclose(scaled * 1000.0, signal_uv)


# --- frequency_resampler -----------------------------------------------

def test_resampler_is_identity_when_fs_equals_target_fs():
    signal = np.arange(10, dtype=float)
    out = frequency_resampler(signal, fs=360, target_fs=360)
    assert out is signal


def test_resampler_default_target_fs_is_360():
    signal = np.random.RandomState(0).randn(250)
    out = frequency_resampler(signal, fs=250)
    assert len(out) == 360


@pytest.mark.parametrize('fs,target_fs,n_samples', [
    (250, 360, 500),   # upsampling
    (360, 250, 720),   # downsampling
    (500, 360, 1000),  # downsampling, non-trivial gcd
])
def test_resampler_output_length_matches_ratio(fs, target_fs, n_samples):
    signal = np.random.RandomState(0).randn(n_samples)
    out = frequency_resampler(signal, fs, target_fs)
    gcd = math.gcd(fs, target_fs)
    expected_len = n_samples * (target_fs // gcd) // (fs // gcd)
    assert len(out) == expected_len


def test_resampler_preserves_dominant_frequency():
    fs = 250
    target_fs = 360
    freq = 10.0
    duration = 2.0
    t = np.arange(0, duration, 1 / fs)
    sine = np.sin(2 * np.pi * freq * t)

    resampled = frequency_resampler(sine, fs, target_fs)

    spectrum = np.fft.rfft(resampled)
    freqs = np.fft.rfftfreq(len(resampled), d=1 / target_fs)
    peak_freq = freqs[np.argmax(np.abs(spectrum))]

    assert peak_freq == pytest.approx(freq, abs=0.5)


def test_resampler_preserves_amplitude():
    fs = 250
    target_fs = 360
    freq = 5.0
    duration = 4.0
    t = np.arange(0, duration, 1 / fs)
    sine = 2.0 * np.sin(2 * np.pi * freq * t)

    resampled = frequency_resampler(sine, fs, target_fs)

    assert np.max(np.abs(resampled)) == pytest.approx(2.0, abs=0.1)
