"""
Non-integer sampling frequency handling, required for clock drift correction between
the ECG recorder and the accelerometer platform.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.normalization import (
    frequency_resampler,
    resampled_length,
    resampling_ratio,
)


def bandlimited(t, seed=0, n_components=60, f_max=90.0):
    """Analytic band-limited signal, so the truth is known on any sampling grid."""
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(0.5, f_max, n_components)
    amps = rng.uniform(0.1, 1.0, n_components) / np.sqrt(n_components)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_components)
    return sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))


# --- resampling_ratio ---------------------------------------------------

def test_ratio_is_exact_for_integer_rates():
    assert resampling_ratio(250, 360) == (36, 25)
    assert resampling_ratio(360, 250) == (25, 36)
    assert resampling_ratio(500, 360) == (18, 25)


def test_ratio_recovers_drifted_rate_instead_of_truncating():
    up, down = resampling_ratio(360.4, 360.0)
    assert (up, down) == (900, 901)


def test_ratio_is_unity_only_for_equal_rates():
    assert resampling_ratio(360.0, 360.0) == (1, 1)
    assert resampling_ratio(360.4, 360.0) != (1, 1)


@pytest.mark.parametrize('fs,target_fs', [(0, 360), (-360, 360), (np.nan, 360),
                                          (np.inf, 360), (360, 0), (360, -1)])
def test_ratio_rejects_invalid_rates(fs, target_fs):
    with pytest.raises(ValueError):
        resampling_ratio(fs, target_fs)


# --- frequency_resampler ------------------------------------------------

def test_drift_correction_produces_expected_length():
    fs, target_fs, duration = 360.4, 360.0, 60.0
    n_in = int(duration * fs)
    out = frequency_resampler(np.zeros(n_in), fs, target_fs)
    assert len(out) == int(round(n_in * target_fs / fs))
    assert len(out) != n_in


def test_drift_correction_is_accurate_against_analytic_truth():
    fs, target_fs, duration = 360.4, 360.0, 60.0
    n_in = int(duration * fs)
    x = bandlimited(np.arange(n_in) / fs)

    out = frequency_resampler(x, fs, target_fs)
    reference = bandlimited(np.arange(len(out)) / target_fs)

    edge = int(2 * target_fs)
    core = slice(edge, len(out) - edge)
    error = np.sqrt(np.mean((out[core] - reference[core]) ** 2)) / np.std(reference)
    assert error < 1e-3


def test_drift_correction_does_not_explode_memory():
    """A 900/901 ratio must not be routed through 900-fold upsampling."""
    fs, target_fs = 360.4, 360.0
    n_in = 360 * 600
    out = frequency_resampler(np.zeros(n_in), fs, target_fs)
    assert abs(len(out) - int(round(n_in * target_fs / fs))) <= 1


def test_multichannel_resamples_along_time_axis():
    x = np.random.RandomState(0).randn(720, 2)
    out = frequency_resampler(x, fs=360, target_fs=250)
    assert out.shape == (500, 2)


def test_multichannel_drift_correction_keeps_channel_count():
    x = np.random.RandomState(0).randn(3604, 2)
    out = frequency_resampler(x, fs=360.4, target_fs=360.0)
    assert out.shape[1] == 2
    assert out.shape[0] == int(round(3604 * 360.0 / 360.4))


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError):
        frequency_resampler(np.array([]), fs=360, target_fs=250)


# --- resampled_length ---------------------------------------------------

@pytest.mark.parametrize('n,fs,target_fs', [(500, 250, 360), (720, 360, 250),
                                            (1000, 500, 360), (3604, 360.4, 360.0),
                                            (100, 360, 360)])
def test_resampled_length_predicts_actual_output(n, fs, target_fs):
    predicted = resampled_length(n, fs, target_fs)
    actual = len(frequency_resampler(np.zeros(n), fs, target_fs))
    assert predicted == actual
