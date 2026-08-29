"""
Noise mixing tests.

The signal-to-noise ratio is the axis every result in the synthetic environment is
plotted against, so a systematic error in its calibration would shift the whole
comparison without producing a single visible symptom. These tests pin the estimators
to values that can be derived by hand and check that a requested ratio is actually
realised.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.noise_mixing import (
    NSTDB_SNR_LEVELS,
    mix_at_snr,
    noise_gain,
    noise_power,
    nstdb_protocol_mask,
    qrs_amplitude,
    sample_noise_window,
    signal_power,
    split_noise,
)

FS = 360.0


QRS_WINDOW = np.hanning(10)
QRS_PEAK = QRS_WINDOW.max()          # 0.9698, the window is not sampled at its apex


def synthetic_ecg(n_beats=60, fs=FS, bpm=75, amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    rr = int(60.0 / bpm * fs)
    signal = np.zeros(n_beats * rr)
    r_peaks = np.arange(n_beats) * rr + rr // 2
    for peak in r_peaks:
        signal[peak - 5:peak + 5] += QRS_WINDOW * amplitude
        signal[peak + 20:peak + 60] += np.hanning(40) * amplitude * 0.2
    return signal, r_peaks


# --- estimators ---------------------------------------------------------

def test_qrs_amplitude_recovers_known_value():
    signal, r_peaks = synthetic_ecg(amplitude=1.0)
    assert qrs_amplitude(signal, r_peaks, FS) == pytest.approx(QRS_PEAK, rel=0.02)


def test_qrs_amplitude_scales_linearly():
    signal, r_peaks = synthetic_ecg(amplitude=1.0)
    base = qrs_amplitude(signal, r_peaks, FS)
    assert qrs_amplitude(3.0 * signal, r_peaks, FS) == pytest.approx(3.0 * base, rel=1e-9)


def test_qrs_amplitude_survives_a_few_corrupted_beats():
    """Trimming is the only thing standing between this estimate and a large artefact."""
    signal, r_peaks = synthetic_ecg(amplitude=1.0)
    for peak in r_peaks[[3, 11, 25]]:
        signal[peak - 5:peak + 5] += QRS_WINDOW * 40.0
    assert qrs_amplitude(signal, r_peaks, FS) == pytest.approx(QRS_PEAK, rel=0.10)


def test_qrs_amplitude_is_independent_of_heart_rate():
    values = [qrs_amplitude(*synthetic_ecg(bpm=bpm)[:2], FS) for bpm in (50, 75, 120)]
    assert max(values) - min(values) < 0.05 * np.mean(values)


def test_signal_power_follows_the_nst_definition():
    signal, r_peaks = synthetic_ecg(amplitude=2.0)
    amplitude = qrs_amplitude(signal, r_peaks, FS)
    assert signal_power(signal, r_peaks, FS) == pytest.approx(amplitude ** 2 / 8.0)


def test_noise_power_recovers_variance_of_white_noise():
    rng = np.random.RandomState(0)
    sigma = 0.3
    noise = sigma * rng.randn(int(300 * FS))
    assert noise_power(noise, FS) == pytest.approx(sigma ** 2, rel=0.05)


def test_noise_power_ignores_constant_offset():
    rng = np.random.RandomState(0)
    noise = 0.3 * rng.randn(int(60 * FS))
    assert noise_power(noise + 25.0, FS) == pytest.approx(noise_power(noise, FS), rel=1e-6)


def test_noise_power_rejects_too_short_input():
    with pytest.raises(ValueError, match='shorter'):
        noise_power(np.zeros(10), FS)


# --- gain ---------------------------------------------------------------

@pytest.mark.parametrize('snr_db', NSTDB_SNR_LEVELS)
def test_noise_gain_inverts_the_snr_definition(snr_db):
    power_signal, power_noise = 4.0, 0.25
    gain = noise_gain(power_signal, power_noise, snr_db)
    realised = 10.0 * np.log10(power_signal / (power_noise * gain ** 2))
    assert realised == pytest.approx(snr_db)


def test_noise_gain_rejects_nonpositive_powers():
    with pytest.raises(ValueError):
        noise_gain(0.0, 1.0, 6.0)
    with pytest.raises(ValueError):
        noise_gain(1.0, 0.0, 6.0)


# --- mixing -------------------------------------------------------------

@pytest.mark.parametrize('snr_db', NSTDB_SNR_LEVELS)
def test_requested_snr_is_realised(snr_db):
    clean, r_peaks = synthetic_ecg()
    rng = np.random.RandomState(1)
    noise = rng.randn(clean.size)
    _, meta = mix_at_snr(clean, noise, r_peaks, FS, snr_db)
    assert meta['snr_db_realised'] == pytest.approx(snr_db, abs=1e-9)


def test_lower_snr_produces_a_larger_gain():
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    gains = [mix_at_snr(clean, noise, r_peaks, FS, snr)[1]['gain']
             for snr in (24.0, 12.0, 0.0, -6.0)]
    assert all(a < b for a, b in zip(gains, gains[1:]))


def test_mixing_preserves_length_and_leaves_clean_untouched():
    clean, r_peaks = synthetic_ecg()
    original = clean.copy()
    noise = np.random.RandomState(1).randn(clean.size)
    noisy, _ = mix_at_snr(clean, noise, r_peaks, FS, 6.0)
    assert noisy.shape == clean.shape
    assert np.array_equal(clean, original)


def test_mixing_rejects_length_mismatch():
    clean, r_peaks = synthetic_ecg()
    with pytest.raises(ValueError, match='length mismatch'):
        mix_at_snr(clean, np.zeros(clean.size - 1), r_peaks, FS, 6.0)


def test_precomputed_signal_power_gives_the_same_result():
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    power = signal_power(clean, r_peaks, FS)
    a, _ = mix_at_snr(clean, noise, r_peaks, FS, 6.0, convention='nst')
    b, _ = mix_at_snr(clean, noise, r_peaks, FS, 6.0, power_clean=power,
                      convention='nst')
    assert np.allclose(a, b)


# --- noise handling -----------------------------------------------------

def test_split_noise_is_disjoint_and_covers_everything():
    noise = np.arange(1000, dtype=float)
    parts = split_noise(noise)
    assert sum(part.size for part in parts.values()) == noise.size
    joined = np.concatenate([parts['train'], parts['val'], parts['test']])
    assert np.array_equal(joined, noise)


def test_split_noise_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match='sum to one'):
        split_noise(np.zeros(100), fractions=(0.5, 0.2, 0.2))


def test_sampled_windows_come_from_many_distinct_offsets():
    noise = np.arange(100000, dtype=float)
    rng = np.random.default_rng(0)
    starts = {sample_noise_window(noise, 1024, rng)[0] for _ in range(500)}
    assert len(starts) > 450


def test_sample_window_rejects_too_short_stretch():
    with pytest.raises(ValueError, match='shorter'):
        sample_noise_window(np.zeros(100), 1024, np.random.default_rng(0))


# --- NSTDB protocol -----------------------------------------------------

def test_protocol_leaves_the_first_five_minutes_clean():
    n = int(1800 * FS)
    mask = nstdb_protocol_mask(n, FS)
    assert not mask[:int(300 * FS)].any()


def test_protocol_alternates_two_minute_blocks():
    n = int(1800 * FS)
    mask = nstdb_protocol_mask(n, FS)
    block = int(120 * FS)
    first_noisy = mask[int(300 * FS):int(300 * FS) + block]
    first_clean = mask[int(300 * FS) + block:int(300 * FS) + 2 * block]
    assert first_noisy.all()
    assert not first_clean.any()


def test_protocol_noisy_fraction_is_close_to_half_after_the_learning_period():
    n = int(1800 * FS)
    mask = nstdb_protocol_mask(n, FS)
    after = mask[int(300 * FS):]
    assert after.mean() == pytest.approx(0.5, abs=0.05)


# --- konwencje SNR ------------------------------------------------------

from preparing.noise_mixing import (
    CONVENTIONS,
    WANG_SNR_LEVELS,
    mean_square,
    power_ratio_snr,
    powers_for_convention,
)


@pytest.mark.parametrize('snr_db', WANG_SNR_LEVELS)
def test_power_ratio_convention_realises_equation_11(snr_db):
    """The realised ratio must satisfy equation (11) measured on the mixed signal."""
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    noisy, meta = mix_at_snr(clean, noise, r_peaks, FS, snr_db, convention='power_ratio')
    assert power_ratio_snr(clean, noisy) == pytest.approx(snr_db, abs=1e-6)


@pytest.mark.parametrize('snr_db', NSTDB_SNR_LEVELS)
def test_nst_convention_still_realises_its_own_definition(snr_db):
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    _, meta = mix_at_snr(clean, noise, r_peaks, FS, snr_db, convention='nst')
    assert meta['snr_db_realised'] == pytest.approx(snr_db, abs=1e-9)


def test_the_two_conventions_disagree_substantially():
    """A ratio of six decibels means a different amount of noise under each."""
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    gain_power = mix_at_snr(clean, noise, r_peaks, FS, 6.0,
                            convention='power_ratio')[1]['gain']
    gain_nst = mix_at_snr(clean, noise, r_peaks, FS, 6.0, convention='nst')[1]['gain']
    assert gain_nst / gain_power > 2.0


def test_metadata_always_reports_the_power_ratio_value():
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    _, meta = mix_at_snr(clean, noise, r_peaks, FS, 6.0, convention='nst')
    assert 'snr_db_power_ratio' in meta
    assert meta['snr_db_power_ratio'] != pytest.approx(6.0, abs=0.5)


def test_power_ratio_convention_needs_no_annotations():
    clean, _ = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    powers = powers_for_convention(clean, noise, FS, r_peaks=None,
                                   convention='power_ratio')
    assert powers[0] == pytest.approx(mean_square(clean))


def test_nst_convention_without_annotations_is_rejected():
    clean, _ = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    with pytest.raises(ValueError, match='r_peaks'):
        powers_for_convention(clean, noise, FS, r_peaks=None, convention='nst')


def test_unknown_convention_is_rejected():
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    with pytest.raises(ValueError, match='unknown convention'):
        mix_at_snr(clean, noise, r_peaks, FS, 6.0, convention='rms')


def test_power_ratio_snr_of_identical_signals_is_infinite():
    clean, _ = synthetic_ecg()
    assert power_ratio_snr(clean, clean) == float('inf')


def test_power_ratio_snr_matches_hand_computation():
    clean = np.array([3.0, 4.0, 0.0, 0.0])
    other = np.array([3.0, 4.0, 1.0, 1.0])
    expected = 10.0 * np.log10(25.0 / 2.0)
    assert power_ratio_snr(clean, other) == pytest.approx(expected)


def test_conventions_tuple_is_exhaustive():
    clean, r_peaks = synthetic_ecg()
    noise = np.random.RandomState(1).randn(clean.size)
    for convention in CONVENTIONS:
        _, meta = mix_at_snr(clean, noise, r_peaks, FS, 3.0, convention=convention)
        assert meta['convention'] == convention
