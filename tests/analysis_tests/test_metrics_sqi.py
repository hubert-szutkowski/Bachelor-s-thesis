"""
Signal quality index tests.

Two groups. The first checks that each index measures what its definition says, using
signals whose answer is known: a Gaussian has a kurtosis of three, a symmetric signal has
no skewness, a tone inside the QRS band puts all its energy there.

The second group is the one that matters for the thesis. It demonstrates, on this
implementation and not only in the cited paper, that a band-pass filter raises the
spectral index while the true signal to noise ratio falls. That failure is the reason the
evaluation carries control methods, and a test that fails if the vulnerability were ever
quietly removed is worth more than a paragraph saying it exists.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.metrics_reference import snr
from analysis.metrics_sqi import (
    QRS_BAND,
    SNR_THRESHOLDS_DB,
    SQI_FIELDS,
    TYPICAL_THRESHOLDS,
    basesqi,
    ksqi,
    psqi,
    rank_correlation,
    sqi_panel,
    ssqi,
    validate_panel,
)

FS = 360.0
N = 4096


def beats(n=N, amplitude=1.2, bpm=75):
    """
    A beat with physiological proportions, which the spectral tests need.

    A train of narrow pulses carries almost all its energy inside the QRS band already, so
    a filter restricted to that band would not damage it and the vulnerability under test
    would not appear. This waveform puts a P wave of 80 ms and a T wave of 160 ms outside
    that band, where a narrow filter destroys them: measured energy shares are 24 percent
    below 5 Hz, 23 percent inside 5 to 15 Hz and 44 percent above.
    """
    signal = np.zeros(n)
    step = int(60.0 / bpm * FS)
    for peak in range(int(FS), n - int(FS), step):
        signal[peak - 90:peak - 61] += np.hanning(29) * 0.125 * amplitude   # P
        signal[peak - 14:peak - 6] -= np.hanning(8) * 0.125 * amplitude     # Q
        signal[peak - 4:peak + 4] += np.hanning(8) * amplitude              # R
        signal[peak + 4:peak + 12] -= np.hanning(8) * 0.21 * amplitude      # S
        signal[peak + 40:peak + 98] += np.hanning(58) * 0.25 * amplitude    # T
    return signal


def tone(frequency, n=N, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(n) / FS)


# --- each index measures what its definition says ------------------------

def test_gaussian_noise_has_a_kurtosis_of_three():
    """The Pearson convention, which is the one the quoted thresholds assume."""
    noise = np.random.default_rng(0).standard_normal(200000)
    assert ksqi(noise) == pytest.approx(3.0, abs=0.1)


def test_the_fisher_convention_subtracts_three():
    noise = np.random.default_rng(0).standard_normal(200000)
    assert ksqi(noise, fisher=True) == pytest.approx(ksqi(noise) - 3.0, abs=1e-9)


def test_a_beat_train_is_far_more_peaked_than_noise():
    """Sharp complexes against a quiet baseline are a heavy tailed distribution."""
    noise = np.random.default_rng(0).standard_normal(N)
    assert ksqi(beats()) > 3.0 * ksqi(noise)


def test_adding_noise_pulls_the_kurtosis_towards_gaussian():
    clean = beats()
    rng = np.random.default_rng(0)
    heavy = clean + 1.5 * rng.standard_normal(N)
    light = clean + 0.05 * rng.standard_normal(N)
    assert abs(ksqi(heavy) - 3.0) < abs(ksqi(light) - 3.0)


def test_a_symmetric_signal_has_no_skewness():
    assert ssqi(tone(5.0)) == pytest.approx(0.0, abs=0.05)


def test_a_beat_train_is_skewed():
    """The R wave rises far above the baseline; nothing goes equally far below."""
    assert abs(ssqi(beats())) > 1.0


def test_a_tone_inside_the_qrs_band_puts_its_energy_there():
    assert psqi(tone(10.0), FS) > 0.95


def test_a_tone_outside_it_does_not():
    assert psqi(tone(35.0), FS) < 0.05


def test_muscle_noise_lowers_the_spectral_index():
    """Zhao & Zhang: high frequency content rises and the ratio falls."""
    clean = beats()
    rng = np.random.default_rng(0)
    emg = 0.4 * rng.standard_normal(N)
    assert psqi(clean + emg, FS) < psqi(clean, FS)


def test_drift_lowers_the_baseline_index():
    clean = beats()
    drift = 0.8 * tone(0.25)
    assert basesqi(clean + drift, FS) < basesqi(clean, FS)


def test_the_baseline_index_stays_near_one_without_drift():
    assert basesqi(beats(), FS) > 0.9


# --- the vulnerability that shapes the whole evaluation ------------------

def test_a_band_pass_filter_raises_the_spectral_index():
    """
    Moeyersons et al. 2019, reproduced on this implementation.

    A segment pre-processed with a Butterworth band-pass scores higher than the same
    segment untouched. The default static filter of this project is a Butterworth over
    0.5 to 40 hertz, which is that filter.
    """
    rng = np.random.default_rng(0)
    noisy = beats() + 0.4 * rng.standard_normal(N) + 0.8 * tone(0.25)

    sos = butter(4, [0.5, 40.0], btype='bandpass', fs=FS, output='sos')
    filtered = sosfiltfilt(sos, noisy)

    assert psqi(filtered, FS) > psqi(noisy, FS)


def test_a_filter_over_the_qrs_band_alone_drives_the_index_towards_one():
    """
    Measured: 0.32 before, 0.96 after, because the index and the filter share a band.

    Nothing about the P and T waves it removed enters the number.
    """
    clean = beats()
    noisy = clean + 0.2 * np.random.default_rng(0).standard_normal(N)

    sos = butter(4, [5.0, 15.0], btype='bandpass', fs=FS, output='sos')
    destructive = sosfiltfilt(sos, noisy)

    assert psqi(noisy, FS) < 0.5
    assert psqi(destructive, FS) > 0.9


def test_the_destructive_filter_beats_an_honest_one_on_the_spectral_index():
    """
    Which is the ranking a table built on this index alone would produce.

    The honest filter improves the ratio and the destructive one ruins it, and the index
    prefers the destructive one.
    """
    clean = beats()
    # poziom szumu dobrany tak, by uczciwy filtr byl faktycznie lepszy: przy silniejszym
    # szumie odrzucenie wszystkiego poza pasmem QRS podnosi rowniez prawdziwe SNR, bo
    # zespol niesie wtedy wiekszosc energii, i obie miary sie zgadzaja
    noisy = clean + 0.2 * np.random.default_rng(0).standard_normal(N)

    honest = sosfiltfilt(butter(4, [0.5, 40.0], btype='bandpass', fs=FS, output='sos'), noisy)
    destructive = sosfiltfilt(butter(4, [5.0, 15.0], btype='bandpass', fs=FS, output='sos'), noisy)

    # zmierzone: SNR +2.01 wobec +0.89 dB, pSQI 0.354 wobec 0.962
    assert snr(clean, honest) > snr(clean, destructive)
    assert psqi(destructive, FS) > 2.5 * psqi(honest, FS)


def test_the_kurtosis_index_rewards_sparsity_rather_than_quality():
    """
    A hard threshold leaves isolated spikes and raises the index while removing the waves.

    The second failure mode of the panel, and the reason detection and the reference
    metrics are read alongside it.
    """
    clean = beats()
    sparse = clean.copy()
    sparse[np.abs(sparse) < 0.5 * np.abs(sparse).max()] = 0.0

    # wskaznik przedklada uszkodzony przebieg nad sam sygnal odniesienia
    assert ksqi(sparse) > 2.0 * ksqi(clean)
    assert np.isfinite(snr(clean, sparse))
    assert snr(clean, sparse) < 10.0


# --- validation against a known ratio ------------------------------------

def test_the_spectral_index_tracks_the_true_ratio_on_synthetic_material():
    """
    The validation the panel needs before it is read where nothing is known.

    Published values give the scale: 0.94 for a learned index, 0.65 for an autocorrelation
    one.
    """
    clean = beats()
    rng = np.random.default_rng(0)

    windows, ratios = [], []
    for level in np.linspace(0.02, 1.0, 25):
        noisy = clean + level * rng.standard_normal(N)
        windows.append(noisy)
        ratios.append(snr(clean, noisy))

    correlations = validate_panel(windows, FS, ratios)

    # zmierzone: ksqi 0.937, ssqi 0.882, psqi 0.633, basesqi 0.391. Wartosci
    # publikowane daja skale: 0.94 dla wskaznika uczonego, 0.65 dla autokorelacyjnego
    assert correlations['ksqi'] > 0.9
    assert correlations['ssqi'] > 0.8
    assert correlations['psqi'] > 0.6


def test_the_baseline_index_tracks_the_ratio_worst_against_broadband_noise():
    """
    Which is what it should do: it measures drift, and this noise has none.

    Recorded so that a weak correlation here is read as the index doing its job rather
    than as the index being broken.
    """
    clean = beats()
    rng = np.random.default_rng(0)

    windows, ratios = [], []
    for level in np.linspace(0.02, 1.0, 25):
        noisy = clean + level * rng.standard_normal(N)
        windows.append(noisy)
        ratios.append(snr(clean, noisy))

    correlations = validate_panel(windows, FS, ratios)
    assert correlations['basesqi'] < correlations['ksqi']


def test_the_rank_correlation_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match='against'):
        rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_the_rank_correlation_needs_enough_usable_points():
    assert math.isnan(rank_correlation([1.0, float('nan')], [1.0, 2.0]))


def test_a_negative_correlation_would_be_visible():
    """An index measuring something other than quality has to be detectable as such."""
    assert rank_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


# --- the panel and its constants -----------------------------------------

def test_the_panel_produces_every_declared_index():
    assert set(sqi_panel(beats(), FS)) == set(SQI_FIELDS)


def test_no_composite_score_is_formed():
    """
    Shahriari et al. 2017: a multivariate combination gave no improvement over one index.

    A composite would also hide which index a method exploited, which is the question the
    band-pass vulnerability makes worth asking.
    """
    assert 'composite' not in SQI_FIELDS
    assert 'overall' not in SQI_FIELDS


def test_the_qrs_band_is_centred_at_ten_hertz_and_ten_wide():
    """Zhao & Zhang 2018."""
    low, high = QRS_BAND
    assert (low + high) / 2.0 == pytest.approx(10.0)
    assert high - low == pytest.approx(10.0)


def test_the_denominator_band_is_a_parameter():
    """
    Zhao & Zhang describe the denominator as the overall energy; Li and Clifford use
    5 to 40 hertz. The two disagree over whether baseline wander counts, so the choice is
    explicit and travels with the result.
    """
    signal = beats() + 0.8 * tone(0.25)
    wide = psqi(signal, FS, total_band=(0.5, 40.0))
    narrow = psqi(signal, FS, total_band=(5.0, 40.0))
    assert wide < narrow


def test_the_quoted_thresholds_are_kept_for_orientation_only():
    """
    Rahman et al. 2022 found fixed thresholds not to carry between datasets.

    They are recorded so the values can be cited, and nothing in this module applies them.
    """
    assert TYPICAL_THRESHOLDS['ksqi']['clean_above'] == 5.0
    assert TYPICAL_THRESHOLDS['psqi']['clean_between'] == (0.5, 0.8)


def test_the_ratio_thresholds_come_from_the_measurement_not_the_index():
    """Smital et al. 2020: 5 dB for detection, 18 dB for full waveform analysis."""
    assert SNR_THRESHOLDS_DB['qrs_detection'] == 5.0
    assert SNR_THRESHOLDS_DB['waveform_analysis'] == 18.0


# --- edge cases ----------------------------------------------------------

def test_a_constant_window_has_no_indices():
    constant = np.full(N, 2.0)
    assert math.isnan(ksqi(constant))
    assert math.isnan(ssqi(constant))


def test_a_window_too_short_for_a_spectrum_is_rejected():
    with pytest.raises(ValueError, match='too short'):
        psqi(np.zeros(4), FS)


def test_a_silent_window_has_no_spectral_index():
    assert math.isnan(psqi(np.zeros(N), FS))


def test_the_indices_do_not_depend_on_a_constant_offset():
    """The spectrum is taken with the mean removed, so a shift changes nothing."""
    signal = beats()
    assert psqi(signal + 10.0, FS) == pytest.approx(psqi(signal, FS), rel=1e-9)
    assert ksqi(signal + 10.0) == pytest.approx(ksqi(signal), rel=1e-9)
