"""
Detection scoring tests.

The matching logic is the part of this module where an error stays invisible: a wrong
assignment rule still returns plausible percentages. The cases below pin down the
one-to-one constraint, the tolerance boundary and the degenerate inputs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.qrs_detection import (
    TOLERANCE_MS,
    detect_qrs,
    detection_metrics,
    match_detections,
    score_signal,
)

FS = 360.0


# --- matching -----------------------------------------------------------

def test_perfect_match_scores_one():
    reference = np.arange(10) * 300
    result = detection_metrics(reference, reference, FS)
    assert result['tp'] == 10
    assert result['fp'] == 0
    assert result['fn'] == 0
    assert result['f1'] == pytest.approx(1.0)
    assert result['der'] == pytest.approx(0.0)


def test_missing_detection_counts_as_false_negative():
    reference = np.arange(10) * 300
    detected = np.delete(reference, 4)
    result = detection_metrics(detected, reference, FS)
    assert (result['tp'], result['fp'], result['fn']) == (9, 0, 1)
    assert result['sensitivity'] == pytest.approx(0.9)
    assert result['positive_predictivity'] == pytest.approx(1.0)


def test_extra_detection_counts_as_false_positive():
    reference = np.arange(10) * 300
    detected = np.append(reference, 150)
    result = detection_metrics(detected, reference, FS)
    assert (result['tp'], result['fp'], result['fn']) == (10, 1, 0)
    assert result['sensitivity'] == pytest.approx(1.0)


def test_two_detections_near_one_beat_yield_one_true_positive():
    """Without a one-to-one rule this would report two true positives."""
    reference = np.array([1000])
    detected = np.array([1000 - 10, 1000 + 10])
    result = detection_metrics(detected, reference, FS)
    assert result['tp'] == 1
    assert result['fp'] == 1
    assert result['fn'] == 0


def test_closest_detection_wins_the_pairing():
    reference = np.array([1000])
    detected = np.array([1000 - 40, 1000 + 3])
    match = match_detections(detected, reference, FS)
    matched_detection = detected[match['pairs'][0][0]]
    assert matched_detection == 1003


def test_detection_exactly_at_tolerance_is_matched():
    tolerance_samples = int(TOLERANCE_MS * 1e-3 * FS)
    reference = np.array([1000])
    at_boundary = np.array([1000 + tolerance_samples])
    assert detection_metrics(at_boundary, reference, FS)['tp'] == 1


def test_detection_one_sample_past_tolerance_is_not_matched():
    tolerance_samples = int(TOLERANCE_MS * 1e-3 * FS)
    reference = np.array([1000])
    outside = np.array([1000 + tolerance_samples + 1])
    result = detection_metrics(outside, reference, FS)
    assert result['tp'] == 0
    assert result['fp'] == 1
    assert result['fn'] == 1


def test_der_is_normalised_by_reference_beats():
    reference = np.arange(100) * 300
    detected = np.append(np.delete(reference, [1, 2]), [50, 80])
    result = detection_metrics(detected, reference, FS)
    assert result['der'] == pytest.approx((result['fp'] + result['fn']) / 100)


def test_offsets_report_systematic_delay():
    reference = np.arange(20) * 300
    shift = 18
    result = detection_metrics(reference + shift, reference, FS)
    assert result['tp'] == 20
    assert result['offset_mean_ms'] == pytest.approx(shift / FS * 1e3)
    assert result['offset_std_ms'] == pytest.approx(0.0, abs=1e-9)


def test_unsorted_input_is_handled():
    reference = np.arange(10) * 300
    shuffled = np.random.RandomState(0).permutation(reference)
    assert detection_metrics(shuffled, reference, FS)['tp'] == 10


@pytest.mark.parametrize('detected,reference,expected', [
    (np.array([], int), np.arange(5) * 300, (0, 0, 5)),
    (np.arange(5) * 300, np.array([], int), (0, 5, 0)),
    (np.array([], int), np.array([], int), (0, 0, 0)),
])
def test_empty_inputs(detected, reference, expected):
    match = match_detections(detected, reference, FS)
    assert (match['tp'], match['fp'], match['fn']) == expected


def test_nonpositive_tolerance_is_rejected():
    with pytest.raises(ValueError):
        match_detections(np.array([1]), np.array([1]), FS, tolerance_ms=0.0)


# --- detector -----------------------------------------------------------

def test_detector_finds_the_expected_number_of_beats():
    nk = pytest.importorskip('neurokit2')
    fs, duration, heart_rate = 360, 30, 72
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs,
                          heart_rate=heart_rate, random_state=0)
    detected = detect_qrs(ecg, fs)
    expected = duration * heart_rate / 60
    assert abs(len(detected) - expected) <= 2


def test_detector_rejects_non_finite_input():
    signal = np.zeros(1000)
    signal[10] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        detect_qrs(signal, FS)


def test_detector_rejects_signal_shorter_than_one_second():
    with pytest.raises(ValueError, match='shorter'):
        detect_qrs(np.zeros(100), FS)


def test_score_signal_on_clean_simulation_is_near_perfect():
    nk = pytest.importorskip('neurokit2')
    fs = 360
    ecg = nk.ecg_simulate(duration=60, sampling_rate=fs, heart_rate=72, random_state=1)
    reference = detect_qrs(ecg, fs)
    result = score_signal(ecg, reference, fs)
    assert result['f1'] == pytest.approx(1.0)
