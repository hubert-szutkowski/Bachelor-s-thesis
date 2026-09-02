"""
Detection scoring tests.

The test that carries the module is the one showing a detector which finds every beat and
places each one eighty milliseconds late: it scores as perfect at the standard tolerance
and falls apart at the strict one. That is the failure a single wide window hides, and the
reason every score here is reported at two.

The rest checks that the tolerance boundaries and the reliability zones are the published
ones, since both are quoted in the thesis and neither may drift.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.qrs_score import (
    RELIABILITY_ZONES,
    STANDARD_TOLERANCE_MS,
    STRICT_TOLERANCE_MS,
    format_score,
    reliability_zone,
    score_detection,
    score_waveform,
    timing_sensitivity,
)

FS = 360.0


def beats(count=120, interval=288):
    return np.arange(1, count + 1) * interval


def shifted(peaks, milliseconds, fs=FS):
    return np.asarray(peaks) + int(round(milliseconds * 1e-3 * fs))


# --- the failure a single wide window hides ------------------------------

def test_a_detector_late_by_eighty_milliseconds_looks_perfect_at_the_standard_window():
    """
    Which is why the standard tolerance is not reported alone.

    A hundred and fifty milliseconds is more than half a QRS complex, so a detection that
    is consistently misplaced still counts as correct.
    """
    reference = beats()
    late = shifted(reference, 80.0)

    score = score_detection(late, reference, FS)
    assert score['standard']['f1'] == pytest.approx(1.0)


def test_the_same_detector_falls_apart_at_the_strict_window():
    reference = beats()
    late = shifted(reference, 80.0)

    score = score_detection(late, reference, FS)
    assert score['strict']['f1'] == pytest.approx(0.0)
    assert score['f1_loss'] == pytest.approx(1.0)


def test_a_detector_on_time_scores_the_same_at_both_windows():
    reference = beats()
    score = score_detection(reference, reference, FS)

    assert score['standard']['f1'] == pytest.approx(1.0)
    assert score['strict']['f1'] == pytest.approx(1.0)
    assert score['f1_loss'] == pytest.approx(0.0)


def test_the_offset_is_reported_and_recovers_the_shift():
    """The quantity the wide window hides, and the one variability analysis depends on."""
    reference = beats()
    score = score_detection(shifted(reference, 30.0), reference, FS)
    assert abs(score['standard']['offset_mean_ms']) == pytest.approx(30.0, abs=3.0)


def test_the_error_rate_rises_when_the_window_is_tightened():
    reference = beats()
    score = score_detection(shifted(reference, 80.0), reference, FS)
    assert score['der_rise'] > 1.0


# --- the curve over tolerances -------------------------------------------

def test_the_score_falls_steeply_for_a_misplaced_detector():
    """
    Reklewski & Augustyniak 2025 measured a detection error rate moving from 8.1 to 67.2
    percent for a change of eleven milliseconds in the tolerance.

    A method whose curve falls steeply is finding the beats and misplacing them.
    """
    reference = beats()
    rows = timing_sensitivity(shifted(reference, 60.0), reference, FS)

    scores = [row['f1'] for row in rows]
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(0.0)
    assert scores == sorted(scores, reverse=True)


def test_the_curve_stays_flat_for_a_detector_on_time():
    reference = beats()
    rows = timing_sensitivity(reference, reference, FS)
    assert all(row['f1'] == pytest.approx(1.0) for row in rows)


def test_the_curve_is_ordered_from_the_widest_window_down():
    reference = beats()
    rows = timing_sensitivity(reference, reference, FS)
    tolerances = [row['tolerance_ms'] for row in rows]
    assert tolerances == sorted(tolerances, reverse=True)


# --- the tolerances themselves -------------------------------------------

def test_the_standard_window_is_the_one_the_norm_defines():
    """ANSI/AAMI EC38 and EC57, 150 ms."""
    assert STANDARD_TOLERANCE_MS == 150.0


def test_the_strict_window_is_the_one_timing_studies_use():
    """50 ms in wearable and benchmarking work; Kristof et al. use 75."""
    assert STRICT_TOLERANCE_MS == 50.0


def test_a_strict_window_wider_than_the_standard_one_is_rejected():
    reference = beats()
    with pytest.raises(ValueError, match='tighter'):
        score_detection(reference, reference, FS, standard_ms=50.0, strict_ms=150.0)


def test_both_tolerances_travel_with_the_score():
    """So that a number in a table can be read without knowing how it was produced."""
    reference = beats()
    score = score_detection(reference, reference, FS)
    assert score['standard_tolerance_ms'] == 150.0
    assert score['strict_tolerance_ms'] == 50.0


# --- missed and spurious beats -------------------------------------------

def test_a_missed_beat_lowers_the_sensitivity():
    reference = beats()
    partial = reference[::2]
    score = score_detection(partial, reference, FS)
    assert score['standard']['sensitivity'] == pytest.approx(0.5, abs=0.02)


def test_a_spurious_beat_lowers_the_predictivity():
    reference = beats()
    extra = np.sort(np.concatenate([reference, reference[:30] + 144]))
    score = score_detection(extra, reference, FS)
    assert score['standard']['positive_predictivity'] < 1.0


def test_the_error_rate_counts_both_kinds_of_mistake():
    reference = beats()
    partial = reference[::2]
    score = score_detection(partial, reference, FS)
    assert score['standard']['der'] == pytest.approx(0.5, abs=0.02)


# --- reliability zones ---------------------------------------------------

def test_the_zones_are_the_published_boundaries():
    """Fariha et al. 2020, Kim & Shin 2016, Reklewski et al. 2024."""
    assert reliability_zone(20.0)['zone'] == 'pewna'
    assert reliability_zone(8.0)['zone'] == 'dobra'
    assert reliability_zone(2.0)['zone'] == 'przejsciowa'
    assert reliability_zone(-5.0)['zone'] == 'niepewna'
    assert reliability_zone(-20.0)['zone'] == 'zawodna'


def test_the_boundaries_are_ordered():
    boundaries = [boundary for boundary, _, _ in RELIABILITY_ZONES]
    assert boundaries == sorted(boundaries, reverse=True)


def test_the_grid_of_this_work_spans_four_zones():
    """
    Minus nine to eleven decibels, and the noise used is electrode motion.

    Which is the hardest case in the published breakdown, so the lower half of the grid is
    where the detector rather than the filter decides the result.
    """
    zones = {reliability_zone(level)['zone']
             for level in (-9.0, -5.0, -1.0, 3.0, 7.0, 11.0)}
    assert zones == {'niepewna', 'przejsciowa', 'dobra'}


def test_the_zone_travels_with_the_score_when_the_ratio_is_given():
    reference = beats()
    score = score_detection(reference, reference, FS, snr_db=-9.0)
    assert score['reliability']['zone'] == 'niepewna'


def test_no_zone_is_claimed_without_a_ratio():
    reference = beats()
    assert 'reliability' not in score_detection(reference, reference, FS)


def test_an_unknown_ratio_is_named_as_such():
    assert reliability_zone(float('nan'))['zone'] == 'nieznana'


# --- scoring a waveform --------------------------------------------------

def ecg(n=4096, fs=FS, bpm=75, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    peaks = np.arange(int(fs), n - int(fs), int(60.0 / bpm * fs))
    for peak in peaks:
        signal[peak - 4:peak + 4] += np.hanning(8) * 1.2
        signal[peak + 40:peak + 98] += np.hanning(58) * 0.25
    return signal + noise * rng.standard_normal(n), peaks


def test_a_clean_waveform_scores_well_at_both_windows():
    signal, peaks = ecg(noise=0.02)
    score = score_waveform(signal, peaks, FS)
    assert score['standard']['f1'] > 0.9
    assert score['n_detected'] > 0


def test_the_score_carries_the_number_of_detections():
    signal, peaks = ecg()
    score = score_waveform(signal, peaks, FS)
    assert isinstance(score['n_detected'], int)


def test_a_ratio_can_be_attached_when_scoring_a_waveform():
    signal, peaks = ecg()
    score = score_waveform(signal, peaks, FS, snr_db=3.0)
    assert score['reliability']['zone'] == 'przejsciowa'


# --- reporting -----------------------------------------------------------

def test_the_summary_shows_both_windows_and_the_loss():
    reference = beats()
    text = format_score(score_detection(shifted(reference, 80.0), reference, FS,
                                        snr_db=-9.0))
    assert 'standard' in text and 'scisle' in text
    assert 'spadek F1' in text
    assert 'niepewna' in text


def test_the_summary_omits_the_zone_when_no_ratio_was_given():
    reference = beats()
    text = format_score(score_detection(reference, reference, FS))
    assert 'strefa' not in text


def test_a_detector_that_matched_nothing_scores_zero_not_unknown():
    """
    The distinction between total failure and absence of data.

    Returning an unknown value would drop the method from every aggregate as non-finite,
    so the method that failed hardest would vanish from the table instead of lowering it.
    """
    reference = beats()
    hopeless = shifted(reference, 400.0)

    score = score_detection(hopeless, reference, FS)
    assert score['standard']['f1'] == 0.0
    assert score['strict']['f1'] == 0.0


def test_a_comparison_with_nothing_on_either_side_stays_unknown():
    from preparing.qrs_detection import detection_metrics

    metrics = detection_metrics(np.array([]), np.array([]), FS)
    assert math.isnan(metrics['f1'])
