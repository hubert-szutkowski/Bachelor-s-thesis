"""
Heart rate variability tests.

Two things carry the module and both are asserted rather than described. A single
corrupted beat has to visibly dominate a short recording, which is why the artefact burden
is reported next to every index. And a recording too short for an index has to yield no
value at all rather than a plausible looking one, since values from different durations are
not interchangeable.

The last section is the reason the module exists: a filter that improves the waveform while
shifting the complexes has to be distinguishable from one that leaves them alone, and no
waveform metric makes that distinction.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.hrv import (
    ARTEFACT_TOLERANCE,
    HRV_FIELDS,
    MINIMUM_DURATION_S,
    REFERENCE_DURATION_S,
    TIME_FIELDS,
    artefact_fraction,
    artefact_mask,
    correct_artefacts,
    format_panel,
    hrv_agreement,
    hrv_panel,
    rr_intervals,
)

FS = 360.0


def peaks_at(bpm=75.0, duration_s=300.0, jitter_ms=20.0, fs=FS, seed=0):
    """A beat train with a physiological amount of variability."""
    rng = np.random.default_rng(seed)
    interval = 60.0 / bpm * fs
    # o dwa uderzenia wiecej: n uderzen daje n-1 interwalow, wiec bez zapasu odcinek
    # wychodzi krotszy niz zamowiony i wpada ponizej progu dla najnizszego pasma
    count = int(duration_s * fs / interval) + 2
    jitter = rng.normal(0.0, jitter_ms * 1e-3 * fs, size=count)
    return np.cumsum(np.full(count, interval) + jitter).astype(int) + int(fs)


# --- intervals -----------------------------------------------------------

def test_a_steady_rate_gives_the_interval_it_implies():
    peaks = np.arange(0, 100) * 288
    assert rr_intervals(peaks, FS) == pytest.approx(800.0)


def test_a_single_beat_has_no_interval():
    with pytest.raises(ValueError, match='at least two beats'):
        rr_intervals([100], FS)


def test_unsorted_beats_are_rejected():
    with pytest.raises(ValueError, match='increasing'):
        rr_intervals([100, 400, 300], FS)


# --- artefact burden -----------------------------------------------------

def test_an_interval_outside_the_physiological_range_is_flagged():
    rr = np.array([800.0, 800.0, 150.0, 800.0, 800.0])
    assert artefact_mask(rr)[2]


def test_an_interval_departing_from_its_neighbours_is_flagged():
    """A heart rate does not change that fast between consecutive beats."""
    rr = np.array([800.0, 800.0, 800.0, 1300.0, 800.0, 800.0, 800.0])
    assert artefact_mask(rr)[3]


def test_a_steady_series_carries_no_artefact():
    rr = 800.0 + np.random.default_rng(0).normal(0.0, 15.0, size=200)
    assert artefact_fraction(rr) == 0.0


def test_correction_replaces_the_flagged_intervals_only():
    rr = np.array([800.0, 800.0, 800.0, 1500.0, 800.0, 800.0, 800.0])
    corrected = correct_artefacts(rr)

    assert corrected[3] == pytest.approx(800.0, abs=50.0)
    assert np.array_equal(corrected[[0, 1, 2, 4, 5, 6]], rr[[0, 1, 2, 4, 5, 6]])


def test_correction_leaves_a_clean_series_untouched():
    rr = 800.0 + np.random.default_rng(0).normal(0.0, 15.0, size=200)
    assert np.array_equal(correct_artefacts(rr), rr)


# --- what one bad beat does ----------------------------------------------

def test_a_single_artefact_dominates_the_successive_difference_statistic():
    """
    Bourdillon et al. 2022 measured a rise of 413 percent from one artefact.

    Reproduced here on the same statistic, which is why the burden is reported next to
    every value rather than left for the reader to ask about.
    """
    rr = 800.0 + np.random.default_rng(0).normal(0.0, 15.0, size=300)
    spoiled = rr.copy()
    spoiled[150] = 1600.0

    clean = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    damaged = float(np.sqrt(np.mean(np.diff(spoiled) ** 2)))
    assert damaged > 3.0 * clean


def test_correction_brings_the_statistic_back():
    rr = 800.0 + np.random.default_rng(0).normal(0.0, 15.0, size=300)
    spoiled = rr.copy()
    spoiled[150] = 1600.0

    clean = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    repaired = float(np.sqrt(np.mean(np.diff(correct_artefacts(spoiled)) ** 2)))
    assert repaired == pytest.approx(clean, rel=0.15)


def test_the_tolerances_are_the_measured_ones():
    """Bourdillon et al. 2022: 0.9 percent for the time domain, 1.4 for the spectral."""
    assert ARTEFACT_TOLERANCE['rmssd'] == 0.009
    assert ARTEFACT_TOLERANCE['lf_hf'] == 0.014


def test_a_burden_above_the_tolerance_is_marked():
    peaks = peaks_at(duration_s=300.0)
    spoiled = np.asarray(peaks, dtype=float)
    for index in range(20, len(spoiled), 40):
        spoiled[index] += 0.35 * 288

    panel = hrv_panel(spoiled.astype(int), FS, correct=False)
    assert panel['artefact_fraction'] > ARTEFACT_TOLERANCE['rmssd']
    assert panel['tolerance_exceeded']['rmssd']


# --- length ---------------------------------------------------------------

def test_a_short_recording_yields_only_what_it_supports():
    """
    Eleven seconds is one window of 4096 samples at 360 Hz.

    It supports the successive difference statistic and nothing spectral, which is why
    variability is computed over a recording and never over a window.
    """
    with pytest.warns(UserWarning, match='reference'):
        panel = hrv_panel(peaks_at(duration_s=11.4), FS)

    assert np.isfinite(panel['rmssd'])
    assert math.isnan(panel['lf_power'])
    assert math.isnan(panel['pnn50'])


def test_a_five_minute_recording_supports_everything():
    panel = hrv_panel(peaks_at(duration_s=300.0), FS)
    for field in HRV_FIELDS:
        assert panel['admissible'][field], field
        assert np.isfinite(panel[field]), field


def test_a_recording_below_the_reference_length_raises_a_warning():
    with pytest.warns(UserWarning, match='not\\s+interchangeable'):
        hrv_panel(peaks_at(duration_s=120.0), FS)


def test_the_minimum_durations_are_the_published_ones():
    """Uryga et al. 2025, Baek et al. 2015, Shaffer & Ginsberg 2017."""
    assert MINIMUM_DURATION_S['rmssd'] == 10.0
    assert MINIMUM_DURATION_S['pnn50'] == 60.0
    assert MINIMUM_DURATION_S['lf_power'] == 150.0
    assert MINIMUM_DURATION_S['vlf_power'] == 300.0
    assert REFERENCE_DURATION_S == 300.0


def test_an_index_below_its_own_minimum_is_not_computed():
    """Not a plausible looking number: values from different durations do not compare."""
    with pytest.warns(UserWarning):
        panel = hrv_panel(peaks_at(duration_s=30.0), FS)
    assert not panel['admissible']['pnn50']
    assert math.isnan(panel['pnn50'])


# --- the indices themselves ----------------------------------------------

def test_a_steadier_rhythm_gives_a_smaller_spread():
    steady = hrv_panel(peaks_at(jitter_ms=5.0), FS)
    variable = hrv_panel(peaks_at(jitter_ms=40.0), FS)
    assert steady['sdnn'] < variable['sdnn']
    assert steady['rmssd'] < variable['rmssd']


def test_the_mean_interval_recovers_the_rate():
    panel = hrv_panel(peaks_at(bpm=60.0), FS)
    assert panel['mean_nn'] == pytest.approx(1000.0, rel=0.02)


def test_the_normalised_powers_sum_to_one():
    panel = hrv_panel(peaks_at(duration_s=300.0), FS)
    assert panel['lf_norm'] + panel['hf_norm'] == pytest.approx(1.0)


def test_the_panel_reports_the_burden_next_to_the_values():
    """A value without it cannot be read, so the two are not separable."""
    panel = hrv_panel(peaks_at(), FS)
    assert 'artefact_fraction' in panel
    assert 'n_intervals' in panel
    assert 'duration_s' in panel


# --- the safety check the module exists for ------------------------------

def test_a_filter_that_shifts_the_beats_is_visible():
    """
    Which no waveform metric shows.

    A method may raise the signal to noise ratio and move every complex; the variability
    computed from its output then departs from the reference.
    """
    reference = peaks_at(duration_s=300.0)
    rng = np.random.default_rng(1)
    shifted = np.asarray(reference) + rng.integers(-12, 13, size=len(reference))

    agreement = hrv_agreement(reference, np.sort(shifted).astype(int), FS)
    assert agreement['relative_departure']['rmssd'] > 0.1


def test_a_filter_that_leaves_the_beats_alone_shows_no_departure():
    reference = peaks_at(duration_s=300.0)
    agreement = hrv_agreement(reference, reference, FS)
    for field in TIME_FIELDS:
        assert agreement['relative_departure'][field] == pytest.approx(0.0, abs=1e-12)


def test_the_worst_departure_is_the_one_reported():
    reference = peaks_at(duration_s=300.0)
    rng = np.random.default_rng(2)
    shifted = np.sort(np.asarray(reference) + rng.integers(-8, 9, size=len(reference)))

    agreement = hrv_agreement(reference, shifted.astype(int), FS)
    assert agreement['worst_departure'] == pytest.approx(
        max(agreement['relative_departure'].values()))


def test_the_agreement_carries_both_panels():
    reference = peaks_at(duration_s=300.0)
    agreement = hrv_agreement(reference, reference, FS)
    assert agreement['reference']['n_intervals'] == agreement['estimate']['n_intervals']


# --- reporting -----------------------------------------------------------

def test_the_summary_names_every_index_and_its_status():
    text = format_panel(hrv_panel(peaks_at(duration_s=300.0), FS))
    for field in HRV_FIELDS:
        assert field in text
    assert 'artefakty' in text


def test_the_summary_marks_what_the_recording_was_too_short_for():
    with pytest.warns(UserWarning):
        text = format_panel(hrv_panel(peaks_at(duration_s=20.0), FS))
    assert 'za krotko' in text
