"""
Window generator tests.

Two things are asserted. Every window has the width it declares, since a window of any
other size reaches a filter or a tensor and fails far from here. And the generator holds
one window at a time, since the whole reason for generating rather than collecting is
that a recording cut with an overlap does not fit in memory comfortably.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from preparing.windows import (
    STATIC_FILTER_WIDTH,
    beat_windows,
    count_windows,
    mean_rr,
    sliding_windows,
    stride_from_overlap,
)

FS = 360.0
N = 200000


def regular_peaks(bpm=75, n_samples=N, fs=FS):
    rr = int(round(60.0 / bpm * fs))
    return np.arange(rr, n_samples - rr, rr, dtype=np.int64)


# --- stride --------------------------------------------------------------

def test_no_overlap_abuts_the_windows():
    assert stride_from_overlap(4096, 0.0) == 4096


def test_half_overlap_advances_by_half():
    assert stride_from_overlap(4096, 0.5) == 2048


def test_an_overlap_of_one_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        stride_from_overlap(4096, 1.0)


def test_a_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        stride_from_overlap(4096, -0.1)


def test_a_nonpositive_width_is_rejected():
    with pytest.raises(ValueError, match='width'):
        stride_from_overlap(0, 0.5)


# --- sliding windows -----------------------------------------------------

@pytest.mark.parametrize('overlap', [0.0, 0.25, 0.5, 0.75])
def test_every_window_has_the_declared_width(overlap):
    for start, stop in sliding_windows(N, STATIC_FILTER_WIDTH, overlap):
        assert stop - start == STATIC_FILTER_WIDTH


@pytest.mark.parametrize('overlap', [0.0, 0.5, 0.75])
def test_windows_stay_inside_the_recording(overlap):
    windows = list(sliding_windows(N, STATIC_FILTER_WIDTH, overlap))
    assert windows[0][0] >= 0
    assert windows[-1][1] <= N


def test_overlap_multiplies_the_number_of_measurements():
    """More windows from one recording is the reason to overlap for a static filter."""
    plain = len(list(sliding_windows(N, STATIC_FILTER_WIDTH, 0.0)))
    overlapped = len(list(sliding_windows(N, STATIC_FILTER_WIDTH, 0.5)))
    assert overlapped == pytest.approx(2 * plain, abs=2)


def test_consecutive_windows_advance_by_the_stride():
    windows = list(sliding_windows(N, STATIC_FILTER_WIDTH, 0.5))
    starts = np.array([start for start, _ in windows])
    assert np.all(np.diff(starts) == stride_from_overlap(STATIC_FILTER_WIDTH, 0.5))


def test_the_trailing_remainder_is_dropped_not_padded():
    windows = list(sliding_windows(STATIC_FILTER_WIDTH + 10, STATIC_FILTER_WIDTH, 0.0))
    assert len(windows) == 1
    assert windows[0] == (0, STATIC_FILTER_WIDTH)


def test_a_recording_shorter_than_one_window_yields_nothing():
    assert list(sliding_windows(STATIC_FILTER_WIDTH - 1, STATIC_FILTER_WIDTH)) == []


def test_the_generator_holds_one_window_at_a_time():
    """A list would defeat the purpose; the object handed back must be lazy."""
    windows = sliding_windows(N, STATIC_FILTER_WIDTH, 0.5)
    assert not isinstance(windows, (list, tuple))
    assert next(windows) == (0, STATIC_FILTER_WIDTH)


def test_counting_agrees_with_generating():
    for overlap in (0.0, 0.25, 0.5, 0.75):
        assert count_windows(N, STATIC_FILTER_WIDTH, overlap) == \
               len(list(sliding_windows(N, STATIC_FILTER_WIDTH, overlap)))


# --- beat windows --------------------------------------------------------

def test_mean_rr_recovers_a_known_heart_rate():
    assert 60.0 * FS / mean_rr(regular_peaks(bpm=75)) == pytest.approx(75.0, rel=0.01)


def test_mean_rr_rejects_a_single_beat():
    with pytest.raises(ValueError, match='two beats'):
        mean_rr(np.array([100]))


def test_mean_rr_rejects_unsorted_peaks():
    with pytest.raises(ValueError, match='increasing'):
        mean_rr(np.array([100, 400, 300]))


@pytest.mark.parametrize('bpm', [50, 60, 75, 90, 120])
def test_beat_windows_have_one_width_within_a_patient(bpm):
    widths = {stop - start
              for start, stop in beat_windows(regular_peaks(bpm=bpm), N)}
    assert len(widths) == 1


def test_the_width_follows_the_patient_heart_rate():
    """
    Width differs between patients, which is the price of sizing it from the rhythm.

    Architectures that resample every cycle to a fixed length are unaffected; one that
    expects a single width across patients needs it fixed by the caller instead.
    """
    slow = next(iter(beat_windows(regular_peaks(bpm=50), N)))
    fast = next(iter(beat_windows(regular_peaks(bpm=120), N)))
    assert (slow[1] - slow[0]) > (fast[1] - fast[0])


def test_a_beat_sits_at_the_centre_of_its_window():
    peaks = regular_peaks()
    centres = {(start + stop) // 2 for start, stop in beat_windows(peaks, N)}
    assert centres <= set(peaks.tolist())


def test_the_span_scales_the_width():
    peaks = regular_peaks()
    narrow = next(iter(beat_windows(peaks, N, span=1.0)))
    wide = next(iter(beat_windows(peaks, N, span=2.0)))
    assert (wide[1] - wide[0]) == pytest.approx(2 * (narrow[1] - narrow[0]), abs=2)


def test_windows_near_the_ends_are_skipped():
    peaks = np.array([10, 5000, 10000, N - 10], dtype=np.int64)
    windows = list(beat_windows(peaks, N))
    assert all(start >= 0 and stop <= N for start, stop in windows)
    centres = [(start + stop) // 2 for start, stop in windows]
    assert 10 not in centres and N - 10 not in centres


def test_stride_in_beats_thins_the_windows():
    peaks = regular_peaks()
    every = len(list(beat_windows(peaks, N, stride_beats=1)))
    third = len(list(beat_windows(peaks, N, stride_beats=3)))
    assert third == pytest.approx(every / 3, rel=0.1)


def test_a_nonpositive_stride_is_rejected():
    with pytest.raises(ValueError, match='stride_beats'):
        list(beat_windows(regular_peaks(), N, stride_beats=0))


def test_a_nonpositive_span_is_rejected():
    with pytest.raises(ValueError, match='span'):
        list(beat_windows(regular_peaks(), N, span=0.0))
