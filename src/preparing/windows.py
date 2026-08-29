"""
Window generators.

Windows are produced as index pairs by a generator rather than collected into an array,
so a caller can noise one window, filter it, record its metrics and let it go before
asking for the next. Peak memory then stays at one window regardless of how long the
recording is, and a record that would occupy a gigabyte once cut with an overlap never
exists in that form.

Two rules cover the methods used here. `sliding` returns windows of a fixed width and is
what the static filters and the window based networks need. `beat` sizes the window from
the patient's own interbeat distance and centres it on a complex, which is what the
networks built around the cardiac cycle need.

Overlap matters for a different reason in each case. For a static filter it multiplies
the number of measurements taken from one recording, which is worth having when the
statistics are computed over those measurements. For a network it multiplies the
training material and shifts the phase at which a complex meets the window edge, which
makes the model less dependent on that phase.
"""

from typing import Iterator

import numpy as np

STATIC_FILTER_WIDTH = 4096
DEFAULT_OVERLAP = 0.5


def stride_from_overlap(width: int, overlap: float) -> int:
    """
    Step between consecutive windows.

    `overlap` is a fraction of the width: zero abuts the windows, one half makes each
    window share its first half with its predecessor. Values at or above one would place
    the windows on top of each other and never advance.
    """
    width = int(width)
    if width <= 0:
        raise ValueError(f'width must be positive, got {width}')
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f'overlap must lie in [0, 1), got {overlap}')

    stride = int(round(width * (1.0 - overlap)))
    return max(1, stride)


def sliding_windows(n_samples: int, width: int = STATIC_FILTER_WIDTH,
                    overlap: float = DEFAULT_OVERLAP) -> Iterator:
    """
    Yields `(start, stop)` for windows of fixed width across a recording.

    The trailing remainder is dropped rather than padded. Padding would introduce a
    discontinuity the filter would respond to, and the response would enter the metrics
    as if it came from the signal.
    """
    stride = stride_from_overlap(width, overlap)
    width = int(width)
    for start in range(0, int(n_samples) - width + 1, stride):
        yield start, start + width


def mean_rr(r_peaks: np.ndarray) -> float:
    """Mean interbeat distance in samples."""
    r_peaks = np.asarray(r_peaks, dtype=np.int64).ravel()
    if r_peaks.size < 2:
        raise ValueError(f'at least two beats are required, got {r_peaks.size}')
    if np.any(np.diff(r_peaks) <= 0):
        raise ValueError('beat positions must be strictly increasing')
    return float(np.diff(r_peaks).mean())


def beat_windows(r_peaks: np.ndarray, n_samples: int, span: float = 1.0,
                 stride_beats: int = 1) -> Iterator:
    """
    Yields `(start, stop)` for windows centred on beats and sized from the mean interval.

    Each window reaches `span` mean interbeat distances to either side of a complex, so
    its width is twice that and is constant within a patient while differing between
    patients. Networks that resample every cycle to a fixed length are unaffected; a
    network expecting one width across patients is not, and needs the width fixed by the
    caller instead.

    Windows whose centre lies closer to an end of the recording than half the width are
    skipped rather than shifted inwards, since shifting would break the centring the mode
    exists to provide.
    """
    r_peaks = np.asarray(r_peaks, dtype=np.int64).ravel()
    stride_beats = int(stride_beats)
    if stride_beats <= 0:
        raise ValueError(f'stride_beats must be positive, got {stride_beats}')
    if span <= 0:
        raise ValueError(f'span must be positive, got {span}')

    half = int(round(span * mean_rr(r_peaks)))
    for peak in r_peaks[::stride_beats]:
        start, stop = int(peak) - half, int(peak) + half
        if start >= 0 and stop <= int(n_samples):
            yield start, stop


def count_windows(n_samples: int, width: int = STATIC_FILTER_WIDTH,
                  overlap: float = DEFAULT_OVERLAP) -> int:
    """Number of sliding windows without generating them, for progress reporting."""
    stride = stride_from_overlap(width, overlap)
    span = int(n_samples) - int(width)
    return 0 if span < 0 else span // stride + 1
