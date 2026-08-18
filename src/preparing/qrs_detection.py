"""
QRS detection and detection scoring.

The detector serves two roles in this project. It supplies beat locations for the
stacked cardiac cycle representation, and it acts as the measuring instrument behind
the downstream-task metric, where the F1 score of beat detection is compared before
and after denoising.

The second role imposes a requirement the first does not: the detector must stay
fixed across every method under comparison, and its own preprocessing must be
declared.

The `neurokit` method was selected after benchmarking six detectors on nine MIT-BIH
records, five easy and four heavily corrupted. It scored the highest F1 and, unlike
`kalidas2017`, carries no wavelet stage, which would have placed wavelet processing
inside the instrument used to evaluate two wavelet-based denoising methods.

Its cleaning stage passes roughly 1.2-19 Hz, attenuating a 0.25 Hz baseline component
by about 60 dB and removing all but 0.06 percent of the energy above 40 Hz. Baseline
wander and muscle noise are both artefact types this project sets out to remove, so
the detector partially performs the task it is meant to measure. Reporting the F1
score of the unprocessed noisy signal alongside the filtered ones lets the reader
separate the two effects.
"""

from typing import Optional

import numpy as np

DEFAULT_METHOD = 'neurokit'
TOLERANCE_MS = 150.0


def detect_qrs(signal: np.ndarray, fs: float, method: str = DEFAULT_METHOD,
               clean: bool = True) -> np.ndarray:
    """
    Locates QRS complexes.

    Parameters
    ----------
    signal : np.ndarray
        Single-channel signal.
    fs : float
        Sampling frequency in Hz.
    method : str
        Detector name understood by `neurokit2.ecg_peaks`.
    clean : bool
        Apply `neurokit2.ecg_clean` with the matching method first. Leave this enabled
        for results that are comparable with the literature; disable it only for a
        sensitivity analysis of how much the detector's own filtering contributes.

    Returns
    -------
    r_peaks : np.ndarray
        Sample indices of the detected beats, in increasing order.
    """
    import neurokit2 as nk

    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < int(fs):
        raise ValueError(f'signal shorter than one second: {signal.size} samples at {fs} Hz')
    if not np.all(np.isfinite(signal)):
        raise ValueError('signal contains NaN or infinite values')

    prepared = nk.ecg_clean(signal, sampling_rate=fs, method=method) if clean else signal
    _, info = nk.ecg_peaks(prepared, sampling_rate=fs, method=method)
    return np.asarray(info['ECG_R_Peaks'], dtype=int)


def match_detections(detected: np.ndarray, reference: np.ndarray, fs: float,
                     tolerance_ms: float = TOLERANCE_MS) -> dict:
    """
    One-to-one matching of detected beats against reference annotations.

    Every reference annotation may claim at most one detection and vice versa. Without
    this constraint a burst of detections around a single beat would be counted as
    several true positives, which inflates sensitivity while hiding the false alarms.
    Pairs are assigned greedily in order of increasing distance, which is the
    assignment used by the WFDB comparison tools.

    Parameters
    ----------
    detected : np.ndarray
        Sample indices produced by the detector.
    reference : np.ndarray
        Sample indices of the reference annotations.
    fs : float
        Sampling frequency in Hz.
    tolerance_ms : float
        Matching window in milliseconds, measured on each side. The value of 150 ms
        follows ANSI/AAMI EC57.

    Returns
    -------
    result : dict
        Counts `tp`, `fp`, `fn`, the matched index pairs, and the signed offsets of
        matched detections in samples.
    """
    detected = np.sort(np.asarray(detected, dtype=int).ravel())
    reference = np.sort(np.asarray(reference, dtype=int).ravel())
    if tolerance_ms <= 0:
        raise ValueError(f'tolerance_ms must be positive, got {tolerance_ms}')

    tolerance = tolerance_ms * 1e-3 * fs

    candidates = []
    for det_idx, position in enumerate(detected):
        lo = np.searchsorted(reference, position - tolerance, side='left')
        hi = np.searchsorted(reference, position + tolerance, side='right')
        for ref_idx in range(lo, hi):
            candidates.append((abs(position - reference[ref_idx]), det_idx, ref_idx))

    candidates.sort()
    used_detected: set = set()
    used_reference: set = set()
    pairs = []
    for _, det_idx, ref_idx in candidates:
        if det_idx in used_detected or ref_idx in used_reference:
            continue
        used_detected.add(det_idx)
        used_reference.add(ref_idx)
        pairs.append((det_idx, ref_idx))

    offsets = np.array([detected[d] - reference[r] for d, r in pairs], dtype=float)
    return {
        'tp': len(pairs),
        'fp': int(detected.size - len(pairs)),
        'fn': int(reference.size - len(pairs)),
        'pairs': pairs,
        'offsets': offsets,
        'n_detected': int(detected.size),
        'n_reference': int(reference.size),
    }


def detection_metrics(detected: np.ndarray, reference: np.ndarray, fs: float,
                      tolerance_ms: float = TOLERANCE_MS) -> dict:
    """
    Sensitivity, positive predictivity, F1 score and detection error rate.

    Se  = TP / (TP + FN)
    P+  = TP / (TP + FP)
    F1  = 2 Se P+ / (Se + P+)
    DER = (FP + FN) / (TP + FN)

    The detection error rate is normalised by the number of reference beats, following
    the convention used in the QRS detection literature. Unlike the other three it is
    not bounded by one.

    Returns
    -------
    metrics : dict
        Counts, the four measures, and the mean and standard deviation of the temporal
        offset of matched detections in milliseconds.
    """
    match = match_detections(detected, reference, fs, tolerance_ms)
    tp, fp, fn = match['tp'], match['fp'], match['fn']

    sensitivity = tp / (tp + fn) if tp + fn else float('nan')
    predictivity = tp / (tp + fp) if tp + fp else float('nan')
    if np.isnan(sensitivity) or np.isnan(predictivity) or sensitivity + predictivity == 0:
        f1 = float('nan')
    else:
        f1 = 2.0 * sensitivity * predictivity / (sensitivity + predictivity)
    der = (fp + fn) / (tp + fn) if tp + fn else float('nan')

    offsets_ms = match['offsets'] / fs * 1e3
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'positive_predictivity': predictivity,
        'f1': f1,
        'der': der,
        'offset_mean_ms': float(np.mean(offsets_ms)) if offsets_ms.size else float('nan'),
        'offset_std_ms': float(np.std(offsets_ms)) if offsets_ms.size else float('nan'),
        'n_detected': match['n_detected'],
        'n_reference': match['n_reference'],
    }


def score_signal(signal: np.ndarray, reference: np.ndarray, fs: float,
                 method: str = DEFAULT_METHOD, clean: bool = True,
                 tolerance_ms: float = TOLERANCE_MS) -> dict:
    """
    Runs the detector on a signal and scores it against reference annotations.

    Convenience wrapper for the downstream-task metric: call it once on the noisy
    signal to obtain the baseline, then once per denoising method.
    """
    detected = detect_qrs(signal, fs, method=method, clean=clean)
    return detection_metrics(detected, reference, fs, tolerance_ms)
