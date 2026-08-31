"""
Reading of exported datasets and separation of their segments into training parts.

An archive carries beat positions twice. `r_peaks` holds the reference annotations of the
database and exists to score detection against; nothing may take it as an input.
`r_peaks_detected` holds what the detector found on the noisy window, which is what a
method would have outside a database, and is what the architectures built around cardiac
cycles are given. Confusing the two would hand those architectures a segmentation no
detector could produce, and the advantage would be largest exactly where the signal is
worst.

Kept clear of PyTorch on purpose. What an archive must contain, and how its segments may
be divided without a patient appearing on both sides, is a property of the data rather
than of the framework that will consume it; keeping the two apart lets the contract be
tested wherever numpy runs, including where no accelerator or matching CUDA runtime is
installed.
"""

import numpy as np


def read_r_peaks(payload, n_segments: int, key: str = 'r_peaks'):
    '''
    Beat positions in either of the two layouts an archive may carry.

    `scripts/export_dataset.py` writes them flat: one concatenated vector of indices and a
    vector of offsets marking where each segment begins. Older archives hold an array of
    objects, which numpy can only read back with pickle enabled. The flat layout is
    preferred and read without pickle.
    '''
    if key not in payload:
        return None

    offset_key = f'{key}_offset'
    if offset_key in payload:
        flat = np.asarray(payload[key], dtype=np.int64)
        offsets = np.asarray(payload[offset_key], dtype=np.int64)
        if offsets.size != n_segments + 1:
            raise ValueError(f'{offset_key} holds {offsets.size} entries, '
                             f'expected {n_segments + 1}')
        return [flat[offsets[i]:offsets[i + 1]] for i in range(n_segments)]

    return list(payload[key])


def load_dataset(path: str, limit: int = 0) -> dict:
    try:
        payload = np.load(path, allow_pickle=False)
        list(payload.keys())
    except ValueError:
        payload = np.load(path, allow_pickle=True)

    for key in ('noisy', 'clean'):
        if key not in payload:
            raise KeyError(f"'{key}' missing from {path}")

    noisy = np.asarray(payload['noisy'], dtype=np.float64)
    clean = np.asarray(payload['clean'], dtype=np.float64)
    if noisy.shape != clean.shape:
        raise ValueError(f'noisy {noisy.shape} and clean {clean.shape} must have the same shape')

    r_peaks = read_r_peaks(payload, noisy.shape[0])
    r_peaks_detected = read_r_peaks(payload, noisy.shape[0], 'r_peaks_detected')
    patient = np.asarray(payload['patient']) if 'patient' in payload else None

    if 'fs' not in payload:
        raise KeyError(
            f"'fs' missing from {path}; a wrong sampling frequency shifts every quantity "
            f'expressed in hertz without raising anything, so it is not defaulted')
    fs = float(payload['fs'])

    if limit:
        noisy, clean = noisy[:limit], clean[:limit]
        if r_peaks is not None:
            r_peaks = r_peaks[:limit]
        if r_peaks_detected is not None:
            r_peaks_detected = r_peaks_detected[:limit]
        if patient is not None:
            patient = patient[:limit]

    return {'noisy': noisy, 'clean': clean, 'r_peaks': r_peaks,
            'r_peaks_detected': r_peaks_detected, 'patient': patient, 'fs': fs}


def split_indices(n: int, val_fraction: float) -> tuple:
    '''
    Contiguous split, used only when the archive carries no patient labels.

    Consecutive segments of one recording are strongly correlated, so a shuffle before the
    split would leak the validation subject into training. A contiguous cut is better than
    a shuffle but still lands inside whichever patient happens to straddle it.
    '''
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must lie in (0, 1)')
    cut = int(round(n * (1.0 - val_fraction)))
    cut = max(1, min(cut, n - 1))
    return np.arange(cut), np.arange(cut, n)


def split_by_patient(patient: np.ndarray, val_fraction: float) -> tuple:
    '''
    Whole patients to one side of the boundary or the other.

    This is the split the archive was built to support. A cut on segment index puts part
    of one recording in training and the rest in validation, and the model then recognises
    a waveform rather than denoising an unfamiliar one; the metric improves and nothing
    warns. Patients are taken smallest first until the requested share of segments is
    reached, which keeps the realised fraction close to the request without ever splitting
    anyone.
    '''
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must lie in (0, 1)')

    patient = np.asarray(patient)
    unique, counts = np.unique(patient, return_counts=True)
    if unique.size < 2:
        raise ValueError(f'a patient wise split needs at least two patients, got {unique.size}')

    target = val_fraction * patient.size
    order = np.argsort(counts)
    chosen, total = [], 0.0
    for position in order:
        if total >= target or len(chosen) == unique.size - 1:
            break
        chosen.append(unique[position])
        total += counts[position]

    in_val = np.isin(patient, chosen)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)
