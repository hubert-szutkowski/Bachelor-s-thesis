"""
Calibrated mixing of electrode motion artefact into clean ECG.

Only the `em` record from the MIT-BIH Noise Stress Test Database is used. That is the
same choice the database authors made when they generated the twelve premixed records
`118e*` and `119e*`, so signal-to-noise ratios produced here carry the same meaning as
those quoted throughout the literature built on that database. The record already
contains substantial baseline wander and muscle noise, which makes a weighted mixture
of `em`, `ma` and `bw` double count both.

Two signal-to-noise conventions live side by side because the literature is split.

`power_ratio` is the default and follows the deep learning denoising papers, including
equations (11) and (12) of Wang et al. 2023: both powers are plain mean squares over the
segment. Results produced with it can be placed next to the figures published by the
works whose architectures this project reimplements.

`nst` follows the WFDB tool behind the premixed NSTDB records: signal power from the
trimmed peak-to-peak QRS amplitude, noise power from a trimmed short-term root mean
square. It exists because mean square power of an ECG grows with the square of the heart
rate, so the same waveform recorded faster reports more power without becoming any more
visible. That property makes it the better measure in principle and the wrong one in
practice, since almost nobody uses it.

An attempt to reproduce the premixed records `118e*` under the `nst` convention did not
succeed: the injected noise in those files is larger than the definition predicts by a
constant factor of about 24, and the noise-free stretches are not noise free. Those files
are therefore not used as a reference here. The finding is recorded in
`scripts/validate_noise_mixing.py` and `scripts/diagnose_noise_power.py`.

Whichever convention is selected, `mix_at_snr` always reports the ratio under
`power_ratio` as well, so a table of results stays comparable with either family.
"""

from typing import Optional

import numpy as np

QRS_WINDOW_MS = 50.0
TRIM_FRACTION = 0.05
MAX_BEATS = 300
NOISE_CHUNK_S = 1.0
MAX_NOISE_CHUNKS = 300

NSTDB_SNR_LEVELS = (-6.0, 0.0, 6.0, 12.0, 18.0, 24.0)
NSTDB_LEARNING_S = 300.0
NSTDB_BLOCK_S = 120.0

CONVENTIONS = ('power_ratio', 'nst')
DEFAULT_CONVENTION = 'power_ratio'

# Wang et al. 2023, sekcja 2.4.2
WANG_SNR_LEVELS = (-1.0, 3.0, 7.0)
WANG_RECORDS_MLII = ('100', '103', '106', '109', '115', '116', '123', '202',
                     '205', '209', '220', '223', '230', '231', '234')
WANG_RECORDS_V1 = ('106', '109', '115', '116', '202', '205', '209', '220',
                   '223', '230', '231', '234')
WANG_SEGMENT_LENGTH = 1024


def qrs_amplitude(signal: np.ndarray, r_peaks: np.ndarray, fs: float,
                  window_ms: float = QRS_WINDOW_MS, trim: float = TRIM_FRACTION,
                  max_beats: int = MAX_BEATS) -> float:
    """
    Trimmed peak-to-peak QRS amplitude, the estimator `nst` uses for signal size.

    Measures the range of the signal within `window_ms` on each side of every
    annotated beat, discards the `trim` fraction at each tail and averages the rest.
    Trimming is what makes the estimate survive a handful of beats corrupted by a large
    artefact; without it a single disturbed beat moves the result by tens of percent.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    r_peaks = np.asarray(r_peaks, dtype=int).ravel()

    if r_peaks.size == 0:
        raise ValueError('at least one beat annotation is required')
    if not 0.0 <= trim < 0.5:
        raise ValueError(f'trim must lie in [0, 0.5), got {trim}')

    half = int(round(window_ms * 1e-3 * fs))
    amplitudes = []
    for peak in r_peaks[:max_beats]:
        start, stop = max(0, peak - half), min(signal.size, peak + half + 1)
        if stop - start > 2:
            amplitudes.append(np.ptp(signal[start:stop]))

    if not amplitudes:
        raise ValueError('no beat produced a usable measurement window')

    amplitudes = np.sort(np.asarray(amplitudes))
    cut = int(round(trim * amplitudes.size))
    kept = amplitudes[cut:amplitudes.size - cut] if amplitudes.size - 2 * cut > 0 else amplitudes
    return float(kept.mean())


def signal_power(signal: np.ndarray, r_peaks: np.ndarray, fs: float, **kwargs) -> float:
    """
    Signal power as defined by `nst`: the squared QRS amplitude divided by eight.

    The divisor is exact for a sinusoid, where a peak-to-peak amplitude of A gives a
    mean square of A squared over eight, and is used here as the same convention.
    """
    amplitude = qrs_amplitude(signal, r_peaks, fs, **kwargs)
    return amplitude ** 2 / 8.0


def noise_power(noise: np.ndarray, fs: float, chunk_s: float = NOISE_CHUNK_S,
                trim: float = TRIM_FRACTION, max_chunks: int = MAX_NOISE_CHUNKS) -> float:
    """
    Noise power as defined by `nst`.

    Splits the leading part of the record into chunks of `chunk_s`, takes the root mean
    square deviation from the mean of each chunk, trims the extreme values and squares
    the average. Removing the chunk mean before the root mean square keeps very low
    frequency drift from dominating the estimate.
    """
    noise = np.asarray(noise, dtype=np.float64).ravel()
    chunk = int(round(chunk_s * fs))
    if chunk < 2:
        raise ValueError(f'chunk of {chunk} samples is too short')
    if noise.size < chunk:
        raise ValueError(f'noise shorter than one chunk: {noise.size} < {chunk}')

    n_chunks = min(max_chunks, noise.size // chunk)
    values = []
    for index in range(n_chunks):
        block = noise[index * chunk:(index + 1) * chunk]
        values.append(np.sqrt(np.mean((block - block.mean()) ** 2)))

    values = np.sort(np.asarray(values))
    cut = int(round(trim * values.size))
    kept = values[cut:values.size - cut] if values.size - 2 * cut > 0 else values
    return float(kept.mean() ** 2)


def mean_square(signal: np.ndarray) -> float:
    """
    Mean square value, the power definition of the deep learning denoising literature.

    Equations (11) and (12) of Wang et al. 2023 express the ratio as a quotient of sums
    over a segment of fixed length; with equal lengths the sums may be replaced by means
    without changing the ratio.
    """
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.mean(signal ** 2))


def powers_for_convention(clean: np.ndarray, noise: np.ndarray, fs: float,
                          r_peaks: Optional[np.ndarray] = None,
                          convention: str = DEFAULT_CONVENTION) -> tuple:
    """
    Signal and noise power under the requested convention.

    `power_ratio` follows the deep learning denoising literature: both powers are plain
    mean squares over the segment. `nst` follows the WFDB tool behind the premixed NSTDB
    records: signal power from the trimmed peak-to-peak QRS amplitude, noise power from
    the trimmed short-term root mean square.

    The two are not interchangeable. For the same waveform they differ by more than an
    order of magnitude, so a ratio of minus six decibels means a different amount of
    noise under each. Whichever is chosen has to be stated alongside every result.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f'unknown convention {convention!r}; available: {CONVENTIONS}')

    if convention == 'power_ratio':
        return mean_square(clean), mean_square(noise)

    if r_peaks is None:
        raise ValueError("the 'nst' convention needs r_peaks to size the QRS complex")
    return signal_power(clean, r_peaks, fs), noise_power(noise, fs)


def power_ratio_snr(clean: np.ndarray, other: np.ndarray) -> float:
    """
    Ratio of the clean signal to its difference from another signal, in decibels.

    Equation (11) of Wang et al. 2023 when `other` is the noisy signal, equation (12)
    when it is the denoised one. The improvement reported in that work is the difference
    between the two.
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    other = np.asarray(other, dtype=np.float64).ravel()
    if clean.size != other.size:
        raise ValueError(f'length mismatch: {clean.size} and {other.size}')

    residual = float(np.sum((other - clean) ** 2))
    if residual == 0:
        return float('inf')
    return float(10.0 * np.log10(np.sum(clean ** 2) / residual))


def noise_gain(power_signal: float, power_noise: float, snr_db: float) -> float:
    """
    Multiplicative gain that brings the noise to the requested signal-to-noise ratio.

    Solves SNR = 10 log10(S / (N a^2)) for a.
    """
    if power_signal <= 0:
        raise ValueError(f'signal power must be positive, got {power_signal}')
    if power_noise <= 0:
        raise ValueError(f'noise power must be positive, got {power_noise}')
    return float(np.sqrt(power_signal / (power_noise * 10.0 ** (snr_db / 10.0))))


def split_noise(noise: np.ndarray, fractions=(0.6, 0.2, 0.2)) -> dict:
    """
    Splits a noise record into disjoint stretches along the time axis.

    The noise records last half an hour while a full training set draws hundreds of
    thousands of windows from them, so every sample is reused many times. Reusing the
    same stretch across the training and test sets lets a network memorise one
    realisation of the noise instead of learning to remove it, and the resulting
    optimism is invisible in the metrics.
    """
    noise = np.asarray(noise, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=float)
    if fractions.size != 3:
        raise ValueError('three fractions are required: train, validation, test')
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f'fractions must sum to one, got {fractions.sum()}')

    n = noise.shape[0]
    bounds = np.concatenate([[0], np.cumsum(np.round(fractions * n)).astype(int)])
    bounds[-1] = n
    return {
        'train': noise[bounds[0]:bounds[1]],
        'val': noise[bounds[1]:bounds[2]],
        'test': noise[bounds[2]:bounds[3]],
    }


def sample_noise_window(noise: np.ndarray, length: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Draws a window of `length` samples starting at a random continuous offset.

    Offsets are drawn from the whole admissible range rather than from a grid of
    non-overlapping tiles. Tiling an eighteen minute stretch into windows of a thousand
    samples leaves a few hundred distinct windows, which a training run exhausts many
    times over; continuous offsets give hundreds of thousands.
    """
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape[0] < length:
        raise ValueError(f'noise stretch of {noise.shape[0]} samples is shorter '
                         f'than the requested window of {length}')
    start = int(rng.integers(0, noise.shape[0] - length + 1))
    return noise[start:start + length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, r_peaks: np.ndarray,
               fs: float, snr_db: float,
               power_clean: Optional[float] = None,
               convention: str = DEFAULT_CONVENTION) -> tuple:
    """
    Adds noise to a clean signal at a calibrated signal-to-noise ratio.

    Parameters
    ----------
    clean : np.ndarray
        Clean signal, used as the training target.
    noise : np.ndarray
        Noise stretch of the same length.
    r_peaks : np.ndarray
        Beat locations in `clean`. Only the `nst` convention needs them.
    fs : float
        Sampling frequency in Hz.
    snr_db : float
        Requested ratio.
    power_clean : float, optional
        Precomputed signal power. Pass it when mixing many segments of one record to
        avoid recomputing the estimate for every segment.
    convention : str
        `power_ratio` or `nst`.

    Returns
    -------
    noisy : np.ndarray
    meta : dict
        Gain applied, both power estimates, the realised ratio under the chosen
        convention, and the ratio expressed under `power_ratio` so that results stay
        comparable with either family of publications.
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    noise = np.asarray(noise, dtype=np.float64).ravel()
    if clean.size != noise.size:
        raise ValueError(f'length mismatch: clean {clean.size}, noise {noise.size}')

    # moc mierzona na tym, co faktycznie zostanie dodane, inaczej odjecie skladowej
    # stalej rozjezdza sie z kalibracja o roznice miedzy srednim kwadratem a wariancja
    centred = noise - noise.mean()
    measured_clean, power_noise = powers_for_convention(
        clean, centred, fs, r_peaks=r_peaks, convention=convention)
    if power_clean is None:
        power_clean = measured_clean

    gain = noise_gain(power_clean, power_noise, snr_db)
    scaled = gain * centred
    noisy = clean + scaled

    return noisy, {
        'gain': gain,
        'convention': convention,
        'power_signal': power_clean,
        'power_noise': power_noise,
        'snr_db_requested': float(snr_db),
        'snr_db_realised': float(10.0 * np.log10(power_clean / (power_noise * gain ** 2))),
        'snr_db_power_ratio': power_ratio_snr(clean, noisy),
    }


def nstdb_protocol_mask(n_samples: int, fs: float,
                        learning_s: float = NSTDB_LEARNING_S,
                        block_s: float = NSTDB_BLOCK_S) -> np.ndarray:
    """
    Boolean mask marking the stretches where the premixed NSTDB records carry noise.

    The database protocol leaves the first five minutes clean, then alternates two
    minute noisy and two minute clean blocks to the end of the record. Reproducing it
    is what allows a generated record to be compared sample by sample against the
    official `118e06` and its siblings, which turns the calibration from an assertion
    into a verified property.
    """
    mask = np.zeros(int(n_samples), dtype=bool)
    learning = int(round(learning_s * fs))
    block = int(round(block_s * fs))
    if block < 1:
        raise ValueError(f'block of {block} samples is too short')

    position = learning
    noisy = True
    while position < n_samples:
        if noisy:
            mask[position:min(position + block, n_samples)] = True
        position += block
        noisy = not noisy
    return mask


def mix_nstdb_protocol(clean: np.ndarray, noise: np.ndarray, r_peaks: np.ndarray,
                       fs: float, snr_db: float) -> tuple:
    """
    Generates a record following the NSTDB protocol, for validation against `118e*`.

    The noise record is repeated from the beginning when it runs out, as `nst` does.
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    noise = np.asarray(noise, dtype=np.float64).ravel()

    repeats = int(np.ceil(clean.size / noise.size))
    tiled = np.tile(noise, repeats)[:clean.size]

    power_clean = signal_power(clean, r_peaks, fs)
    power_noise = noise_power(noise, fs)
    gain = noise_gain(power_clean, power_noise, snr_db)

    mask = nstdb_protocol_mask(clean.size, fs)
    noisy = clean.copy()
    noisy[mask] += gain * (tiled[mask] - noise.mean())

    return noisy, {
        'gain': gain,
        'power_signal': power_clean,
        'power_noise': power_noise,
        'snr_db_requested': float(snr_db),
        'noisy_fraction': float(mask.mean()),
    }
