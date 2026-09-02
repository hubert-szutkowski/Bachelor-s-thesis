"""
Heart rate variability, used as a check on the filtering rather than as a score for it.

The role of this module is decided by two findings and neither is a matter of taste.

**Variability is computed from detected beats, so a filter that moves a QRS complex moves
every index derived from it.** Denoising changes the outputs rather than leaving them
alone, and reliable variability depends mainly on accurate detection of the R wave, which
makes it a downstream safety endpoint rather than a primary score for a denoiser (Saleem
et al. 2022 [10]_; Eguchi & Aoki 2023 [5]_; Cavalieri & Bertemes 2020 [3]_). A method that
improves the signal to noise ratio and shifts the beats is worse than one that improves it
less and leaves them where they were, and only this module can tell the two apart.

**A very small proportion of corrupted beats invalidates the whole analysis.** Bourdillon
et al. 2022 [2]_ measured the boundaries: the root mean square of successive differences
becomes significantly biased above 0.9 percent of artefact, the spectral powers above 1.4
percent, and a single artefact raised that statistic by 413 percent in supine data.
Reporting a variability index without reporting the artefact burden alongside it is
therefore not a partial result but a misleading one, and `hrv_panel` refuses to separate
the two.

Length matters as much as burden. Five minutes remains the reference for short term
analysis, and values from different durations are not interchangeable (Shaffer & Ginsberg
2017 [11]_; Baek et al. 2015 [1]_). Individual indices tolerate less: the successive
difference statistic can be estimated from ten to thirty seconds, the proportion of long
differences from about a minute, the low and high frequency powers need one to two and a
half minutes of clean data and the very low frequency power around five (Uryga et al. 2025
[13]_; Baek et al. 2015 [1]_; Shaffer & Ginsberg 2017 [11]_). `MINIMUM_DURATION_S` holds
those boundaries and every index is refused below its own.

The consequence for this work is that variability is not a per window quantity. A window
of 4096 samples at 360 Hz spans 11.4 seconds, which admits one time domain index and no
spectral one, so the panel is computed over a whole recording and never over a window.

The indices most easily damaged are worth knowing when reading a result: the proportion of
long differences and the ratio of spectral powers are the most sensitive to errors in beat
timing (Rohr et al. 2024 [9]_), while the successive difference statistic and the standard
deviation are the most sensitive to added artefact (Saleem et al. 2022 [10]_; Stapelberg
et al. 2018 [12]_). Normalised spectral measures are the more robust of the set.

References
----------
.. [1] Baek, H. J., Cho, C.-H., Cho, J., & Woo, J.-M. (2015). Reliability of Ultra-Short-
       Term Analysis as a Surrogate of Standard 5-Min Analysis of Heart Rate Variability.
       Telemedicine and e-Health, 21(5), 404-414. https://doi.org/10.1089/tmj.2014.0104
.. [2] Bourdillon, N., Yazdani, S., Vesin, J.-M., Schmitt, L., & Millet, G. P. (2022).
       RMSSD Is More Sensitive to Artifacts Than Frequency-Domain Parameters. Journal of
       Sports Science & Medicine, 21(2), 260-266. https://doi.org/10.52082/jssm.2022.260
.. [3] Cavalieri, R., & Bertemes-Filho, P. (2020). Determination of maximum noise level in
       an ECG channel under SURE Wavelet filtering for HRV extraction. Revista Mexicana de
       Ingenieria Biomedica. https://doi.org/10.17488/rmib.41.2.5
.. [4] Cilhoroz, B. T., Giles, D., Zaleski, A., Taylor, B., Fernhall, B., & Pescatello, L.
       (2020). Validation of the Polar V800 heart rate monitor and comparison of artifact
       correction methods among adults with hypertension. PLoS ONE, 15.
       https://doi.org/10.1371/journal.pone.0240220
.. [5] Eguchi, K., & Aoki, R. (2023). Practical R-R Interval Editing for Heart Rate
       Variability Analysis Using Single-Channel Wearable ECG Devices. IEEE Access, 11,
       25543-25582. https://doi.org/10.1109/ACCESS.2023.3253933
.. [6] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate
       variability time series artefact correction using novel beat classification. Journal
       of Medical Engineering & Technology, 43(3), 173-181.
       https://doi.org/10.1080/03091902.2019.1640306
.. [7] Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability
       in Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12).
       https://doi.org/10.3390/s21123998
.. [8] Ren, J., Zhang, R., Cao, X., & Kong, X. (2023). Experimental evaluation of ECG
       signal denoising methods based on HRV indices. Energy and Buildings.
       https://doi.org/10.1016/j.enbuild.2023.113797
.. [9] Rohr, M., Tarvainen, M., Miri, S., Guney, G., Vehkaoja, A., & Hoog Antink, C.
       (2024). An extensive quantitative analysis of the effects of errors in beat-to-beat
       intervals on all commonly used HRV parameters. Scientific Reports, 14.
       https://doi.org/10.1038/s41598-023-50701-4
.. [10] Saleem, S., Khandoker, A. H., Alkhodari, M., Hadjileontiadis, L. J., & Jelinek,
        H. F. (2022). A two-step pre-processing tool to remove Gaussian and ectopic noise
        for heart rate variability analysis. Scientific Reports, 12.
        https://doi.org/10.1038/s41598-022-21776-2
.. [11] Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate Variability
        Metrics and Norms. Frontiers in Public Health, 5.
        https://doi.org/10.3389/fpubh.2017.00258
.. [12] Stapelberg, N. J. C., Neumann, D. L., Shum, D. H. K., McConnell, H., &
        Hamilton-Craig, I. (2018). The sensitivity of 38 heart rate variability measures to
        the addition of artifact in human and artificial 24-hr cardiac recordings. Annals of
        Noninvasive Electrocardiology, 23. https://doi.org/10.1111/anec.12483
.. [13] Uryga, A., Olszewski, B., Pietron, D., & Kasprowicz, M. (2025). Impact of signal
        length and window size on heart rate variability and pulse rate variability
        metrics. Physiological Measurement, 46. https://doi.org/10.1088/1361-6579/adece2
.. [14] Sauerbier, F., et al. (2024). Impact of QRS misclassifications on
        heart-rate-variability parameters. PLOS ONE, 19, e0304893.
        https://doi.org/10.1371/journal.pone.0304893
"""

import warnings
from typing import Optional, Sequence

import numpy as np

TIME_FIELDS = ('mean_nn', 'sdnn', 'rmssd', 'pnn50')
FREQUENCY_FIELDS = ('vlf_power', 'lf_power', 'hf_power', 'lf_hf', 'lf_norm', 'hf_norm')
HRV_FIELDS = TIME_FIELDS + FREQUENCY_FIELDS

# Uryga et al. 2025 [13], Baek et al. 2015 [1], Shaffer & Ginsberg 2017 [11].
# Ponizej tych dlugosci wskaznik nie jest liczony, bo wartosci z roznych czasow trwania
# nie sa wymienne.
MINIMUM_DURATION_S = {
    'mean_nn': 10.0,
    'rmssd': 10.0,
    'pnn50': 60.0,
    'sdnn': 60.0,
    'lf_power': 150.0,
    'hf_power': 60.0,
    'lf_hf': 150.0,
    'lf_norm': 150.0,
    'hf_norm': 150.0,
    'vlf_power': 300.0,
}
REFERENCE_DURATION_S = 300.0

# Bourdillon et al. 2022 [2]: powyzej tych udzialow artefaktow wskaznik jest istotnie
# obciazony. Pojedynczy artefakt podniosl RMSSD o 413 procent w pozycji lezacej.
ARTEFACT_TOLERANCE = {'rmssd': 0.009, 'sdnn': 0.009, 'pnn50': 0.009,
                      'lf_power': 0.014, 'hf_power': 0.014, 'lf_hf': 0.014,
                      'lf_norm': 0.014, 'hf_norm': 0.014, 'vlf_power': 0.014,
                      'mean_nn': 0.014}

# Zakres fizjologiczny i prog odchylenia od mediany sasiedztwa. Regula jest deklaracja
# tej pracy, nie cytatem: publikowane algorytmy roznia sie szczegolami, a Lipponen
# i Tarvainen 2019 [6] podaja, ze poprawnie zwalidowana korekta zostawia blad ponizej
# dwoch procent.
PHYSIOLOGICAL_RANGE_MS = (300.0, 2000.0)
RELATIVE_DEVIATION = 0.20

# Shaffer & Ginsberg 2017 [11].
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
RESAMPLE_HZ = 4.0


def rr_intervals(peaks: Sequence[int], fs: float) -> np.ndarray:
    """Interbeat intervals in milliseconds, from beat positions in samples."""
    peaks = np.asarray(peaks, dtype=np.float64).ravel()
    if peaks.size < 2:
        raise ValueError(f'at least two beats are needed for an interval, got {peaks.size}')
    if np.any(np.diff(peaks) <= 0):
        raise ValueError('beat positions must be strictly increasing')
    return np.diff(peaks) / float(fs) * 1000.0


def artefact_mask(rr: np.ndarray, physiological=PHYSIOLOGICAL_RANGE_MS,
                  deviation: float = RELATIVE_DEVIATION) -> np.ndarray:
    """
    Which intervals are implausible, either in themselves or against their neighbours.

    Two rules. An interval outside the physiological range cannot be a normal beat. An
    interval departing from the median of the surrounding ones by more than `deviation`
    is a missed or an extra detection, since a heart rate does not change that fast
    between consecutive beats.

    The rule is a declared convention rather than a citation: published algorithms differ
    in their details, and a properly validated correction leaves a mean error below two
    percent (Lipponen & Tarvainen 2019 [6]_).
    """
    rr = np.asarray(rr, dtype=np.float64).ravel()
    if rr.size == 0:
        return np.zeros(0, dtype=bool)

    low, high = float(physiological[0]), float(physiological[1])
    implausible = (rr < low) | (rr > high)

    if rr.size < 3:
        return implausible

    from scipy.ndimage import median_filter

    local = median_filter(rr, size=5, mode='nearest')
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.abs(rr - local) / np.where(local > 0, local, np.nan)
    return implausible | (np.nan_to_num(relative, nan=0.0) > float(deviation))


def artefact_fraction(rr: np.ndarray, **kwargs) -> float:
    """
    Share of the intervals judged corrupted.

    Reported next to every index, because the boundaries above which an index is biased
    are of the order of one percent and a result without this number cannot be read.
    """
    rr = np.asarray(rr, dtype=np.float64).ravel()
    return 0.0 if rr.size == 0 else float(artefact_mask(rr, **kwargs).mean())


def correct_artefacts(rr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Replaces corrupted intervals by interpolation between the ones that stand.

    Mild editing is safer than leaving artefacts in place: corrections under thirty
    milliseconds did not alter the indices materially, while a single uncorrected artefact
    dominated them (Bourdillon et al. 2022 [2]_).
    """
    rr = np.asarray(rr, dtype=np.float64).ravel().copy()
    bad = artefact_mask(rr, **kwargs)
    if not bad.any() or bad.all():
        return rr

    index = np.arange(rr.size)
    rr[bad] = np.interp(index[bad], index[~bad], rr[~bad])
    return rr


def _time_domain(rr: np.ndarray) -> dict:
    differences = np.diff(rr)
    return {
        'mean_nn': float(np.mean(rr)),
        'sdnn': float(np.std(rr, ddof=1)) if rr.size > 1 else float('nan'),
        'rmssd': float(np.sqrt(np.mean(differences ** 2))) if differences.size else float('nan'),
        'pnn50': float(np.mean(np.abs(differences) > 50.0)) if differences.size else float('nan'),
    }


def _frequency_domain(rr: np.ndarray, resample_hz: float = RESAMPLE_HZ) -> dict:
    """Spectral powers of the interval series, resampled onto an even grid."""
    from scipy.signal import welch

    times = np.cumsum(rr) / 1000.0
    times = times - times[0]
    if times[-1] <= 0:
        return {field: float('nan') for field in FREQUENCY_FIELDS}

    grid = np.arange(0.0, times[-1], 1.0 / resample_hz)
    if grid.size < 16:
        return {field: float('nan') for field in FREQUENCY_FIELDS}

    series = np.interp(grid, times, rr)
    series = series - series.mean()

    nperseg = min(series.size, 256)
    frequency, power = welch(series, fs=resample_hz, nperseg=nperseg)

    def band(limits):
        inside = (frequency >= limits[0]) & (frequency < limits[1])
        return float(np.trapezoid(power[inside], frequency[inside])) if inside.any() else 0.0

    vlf, lf, hf = band(VLF_BAND), band(LF_BAND), band(HF_BAND)
    total = lf + hf
    return {
        'vlf_power': vlf,
        'lf_power': lf,
        'hf_power': hf,
        'lf_hf': lf / hf if hf > 0 else float('nan'),
        'lf_norm': lf / total if total > 0 else float('nan'),
        'hf_norm': hf / total if total > 0 else float('nan'),
    }


def hrv_panel(peaks: Sequence[int], fs: float, correct: bool = True,
              minimum_duration=MINIMUM_DURATION_S,
              tolerance=ARTEFACT_TOLERANCE) -> dict:
    """
    Every index a recording is long enough and clean enough to support.

    An index whose recording is shorter than its own minimum is returned as not a number
    rather than computed, since values from different durations are not interchangeable.
    An index whose artefact burden exceeds its own tolerance is computed but marked, and
    the burden is returned alongside so that no value can be read without it.

    Takes the beats of a whole recording, never of a window: a window of 4096 samples at
    360 Hz spans 11.4 seconds, which supports one time domain index and no spectral one.
    """
    rr_raw = rr_intervals(peaks, fs)
    burden = artefact_fraction(rr_raw)
    rr = correct_artefacts(rr_raw) if correct else rr_raw

    duration = float(np.sum(rr) / 1000.0)
    values = {**_time_domain(rr), **_frequency_domain(rr)}

    admissible, exceeded = {}, {}
    for field in HRV_FIELDS:
        long_enough = duration >= float(minimum_duration.get(field, 0.0))
        admissible[field] = bool(long_enough)
        exceeded[field] = bool(burden > float(tolerance.get(field, 1.0)))
        if not long_enough:
            values[field] = float('nan')

    if duration < REFERENCE_DURATION_S:
        warnings.warn(
            f'{duration:.1f} s of intervals against the {REFERENCE_DURATION_S:.0f} s '
            f'reference for short term analysis; values from different durations are not '
            f'interchangeable', stacklevel=2)

    return {
        **values,
        'artefact_fraction': burden,
        'n_intervals': int(rr.size),
        'duration_s': duration,
        'admissible': admissible,
        'tolerance_exceeded': exceeded,
        'corrected': bool(correct),
    }


def hrv_agreement(reference_peaks: Sequence[int], estimate_peaks: Sequence[int],
                  fs: float, fields=TIME_FIELDS, **kwargs) -> dict:
    """
    How far a method moved the variability away from what the reference beats give.

    This is the safety check the module exists for. A filter that raises the signal to
    noise ratio while shifting the complexes will show a large relative departure here,
    and a filter that leaves the beats where they were will show none, which is a
    distinction no waveform metric makes.

    References
    ----------
    Eguchi & Aoki 2023 [5]_; Sauerbier et al. 2024 [14]_.
    """
    reference = hrv_panel(reference_peaks, fs, **kwargs)
    estimate = hrv_panel(estimate_peaks, fs, **kwargs)

    departures = {}
    for field in fields:
        first, second = reference[field], estimate[field]
        if not (np.isfinite(first) and np.isfinite(second)) or first == 0.0:
            departures[field] = float('nan')
        else:
            departures[field] = float(abs(second - first) / abs(first))

    return {
        'reference': reference,
        'estimate': estimate,
        'relative_departure': departures,
        'worst_departure': float(np.nanmax(list(departures.values()))
                                 if departures else float('nan')),
    }


def format_panel(panel: dict) -> str:
    """Readable summary of one recording, for the console and for the thesis appendix."""
    lines = [
        f"interwaly {panel['n_intervals']}, czas {panel['duration_s']:.1f} s, "
        f"artefakty {100 * panel['artefact_fraction']:.2f} %"
        f"{' (skorygowane)' if panel['corrected'] else ''}",
        '',
        '%-12s %14s %10s %10s' % ('wskaznik', 'wartosc', 'dlugosc', 'artefakty'),
        '-' * 50,
    ]
    for field in HRV_FIELDS:
        value = panel[field]
        lines.append('%-12s %14s %10s %10s' % (
            field,
            '-' if not np.isfinite(value) else f'{value:.4f}',
            'ok' if panel['admissible'][field] else 'za krotko',
            'przekr.' if panel['tolerance_exceeded'][field] else 'ok'))
    return '\n'.join(lines)
