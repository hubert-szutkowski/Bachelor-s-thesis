"""
Checks that the evaluation measures what it claims to, before it is trusted with results.

An error in a metric moves every method in the same direction, so no comparison between
methods reveals it. The only way to catch one is to measure something whose answer is
known in advance, and this module holds three such measurements.

**A gain that can be derived on paper.** White noise spread evenly over the band up to
half the sampling frequency, passed through a low pass filter with a cutoff below that,
keeps only the fraction of its power that fits inside the passband. If the signal lies
entirely inside the passband too, the ratio improves by exactly

.. math:: \\Delta\\mathrm{SNR} = 10 \\log_{10} \\frac{f_s / 2}{f_c}

with nothing measured and nothing fitted. A pipeline that returns a different number has
a fault in the metric, in the filter or in the mixing, and the three can be separated by
which of the checks fails. Starting from a clean recording, adding synthetic noise at
preset ratios and confirming that the metrics move as expected is the standard validation
pattern in this literature (Jenkal et al. 2023 [7]_; Hesar & Hesar 2024 [5]_; De Fazio et
al. 2026 [4]_; Ali et al. 2023 [2]_).

**Agreement between metrics rather than trust in one.** Reviews evaluate denoisers on the
ratio, the squared error, the difference and the correlation together, because a single
score can rank methods wrongly: one comparison found one method better on the root mean
square error and another better on the improvement in the ratio, on the same material
(Shi et al. 2021 [12]_; Chatterjee et al. 2020 [3]_; Sarafan et al. 2022 [11]_).
`metric_concordance` measures how far the metrics agree on an ordering and names the pairs
that do not, so that a disagreement is reported rather than resolved by whichever metric
was put in the first column.

**Two control methods.** Scoring the unfiltered input against the clean signal is common
practice and anchors every other number (Venkata et al. 2026 [14]_; Shi et al. 2021 [12]_).
A deliberately destructive filter is not: published work reports methods that damaged the
morphology, but as outcomes of a comparison rather than as controls inserted on purpose
(Sarafan et al. 2022 [11]_; Liu et al. 2021 [8]_). Including one here is therefore a step
beyond typical practice and is justified as a falsification control: the quality indices of
`analysis.metrics_sqi` can be raised by a filter that removes signal, so a panel that ranks
the destructive control highly has been shown to be unusable rather than merely suspected
of it.

References
----------
.. [1] Ahmad, A., Lais, M., Awan, D., Zia, M., Khan, M. U., & Farooq, E. (2025). ECG
       Denoising Using Wavelet Transform and Wiener Filter. Physical Education, Health and
       Social Sciences. https://doi.org/10.63163/jpehss.v3i3.685
.. [2] Ali, M., Ali, S., & Khorsheed, A. (2023). ECG Signal Denoising Using Discrete
       Wavelet Transform. Journal of Duhok University.
       https://doi.org/10.26682/csjuod.2023.26.2.42
.. [3] Chatterjee, S., Thakur, R. S., Yadav, R. N., Gupta, L., & Raghuvanshi, D. K. (2020).
       Review of noise removal techniques in ECG signals. IET Signal Processing, 14(9),
       569-590. https://doi.org/10.1049/iet-spr.2020.0104
.. [4] De Fazio, R., Al-Naami, B., Rawash, Y., Al-Hinnawi, A., Al-Zaben, A., & Visconti, P.
       (2026). A novel stretched-compressed exponential low-pass filter and its application
       to electrocardiogram signal denoising. IJECE.
       https://doi.org/10.11591/ijece.v16i1.pp230-245
.. [5] Hesar, H. D., & Hesar, A. D. (2024). Adaptive augmented cubature Kalman
       filter/smoother for ECG denoising. Biomedical Engineering Letters, 14, 689-705.
       https://doi.org/10.1007/s13534-024-00362-7
.. [6] Hesar, H. D., & Mohebbi, M. (2017). ECG Denoising Using Marginalized Particle
       Extended Kalman Filter With an Automatic Particle Weighting Strategy. IEEE Journal
       of Biomedical and Health Informatics, 21, 635-644.
       https://doi.org/10.1109/JBHI.2016.2582340
.. [7] Jenkal, W., Latif, R., & Laaboubi, M. (2023). ECG Signal Denoising Using an Improved
       Hybrid DWT-ADTF Approach. Cardiovascular Engineering and Technology, 15, 77-94.
       https://doi.org/10.1007/s13239-023-00698-8
.. [8] Liu, R.-X., Shu, M., & Chen, C. (2021). ECG Signal Denoising and Reconstruction
       Based on Basis Pursuit. Applied Sciences. https://doi.org/10.3390/app11041591
.. [9] Malghan, P. G., & Hota, M. K. (2020). A review on ECG filtering techniques for
       rhythm analysis. Research on Biomedical Engineering, 36, 171-186.
       https://doi.org/10.1007/s42600-020-00057-9
.. [10] Mohguen, W., & Bouguezel, S. (2021). Denoising the ECG Signal Using Ensemble
        Empirical Mode Decomposition. Engineering, Technology & Applied Science Research.
        https://doi.org/10.48084/etasr.4302
.. [11] Sarafan, S., Vuong, H. T., Jilani, D., Malhotra, S., Lau, M. P. H.,
        Vishwanath, M., Ghirmai, T., & Cao, H. (2022). A Novel ECG Denoising Scheme Using
        the Ensemble Kalman Filter. EMBC, 2005-2008.
        https://doi.org/10.1109/EMBC48229.2022.9871884
.. [12] Shi, H., Liu, R.-X., Chen, C., Shu, M., & Wang, Y. (2021). ECG Baseline Estimation
        and Denoising With Group Sparse Regularization. IEEE Access, 9, 23595-23607.
        https://doi.org/10.1109/ACCESS.2021.3056459
.. [13] Sraitih, M., & Jabrane, Y. (2021). A denoising performance comparison based on ECG
        Signal Decomposition and local means filtering. Biomedical Signal Processing and
        Control, 69, 102903. https://doi.org/10.1016/j.bspc.2021.102903
.. [14] Venkata, K., Reddy, S., & Balaji, M. (2026). Electrocardiogram signal denoising and
        heart disease classification. Bulletin of Electrical Engineering and Informatics.
        https://doi.org/10.11591/eei.v15i3.11276
"""

import math
from typing import Optional, Sequence

import numpy as np

try:
    from analysis.metrics_reference import correlation, prd, snr
except ImportError:
    from .metrics_reference import correlation, prd, snr

# Filtr rzedu skonczonego przepuszcza czesc mocy poza pasmem, wiec zmierzony zysk jest
# nieco nizszy od analitycznego. Prog dobrany na podstawie pomiaru dla Butterwortha
# rzedu 8, ktory odtwarza przewidywanie z dokladnoscia ponizej 0.5 dB.
ANALYTIC_TOLERANCE_DB = 1.0

# Ponizej tej zgodnosci rangowej miedzy metrykami wynik jest oznaczany jako sporny.
# Shi et al. 2021 [12] pokazuja przypadek, w ktorym dwie metryki daja odwrotny ranking.
CONCORDANCE_WARNING = 0.7


def analytic_lowpass_gain_db(cutoff_hz: float, fs: float) -> float:
    """
    Improvement a low pass filter must give on white noise, derived rather than measured.

    White noise spread evenly to half the sampling frequency keeps the fraction of its
    power that fits below the cutoff. A signal lying entirely inside the passband is
    unchanged, so the ratio improves by the reciprocal of that fraction. Nothing here is
    fitted, which is what makes it a check.
    """
    nyquist = float(fs) / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(f'the cutoff must lie between zero and {nyquist} Hz, '
                         f'got {cutoff_hz}')
    return 10.0 * math.log10(nyquist / float(cutoff_hz))


def check_lowpass_gain(fs: float, cutoff_hz: float = 40.0, order: int = 8,
                       n_samples: int = 200000, sigma: float = 0.5,
                       tolerance_db: float = ANALYTIC_TOLERANCE_DB,
                       seed: int = 0) -> dict:
    """
    Measures the gain on a case whose answer is known and compares it with the derivation.

    The signal is a slow tone well inside the passband, so the filter cannot damage it and
    every decibel of the result comes from the noise it removed. A departure larger than
    `tolerance_db` means a fault in the metric, in the filter or in the mixing, and which
    of the three it is follows from which of the other checks also fails.
    """
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    time = np.arange(n_samples) / fs
    clean = np.sin(2.0 * np.pi * 1.0 * time)
    noisy = clean + sigma * rng.standard_normal(n_samples)

    sos = butter(order, cutoff_hz, btype='lowpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, noisy)

    predicted = analytic_lowpass_gain_db(cutoff_hz, fs)
    measured = snr(clean, filtered) - snr(clean, noisy)

    return {
        'predicted_db': predicted,
        'measured_db': float(measured),
        'error_db': float(measured - predicted),
        'tolerance_db': float(tolerance_db),
        'passed': bool(abs(measured - predicted) <= tolerance_db),
        'cutoff_hz': float(cutoff_hz),
        'order': int(order),
    }


def check_monotonicity(clean: np.ndarray, noise: np.ndarray, fs: float,
                       levels_db: Sequence[float] = (-9.0, -5.0, -1.0, 3.0, 7.0, 11.0)
                       ) -> dict:
    """
    Confirms that every metric moves the way it must as the noise is turned down.

    The ratio has to rise, the difference and the errors have to fall, the correlation has
    to rise. A metric that moves the other way is inverted somewhere, and the ordering is
    the only thing being tested here, not any particular value.
    """
    try:
        from preparing.noise_mixing import mix_at_snr
    except ImportError:
        from ..preparing.noise_mixing import mix_at_snr

    levels = sorted(levels_db)
    ratios, differences, correlations = [], [], []
    for level in levels:
        noisy, _ = mix_at_snr(clean, noise, None, fs, level)
        ratios.append(snr(clean, noisy))
        differences.append(prd(clean, noisy))
        correlations.append(correlation(clean, noisy))

    rising = ratios == sorted(ratios)
    falling = differences == sorted(differences, reverse=True)
    improving = correlations == sorted(correlations)

    return {
        'levels_db': levels,
        'snr': ratios,
        'prd': differences,
        'correlation': correlations,
        'snr_rises': bool(rising),
        'prd_falls': bool(falling),
        'correlation_rises': bool(improving),
        'passed': bool(rising and falling and improving),
    }


def metric_concordance(scores: dict, higher_is_better: Optional[dict] = None) -> dict:
    """
    How far the metrics agree on an ordering of the methods.

    Shi et al. 2021 report a case where one method wins on the root mean square error and
    another on the improvement in the ratio, over the same material, which is why a single
    metric cannot be trusted to rank. Pairs falling below `CONCORDANCE_WARNING` are named
    so that a disagreement is reported rather than settled by whichever metric happened to
    be put first in the table.

    `scores` maps a metric to a mapping of method to value.
    """
    from scipy.stats import spearmanr

    defaults = {'snr_out': True, 'snr_improvement': True, 'correlation': True,
                'prd': False, 'prd1': False, 'mse': False, 'rmse': False, 'mae': False}
    direction = {**defaults, **(higher_is_better or {})}

    metrics = sorted(scores)
    if len(metrics) < 2:
        raise ValueError('at least two metrics are needed for a concordance')

    methods = sorted(set.intersection(*(set(scores[metric]) for metric in metrics)))
    if len(methods) < 3:
        raise ValueError(f'at least three methods are needed to compare orderings, '
                         f'got {len(methods)}')

    ranked = {}
    for metric in metrics:
        values = np.array([scores[metric][method] for method in methods], dtype=np.float64)
        # kazda metryka sprowadzona do wspolnego kierunku, zeby korelacja mierzyla
        # zgodnosc uporzadkowania, a nie przypadkowy znak definicji
        ranked[metric] = values if direction.get(metric, True) else -values

    pairs, disputed = {}, []
    for first in range(len(metrics)):
        for second in range(first + 1, len(metrics)):
            a, b = metrics[first], metrics[second]
            usable = np.isfinite(ranked[a]) & np.isfinite(ranked[b])
            if usable.sum() < 3:
                value = float('nan')
            else:
                value = float(spearmanr(ranked[a][usable], ranked[b][usable]).statistic)
            pairs[(a, b)] = value
            if np.isfinite(value) and value < CONCORDANCE_WARNING:
                disputed.append({'metrics': (a, b), 'correlation': value})

    finite = [value for value in pairs.values() if np.isfinite(value)]
    return {
        'methods': methods,
        'metrics': metrics,
        'pairs': pairs,
        'disputed': disputed,
        'worst': float(min(finite)) if finite else float('nan'),
        'passed': not disputed,
        'threshold': CONCORDANCE_WARNING,
    }


def check_controls(clean: np.ndarray, noisy: np.ndarray, identity: np.ndarray,
                   destructive: np.ndarray, fs: float) -> dict:
    """
    Confirms that the two control methods behave as controls must.

    The unfiltered one has to score exactly what the input scores, since it is the input;
    anything else means the pipeline is altering a waveform it was told to pass through.
    The destructive one has to lose on the reference metrics, and if it wins on a quality
    index then that index has been shown to reward the removal of signal rather than the
    removal of noise.
    """
    try:
        from analysis.metrics_sqi import psqi
    except ImportError:
        from .metrics_sqi import psqi

    input_ratio = snr(clean, noisy)
    identity_ratio = snr(clean, identity)
    destructive_ratio = snr(clean, destructive)

    return {
        'input_snr_db': float(input_ratio),
        'identity_snr_db': float(identity_ratio),
        'destructive_snr_db': float(destructive_ratio),
        'identity_matches_input': bool(np.allclose(identity, noisy)),
        'destructive_loses_on_reference': bool(destructive_ratio < input_ratio + 1e-9
                                               or destructive_ratio < identity_ratio),
        'input_psqi': float(psqi(noisy, fs)),
        'destructive_psqi': float(psqi(destructive, fs)),
        'destructive_wins_on_quality': bool(psqi(destructive, fs) > psqi(noisy, fs)),
    }


def run_sanity_suite(fs: float = 360.0, **kwargs) -> dict:
    """
    Every check that needs no data of its own, in one call.

    Run before a results table is produced and again whenever a metric is changed. The
    report is a record: a table produced after a failed suite has no standing.
    """
    rng = np.random.default_rng(0)
    n = 4096
    clean = np.zeros(n)
    for peak in range(int(fs), n - int(fs), 288):
        clean[peak - 4:peak + 4] += np.hanning(8) * 1.2
        clean[peak + 40:peak + 98] += np.hanning(58) * 0.25
    noise = rng.standard_normal(n)

    checks = {
        'analytic_lowpass_gain': check_lowpass_gain(fs, **kwargs),
        'metric_monotonicity': check_monotonicity(clean, noise, fs),
    }
    return {'checks': checks,
            'passed': all(check['passed'] for check in checks.values())}


def format_suite(report: dict) -> str:
    """Readable summary of a suite, for the console and for the thesis appendix."""
    lines = ['%-26s %10s' % ('sprawdzenie', 'wynik'), '-' * 40]
    for name, check in report['checks'].items():
        lines.append('%-26s %10s' % (name, 'ok' if check['passed'] else 'BLAD'))

    gain = report['checks']['analytic_lowpass_gain']
    lines += ['',
              f"zysk analityczny {gain['predicted_db']:.3f} dB, "
              f"zmierzony {gain['measured_db']:.3f} dB, "
              f"blad {gain['error_db']:+.3f} dB"]

    monotone = report['checks']['metric_monotonicity']
    lines.append(f"monotonicznosc: SNR rosnie {monotone['snr_rises']}, "
                 f"PRD maleje {monotone['prd_falls']}, "
                 f"korelacja rosnie {monotone['correlation_rises']}")
    return '\n'.join(lines)
