"""
Comparison of methods, at the level where the observations are independent.

The whole of this module exists because of one fact: windows cut from one recording are
not independent observations. They share a patient, an electrode placement, a heart and
often the same beats, and with an overlap they literally share samples. Treating a hundred
thousand of them as a sample would produce significance at any threshold for differences
of no consequence, because the effective size of the sample is the number of patients.

Methodology work names this directly. Non-independence is flagged as a technical problem
for the calculation of effect sizes and confidence intervals (Nakagawa & Cuthill 2007 [7]_;
Carrara & Papadopoulo 2023 [4]_), studies using overlapping windows compare methods across
subjects rather than across windows (Carrara & Papadopoulo 2023 [4]_; Ashraf et al. 2020
[1]_), and the length and overlap of the window have been shown to change variability and
significance while leaving the effect size largely alone (Burden et al. 2014 [3]_). Window
level metrics are therefore aggregated within each patient before any test is run, and
`aggregate_by_patient` is the only entry into the rest of this module.

The design that follows is repeated measures with seventeen methods on the same
recordings, so the workflow is an omnibus test, then paired contrasts, then a correction
for their number.

    Friedman across all methods           repeated measures, no normality assumed [8]_
    paired Wilcoxon for each pair         same recordings, paired contrasts [8]_
    Holm correction over the pairs        more powerful than Bonferroni at equal
                                          familywise control [6]_ [2]_

Holm rather than Bonferroni because Bonferroni is the more conservative of the two at the
same guarantee, and rather than Benjamini-Hochberg because that controls the false
discovery rate instead of the familywise error and belongs to an exploratory panel rather
than to the main table (Lesack & Naugler 2011 [6]_; Keselman et al. 1999 [5]_).

Effect sizes and intervals are reported for every primary comparison, not only p values,
which convey neither magnitude nor precision (Schober et al. 2018 [10]_; Nakagawa & Cuthill
2007 [7]_; Williams et al. 2023 [12]_). This matters more here than usual: with five held
out patients a hypothesis test has very little power, and an effect with a wide interval
is a more honest result than a p value that failed to cross a threshold.

One convention is declared rather than cited. The literature surveyed supports reporting
intervals but does not settle whether an interval for a quantity in decibels should be
computed on the decibel scale or on the linear one and then transformed. Since the
logarithm is not linear the two differ, and `confidence_interval` therefore takes the scale
as an argument, defaults to the linear route for ratio metrics and records which was used.

References
----------
.. [1] Ashraf, H., Waris, A., Gilani, S. O., Kashif, A. A., Jamil, M., Jochumsen, M., &
       Niazi, I. K. (2020). Evaluation of windowing techniques for intramuscular EMG-based
       diagnostic, rehabilitative and assistive devices. Journal of Neural Engineering, 18.
       https://doi.org/10.1088/1741-2552/abcc7f
.. [2] Avci, H., & Dag, O. (2024). A Comprehensive Monte Carlo Simulation Study on
       Multiple Comparison Methods after ANOVA. Journal of Advanced Research in Natural
       and Applied Sciences. https://doi.org/10.28979/jarnas.1429315
.. [3] Burden, A. M., Lewis, S. E., & Willcox, E. (2014). The effect of manipulating root
       mean square window length and overlap on reliability, inter-individual variability,
       statistical significance and clinical relevance of electromyograms. Manual Therapy,
       19(6), 595-601. https://doi.org/10.1016/j.math.2014.06.003
.. [4] Carrara, I., & Papadopoulo, T. (2023). Pseudo-online framework for BCI evaluation:
       a MOABB perspective. Journal of Neural Engineering, 21.
       https://doi.org/10.1088/1741-2552/ad171a
.. [5] Keselman, H. J., Cribbie, R. A., & Holland, B. (1999). The pairwise multiple
       comparison multiplicity problem. Psychological Methods, 4(1), 58-69.
       https://doi.org/10.1037/1082-989X.4.1.58
.. [6] Lesack, K., & Naugler, C. (2011). An open-source software program for performing
       Bonferroni and related corrections for multiple comparisons. Journal of Pathology
       Informatics, 2. https://doi.org/10.4103/2153-3539.91130
.. [7] Nakagawa, S., & Cuthill, I. C. (2007). Effect size, confidence interval and
       statistical significance: a practical guide for biologists. Biological Reviews, 82.
       https://doi.org/10.1111/j.1469-185X.2007.00027.x
.. [8] Peres, F. F. (2025). Effect sizes for nonparametric tests. Biochemia Medica, 36.
       https://doi.org/10.11613/BM.2026.010101
.. [9] Narkevich, A. N., Vinogradov, K. A., & Grjibovski, A. M. (2020). Multiple
       comparisons in biomedical research: the problem and its solutions. Human Ecology,
       55-64. https://doi.org/10.33396/1728-0869-2020-10-55-64
.. [10] Schober, P., Bossers, S. M., & Schwarte, L. A. (2018). Statistical Significance
       Versus Clinical Importance of Observed Effect Sizes. Anesthesia & Analgesia, 126(3),
       1068-1072. https://doi.org/10.1213/ANE.0000000000002798
.. [11] Lee, D. K. (2016). Alternatives to P value: confidence interval and effect size.
       Korean Journal of Anesthesiology, 69(6), 555-562.
       https://doi.org/10.4097/kjae.2016.69.6.555
.. [12] Williams, S., Carson, R. G., & Toth, K. (2023). Moving beyond P values in The
       Journal of Physiology: a primer on the value of effect sizes and confidence
       intervals. The Journal of Physiology, 601.
       https://doi.org/10.1113/JP285575
"""

import warnings
from typing import Optional, Sequence

import numpy as np

# Ponizej tej liczby pacjentow test hipotezy praktycznie nie ma mocy. Nie jest to prog
# odrzucenia, tylko granica, przy ktorej modul ostrzega, ze rozmiar efektu i przedzial
# ufnosci sa jedynym sensownym wynikiem.
MIN_PATIENTS_FOR_TESTING = 8

DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP = 10000


def aggregate_by_patient(rows, metric: str, method_key: str = 'method',
                         patient_key: str = 'patient', statistic: str = 'mean') -> dict:
    """
    Window level values collapsed to one value per method and patient.

    The only way into the tests below. A test run over windows would treat correlated
    observations as independent and report significance for differences of no consequence,
    since the effective size of the sample is the number of patients, not of windows.

    References
    ----------
    Nakagawa & Cuthill 2007 [7]_; Carrara & Papadopoulo 2023 [4]_; Burden et al. 2014 [3]_.
    """
    if statistic not in ('mean', 'median'):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")

    collected: dict = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        collected.setdefault(row[method_key], {}).setdefault(row[patient_key], []).append(
            float(value))

    reduce = np.mean if statistic == 'mean' else np.median
    return {method: {patient: float(reduce(values)) for patient, values in patients.items()}
            for method, patients in collected.items()}


def paired_matrix(aggregated: dict, methods: Optional[Sequence[str]] = None) -> tuple:
    """
    The aggregate as a matrix of patients by methods, keeping only complete cases.

    A repeated measures test needs every method measured on every patient. A patient a
    method failed on is dropped for all of them, so the comparison stays paired, and the
    count of what was dropped is returned rather than left implicit.
    """
    methods = sorted(aggregated) if methods is None else list(methods)
    if not methods:
        raise ValueError('no method to compare')

    patients = set.intersection(*(set(aggregated[method]) for method in methods))
    complete = sorted(patients)
    if not complete:
        raise ValueError('no patient carries a value for every method')

    everyone = set().union(*(set(aggregated[method]) for method in methods))
    matrix = np.array([[aggregated[method][patient] for method in methods]
                       for patient in complete], dtype=np.float64)

    return matrix, methods, complete, sorted(everyone - patients)


def friedman(matrix: np.ndarray) -> dict:
    """
    Omnibus test across every method, on the same patients.

    Repeated measures and free of any assumption of normality, which is what a handful of
    patients and a bounded metric call for. `kendalls_w` is the concordance form of the
    effect size, between zero and one, and is reported because the statistic alone says
    nothing about how large the disagreement between methods is.

    References
    ----------
    Peres 2025 [8]_ for the nonparametric effect size; Lee 2016 [11]_ for the omnibus
    before pairwise testing.
    """
    from scipy.stats import friedmanchisquare

    matrix = np.asarray(matrix, dtype=np.float64)
    n_patients, n_methods = matrix.shape
    if n_methods < 3:
        raise ValueError(f'an omnibus test needs at least three methods, got {n_methods}')
    if n_patients < 2:
        raise ValueError(f'an omnibus test needs at least two patients, got {n_patients}')

    if n_patients < MIN_PATIENTS_FOR_TESTING:
        warnings.warn(
            f'{n_patients} patients against {n_methods} methods: the test has very little '
            f'power, so read the effect sizes and intervals rather than the p value',
            stacklevel=2)

    statistic, p_value = friedmanchisquare(*matrix.T)
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'kendalls_w': float(statistic / (n_patients * (n_methods - 1))),
        'n_patients': int(n_patients),
        'n_methods': int(n_methods),
    }


def rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """
    Matched pairs rank biserial correlation, the effect size of a signed rank test.

    Runs from minus one to one and reads as the share of the signed rank mass favouring
    one method over the other; zero means the two are interchangeable on this material.

    References
    ----------
    Peres 2025 [8]_.
    """
    from scipy.stats import rankdata

    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    difference = difference[difference != 0.0]
    if difference.size == 0:
        return 0.0

    ranks = rankdata(np.abs(difference))
    positive = float(ranks[difference > 0].sum())
    negative = float(ranks[difference < 0].sum())
    total = positive + negative
    return 0.0 if total == 0.0 else (positive - negative) / total


def paired_wilcoxon(first: np.ndarray, second: np.ndarray) -> dict:
    """
    Signed rank test between two methods measured on the same patients.

    Used only after the omnibus, and only on patient level values.

    References
    ----------
    Peres 2025 [8]_.
    """
    from scipy.stats import wilcoxon

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f'unpaired samples: {first.shape} against {second.shape}')

    difference = first - second
    if np.all(difference == 0.0):
        return {'statistic': 0.0, 'p_value': 1.0, 'rank_biserial': 0.0,
                'median_difference': 0.0, 'n': int(first.size)}

    statistic, p_value = wilcoxon(first, second)
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'rank_biserial': rank_biserial(first, second),
        'median_difference': float(np.median(difference)),
        'n': int(first.size),
    }


def holm(p_values: Sequence[float], alpha: float = DEFAULT_ALPHA) -> dict:
    """
    Holm correction over a family of comparisons.

    Controls the familywise error rate like Bonferroni does but rejects more, since each
    p value is weighted by how many comparisons remain rather than by how many there were.
    Preferred here over Benjamini-Hochberg, which controls the false discovery rate and
    belongs to an exploratory panel rather than to the main table.

    References
    ----------
    Lesack & Naugler 2011 [6]_; Avci & Dag 2024 [2]_; Keselman et al. 1999 [5]_;
    Narkevich et al. 2020 [9]_.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.size == 0:
        return {'adjusted': np.empty(0), 'rejected': np.empty(0, dtype=bool),
                'alpha': float(alpha), 'method': 'holm'}
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError('p values must lie in [0, 1]')

    order = np.argsort(p_values)
    count = p_values.size

    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (count - position) * p_values[index])
        adjusted[index] = min(1.0, running)

    return {
        'adjusted': adjusted,
        'rejected': adjusted <= alpha,
        'alpha': float(alpha),
        'method': 'holm',
    }


def confidence_interval(values: Sequence[float], alpha: float = DEFAULT_ALPHA,
                        scale: str = 'linear', n_bootstrap: int = DEFAULT_BOOTSTRAP,
                        seed: int = 0) -> dict:
    """
    Percentile bootstrap interval for the mean, on a stated scale.

    `scale='decibel'` converts each value out of decibels into the power ratio it stands
    for, forms the interval there and converts the result back. That is not the same as an
    interval taken on the decibel values directly, because the logarithm is not linear, and
    the surveyed literature does not settle which is correct. The route is therefore an
    argument, the linear one is the default for ratio quantities, and the choice travels
    with the result instead of being assumed.

    References
    ----------
    Nakagawa & Cuthill 2007 [7]_; Schober et al. 2018 [10]_; Williams et al. 2023 [12]_.
    """
    if scale not in ('linear', 'decibel'):
        raise ValueError(f"scale must be 'linear' or 'decibel', got {scale!r}")

    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {'mean': float(values[0]) if values.size else float('nan'),
                'low': float('nan'), 'high': float('nan'),
                'n': int(values.size), 'scale': scale}

    working = 10.0 ** (values / 10.0) if scale == 'decibel' else values

    rng = np.random.default_rng(seed)
    draws = rng.choice(working, size=(int(n_bootstrap), working.size), replace=True)
    means = draws.mean(axis=1)

    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    centre = float(working.mean())

    if scale == 'decibel':
        centre, low, high = (10.0 * np.log10(value) for value in (centre, low, high))

    return {'mean': float(centre), 'low': float(low), 'high': float(high),
            'n': int(values.size), 'scale': scale}


def compare_methods(rows, metric: str, alpha: float = DEFAULT_ALPHA,
                    scale: str = 'linear', methods: Optional[Sequence[str]] = None,
                    statistic: str = 'mean') -> dict:
    """
    The whole workflow, from window level rows to a corrected table of contrasts.

    Aggregates by patient, runs the omnibus, then every pairwise contrast with its effect
    size, then corrects their p values together. The pairwise results are returned whether
    or not the omnibus was significant, with the omnibus alongside them, so that the
    decision of what to report stays with the person writing the chapter.
    """
    from itertools import combinations

    aggregated = aggregate_by_patient(rows, metric, statistic=statistic)
    matrix, names, patients, dropped = paired_matrix(aggregated, methods)

    omnibus = friedman(matrix)

    contrasts = []
    for first, second in combinations(range(len(names)), 2):
        outcome = paired_wilcoxon(matrix[:, first], matrix[:, second])
        outcome['first'] = names[first]
        outcome['second'] = names[second]
        contrasts.append(outcome)

    correction = holm([contrast['p_value'] for contrast in contrasts], alpha=alpha)
    for contrast, adjusted, rejected in zip(contrasts, correction['adjusted'],
                                            correction['rejected']):
        contrast['p_adjusted'] = float(adjusted)
        contrast['significant'] = bool(rejected)

    intervals = {name: confidence_interval(matrix[:, index], alpha=alpha, scale=scale)
                 for index, name in enumerate(names)}

    return {
        'metric': metric,
        'methods': names,
        'patients': patients,
        'dropped_patients': dropped,
        'per_patient': matrix,
        'omnibus': omnibus,
        'contrasts': contrasts,
        'intervals': intervals,
        'correction': correction['method'],
        'alpha': float(alpha),
    }


def format_comparison(comparison: dict, top: int = 10) -> str:
    """Readable summary of a comparison, for the console and for the thesis appendix."""
    omnibus = comparison['omnibus']
    lines = [
        f"metryka {comparison['metric']} | {omnibus['n_methods']} metod, "
        f"{omnibus['n_patients']} pacjentow",
        f"Friedman chi2 = {omnibus['statistic']:.3f}, p = {omnibus['p_value']:.4g}, "
        f"W Kendalla = {omnibus['kendalls_w']:.3f}",
        '',
        '%-18s %10s %10s %10s %8s' % ('metoda', 'srednia', 'dolny', 'gorny', 'n'),
        '-' * 60,
    ]
    for name in comparison['methods']:
        interval = comparison['intervals'][name]
        lines.append('%-18s %10.4f %10.4f %10.4f %8d' % (
            name, interval['mean'], interval['low'], interval['high'], interval['n']))

    lines += ['', f"kontrasty parami, korekta {comparison['correction']}, "
                  f"alpha = {comparison['alpha']}",
              '%-18s %-18s %10s %10s %8s' % ('metoda A', 'metoda B', 'p', 'p popr.', 'r'),
              '-' * 68]
    ranked = sorted(comparison['contrasts'], key=lambda item: item['p_adjusted'])
    for contrast in ranked[:top]:
        lines.append('%-18s %-18s %10.4g %10.4g %+8.3f%s' % (
            contrast['first'], contrast['second'], contrast['p_value'],
            contrast['p_adjusted'], contrast['rank_biserial'],
            ' *' if contrast['significant'] else ''))

    if comparison['dropped_patients']:
        lines.append('')
        lines.append('pominieto pacjentow bez kompletu metod: '
                     + ', '.join(comparison['dropped_patients']))
    return '\n'.join(lines)
