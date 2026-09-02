"""
Detection scored against the reference annotations, at two tolerances rather than one.

The standard is settled: a detection counts as correct when it falls within 150 ms of an
annotated beat, defined by ANSI/AAMI EC38 and EC57 (Jia et al. 2020 [5]_; Fariha et al.
2020 [3]_). That window is used here and cited as such.

It is also too wide to be used alone. A tolerance of 150 ms is more than half a QRS
complex, so a detector that finds every beat but places each one eighty milliseconds off
scores as perfect, and the timing error it carries goes on to corrupt every interbeat
interval derived from it. Recent work makes the point directly: sensitivity, positive
predictivity, the detection error rate and the F measure all depend on the tolerance and a
wide fixed window masks temporal inaccuracy (Reklewski et al. 2024 [10]_; Reklewski &
Augustyniak 2025 [11]_; Porr & Macfarlane 2019 [9]_). A case study measured the size of
it: tightening the admitted jitter from 97.2 to 86.1 milliseconds moved the detection error
rate from 8.1 to 67.2 percent on the same recording [11]_.

Every score is therefore reported at both the standard window and a stricter one of 50 ms,
which is the tolerance wearable and benchmarking studies use when interbeat timing matters
(Blasing et al. 2022 [1]_; Liu et al. 2018 [7]_; Kristof et al. 2023 [6]_ use 75 ms). The
offset of the matched detections is reported alongside, since that is the quantity the wide
window hides and the one variability analysis depends on (Ruha et al. 1997 [12]_;
Weinstein et al. 2026 [13]_).

**Detection is a control, not a substitute.** It says whether the filtered signal still
supports beat based analysis, which is a downstream utility question rather than a measure
of how much noise was removed, and acceptable detection does not guarantee that the
waveform is usable for anything wider (Kristof et al. 2023 [6]_; De Kruijf et al. 2026
[2]_). It belongs in the results table next to the ratio and the difference, not in place
of them.

The reliability of the whole score depends on the ratio it was measured at, and the
boundaries are known. Above twelve decibels detection is near its ceiling; between five and
twelve it is usually good but detector dependent; between zero and five methods separate
strongly; below zero many common detectors degrade sharply; and around minus twelve some
fail outright, with electrode motion the hardest case of all (Fariha et al. 2020 [3]_;
Kim & Shin 2016 [4]_; Reklewski et al. 2024 [10]_). Pan-Tompkins falls from near perfect
at twenty-four decibels to about sixty percent sensitivity and seventy percent predictivity
at minus six [10]_.

That places the grid used in this work, from minus nine to eleven decibels, across every
one of those regions, and the noise used is electrode motion. `reliability_zone` names the
region a measurement came from so that a low score at minus nine is read as the expected
behaviour of the detector rather than as a failure of the filter that preceded it. The
same boundary applies to the architecture built on cardiac cycles, whose segmentation comes
from the same detector.

References
----------
.. [1] Blasing, D., Buder, A., Reiser, J. E., Nisser, M., Derlien, S., & Vollmer, M.
       (2022). ECG performance in simultaneous recordings of five wearable devices using a
       new morphological noise-to-signal index and Smith-Waterman-based RR interval
       comparisons. PLoS ONE, 17. https://doi.org/10.1371/journal.pone.0274994
.. [2] De Kruijf, N., De Boer, M. M., Tieleman, R., Taverne, Y., Kavousi, M., De Groot, N.,
       & Van Schie, M. S. (2026). Performance analysis of QRS detectors on real-life
       continuous rhythm monitoring surface electrocardiograms. Europace, 28.
       https://doi.org/10.1093/europace/euag105.1258
.. [3] Fariha, M. A. Z., Ikeura, R., Hayakawa, S., & Tsutsumi, S. (2020). Analysis of
       Pan-Tompkins Algorithm Performance with Noisy ECG Signals. Journal of Physics:
       Conference Series, 1532. https://doi.org/10.1088/1742-6596/1532/1/012022
.. [4] Kim, J., & Shin, H. (2016). Simple and Robust Realtime QRS Detection Algorithm
       Based on Spatiotemporal Characteristic of the QRS Complex. PLoS ONE, 11.
       https://doi.org/10.1371/journal.pone.0150144
.. [5] Jia, M., Li, F., Wu, J., Chen, Z., & Pu, Y. (2020). Robust QRS Detection Using
       High-Resolution Wavelet Packet Decomposition and Time-Attention Convolutional
       Neural Network. IEEE Access, 8, 16979-16988.
       https://doi.org/10.1109/ACCESS.2020.2967775
.. [6] Kristof, F., Kapsecker, M., Nissen, L., Brimicombe, J., Cowie, M., Ding, Z.,
       Dymond, A., Jonas, S., Linden, H., Lip, G., Williams, K., Mant, J., & Charlton, P.
       (2023). QRS detection in single-lead, telehealth electrocardiogram signals:
       Benchmarking open-source algorithms. PLOS Digital Health, 3, e0000538.
       https://doi.org/10.1371/journal.pdig.0000538
.. [7] Liu, F., Liu, C., Jiang, X., Zhang, Z., Zhang, Y., Li, J., & Wei, S. (2018).
       Performance Analysis of Ten Common QRS Detectors on Different ECG Application
       Cases. Journal of Healthcare Engineering, 2018.
       https://doi.org/10.1155/2018/9050812
.. [8] Sharma, T., & Sharma, K. K. (2017). QRS complex detection in ECG signals using
       locally adaptive weighted total variation denoising. Computers in Biology and
       Medicine, 87, 187-199. https://doi.org/10.1016/j.compbiomed.2017.05.027
.. [9] Porr, B., & Macfarlane, P. (2019). A new QRS detector stress test combining
       temporal jitter and F-score (JF) reveals significant performance differences amongst
       popular detectors. PLOS ONE. https://doi.org/10.1371/journal.pone.0309739
.. [10] Reklewski, W., Miskowicz, M., & Augustyniak, P. (2024). QRS Detector Performance
        Evaluation Aware of Temporal Accuracy and Presence of Noise. Sensors, 24.
        https://doi.org/10.3390/s24051698
.. [11] Reklewski, W., & Augustyniak, P. (2025). How the Level of Noise Affects Temporal
        Accuracy of a QRS Detector - Case Study. Sensors, 26.
        https://doi.org/10.3390/s26010015
.. [12] Ruha, A., Sallinen, S., & Nissila, S. (1997). A real-time microprocessor QRS
        detector system with a 1-ms timing accuracy for the measurement of ambulatory HRV.
        IEEE Transactions on Biomedical Engineering, 44(3), 159-167.
        https://doi.org/10.1109/10.554762
.. [13] Weinstein, A. J., Rodino, J., & Otero, M. (2026). Effects of electrocardiogram QRS
        detection algorithms in heart rate variability metrics. Scientific Reports, 16.
        https://doi.org/10.1038/s41598-026-49215-6
.. [14] Yan, H., Yang, Z., Gao, J., & Wang, X. (2025). QRS detection in noisy
        electrocardiogram using an adaptively regularized numerical differentiation method.
        Biomedical Signal Processing and Control, 105, 107666.
        https://doi.org/10.1016/j.bspc.2025.107666
"""

from typing import Optional, Sequence

import numpy as np

try:
    from preparing.qrs_detection import TOLERANCE_MS, detect_qrs, detection_metrics
except ImportError:
    from ..preparing.qrs_detection import TOLERANCE_MS, detect_qrs, detection_metrics

# ANSI/AAMI EC38 i EC57 [5], [3].
STANDARD_TOLERANCE_MS = TOLERANCE_MS

# Okno stosowane tam, gdzie liczy sie czas wystapienia uderzenia, a nie sam fakt jego
# wykrycia [1], [7]. Kristof et al. [6] uzywaja 75 ms.
STRICT_TOLERANCE_MS = 50.0

SCORE_FIELDS = ('sensitivity', 'positive_predictivity', 'f1', 'der',
                'offset_mean_ms', 'offset_std_ms')

# Granice wiarygodnosci detekcji wzgledem stosunku sygnalu do szumu [3], [4], [10], [11].
# Nazwa strefy towarzyszy kazdemu pomiarowi, zeby niski wynik przy -9 dB byl czytany jako
# spodziewane zachowanie detektora, a nie jako porazka filtru, ktory go poprzedzil.
RELIABILITY_ZONES = (
    (12.0, 'pewna', 'detekcja blisko sufitu'),
    (5.0, 'dobra', 'zwykle dobra, ale zalezna od detektora'),
    (0.0, 'przejsciowa', 'metody zaczynaja sie wyraznie roznicowac'),
    (-12.0, 'niepewna', 'wiele detektorow wyraznie sie psuje'),
    (float('-inf'), 'zawodna', 'czesc detektorow zawodzi calkowicie, zwlaszcza przy em'),
)


def reliability_zone(snr_db: float) -> dict:
    """
    Which region of the published breakdown a measurement came from.

    Attached to every score so that a poor result is read against what the detector is
    known to do at that ratio, rather than charged to the filter that produced the signal.
    """
    if not np.isfinite(snr_db):
        return {'zone': 'nieznana', 'note': 'brak stosunku sygnalu do szumu',
                'snr_db': float(snr_db)}

    for boundary, name, note in RELIABILITY_ZONES:
        if snr_db >= boundary:
            return {'zone': name, 'note': note, 'snr_db': float(snr_db)}
    return {'zone': 'zawodna', 'note': RELIABILITY_ZONES[-1][2], 'snr_db': float(snr_db)}


def score_detection(detected: Sequence[int], reference: Sequence[int], fs: float,
                    standard_ms: float = STANDARD_TOLERANCE_MS,
                    strict_ms: float = STRICT_TOLERANCE_MS,
                    snr_db: Optional[float] = None) -> dict:
    """
    Detection scored at both tolerances, with the timing offset alongside.

    The standard window answers whether the beat was found; the strict one answers whether
    it was found in the right place. Reporting only the first would let a detector that is
    consistently eighty milliseconds late score as perfect, and that error propagates into
    every interbeat interval computed afterwards.
    """
    if strict_ms >= standard_ms:
        raise ValueError(f'the strict tolerance must be tighter than the standard one, '
                         f'got {strict_ms} against {standard_ms}')

    standard = detection_metrics(detected, reference, fs, tolerance_ms=standard_ms)
    strict = detection_metrics(detected, reference, fs, tolerance_ms=strict_ms)

    score = {
        'standard': standard,
        'strict': strict,
        'standard_tolerance_ms': float(standard_ms),
        'strict_tolerance_ms': float(strict_ms),
        'f1_loss': _difference(standard.get('f1'), strict.get('f1')),
        'der_rise': _difference(strict.get('der'), standard.get('der')),
    }
    if snr_db is not None:
        score['reliability'] = reliability_zone(snr_db)
    return score


def _difference(first, second) -> float:
    """Difference between two measures, or not a number when either is missing."""
    if first is None or second is None:
        return float('nan')
    if not (np.isfinite(first) and np.isfinite(second)):
        return float('nan')
    return float(first - second)


def score_waveform(signal: np.ndarray, reference: Sequence[int], fs: float,
                   method: Optional[str] = None, snr_db: Optional[float] = None,
                   **kwargs) -> dict:
    """
    Runs the detector over a waveform and scores what it found.

    The detector is the same one used everywhere else in this work, so that a difference
    between two methods is a difference between the waveforms they produced rather than
    between the instruments used to read them.
    """
    detected = detect_qrs(signal, fs) if method is None else detect_qrs(signal, fs,
                                                                        method=method)
    score = score_detection(detected, reference, fs, snr_db=snr_db, **kwargs)
    score['n_detected'] = int(np.asarray(detected).size)
    return score


def timing_sensitivity(detected: Sequence[int], reference: Sequence[int], fs: float,
                       tolerances_ms: Sequence[float] = (150.0, 100.0, 75.0, 50.0, 25.0)
                       ) -> list:
    """
    How the score changes as the admitted window is tightened.

    The curve is the evidence that a single wide window is not enough. Reklewski and
    Augustyniak 2025 measured a detection error rate moving from 8.1 to 67.2 percent for a
    change of eleven milliseconds in the tolerance; a method whose curve falls steeply is
    finding the beats and misplacing them.
    """
    rows = []
    for tolerance in sorted(tolerances_ms, reverse=True):
        metrics = detection_metrics(detected, reference, fs, tolerance_ms=tolerance)
        rows.append({'tolerance_ms': float(tolerance), **{
            field: metrics.get(field) for field in SCORE_FIELDS}})
    return rows


def format_score(score: dict) -> str:
    """Readable summary of one scored waveform, for the console and for the appendix."""
    lines = ['%-10s %10s %10s %10s %10s %12s' % (
        'okno', 'Se [%]', 'P+ [%]', 'F1 [%]', 'DER [%]', 'offset [ms]'), '-' * 68]

    for label, key in (('standard', 'standard'), ('scisle', 'strict')):
        metrics = score[key]
        lines.append('%-10s %10.2f %10.2f %10.2f %10.2f %12s' % (
            f"{label} {score[f'{key}_tolerance_ms']:.0f}",
            100 * metrics['sensitivity'], 100 * metrics['positive_predictivity'],
            100 * metrics['f1'], 100 * metrics['der'],
            f"{metrics['offset_mean_ms']:+.1f} ± {metrics['offset_std_ms']:.1f}"))

    lines.append('')
    lines.append(f"spadek F1 po zaostrzeniu okna: {100 * score['f1_loss']:.2f} pkt proc.")
    if 'reliability' in score:
        zone = score['reliability']
        lines.append(f"strefa wiarygodnosci przy {zone['snr_db']:+.1f} dB: "
                     f"{zone['zone']} ({zone['note']})")
    return '\n'.join(lines)
