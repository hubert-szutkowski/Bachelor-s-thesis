"""
Signal quality indices, for the environment where no clean reference exists.

These are the only measurements available on the wearable recordings, and therefore the
only ones on which all seventeen methods can be ranked together. That makes their
weaknesses the weaknesses of the whole comparison, and three of them matter enough to be
stated before any of the code.

**A band-pass filter inflates the spectral index without improving the signal.**
Moeyersons et al. 2019 [4]_ measured it: segments pre-processed with a 1 to 40 hertz
Butterworth band-pass scored higher on a spectral quality index than the same segments
untouched, because the index has no filtering operation of its own and rewards a spectrum
that already looks filtered. The default band-pass in this project is a Butterworth over
0.5 to 40 hertz, which is the same filter. A ranking of methods on `psqi` alone would
therefore reward seven static filters for being what they are, and the two families they
are being compared against would lose for a reason that has nothing to do with quality.
`analysis.metrics_reference` and the detection score exist partly to catch this, and the
control methods described in the evaluation plan exist entirely for it.

**Fixed thresholds do not carry between datasets.** Rahman et al. 2022 [5]_ found the
behaviour of statistical indices to fluctuate considerably across datasets and fixed
thresholds to be biased towards clean recordings. The values they quote are kept here as
`TYPICAL_THRESHOLDS` for orientation only; this work reports the indices as continuous
quantities and never classifies a segment on them.

**Agreement with a human reader is poor.** Syversen et al. 2024 [8]_ found quality indices
to agree with a cardiologist's judgement of clinical usefulness in between four and ten
cases out of sixteen. An index is evidence about a spectrum, not about whether a
recording can be read.

What the panel is good for is comparison under a validation. Correlation with a true
signal to noise ratio is strong where both can be computed: a learned index reached a
Spearman correlation of 0.94 with the preset ratio on synthetic material (Abdelazez et al.
2022 [1]_) and an autocorrelation index reached 0.65 (Moeyersons et al. 2019 [4]_). The
synthetic environment of this work has both quantities, so the panel is validated there
before it is trusted on the wearable recordings.

Combining the indices is not obviously worth it. Shahriari et al. 2017 [7]_ report that a
multivariate combination of the baseline, kurtosis and skewness indices gave no
improvement over the baseline index alone; heuristic fusion of four indices reached 94.67
percent in a three level classification (Zhao & Zhang 2018 [9]_) and a support vector
machine over thirteen metrics reached 80.26 percent in a five level one (Li et al. 2014
[3]_). The indices are therefore reported separately here and no composite score is formed.

References
----------
.. [1] Abdelazez, M., Rajan, S., & Chan, A. D. C. (2022). Signal Quality Assessment of
       Compressively Sensed Electrocardiogram. IEEE Transactions on Biomedical
       Engineering, 69(11), 3397-3406. https://doi.org/10.1109/TBME.2022.3170047
.. [2] Kuetche, F., Alexendre, N., Pascal, N., Colince, W., & Thierry, S. (2023). Signal
       quality indices evaluation for robust ECG signal quality assessment systems.
       Biomedical Physics & Engineering Express, 9. https://doi.org/10.1088/2057-1976/ace9e0
.. [3] Li, Q., Rajagopalan, C., & Clifford, G. D. (2014). A machine learning approach to
       multi-level ECG signal quality classification. Computer Methods and Programs in
       Biomedicine, 117(3), 435-447. https://doi.org/10.1016/j.cmpb.2014.09.002
.. [4] Moeyersons, J., Smets, E., Morales, J., Villa, A., De Raedt, W., Testelmans, D.,
       Buyse, B., Van Hoof, C., Willems, R., Van Huffel, S., & Varon, C. (2019). Artefact
       detection and quality assessment of ambulatory ECG signals. Computer Methods and
       Programs in Biomedicine, 182, 105050. https://doi.org/10.1016/j.cmpb.2019.105050
.. [5] Rahman, S., Karmakar, C., Natgunanathan, I., Yearwood, J., & Palaniswami, M.
       (2022). Robustness of electrocardiogram signal quality indices. Journal of the
       Royal Society Interface, 19. https://doi.org/10.1098/rsif.2022.0012
.. [6] Smital, L., Haider, C. R., Vitek, M., Leinveber, P., Jurak, P., Nemcova, A.,
       Smisek, R., Marsanova, L., Provaznik, I., Felton, C. L., Gilbert, B. K., &
       Holmes, D. R. (2020). Real-Time Quality Assessment of Long-Term ECG Signals
       Recorded by Wearables in Free-Living Conditions. IEEE Transactions on Biomedical
       Engineering, 67(9), 2721-2734. https://doi.org/10.1109/TBME.2020.2969719
.. [7] Shahriari, Y., Fidler, R., Pelter, M. M., Bai, Y., Villaroman, A., & Hu, X. (2017).
       Electrocardiogram Signal Quality Assessment Based on Structural Image Similarity
       Metric. IEEE Transactions on Biomedical Engineering, 65(4), 745-753.
       https://doi.org/10.1109/TBME.2017.2717876
.. [8] Syversen, A., Zhang, Z., Batty, J. A., Kaisti, M., Jayne, D., & Wong, D. C. (2024).
       Assessment of ECG Signal Quality Index Algorithms Using Synthetic ECG Data.
       Computing in Cardiology. https://doi.org/10.22489/CinC.2024.270
.. [9] Zhao, Z., & Zhang, Y. (2018). SQI Quality Evaluation Mechanism of Single-Lead ECG
       Signal Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation.
       Frontiers in Physiology, 9, 727. https://doi.org/10.3389/fphys.2018.00727
"""

from typing import Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

SQI_FIELDS = ('ksqi', 'ssqi', 'psqi', 'basesqi')

# Zhao & Zhang 2018 [9]: energia zespolu QRS skupiona w pasmie o srodku 10 Hz i
# szerokosci 10 Hz.
QRS_BAND = (5.0, 15.0)

# Mianownik pSQI. Zhao & Zhang mowia o "calkowitej energii sygnalu", a szeroko uzywana
# implementacja Li & Clifford [3] bierze 5-40 Hz. Roznica jest istotna, bo dolna granica
# decyduje, czy wedrowanie linii izoelektrycznej wchodzi do mianownika, wiec jest
# parametrem i trafia do zapisu wyniku.
TOTAL_BAND = (5.0, 40.0)

# Shahriari et al. 2017 [7]: aktywnosc ponizej 1 Hz traktowana jako pochodzaca
# z sygnalu zlej jakosci.
BASELINE_BAND = (0.0, 1.0)
BASELINE_TOTAL_BAND = (0.0, 40.0)

# Rahman et al. 2022 [5]. Zachowane wylacznie dla orientacji: te same prace pokazuja,
# ze progi stale nie przenosza sie miedzy zbiorami danych i sa obciazone na korzysc
# nagran czystych. Ta praca traktuje wskazniki jako wielkosci ciagle.
TYPICAL_THRESHOLDS = {
    'ksqi': {'clean_above': 5.0},
    'ssqi': {'clean_between': (-0.8, 0.8)},
    'psqi': {'clean_between': (0.5, 0.8)},
}

# Smital et al. 2020 [6]: progi na prawdziwym stosunku sygnalu do szumu, nie na SQI.
# Ponizej 5 dB detekcja QRS przestaje byc wiarygodna, ponizej 18 dB analiza pelnego
# przebiegu. Punkty odniesienia dla siatki SNR uzytej w tej pracy.
SNR_THRESHOLDS_DB = {'qrs_detection': 5.0, 'waveform_analysis': 18.0}


def _spectrum(signal: np.ndarray, fs: float, nperseg: Optional[int] = None) -> tuple:
    """Power spectral density by Welch's method, on the signal with its mean removed."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 8:
        raise ValueError(f'a window of {signal.size} samples is too short for a spectrum')

    if nperseg is None:
        nperseg = min(signal.size, 1024)
    frequency, power = welch(signal - signal.mean(), fs=fs, nperseg=int(nperseg))
    return frequency, power


def _band_power(frequency: np.ndarray, power: np.ndarray, band: tuple) -> float:
    low, high = float(band[0]), float(band[1])
    inside = (frequency >= low) & (frequency <= high)
    if not inside.any():
        return 0.0
    return float(np.trapezoid(power[inside], frequency[inside]))


def ksqi(signal: np.ndarray, fisher: bool = False) -> float:
    """
    Kurtosis of the waveform, the fourth standardised moment.

    A clean ECG is dominated by sharp, rare complexes against a quiet baseline, which is a
    heavy tailed distribution; noise pulls it towards Gaussian. Reported in the Pearson
    convention, where a Gaussian scores three, because the thresholds quoted in the
    literature assume it (`fisher=True` subtracts the three).

    Rewards sparsity, which is not the same as quality: a method that thresholds the
    waveform into isolated spikes raises this index while destroying the P and T waves.

    References
    ----------
    Shahriari et al. 2017 [7]_; Rahman et al. 2022 [5]_.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 4 or signal.std() == 0.0:
        return float('nan')
    return float(kurtosis(signal, fisher=fisher, bias=False))


def ssqi(signal: np.ndarray) -> float:
    """
    Skewness of the waveform, the third standardised moment.

    A clean ECG is asymmetric, since the R wave rises far above the baseline while nothing
    goes equally far below it; noise is symmetric and pulls the value towards zero.

    Sign depends on the lead and on which way the complex points, so its absolute value is
    what carries the information.

    References
    ----------
    Shahriari et al. 2017 [7]_; Rahman et al. 2022 [5]_.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 3 or signal.std() == 0.0:
        return float('nan')
    return float(skew(signal, bias=False))


def psqi(signal: np.ndarray, fs: float, qrs_band: tuple = QRS_BAND,
         total_band: tuple = TOTAL_BAND, nperseg: Optional[int] = None) -> float:
    """
    Share of the spectrum that falls in the band of the QRS complex.

    The numerator is the band centred at ten hertz and ten hertz wide, which is where the
    energy of the complex sits; the denominator is the wider band the signal is expected to
    occupy. Muscle noise raises the high frequency content and lowers the ratio.

    **Inflatable by filtering.** The index has no filtering operation of its own and scores
    a band-passed segment higher than the same segment untouched, whether or not the
    filtering helped (Moeyersons et al. 2019 [4]_). A band-pass filter over the same band
    as `total_band` approaches the value one by construction. Never rank filtering methods
    on this index alone.

    References
    ----------
    Zhao & Zhang 2018 [9]_ for the band; Li et al. 2014 [3]_ for the usual denominator;
    Moeyersons et al. 2019 [4]_ for the vulnerability.
    """
    frequency, power = _spectrum(signal, fs, nperseg)
    total = _band_power(frequency, power, total_band)
    if total <= 0.0:
        return float('nan')
    return _band_power(frequency, power, qrs_band) / total


def basesqi(signal: np.ndarray, fs: float, baseline_band: tuple = BASELINE_BAND,
            total_band: tuple = BASELINE_TOTAL_BAND,
            nperseg: Optional[int] = None) -> float:
    """
    One minus the share of the spectrum lying below one hertz.

    Activity there is taken to be baseline wander rather than signal, so a value near one
    means little drift. Reported alongside the others because Shahriari et al. 2017 [7]_
    found it to carry as much as their multivariate combination of three indices did.

    References
    ----------
    Shahriari et al. 2017 [7]_.
    """
    frequency, power = _spectrum(signal, fs, nperseg)
    total = _band_power(frequency, power, total_band)
    if total <= 0.0:
        return float('nan')
    return 1.0 - _band_power(frequency, power, baseline_band) / total


def sqi_panel(signal: np.ndarray, fs: float, **kwargs) -> dict:
    """
    Every index of one window, reported separately.

    No composite score is formed. A multivariate combination of the baseline, kurtosis and
    skewness indices gave no improvement over the baseline index alone in Shahriari et al.
    2017 [7]_, and a composite would hide which index a method exploited.
    """
    return {
        'ksqi': ksqi(signal),
        'ssqi': ssqi(signal),
        'psqi': psqi(signal, fs, **kwargs),
        'basesqi': basesqi(signal, fs),
    }


def rank_correlation(indices, references) -> float:
    """
    Spearman correlation between an index and a quantity known to be true.

    The validation the panel needs before it is used where nothing is known. Published
    values give a scale to read the result against: 0.94 for a learned index against a
    preset ratio (Abdelazez et al. 2022 [1]_) and 0.65 for an autocorrelation index
    (Moeyersons et al. 2019 [4]_). A value near zero, or negative, means the index is
    measuring something other than quality on this material.
    """
    from scipy.stats import spearmanr

    indices = np.asarray(indices, dtype=np.float64).ravel()
    references = np.asarray(references, dtype=np.float64).ravel()
    if indices.size != references.size:
        raise ValueError(f'{indices.size} indices against {references.size} references')

    usable = np.isfinite(indices) & np.isfinite(references)
    if usable.sum() < 3:
        return float('nan')
    return float(spearmanr(indices[usable], references[usable]).statistic)


def validate_panel(windows, fs: float, snr_values, **kwargs) -> dict:
    """
    Correlation of every index with the true ratio, over a set of windows.

    Run on the synthetic material, where both are known, before any index is read on the
    wearable recordings. An index that fails here has no standing there.
    """
    values = {field: [] for field in SQI_FIELDS}
    for window in windows:
        panel = sqi_panel(window, fs, **kwargs)
        for field in SQI_FIELDS:
            values[field].append(panel[field])

    return {field: rank_correlation(values[field], snr_values) for field in SQI_FIELDS}
