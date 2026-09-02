"""
Metrics that need a clean reference, for the synthetic environment.

Four choices here were settled against the literature rather than assumed, and each of
them changes a number in the results table.

The percentage root mean square difference puts the **clean** signal in the denominator.
That is the dominant convention (Blanco-Velasco et al. 2005; Chiang et al. 2019; Bing et
al. 2020) and the one this work follows. Equation (14) of Wang et al. 2023 places the
denoised signal there instead, which is a minority reading and most likely a typographical
error; their published figures should be read with that in mind.

Both the raw and the mean removed forms are computed. The raw form depends on the direct
current level of the signal, and Blanco-Velasco et al. warn that relying on it alone can
lead to wrong conclusions; the mean removed form, usually written PRD1, isolates the
alternating component and is the one to prefer wherever a baseline offset may be present.
Reporting both costs nothing and makes the difference visible instead of arguable.

Improvement in signal to noise ratio is reported alongside the output ratio, never alone.
Chiang et al. 2019 show the improvement to be inversely related to the input ratio: the
same method scores a much larger improvement at minus nine decibels than at eleven, so a
table of improvements alone would rank methods by the ratio they happened to be tested at.
The output ratio gives the absolute level, the improvement gives the gain, and the two
together are what the literature reports.

Values are computed on the waveform in millivolts, without amplitude normalisation. There
is no consensus on normalisation across the published comparisons, and reported values
span from 0.095 to 32.61 percent depending on scale and noise type, so an absolute value
taken from another paper is not directly comparable with one taken here. What is
comparable is the ordering of methods measured the same way, which is what this work
reports.

References
----------
.. [1] Blanco-Velasco, M., Cruz-Roldan, F., Godino-Llorente, J. I., Blanco-Velasco, J.,
       Armiens-Aparicio, C., & Lopez-Ferreras, F. (2005). On the use of PRD and CR
       parameters for ECG compression. Medical Engineering & Physics, 27(9), 798-802.
       https://doi.org/10.1016/j.medengphy.2005.02.007
.. [2] Chiang, H.-T., Hsieh, Y.-Y., Fu, S.-W., Hung, K.-H., Tsao, Y., & Chien, S.-Y.
       (2019). Noise Reduction in ECG Signals Using Fully Convolutional Denoising
       Autoencoders. IEEE Access, 7, 60806-60813.
       https://doi.org/10.1109/ACCESS.2019.2912036
.. [3] Bing, P., Liu, W., Wang, Z., & Zhang, Z. (2020). Noise Reduction in ECG Signal
       Using an Effective Hybrid Scheme. IEEE Access, 8, 160790-160801.
       https://doi.org/10.1109/ACCESS.2020.3021068
.. [4] Chen, X., Lin, J., Huang, C., & He, L. (2019). A novel method based on Adaptive
       Periodic Segment Matrix and Singular Value Decomposition for removing EMG artifact
       in ECG signal. Biomedical Signal Processing and Control, 62, 102060.
       https://doi.org/10.1016/j.bspc.2020.102060
.. [5] Elgendi, M., Mohamed, A., & Ward, R. (2017). Efficient ECG Compression and QRS
       Detection for E-Health Applications. Scientific Reports, 7, 459.
       https://doi.org/10.1038/s41598-017-00540-x
.. [6] Miaou, S.-G., & Lin, C.-L. (2002). A quality-on-demand algorithm for wavelet-based
       compression of electrocardiogram signals. IEEE Transactions on Biomedical
       Engineering, 49(3), 233-239. https://doi.org/10.1109/10.983457
.. [7] Wang, G., et al. (2023). Deep Convolutional Generative Adversarial Network with
       LSTM for ECG Denoising. Computational and Mathematical Methods in Medicine.
.. [8] Singh, P., & Sharma, A. (2022). Attention-Based Convolutional Denoising
       Autoencoder for Two-Lead ECG Denoising and Arrhythmia Classification. IEEE
       Transactions on Instrumentation and Measurement, 71, 1-10.
       https://doi.org/10.1109/TIM.2022.3197757
.. [9] Mir, H. Y., & Singh, O. (2024). A Novel Approach for Denoising ECG Signals
       Corrupted with White Gaussian Noise Using Wavelet Packet Transform and
       Soft-Thresholding. International Journal of Computing and Digital Systems.
       https://doi.org/10.12785/ijcds/150196
"""

import math
from typing import Optional

import numpy as np

METRIC_FIELDS = ('snr_out', 'snr_in', 'snr_improvement',
                 'mse', 'rmse', 'mae', 'prd', 'prd1', 'correlation')


def _aligned(first: np.ndarray, second: np.ndarray, span: Optional[tuple] = None) -> tuple:
    """Both waveforms as one dimensional arrays of equal length, optionally trimmed."""
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()

    length = min(first.size, second.size)
    if length == 0:
        raise ValueError('an empty waveform has no metrics')
    first, second = first[:length], second[:length]

    if span is not None:
        start, stop = int(span[0]), int(span[1])
        if not 0 <= start < stop <= length:
            raise ValueError(f'span {span} lies outside a waveform of {length} samples')
        first, second = first[start:stop], second[start:stop]

    return first, second


def snr(clean: np.ndarray, estimate: np.ndarray, span: Optional[tuple] = None) -> float:
    """
    Ratio of the clean signal to what separates it from the estimate, in decibels.

    Infinite for a perfect reconstruction, which is a fact rather than a failure and is
    reported as such; an aggregate that meets one has a problem the aggregate should show.

    .. math:: \\mathrm{SNR} = 10 \\log_{10}
              \\frac{\\sum_n x_n^2}{\\sum_n (x_n - \\hat{x}_n)^2}

    References
    ----------
    Chiang et al. 2019 [2]_, eq. (6); Wang et al. 2023 [7]_, eq. (11) and (12).
    """
    clean, estimate = _aligned(clean, estimate, span)
    residual = float(np.sum((clean - estimate) ** 2))
    power = float(np.sum(clean ** 2))

    if power == 0.0:
        return float('nan')
    if residual == 0.0:
        return float('inf')
    return 10.0 * math.log10(power / residual)


def snr_improvement(clean: np.ndarray, noisy: np.ndarray, estimate: np.ndarray,
                    span: Optional[tuple] = None) -> float:
    """
    How many decibels the method gained, which is meaningless without the level it gained
    them from.

    Chiang et al. 2019 [2]_ show the improvement to be inversely related to the input
    ratio: a method tested at a low ratio outscores the same method tested at a high one.
    Mir and Singh 2024 [9]_ report both quantities for that reason. Always read next to
    `snr_out`.

    References
    ----------
    Chiang et al. 2019 [2]_; Mir & Singh 2024 [9]_.
    """
    return snr(clean, estimate, span) - snr(clean, noisy, span)


def mean_squared_error(clean: np.ndarray, estimate: np.ndarray,
                       span: Optional[tuple] = None) -> float:
    clean, estimate = _aligned(clean, estimate, span)
    return float(np.mean((clean - estimate) ** 2))


def root_mean_squared_error(clean: np.ndarray, estimate: np.ndarray,
                            span: Optional[tuple] = None) -> float:
    return math.sqrt(mean_squared_error(clean, estimate, span))


def mean_absolute_error(clean: np.ndarray, estimate: np.ndarray,
                        span: Optional[tuple] = None) -> float:
    """
    Reported alongside the squared error rather than instead of it.

    Squaring weights the largest deviations most, which pushes a model towards smoothing
    the sharp parts of the waveform, and the sharp parts of an ECG are the diagnostic ones
    (Singh & Sharma 2022 [8]_). The absolute error weights them evenly. The literature
    surveyed does not compare the two directly for ECG denoising, so both are reported and
    neither is claimed to be the better measure; each architecture keeps the loss its own
    publication used.

    References
    ----------
    Singh & Sharma 2022 [8]_.
    """
    clean, estimate = _aligned(clean, estimate, span)
    return float(np.mean(np.abs(clean - estimate)))


def prd(clean: np.ndarray, estimate: np.ndarray, span: Optional[tuple] = None,
        remove_mean: bool = False) -> float:
    """
    Percentage root mean square difference, with the clean signal in the denominator.

    .. math:: \\mathrm{PRD} = 100 \\sqrt{
              \\frac{\\sum_n (x_n - \\hat{x}_n)^2}{\\sum_n x_n^2}}

    The clean signal stands in the denominator, which is the dominant convention
    (Blanco-Velasco et al. 2005 [1]_; Miaou & Lin 2002 [6]_; Chiang et al. 2019 [2]_;
    Bing et al. 2020 [3]_). Wang et al. 2023 [7]_ place the denoised signal there in their
    eq. (14); that is a minority reading and most probably a typographical error, so their
    published values should be read accordingly.

    `remove_mean` gives the form usually written PRD1. The raw form depends on the direct
    current level, and Blanco-Velasco et al. [1]_ warn that using it alone can lead to
    wrong conclusions; PRD1 removes the mean from both signals first and measures the
    alternating component alone (Chen et al. 2019 [4]_).

    References
    ----------
    Blanco-Velasco et al. 2005 [1]_; Chen et al. 2019 [4]_; Elgendi et al. 2017 [5]_.
    """
    clean, estimate = _aligned(clean, estimate, span)

    if remove_mean:
        clean = clean - clean.mean()
        estimate = estimate - estimate.mean()

    power = float(np.sum(clean ** 2))
    if power == 0.0:
        return float('nan')
    return 100.0 * math.sqrt(float(np.sum((clean - estimate) ** 2)) / power)


def correlation(clean: np.ndarray, estimate: np.ndarray,
                span: Optional[tuple] = None) -> float:
    """
    Pearson correlation between the clean waveform and the estimate.

    Blind to scale, so it separates a method that lost the shape of the beat from one that
    only lost its amplitude; the ratio metrics cannot tell those apart. Reported here as a
    supplement, since the surveyed comparisons do not use it as a primary criterion.
    """
    clean, estimate = _aligned(clean, estimate, span)
    if clean.std() == 0.0 or estimate.std() == 0.0:
        return float('nan')
    return float(np.corrcoef(clean, estimate)[0, 1])


def reference_metrics(clean: np.ndarray, noisy: np.ndarray, estimate: np.ndarray,
                      span: Optional[tuple] = None) -> dict:
    """
    Every reference metric of one window, in one pass.

    `span` restricts the comparison to the part a method actually reconstructed, which
    matters for the architecture built on cardiac cycles: it reaches from the first cycle
    to the last and returns the input unchanged outside them.
    """
    return {
        'snr_out': snr(clean, estimate, span),
        'snr_in': snr(clean, noisy, span),
        'snr_improvement': snr_improvement(clean, noisy, estimate, span),
        'mse': mean_squared_error(clean, estimate, span),
        'rmse': root_mean_squared_error(clean, estimate, span),
        'mae': mean_absolute_error(clean, estimate, span),
        'prd': prd(clean, estimate, span, remove_mean=False),
        'prd1': prd(clean, estimate, span, remove_mean=True),
        'correlation': correlation(clean, estimate, span),
    }


def aggregate(rows, fields=METRIC_FIELDS) -> dict:
    """
    Mean and spread of each metric over a set of windows, ignoring what is not finite.

    A perfect reconstruction gives an infinite ratio and a diverged filter gives a
    not-a-number; either would swallow an average whole. The count of what was dropped is
    returned alongside, so a method that produced mostly unusable windows cannot hide
    behind the average of the few that worked.
    """
    summary: dict = {}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[field] = {
            'mean': float(finite.mean()) if finite.size else float('nan'),
            'std': float(finite.std(ddof=1)) if finite.size > 1 else float('nan'),
            'median': float(np.median(finite)) if finite.size else float('nan'),
            'n': int(finite.size),
            'n_dropped': int(values.size - finite.size),
        }
    return summary
