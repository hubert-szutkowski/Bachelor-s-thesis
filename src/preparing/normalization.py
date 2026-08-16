from fractions import Fraction

import numpy as np
from scipy.signal import resample, resample_poly

TARGET_FS = 360.0
MAX_DENOMINATOR = 10000
POLYPHASE_MAX_DENOMINATOR = 64


def signal_scaling(signal_uv):
    """
    Scale the signal from microvolts to millivolts.

    MIT-BIH records are stored in millivolts (gain 200 ADU/mV), so this brings
    Neurobit exports onto the same amplitude scale.

    Parameters:
        signal_uv (np.ndarray): The input signal in microvolts.

    Returns:
        np.ndarray: The scaled signal in millivolts.
    """
    return signal_uv / 1000.0


def resampling_ratio(fs, target_fs, max_denominator=MAX_DENOMINATOR):
    """
    Rational approximation of the resampling ratio target_fs / fs.

    Accepts non-integer sampling frequencies. A measured rate of 360.4 Hz truncated
    to an integer would yield a ratio of 1/1, leaving a 2 s drift over a half-hour
    recording, so the ratio is built from the float value instead.

    Parameters:
        fs (float): Original sampling frequency in Hz.
        target_fs (float): Desired sampling frequency in Hz.
        max_denominator (int): Upper bound on the denominator of the approximation.

    Returns:
        tuple: (up, down) integers such that up / down approximates target_fs / fs.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'fs must be a positive finite number, got {fs}')
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError(f'target_fs must be a positive finite number, got {target_fs}')
    if max_denominator < 1:
        raise ValueError(f'max_denominator must be at least 1, got {max_denominator}')

    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(max_denominator)
    return ratio.numerator, ratio.denominator


def frequency_resampler(signal, fs, target_fs=TARGET_FS,
                        max_denominator=MAX_DENOMINATOR,
                        polyphase_max_denominator=POLYPHASE_MAX_DENOMINATOR):
    """
    Resample a signal to a target sampling frequency.

    Two methods are used depending on how well the ratio is approximated by a small
    fraction.

    Device rate conversions such as 500 Hz to 360 Hz reduce to 18/25 and are handled
    by polyphase filtering, which introduces no periodicity assumption and behaves
    well at the edges of the record.

    Clock drift corrections such as 360.4 Hz to 360 Hz reduce to 900/901. Polyphase
    filtering would upsample the signal 900 times, requiring several gigabytes for a
    half-hour record, so these are handled by band-limited Fourier resampling onto an
    output of the exact required length. Measured relative error for content up to
    90 Hz is of the order of 1e-5, roughly two orders of magnitude better than a cubic
    spline and three better than linear interpolation.

    The first axis is treated as time, matching the layout returned by
    `wfdb.rdrecord(...).p_signal`.

    Parameters:
        signal (np.ndarray): Signal to resample, time along the first axis.
        fs (float): Original sampling frequency in Hz. May be non-integer.
        target_fs (float): Desired sampling frequency in Hz. Defaults to 360.
        max_denominator (int): Upper bound on the denominator of the rational ratio.
        polyphase_max_denominator (int): Denominators up to this value use polyphase
            filtering; larger ones use Fourier resampling.

    Returns:
        np.ndarray: The resampled signal. The input object itself is returned when no
            conversion is required.
    """
    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError('signal is empty')

    up, down = resampling_ratio(fs, target_fs, max_denominator)
    if up == down:
        return signal

    if down <= polyphase_max_denominator and up <= polyphase_max_denominator:
        return resample_poly(signal, up, down)

    n_out = int(round(signal.shape[0] * float(target_fs) / float(fs)))
    if n_out < 1:
        raise ValueError(f'resampling would leave {n_out} samples')
    return resample(signal, n_out)


def resampled_length(n_samples, fs, target_fs, max_denominator=MAX_DENOMINATOR,
                     polyphase_max_denominator=POLYPHASE_MAX_DENOMINATOR):
    """
    Number of samples `frequency_resampler` will return, without performing the work.

    Parameters:
        n_samples (int): Length of the input along the time axis.
        fs (float): Original sampling frequency in Hz.
        target_fs (float): Desired sampling frequency in Hz.
        max_denominator (int): Upper bound on the denominator of the rational ratio.
        polyphase_max_denominator (int): Threshold between the two methods.

    Returns:
        int: Expected output length.
    """
    up, down = resampling_ratio(fs, target_fs, max_denominator)
    if up == down:
        return int(n_samples)
    if down <= polyphase_max_denominator and up <= polyphase_max_denominator:
        return -(-int(n_samples) * up // down)
    return int(round(int(n_samples) * float(target_fs) / float(fs)))
