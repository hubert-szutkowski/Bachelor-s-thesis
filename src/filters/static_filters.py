import numpy as np
import scipy.signal as sp_signal
import pywt
from PyEMD import EMD


def _as_1d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {arr.shape}")
    return arr


def fir_filter(data: np.ndarray, numtaps: int, cutoff, width: float = None, window: str = 'hamming', pass_zero: bool = True, scale: bool = True, fs: float = None, zero_phase: bool = True) -> np.ndarray: #[5]
    """
    FIR filter design using the window method.

    Parameters
    ----------
    data : array_like
        The input signal to be filtered.
    numtaps : int
        The number of taps in the FIR filter.
    cutoff : float or array_like
        The cutoff frequency of the filter, in Hz. For lowpass and highpass filters, this is a single frequency. For bandpass and bandstop filters, this is a two-element array-like object.
    width : float, optional
        The width of the transition region (in Hz). If None, the window parameter is used instead.
    window : str or tuple, optional
        Desired window to use. See `scipy.signal.get_window` for a list of windows and required parameters. Default is 'hamming'.
    pass_zero : bool or str, optional
        If True, the filter will be a lowpass or bandstop filter. If False, it will be a highpass or bandpass filter.
    scale : bool, optional
        Whether to scale the coefficients so that the frequency response is exactly unity at a certain frequency. Default is True.
    fs : float, optional
        The sampling frequency of the signal, in Hz. Required.
    zero_phase : bool, optional
        If True, apply the filter forward and backward (`filtfilt`) so that no phase distortion is introduced. Default is True.

    Returns
    -------
    filtered_signal : ndarray
        The filtered signal.
    """

    data = _as_1d(data)
    if fs is None:
        raise ValueError("fs must be provided (in Hz)")

    cutoff_arr = np.atleast_1d(np.asarray(cutoff, dtype=np.float64))
    if np.any(cutoff_arr <= 0) or np.any(cutoff_arr >= 0.5 * fs):
        raise ValueError(f"cutoff must lie in (0, fs/2) = (0, {0.5 * fs})")

    needs_odd = (pass_zero is False and cutoff_arr.size == 1) or (pass_zero is True and cutoff_arr.size == 2)
    if needs_odd and numtaps % 2 == 0:
        numtaps += 1

    taps = sp_signal.firwin(numtaps, cutoff_arr if cutoff_arr.size > 1 else cutoff_arr[0],
                            width=width, window=window, pass_zero=pass_zero, scale=scale, fs=fs)

    if zero_phase:
        padlen = min(3 * len(taps), data.size - 1)
        return sp_signal.filtfilt(taps, 1.0, data, padlen=padlen)
    return sp_signal.lfilter(taps, 1.0, data)


def iir_filter(data: np.ndarray, order: int, Wn, rp: float = None, rs: float = None, btype: str = 'lowpass', analog: bool = False, fs: float = None, ftype: str = 'butter', zero_phase: bool = True) -> np.ndarray: #[5]
    """
    IIR filter design and application.

    Parameters
    ----------
    data : array_like
        The input signal to be filtered.
    order : int
        The order of the filter.
    Wn : float or array_like
        The critical frequency or frequencies, in Hz. For lowpass and highpass filters, this is a single frequency. For bandpass and bandstop filters, this is a two-element array-like object.
    rp : float, optional
        For Chebyshev and elliptic filters, the maximum ripple in the passband (dB).
    rs : float, optional
        For Chebyshev and elliptic filters, the minimum attenuation in the stopband (dB).
    btype : str, optional
        The type of filter to design: 'lowpass', 'highpass', 'bandpass' or 'bandstop'. Default is 'lowpass'.
    analog : bool, optional
        If True, return an analog filter. If False, return a digital filter. Default is False.
    fs : float, optional
        The sampling frequency of the signal, in Hz. Required.
    ftype : str, optional
        The type of IIR filter to design: 'butter', 'cheby1', 'cheby2' or 'ellip'. Default is 'butter'.
    zero_phase : bool, optional
        If True, apply the filter forward and backward (`sosfiltfilt`) so that no phase distortion is introduced. Default is True.

    Returns
    -------
    filtered_signal : ndarray
        The filtered signal.
    """

    data = _as_1d(data)
    if fs is None:
        raise ValueError("fs must be provided (in Hz)")

    Wn_arr = np.atleast_1d(np.asarray(Wn, dtype=np.float64))
    if np.any(Wn_arr <= 0) or np.any(Wn_arr >= 0.5 * fs):
        raise ValueError(f"Wn must lie in (0, fs/2) = (0, {0.5 * fs})")

    sos = sp_signal.iirfilter(order, Wn_arr if Wn_arr.size > 1 else Wn_arr[0], rp=rp, rs=rs,
                              btype=btype, analog=analog, ftype=ftype, output='sos', fs=fs)

    if zero_phase:
        padlen = min(3 * (2 * len(sos) + 1), data.size - 1)
        return sp_signal.sosfiltfilt(sos, data, padlen=padlen)
    return sp_signal.sosfilt(sos, data)


def moving_average(data: np.ndarray, smooth_interval: int = 2) -> np.ndarray: #[5]
    '''
    Function to smooth data using a zero-delay moving average filter.

    Parameters
    ----------
    data : np.ndarray
        The input data to be smoothed.
    smooth_interval : int, optional
        The number of data points to include in the moving average. Forced to the nearest odd value so that the window stays symmetric. Default is 2.

    Returns
    -------
    new_data : np.ndarray
        The smoothed data.
    '''

    data = _as_1d(data)
    if smooth_interval < 2:
        return data
    if smooth_interval % 2 == 0:
        smooth_interval += 1
    if smooth_interval > data.size:
        raise ValueError(f"smooth_interval ({smooth_interval}) exceeds signal length ({data.size})")

    half = smooth_interval // 2
    padded = np.pad(data, half, mode='reflect')
    window = np.ones(smooth_interval) / smooth_interval
    return np.convolve(padded, window, mode='valid')


def moving_median(data: np.ndarray, window_size: int = 2) -> np.ndarray: #[5]
    '''
    Function to smooth data using a median filter.

    Parameters
    ----------
    data : array_like
        The input data to be smoothed.
    window_size : int, optional
        The number of data points to include in the moving median. Forced to the nearest odd value. Default is 2.

    Returns
    -------
    new_data : ndarray
        The smoothed data.
    '''

    data = _as_1d(data)
    if window_size % 2 == 0:
        window_size += 1
    if window_size > data.size:
        raise ValueError(f"window_size ({window_size}) exceeds signal length ({data.size})")
    return sp_signal.medfilt(data, kernel_size=window_size)


def wavelet_denoising(data: np.ndarray, wavelet: str = 'db4', level: int = 1, mode: str = 'soft') -> np.ndarray: #[5]
    '''
    Function to denoise a signal by thresholding its detail wavelet coefficients.

    Parameters
    ----------
    data : array_like
        The input data to be denoised.
    wavelet : str, optional
        The type of wavelet to use. Default is 'db4'.
    level : int, optional
        The level of decomposition to use. Clipped to the maximum useful level for the given signal length. Default is 1.
    mode : str, optional
        Thresholding mode passed to `pywt.threshold`. Default is 'soft'.

    Returns
    -------
    new_data : ndarray
        The denoised data.
    '''

    data = _as_1d(data)
    max_level = pywt.dwt_max_level(data.size, pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(level, max_level))

    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(data.size))
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode=mode) for c in coeffs[1:]]

    return pywt.waverec(coeffs, wavelet)[:data.size]


def wavelet_baseline_removal(data: np.ndarray, wavelet: str = 'db8', level: int = 9) -> np.ndarray: #[5]
    '''
    Function to remove low-frequency noise (baseline wander) from ECG signals
    using the Discrete Wavelet Transform (DWT).

    Parameters
    ----------
    data : np.ndarray
        The input ECG signal (1D array).
    wavelet : str, optional
        The wavelet basis function to use. Default is 'db8'.
    level : int, optional
        The decomposition level. Clipped to the maximum useful level for the given signal length. Default is 9.

    Returns
    -------
    denoised_data : np.ndarray
        The ECG signal with low-frequency noise removed.
    '''

    data = _as_1d(data)
    max_level = pywt.dwt_max_level(data.size, pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(level, max_level))

    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    return pywt.waverec(coeffs, wavelet)[:data.size]


def emd_ecg_denoising(data: np.ndarray, max_imf: int = 6, noise_components: int = 3) -> np.ndarray: #[5]
    '''
    Function to remove low-frequency noise from ECG signals using Empirical Mode Decomposition (EMD).

    Parameters
    ----------
    data : np.ndarray
        The input ECG signal to be denoised.
    max_imf : int, optional
        The maximum number of Intrinsic Mode Functions (IMFs) to extract. Default is 6.
    noise_components : int, optional
        The number of the last components (IMFs + residue) to be summed and considered as noise. Default is 3.

    Returns
    -------
    denoised_data : np.ndarray
        The ECG signal with low-frequency noise removed.
    '''

    data = _as_1d(data)
    components = EMD().emd(data, max_imf=max_imf)
    if components.shape[0] <= noise_components:
        return data
    noise_estimate = np.sum(components[-noise_components:], axis=0)
    return data - noise_estimate
