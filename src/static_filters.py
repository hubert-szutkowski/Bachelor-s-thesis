import numpy as np
import scipy.signal as signal
import pywt
from PyEMD import EMD

def fir_filter(signal: np.ndarray, numtaps: int, cutoff: float, width: float = None, window: str = 'hamming', pass_zero: bool = True, scale: bool = True, fs: float = None) -> np.ndarray: #[5]
    """
    FIR filter design using the window method.

    Parameters
    ----------
    signal : array_like
        The input signal to be filtered.
    numtaps : int
        The number of taps in the FIR filter. Must be odd.
    cutoff : float or array_like
        The cutoff frequency of the filter. For lowpass and highpass filters, this is a single frequency. For bandpass and bandstop filters, this is a two-element array-like object.
    width : float, optional
        The width of the transition region (in Hz). If None, it will be set to 0.1 * cutoff.
    window : str or tuple, optional
        Desired window to use. See `scipy.signal.get_window` for a list of windows and required parameters. Default is 'hamming'.
    pass_zero : bool, optional
        If True, the filter will be a lowpass or bandstop filter. If False, it will be a highpass or bandpass
    fs : float, optional
        The sampling frequency of the signal. If None, it will be set to 2 * cutoff.
    
    Returns
    -------
    filtered_signal : ndarray
        The filtered signal.
    """

    if width is None:
        width = 0.1 * cutoff
    if fs is None:
        fs = 2 * cutoff
    nyq = 0.5 * fs
    normalized_cutoff = np.array(cutoff) / nyq
    normalized_width = width / nyq
    taps = signal.firwin(numtaps, normalized_cutoff, window=window, pass_zero=pass_zero, scale=scale, width=normalized_width)
    filtered_signal = signal.lfilter(taps, 1.0, signal)
    return filtered_signal


def iir_filter(signal: np.ndarray, order: int, Wn: np.array, rp: float = None, rs: float = None, btype: str = 'lowpass', analog: bool = False, fs: float = None, output: str = 'ba', ftype: str = 'butter') -> np.ndarray: #[5]
    """
    IIR filter design using the Butterworth method.

    Parameters
    ----------
    signal : array_like
        The input signal to be filtered.
    order : int
        The order of the filter.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass filters, this is a single frequency. For bandpass and bandstop filters, this is a two-element array-like object.
    rp : float
        For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB)
    rs : float
        For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB)
    btype : str, optional
        The type of filter to design. Must be one of 'lowpass', 'highpass', 'bandpass', or 'bandstop'. Default is 'lowpass'.
    analog : bool, optional
        If True, return an analog filter. If False, return a digital filter. Default is False.
    fs : float, optional
        The sampling frequency of the signal. If None, it will be set to 2 * Wn.
    output : str, optional
        Type of output: 'ba' for numerator/denominator, 'zpk' for zeros/poles/gain, 'sos' for second-order sections. Default is 'ba'.
    ftype : str, optional
        The type of IIR filter to design. Must be one of 'butter', 'cheby1', 'cheby2', or 'ellip'. Default is 'butter'.
    
    Returns
    -------
    filtered_signal : ndarray
        The filtered signal.
    """

    if fs is None:
        fs = 2 * np.max(Wn)
    nyq = 0.5 * fs
    normalized_Wn = np.array(Wn) / nyq
    filter_coeff = signal.iirfilter(order, normalized_Wn, rp=rp, rs=rs, btype=btype, analog=analog, output=output, ftype=ftype, fs=fs)
    filtered_signal = signal.lfilter(filter_coeff[0], filter_coeff[1], signal)
    return filtered_signal


def moving_average(data: np.ndarray, smooth_interval: int = 2) -> np.ndarray: #[5]
    '''
    Function to smooth data using an optimized moving average filter.

    Parameters
    ----------
    data : np.ndarray
        The input data to be smoothed.
    smooth_interval : int, optional
        The number of data points to include in the moving average. Default is 2.
        
    Returns
    -------
    new_data : np.ndarray
        The smoothed data.
    '''

    if smooth_interval > len(data):
        print("Smooth interval > length of data")
        return data
    if smooth_interval < 2:
        return data
    window = np.ones(smooth_interval) / smooth_interval
    new_data = np.convolve(data, window, mode='same')
    return new_data


def moving_median(data: np.ndarray, window_size: int = 2) -> np.ndarray: #[5]

    '''
    ----------
    data : array_like
        The input data to be smoothed.
    window_size : int, optional
        The number of data points to include in the moving median. Default is 2.
    Returns
    -------
    new_data : ndarray
        The smoothed data.
    '''

    if window_size > len(data):
        print("Window size > length of data")
        return data
    if window_size % 2 == 0:
        window_size += 1
    new_data = signal.medfilt(data, kernel_size=window_size)
    return new_data


def wavelet_denoising(data: np.ndarray, wavelet: str = 'db4', level: int = 1) -> np.ndarray: #[5]
    '''
    ----------
    data : array_like
        The input data to be denoised.
    wavelet : str, optional
        The type of wavelet to use. Default is 'db4'.
    level : int, optional
        The level of decomposition to use. Default is 1.
    Returns
    -------
    new_data : ndarray
        The denoised data.
    '''

    coeffs = pywt.dwt(data, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(data))) * (1 / np.sqrt(2))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
    new_data = pywt.waverec(coeffs, wavelet)
    return new_data

def wavelet_baseline_removal(data: np.ndarray, wavelet: str = 'db8', level: int = 9) -> np.ndarray: #[5]
    '''
    Function to remove low-frequency noise (baseline wander) from ECG signals 
    using the Discrete Wavelet Transform (DWT).

    Parameters
    ----------
    data : np.ndarray
        The input ECG signal (1D array).
    wavelet : str, optional
        The wavelet basis function to use. Default is 'db8' (Daubechies 8), also you can use 'db9', 'sym10', 'db1' .
    level : int, optional
        The decomposition level. Default is 9. 

    Returns
    -------
    denoised_data : np.ndarray
        The ECG signal with low-frequency noise removed.
    '''

    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    denoised_data = pywt.waverec(coeffs, wavelet)
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]
    return denoised_data


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

    emd = EMD()
    components = emd.emd(data, max_imf=max_imf)
    if components.shape[0] < noise_components:
        return data
    noise_estimate = np.sum(components[-noise_components:], axis=0)
    denoised_data = data - noise_estimate
    return denoised_data