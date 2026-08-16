'''
Signal transforms shared by the deep architectures.

Kept free of any PyTorch import on purpose: the encode / decode pairs are pure NumPy and
SciPy, they are exercised while the cache is built and while metrics are computed, and both
of those steps must be runnable without a CUDA-capable install.

`sced_net_model` and `FSSTH_model` re-export these names, so existing imports keep working.
'''

import numpy as np
from scipy import signal
from scipy.signal import resample
from PyEMD import EMD


class ECGSignalProcessor:
    '''
    Handles the creation of the Stacked Cardiac Cycle (SCC) tensor 
    and the reconstruction of the 1D signal.
    '''
    def __init__(self, target_length: int = 384, num_cycles: int = 32, min_cycle_length: int = 8):
        self.target_length = target_length
        self.num_cycles = num_cycles
        self.min_cycle_length = min_cycle_length

    def fourier_interpolate(self, cycle: np.ndarray, target_len: int) -> np.ndarray:
        '''Resamples a cycle to a target length in the Fourier domain.'''
        cycle = np.asarray(cycle, dtype=np.float64)
        if cycle.size == target_len:
            return cycle
        return resample(cycle, target_len)

    def segment_bounds(self, ecg_length: int, r_peaks: np.ndarray) -> list:
        '''
        Returns contiguous [start, end) bounds of consecutive cardiac cycles,
        each delimited by the midpoints between neighbouring R-peaks.
        '''
        r_peaks = np.asarray(r_peaks, dtype=int)
        if r_peaks.size < 2:
            raise ValueError(f"at least 2 R-peaks are required, got {r_peaks.size}")

        bounds = []
        start = max(0, int(r_peaks[0]) - (int(r_peaks[1]) - int(r_peaks[0])) // 2)
        for i in range(len(r_peaks) - 1):
            if len(bounds) >= self.num_cycles:
                break
            end = min(ecg_length, (int(r_peaks[i]) + int(r_peaks[i + 1])) // 2)
            if end > start:
                bounds.append((start, end))
            start = end
        return bounds

    def preprocess(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, stats: tuple = None) -> tuple:
        '''
        Segments ECG into SCC Tensor.

        Parameters
        ----------
        ecg_signal : np.ndarray
            The 1D ECG signal to be segmented.
        r_peaks : np.ndarray
            Sample indices of the detected R-peaks.
        stats : tuple, optional
            (g_min, g_max) to reuse for normalization. Pass the statistics obtained from the
            noisy signal when encoding the corresponding clean target, otherwise the network
            has to learn an implicit rescaling and the reported SNR is biased.

        Returns
        -------
        scc_matrix : np.ndarray
            SCC tensor of shape (num_cycles, target_length).
        original_lengths : list
            Sample count of every cycle before resampling; 0 marks a padded row.
        g_min : float
            Minimum used for normalization.
        g_max : float
            Maximum used for normalization.
        '''
        ecg_signal = np.asarray(ecg_signal, dtype=np.float64).ravel()
        bounds = self.segment_bounds(ecg_signal.size, r_peaks)

        scc_matrix = np.zeros((self.num_cycles, self.target_length), dtype=np.float64)
        original_lengths = [0] * self.num_cycles

        for i, (start, end) in enumerate(bounds):
            if end - start < self.min_cycle_length:
                continue
            scc_matrix[i] = self.fourier_interpolate(ecg_signal[start:end], self.target_length)
            original_lengths[i] = end - start

        valid = np.array(original_lengths) > 0
        if not valid.any():
            raise ValueError("no valid cardiac cycle could be extracted")

        if stats is None:
            g_min = float(scc_matrix[valid].min())
            g_max = float(scc_matrix[valid].max())
        else:
            g_min, g_max = float(stats[0]), float(stats[1])

        scale = g_max - g_min
        if scale <= 0:
            scale = 1.0
        scc_matrix = (scc_matrix - g_min) / scale
        scc_matrix[~valid] = 0.0

        return scc_matrix, original_lengths, g_min, g_max

    def postprocess(self, scc_matrix: np.ndarray, original_lengths: list, g_min: float, g_max: float) -> np.ndarray:
        '''Reconstructs the 1D ECG signal from the denoised SCC Tensor.'''
        scc_matrix = np.asarray(scc_matrix, dtype=np.float64).reshape(self.num_cycles, self.target_length)

        scale = g_max - g_min
        if scale <= 0:
            scale = 1.0
        scc_matrix = scc_matrix * scale + g_min

        reconstructed_1d = []
        for i in range(min(self.num_cycles, len(original_lengths))):
            orig_len = int(original_lengths[i])
            if orig_len == 0:
                continue
            reconstructed_1d.append(self.fourier_interpolate(scc_matrix[i, :], orig_len))

        if not reconstructed_1d:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(reconstructed_1d)


DEFAULT_FS = 250
DEFAULT_NPERSEG = 256
DEFAULT_NOVERLAP = 224
DEFAULT_N_FREQ = 129
DEFAULT_N_FRAMES = 32


def stft_proxy_transform(ecg_signal: np.ndarray, fs: int = DEFAULT_FS, nperseg: int = DEFAULT_NPERSEG,
                         noverlap: int = DEFAULT_NOVERLAP, n_freq: int = DEFAULT_N_FREQ,
                         n_frames: int = DEFAULT_N_FRAMES, stats: dict = None) -> tuple:
    '''
    Structural proxy for the High-order Synchrosqueezing Transform (FSSTH) of [10].
    A standard STFT is used so that the tensor dimensions match the ones expected by DeepCEDNet.
    Replace with a true FSSTH implementation before reporting results as FSSTH-based.

    Parameters
    ----------
    ecg_signal : np.ndarray
        The 1D ECG segment to transform.
    fs : int, optional
        Sampling frequency in Hz.
    nperseg, noverlap : int, optional
        STFT window length and overlap in samples.
    n_freq, n_frames : int, optional
        Size of the retained time-frequency patch.
    stats : dict, optional
        Normalization statistics {'loc', 'scale'} to reuse. Pass the statistics obtained from the
        noisy segment when transforming the corresponding clean target.

    Returns
    -------
    t_f_matrix : np.ndarray
        Real/imaginary time-frequency tensor of shape (2, n_freq, n_frames).
    meta : dict
        Everything required by `stft_proxy_inverse` to return to the time domain.
    '''
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64).ravel()

    _, _, Zxx = signal.stft(ecg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if Zxx.shape[0] < n_freq or Zxx.shape[1] < n_frames:
        raise ValueError(f"STFT produced a {Zxx.shape} grid, which is smaller than the requested ({n_freq}, {n_frames}) patch")

    t_f_matrix = np.stack((np.real(Zxx[:n_freq, :n_frames]), np.imag(Zxx[:n_freq, :n_frames])), axis=0)

    if stats is None:
        stats = {
            'loc': np.mean(t_f_matrix, axis=(1, 2), keepdims=True),
            'scale': np.std(t_f_matrix, axis=(1, 2), keepdims=True) + 1e-8,
        }
    t_f_matrix = (t_f_matrix - stats['loc']) / stats['scale']

    meta = {
        'stats': stats,
        'fs': fs,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'n_freq': n_freq,
        'n_frames': n_frames,
        'full_shape': Zxx.shape,
        'signal_length': ecg_signal.size,
    }
    return t_f_matrix.astype(np.float32), meta


def stft_proxy_inverse(t_f_matrix: np.ndarray, meta: dict) -> np.ndarray:
    '''
    Inverse of `stft_proxy_transform`. Required to evaluate SNR, MSE, RMSE and PRD
    in the time domain, which is the only domain in which those metrics are comparable
    across architectures.

    Parameters
    ----------
    t_f_matrix : np.ndarray
        Denormalized-ready tensor of shape (2, n_freq, n_frames).
    meta : dict
        Metadata returned by `stft_proxy_transform`.

    Returns
    -------
    reconstructed : np.ndarray
        1D signal of length meta['signal_length'].
    '''
    t_f_matrix = np.asarray(t_f_matrix, dtype=np.float64).reshape(2, meta['n_freq'], meta['n_frames'])
    t_f_matrix = t_f_matrix * meta['stats']['scale'] + meta['stats']['loc']

    Zxx = np.zeros(meta['full_shape'], dtype=np.complex128)
    Zxx[:meta['n_freq'], :meta['n_frames']] = t_f_matrix[0] + 1j * t_f_matrix[1]

    _, reconstructed = signal.istft(Zxx, fs=meta['fs'], nperseg=meta['nperseg'], noverlap=meta['noverlap'])

    n = meta['signal_length']
    if reconstructed.size >= n:
        return reconstructed[:n]
    return np.pad(reconstructed, (0, n - reconstructed.size))


class MIEMD_Filter:
    '''
    Empirical Mode Decomposition for initial ECG denoising.
    Extracts IMFs, applies soft thresholding based on noise estimation, and reconstructs.
    '''
    def __init__(self, threshold_multiplier: float = 1.0):
        self.emd = EMD()
        self.threshold_multiplier = threshold_multiplier

    def soft_thresholding(self, imf: np.ndarray, threshold: float) -> np.ndarray:
        '''Applies soft thresholding to a single IMF.'''
        return np.sign(imf) * np.maximum(np.abs(imf) - threshold, 0.0)

    def process(self, ecg_signal: np.ndarray) -> np.ndarray:
        '''
        Decomposes signal, thresholds high-frequency IMFs, and reconstructs.
        '''
        # 1. Decomposition into IMFs
        imfs = self.emd.emd(ecg_signal)
        
        if imfs.shape[0] == 0:
            return ecg_signal
            
        processed_imfs = np.zeros_like(imfs)
        
        # 2. Noise Estimation & Thresholding
        # Usually, the first IMF (IMF1) contains the most high-frequency noise.
        # We estimate universal threshold based on the median absolute deviation (MAD) of IMF1
        sigma = np.median(np.abs(imfs[0] - np.median(imfs[0]))) / 0.6745
        threshold = self.threshold_multiplier * sigma * np.sqrt(2 * np.log(len(ecg_signal)))
        
        for i in range(imfs.shape[0]):
            # Apply thresholding primarily to the first few IMFs (high frequency noise)
            # and drop the last IMF to remove baseline wander (trend)
            if i < 3: 
                processed_imfs[i] = self.soft_thresholding(imfs[i], threshold)
            elif i == imfs.shape[0] - 1:
                processed_imfs[i] = np.zeros_like(imfs[i]) # Remove baseline wander
            else:
                processed_imfs[i] = imfs[i]
                
        # 3. Reconstruction
        denoised_signal = np.sum(processed_imfs, axis=0)
        return denoised_signal


def required_segment_length(nperseg: int = DEFAULT_NPERSEG, noverlap: int = DEFAULT_NOVERLAP,
                            n_frames: int = DEFAULT_N_FRAMES) -> int:
    '''
    Segment length that produces exactly `n_frames` STFT frames.

    `stft_proxy_transform` keeps a fixed (n_freq, n_frames) patch because DeepCEDNet expects
    a fixed input tensor. If the signal is longer than this helper returns, the trailing
    frames are silently discarded and `stft_proxy_inverse` cannot reconstruct the tail of the
    waveform, which corrupts every time-domain metric. At nperseg=256 and noverlap=224 the
    hop is 32 samples, so 32 frames correspond to 992 samples, not 1024.
    '''
    hop = nperseg - noverlap
    if hop <= 0:
        raise ValueError('noverlap must be smaller than nperseg')
    return (n_frames - 1) * hop
