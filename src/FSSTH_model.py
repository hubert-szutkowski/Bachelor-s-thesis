import torch
import torch.nn as nn
import numpy as np
from scipy import signal
#Citation [10]

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


class ConvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.relu(self.conv(x))))

class DeconvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=3, stride=2, 
            padding=1, output_padding=(0, 1)
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.relu(self.deconv(x))))

class DeepCEDNet(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        
        self.enc1 = ConvBlock2D(2, 8, stride=1, dropout=dropout)
        self.enc2 = ConvBlock2D(8, 8, stride=2, dropout=dropout)
        
        self.enc3 = ConvBlock2D(8, 16, stride=1, dropout=dropout)
        self.enc4 = ConvBlock2D(16, 16, stride=2, dropout=dropout)
        
        self.enc5 = ConvBlock2D(16, 32, stride=1, dropout=dropout)
        self.enc6 = ConvBlock2D(32, 32, stride=2, dropout=dropout)
        
        self.enc7 = ConvBlock2D(32, 64, stride=1, dropout=dropout)
        self.enc8 = ConvBlock2D(64, 64, stride=2, dropout=dropout)
        
        self.enc9 = ConvBlock2D(64, 128, stride=1, dropout=dropout)
        self.enc10 = ConvBlock2D(128, 128, stride=2, dropout=dropout)
        
        self.enc11 = ConvBlock2D(128, 256, stride=1, dropout=dropout)
        
        self.dec1_up = DeconvBlock2D(256, 384, dropout=dropout)
        self.dec1_conv = ConvBlock2D(512, 128, stride=1, dropout=dropout)
        
        self.dec2_up = DeconvBlock2D(128, 64, dropout=dropout)
        self.dec2_conv = ConvBlock2D(128, 64, stride=1, dropout=dropout)
        
        self.dec3_up = DeconvBlock2D(64, 32, dropout=dropout)
        self.dec3_conv = ConvBlock2D(64, 32, stride=1, dropout=dropout)
        
        self.dec4_up = DeconvBlock2D(32, 16, dropout=dropout)
        self.dec4_conv = ConvBlock2D(32, 16, stride=1, dropout=dropout)
        
        self.dec5_up = DeconvBlock2D(16, 8, dropout=dropout)
        self.dec5_conv = ConvBlock2D(16, 8, stride=1, dropout=dropout)
        
        self.final_conv = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        e9 = self.enc9(e8)
        e10 = self.enc10(e9)
        
        bottleneck = self.enc11(e10)
        
        d1 = self.dec1_up(bottleneck)
        d1 = torch.cat([d1, e9], dim=1)
        d1 = self.dec1_conv(d1)
        
        d2 = self.dec2_up(d1)
        d2 = torch.cat([d2, e7], dim=1)
        d2 = self.dec2_conv(d2)
        
        d3 = self.dec3_up(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d3 = self.dec3_conv(d3)
        
        d4 = self.dec4_up(d3)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4_conv(d4)
        
        d5 = self.dec5_up(d4)
        d5 = torch.cat([d5, e1], dim=1)
        d5 = self.dec5_conv(d5)
        
        out = self.final_conv(d5)
        return out