import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks, resample

#Citation [7]
#SIGNAL PROCESSING (PRE & POST)


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


#SCED-Net NEURAL ARCHITECTURE


class LNC_Block(nn.Module):
    '''
    Local/Non-local Cycle observation block.
    Uses parallel 2D convolutions with dilation only in the cycle dimension (height).
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # We need to maintain spatial dimensions.
        # Kernel is (3, 7). Height (cycle dimension) padding depends on dilation.
        # Width (time dimension) padding is always 3 to maintain dimension with kernel=7.
        
        # Branch 1: Dilation (1, 1) -> standard conv. Padding needed for H=3 is 1.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), 
                               dilation=(1, 1), padding=(1, 3))
        
        # Branch 2: Dilation (2, 1). Padding needed for H=3 with dilation 2 is 2.
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), 
                               dilation=(2, 1), padding=(2, 3))
        
        # Branch 3: Dilation (4, 1). Padding needed for H=3 with dilation 4 is 4.
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), 
                               dilation=(4, 1), padding=(4, 3))
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 1x1 Convolution to reduce channels after concatenation (3 * out_channels -> out_channels)
        self.reduce_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
        # Skip connection adapter if channel dimensions change
        self.skip_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        b2 = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        b3 = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        
        # Concatenate along channel dimension
        merged = torch.cat([b1, b2, b3], dim=1)
        out = self.reduce_conv(merged)
        
        # Residual connection
        return out + self.skip_adapter(x)


class SCED_Net(nn.Module):
    '''
    Full Encoder-Decoder Architecture for SCC Tensor Denoising.
    Input shape: (Batch, Channels=1, Cycles=32, Time=384)
    '''
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.enc1 = LNC_Block(1, 16)
        self.enc2 = LNC_Block(16, 32)
        self.enc3 = LNC_Block(32, 64)
        self.enc4 = LNC_Block(64, 128)
        
        # Max Pooling (reduces ONLY time dimension by half: 1x2)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # --- LATENT SPACE ---
        self.bottleneck = LNC_Block(128, 256)
        
        # --- DECODER ---
        # Up-sampling only in time dimension (1x2)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2))
        self.dec4 = LNC_Block(128 + 128, 128) # +128 from skip connection
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=(1, 2), stride=(1, 2))
        self.dec3 = LNC_Block(64 + 64, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.dec2 = LNC_Block(32 + 32, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=(1, 2), stride=(1, 2))
        self.dec1 = LNC_Block(16 + 16, 16)
        
        # Output layer to restore 1 channel
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder phase with skip connections saved
        e1 = self.enc1(x)         # [B, 16, 32, 384]
        p1 = self.pool(e1)        # [B, 16, 32, 192]
        
        e2 = self.enc2(p1)        # [B, 32, 32, 192]
        p2 = self.pool(e2)        # [B, 32, 32, 96]
        
        e3 = self.enc3(p2)        # [B, 64, 32, 96]
        p3 = self.pool(e3)        # [B, 64, 32, 48]
        
        e4 = self.enc4(p3)        # [B, 128, 32, 48]
        p4 = self.pool(e4)        # [B, 128, 32, 24]
        
        # Bottleneck
        bn = self.bottleneck(p4)  # [B, 256, 32, 24]
        
        # Decoder phase with concatenation (U-Net style)
        u4 = self.up4(bn)         # [B, 128, 32, 48]
        u4 = torch.cat([u4, e4], dim=1) # [B, 256, 32, 48]
        d4 = self.dec4(u4)        # [B, 128, 32, 48]
        
        u3 = self.up3(d4)         # [B, 64, 32, 96]
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)        # [B, 64, 32, 96]
        
        u2 = self.up2(d3)         # [B, 32, 32, 192]
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)        # [B, 32, 32, 192]
        
        u1 = self.up1(d2)         # [B, 16, 32, 384]
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)        # [B, 16, 32, 384]
        
        out = self.out_conv(d1)   # [B, 1, 32, 384]
        return out


