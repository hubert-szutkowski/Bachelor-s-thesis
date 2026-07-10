import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.fft import fft, ifft


#SIGNAL PROCESSING (PRE & POST)


class ECGSignalProcessor:
    '''
    Handles the creation of the Stacked Cardiac Cycle (SCC) tensor 
    and the reconstruction of the 1D signal.
    '''
    def __init__(self, target_length: int = 384, num_cycles: int = 32):
        self.target_length = target_length
        self.num_cycles = num_cycles
        
    def fourier_interpolate(self, cycle: np.ndarray, target_len: int) -> np.ndarray:
        '''Resamples a cycle to a target length using zero-padding in Fourier domain.'''
        N = len(cycle)
        if N == target_len:
            return cycle
            
        freq_domain = fft(cycle)
        if target_len > N:
            # Upsampling: pad with zeros in the middle (high frequencies)
            pad_len = target_len - N
            insert_idx = N // 2 + 1
            padded_freq = np.insert(freq_domain, insert_idx, np.zeros(pad_len))
            resampled = np.real(ifft(padded_freq)) * (target_len / N)
        else:
            # Downsampling: truncate high frequencies
            keep_len = target_len // 2
            truncated_freq = np.concatenate((freq_domain[:keep_len], freq_domain[-keep_len:]))
            if target_len % 2 != 0: # handle odd lengths
                 truncated_freq = np.concatenate((freq_domain[:keep_len+1], freq_domain[-keep_len:]))
            resampled = np.real(ifft(truncated_freq)) * (target_len / N)
        return resampled

    def preprocess(self, ecg_signal: np.ndarray, r_peaks: np.ndarray) -> tuple:
        '''
        Segments ECG into SCC Tensor.
        Returns: SCC Tensor (32, 384), original cycle lengths, global min, global max
        '''
        cycles = []
        original_lengths = []
        
        # Segment between R-peaks (simplified: midpoints between R-peaks)
        for i in range(len(r_peaks) - 1):
            if len(cycles) >= self.num_cycles:
                break
                
            start = r_peaks[i] - (r_peaks[i] - (0 if i==0 else r_peaks[i-1])) // 2
            end = r_peaks[i] + (r_peaks[i+1] - r_peaks[i]) // 2
            
            cycle = ecg_signal[start:end]
            original_lengths.append(len(cycle))
            
            interpolated = self.fourier_interpolate(cycle, self.target_length)
            cycles.append(interpolated)
            
        # Pad with zeros if less than required cycles (handling edge case)
        while len(cycles) < self.num_cycles:
            cycles.append(np.zeros(self.target_length))
            original_lengths.append(0)
            
        scc_matrix = np.vstack(cycles)
        
        # Min-Max Normalization
        g_min = np.min(scc_matrix)
        g_max = np.max(scc_matrix)
        if g_max > g_min:
            scc_matrix = (scc_matrix - g_min) / (g_max - g_min)
            
        return scc_matrix, original_lengths, g_min, g_max

    def postprocess(self, scc_matrix: np.ndarray, original_lengths: list, g_min: float, g_max: float) -> np.ndarray:
        '''Reconstructs the 1D ECG signal from the denoised SCC Tensor.'''
        # Denormalize
        scc_matrix = scc_matrix * (g_max - g_min) + g_min
        
        reconstructed_1d = []
        for i in range(min(self.num_cycles, len(original_lengths))):
            orig_len = original_lengths[i]
            if orig_len == 0:
                continue
            
            denoised_cycle = scc_matrix[i, :]
            restored_cycle = self.fourier_interpolate(denoised_cycle, orig_len)
            reconstructed_1d.extend(restored_cycle)
            
        return np.array(reconstructed_1d)


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


