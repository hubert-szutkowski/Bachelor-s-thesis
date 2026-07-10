import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class DWT_1D_Layer(nn.Module):
    '''
    Discrete Wavelet Transform Layer.
    Splits features into Low-Pass (LPF) and High-Pass (HPF) components with down-sampling by 2.
    Then applies separate convolutions and concatenates the results.
    '''
    def __init__(self, in_channels: int, out_channels: int, wavelet_name: str = 'db6'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Load wavelet coefficients
        wavelet = pywt.Wavelet(wavelet_name)
        # Reverse coefficients for cross-correlation used in PyTorch Conv1d
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        
        # Reshape to (out_channels, in_channels/groups, kernel_size) for depthwise convolution
        # We use groups=in_channels to apply the same filter to each channel independently
        self.register_buffer('filter_lpf', dec_lo.view(1, 1, -1).repeat(in_channels, 1, 1))
        self.register_buffer('filter_hpf', dec_hi.view(1, 1, -1).repeat(in_channels, 1, 1))
        
        # Calculate padding to halve the spatial dimension for wavelet length 12
        self.pad_dwt = 5 
        
        # Separate convolutions for LPF and HPF branches (stride=1 to maintain dimension)
        branch_out_channels = out_channels // 2
        
        self.conv_lpf = nn.Conv1d(in_channels, branch_out_channels, kernel_size=8, stride=1, padding='same')
        self.conv_hpf = nn.Conv1d(in_channels, branch_out_channels, kernel_size=8, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise 1D Convolution for Wavelet Decomposition (Downsampling by 2)
        x_lpf = F.conv1d(x, self.filter_lpf, stride=2, groups=self.in_channels, padding=self.pad_dwt)
        x_hpf = F.conv1d(x, self.filter_hpf, stride=2, groups=self.in_channels, padding=self.pad_dwt)
        
        # Feature extraction on each frequency band
        feat_lpf = self.conv_lpf(x_lpf)
        feat_hpf = self.conv_hpf(x_hpf)
        
        # Concatenate along the channel dimension
        out = torch.cat([feat_lpf, feat_hpf], dim=1)
        return out


class IDWT_1D_Layer(nn.Module):
    '''
    Inverse Discrete Wavelet Transform Layer.
    Applies Deconvolution, slices features, and reconstructs using up-sampled HPF/LPF filters.
    '''
    def __init__(self, in_channels: int, out_channels: int, wavelet_name: str = 'db6'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        wavelet = pywt.Wavelet(wavelet_name)
        # Reconstruction filters
        rec_lo = torch.tensor(wavelet.rec_lo[::-1], dtype=torch.float32)
        rec_hi = torch.tensor(wavelet.rec_hi[::-1], dtype=torch.float32)
        
        self.register_buffer('filter_rec_lpf', rec_lo.view(1, 1, -1).repeat(out_channels, 1, 1))
        self.register_buffer('filter_rec_hpf', rec_hi.view(1, 1, -1).repeat(out_channels, 1, 1))
        
        self.pad_idwt = 5
        
        # Initial Deconvolution to process features before reconstruction
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels * 2, kernel_size=8, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Feature processing
        x_processed = self.deconv(x)
        
        # 2. Slice into two paths
        feat_lpf = x_processed[:, :self.out_channels, :]
        feat_hpf = x_processed[:, self.out_channels:, :]
        
        # 3. Upsampling and Reconstruction filtering via transposed convolution
        y_lpf = F.conv_transpose1d(feat_lpf, self.filter_rec_lpf, stride=2, groups=self.out_channels, padding=self.pad_idwt)
        y_hpf = F.conv_transpose1d(feat_hpf, self.filter_rec_hpf, stride=2, groups=self.out_channels, padding=self.pad_idwt)
        
        # 4. Add results
        out = y_lpf + y_hpf
        return out


class ConvBlock(nn.Module):
    '''Standard Convolutional Block with ELU, BatchNorm and Dropout.'''
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int | str, is_transpose: bool = False):
        super().__init__()
        layers = []
        if is_transpose:
            # output_padding=0 works perfectly for halving/doubling with these specific kernels
            layers.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=k, stride=s, padding=p))
        else:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p))
            
        layers.extend([
            nn.ELU(),
            nn.BatchNorm1d(out_c),
            nn.Dropout(p=0.10)
        ])
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletCNNAutoencoder(nn.Module):
    '''
    Backward-type FCN Autoencoder for ECG Denoising.
    Input: (Batch, 1, 1024)
    '''
    def __init__(self):
        super().__init__()
        
        # ENCODER
        # Shallow layers: Standard Convolutions
        self.enc1 = ConvBlock(1, 40, k=16, s=2, p=7)   # Out: 40 x 512
        self.enc2 = ConvBlock(40, 20, k=16, s=2, p=7)  # Out: 20 x 256
        
        # Deep layers: DWT Blocks
        self.enc3 = nn.Sequential(DWT_1D_Layer(20, 20), nn.ELU(), nn.BatchNorm1d(20), nn.Dropout(0.1)) # Out: 20 x 128
        self.enc4 = nn.Sequential(DWT_1D_Layer(20, 20), nn.ELU(), nn.BatchNorm1d(20), nn.Dropout(0.1)) # Out: 20 x 64
        self.enc5 = nn.Sequential(DWT_1D_Layer(20, 40), nn.ELU(), nn.BatchNorm1d(40), nn.Dropout(0.1)) # Out: 40 x 32
        
        # Bottleneck (Enc 6)
        self.enc6 = ConvBlock(40, 1, k=15, s=1, p='same') # Out: 1 x 32
        
        # DECODER
        # Bottleneck (Dec 1)
        self.dec1 = ConvBlock(1, 1, k=15, s=1, p='same') # Out: 1 x 32
        
        # Deep layers: IDWT Blocks (matching Encoder's DWT layers)
        self.dec2 = nn.Sequential(IDWT_1D_Layer(1, 40), nn.ELU(), nn.BatchNorm1d(40), nn.Dropout(0.1)) # Out: 40 x 64
        self.dec3 = nn.Sequential(IDWT_1D_Layer(40, 20), nn.ELU(), nn.BatchNorm1d(20), nn.Dropout(0.1)) # Out: 20 x 128
        self.dec4 = nn.Sequential(IDWT_1D_Layer(20, 20), nn.ELU(), nn.BatchNorm1d(20), nn.Dropout(0.1)) # Out: 20 x 256
        
        # Shallow layers: Standard Transposed Convolutions
        self.dec5 = ConvBlock(20, 20, k=16, s=2, p=7, is_transpose=True) # Out: 20 x 512
        self.dec6 = ConvBlock(20, 40, k=16, s=2, p=7, is_transpose=True) # Out: 40 x 1024
        
        # Output layer (No activation or dropout for signal reconstruction)
        self.out_layer = nn.Conv1d(40, 1, k=15, s=1, padding='same')     # Out: 1 x 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        
        # Output
        x = self.out_layer(x)
        return x