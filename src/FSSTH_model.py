import torch
import torch.nn as nn
import numpy as np
from scipy import signal

def fssth_transform_mock(ecg_signal: np.ndarray) -> np.ndarray:
    '''
    Placeholder for High-order Synchrosqueezing Transform (FSSTH).
    Standard STFT is used here as a structural proxy to match tensor dimensions.
    '''
    f, t, Zxx = signal.stft(ecg_signal, fs=250, nperseg=256, noverlap=224)
    
    real_part = np.real(Zxx[:129, :32])
    imag_part = np.imag(Zxx[:129, :32])
    
    t_f_matrix = np.stack((real_part, imag_part), axis=0)
    
    mean = np.mean(t_f_matrix, axis=(1, 2), keepdims=True)
    std = np.std(t_f_matrix, axis=(1, 2), keepdims=True)
    t_f_matrix = (t_f_matrix - mean) / (std + 1e-8)
    
    return t_f_matrix

class ConvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.relu(self.conv(x))))

class DeconvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=3, stride=2, 
            padding=1, output_padding=(0, 1)
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.relu(self.deconv(x))))

class DeepCEDNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = ConvBlock2D(2, 8, stride=1)
        self.enc2 = ConvBlock2D(8, 8, stride=2)
        
        self.enc3 = ConvBlock2D(8, 16, stride=1)
        self.enc4 = ConvBlock2D(16, 16, stride=2)
        
        self.enc5 = ConvBlock2D(16, 32, stride=1)
        self.enc6 = ConvBlock2D(32, 32, stride=2)
        
        self.enc7 = ConvBlock2D(32, 64, stride=1)
        self.enc8 = ConvBlock2D(64, 64, stride=2)
        
        self.enc9 = ConvBlock2D(64, 128, stride=1)
        self.enc10 = ConvBlock2D(128, 128, stride=2)
        
        self.enc11 = ConvBlock2D(128, 256, stride=1)
        
        self.dec1_up = DeconvBlock2D(256, 384)
        self.dec1_conv = ConvBlock2D(512, 128, stride=1)
        
        self.dec2_up = DeconvBlock2D(128, 64)
        self.dec2_conv = ConvBlock2D(128, 64, stride=1)
        
        self.dec3_up = DeconvBlock2D(64, 32)
        self.dec3_conv = ConvBlock2D(64, 32, stride=1)
        
        self.dec4_up = DeconvBlock2D(32, 16)
        self.dec4_conv = ConvBlock2D(32, 16, stride=1)
        
        self.dec5_up = DeconvBlock2D(16, 8)
        self.dec5_conv = ConvBlock2D(16, 8, stride=1)
        
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