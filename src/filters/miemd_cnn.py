import numpy as np
import torch
import torch.nn as nn
try:
    from .signal_transforms import MIEMD_Filter
except ImportError:
    from signal_transforms import MIEMD_Filter

#Citation [9]

#MIEMD PRE-FILTERING


#DEEP CNN DENOISING


class ECGD_Net_CNN(nn.Module):
    '''
    Deep CNN for Stage II Denoising.
    Fully convolutional encoder-decoder for 1D signals and regression output.
    Input shape expected: (Batch, Channels=1, Sequence_Length), with Sequence_Length divisible by 4.
    Output shape: identical to the input.
    '''
    def __init__(self, seq_length: int = 1024):
        super().__init__()
        self.seq_length = seq_length
        
        # 5 Convolutional layers with kernel size 5 for optimal balance (as requested)
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length becomes seq_length // 2
            
            # Block 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length becomes seq_length // 4
            
            # Block 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            
            # Block 4
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            
            # Block 5
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU()
        )
        
        self.reconstructor = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding='same')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected input of shape (Batch, 1, Sequence_Length), got {tuple(x.shape)}")
        if x.shape[2] % 4 != 0:
            raise ValueError(f"sequence length must be divisible by 4, got {x.shape[2]}")
        features = self.feature_extractor(x)
        return self.reconstructor(features)
