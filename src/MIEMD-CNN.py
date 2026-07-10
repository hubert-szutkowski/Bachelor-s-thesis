import numpy as np
import torch
import torch.nn as nn
from PyEMD import EMD



#MIEMD PRE-FILTERING


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


#DEEP CNN DENOISING


class ECGD_Net_CNN(nn.Module):
    '''
    Deep CNN for Stage II Denoising.
    Corrected architecture for 1D signals and regression output.
    Input shape expected: (Batch, Channels=1, Sequence_Length)
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
        
        # Calculate flattened size dynamically
        flattened_size = 32 * (seq_length // 4)
        
        # Fully Connected Layers for Reconstruction
        self.reconstructor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            # Output layer matches the original sequence length for regression! (Not SoftMax 1x2)
            nn.Linear(256, seq_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = self.reconstructor(features)
        # Reshape back to (Batch, Channels, Seq_Length)
        return output.view(-1, 1, self.seq_length)