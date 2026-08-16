import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Citation [8]

class CNNSWT_1D(nn.Module):
    '''
    CNN-SWT Layer.
    Uses a single trainable kernel to generate orthogonal wavelet-like filters 
    (reversal, alternating signs) to drastically reduce parameter count.
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # Output channels must be divisible by 4.
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for CNN-SWT"
        self.base_out = out_channels // 4
        
        # Single trainable kernel
        self.weight = nn.Parameter(torch.Tensor(self.base_out, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        
        self.padding = kernel_size // 2

        # Alternating sign mask
        alt_signs = torch.ones(kernel_size)
        alt_signs[1::2] = -1
        self.register_buffer('alt_signs', alt_signs.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base kernel
        w_base = self.weight
        
        # Reversed kernel
        w_rev = torch.flip(w_base, dims=[2])
        
        # Alternating-sign kernel
        w_alt = w_base * self.alt_signs
        
        # Reversed alternating kernel
        w_rev_alt = w_rev * self.alt_signs
        
        # Concatenate filters
        w_combined = torch.cat([w_base, w_rev, w_alt, w_rev_alt], dim=0)
        
        # Single convolution
        return F.conv1d(x, w_combined, bias=self.bias, padding=self.padding)


class Time2Vec(nn.Module):
    '''
    Time2Vec Positional Encoding.
    E_{i,j} = X_{i,j} + w_j * i + phi_j (Linear)
    E_{i,j} = X_{i,j} + sin(w_j * i + phi_j) (Periodic)
    '''
    def __init__(self, seq_len: int, features: int):
        super().__init__()
        # Linear component
        self.w_linear = nn.Parameter(torch.Tensor(seq_len, features))
        self.phi_linear = nn.Parameter(torch.Tensor(seq_len, features))
        
        # Periodic component
        self.w_periodic = nn.Parameter(torch.Tensor(seq_len, features))
        self.phi_periodic = nn.Parameter(torch.Tensor(seq_len, features))
        
        nn.init.uniform_(self.w_linear, -0.1, 0.1)
        nn.init.uniform_(self.phi_linear, -0.1, 0.1)
        nn.init.uniform_(self.w_periodic, -0.1, 0.1)
        nn.init.uniform_(self.phi_periodic, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, Seq, Features)
        # Time index encoding
        time_seq = (torch.arange(x.size(1), device=x.device, dtype=x.dtype) / max(x.size(1) - 1, 1)).unsqueeze(1).expand(-1, x.size(2))
        
        t_linear = self.w_linear * time_seq + self.phi_linear
        t_periodic = torch.sin(self.w_periodic * time_seq + self.phi_periodic)
        
        # Add encoding
        encoding = t_linear + t_periodic
        return x + encoding


class TransformerEncoderBlock(nn.Module):
    '''2-Layer Transformer Encoder with 16 heads, 64-node FFN, BN, and Dropout.'''
    def __init__(self, d_model: int, nhead: int = 16, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        # BatchNorm instead of LayerNorm
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.bn1 = nn.BatchNorm1d(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head Attention + Residual + BatchNorm
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        # BN expects (B, Channels, Seq) so we permute
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        
        # Feed-Forward + Residual + BatchNorm
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        return x


class ConvBlock(nn.Module):
    '''Standard Convolutional Block for Encoder/Decoder.'''
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_c)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class CSTRANS_Denoising(nn.Module):
    '''
    U-Net style architecture for ECG Denoising integrating CNN-SWT and Transformer.
    Input: (Batch, 1, 2048)
    '''
    def __init__(self):
        super().__init__()
        
        self.enc1 = ConvBlock(1, 16)      # Out: (16, 2048)
        self.enc2 = ConvBlock(16, 32)     # Out: (32, 1024)
        
        self.enc3_conv = ConvBlock(32, 64)
        self.drop3 = nn.Dropout(0.5)      # Out: (64, 512)
        
        self.enc4_conv = ConvBlock(64, 64)
        self.drop4 = nn.Dropout(0.5)      # Out: (64, 256)
        
        self.pool = nn.MaxPool1d(2)
        
        # CNN-SWT + TRANSFORMER (Bottleneck)
        self.cnn_swt = CNNSWT_1D(in_channels=64, out_channels=64)
        self.time2vec = Time2Vec(seq_len=256, features=64)
        
        self.transformer = nn.Sequential(
            TransformerEncoderBlock(d_model=64, nhead=16, dim_feedforward=64, dropout=0.1),
            TransformerEncoderBlock(d_model=64, nhead=16, dim_feedforward=64, dropout=0.1)
        )

        # DECODER
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = ConvBlock(128, 64)    # 64 (Trans) + 64 (Enc4)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ConvBlock(128, 32)    # 64 (Dec4) + 64 (Enc3)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ConvBlock(64, 16)     # 32 (Dec3) + 32 (Enc2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(32, 16)     # 16 (Dec2) + 16 (Enc1)
        
        self.out_dense = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        e1 = self.enc1(x)                                # (B, 16, 2048)
        p1 = self.pool(e1)                               # (B, 16, 1024)
        
        e2 = self.enc2(p1)                               # (B, 32, 1024)
        p2 = self.pool(e2)                               # (B, 32, 512)
        
        e3 = self.drop3(self.enc3_conv(p2))              # (B, 64, 512)
        p3 = self.pool(e3)                               # (B, 64, 256)
        
        e4 = self.drop4(self.enc4_conv(p3))              # (B, 64, 256) - To be concatenated
        
        
        swt_out = self.cnn_swt(e4)                       # (B, 64, 256)
        
        # (B, Seq, Features)
        trans_in = swt_out.transpose(1, 2)
        t2v_out = self.time2vec(trans_in)
        trans_out = self.transformer(t2v_out)
        
        # Back to (B, C, Seq)
        bottleneck = trans_out.transpose(1, 2)           # (B, 64, 256)
        
        
        d4 = torch.cat([bottleneck, e4], dim=1)          # (B, 128, 256)
        d4 = self.dec4(d4)                               # (B, 64, 256)
        d4 = self.up4(d4)                                # (B, 64, 512)
        
        d3 = torch.cat([d4, e3], dim=1)                  # (B, 128, 512)
        d3 = self.dec3(d3)                               # (B, 32, 512)
        d3 = self.up3(d3)                                # (B, 32, 1024)
        
        d2 = torch.cat([d3, e2], dim=1)                  # (B, 64, 1024)
        d2 = self.dec2(d2)                               # (B, 16, 1024)
        d2 = self.up2(d2)                                # (B, 16, 2048)
        
        d1 = torch.cat([d2, e1], dim=1)                  # (B, 32, 2048)
        d1 = self.dec1(d1)                               # (B, 16, 2048)
        
        out = self.out_dense(d1)                         # (B, 1, 2048)
        return out


class CSTRANS_Classification(nn.Module):
    '''
    Classification Module processing 3 consecutive beats and merging with R-R intervals.
    Inputs: 3x (Batch, 1, 360) arrays, and 1x (Batch, 4) R-R intervals.
    '''
    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Shared feature extractor
        self.shared_conv = nn.Sequential(
            ConvBlock(1, 16),
            nn.MaxPool1d(2),          # 180
            ConvBlock(16, 32),
            nn.MaxPool1d(2),          # 90
            ConvBlock(32, 64),
            nn.Dropout(0.5)           # (64, 90)
        )
        
        self.shared_swt = CNNSWT_1D(in_channels=64, out_channels=64)
        self.shared_time2vec = Time2Vec(seq_len=90, features=64)
        self.shared_transformer = nn.Sequential(
            TransformerEncoderBlock(d_model=64, nhead=16, dim_feedforward=64, dropout=0.1),
            TransformerEncoderBlock(d_model=64, nhead=16, dim_feedforward=64, dropout=0.1)
        )
        
        # 90 * 64 = 5760
        self.flatten = nn.Flatten()
        self.dense_feat = nn.Linear(5760, 64)
        
        # 3 beats + 4 RR intervals
        self.classifier = nn.Sequential(
            nn.Linear(196, 66),
            nn.ReLU(),
            nn.Linear(66, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        '''Shared feature extraction.'''
        c = self.shared_conv(x)
        s = self.shared_swt(c)
        
        t_in = s.transpose(1, 2)
        t2v = self.shared_time2vec(t_in)
        t_out = self.shared_transformer(t2v)
        
        flat = self.flatten(t_out)
        return self.dense_feat(flat)

    def forward(self, beat1, beat2, beat3, rr_intervals) -> torch.Tensor:
        f1 = self.extract_features(beat1)
        f2 = self.extract_features(beat2)
        f3 = self.extract_features(beat3)
        
        # Merge features and RR intervals
        merged = torch.cat([f1, f2, f3, rr_intervals], dim=1) # (B, 196)
        
        logits = self.classifier(merged)
        return logits # Softmax handled by loss



class ImprovedMSELoss(nn.Module):
    '''
    MSE Loss weighted by Peak Factor (C_p) to preserve amplitudes of P, QRS, and T waves.
    '''
    def __init__(self, peak_weight: float = 2.0):
        super().__init__()
        self.peak_weight = peak_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_mse = self.mse(pred, target)
        
        # Peak-weighted error
        # Target amplitude as weight mask
        peak_factor = 1.0 + (self.peak_weight * torch.abs(target))
        
        improved_mse = base_mse * peak_factor
        return improved_mse.mean()


class FocalLoss(nn.Module):
    '''
    Focal Loss to handle highly imbalanced datasets like AAMI (N, S, V, F, Q).
    '''
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

