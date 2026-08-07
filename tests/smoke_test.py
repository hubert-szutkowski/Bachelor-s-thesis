import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from wavelet_model import WaveletCNNAutoencoder
from sced_net_model import SCED_Net, ECGSignalProcessor
from wavelet_transformers import CSTRANS_Denoising, ImprovedMSELoss
from miemd_cnn import ECGD_Net_CNN
from FSSTH_model import DeepCEDNet, stft_proxy_transform, stft_proxy_inverse


def report(name, expected, actual, params=None):
    status = 'OK  ' if tuple(actual) == tuple(expected) else 'FAIL'
    extra = f'  params={params/1e6:8.3f} M' if params is not None else ''
    print(f'  {status} {name:22s} {tuple(actual)} expected {tuple(expected)}{extra}')
    return status == 'OK  '


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_models():
    print('=== forward pass ===')
    torch.manual_seed(0)
    ok = True

    cases = [
        ('WaveletCNNAutoencoder', WaveletCNNAutoencoder(), (2, 1, 1024)),
        ('SCED_Net', SCED_Net(), (2, 1, 32, 384)),
        ('CSTRANS_Denoising', CSTRANS_Denoising(), (2, 1, 2048)),
        ('ECGD_Net_CNN', ECGD_Net_CNN(), (2, 1, 1024)),
        ('DeepCEDNet', DeepCEDNet(), (2, 2, 129, 32)),
    ]

    for name, model, shape in cases:
        model.eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x)
        ok &= report(name, shape, y.shape, count_params(model))

    print('\n=== backward pass ===')
    for name, model, shape in cases:
        model.train()
        x = torch.randn(*shape, requires_grad=False)
        target = torch.randn(*shape)
        loss = ImprovedMSELoss()(model(x), target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        finite = all(g is not None and torch.isfinite(g).all() for g in grads)
        print(f'  {"OK  " if finite else "FAIL"} {name:22s} loss={loss.item():.4f} finite grads={finite}')
        ok &= finite

    return ok


def test_scc_roundtrip():
    print('\n=== SCC round trip ===')
    fs = 360
    n_beats = 40
    rr = int(0.8 * fs)
    signal = np.zeros(n_beats * rr)
    r_peaks = np.arange(n_beats) * rr + rr // 2
    for r in r_peaks:
        signal[r - 5:r + 5] += np.hanning(10) * 2.0
    signal += 0.1 * np.sin(2 * np.pi * 5 * np.arange(signal.size) / fs)

    proc = ECGSignalProcessor(target_length=384, num_cycles=32)
    scc, lengths, g_min, g_max = proc.preprocess(signal, r_peaks)
    rec = proc.postprocess(scc, lengths, g_min, g_max)

    covered = sum(lengths)
    err = np.abs(rec - signal[:rec.size]).max()
    print(f'  scc shape {scc.shape}, range [{scc.min():.3f}, {scc.max():.3f}]')
    print(f'  cycles used {int(np.sum(np.array(lengths) > 0))}/32, samples covered {covered}')
    print(f'  {"OK  " if err < 1e-6 else "FAIL"} max reconstruction error {err:.3e}')

    scc_t, _, _, _ = proc.preprocess(signal * 3.0, r_peaks, stats=(g_min, g_max))
    shared = np.allclose((scc_t * (g_max - g_min) + g_min)[np.array(lengths) > 0],
                         (scc * (g_max - g_min) + g_min)[np.array(lengths) > 0] * 3.0, atol=1e-8)
    print(f'  {"OK  " if shared else "FAIL"} shared normalization statistics honoured')
    return err < 1e-6 and shared


def test_stft_roundtrip():
    print('\n=== STFT proxy round trip ===')
    fs = 250
    x = np.random.RandomState(0).randn(1024)
    tf, meta = stft_proxy_transform(x, fs=fs)
    rec = stft_proxy_inverse(tf, meta)
    err = np.abs(rec - x).max()
    print(f'  tf shape {tf.shape}, dtype {tf.dtype}')
    print(f'  {"OK  " if err < 1e-6 else "INFO"} max reconstruction error {err:.3e}')
    return tf.shape == (2, 129, 32) and rec.shape == x.shape


if __name__ == '__main__':
    results = [test_models(), test_scc_roundtrip(), test_stft_roundtrip()]
    print(f'\n{"ALL PASSED" if all(results) else "FAILURES PRESENT"}')
    sys.exit(0 if all(results) else 1)
