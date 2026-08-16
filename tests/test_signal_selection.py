'''
Contract tests for the representation layer.

Runs without PyTorch: `signal_selection` imports the architectures lazily, so the encode /
decode pairs can be verified on any machine.
'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from train.signal_selection import available_models, describe, select_signal


def synthetic_ecg(n_beats=40, fs=360, bpm=75, seed=0):
    rng = np.random.RandomState(seed)
    rr = int(60.0 / bpm * fs)
    signal = np.zeros(n_beats * rr)
    r_peaks = np.arange(n_beats) * rr + rr // 2
    for r in r_peaks:
        signal[r - 5:r + 5] += np.hanning(10) * 2.0
        signal[r + 20:r + 60] += np.hanning(40) * 0.3
    signal += 0.05 * rng.randn(signal.size)
    return signal, r_peaks


def test_registry():
    print('=== registry ===')
    for name in available_models():
        print('  ' + describe(name))
    return len(available_models()) == 5


def test_shapes_and_roundtrip():
    print('\n=== encode / decode contract ===')
    fs = 360
    signal, r_peaks = synthetic_ecg(fs=fs)
    ok = True

    for name in available_models():
        spec = select_signal(name, fs=fs)

        if spec.requires_r_peaks:
            segment, peaks = signal, r_peaks
        else:
            segment, peaks = signal[:spec.segment_length], None

        encoded, meta = spec.encode(segment, r_peaks=peaks)
        decoded = spec.decode(encoded, meta)

        shape_ok = encoded.shape == tuple(spec.tensor_shape)
        dtype_ok = encoded.dtype == np.float32

        reference_full = spec.reference_waveform(segment, r_peaks=peaks)
        n = min(decoded.size, reference_full.size)
        reference = reference_full[:n]
        error = np.abs(decoded[:n] - reference).max()
        scale = np.abs(reference).max()
        relative = error / scale if scale > 0 else error
        roundtrip_ok = relative < 0.05

        determinism_ok = np.array_equal(encoded, spec.encode(segment, r_peaks=peaks)[0])

        passed = shape_ok and dtype_ok and roundtrip_ok and determinism_ok
        ok &= passed
        lossy = ' (vs MIEMD output)' if spec.is_lossy else ''
        print(f'  {"OK  " if passed else "FAIL"} {name:12s} shape={encoded.shape} '
              f'dtype={encoded.dtype} rel_err={relative:.2e} '
              f'deterministic={determinism_ok}{lossy}')

    return ok


def test_shared_statistics():
    print('\n=== shared normalization statistics ===')
    fs = 360
    signal, r_peaks = synthetic_ecg(fs=fs)
    ok = True

    for name in available_models():
        if name == 'ecgd_net':
            continue

        spec = select_signal(name, fs=fs)
        if spec.requires_r_peaks:
            noisy, peaks = signal, r_peaks
        else:
            noisy, peaks = signal[:spec.segment_length], None
        clean = noisy * 0.5

        x, meta = spec.encode(noisy, r_peaks=peaks)
        y, _ = spec.encode(clean, r_peaks=peaks, stats=meta['stats'])

        decoded_x = spec.decode(x, meta)
        decoded_y = spec.decode(y, meta)
        n = min(decoded_x.size, decoded_y.size)
        ratio = np.median(np.abs(decoded_y[:n]) / (np.abs(decoded_x[:n]) + 1e-9))

        passed = abs(ratio - 0.5) < 0.05
        ok &= passed
        print(f'  {"OK  " if passed else "FAIL"} {name:12s} '
              f'target/input amplitude ratio = {ratio:.4f} (expected 0.5)')

    return ok


def test_cache_budget():
    print('\n=== cache footprint per 10 000 segments ===')
    for name in available_models():
        spec = select_signal(name)
        total = spec.bytes_per_sample * 10000 * 2
        print(f'  {name:12s} {spec.bytes_per_sample / 1024:7.1f} KiB/sample '
              f'-> {total / 1024 ** 3:6.3f} GiB (input + target)')
    return True


if __name__ == '__main__':
    results = [
        test_registry(),
        test_shapes_and_roundtrip(),
        test_shared_statistics(),
        test_cache_budget(),
    ]
    print(f'\n{"ALL PASSED" if all(results) else "FAILURES PRESENT"}')
    sys.exit(0 if all(results) else 1)
