"""
The seven static methods, adapted to the registry.

Each adapter does three things and nothing else: it takes the sampling frequency from the
context rather than from the parameter file, converts window lengths given in
milliseconds into samples, and calls the underlying implementation. Keeping them thin is
deliberate, so that what is measured is the method as published rather than an adapter
that improved it.

Window lengths are configured in milliseconds rather than in samples. Everything in this
work runs at 360 Hz, so the two are interchangeable today, but a length in milliseconds
states what the filter does — a moving average over 25 ms smooths half a QRS complex —
where a length of nine samples states only how it is implemented. It also means that a
change of rate later moves no parameter: nine samples smooth 25 ms at 360 Hz and 36 ms at
250 Hz, and nothing in a length expressed in samples would announce the difference.

Wavelet baseline removal carries a guard the others do not need. The number of
decomposition levels a window admits is bounded by its length, and removing the
approximation after `level` levels removes everything below fs / 2^(level+1). A window of
1024 samples at 360 Hz admits six levels with `db8`, which puts that boundary at 2.8 Hz,
inside the P and T waves; the method would then remove the baseline together with a good
part of the beat and report a plausible waveform while doing it. The adapter computes the
boundary it will actually achieve and refuses when it lands above `max_cut_hz`, which is
also the reason the static filters are measured on windows of 4096 samples rather than
1024.
"""

import numpy as np
import pywt

try:
    from .registry import register
    from .static_filters import (
        emd_ecg_denoising,
        fir_filter,
        iir_filter,
        moving_average,
        moving_median,
        wavelet_baseline_removal,
        wavelet_denoising,
    )
except ImportError:
    from registry import register
    from static_filters import (
        emd_ecg_denoising,
        fir_filter,
        iir_filter,
        moving_average,
        moving_median,
        wavelet_baseline_removal,
        wavelet_denoising,
    )

DEFAULT_MAX_CUT_HZ = 1.0

# Szerokosc pasma przejsciowego okna FIR wyraza sie jako df = k * fs / N.
# Wspolczynniki k dla typowych okien.
TRANSITION_CONSTANT = {'rectangular': 0.9, 'boxcar': 0.9, 'hann': 3.1, 'hanning': 3.1,
                       'hamming': 3.3, 'blackman': 5.5, 'bartlett': 2.9}
FILTFILT_PAD_FACTOR = 3


def _odd_samples(milliseconds: float, fs: float) -> int:
    """Window length in samples, forced odd so that the window has a centre sample."""
    samples = int(round(milliseconds * 1e-3 * fs))
    samples = max(1, samples)
    return samples if samples % 2 else samples + 1


def fir_transition_hz(numtaps: int, fs: float, window: str = 'hamming') -> dict:
    """
    The narrowest transition a windowed FIR filter of this length can produce.

    A band edge placed below that width has no stopband beneath it: the filter simply
    passes what it was meant to reject. At 360 Hz a Hamming design of 101 coefficients
    spans 11.8 Hz, so a lower edge of 0.5 Hz leaves the baseline drift almost untouched
    while the frequency response plot still shows the requested band. Lengthening the
    filter is the only remedy, and the length is bounded in turn by the window it has to
    fit inside.
    """
    numtaps = int(numtaps)
    constant = TRANSITION_CONSTANT.get(str(window).lower(), 3.3)
    return {
        'numtaps': numtaps,
        'transition_hz': float(constant * fs / numtaps),
        'padlen': FILTFILT_PAD_FACTOR * numtaps,
        'min_samples': FILTFILT_PAD_FACTOR * numtaps + 1,
    }


def baseline_cut_hz(n_samples: int, fs: float, wavelet: str, level: int) -> dict:
    """
    The frequency below which wavelet baseline removal will actually cut.

    `level` is what was asked for, `effective` what the window admits, and `cut_hz` the
    boundary that follows from it. Reported rather than merely enforced, because the
    number belongs in the methodology chapter.
    """
    admitted = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
    effective = min(int(level), int(admitted))
    return {
        'requested_level': int(level),
        'max_level': int(admitted),
        'effective_level': effective,
        'cut_hz': float(fs / 2 ** (effective + 1)),
    }


# --- adapters ------------------------------------------------------------

def _fir_bandpass(signal, context, low_hz=0.5, high_hz=40.0, numtaps_ms=1700.0,
                  window='hamming', zero_phase=True, **_):
    numtaps = _odd_samples(numtaps_ms, context.fs)
    report = fir_transition_hz(numtaps, context.fs, window)

    if zero_phase and signal.size <= report['padlen']:
        raise ValueError(
            f"fir_bandpass: {numtaps} coefficients need at least {report['min_samples']} "
            f"samples for zero phase filtering, and the window holds {signal.size}; "
            f"shorten numtaps_ms or use a longer window")

    return fir_filter(signal, numtaps=numtaps, cutoff=[low_hz, high_hz],
                      window=window, pass_zero=False, fs=context.fs,
                      zero_phase=zero_phase)


def _iir_bandpass(signal, context, order=4, low_hz=0.5, high_hz=40.0,
                  ftype='butter', zero_phase=True, rp=None, rs=None, **_):
    return iir_filter(signal, order=order, Wn=[low_hz, high_hz], btype='bandpass',
                      ftype=ftype, fs=context.fs, zero_phase=zero_phase, rp=rp, rs=rs)


def _moving_average(signal, context, window_ms=25.0, **_):
    return moving_average(signal, smooth_interval=_odd_samples(window_ms, context.fs))


def _moving_median(signal, context, window_ms=60.0, **_):
    return moving_median(signal, window_size=_odd_samples(window_ms, context.fs))


def _wavelet_denoising(signal, context, wavelet='db4', level=5, mode='soft', **_):
    admitted = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    return wavelet_denoising(signal, wavelet=wavelet, level=min(int(level), admitted),
                             mode=mode)


def _wavelet_baseline(signal, context, wavelet='db8', level=9,
                      max_cut_hz=DEFAULT_MAX_CUT_HZ, **_):
    report = baseline_cut_hz(signal.size, context.fs, wavelet, level)
    if report['cut_hz'] > max_cut_hz:
        raise ValueError(
            f"wavelet_baseline: a window of {signal.size} samples at {context.fs:g} Hz "
            f"admits {report['max_level']} levels with '{wavelet}', which cuts below "
            f"{report['cut_hz']:.2f} Hz and would remove the P and T waves along with the "
            f"baseline; use a longer window or raise max_cut_hz deliberately")
    return wavelet_baseline_removal(signal, wavelet=wavelet,
                                    level=report['effective_level'])


def _emd_denoising(signal, context, max_imf=6, noise_components=3, **_):
    return emd_ecg_denoising(signal, max_imf=max_imf, noise_components=noise_components)


# --- registration --------------------------------------------------------

def register_static_filters() -> list:
    """Registers the seven static methods, skipping any already present."""
    entries = [
        ('fir_bandpass', _fir_bandpass,
         'FIR pasmowoprzepustowy, faza zerowa'),
        ('iir_bandpass', _iir_bandpass,
         'IIR pasmowoprzepustowy na sekcjach drugiego rzedu'),
        ('moving_average', _moving_average,
         'srednia ruchoma'),
        ('moving_median', _moving_median,
         'mediana ruchoma'),
        ('wavelet_denoising', _wavelet_denoising,
         'progowanie wspolczynnikow falkowych'),
        ('wavelet_baseline', _wavelet_baseline,
         'usuniecie wedrowania linii izoelektrycznej przez odrzucenie aproksymacji'),
        ('emd_denoising', _emd_denoising,
         'empiryczna dekompozycja modalna z odrzuceniem pierwszych skladowych'),
    ]

    registered = []
    for name, fn, description in entries:
        register(name, 'static', fn, description=description)
        registered.append(name)
    return registered
