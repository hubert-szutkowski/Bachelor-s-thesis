"""
The five adaptive methods, adapted to the registry.

All of them cancel noise against a reference derived from the accelerometers, so all of
them declare `requires_reference` and none of them can run on MIT-BIH. That is not a gap
to be filled but the shape of the study: the synthetic environment compares seven static
methods against five networks, and only the recordings made with the wearable platform
compare all seventeen.

Three things happen in these adapters that do not happen in the static ones.

The reference is standardised to unit variance before it reaches any of them. The
stability bound of the least mean squares update is inversely proportional to the power of
the reference, so a step size chosen for an accelerometer reported in units of gravity
diverges for the same sensor reported in raw converter counts. Fixing the power once makes
the step size a property of the algorithm rather than of the acquisition chain, and lets
one configured value carry from one recording to the next. It is applied here rather than
left to the caller because forgetting it produces a filter that either diverges or never
converges, and the second is indistinguishable from a filter that simply does not work.

The modified least mean squares method needs a low pass version of the ECG for its weight
update. Taking it from the context would make the result depend on whatever the caller
happened to supply; the adapter computes it instead, from a cutoff recorded in the
parameters, so the method is reproducible from its own record.

Two of the implementations return more than a waveform. The adapters keep the denoised
signal and discard the rest, since the weight history of a GALL filter over a window of
4096 samples with order 32 is a megabyte per window and is of interest only when a
particular run is being examined by hand.
"""

import numpy as np

try:
    from .adaptive_filters import (
        blms_ecg_filter,
        hybrid_gall_kalman_ecg_filter,
        modified_lms_anc,
        rls_anc,
    )
    from .gall_filter import gall_filter
    from .reference import principal_channel, reference_matrix, standardise
    from .registry import register
    from .static_filters import iir_filter
except ImportError:
    from adaptive_filters import (
        blms_ecg_filter,
        hybrid_gall_kalman_ecg_filter,
        modified_lms_anc,
        rls_anc,
    )
    from gall_filter import gall_filter
    from reference import principal_channel, reference_matrix, standardise
    from registry import register
    from static_filters import iir_filter

DEFAULT_LOWPASS_HZ = 40.0
DEFAULT_LOWPASS_ORDER = 4


def prepare_reference(context, standardise_reference: bool = True,
                      svm: bool = False) -> np.ndarray:
    """
    The reference as the adaptive filters should see it.

    `svm` collapses each accelerometer into the magnitude of its acceleration, which
    discards direction and leaves one channel per sensor. It exists because the lattice
    method takes a single reference by construction, and because a comparison between
    three channels and one is a question worth answering rather than assuming.
    """
    reference = reference_matrix(context.reference, svm=svm)
    return standardise(reference) if standardise_reference else reference


def _lowpass(signal, context, cutoff_hz, order):
    """The low pass ECG the modified least mean squares update adapts on."""
    return iir_filter(signal, order=order, Wn=cutoff_hz, btype='lowpass',
                      fs=context.fs, ftype='butter', zero_phase=True)


# --- adapters ------------------------------------------------------------

def _lms_anc(signal, context, mu=0.001, filter_order=5, lowpass_hz=DEFAULT_LOWPASS_HZ,
             lowpass_order=DEFAULT_LOWPASS_ORDER, standardise_reference=True,
             svm=False, **_):
    reference = prepare_reference(context, standardise_reference, svm)
    guide = _lowpass(signal, context, lowpass_hz, lowpass_order)
    return modified_lms_anc(signal, guide, reference=reference, mu=mu,
                            filter_order=filter_order)


def _rls_anc(signal, context, filter_order=5, lam=0.99, delta=1.0,
             standardise_reference=True, svm=False, **_):
    reference = prepare_reference(context, standardise_reference, svm)
    return rls_anc(signal, reference=reference, filter_order=filter_order,
                   lam=lam, delta=delta)


def _blms_anc(signal, context, L=10, mu=0.01, filter_order=32,
              standardise_reference=True, svm=False, **_):
    reference = prepare_reference(context, standardise_reference, svm)
    return blms_ecg_filter(signal, reference=reference, L=L, mu=mu,
                           filter_order=filter_order)


def _gall_anc(signal, context, M=5, beta=1.0, alpha=0.0, epsi=1e-3,
              standardise_reference=True, reduction='pca', **_):
    """
    The lattice method, which takes one reference channel by construction.

    `reduction` decides how the channels become one. `pca` projects them onto their first
    principal component and is the default; `svm` takes the magnitude of the acceleration
    and is kept only so that the comparison can be shown. Rectifying the channels destroys
    the linear relationship the canceller acts on, and measured on synthetic material the
    magnitude leaves the artefact untouched while the projection removes most of it.
    """
    reference = prepare_reference(context, standardise_reference, svm=(reduction == 'svm'))

    if reduction == 'pca':
        reference = principal_channel(reference)
    elif reduction == 'svm':
        reference = standardise(reference.sum(axis=0, keepdims=True))
    elif reduction != 'first':
        raise ValueError(f"unknown reduction {reduction!r}; available: "
                         f"('pca', 'svm', 'first')")

    error, _, _, _ = gall_filter(x=reference[0], d=signal, M=M, beta=beta,
                                 alpha=alpha, epsi=epsi)
    return error


def _gall_kalman(signal, context, M_gall=5, beta_gall=1.0, alpha_gall=0.0,
                 epsi_gall=1e-3, lambda_K=0.99, delta_K=1.0,
                 standardise_reference=True, svm=False, **_):
    reference = prepare_reference(context, standardise_reference, svm)
    clean, _, _ = hybrid_gall_kalman_ecg_filter(
        signal, reference=reference, M_gall=M_gall, beta_gall=beta_gall,
        alpha_gall=alpha_gall, epsi_gall=epsi_gall, lambda_K=lambda_K, delta_K=delta_K)
    return clean


# --- registration --------------------------------------------------------

def register_adaptive_filters() -> list:
    """Registers the five adaptive methods, every one of them needing a reference."""
    entries = [
        ('lms_anc', _lms_anc,
         'LMS wielokanalowy z adaptacja na sygnale dolnoprzepustowym'),
        ('rls_anc', _rls_anc,
         'rekurencyjna metoda najmniejszych kwadratow'),
        ('blms_anc', _blms_anc,
         'blokowy LMS'),
        ('gall_anc', _gall_anc,
         'gradientowa adaptacyjna krata Laguerrea, jeden kanal odniesienia'),
        ('gall_kalman', _gall_kalman,
         'kraty GALL na kazdym kanale, polaczone filtrem Kalmana'),
    ]

    registered = []
    for name, fn, description in entries:
        register(name, 'adaptive', fn, requires_reference=True, description=description)
        registered.append(name)
    return registered
