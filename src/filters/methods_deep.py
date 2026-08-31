"""
The five networks, adapted to the registry.

A network differs from the other twelve methods in three ways, and the adapter exists to
hide all three from the caller.

It carries weights. Loading a checkpoint takes longer than filtering a window, so the
loaded model is kept in a cache keyed by its path and its device and reused across the
several thousand windows of an evaluation. Nothing else in the registry holds state.

It has its own idea of how long a window is. Three of them read the time domain at 1024,
992 and 2048 samples, one reads a stack of cardiac cycles and one a time-frequency map,
while the evaluation hands every method the same 4096 samples. The adapter cuts the window
into pieces of the size its network expects, runs them as one batch and lays the results
back down. The pieces are laid end to end and the final one is aligned to the end of the
window, with only its tail beyond the previous boundary kept, so every sample is produced
exactly once and no two estimates are averaged. Averaging would quietly improve a network
whose errors happen to be independent between pieces.

One of them reconstructs less than the whole window. The cardiac cycle architecture spans
from half an interbeat distance before the first beat to the midpoint after the last, so
the head and the tail of a window have no cycle to belong to. Those samples are returned
as they came in and the reconstructed span is reported in the result, which leaves the
uncovered edges carrying their noise: that can count against the architecture but never
for it.

It works in a representation of its own. `train.signal_selection` already holds the
encode and decode pair for each architecture, so the adapter goes through that rather than
reimplementing any of it, and the waveform that comes back is in the time domain
regardless of the domain the network worked in.

PyTorch is imported inside the functions rather than at module level. Registration then
costs nothing on a machine without it, and the twelve methods that need no accelerator
stay usable where no matching runtime is installed.
"""

from pathlib import Path

import numpy as np

try:
    from .registry import register
except ImportError:
    from registry import register

DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'auto'

_MODEL_CACHE: dict = {}


def resolve_device(device: str = DEFAULT_DEVICE):
    """The device to run on, preferring an accelerator when one is present."""
    import torch

    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def load_model(model_name: str, checkpoint, device: str = DEFAULT_DEVICE):
    """
    A trained network, loaded once and kept.

    Loading takes longer than filtering a window, and an evaluation runs several thousand
    of them; reloading each time would dominate the measured cost of the method and make
    the timing column of the results table meaningless.
    """
    import torch

    try:
        from train.signal_selection import build_model
    except ImportError:
        from ..train.signal_selection import build_model

    checkpoint = Path(checkpoint)
    key = (model_name, str(checkpoint.resolve()), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not checkpoint.exists():
        raise FileNotFoundError(f'{model_name}: no checkpoint at {checkpoint}')

    target = resolve_device(device)
    model = build_model(model_name)
    payload = torch.load(checkpoint, map_location=target, weights_only=False)
    state = payload.get('model_state_dict', payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.to(target).eval()

    _MODEL_CACHE[key] = (model, target)
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Drops every loaded network. For tests and for freeing device memory."""
    _MODEL_CACHE.clear()


def piece_bounds(n_samples: int, piece: int) -> list:
    """
    Where to cut a window so that every sample is produced exactly once.

    Pieces are laid end to end. When the length does not divide evenly the last piece is
    aligned to the end of the window and overlaps its predecessor; only the part beyond
    the previous boundary is taken from it. Overlap-averaging is avoided on purpose,
    because averaging two estimates whose errors are independent lowers the error without
    the network having done anything.
    """
    if piece <= 0:
        raise ValueError(f'the piece length must be positive, got {piece}')
    if n_samples < piece:
        raise ValueError(f'a window of {n_samples} samples is shorter than the {piece} '
                         f'this architecture reads')

    bounds, start = [], 0
    while start + piece <= n_samples:
        bounds.append((start, start + piece, start))
        start += piece

    if start < n_samples:
        last = n_samples - piece
        bounds.append((last, n_samples, start))
    return bounds


# --- the one adapter, parametrised by architecture -----------------------

def _run_network(signal, context, model_name, batch_size=DEFAULT_BATCH_SIZE,
                 device=DEFAULT_DEVICE, **_):
    import torch

    try:
        from train.signal_selection import select_signal
    except ImportError:
        from ..train.signal_selection import select_signal

    model, target = load_model(model_name, context.checkpoint, device)
    spec = select_signal(model_name, fs=context.fs)

    if spec.requires_r_peaks and context.r_peaks is None:
        raise ValueError(f'{model_name} works on cardiac cycles and needs r_peaks')

    if spec.segment_length is None:
        return _run_beatwise(signal, context, model, target, spec)

    bounds = piece_bounds(signal.size, spec.segment_length)
    encoded, metas = [], []
    for start, stop, _ in bounds:
        peaks = None
        if spec.requires_r_peaks:
            inside = context.r_peaks[(context.r_peaks >= start) & (context.r_peaks < stop)]
            peaks = inside - start
        tensor, meta = spec.encode(signal[start:stop], r_peaks=peaks)
        encoded.append(tensor)
        metas.append(meta)

    batch = torch.from_numpy(np.stack(encoded)).to(target)
    outputs = []
    with torch.no_grad():
        for first in range(0, batch.shape[0], int(batch_size)):
            outputs.append(model(batch[first:first + int(batch_size)]).cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)

    filtered = np.empty_like(signal)
    for (start, stop, write_from), piece, meta in zip(bounds, outputs, metas):
        waveform = np.asarray(spec.decode(piece, meta), dtype=np.float64).ravel()
        offset = write_from - start
        filtered[write_from:stop] = waveform[offset:]
    return filtered


def _run_beatwise(signal, context, model, target, spec):
    """
    The cardiac cycle architecture, which spans whole heartbeats rather than samples.

    Its representation reaches from half an interbeat distance before the first annotated
    beat to the midpoint after the last one, so a window of 4096 samples holding twelve
    beats reconstructs 3168 of them and leaves the head and the tail without a cycle to
    belong to. The waveform returned is still of full length, with the input copied through
    outside the reconstructed span, and the span is reported so that the comparison can be
    restricted to it. Copying the input through means the uncovered edges keep their noise,
    which can only count against this architecture, never for it.
    """
    import torch

    processor = spec.representation.processor
    bounds = processor.segment_bounds(signal.size, np.asarray(context.r_peaks))
    if not bounds:
        raise ValueError('no cardiac cycle fits inside this window')
    start, stop = int(bounds[0][0]), int(bounds[-1][1])

    tensor, meta = spec.encode(signal, r_peaks=context.r_peaks)
    with torch.no_grad():
        output = model(torch.from_numpy(tensor[None, ...]).to(target)).cpu().numpy()[0]
    waveform = np.asarray(spec.decode(output, meta), dtype=np.float64).ravel()

    if waveform.size != stop - start:
        raise ValueError(f'the reconstruction holds {waveform.size} samples against the '
                         f'{stop - start} its cycles span')

    filtered = signal.copy()
    filtered[start:stop] = waveform
    return filtered, (start, stop)


# --- registration --------------------------------------------------------

DEEP_METHODS = (
    ('wavelet_cnn', 'autoenkoder splotowy z warstwami falkowymi, okno 1024'),
    ('sced_net', 'siec na stosie cykli serca'),
    ('cstrans', 'transformer z uwaga na dlugim kontekscie, okno 2048'),
    ('ecgd_net', 'siec splotowa na sygnale wstepnie rozlozonym MIEMD, okno 1024'),
    ('deepcednet', 'autoenkoder w dziedzinie czas-czestotliwosc, okno 992'),
)


def _make_adapter(model_name: str):
    def adapter(signal, context, **params):
        output = _run_network(signal, context, model_name, **params)
        if isinstance(output, tuple):
            output, adapter.last_covered_span = output
        else:
            adapter.last_covered_span = None
        return output

    adapter.__name__ = f'_{model_name}'
    adapter.last_covered_span = None
    return adapter


def register_deep_filters() -> list:
    """
    Registers the five networks. Every one of them needs a checkpoint.

    `requires_r_peaks` is read from the representation rather than written out here, so a
    change of representation cannot leave the declaration behind.
    """
    try:
        from train.signal_selection import select_signal
    except ImportError:
        from ..train.signal_selection import select_signal

    registered = []
    for name, description in DEEP_METHODS:
        spec = select_signal(name)
        register(name, 'deep', _make_adapter(name),
                 defaults={'batch_size': DEFAULT_BATCH_SIZE, 'device': DEFAULT_DEVICE},
                 requires_checkpoint=True,
                 requires_r_peaks=spec.requires_r_peaks,
                 description=description)
        registered.append(name)
    return registered
