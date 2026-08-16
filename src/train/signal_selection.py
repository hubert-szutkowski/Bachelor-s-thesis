'''
Signal representation selection layer.

Every denoising architecture in this project consumes a different representation of the
same ECG recording:

    WaveletCNNAutoencoder   (1, 1024)        time domain
    CSTRANS_Denoising       (1, 2048)        time domain
    ECGD_Net_CNN            (1, 1024)        time domain, MIEMD pre-filtered input
    SCED_Net                (1, 32, 384)     stacked cardiac cycle tensor
    DeepCEDNet              (2, 129, 32)     time-frequency map (real / imaginary)

A single training loop can therefore only be written once the representation is abstracted
away. `select_signal(model_name)` returns a `SignalSpec` that carries the tensor layout and
a matched `encode` / `decode` pair.

`encode` runs offline while the cache is built, `decode` returns to the time domain and is
required because SNR, MSE, RMSE and PRD are only comparable across architectures when they
are evaluated on the time-domain waveform. The training loss itself is computed in the
native domain of the model, so `decode` never has to be differentiable.
'''

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from filters.signal_transforms import ECGSignalProcessor, MIEMD_Filter, required_segment_length
from filters.signal_transforms import (
    DEFAULT_FS,
    DEFAULT_NPERSEG,
    DEFAULT_NOVERLAP,
    DEFAULT_N_FREQ,
    DEFAULT_N_FRAMES,
    stft_proxy_transform,
    stft_proxy_inverse,
)

DOMAIN_TIME = 'time'
DOMAIN_CARDIAC_CYCLE = 'cardiac_cycle'
DOMAIN_TIME_FREQUENCY = 'time_frequency'


class SignalRepresentation:
    '''
    Base class of the encode / decode pair.

    Contract enforced by tests/test_signal_selection.py:
        1. encode returns a float32 array whose shape equals `tensor_shape`
        2. decode(encode(x)[0], meta) reproduces x up to the resampling error
        3. encode is deterministic, so a cache built once stays reproducible
        4. passing `stats` reuses the normalization of another signal, which is mandatory
           when encoding a clean target that belongs to an already encoded noisy input
    '''

    domain: str = DOMAIN_TIME
    tensor_shape: tuple = ()
    requires_r_peaks: bool = False

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        raise NotImplementedError

    def decode(self, y: np.ndarray, meta: dict) -> np.ndarray:
        raise NotImplementedError

    def reference_waveform(self, x: np.ndarray, r_peaks: np.ndarray = None) -> np.ndarray:
        '''
        Time-domain signal that `decode(encode(x))` is expected to reproduce.

        Identity for every representation that only changes the basis. Representations that
        also alter the signal content, such as the MIEMD pre-filter, override this so that
        the round-trip contract stays testable without pretending the filter is invertible.
        '''
        return np.asarray(x, dtype=np.float64).ravel()

    @property
    def bytes_per_sample(self) -> int:
        '''Float32 footprint of one encoded sample, used to size the cache.'''
        return int(np.prod(self.tensor_shape)) * 4


class TimeSeriesSignal(SignalRepresentation):
    '''
    Plain 1-D window with per-segment normalization.

    Parameters
    ----------
    segment_length : int
        Number of samples per training example.
    normalization : str
        'zscore', 'minmax' or 'none'.
    '''

    domain = DOMAIN_TIME
    requires_r_peaks = False

    def __init__(self, segment_length: int = 1024, normalization: str = 'zscore'):
        if normalization not in ('zscore', 'minmax', 'none'):
            raise ValueError(f"unknown normalization '{normalization}'")
        self.segment_length = segment_length
        self.normalization = normalization
        self.tensor_shape = (1, segment_length)

    def compute_stats(self, x: np.ndarray) -> dict:
        if self.normalization == 'zscore':
            return {'loc': float(np.mean(x)), 'scale': float(np.std(x)) + 1e-8}
        if self.normalization == 'minmax':
            lo, hi = float(np.min(x)), float(np.max(x))
            return {'loc': lo, 'scale': max(hi - lo, 1e-8)}
        return {'loc': 0.0, 'scale': 1.0}

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.segment_length:
            raise ValueError(f'expected {self.segment_length} samples, got {x.size}')

        if stats is None:
            stats = self.compute_stats(x)
        encoded = (x - stats['loc']) / stats['scale']
        return encoded.astype(np.float32).reshape(1, -1), {'stats': stats, 'signal_length': x.size}

    def decode(self, y: np.ndarray, meta: dict) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        stats = meta['stats']
        return y * stats['scale'] + stats['loc']


class MIEMDTimeSeriesSignal(TimeSeriesSignal):
    '''
    Stage-I MIEMD decomposition followed by the same normalization as `TimeSeriesSignal`.

    The decomposition is applied to the network input only; the clean target keeps its raw
    waveform, otherwise the network would be trained to reproduce the artefacts that MIEMD
    already removed.

    Empirical Mode Decomposition costs on the order of a second per segment, which rules out
    running it inside a DataLoader worker. `ECGDenoisingDataset` therefore pre-computes and
    caches the whole encoded set before the first epoch.
    '''

    def __init__(self, segment_length: int = 1024, normalization: str = 'zscore',
                 threshold_multiplier: float = 1.0):
        super().__init__(segment_length=segment_length, normalization=normalization)
        self.threshold_multiplier = threshold_multiplier
        self._filter = None

    def _get_filter(self):
        if self._filter is None:
            self._filter = MIEMD_Filter(threshold_multiplier=self.threshold_multiplier)
        return self._filter

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.segment_length:
            raise ValueError(f'expected {self.segment_length} samples, got {x.size}')

        if stats is None:
            filtered = self._get_filter().process(x)
            if filtered.size != x.size:
                filtered = np.resize(filtered, x.size)
            return super().encode(filtered, stats=None)
        return super().encode(x, stats=stats)

    def reference_waveform(self, x: np.ndarray, r_peaks: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        filtered = self._get_filter().process(x)
        if filtered.size != x.size:
            filtered = np.resize(filtered, x.size)
        return filtered


class CardiacCycleSignal(SignalRepresentation):
    '''
    Stacked Cardiac Cycle tensor built by `filters.signal_transforms.ECGSignalProcessor`.

    The segment length is not fixed: one example spans `num_cycles` heartbeats, which at
    360 Hz and 75 bpm is roughly 25 s of recording.
    '''

    domain = DOMAIN_CARDIAC_CYCLE
    requires_r_peaks = True

    def __init__(self, num_cycles: int = 32, target_length: int = 384, min_cycle_length: int = 8):
        self.processor = ECGSignalProcessor(target_length=target_length,
                                            num_cycles=num_cycles,
                                            min_cycle_length=min_cycle_length)
        self.num_cycles = num_cycles
        self.target_length = target_length
        self.tensor_shape = (1, num_cycles, target_length)

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        if r_peaks is None:
            raise ValueError('CardiacCycleSignal requires r_peaks')

        x = np.asarray(x, dtype=np.float64).ravel()
        scc, lengths, g_min, g_max = self.processor.preprocess(x, r_peaks, stats=stats)
        meta = {'stats': (g_min, g_max), 'lengths': lengths, 'signal_length': x.size}
        return scc.astype(np.float32)[None, ...], meta

    def decode(self, y: np.ndarray, meta: dict) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(self.num_cycles, self.target_length)
        g_min, g_max = meta['stats']
        return self.processor.postprocess(y, meta['lengths'], g_min, g_max)


class TimeFrequencySignal(SignalRepresentation):
    '''
    Real / imaginary time-frequency map produced by `filters.signal_transforms.stft_proxy_transform`.

    The transform is a standard STFT standing in for the FSSTH of [10]. Swap it for a true
    synchrosqueezing implementation before describing the results as FSSTH-based.
    '''

    domain = DOMAIN_TIME_FREQUENCY
    requires_r_peaks = False

    def __init__(self, segment_length: int = None, fs: int = DEFAULT_FS,
                 nperseg: int = DEFAULT_NPERSEG, noverlap: int = DEFAULT_NOVERLAP,
                 n_freq: int = DEFAULT_N_FREQ, n_frames: int = DEFAULT_N_FRAMES):
        exact = required_segment_length(nperseg, noverlap, n_frames)
        if segment_length is None:
            segment_length = exact
        elif segment_length != exact:
            raise ValueError(
                f'segment_length must be {exact} for nperseg={nperseg}, noverlap={noverlap} '
                f'and n_frames={n_frames}; {segment_length} would leave '
                f'{segment_length // (nperseg - noverlap) + 1 - n_frames} frame(s) outside the '
                f'retained patch and make the inverse transform lose the tail of the waveform')
        self.segment_length = segment_length
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.n_freq = n_freq
        self.n_frames = n_frames
        self.tensor_shape = (2, n_freq, n_frames)

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.segment_length:
            raise ValueError(f'expected {self.segment_length} samples, got {x.size}')

        tensor, meta = stft_proxy_transform(x, fs=self.fs, nperseg=self.nperseg,
                                            noverlap=self.noverlap, n_freq=self.n_freq,
                                            n_frames=self.n_frames, stats=stats)
        return tensor, meta

    def decode(self, y: np.ndarray, meta: dict) -> np.ndarray:
        return stft_proxy_inverse(y, meta)


@dataclass(frozen=True)
class SignalSpec:
    '''
    Everything the training loop needs to know about a model without importing it.

    Attributes
    ----------
    model_name : str
        Registry key.
    model_class : str
        Class name inside `module`, resolved lazily by `build_model` so that this module
        stays importable without PyTorch.
    module : str
        Dotted path of the module holding the architecture.
    representation : SignalRepresentation
        Encode / decode pair.
    segment_length : int or None
        Samples per example, or None when the example spans a whole number of heartbeats.
    default_loss : str
        Key understood by `training.build_criterion`.
    '''

    model_name: str
    model_class: str
    module: str
    representation: SignalRepresentation
    segment_length: Optional[int]
    default_loss: str = 'improved_mse'
    model_kwargs: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return self.representation.domain

    @property
    def tensor_shape(self) -> tuple:
        return self.representation.tensor_shape

    @property
    def requires_r_peaks(self) -> bool:
        return self.representation.requires_r_peaks

    @property
    def is_spatial(self) -> bool:
        '''True for 4-D batches, which are the only ones that benefit from channels_last.'''
        return len(self.tensor_shape) == 3

    @property
    def bytes_per_sample(self) -> int:
        return self.representation.bytes_per_sample

    def encode(self, x: np.ndarray, r_peaks: np.ndarray = None, stats: Any = None) -> tuple:
        return self.representation.encode(x, r_peaks=r_peaks, stats=stats)

    def decode(self, y: np.ndarray, meta: dict) -> np.ndarray:
        return self.representation.decode(y, meta)

    def reference_waveform(self, x: np.ndarray, r_peaks: np.ndarray = None) -> np.ndarray:
        return self.representation.reference_waveform(x, r_peaks=r_peaks)

    @property
    def is_lossy(self) -> bool:
        '''True when encode alters the signal content, not only its basis.'''
        return type(self.representation).reference_waveform is not SignalRepresentation.reference_waveform


def _registry(fs: int = DEFAULT_FS) -> dict:
    return {
        'wavelet_cnn': SignalSpec(
            model_name='wavelet_cnn',
            model_class='WaveletCNNAutoencoder',
            module='filters.wavelet_model',
            representation=TimeSeriesSignal(segment_length=1024),
            segment_length=1024,
        ),
        'cstrans': SignalSpec(
            model_name='cstrans',
            model_class='CSTRANS_Denoising',
            module='filters.wavelet_transformers',
            representation=TimeSeriesSignal(segment_length=2048),
            segment_length=2048,
        ),
        'ecgd_net': SignalSpec(
            model_name='ecgd_net',
            model_class='ECGD_Net_CNN',
            module='filters.miemd_cnn',
            representation=MIEMDTimeSeriesSignal(segment_length=1024),
            segment_length=1024,
            model_kwargs={'seq_length': 1024},
        ),
        'sced_net': SignalSpec(
            model_name='sced_net',
            model_class='SCED_Net',
            module='filters.sced_net_model',
            representation=CardiacCycleSignal(num_cycles=32, target_length=384),
            segment_length=None,
        ),
        'deepcednet': SignalSpec(
            model_name='deepcednet',
            model_class='DeepCEDNet',
            module='filters.FSSTH_model',
            representation=TimeFrequencySignal(fs=fs),
            segment_length=required_segment_length(),
        ),
    }


def available_models() -> list:
    '''Registry keys accepted by `select_signal` and `build_model`.'''
    return sorted(_registry().keys())


def select_signal(model_name: str, fs: int = DEFAULT_FS, **representation_kwargs) -> SignalSpec:
    '''
    Returns the `SignalSpec` bound to a model.

    Parameters
    ----------
    model_name : str
        One of `available_models()`.
    fs : int, optional
        Sampling frequency in Hz, forwarded to the time-frequency representation.
    **representation_kwargs
        Overrides applied to the constructor of the representation, for example
        `segment_length=2048` or `num_cycles=16`.

    Returns
    -------
    spec : SignalSpec

    Examples
    --------
    >>> spec = select_signal('sced_net')
    >>> spec.tensor_shape
    (1, 32, 384)
    >>> spec.requires_r_peaks
    True
    '''
    registry = _registry(fs=fs)
    if model_name not in registry:
        raise KeyError(f"unknown model '{model_name}'; available: {sorted(registry)}")

    spec = registry[model_name]
    if not representation_kwargs:
        return spec

    representation = type(spec.representation)(**representation_kwargs)
    segment_length = getattr(representation, 'segment_length', None)
    return SignalSpec(
        model_name=spec.model_name,
        model_class=spec.model_class,
        module=spec.module,
        representation=representation,
        segment_length=segment_length,
        default_loss=spec.default_loss,
        model_kwargs=dict(spec.model_kwargs),
    )


def build_model(model_name: str, **overrides):
    '''
    Instantiates the architecture bound to `model_name`.

    PyTorch is imported lazily so that the representation layer can be exercised in
    environments without a CUDA-capable install.
    '''
    import importlib

    spec = select_signal(model_name)
    module = importlib.import_module(spec.module)
    model_class = getattr(module, spec.model_class)

    kwargs = dict(spec.model_kwargs)
    kwargs.update(overrides)
    return model_class(**kwargs)


def describe(model_name: str) -> str:
    '''One-line summary used by the CLI and the logs.'''
    spec = select_signal(model_name)
    seg = spec.segment_length if spec.segment_length is not None else 'variable'
    return (f'{spec.model_name:12s} {spec.module}.{spec.model_class:22s} '
            f'domain={spec.domain:14s} tensor={spec.tensor_shape} '
            f'segment={seg} r_peaks={spec.requires_r_peaks} '
            f'{spec.bytes_per_sample / 1024:.1f} KiB/sample')
