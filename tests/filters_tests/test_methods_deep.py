"""
Deep method tests.

Split in two. The cutting of a window into the pieces an architecture reads, and the
declarations each network makes to the registry, are plain numpy and run anywhere. Loading
a checkpoint and pushing a batch through a network needs PyTorch and a matching runtime,
so those tests skip where neither is installed and run on the machine that will do the
training.

The property that carries the first half is coverage: every sample of the window must be
produced by exactly one piece. Producing it twice and averaging would lower the error of a
network whose mistakes happen to be independent between pieces, which is an improvement
the network did not earn.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.methods_deep import DEEP_METHODS, clear_model_cache, piece_bounds
from filters.registry import (
    FilterContext,
    applicable_filters,
    apply_filter,
    available_filters,
    configured_params,
    get_spec,
    load_config,
    reset_config,
    unregister,
)

FS = 360.0
WIDTH = 4096
CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'filters.yaml'


@pytest.fixture(autouse=True)
def registry():
    from filters.methods_deep import register_deep_filters

    for name in list(available_filters()):
        unregister(name)
    reset_config()
    clear_model_cache()
    names = register_deep_filters()
    load_config(CONFIG)
    yield names
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    clear_model_cache()


def ecg(n=WIDTH, fs=FS, bpm=75, seed=0):
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    peaks = np.arange(int(fs), n - int(fs), int(60.0 / bpm * fs))
    for peak in peaks:
        signal[peak - 5:peak + 5] += np.hanning(10) * 1.2
        signal[peak + 20:peak + 60] += np.hanning(40) * 0.25
    return signal + 0.15 * rng.standard_normal(n), peaks


# --- cutting a window into pieces ----------------------------------------

@pytest.mark.parametrize('piece', [1024, 2048, 992, 512])
def test_every_sample_is_produced_exactly_once(piece):
    """
    Coverage without overlap is the point.

    Averaging two estimates of the same sample lowers the error of a network whose
    mistakes are independent between pieces, and the network did nothing to earn it.
    """
    bounds = piece_bounds(WIDTH, piece)
    written = np.zeros(WIDTH, dtype=int)
    for _, stop, write_from in bounds:
        written[write_from:stop] += 1
    assert np.all(written == 1)


@pytest.mark.parametrize('piece', [1024, 2048, 992, 512])
def test_every_piece_has_the_length_the_network_reads(piece):
    for start, stop, _ in piece_bounds(WIDTH, piece):
        assert stop - start == piece


def test_pieces_stay_inside_the_window():
    for start, stop, write_from in piece_bounds(WIDTH, 992):
        assert 0 <= start < stop <= WIDTH
        assert start <= write_from < stop


def test_a_length_that_divides_evenly_gives_no_overlap():
    bounds = piece_bounds(4096, 1024)
    assert len(bounds) == 4
    assert all(start == write_from for start, _, write_from in bounds)


def test_a_length_that_does_not_divide_aligns_the_last_piece_to_the_end():
    """992 fits four times into 4096 with 128 samples left over."""
    bounds = piece_bounds(4096, 992)
    assert len(bounds) == 5
    assert bounds[-1][1] == 4096
    assert bounds[-1][0] < bounds[-2][1]
    assert bounds[-1][2] == bounds[-2][1]


def test_a_window_of_exactly_one_piece_is_not_cut():
    assert piece_bounds(1024, 1024) == [(0, 1024, 0)]


def test_a_window_shorter_than_one_piece_is_refused():
    with pytest.raises(ValueError, match='shorter than'):
        piece_bounds(512, 1024)


def test_a_nonpositive_piece_is_refused():
    with pytest.raises(ValueError, match='must be positive'):
        piece_bounds(WIDTH, 0)


# --- what the networks declare -------------------------------------------

def test_exactly_five_networks_are_registered(registry):
    assert len(registry) == 5
    assert available_filters('deep') == sorted(name for name, _ in DEEP_METHODS)


def test_every_network_needs_a_checkpoint(registry):
    for name in registry:
        assert get_spec(name).requires_checkpoint, name


def test_none_of_them_needs_the_accelerometer(registry):
    """
    Which is why the synthetic environment can compare them with the static filters.

    MIT-BIH carries no accelerometer, and a network declaring a need for one would
    disappear from that comparison.
    """
    for name in registry:
        assert not get_spec(name).requires_reference, name


def test_the_beat_based_network_declares_that_it_needs_beats(registry):
    """
    Read from the representation, not written out by hand.

    A change of representation cannot then leave the declaration behind.
    """
    assert get_spec('sced_net').requires_r_peaks
    assert not get_spec('wavelet_cnn').requires_r_peaks


def test_a_network_is_refused_without_its_weights(registry):
    noisy, _ = ecg()
    with pytest.raises(ValueError, match='needs checkpoint'):
        apply_filter('wavelet_cnn', noisy, FilterContext(fs=FS))


def test_the_beat_based_network_is_refused_without_beats(registry, tmp_path):
    noisy, _ = ecg()
    context = FilterContext(fs=FS, checkpoint=tmp_path / 'absent.pt')
    with pytest.raises(ValueError, match='needs .*r_peaks'):
        apply_filter('sced_net', noisy, context)


def test_a_context_with_weights_makes_them_available(registry, tmp_path):
    context = FilterContext(fs=FS, checkpoint=tmp_path / 'weights.pt',
                            r_peaks=np.arange(360, WIDTH - 360, 288))
    report = applicable_filters(context, family='deep')
    assert len(report['runnable']) == 5
    assert report['blocked'] == {}


def test_the_shipped_configuration_covers_every_network(registry):
    import yaml

    with CONFIG.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    assert set(loaded['deep']) == set(registry)


def test_the_checkpoint_path_is_not_configured_in_the_file(registry):
    """It depends on which training run produced the weights, not on the method."""
    for name in registry:
        assert 'checkpoint' not in configured_params(name), name


# --- running the networks ------------------------------------------------

@pytest.fixture
def torch_available():
    """
    Skips where PyTorch cannot actually be used.

    `pytest.importorskip` is not enough: the package can be installed and still fail to
    import when the CUDA runtime it was built against is absent, and that raises something
    other than an import error.
    """
    try:
        import torch
    except Exception as error:                                  # noqa: BLE001
        pytest.skip(f'PyTorch unusable here: {type(error).__name__}')
    return torch


@pytest.fixture
def untrained_checkpoint(torch_available, tmp_path):
    """
    Weights straight from initialisation, saved so the loading path can be exercised.

    An untrained network denoises nothing, which is fine: what is under test here is that
    a window goes in, the pieces are cut, batched and decoded, and a waveform of the same
    length comes back.
    """
    import torch

    from train.signal_selection import build_model

    def save(model_name):
        path = tmp_path / f'{model_name}.pt'
        torch.save(build_model(model_name).state_dict(), path)
        return path

    return save


@pytest.mark.parametrize('name', ['wavelet_cnn', 'cstrans', 'ecgd_net', 'deepcednet'])
def test_a_window_goes_through_and_comes_back_whole(registry, untrained_checkpoint, name):
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint(name), r_peaks=peaks)
    result = apply_filter(name, noisy, context)
    assert result.signal.shape == noisy.shape
    assert np.all(np.isfinite(result.signal))
    assert result.family == 'deep'


def test_the_beat_based_network_goes_through(registry, untrained_checkpoint):
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('sced_net'),
                            r_peaks=peaks)
    result = apply_filter('sced_net', noisy, context)
    assert result.signal.shape == noisy.shape


def test_the_beat_based_network_reports_what_it_reconstructed(registry, untrained_checkpoint):
    """
    Its cycles reach from before the first beat to after the last, not to the window edges.

    Returning a shorter waveform would break the contract every other method keeps, so the
    input is copied through outside the span and the span is reported instead.
    """
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('sced_net'),
                            r_peaks=peaks)
    result = apply_filter('sced_net', noisy, context)

    start, stop = result.covered_span
    assert 0 < start < stop < WIDTH
    assert (stop - start) > 0.7 * WIDTH


def test_outside_the_span_the_beat_based_network_returns_its_input(registry,
                                                                   untrained_checkpoint):
    """
    The uncovered edges keep their noise, which can only count against this architecture.

    Filling them with zeros would look like a very clean stretch to every metric.
    """
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('sced_net'),
                            r_peaks=peaks)
    result = apply_filter('sced_net', noisy, context)

    start, stop = result.covered_span
    assert np.array_equal(result.signal[:start], noisy[:start])
    assert np.array_equal(result.signal[stop:], noisy[stop:])


def test_the_window_based_networks_cover_everything(registry, untrained_checkpoint):
    noisy, peaks = ecg()
    for name in ('wavelet_cnn', 'cstrans', 'deepcednet'):
        context = FilterContext(fs=FS, checkpoint=untrained_checkpoint(name), r_peaks=peaks)
        assert apply_filter(name, noisy, context).covered_span is None, name


def test_the_checkpoint_is_loaded_once_and_reused(registry, untrained_checkpoint):
    """
    Loading takes longer than filtering a window, and an evaluation runs thousands.

    Reloading each time would dominate the measured cost and make the timing column of
    the results table describe the disk rather than the method.
    """
    from filters.methods_deep import _MODEL_CACHE

    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('wavelet_cnn'),
                            r_peaks=peaks)
    apply_filter('wavelet_cnn', noisy, context)
    assert len(_MODEL_CACHE) == 1
    apply_filter('wavelet_cnn', noisy, context)
    assert len(_MODEL_CACHE) == 1


def test_a_missing_checkpoint_file_is_reported_clearly(registry, torch_available, tmp_path):
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=tmp_path / 'nothing.pt', r_peaks=peaks)
    with pytest.raises(FileNotFoundError, match='no checkpoint'):
        apply_filter('wavelet_cnn', noisy, context)


def test_the_batch_size_does_not_change_the_result(registry, untrained_checkpoint):
    """It is a memory setting, not a parameter of the method."""
    noisy, peaks = ecg()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('wavelet_cnn'),
                            r_peaks=peaks)
    small = apply_filter('wavelet_cnn', noisy, context, batch_size=1).signal
    large = apply_filter('wavelet_cnn', noisy, context, batch_size=32).signal
    assert np.allclose(small, large, atol=1e-5)


def test_the_input_is_left_untouched(registry, untrained_checkpoint):
    noisy, peaks = ecg()
    original = noisy.copy()
    context = FilterContext(fs=FS, checkpoint=untrained_checkpoint('wavelet_cnn'),
                            r_peaks=peaks)
    apply_filter('wavelet_cnn', noisy, context)
    assert np.array_equal(noisy, original)


# --- all seventeen together ----------------------------------------------

def test_the_wearable_context_runs_every_method(registry, tmp_path):
    from filters.methods_adaptive import register_adaptive_filters
    from filters.methods_static import register_static_filters

    register_static_filters()
    register_adaptive_filters()

    context = FilterContext(fs=FS, reference=np.zeros((6, WIDTH)),
                            r_peaks=np.arange(360, WIDTH - 360, 288),
                            checkpoint=tmp_path / 'weights.pt')
    report = applicable_filters(context)
    assert len(report['runnable']) == 17
    assert report['blocked'] == {}


def test_the_synthetic_context_runs_twelve_of_them(registry, tmp_path):
    """Seven static and five networks; the adaptive family has no reference there."""
    from filters.methods_adaptive import register_adaptive_filters
    from filters.methods_static import register_static_filters

    register_static_filters()
    register_adaptive_filters()

    context = FilterContext(fs=FS, r_peaks=np.arange(360, WIDTH - 360, 288),
                            checkpoint=tmp_path / 'weights.pt')
    report = applicable_filters(context)
    assert len(report['runnable']) == 12
    assert set(report['blocked']) == set(available_filters('adaptive'))
