"""
Result format tests, and the acceptance test for the whole registry.

The first half covers the record of a run: that rows are on disk before the run ends,
that a method which failed leaves a row saying so, and that what was written can be read
back unchanged. A missing row would make a method that diverged indistinguishable from
one that was never run.

The second half is the acceptance test the pipeline chapter asks for: every registered
method processes one test window without error. Twelve of them run anywhere; the five
networks need PyTorch and a matching runtime, so those skip where neither is present.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.methods_adaptive import register_adaptive_filters
from filters.methods_deep import clear_model_cache, register_deep_filters
from filters.methods_static import register_static_filters
from filters.registry import (
    FilterContext,
    apply_filter,
    available_filters,
    load_config,
    reset_config,
    unregister,
)
from filters.results import (
    RESULT_FIELDS,
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_OK,
    ResultWriter,
    format_summary,
    read_results,
    run_window,
    summarise,
)

FS = 360.0
WIDTH = 4096
CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'filters.yaml'


@pytest.fixture(autouse=True)
def registry():
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    clear_model_cache()
    names = (register_static_filters() + register_adaptive_filters()
             + register_deep_filters())
    load_config(CONFIG)
    yield names
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    clear_model_cache()


def accelerometer(base, seed, n=WIDTH):
    rng = np.random.default_rng(seed)
    return np.stack([base + 0.4 * rng.standard_normal(n),
                     0.5 * base + 0.8 * rng.standard_normal(n),
                     0.2 * base + 1.0 * rng.standard_normal(n)])


@pytest.fixture
def scene():
    """A beat train, two accelerometers and the artefact they explain."""
    rng = np.random.default_rng(0)
    t = np.arange(WIDTH) / FS
    clean = np.zeros(WIDTH)
    peaks = np.arange(int(FS), WIDTH - int(FS), 288)
    for peak in peaks:
        clean[peak - 5:peak + 5] += np.hanning(10) * 1.2
        clean[peak + 20:peak + 60] += np.hanning(40) * 0.25

    first = np.sin(2 * np.pi * 1.7 * t) + 0.3 * rng.standard_normal(WIDTH)
    second = 0.7 * np.roll(first, 5) + 0.6 * np.sin(2 * np.pi * 0.5 * t)
    reference = np.vstack([accelerometer(first, 1), accelerometer(second, 2)])
    noisy = clean + 0.30 * reference[0] + 0.20 * reference[3]
    return clean, noisy, reference, peaks


# --- the record of a run -------------------------------------------------

def test_a_row_reaches_disk_before_the_run_ends(tmp_path, scene):
    """
    A full run takes hours, and a crash in the last one should not cost the first.

    The file is opened at the start and each row appended, so an interrupted run keeps
    everything it produced.
    """
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    context = FilterContext(fs=FS)

    with ResultWriter(path) as writer:
        run_window('iir_bandpass', noisy, context, '100', '100', 3.0, (0, WIDTH), writer)
        assert path.exists()
        assert 'iir_bandpass' in path.read_text(encoding='utf-8')


def test_every_declared_field_is_written(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    header = path.read_text(encoding='utf-8').splitlines()[0]
    assert header.split(',') == list(RESULT_FIELDS)


def test_what_was_written_reads_back(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', -9.0,
                   (512, 512 + WIDTH), writer)

    row, = read_results(path)
    assert row['method'] == 'iir_bandpass'
    assert row['family'] == 'static'
    assert row['status'] == STATUS_OK
    assert row['record'] == '100'
    assert row['snr_db'] == -9.0
    assert row['window_start'] == 512
    assert row['elapsed_s'] > 0.0
    assert row['params']['low_hz'] == 0.5
    assert row['code_version']


def test_a_blocked_method_leaves_a_row(tmp_path, scene):
    """
    Adaptive methods on a database with no accelerometer are the ordinary case.

    The table has to say so rather than leave the reader to notice a gap.
    """
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        outcome = run_window('rls_anc', noisy, FilterContext(fs=FS), '100', '100',
                             3.0, (0, WIDTH), writer)

    assert outcome is None
    row, = read_results(path)
    assert row['status'] == STATUS_BLOCKED
    assert 'reference' in row['message']


def test_a_method_that_raises_leaves_a_row(tmp_path, scene):
    """
    A divergence at one ratio costs that row, not the remaining hours of the run.

    Without this the failure would stop everything, and the partial results would say
    nothing about which method had the problem.
    """
    from filters.registry import register

    def diverging(signal, context, **_):
        raise RuntimeError('krok poza granica stabilnosci')

    register('diverging', 'adaptive', diverging)
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        outcome = run_window('diverging', noisy, FilterContext(fs=FS), '100', '100',
                             3.0, (0, WIDTH), writer)

    assert outcome is None
    row, = read_results(path)
    assert row['status'] == STATUS_ERROR
    assert 'RuntimeError' in row['message']


def test_the_run_continues_after_a_failure(tmp_path, scene):
    from filters.registry import register

    register('diverging', 'static', lambda s, c, **_: 1 / 0)
    _, noisy, _, _ = scene
    context = FilterContext(fs=FS)
    path = tmp_path / 'results.csv'

    with ResultWriter(path) as writer:
        for name in ('iir_bandpass', 'diverging', 'moving_median'):
            run_window(name, noisy, context, '100', '100', 3.0, (0, WIDTH), writer)

    statuses = {row['method']: row['status'] for row in read_results(path)}
    assert statuses == {'iir_bandpass': STATUS_OK, 'diverging': STATUS_ERROR,
                        'moving_median': STATUS_OK}


def test_the_covered_span_is_recorded_when_there_is_one(tmp_path, scene):
    from filters.registry import FilterResult, register

    def partial(signal, context, **_):
        return signal

    spec = register('partial', 'static', partial)
    spec.fn.last_covered_span = (100, 900)

    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('partial', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    row, = read_results(path)
    assert row['covered_start'] == 100 and row['covered_stop'] == 900


def test_a_full_span_leaves_the_columns_empty(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    row, = read_results(path)
    assert row['covered_start'] is None and row['covered_stop'] is None


# --- the sidecar ---------------------------------------------------------

def test_a_sidecar_says_what_the_file_is(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path, config={'purpose': 'filter', 'width': WIDTH}) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    sidecar = json.loads((tmp_path / 'results.json').read_text(encoding='utf-8'))
    assert sidecar['n_rows'] == 1
    assert sidecar['counts'][STATUS_OK] == 1
    assert sidecar['config']['width'] == WIDTH
    assert sidecar['fields'] == list(RESULT_FIELDS)
    assert sidecar['written_at']


def test_the_sidecar_is_written_even_when_the_run_is_interrupted(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    try:
        with ResultWriter(path) as writer:
            run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                       (0, WIDTH), writer)
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

    assert (tmp_path / 'results.json').exists()
    assert len(read_results(path)) == 1


# --- waveforms -----------------------------------------------------------

def test_no_waveform_is_kept_by_default(tmp_path, scene):
    """
    Seventeen methods over the held out pool come to gigabytes almost none of which is
    ever looked at.
    """
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    assert list(tmp_path.glob('*.npz')) == []


def test_every_nth_waveform_is_kept_when_asked(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    waves = tmp_path / 'waveforms.npz'

    with ResultWriter(path, waveform_path=waves, waveform_every=3) as writer:
        for index in range(9):
            run_window('iir_bandpass', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                       (index * 100, index * 100 + WIDTH), writer)

    with np.load(waves) as stored:
        assert len(stored.files) == 3
        assert all(stored[key].shape == (WIDTH,) for key in stored.files)
        assert all(stored[key].dtype == np.float32 for key in stored.files)


def test_a_stored_waveform_can_be_traced_back_to_its_row(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    waves = tmp_path / 'waveforms.npz'

    with ResultWriter(path, waveform_path=waves, waveform_every=1) as writer:
        run_window('iir_bandpass', noisy, FilterContext(fs=FS), '103', '103', -5.0,
                   (2048, 2048 + WIDTH), writer)

    with np.load(waves) as stored:
        key, = stored.files
    method, record, snr, start = key.split('|')
    row, = read_results(path)
    assert method == row['method'] and record == row['record']
    assert float(snr) == row['snr_db'] and int(start) == row['window_start']


def test_asking_for_waveforms_without_a_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='waveform_every'):
        ResultWriter(tmp_path / 'r.csv', waveform_every=5)


def test_writing_outside_the_context_manager_is_rejected(tmp_path):
    writer = ResultWriter(tmp_path / 'r.csv')
    with pytest.raises(RuntimeError, match='context manager'):
        writer.blocked('x', 'static', '100', '100', 0.0, (0, 10), ['reference'])


# --- summary -------------------------------------------------------------

def test_the_summary_counts_each_outcome(tmp_path, scene):
    from filters.registry import register

    register('diverging', 'static', lambda s, c, **_: 1 / 0)
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'

    with ResultWriter(path) as writer:
        for name in ('iir_bandpass', 'iir_bandpass', 'diverging', 'rls_anc'):
            run_window(name, noisy, FilterContext(fs=FS), '100', '100', 3.0,
                       (0, WIDTH), writer)

    summary = summarise(read_results(path))
    assert summary['iir_bandpass']['ok'] == 2
    assert summary['diverging']['error'] == 1
    assert summary['rls_anc']['blocked'] == 1
    assert summary['iir_bandpass']['elapsed_mean_ms'] > 0.0


def test_a_method_that_never_ran_has_no_mean_time(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        run_window('rls_anc', noisy, FilterContext(fs=FS), '100', '100', 3.0,
                   (0, WIDTH), writer)

    assert summarise(read_results(path))['rls_anc']['elapsed_mean_ms'] is None


def test_the_formatted_summary_lists_every_method(tmp_path, scene):
    _, noisy, _, _ = scene
    path = tmp_path / 'results.csv'
    with ResultWriter(path) as writer:
        for name in ('iir_bandpass', 'moving_median', 'rls_anc'):
            run_window(name, noisy, FilterContext(fs=FS), '100', '100', 3.0,
                       (0, WIDTH), writer)

    text = format_summary(read_results(path))
    for name in ('iir_bandpass', 'moving_median', 'rls_anc'):
        assert name in text


# --- B.5: every registered method processes a test window ----------------

def test_the_registry_holds_seventeen_methods(registry):
    assert len(registry) == 17
    assert len(available_filters('static')) == 7
    assert len(available_filters('adaptive')) == 5
    assert len(available_filters('deep')) == 5


@pytest.mark.parametrize('name', ['fir_bandpass', 'iir_bandpass', 'moving_average',
                                  'moving_median', 'wavelet_denoising',
                                  'wavelet_baseline', 'emd_denoising',
                                  'lms_anc', 'rls_anc', 'blms_anc',
                                  'gall_anc', 'gall_kalman'])
def test_each_method_without_weights_processes_a_window(registry, scene, name):
    """The twelve that need no checkpoint; these run wherever numpy runs."""
    _, noisy, reference, peaks = scene
    context = FilterContext(fs=FS, reference=reference, r_peaks=peaks)
    result = apply_filter(name, noisy, context)

    assert result.signal.shape == noisy.shape
    assert np.all(np.isfinite(result.signal))
    assert result.elapsed_s > 0.0
    assert result.code_version


@pytest.fixture
def untrained_checkpoint(tmp_path):
    try:
        import torch
    except Exception as error:                                      # noqa: BLE001
        pytest.skip(f'PyTorch unusable here: {type(error).__name__}')

    from train.signal_selection import build_model

    def save(model_name):
        path = tmp_path / f'{model_name}.pt'
        torch.save(build_model(model_name).state_dict(), path)
        return path

    return save


@pytest.mark.parametrize('name', ['wavelet_cnn', 'sced_net', 'cstrans',
                                  'ecgd_net', 'deepcednet'])
def test_each_network_processes_a_window(registry, scene, untrained_checkpoint, name):
    """The five that need weights; untrained ones exercise the whole path."""
    _, noisy, reference, peaks = scene
    context = FilterContext(fs=FS, reference=reference, r_peaks=peaks,
                            checkpoint=untrained_checkpoint(name))
    result = apply_filter(name, noisy, context)

    assert result.signal.shape == noisy.shape
    assert np.all(np.isfinite(result.signal))


def test_a_whole_run_over_every_method_is_recorded(registry, scene, tmp_path):
    """
    The acceptance test of the pipeline: one window, every method, one file.

    Networks appear as blocked rather than absent, because this run supplies no weights,
    and that is exactly the distinction the format exists to keep.
    """
    _, noisy, reference, peaks = scene
    context = FilterContext(fs=FS, reference=reference, r_peaks=peaks)
    path = tmp_path / 'run.csv'

    with ResultWriter(path, config={'note': 'B.5'}) as writer:
        for name in available_filters():
            run_window(name, noisy, context, '100', '100', 3.0, (0, WIDTH), writer)

    rows = read_results(path)
    assert len(rows) == 17

    summary = summarise(rows)
    assert sum(entry['ok'] for entry in summary.values()) == 12
    assert sum(entry['blocked'] for entry in summary.values()) == 5
    assert sum(entry['error'] for entry in summary.values()) == 0
