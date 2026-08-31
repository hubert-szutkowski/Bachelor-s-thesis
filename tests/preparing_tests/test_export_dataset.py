"""
Data layer tests.

The export script is the last step before the material reaches a network, so anything it
gets wrong is discovered as a training result rather than as an error. Three properties
carry the module: every window has the declared width, no patient appears in two splits,
and the noise a patient meets belongs to that patient alone. The last one is the reason
the noise is partitioned at all, and it is invisible in every metric.

Records are generated into a temporary directory and removed afterwards, so nothing
binary enters version control.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import wfdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from preparing.splitting import patient_of

FS = 360
DURATION_S = 30
BPM = 75
NOISE_S = 240

WITH_LEAD = ('100', '101', '103', '105', '200', '201', '202', '203', '205')
WITHOUT_LEAD = '102'
ALL_RECORDS = tuple(sorted(WITH_LEAD + (WITHOUT_LEAD,)))


def synthetic_ecg(seed: int) -> tuple:
    rng = np.random.RandomState(seed)
    n = FS * DURATION_S
    rr = int(60.0 / BPM * FS)
    r_peaks = np.arange(rr, n - rr, rr)
    channel = np.zeros(n)
    for peak in r_peaks:
        channel[peak - 5:peak + 5] += np.hanning(10) * 1.2
        channel[peak + 20:peak + 60] += np.hanning(40) * 0.2
    channel += 0.02 * rng.randn(n)
    return np.stack([channel, 0.6 * channel], axis=1), r_peaks


@pytest.fixture(scope='session')
def source_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('export_source')
    for index, record_id in enumerate(ALL_RECORDS):
        signal, r_peaks = synthetic_ecg(index)
        names = ['V5', 'V1'] if record_id == WITHOUT_LEAD else ['MLII', 'V1']
        wfdb.wrsamp(record_id, fs=FS, units=['mV', 'mV'], sig_name=names,
                    p_signal=signal, fmt=['16', '16'], write_dir=str(root))
        wfdb.wrann(record_id, 'atr', np.asarray(r_peaks),
                   np.array(['N'] * r_peaks.size), write_dir=str(root))

    rng = np.random.RandomState(7)
    noise = (np.cumsum(rng.randn(FS * NOISE_S)) * 0.004 +
             0.25 * rng.randn(FS * NOISE_S))
    wfdb.wrsamp('em', fs=FS, units=['mV'], sig_name=['noise'],
                p_signal=noise[:, None], fmt=['16'], write_dir=str(root))

    yield root
    shutil.rmtree(root, ignore_errors=True)


def run_export(source: Path, out: Path, *extra) -> int:
    from scripts.export_dataset import main
    return main(['--source', str(source), '--out', str(out),
                 '--records', *ALL_RECORDS, '--test-patients', '2',
                 '--folds', '3', *extra])


@pytest.fixture(scope='session', autouse=True)
def _importable():
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='session')
def model_export(source_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp('export_model')
    assert run_export(source_dir, out, '--purpose', 'model',
                      '--width', '1024', '--snr', '3') == 0
    return out


@pytest.fixture(scope='session')
def filter_export(source_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp('export_filter')
    assert run_export(source_dir, out, '--purpose', 'filter',
                      '--width', '4096', '--snr', '6', '0') == 0
    return out


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# --- shape and content ---------------------------------------------------

def test_model_export_writes_one_archive_per_ratio(model_export):
    assert sorted(p.name for p in model_export.glob('*.npz')) == \
           ['model_development_snrp3.npz', 'model_test_snrp3.npz']


def test_filter_export_writes_one_archive_per_ratio(filter_export):
    assert len(list(filter_export.glob('*.npz'))) == 2


def test_every_window_has_the_declared_width(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    assert data['clean'].shape[1] == 1024
    assert data['noisy'].shape == data['clean'].shape


def test_the_filter_purpose_uses_the_wider_window(filter_export):
    data = load(filter_export / 'filter_test_snrp6.npz')
    assert data['clean'].shape[1] == 4096


def test_arrays_agree_in_length(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    n = data['clean'].shape[0]
    for key in ('noisy', 'record', 'patient', 'start', 'gain', 'snr_db_realised'):
        assert data[key].shape[0] == n
    assert data['r_peaks_offset'].shape[0] == n + 1


def test_windows_are_stored_as_single_precision(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    assert data['clean'].dtype == np.float32
    assert data['noisy'].dtype == np.float32


def test_the_requested_ratio_is_realised(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    assert data['snr_db_realised'] == pytest.approx(3.0, abs=1e-3)


def test_the_noisy_window_differs_from_the_clean_one(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    assert not np.allclose(data['clean'], data['noisy'])


def test_beat_positions_fall_inside_their_window(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    offsets = data['r_peaks_offset']
    for index in range(min(50, offsets.size - 1)):
        peaks = data['r_peaks'][offsets[index]:offsets[index + 1]]
        assert np.all((peaks >= 0) & (peaks < data['clean'].shape[1]))


# --- the properties that cannot be observed downstream --------------------

def test_no_patient_appears_in_both_splits(model_export):
    development = load(model_export / 'model_development_snrp3.npz')['patient']
    test = load(model_export / 'model_test_snrp3.npz')['patient']
    assert not set(development.tolist()) & set(test.tolist())


def test_the_held_out_pool_holds_the_requested_number_of_patients(model_export):
    test = load(model_export / 'model_test_snrp3.npz')['patient']
    assert len(set(test.tolist())) == 2


def test_the_noise_of_a_patient_belongs_to_that_patient_alone(source_dir, tmp_path):
    """
    Partitioning the noise is what makes any patient wise split disjoint in noise too.

    Two archives are exported with the same seed. Windows of one patient must carry the
    same injected noise in both, and no two patients may carry the same stretch. Were the
    stretches shared, a network could learn a waveform it meets again in validation.
    """
    from scripts.export_dataset import partition_noise

    out = tmp_path / 'noise'
    assert run_export(source_dir, out, '--purpose', 'model', '--width', '1024',
                      '--snr', '3', '--splits', 'development') == 0
    data = load(out / 'model_development_snrp3.npz')

    noise = np.asarray(wfdb.rdrecord(str(source_dir / 'em')).p_signal)[:, 0]
    parts = partition_noise(noise, [patient_of(record) for record in WITH_LEAD])
    injected = (data['noisy'] - data['clean']).astype(np.float64)

    def best_match(window, stretch):
        """Largest normalised correlation of the window against any slice of a stretch."""
        window = window - window.mean()
        scores = np.correlate(stretch - stretch.mean(), window, mode='valid')
        energy = np.sqrt(np.sum(window ** 2) * np.sum((stretch - stretch.mean()) ** 2))
        return float(np.max(np.abs(scores)) / energy) if energy else 0.0

    for row in range(0, injected.shape[0], 37):
        own = data['patient'][row]
        others = [key for key in parts if key != own]
        assert best_match(injected[row], parts[own]) > \
               3.0 * max(best_match(injected[row], parts[key]) for key in others)


def test_noise_partitions_are_disjoint_and_cover_the_record():
    from scripts.export_dataset import partition_noise

    noise = np.arange(1000, dtype=float)
    parts = partition_noise(noise, ['c', 'a', 'b', 'a'])
    assert sorted(parts) == ['a', 'b', 'c']
    assert sum(part.size for part in parts.values()) == noise.size
    joined = np.concatenate([parts[key] for key in sorted(parts)])
    assert np.array_equal(joined, noise)


def test_records_of_one_patient_stay_together(model_export):
    for name in ('development', 'test'):
        records = set(load(model_export / f'model_{name}_snrp3.npz')['record'].tolist())
        present = {'201', '202'} & records
        assert present in (set(), {'201', '202'})


def test_the_declared_patient_matches_the_record(model_export):
    data = load(model_export / 'model_development_snrp3.npz')
    for record, patient in zip(data['record'], data['patient']):
        assert patient_of(record) == patient


# --- lead selection ------------------------------------------------------

def test_a_record_without_the_requested_lead_is_skipped(model_export):
    for name in ('development', 'test'):
        records = load(model_export / f'model_{name}_snrp3.npz')['record']
        assert WITHOUT_LEAD not in records.tolist()


def test_every_exported_record_was_requested(model_export):
    records = set(load(model_export / 'model_development_snrp3.npz')['record'].tolist())
    assert records <= set(WITH_LEAD)


# --- manifest ------------------------------------------------------------

def test_the_manifest_describes_every_archive(model_export):
    manifest = json.loads((model_export / 'model_manifest.json').read_text())
    written = {path.name for path in model_export.glob('*.npz')}
    assert {entry['file'] for entry in manifest['archives']} == written


def test_the_manifest_records_the_parameters(model_export):
    manifest = json.loads((model_export / 'model_manifest.json').read_text())
    assert manifest['purpose'] == 'model'
    assert manifest['width'] == 1024
    assert manifest['lead'] == 'MLII'
    assert WITHOUT_LEAD in manifest['skipped']


def test_the_manifest_carries_folds_that_keep_patients_apart(model_export):
    manifest = json.loads((model_export / 'model_manifest.json').read_text())
    assert len(manifest['folds']) == 3
    for fold in manifest['folds']:
        train = {patient_of(record) for record in fold['train']}
        val = {patient_of(record) for record in fold['val']}
        assert not train & val


def test_the_folds_never_reach_into_the_held_out_pool(model_export):
    manifest = json.loads((model_export / 'model_manifest.json').read_text())
    held_out = {patient_of(record) for record in manifest['split']['test']}
    for fold in manifest['folds']:
        used = {patient_of(record) for record in fold['train'] + fold['val']}
        assert not used & held_out


# --- determinism and reporting -------------------------------------------

def test_the_export_is_reproducible(source_dir, tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    for out in (first, second):
        assert run_export(source_dir, out, '--purpose', 'model',
                          '--width', '1024', '--snr', '3', '--seed', '11') == 0
    a = load(first / 'model_development_snrp3.npz')
    b = load(second / 'model_development_snrp3.npz')
    assert np.array_equal(a['noisy'], b['noisy'])


def test_a_dry_run_writes_nothing(source_dir, tmp_path):
    out = tmp_path / 'dry'
    assert run_export(source_dir, out, '--purpose', 'model',
                      '--width', '1024', '--snr', '3', '--dry-run') == 0
    assert list(out.glob('*.npz')) == []


def test_the_window_count_can_be_capped(source_dir, tmp_path):
    out = tmp_path / 'capped'
    assert run_export(source_dir, out, '--purpose', 'model', '--width', '1024',
                      '--snr', '3', '--splits', 'test',
                      '--max-windows-per-record', '2') == 0
    data = load(out / 'model_test_snrp3.npz')
    assert data['clean'].shape[0] == 2 * len(set(data['record'].tolist()))


def test_beat_mode_produces_windows_of_one_width(source_dir, tmp_path):
    out = tmp_path / 'beat'
    assert run_export(source_dir, out, '--purpose', 'model', '--window', 'beat',
                      '--snr', '3', '--splits', 'test') == 0
    data = load(out / 'model_test_snrp3.npz')
    assert data['clean'].ndim == 2
    assert data['clean'].shape[1] == 2 * int(round(60.0 / BPM * FS))


# --- one convention throughout -------------------------------------------

def test_both_purposes_default_to_the_same_convention_and_grid():
    """
    Mixing conventions within one study would make the figures incomparable.

    The same number of decibels means a different amount of noise under `power_ratio`
    than under `nst`, so the choice has to be made once and applied everywhere.
    """
    from scripts.export_dataset import SNR_LEVELS, parse_args, resolve_defaults

    settings = []
    for purpose in ('model', 'filter'):
        args = parse_args(['--purpose', purpose, '--out', '/tmp/unused'])
        resolve_defaults(args)
        settings.append((args.convention, tuple(args.snr)))

    assert settings[0] == settings[1]
    assert settings[0] == ('power_ratio', tuple(SNR_LEVELS))


def test_the_grid_contains_the_published_levels():
    """The central subset reproduces Wang et al. 2023 exactly."""
    from preparing.noise_mixing import WANG_SNR_LEVELS
    from scripts.export_dataset import SNR_LEVELS

    assert set(WANG_SNR_LEVELS) <= set(SNR_LEVELS)


def test_the_grid_is_evenly_spaced():
    from scripts.export_dataset import SNR_LEVELS

    steps = np.diff(np.asarray(SNR_LEVELS))
    assert np.all(steps == steps[0])


def test_the_realised_ratio_matches_the_request_at_every_level(source_dir, tmp_path):
    out = tmp_path / 'grid'
    assert run_export(source_dir, out, '--purpose', 'filter', '--width', '4096',
                      '--snr', '-9', '-1', '11', '--max-windows-per-record', '2') == 0
    for name, requested in (('m9', -9.0), ('m1', -1.0), ('p11', 11.0)):
        data = load(out / f'filter_test_snr{name}.npz')
        assert data['snr_db_realised'] == pytest.approx(requested, abs=1e-3)
        assert data['snr_db_requested'] == pytest.approx(requested)
