"""
Invariants of the exported data layer, and its fit with the training entry point.

These are the checks that would otherwise be made by looking at a training curve and
guessing. A window holding a single not-a-number silently poisons a whole batch through
the loss; an amplitude off by a factor of a thousand trains a model that will never see
that scale again; a signal to noise ratio that misses its target by a decibel shifts the
axis every result is plotted against. None of the three announces itself.

The last section is the acceptance test for the export format: an archive written by
`scripts/export_dataset.py` must be readable by `train/cli.py` without pickle, must carry
its sampling frequency, and must be split by patient rather than by segment index.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from preparing.splitting import patient_of
from scripts.export_dataset import main as export_main
from train.dataset_io import load_dataset, split_by_patient, split_indices

REQUESTED_SNR = (-9.0, -1.0, 7.0)


@pytest.fixture(scope='session')
def archives(wfdb_source, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp('data_layer')
    code = export_main([
        '--purpose', 'model', '--source', str(wfdb_source.root), '--out', str(out),
        '--records', *wfdb_source.records, '--width', '1024',
        '--test-patients', '2', '--folds', '3',
        '--snr', *[str(value) for value in REQUESTED_SNR]])
    assert code == 0
    return out


def archive(root: Path, snr_db: float) -> Path:
    tag = f'{snr_db:+.0f}'.replace('+', 'p').replace('-', 'm')
    return root / f'model_development_snr{tag}.npz'


def read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# --- shapes --------------------------------------------------------------

@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_pairs_agree_in_shape(archives, snr_db):
    data = read(archive(archives, snr_db))
    assert data['noisy'].shape == data['clean'].shape
    assert data['clean'].ndim == 2


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_the_declared_width_is_the_actual_width(archives, snr_db):
    data = read(archive(archives, snr_db))
    assert data['clean'].shape[1] == int(data['width']) == 1024


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_every_side_array_matches_the_segment_count(archives, snr_db):
    data = read(archive(archives, snr_db))
    n = data['clean'].shape[0]
    for key in ('noisy', 'record', 'patient', 'start', 'gain', 'snr_db_realised'):
        assert data[key].shape[0] == n, key
    assert data['r_peaks_offset'].shape[0] == n + 1


def test_the_ratio_does_not_change_the_shape(archives):
    shapes = {read(archive(archives, snr))['clean'].shape for snr in REQUESTED_SNR}
    assert len(shapes) == 1


# --- finiteness ----------------------------------------------------------

@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_no_window_holds_a_non_finite_sample(archives, snr_db):
    """A single one of these poisons a whole batch through the loss."""
    data = read(archive(archives, snr_db))
    assert np.all(np.isfinite(data['clean']))
    assert np.all(np.isfinite(data['noisy']))


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_no_window_is_constant(archives, snr_db):
    """A flat window carries no information and divides by zero in every ratio metric."""
    data = read(archive(archives, snr_db))
    assert np.all(data['clean'].std(axis=1) > 0)
    assert np.all(data['noisy'].std(axis=1) > 0)


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_gains_are_finite_and_positive(archives, snr_db):
    data = read(archive(archives, snr_db))
    assert np.all(np.isfinite(data['gain']))
    assert np.all(data['gain'] > 0)


# --- ranges --------------------------------------------------------------

@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_clean_amplitudes_stay_in_a_plausible_range(archives, snr_db):
    """
    Millivolts, not microvolts and not raw converter counts.

    A unit error of a thousand passes every shape check and produces a model trained on a
    scale it will never meet again.
    """
    data = read(archive(archives, snr_db))
    peak = np.abs(data['clean']).max()
    assert 0.05 < peak < 20.0


def test_noise_grows_as_the_ratio_falls(archives):
    """Monotone by construction; a break here means the gain calculation is inverted."""
    spread = [float(np.std(read(archive(archives, snr))['noisy'] -
                           read(archive(archives, snr))['clean']))
              for snr in sorted(REQUESTED_SNR)]
    assert spread == sorted(spread, reverse=True)


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_the_clean_reference_does_not_depend_on_the_ratio(archives, snr_db):
    """Only the noise changes between archives; the target must not."""
    reference = read(archive(archives, REQUESTED_SNR[0]))['clean']
    assert np.array_equal(read(archive(archives, snr_db))['clean'], reference)


# --- signal to noise distribution ----------------------------------------

@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_the_realised_ratio_hits_the_request_on_every_window(archives, snr_db):
    data = read(archive(archives, snr_db))
    assert data['snr_db_realised'] == pytest.approx(snr_db, abs=1e-3)
    assert float(data['snr_db_requested']) == pytest.approx(snr_db)


@pytest.mark.parametrize('snr_db', REQUESTED_SNR)
def test_the_ratio_recomputed_from_the_waveforms_agrees(archives, snr_db):
    """Recomputed from the stored waveforms, not read back from the stored metadata."""
    from preparing.noise_mixing import power_ratio_snr

    data = read(archive(archives, snr_db))
    measured = [power_ratio_snr(clean, noisy)
                for clean, noisy in zip(data['clean'][:40], data['noisy'][:40])]
    assert np.allclose(measured, snr_db, atol=1e-3)


# --- beat positions ------------------------------------------------------

def test_beat_indices_are_relative_to_their_window(archives):
    data = read(archive(archives, REQUESTED_SNR[0]))
    width = data['clean'].shape[1]
    offsets = data['r_peaks_offset']
    for index in range(offsets.size - 1):
        peaks = data['r_peaks'][offsets[index]:offsets[index + 1]]
        assert np.all((peaks >= 0) & (peaks < width))


def test_the_beat_count_matches_the_known_heart_rate(archives, wfdb_source):
    data = read(archive(archives, REQUESTED_SNR[0]))
    counts = np.diff(data['r_peaks_offset'])
    expected = data['clean'].shape[1] / (60.0 / wfdb_source.bpm * wfdb_source.fs)
    assert counts.mean() == pytest.approx(expected, abs=0.6)


def test_offsets_are_non_decreasing_and_cover_the_vector(archives):
    data = read(archive(archives, REQUESTED_SNR[0]))
    offsets = data['r_peaks_offset']
    assert np.all(np.diff(offsets) >= 0)
    assert offsets[0] == 0
    assert offsets[-1] == data['r_peaks'].size


# --- fit with the training entry point -----------------------------------

def test_the_archive_reads_without_pickle(archives):
    """
    Reading an object array requires executing whatever the file contains.

    The flat layout carries the same beat positions as plain integers, so the archive can
    be opened with pickle disabled.
    """
    with np.load(archive(archives, -1.0), allow_pickle=False) as data:
        assert 'clean' in data.files


def test_the_training_entry_point_loads_the_archive(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    assert payload['noisy'].shape == payload['clean'].shape
    assert payload['patient'] is not None
    assert len(payload['r_peaks']) == payload['noisy'].shape[0]


def test_the_sampling_frequency_survives_the_round_trip(archives, wfdb_source):
    """
    Not defaulted anywhere along the way.

    A default of 250 Hz applied to a 360 Hz database shifts every quantity expressed in
    hertz and raises nothing.
    """
    payload = load_dataset(str(archive(archives, -1.0)))
    assert payload['fs'] == pytest.approx(wfdb_source.fs)


def test_an_archive_without_a_sampling_frequency_is_rejected(tmp_path):
    path = tmp_path / 'nofs.npz'
    np.savez(path, clean=np.zeros((4, 8), np.float32), noisy=np.ones((4, 8), np.float32))
    with pytest.raises(KeyError, match='fs'):
        load_dataset(str(path))


def test_the_training_split_separates_patients(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    train_idx, val_idx = split_by_patient(payload['patient'], 0.25)
    assert not (set(payload['patient'][train_idx].tolist()) &
                set(payload['patient'][val_idx].tolist()))


def test_the_training_split_uses_every_segment_once(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    train_idx, val_idx = split_by_patient(payload['patient'], 0.25)
    assert sorted(np.concatenate([train_idx, val_idx])) == list(range(len(payload['noisy'])))


def test_the_realised_validation_share_is_close_to_the_request(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    _, val_idx = split_by_patient(payload['patient'], 0.3)
    assert len(val_idx) / len(payload['noisy']) == pytest.approx(0.3, abs=0.15)


def test_the_patient_split_never_leaves_a_side_empty(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    for fraction in (0.05, 0.25, 0.5, 0.9):
        train_idx, val_idx = split_by_patient(payload['patient'], fraction)
        assert train_idx.size > 0 and val_idx.size > 0


def test_a_single_patient_cannot_be_split(archives):
    with pytest.raises(ValueError, match='at least two patients'):
        split_by_patient(np.array(['100'] * 10), 0.2)


def test_the_index_split_remains_available_as_a_fallback():
    train_idx, val_idx = split_indices(100, 0.2)
    assert train_idx.size == 80 and val_idx.size == 20


def test_the_patient_split_differs_from_the_index_split(archives):
    """
    The two disagree, which is the whole reason the patient labels are carried.

    A cut on segment index lands inside whichever patient straddles it.
    """
    payload = load_dataset(str(archive(archives, -1.0)))
    _, by_patient = split_by_patient(payload['patient'], 0.25)
    _, by_index = split_indices(len(payload['noisy']), 0.25)

    straddling = set(payload['patient'][by_index].tolist()) & \
                 set(payload['patient'][np.setdiff1d(np.arange(len(payload['noisy'])),
                                                     by_index)].tolist())
    assert straddling or not np.array_equal(np.sort(by_patient), np.sort(by_index))


# --- provenance ----------------------------------------------------------

def test_the_archive_records_how_it_was_made(archives):
    data = read(archive(archives, -1.0))
    assert str(data['convention']) == 'power_ratio'
    assert str(data['lead']) == 'MLII'
    assert str(data['noise']) == 'em'
    assert str(data['window_mode']) == 'sliding'
    assert float(data['overlap']) == pytest.approx(0.5)


def test_every_exported_patient_matches_its_record(archives, wfdb_source):
    data = read(archive(archives, -1.0))
    for record, patient in zip(data['record'], data['patient']):
        assert patient_of(str(record)) == str(patient)
    assert set(data['record'].tolist()) <= set(wfdb_source.with_lead)


# --- beats: what is measured against, and what is consumed ---------------

def test_the_archive_carries_both_annotated_and_detected_beats(archives):
    """
    Two different things that must never be confused.

    Annotations are the truth detection is scored against. Detected beats are what a
    method would have outside a database, and are what the cardiac cycle architectures are
    given; Rasti-Meymandi and Ghaffari locate the beats after the noise is inserted for
    exactly this reason.
    """
    data = read(archive(archives, -1.0))
    for key in ('r_peaks', 'r_peaks_offset',
                'r_peaks_detected', 'r_peaks_detected_offset'):
        assert key in data, key


def test_the_two_sets_of_beats_differ(archives):
    """An identical pair would mean the detector never ran."""
    data = read(archive(archives, -9.0))
    annotated = data['r_peaks_offset']
    detected = data['r_peaks_detected_offset']
    assert annotated.shape == detected.shape
    assert not np.array_equal(data['r_peaks'], data['r_peaks_detected']) or \
           annotated[-1] != detected[-1]


def test_detected_beats_fall_inside_their_window(archives):
    data = read(archive(archives, -1.0))
    width = data['clean'].shape[1]
    offsets = data['r_peaks_detected_offset']
    for index in range(offsets.size - 1):
        peaks = data['r_peaks_detected'][offsets[index]:offsets[index + 1]]
        assert np.all((peaks >= 0) & (peaks < width))


def test_detection_degrades_as_the_ratio_falls(archives):
    """
    Which is the whole point of not handing the models the annotations.

    At a low signal to noise ratio the detector loses beats and finds ones that are not
    there, and a model segmented on annotations would never meet that.
    """
    counts = {}
    for snr in REQUESTED_SNR:
        data = read(archive(archives, snr))
        annotated = np.diff(data['r_peaks_offset']).astype(float)
        detected = np.diff(data['r_peaks_detected_offset']).astype(float)
        counts[snr] = float(np.mean(np.abs(detected - annotated)))
    assert counts[min(REQUESTED_SNR)] >= counts[max(REQUESTED_SNR)]


def test_the_detector_is_recorded_in_the_archive_and_the_manifest(archives):
    data = read(archive(archives, -1.0))
    assert str(data['detector'])
    manifest = json.loads((archives / 'model_manifest.json').read_text())
    assert manifest['detector'] == str(data['detector'])


def test_the_training_entry_point_reads_the_detected_beats(archives):
    payload = load_dataset(str(archive(archives, -1.0)))
    assert payload['r_peaks_detected'] is not None
    assert len(payload['r_peaks_detected']) == payload['noisy'].shape[0]


def test_detection_can_be_switched_off(wfdb_source, tmp_path):
    """For a run that deliberately wants the annotations, and says so."""
    out = tmp_path / 'nodetect'
    assert export_main([
        '--purpose', 'model', '--source', str(wfdb_source.root), '--out', str(out),
        '--records', *wfdb_source.records, '--width', '1024', '--test-patients', '2',
        '--folds', '3', '--snr', '3', '--splits', 'test', '--no-detect']) == 0

    data = read(out / 'model_test_snrp3.npz')
    assert data['r_peaks_detected'].size == 0
    assert str(data['detector']) == 'none'
