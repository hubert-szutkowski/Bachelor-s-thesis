"""
Loader and validation tests, run entirely against the checked-in fixture record in
`tests/data_tests/fixtures/` (see `generate_fixture.py`). No network access.
"""

from pathlib import Path

import numpy as np
import pytest

from data.scripts.loader import load_noise, load_record, validate_record

FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def test_load_record_signal_shape_and_dtype():
    record = load_record('100', database='mitdb', root=FIXTURES)
    assert record.signal.shape[1] == 2
    assert record.signal.dtype == np.float64


def test_load_record_sampling_frequency():
    record = load_record('100', database='mitdb', root=FIXTURES)
    assert record.fs == 360


def test_load_record_annotations_within_signal_range():
    record = load_record('100', database='mitdb', root=FIXTURES)
    assert record.r_peaks.size > 0
    assert record.r_peaks.min() >= 0
    assert record.r_peaks.max() < record.signal.shape[0]


def test_load_record_without_annotations_leaves_them_none():
    record = load_record('100', database='mitdb', root=FIXTURES, with_annotations=False)
    assert record.r_peaks is None
    assert record.symbols is None


def test_load_noise_resolves_zero_gain_to_200_adu_per_mv():
    record = load_noise('bw', root=FIXTURES)
    assert record.signal.shape[1] == 1
    assert record.r_peaks is None


def test_load_noise_rejects_unknown_noise_type():
    with pytest.raises(ValueError, match='noise_type'):
        load_noise('unknown', root=FIXTURES)


def test_validate_record_accepts_the_fixture():
    record = load_record('100', database='mitdb', root=FIXTURES)
    validate_record(record, expected_fs=360)


def test_validate_record_rejects_nan():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.signal[10, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        validate_record(record)


def test_validate_record_rejects_inf():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.signal[10, 0] = np.inf
    with pytest.raises(ValueError, match='NaN'):
        validate_record(record)


def test_validate_record_rejects_constant_channel():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.signal[:, 1] = 0.0
    with pytest.raises(ValueError, match='constant'):
        validate_record(record)


def test_validate_record_rejects_fs_mismatch():
    record = load_record('100', database='mitdb', root=FIXTURES)
    with pytest.raises(ValueError, match='expected fs'):
        validate_record(record, expected_fs=250)


def test_validate_record_rejects_out_of_range_annotations():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.r_peaks = np.append(record.r_peaks, record.signal.shape[0] + 100)
    with pytest.raises(ValueError, match='annotation'):
        validate_record(record)


def test_validate_record_rejects_amplitude_outlier():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.signal[0, 0] = 100.0
    with pytest.raises(ValueError, match='amplitude'):
        validate_record(record)


def test_validate_record_rejects_truncated_signal():
    record = load_record('100', database='mitdb', root=FIXTURES)
    record.signal = record.signal[:-10]
    with pytest.raises(ValueError, match='samples'):
        validate_record(record)
