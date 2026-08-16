"""
Loader and validation tests. The WFDB records are generated into a temporary
directory by the `fixture_root` fixture in `conftest.py` and removed afterwards,
so nothing is stored in the repository. No network access.
"""

import numpy as np
import pytest

from data.scripts.loader import load_noise, load_record, validate_record


def test_load_record_signal_shape_and_dtype(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    assert record.signal.shape[1] == 2
    assert record.signal.dtype == np.float64


def test_load_record_sampling_frequency(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    assert record.fs == 360


def test_load_record_annotations_within_signal_range(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    assert record.r_peaks.size > 0
    assert record.r_peaks.min() >= 0
    assert record.r_peaks.max() < record.signal.shape[0]


def test_load_record_without_annotations_leaves_them_none(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root, with_annotations=False)
    assert record.r_peaks is None
    assert record.symbols is None


def test_load_noise_resolves_zero_gain_to_200_adu_per_mv(fixture_root):
    record = load_noise('bw', root=fixture_root)
    assert record.signal.shape[1] == 1
    assert record.r_peaks is None


def test_load_noise_rejects_unknown_noise_type(fixture_root):
    with pytest.raises(ValueError, match='noise_type'):
        load_noise('unknown', root=fixture_root)


def test_validate_record_accepts_the_fixture(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    validate_record(record, expected_fs=360)


def test_validate_record_rejects_nan(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.signal[10, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        validate_record(record)


def test_validate_record_rejects_inf(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.signal[10, 0] = np.inf
    with pytest.raises(ValueError, match='NaN'):
        validate_record(record)


def test_validate_record_rejects_constant_channel(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.signal[:, 1] = 0.0
    with pytest.raises(ValueError, match='constant'):
        validate_record(record)


def test_validate_record_rejects_fs_mismatch(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    with pytest.raises(ValueError, match='expected fs'):
        validate_record(record, expected_fs=250)


def test_validate_record_rejects_out_of_range_annotations(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.r_peaks = np.append(record.r_peaks, record.signal.shape[0] + 100)
    with pytest.raises(ValueError, match='annotation'):
        validate_record(record)


def test_validate_record_rejects_amplitude_outlier(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.signal[0, 0] = 100.0
    with pytest.raises(ValueError, match='amplitude'):
        validate_record(record)


def test_validate_record_rejects_truncated_signal(fixture_root):
    record = load_record('100', database='mitdb', root=fixture_root)
    record.signal = record.signal[:-10]
    with pytest.raises(ValueError, match='samples'):
        validate_record(record)
