"""
Network-dependent tests for `data/scripts/physionet.py`.

Skipped by default; run with `pytest --run-network`. Only exercises a single small
record so a full run stays fast even against a live PhysioNet connection.
"""

import pytest
import wfdb

from data.scripts.physionet import get_dir_record

pytestmark = pytest.mark.network


def test_mitdb_has_48_records():
    records = wfdb.get_record_list('mitdb')
    assert len(records) == 48


def test_get_dir_record_downloads_a_complete_record(tmp_path):
    paths = get_dir_record(database='mitdb', records=['100'], root=tmp_path)

    assert paths == [tmp_path / 'data' / 'files' / 'mitdb' / '100']
    record_dir = tmp_path / 'data' / 'files' / 'mitdb'
    for extension in ('.dat', '.hea', '.atr'):
        assert (record_dir / f'100{extension}').is_file()


def test_get_dir_record_skips_existing_files_without_force(tmp_path, monkeypatch):
    get_dir_record(database='mitdb', records=['100'], root=tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('wfdb.dl_database must not be called when files already exist')

    monkeypatch.setattr(wfdb, 'dl_database', fail_if_called)
    get_dir_record(database='mitdb', records=['100'], root=tmp_path, force=False)
