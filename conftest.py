"""Shared pytest configuration: network/slow opt-in markers and repo root on sys.path."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--run-network', action='store_true', default=False,
                      help='run tests marked "network" (requires PhysioNet access)')
    parser.addoption('--run-slow', action='store_true', default=False,
                      help='run tests marked "slow"')


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    skip_network = pytest.mark.skip(reason='needs --run-network to run')
    skip_slow = pytest.mark.skip(reason='needs --run-slow to run')
    run_network = config.getoption('--run-network')
    run_slow = config.getoption('--run-slow')

    for item in items:
        if 'network' in item.keywords and not run_network:
            item.add_marker(skip_network)
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)


# --- synthetic WFDB material shared by the data layer tests ----------------

FIXTURE_FS = 360
FIXTURE_DURATION_S = 30
FIXTURE_BPM = 75
FIXTURE_NOISE_S = 240

FIXTURE_WITH_LEAD = ('100', '101', '103', '105', '200', '201', '202', '203', '205')
FIXTURE_WITHOUT_LEAD = '102'
FIXTURE_RECORDS = tuple(sorted(FIXTURE_WITH_LEAD + (FIXTURE_WITHOUT_LEAD,)))


def _synthetic_ecg(seed: int):
    """One ECG-like channel in millivolts, with regular complexes at a known rate."""
    import numpy as np

    rng = np.random.RandomState(seed)
    n = FIXTURE_FS * FIXTURE_DURATION_S
    rr = int(60.0 / FIXTURE_BPM * FIXTURE_FS)
    r_peaks = np.arange(rr, n - rr, rr)

    channel = np.zeros(n)
    for peak in r_peaks:
        channel[peak - 5:peak + 5] += np.hanning(10) * 1.2
        channel[peak + 20:peak + 60] += np.hanning(40) * 0.2
    channel += 0.02 * rng.randn(n)
    return np.stack([channel, 0.6 * channel], axis=1), r_peaks


@pytest.fixture(scope='session')
def wfdb_source(tmp_path_factory):
    """
    A miniature MIT-BIH and NSTDB, with the parameters it was built from.

    Returned as a namespace rather than a bare path so that a test never has to import
    this file; two directories in the tree hold a `conftest.py` and which one wins depends
    on collection order.

    One record carries a different lead name so that the lead selection path is exercised,
    and one patient contributes two records so that patient grouping is exercised too.
    Generation takes milliseconds, which is cheaper than carrying binary fixtures in
    version control.
    """
    import shutil

    import numpy as np
    import wfdb

    root = tmp_path_factory.mktemp('wfdb_source')
    for index, record_id in enumerate(FIXTURE_RECORDS):
        signal, r_peaks = _synthetic_ecg(index)
        names = ['V5', 'V1'] if record_id == FIXTURE_WITHOUT_LEAD else ['MLII', 'V1']
        wfdb.wrsamp(record_id, fs=FIXTURE_FS, units=['mV', 'mV'], sig_name=names,
                    p_signal=signal, fmt=['16', '16'], write_dir=str(root))
        wfdb.wrann(record_id, 'atr', np.asarray(r_peaks),
                   np.array(['N'] * r_peaks.size), write_dir=str(root))

    rng = np.random.RandomState(7)
    n = FIXTURE_FS * FIXTURE_NOISE_S
    noise = np.cumsum(rng.randn(n)) * 0.004 + 0.25 * rng.randn(n)
    wfdb.wrsamp('em', fs=FIXTURE_FS, units=['mV'], sig_name=['noise'],
                p_signal=noise[:, None], fmt=['16'], write_dir=str(root))

    yield SimpleNamespace(
        root=root,
        fs=FIXTURE_FS,
        bpm=FIXTURE_BPM,
        records=FIXTURE_RECORDS,
        with_lead=FIXTURE_WITH_LEAD,
        without_lead=FIXTURE_WITHOUT_LEAD,
    )
    shutil.rmtree(root, ignore_errors=True)
