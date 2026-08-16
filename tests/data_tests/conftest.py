"""
Builds the WFDB records used by the loader tests.

The records are generated into a temporary directory at the start of the session and
removed when it ends, so nothing is written inside the repository and nothing has to
be committed. Generation takes a few milliseconds, which is cheaper than carrying
binary fixtures in version control.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import wfdb

RECORD_ID = '100'
NOISE_ID = 'bw'
FS = 360
DURATION_S = 20
NOISE_DURATION_S = 2
BPM = 75


def synthetic_ecg(fs: int, duration_s: int, bpm: float, seed: int = 0) -> tuple:
    """Two-channel synthetic ECG-like signal with regular R-peaks, in millivolts."""
    rng = np.random.RandomState(seed)
    n = fs * duration_s
    rr = int(60.0 / bpm * fs)
    r_peaks = np.arange(rr // 2, n - rr // 2, rr)

    channel = np.zeros(n)
    for peak in r_peaks:
        channel[peak - 5:peak + 5] += np.hanning(10) * 1.2
        channel[peak + 20:peak + 60] += np.hanning(40) * 0.2
    channel += 0.02 * rng.randn(n)

    signal = np.stack([channel, 0.6 * channel + 0.01 * rng.randn(n)], axis=1)
    return signal, r_peaks


def zero_out_gain(hea_path: Path) -> None:
    """
    Rewrites the ADC gain field to 0, meaning unspecified.

    Reproduces the real NSTDB noise headers `bw.hea`, `ma.hea` and `em.hea`, where the
    gain is absent and WFDB falls back to its default of 200 units per millivolt. The
    loader has to treat that default as a convention rather than a measurement, and
    this fixture is what exercises that path.
    """
    text = hea_path.read_text()
    text = re.sub(r'(?<=\.dat 16 )[0-9.]+(?=\()', '0', text, count=1)
    hea_path.write_text(text)


def build_records(root: Path) -> None:
    """Writes one MIT-BIH-like record with annotations and one noise record."""
    mitdb_dir = root / 'data' / 'files' / 'mitdb'
    nstdb_dir = root / 'data' / 'files' / 'nstdb'
    mitdb_dir.mkdir(parents=True, exist_ok=True)
    nstdb_dir.mkdir(parents=True, exist_ok=True)

    signal, r_peaks = synthetic_ecg(FS, DURATION_S, BPM)
    wfdb.wrsamp(
        RECORD_ID, fs=FS, units=['mV', 'mV'], sig_name=['MLII', 'V1'],
        p_signal=signal, fmt=['16', '16'], write_dir=str(mitdb_dir),
    )
    wfdb.wrann(
        RECORD_ID, 'atr', sample=r_peaks, symbol=['N'] * len(r_peaks),
        write_dir=str(mitdb_dir),
    )

    noise = 0.1 * np.random.RandomState(1).randn(FS * NOISE_DURATION_S, 1)
    wfdb.wrsamp(
        NOISE_ID, fs=FS, units=['mV'], sig_name=['noise'],
        p_signal=noise, fmt=['16'], write_dir=str(nstdb_dir),
    )
    zero_out_gain(nstdb_dir / f'{NOISE_ID}.hea')


@pytest.fixture(scope='session')
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Root directory holding freshly generated WFDB records, removed after the session.

    Session scope keeps generation to a single pass. Sharing the files between tests
    is safe because every test reloads the record from disk, so mutations applied to a
    loaded object never reach the next test.
    """
    root = tmp_path_factory.mktemp('wfdb_records')
    build_records(root)
    yield root
    shutil.rmtree(root, ignore_errors=True)
