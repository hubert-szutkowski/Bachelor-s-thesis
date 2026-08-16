"""
Regenerates the small WFDB fixture record used by `tests/data_tests/test_loader.py`.

Not part of the test suite; run manually with `python generate_fixture.py` after
deleting the old fixture files, then commit the result.
"""

import re
from pathlib import Path

import numpy as np
import wfdb

FIXTURES_ROOT = Path(__file__).resolve().parent
MITDB_DIR = FIXTURES_ROOT / 'data' / 'files' / 'mitdb'
NSTDB_DIR = FIXTURES_ROOT / 'data' / 'files' / 'nstdb'
RECORD_ID = '100'
NOISE_ID = 'bw'
FS = 360
DURATION_S = 20
NOISE_DURATION_S = 2
BPM = 75


def synthetic_ecg(fs: int, duration_s: int, bpm: float, seed: int = 0) -> tuple:
    """Two-channel synthetic ECG-like signal with regular R-peaks, in mV."""
    rng = np.random.RandomState(seed)
    n = fs * duration_s
    rr = int(60.0 / bpm * fs)
    r_peaks = np.arange(rr // 2, n - rr // 2, rr)

    channel = np.zeros(n)
    for r in r_peaks:
        channel[r - 5:r + 5] += np.hanning(10) * 1.2
        channel[r + 20:r + 60] += np.hanning(40) * 0.2
    channel += 0.02 * rng.randn(n)

    signal = np.stack([channel, 0.6 * channel + 0.01 * rng.randn(n)], axis=1)
    return signal, r_peaks


def zero_out_gain(hea_path: Path) -> None:
    """
    Rewrites the ADC gain field to 0 ("unspecified"), matching the real NSTDB
    noise headers (`bw.hea`, `ma.hea`, `em.hea`), where WFDB falls back to the
    library default of 200 ADU/mV.
    """
    text = hea_path.read_text()
    text = re.sub(r'(?<=\.dat 16 )[0-9.]+(?=\()', '0', text, count=1)
    hea_path.write_text(text)


def main() -> None:
    MITDB_DIR.mkdir(parents=True, exist_ok=True)
    signal, r_peaks = synthetic_ecg(FS, DURATION_S, BPM)

    wfdb.wrsamp(
        RECORD_ID, fs=FS, units=['mV', 'mV'], sig_name=['MLII', 'V1'],
        p_signal=signal, fmt=['16', '16'], write_dir=str(MITDB_DIR),
    )
    wfdb.wrann(
        RECORD_ID, 'atr', sample=r_peaks, symbol=['N'] * len(r_peaks),
        write_dir=str(MITDB_DIR),
    )
    print(f'wrote {MITDB_DIR / RECORD_ID}.{{dat,hea,atr}} ({len(r_peaks)} beats)')

    NSTDB_DIR.mkdir(parents=True, exist_ok=True)
    noise = 0.1 * np.random.RandomState(1).randn(FS * NOISE_DURATION_S, 1)
    wfdb.wrsamp(
        NOISE_ID, fs=FS, units=['mV'], sig_name=['noise'],
        p_signal=noise, fmt=['16'], write_dir=str(NSTDB_DIR),
    )
    zero_out_gain(NSTDB_DIR / f'{NOISE_ID}.hea')
    print(f'wrote {NSTDB_DIR / NOISE_ID}.{{dat,hea}} with gain zeroed out')


if __name__ == '__main__':
    main()
