"""Loading and validation of PhysioNet records downloaded by `physionet.py`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]

NOISE_TYPES = ('bw', 'ma', 'em')
MAX_PHYSIOLOGICAL_AMPLITUDE_MV = 50.0


@dataclass
class EcgRecord:
    """A single WFDB record loaded into memory, with signal in physical units."""

    record_id: str
    database: str
    signal: np.ndarray
    fs: float
    channels: list
    units: list
    r_peaks: Optional[np.ndarray]
    symbols: Optional[np.ndarray]
    path: Path


def load_record(record_id: str, database: str = 'mitdb', root: Optional[Path] = None,
                 with_annotations: bool = True) -> EcgRecord:
    """
    Load a WFDB record from `data/files/<database>/<record_id>`.

    Parameters:
        record_id (str): The record to load, e.g. '100'.
        database (str): The database directory the record lives in.
        root (Path, optional): Repository root the record path is resolved against.
            Defaults to the repository root inferred from this file's location.
        with_annotations (bool): Also read the `.atr` annotation file.

    Returns:
        EcgRecord: The record with its signal in physical units (mV).
    """
    root = root if root is not None else ROOT
    record_path = root / 'data' / 'files' / database / record_id

    record = wfdb.rdrecord(str(record_path))

    r_peaks = None
    symbols = None
    if with_annotations:
        annotation = wfdb.rdann(str(record_path), 'atr')
        r_peaks = np.asarray(annotation.sample)
        symbols = np.asarray(annotation.symbol)

    return EcgRecord(
        record_id=record_id,
        database=database,
        signal=np.asarray(record.p_signal, dtype=np.float64),
        fs=float(record.fs),
        channels=list(record.sig_name),
        units=list(record.units),
        r_peaks=r_peaks,
        symbols=symbols,
        path=record_path,
    )


def load_noise(noise_type: str, root: Optional[Path] = None) -> EcgRecord:
    """
    Load an NSTDB noise record ('bw', 'ma' or 'em'). These have no annotations.

    The ADC gain in `bw.hea`, `ma.hea` and `em.hea` is stored as 0, which WFDB
    interprets as unspecified and resolves to the library default of 200 ADU/mV.
    MIT-BIH-derived records (e.g. `118e06`) declare their gain explicitly as
    `200(1024)`. Both resolve to the same 200 ADU/mV scale, but because one is a
    fallback and the other is explicit, this function asserts they agree so that a
    future WFDB version silently changing its default cannot rescale noise mixed
    into a signal by a hidden constant factor.

    Parameters:
        noise_type (str): One of 'bw', 'ma', 'em'.
        root (Path, optional): Repository root the record path is resolved against.
            Defaults to the repository root inferred from this file's location.

    Returns:
        EcgRecord: The noise record with its signal in physical units (mV).
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"unknown noise_type '{noise_type}'; expected one of {NOISE_TYPES}")

    root = root if root is not None else ROOT
    record_path = root / 'data' / 'files' / 'nstdb' / noise_type

    record = wfdb.rdrecord(str(record_path))
    if not all(gain == 200 for gain in record.adc_gain):
        raise ValueError(
            f"noise record '{noise_type}' has unexpected ADC gain {record.adc_gain}; "
            f"expected 200 ADU/mV on every channel, matching MIT-BIH-derived records")

    return EcgRecord(
        record_id=noise_type,
        database='nstdb',
        signal=np.asarray(record.p_signal, dtype=np.float64),
        fs=float(record.fs),
        channels=list(record.sig_name),
        units=list(record.units),
        r_peaks=None,
        symbols=None,
        path=record_path,
    )


def validate_record(record: EcgRecord, expected_fs: Optional[float] = None) -> None:
    """
    Raise ValueError if `record` shows signs of a corrupted or truncated download.

    Checks performed:
        - declared sample count in the header matches the loaded signal
        - no NaN or inf values in the signal
        - no channel is constant (zero standard deviation), a symptom of a
          truncated or all-zero file
        - `expected_fs`, if given, matches the record's sampling frequency
        - annotation sample indices fall within [0, n_samples)
        - signal amplitude stays within a physiologically plausible range

    Parameters:
        record (EcgRecord): The record to validate.
        expected_fs (float, optional): Sampling frequency the record must match.

    Returns:
        None
    """
    n_samples = record.signal.shape[0]
    declared_samples = wfdb.rdheader(str(record.path)).sig_len
    if n_samples != declared_samples:
        raise ValueError(
            f"record '{record.record_id}': header declares {declared_samples} samples, "
            f"but {n_samples} were loaded; the download is likely truncated")

    if not np.all(np.isfinite(record.signal)):
        raise ValueError(f"record '{record.record_id}': signal contains NaN or inf values")

    channel_std = np.std(record.signal, axis=0)
    flat_channels = [record.channels[i] for i in np.flatnonzero(channel_std == 0)]
    if flat_channels:
        raise ValueError(
            f"record '{record.record_id}': channel(s) {flat_channels} are constant, "
            f"which suggests a truncated or corrupted file")

    if expected_fs is not None and record.fs != expected_fs:
        raise ValueError(
            f"record '{record.record_id}': expected fs={expected_fs}, got fs={record.fs}")

    if record.r_peaks is not None and record.r_peaks.size > 0:
        if record.r_peaks.min() < 0 or record.r_peaks.max() >= n_samples:
            raise ValueError(
                f"record '{record.record_id}': annotation indices "
                f"[{record.r_peaks.min()}, {record.r_peaks.max()}] fall outside "
                f"the signal range [0, {n_samples})")

    max_amplitude = np.max(np.abs(record.signal))
    if max_amplitude > MAX_PHYSIOLOGICAL_AMPLITUDE_MV:
        raise ValueError(
            f"record '{record.record_id}': max amplitude {max_amplitude:.1f} mV exceeds "
            f"the physiological sanity limit of {MAX_PHYSIOLOGICAL_AMPLITUDE_MV} mV")


def get_frequency_wfdb(record: EcgRecord) -> float:
    """
    Return the sampling frequency of a WFDB record, as declared in its header.

    Parameters:
        record (EcgRecord): The record to query.

    Returns:
        float: The sampling frequency in Hz.
    """
    return float(wfdb.rdheader(str(record.path)).fs)

def get_frequency_neurobit(record_path: str) -> float:
    """
    Return the sampling frequency of a Neurobit record, based on timestamp inversion
    Open .txt file x_data. Compute time difference and return the inversion of difference as frequency in Hz.
    Parameters:
        record_path (str): The path to the Neurobit record.
    Returns:
        float: The sampling frequency in Hz.
    """
    neurobit_record = np.loadtxt(record_path, delimiter=',', skiprows=1)
    timestamps = neurobit_record[:, 0]
    time_diff = timestamps[1]-timestamps[0]
    return 1/time_diff