"""Download helpers for the PhysioNet MIT-BIH Arrhythmia and NSTDB databases."""

from pathlib import Path
from typing import Optional

import wfdb

ROOT = Path(__file__).resolve().parents[2]

MITDB_RECORD_FILES = ('.dat', '.hea', '.atr')
NOISE_RECORD_FILES = ('.dat', '.hea')


def get_records_ids(database: str = 'mitdb', records: Optional[list] = None) -> list:
    """
    List the record IDs available in a PhysioNet database.

    Parameters:
        database (str): The name of the PhysioNet database.
        records (list, optional): Explicit record IDs to use instead of querying
            PhysioNet. When None, all records in the database are returned.

    Returns:
        list: The requested record IDs.
    """
    if records is not None:
        return records
    return wfdb.get_record_list(database)


def _record_complete(record_dir: Path, record_id: str, extensions: tuple) -> bool:
    """Returns True when every expected file of a record is already on disk."""
    return all((record_dir / f'{record_id}{ext}').exists() for ext in extensions)


def get_dir_record(database: str = 'mitdb', records: Optional[list] = None,
                    root: Optional[Path] = None, force: bool = False) -> list:
    """
    Download records from a PhysioNet database into a single per-database directory.

    Parameters:
        database (str): The name of the PhysioNet database.
        records (list, optional): Record IDs to download. Defaults to the entire
            database.
        root (Path, optional): Repository root the download is anchored to.
            Defaults to the repository root inferred from this file's location.
        force (bool): Re-download records even if they already exist locally.

    Returns:
        list[Path]: Base paths `record_dir / record_id` for every requested record.
    """
    root = root if root is not None else ROOT
    record_ids = get_records_ids(database, records)

    record_dir = root / 'data' / 'files' / database
    record_dir.mkdir(parents=True, exist_ok=True)

    missing = record_ids if force else [
        record_id for record_id in record_ids
        if not _record_complete(record_dir, record_id, MITDB_RECORD_FILES)
    ]
    if missing:
        wfdb.dl_database(database, dl_dir=str(record_dir), records=missing)

    return [record_dir / record_id for record_id in record_ids]


def get_noises(noises: Optional[list] = None, database: str = 'nstdb',
                noised_patients: Optional[list] = None, snr: Optional[list] = None,
                root: Optional[Path] = None, force: bool = False) -> list:
    """
    Download NSTDB noise records and noise-stressed MIT-BIH patients.

    Parameters:
        noises (list): Pure noise record IDs to download ('bw', 'em', 'ma').
        database (str): The name of the PhysioNet database.
        noised_patients (list): Patient IDs with noise-stressed variants.
        snr (list): SNR suffixes to combine with `noised_patients`, e.g. '_6', '06'.
        root (Path, optional): Repository root the download is anchored to.
            Defaults to the repository root inferred from this file's location.
        force (bool): Re-download records even if they already exist locally.

    Returns:
        list[Path]: Base paths `record_dir / record_id` for every requested record.
    """
    root = root if root is not None else ROOT
    noises = noises if noises is not None else ['bw', 'em', 'ma']
    noised_patients = noised_patients if noised_patients is not None else ['118', '119']
    snr = snr if snr is not None else ['_6', '00', '06', '12', '18', '24']

    record_dir = root / 'data' / 'files' / database
    record_dir.mkdir(parents=True, exist_ok=True)

    patient_records = [f'{patient}e{s}' for patient in noised_patients for s in snr]
    record_ids = list(noises) + patient_records

    missing = record_ids if force else [
        record_id for record_id in record_ids
        if not _record_complete(record_dir, record_id, NOISE_RECORD_FILES)
    ]
    if missing:
        wfdb.dl_database(database, dl_dir=str(record_dir), records=missing)

    return [record_dir / record_id for record_id in record_ids]
