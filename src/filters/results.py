"""
Recording of what each method produced, for each window, at each ratio.

A full run is seventeen methods against several thousand windows at six signal to noise
levels, which is of the order of a hundred thousand rows. Two decisions shape the format.

Rows are written as they are produced rather than collected and saved at the end, and the
buffer is emptied after each one. A run of that size takes hours, and a crash in the last
hour would otherwise cost the first five; leaving the rows in the buffer would cost all of
them, which is the same failure wearing the appearance of a solution. Partial results are also useful on their own: the ranking after two patients is
already informative about whether the run is worth finishing.

Failures are rows, not absences. A method that diverged, or that the context could not
supply, gets a row with its status and the reason. Leaving it out would make a method
that failed indistinguishable from one that was never run, and the second is the story
that gets told when nobody remembers.

Waveforms are not stored by default. Seventeen methods over the held out pool at six
ratios come to several gigabytes of float32, and almost none of it is ever looked at. The
figures of the results chapter need a handful of windows, so `waveform_every` keeps every
n-th one and the rest is recomputed from the archive if it is ever wanted.

The measurements themselves are not here. This module records what was produced and what
it cost; the signal to noise ratio, the percentage root mean square difference and the
quality indices are computed by the analysis layer from the waveforms, which keeps a
change of metric from invalidating a run that took hours.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

STATUS_OK = 'ok'
STATUS_BLOCKED = 'blocked'
STATUS_ERROR = 'error'

RESULT_FIELDS = (
    'method', 'family', 'status',
    'record', 'patient', 'snr_db',
    'window_start', 'window_stop',
    'elapsed_s', 'n_reference_channels',
    'covered_start', 'covered_stop',
    'code_version', 'params', 'message',
)


class ResultWriter:
    """
    Streams one row per (method, window, ratio) to a comma separated file.

    Used as a context manager so that the file is closed and the sidecar written even
    when a run is interrupted.
    """

    def __init__(self, path, config: Optional[dict] = None,
                 waveform_path=None, waveform_every: int = 0):
        self.path = Path(path)
        self.config = dict(config or {})
        self.waveform_path = Path(waveform_path) if waveform_path is not None else None
        self.waveform_every = int(waveform_every)

        if self.waveform_every and self.waveform_path is None:
            raise ValueError('waveform_every was set but no waveform_path was given')

        self._handle = None
        self._writer = None
        self._waveforms: dict = {}
        self._counts = {STATUS_OK: 0, STATUS_BLOCKED: 0, STATUS_ERROR: 0}
        self._seen = 0

    # --- lifetime --------------------------------------------------------

    def __enter__(self) -> 'ResultWriter':
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._handle, fieldnames=list(RESULT_FIELDS))
        self._writer.writeheader()
        return self

    def __exit__(self, *_) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

        if self.waveform_path is not None and self._waveforms:
            self.waveform_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.waveform_path, **self._waveforms)

        self._write_sidecar()
        return False

    def _write_sidecar(self) -> None:
        """Everything needed to say what this file is, next to the file itself."""
        sidecar = self.path.with_suffix('.json')
        sidecar.write_text(json.dumps({
            'results': self.path.name,
            'waveforms': None if self.waveform_path is None else self.waveform_path.name,
            'waveform_every': self.waveform_every,
            'written_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'n_rows': sum(self._counts.values()),
            'counts': dict(self._counts),
            'fields': list(RESULT_FIELDS),
            'config': self.config,
        }, indent=2), encoding='utf-8')

    # --- writing ---------------------------------------------------------

    def _row(self, **values) -> None:
        if self._writer is None:
            raise RuntimeError('the writer is used outside its context manager')
        row = {field: '' for field in RESULT_FIELDS}
        row.update({key: value for key, value in values.items() if value is not None})
        self._writer.writerow(row)
        # bufor Pythona trzymalby wiersz w pamieci do zamkniecia pliku, co przekreslaloby
        # sens zapisu strumieniowego: przerwany przebieg zostawilby pusty plik
        self._handle.flush()
        self._counts[row['status']] = self._counts.get(row['status'], 0) + 1

    def write(self, result, record: str, patient: str, snr_db: float,
              window: tuple) -> None:
        """Records a method that ran, and keeps its waveform every n-th time."""
        start, stop = int(window[0]), int(window[1])
        covered = result.covered_span or (None, None)

        self._row(
            method=result.method, family=result.family, status=STATUS_OK,
            record=record, patient=patient, snr_db=f'{float(snr_db):.1f}',
            window_start=start, window_stop=stop,
            elapsed_s=f'{result.elapsed_s:.6f}',
            n_reference_channels=result.n_reference_channels,
            covered_start=covered[0], covered_stop=covered[1],
            code_version=result.code_version,
            params=json.dumps(result.params, sort_keys=True, default=str),
        )

        if self.waveform_every and self._seen % self.waveform_every == 0:
            key = f'{result.method}|{record}|{float(snr_db):+.1f}|{start}'
            self._waveforms[key] = np.asarray(result.signal, dtype=np.float32)
        self._seen += 1

    def blocked(self, method: str, family: str, record: str, patient: str,
                snr_db: float, window: tuple, missing) -> None:
        """
        Records a method the context could not run.

        Adaptive methods on a database with no accelerometer are the ordinary case, and
        the table has to say so rather than leave the reader to notice the gap.
        """
        self._row(method=method, family=family, status=STATUS_BLOCKED,
                  record=record, patient=patient, snr_db=f'{float(snr_db):.1f}',
                  window_start=int(window[0]), window_stop=int(window[1]),
                  message='brak: ' + ', '.join(missing))

    def failed(self, method: str, family: str, record: str, patient: str,
               snr_db: float, window: tuple, error: BaseException) -> None:
        """Records a method that raised, so that a divergence is visible in the table."""
        self._row(method=method, family=family, status=STATUS_ERROR,
                  record=record, patient=patient, snr_db=f'{float(snr_db):.1f}',
                  window_start=int(window[0]), window_stop=int(window[1]),
                  message=f'{type(error).__name__}: {error}'[:300])

    @property
    def counts(self) -> dict:
        return dict(self._counts)


def run_window(name: str, noisy, context, record: str, patient: str,
               snr_db: float, window: tuple, writer: ResultWriter, **overrides):
    """
    Applies one method to one window and records the outcome whatever it is.

    Returns the result, or None when the method was blocked or raised. Catching the
    exception here rather than letting it stop the run is deliberate: one diverging filter
    at one ratio should cost that row, not the remaining hours.
    """
    try:
        from .registry import apply_filter, get_spec
    except ImportError:
        from registry import apply_filter, get_spec

    spec = get_spec(name)
    missing = spec.missing(context)
    if missing:
        writer.blocked(name, spec.family, record, patient, snr_db, window, missing)
        return None

    try:
        result = apply_filter(name, noisy, context, **overrides)
    except Exception as error:                                      # noqa: BLE001
        writer.failed(name, spec.family, record, patient, snr_db, window, error)
        return None

    writer.write(result, record=record, patient=patient, snr_db=snr_db, window=window)
    return result


def read_results(path) -> list:
    """Reads a result file back, with the numeric fields converted."""
    numeric = {'snr_db': float, 'window_start': int, 'window_stop': int,
               'elapsed_s': float, 'n_reference_channels': int,
               'covered_start': int, 'covered_stop': int}

    rows = []
    with Path(path).open('r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            for field, cast in numeric.items():
                if row.get(field):
                    row[field] = cast(row[field])
                else:
                    row[field] = None
            row['params'] = json.loads(row['params']) if row['params'] else {}
            rows.append(row)
    return rows


def summarise(rows) -> dict:
    """
    What a run contained, by method: how many rows, how many failed, how long it took.

    The cost column of the results chapter comes from here; the quality columns do not,
    since nothing in this module measures quality.
    """
    summary: dict = {}
    for row in rows:
        entry = summary.setdefault(row['method'], {
            'family': row['family'], 'ok': 0, 'blocked': 0, 'error': 0,
            'elapsed_total_s': 0.0, 'messages': set()})
        entry[row['status']] = entry.get(row['status'], 0) + 1
        if row['status'] == STATUS_OK and row['elapsed_s'] is not None:
            entry['elapsed_total_s'] += row['elapsed_s']
        if row['message']:
            entry['messages'].add(row['message'][:80])

    for entry in summary.values():
        entry['elapsed_mean_ms'] = (1000.0 * entry['elapsed_total_s'] / entry['ok']
                                    if entry['ok'] else None)
        entry['messages'] = sorted(entry['messages'])
    return summary


def format_summary(rows) -> str:
    """Table of a run, for the console and for the thesis appendix."""
    summary = summarise(rows)
    lines = ['%-18s %-9s %7s %8s %7s %12s' % (
        'metoda', 'rodzina', 'ok', 'zabl.', 'blad', 'sr. czas [ms]'), '-' * 68]
    for name in sorted(summary):
        entry = summary[name]
        mean = entry['elapsed_mean_ms']
        lines.append('%-18s %-9s %7d %8d %7d %12s' % (
            name, entry['family'], entry['ok'], entry['blocked'], entry['error'],
            '-' if mean is None else f'{mean:.2f}'))
    return '\n'.join(lines)
