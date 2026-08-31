"""
Exports windowed, noised ECG material to compressed `.npz` archives.

One script serves both consumers, selected with `--purpose`.

`model` produces the material the networks are trained and validated on. Windows are
narrow, taken with an overlap that multiplies the training material and shifts the phase
at which a complex meets the window edge, and the archive covers the development pool so
that `preparing.splitting.group_kfold` can divide it into folds at training time.

`filter` produces the material the deterministic filters are measured on. Windows are
wide, since a static filter needs no training and a longer window gives its transient
room to decay before the measured part begins, and the archive covers the held out pool
so that the deterministic and the learned methods are scored on the same patients.

Noise is partitioned by patient. Each patient receives one contiguous stretch of the
noise record and every window of that patient draws from that stretch alone, so any
split made along patient lines is disjoint in noise as well without the caller having to
arrange it. Without that, a network could learn the particular waveform of a noise
stretch it will meet again in validation, and the metric would improve for a reason that
has nothing to do with denoising.

Every archive carries the sampling frequency alongside the waveforms. `train/cli.py`
falls back to 250 Hz when it is absent, which is silently wrong for a 360 Hz database and
would shift every quantity expressed in hertz without raising anything.

Beat positions are stored flat, as one concatenated vector of indices with a companion
vector of offsets, rather than as an array of objects. An object array can only be read
back with pickle enabled, which means executing whatever the file contains; the flat form
carries the same information as plain integers.

Noise is mixed per window rather than across the whole record. The windows are scored
individually and never reassembled, so a window is the natural unit, and the noise record
is in any case no longer than a recording.

One signal to noise convention is used throughout. `power_ratio` is the plain ratio of
sums of squares of equations (11) and (12) of Wang et al. 2023, which is what the deep
learning denoising literature reports and therefore what makes these results comparable
with the works whose architectures are reimplemented here. The `nst` convention of the
WFDB tool remains available in `preparing.noise_mixing` for reference, but the same
number means a different amount of noise under each, so mixing them within one study
would make the figures incomparable with each other.

The levels are spaced four decibels apart and the three central ones are exactly those
of Wang et al., so that subset reproduces the published setting while the wings extend
the range into the region where the methods begin to separate.

Requires network access to PhysioNet unless `--source` points at a local copy.

Usage
-----
    python scripts/export_dataset.py --purpose model --out data/npz
    python scripts/export_dataset.py --purpose filter --out data/npz --snr 6 0 -6
    python scripts/export_dataset.py --purpose model --out data/npz --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preparing.noise_mixing import (
    DEFAULT_CONVENTION,
    mix_at_snr,
    power_ratio_snr,
    sample_noise_window,
)
from preparing.normalization import frequency_resampler, resampled_length
from preparing.splitting import group_kfold, holdout_split, patient_of
from preparing.windows import STATIC_FILTER_WIDTH, beat_windows, count_windows, sliding_windows

BEAT_SYMBOLS = set('NLRBAaJSVrFejnEf/Q?')
MITDB_RECORDS = ('100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                 '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                 '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                 '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                 '222', '223', '228', '230', '231', '232', '233', '234')

PURPOSE_DEFAULTS = {
    'model': {'width': 1024, 'overlap': 0.5, 'splits': ('development', 'test')},
    'filter': {'width': STATIC_FILTER_WIDTH, 'overlap': 0.5, 'splits': ('test',)},
}

# Jedna siatka dla obu przeznaczen, w konwencji power_ratio. Krok 4 dB, a trzy srodkowe
# poziomy to dokladnie te z Wang et al. 2023, wiec podzbior pozostaje porownywalny
# z publikacja, a skrzydla pokazuja, gdzie metody zaczynaja sie roznic.
SNR_LEVELS = (-9.0, -5.0, -1.0, 3.0, 7.0, 11.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--purpose', required=True, choices=sorted(PURPOSE_DEFAULTS),
                        help='model: training material; filter: measurement material')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=None,
                        help='local WFDB directory; PhysioNet is used when omitted')

    parser.add_argument('--records', nargs='+', default=list(MITDB_RECORDS))
    parser.add_argument('--lead', default='MLII',
                        help='signal name to export; records without it are skipped')
    parser.add_argument('--noise', default='em', choices=['em', 'ma', 'bw'])

    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--window', default='sliding', choices=['sliding', 'beat'])
    parser.add_argument('--span', type=float, default=1.0,
                        help='beat mode: mean interbeat distances to either side')
    parser.add_argument('--stride-beats', type=int, default=1)

    parser.add_argument('--snr', nargs='+', type=float, default=None)
    parser.add_argument('--convention', default=DEFAULT_CONVENTION,
                        choices=['power_ratio', 'nst'],
                        help='leave at the default; the two are not interchangeable')

    parser.add_argument('--splits', nargs='+', default=None,
                        choices=['development', 'test'])
    parser.add_argument('--test-patients', type=int, default=5)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--target-fs', type=float, default=None,
                        help='resample both signal and noise before windowing')
    parser.add_argument('--max-windows-per-record', type=int, default=None)
    parser.add_argument('--limit-records', type=int, default=None)
    parser.add_argument('--no-compress', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='report window counts and archive sizes without writing')
    return parser.parse_args(argv)


def resolve_defaults(args) -> None:
    """Fills the parameters left unset from the defaults of the chosen purpose."""
    defaults = PURPOSE_DEFAULTS[args.purpose]
    if args.width is None:
        args.width = defaults['width']
    if args.overlap is None:
        args.overlap = defaults['overlap']
    if args.splits is None:
        args.splits = list(defaults['splits'])
    if args.snr is None:
        args.snr = list(SNR_LEVELS)


def beat_annotations(annotation) -> np.ndarray:
    """Keeps beat annotations only; rhythm and quality markers are not beats."""
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample)
    return samples[np.isin(symbols, list(BEAT_SYMBOLS))]


def read_record(record_id: str, lead: str, source, wfdb) -> tuple:
    """
    Reads one record and returns the requested lead with its beat annotations.

    Selecting the lead by name rather than by position matters: the first signal of a
    MIT-BIH record is usually but not always the modified limb lead, and exporting
    whichever signal happens to come first would mix two different projections of the
    heart into one training set.
    """
    if source is None:
        record = wfdb.rdrecord(record_id, pn_dir='mitdb')
        annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
    else:
        path = str(Path(source) / record_id)
        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')

    names = [str(name).strip() for name in record.sig_name]
    if lead not in names:
        return None, None, None
    channel = names.index(lead)
    return (np.asarray(record.p_signal)[:, channel].astype(np.float64),
            beat_annotations(annotation), float(record.fs))


def read_noise(noise_id: str, source, wfdb) -> tuple:
    if source is None:
        record = wfdb.rdrecord(noise_id, pn_dir='nstdb')
    else:
        record = wfdb.rdrecord(str(Path(source) / noise_id))
    return np.asarray(record.p_signal)[:, 0].astype(np.float64), float(record.fs)


def partition_noise(noise: np.ndarray, patients) -> dict:
    """
    One contiguous stretch of noise per patient.

    Disjointness in noise then follows disjointness in patients automatically, in the
    held out split and in every cross validation fold alike, without the caller having
    to keep the two arrangements in step.
    """
    patients = sorted(set(patients))
    bounds = np.linspace(0, noise.size, len(patients) + 1).astype(int)
    return {patient: noise[bounds[index]:bounds[index + 1]]
            for index, patient in enumerate(patients)}


def window_bounds(signal_length: int, r_peaks: np.ndarray, args) -> list:
    if args.window == 'sliding':
        bounds = sliding_windows(signal_length, args.width, args.overlap)
    else:
        bounds = beat_windows(r_peaks, signal_length, args.span, args.stride_beats)
    bounds = list(bounds)
    if args.max_windows_per_record is not None:
        bounds = bounds[:args.max_windows_per_record]
    return bounds


def build_split(records, loaded, noise_parts, snr_db, args, rng) -> dict:
    """Cuts, noises and collects every window of one split at one signal to noise ratio."""
    clean_windows, noisy_windows = [], []
    record_ids, patients, starts = [], [], []
    gains, realised = [], []
    peaks_flat, peaks_offset = [], [0]

    for record_id in records:
        signal, r_peaks, _ = loaded[record_id]
        patient = patient_of(record_id)
        stretch = noise_parts[patient]

        for start, stop in window_bounds(signal.size, r_peaks, args):
            clean = signal[start:stop]
            width = stop - start
            if stretch.size < width:
                raise ValueError(
                    f'noise stretch of patient {patient} holds {stretch.size} samples, '
                    f'shorter than the window of {width}; lower --width or --overlap')

            inside = r_peaks[(r_peaks >= start) & (r_peaks < stop)] - start
            noise = sample_noise_window(stretch, width, rng)
            noisy, meta = mix_at_snr(clean, noise, inside if inside.size else None,
                                     args.target_fs or loaded[record_id][2], snr_db,
                                     convention=args.convention)

            clean_windows.append(clean.astype(np.float32))
            noisy_windows.append(noisy.astype(np.float32))
            record_ids.append(record_id)
            patients.append(patient)
            starts.append(start)
            gains.append(meta['gain'])
            realised.append(meta['snr_db_power_ratio'])
            peaks_flat.append(inside.astype(np.int32))
            peaks_offset.append(peaks_offset[-1] + inside.size)

    if not clean_windows:
        raise ValueError('no window was produced; check --width against the record length')

    return {
        'clean': np.stack(clean_windows),
        'noisy': np.stack(noisy_windows),
        'record': np.asarray(record_ids),
        'patient': np.asarray(patients),
        'start': np.asarray(starts, dtype=np.int64),
        'gain': np.asarray(gains, dtype=np.float32),
        'snr_db_realised': np.asarray(realised, dtype=np.float32),
        'r_peaks': np.concatenate(peaks_flat) if peaks_flat else np.empty(0, np.int32),
        'r_peaks_offset': np.asarray(peaks_offset, dtype=np.int64),
    }


def archive_name(purpose: str, split: str, snr_db: float) -> str:
    tag = f'{snr_db:+.0f}'.replace('+', 'p').replace('-', 'm')
    return f'{purpose}_{split}_snr{tag}.npz'


def main(argv=None) -> int:
    args = parse_args(argv)
    resolve_defaults(args)

    try:
        import wfdb
    except ImportError:
        print('wfdb is required: pip install wfdb', file=sys.stderr)
        return 1

    records = list(args.records)[:args.limit_records]
    split = holdout_split(records, n_test_patients=args.test_patients, seed=args.seed)

    print(f'przeznaczenie {args.purpose} | okno {args.width} | zakladka {args.overlap} '
          f'| tryb {args.window} | konwencja {args.convention}')
    print(f'SNR {args.snr} | szum {args.noise} | odprowadzenie {args.lead}')
    print()

    loaded, skipped = {}, []
    for record_id in records:
        signal, r_peaks, fs = read_record(record_id, args.lead, args.source, wfdb)
        if signal is None:
            skipped.append(record_id)
            continue
        if args.target_fs and args.target_fs != fs:
            scale = args.target_fs / fs
            signal = frequency_resampler(signal, fs, args.target_fs)
            r_peaks = np.round(r_peaks * scale).astype(np.int64)
            r_peaks = r_peaks[r_peaks < signal.size]
            fs = args.target_fs
        loaded[record_id] = (signal, r_peaks, fs)

    if skipped:
        print(f'pominieto (brak {args.lead}): {", ".join(skipped)}')
    if not loaded:
        print(f'zaden rekord nie zawiera odprowadzenia {args.lead}', file=sys.stderr)
        return 1

    kept = set(loaded)
    split = {name: [record for record in part if record in kept]
             for name, part in split.items()}

    noise, noise_fs = read_noise(args.noise, args.source, wfdb)
    if args.target_fs and args.target_fs != noise_fs:
        noise = frequency_resampler(noise, noise_fs, args.target_fs)
    noise_parts = partition_noise(noise, [patient_of(record) for record in loaded])

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'purpose': args.purpose, 'width': args.width, 'overlap': args.overlap,
        'window': args.window, 'convention': args.convention, 'noise': args.noise,
        'lead': args.lead, 'seed': args.seed, 'snr_db': args.snr,
        'target_fs': args.target_fs, 'skipped': skipped,
        'split': split, 'archives': [],
    }
    if 'development' in args.splits:
        manifest['folds'] = [
            {'train': train, 'val': val}
            for train, val in group_kfold(split['development'], args.folds)]

    total_bytes = 0
    for name in args.splits:
        part = split[name]
        estimate = sum(count_windows(loaded[record][0].size, args.width, args.overlap)
                       for record in part) if args.window == 'sliding' else None

        for snr_db in args.snr:
            if args.dry_run:
                if estimate is None:
                    print(f'{name:>12} {snr_db:>6.1f} dB   tryb beat, liczba okien '
                          f'zalezy od tetna')
                    continue
                size = 2 * estimate * args.width * 4
                total_bytes += size
                print(f'{name:>12} {snr_db:>6.1f} dB   {estimate:>7d} okien   '
                      f'{size / 1e6:>8.1f} MB')
                continue

            rng = np.random.default_rng(args.seed)
            data = build_split(part, loaded, noise_parts, snr_db, args, rng)
            path = args.out / archive_name(args.purpose, name, snr_db)
            saver = np.savez if args.no_compress else np.savez_compressed
            saver(path,
                  snr_db_requested=np.float32(snr_db),
                  fs=np.float64(args.target_fs or loaded[part[0]][2]),
                  width=np.int64(args.width),
                  overlap=np.float64(args.overlap),
                  window_mode=np.asarray(args.window),
                  convention=np.asarray(args.convention),
                  lead=np.asarray(args.lead),
                  noise=np.asarray(args.noise),
                  **data)

            size = path.stat().st_size
            total_bytes += size
            manifest['archives'].append(
                {'file': path.name, 'split': name, 'snr_db': snr_db,
                 'n_windows': int(data['clean'].shape[0]), 'bytes': int(size)})
            print(f'{path.name:>34}   {data["clean"].shape[0]:>7d} okien   '
                  f'{size / 1e6:>8.1f} MB   '
                  f'SNR zrealizowane {data["snr_db_realised"].mean():+.3f} dB')

    print()
    print(f'razem {total_bytes / 1e6:.1f} MB')

    if not args.dry_run:
        manifest_path = args.out / f'{args.purpose}_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'manifest: {manifest_path.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
