"""
Validates the QRS detector against MIT-BIH reference annotations.

Run once before using the detector for the downstream-task metric. The record set is
deliberately biased towards difficult recordings, so an overall sensitivity near 98
percent is expected; a markedly lower figure points at a wrong channel, wrong units or
a sampling frequency mismatch rather than at the detector.

Requires network access to PhysioNet on the first run.

Usage
-----
    python scripts/validate_qrs_detector.py
    python scripts/validate_qrs_detector.py --records 100 101 103 --channel 0
    python scripts/validate_qrs_detector.py --records 105 108 203 --no-clean
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preparing.qrs_detection import detection_metrics, detect_qrs

EASY_RECORDS = ['100', '101', '103', '106', '112']
NOISY_RECORDS = ['105', '108', '203', '228']
BEAT_SYMBOLS = set('NLRBAaJSVrFejnEf/Q?')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--records', nargs='+', default=EASY_RECORDS + NOISY_RECORDS,
                        help='MIT-BIH record identifiers')
    parser.add_argument('--channel', type=int, default=0,
                        help='signal channel index, 0 is usually MLII')
    parser.add_argument('--no-clean', action='store_true',
                        help='skip the detector internal band-pass, for sensitivity analysis')
    parser.add_argument('--tolerance-ms', type=float, default=150.0)
    parser.add_argument('--method', default='neurokit',
                        help='detector name understood by neurokit2.ecg_peaks')
    parser.add_argument('--compare-methods', action='store_true',
                        help='run several detectors on the same records and rank them')
    return parser.parse_args(argv)


CANDIDATE_METHODS = ['pantompkins1985', 'neurokit', 'hamilton2002',
                     'elgendi2010', 'engzeemod2012', 'kalidas2017']


def load_records(record_ids: list, channel: int) -> list:
    """Reads the records once so every detector is scored on identical data."""
    import wfdb

    loaded = []
    for record_id in record_ids:
        try:
            record = wfdb.rdrecord(record_id, pn_dir='mitdb')
            annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        except Exception as error:
            print(f'{record_id}: {type(error).__name__}: {str(error)[:60]}', file=sys.stderr)
            continue
        loaded.append((record_id, np.asarray(record.p_signal)[:, channel],
                       record.fs, beat_annotations(annotation)))
    return loaded


def compare_methods(args) -> int:
    """
    Ranks detectors on the same records.

    Note on `kalidas2017`: it is built on a stationary wavelet transform, so choosing it
    reintroduces wavelet processing inside the measuring instrument. Two of the methods
    under comparison in this project are themselves wavelet based, which makes the
    comparison partly circular. Prefer a detector whose internal processing does not
    overlap with the methods being evaluated, unless the accuracy gain is large enough
    to justify documenting the caveat.
    """
    loaded = load_records(args.records, args.channel)
    if not loaded:
        print('no record could be read', file=sys.stderr)
        return 1

    header = f'{"metoda":>16} {"TP":>7} {"FP":>6} {"FN":>6} ' \
             f'{"Se [%]":>8} {"P+ [%]":>8} {"F1 [%]":>8} {"DER [%]":>8}'
    print(header)
    print('-' * len(header))

    ranking = []
    for method in CANDIDATE_METHODS:
        tp = fp = fn = 0
        failed = False
        for _, signal, fs, reference in loaded:
            try:
                detected = detect_qrs(signal, fs, method=method, clean=not args.no_clean)
            except Exception as error:
                print(f'{method:>16}  {type(error).__name__}: {str(error)[:50]}')
                failed = True
                break
            result = detection_metrics(detected, reference, fs,
                                       tolerance_ms=args.tolerance_ms)
            tp += result['tp']; fp += result['fp']; fn += result['fn']
        if failed:
            continue

        sensitivity = tp / (tp + fn) if tp + fn else float('nan')
        predictivity = tp / (tp + fp) if tp + fp else float('nan')
        f1 = 2 * sensitivity * predictivity / (sensitivity + predictivity)
        der = (fp + fn) / (tp + fn)
        ranking.append((f1, method))
        print(f'{method:>16} {tp:>7} {fp:>6} {fn:>6} {100 * sensitivity:>8.2f} '
              f'{100 * predictivity:>8.2f} {100 * f1:>8.2f} {100 * der:>8.2f}')

    if ranking:
        ranking.sort(reverse=True)
        print(f'\nnajwyzsze F1: {ranking[0][1]} ({100 * ranking[0][0]:.2f} %)')
        print('uwaga: kalidas2017 opiera sie na transformacie falkowej, co wprowadza '
              'falki do przyrzadu pomiarowego')
    return 0


def beat_annotations(annotation) -> np.ndarray:
    """Keeps only beat annotations; rhythm and quality markers are not beats."""
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample)
    return samples[np.isin(symbols, list(BEAT_SYMBOLS))]


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import wfdb
    except ImportError:
        print('wfdb is required: pip install wfdb', file=sys.stderr)
        return 1

    if args.compare_methods:
        return compare_methods(args)

    header = f'{"record":>8} {"beats":>7} {"TP":>7} {"FP":>5} {"FN":>5} ' \
             f'{"Se [%]":>8} {"P+ [%]":>8} {"F1 [%]":>8} {"DER [%]":>8} {"offset":>10}'
    print(header)
    print('-' * len(header))

    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'n_reference': 0}
    failures = []

    for record_id in args.records:
        try:
            record = wfdb.rdrecord(record_id, pn_dir='mitdb')
            annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        except Exception as error:
            print(f'{record_id:>8}  {type(error).__name__}: {str(error)[:60]}')
            failures.append(record_id)
            continue

        signal = np.asarray(record.p_signal)[:, args.channel]
        reference = beat_annotations(annotation)

        detected = detect_qrs(signal, record.fs, method=args.method,
                              clean=not args.no_clean)
        result = detection_metrics(detected, reference, record.fs,
                                   tolerance_ms=args.tolerance_ms)

        for key in ('tp', 'fp', 'fn', 'n_reference'):
            totals[key] += result[key]

        print(f'{record_id:>8} {result["n_reference"]:>7} {result["tp"]:>7} '
              f'{result["fp"]:>5} {result["fn"]:>5} '
              f'{100 * result["sensitivity"]:>8.2f} '
              f'{100 * result["positive_predictivity"]:>8.2f} '
              f'{100 * result["f1"]:>8.2f} {100 * result["der"]:>8.2f} '
              f'{result["offset_mean_ms"]:>7.1f} ms')

    if totals['n_reference'] == 0:
        print('\nno record could be read', file=sys.stderr)
        return 1

    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    sensitivity = tp / (tp + fn)
    predictivity = tp / (tp + fp)
    f1 = 2 * sensitivity * predictivity / (sensitivity + predictivity)
    der = (fp + fn) / (tp + fn)

    print('-' * len(header))
    print(f'{"TOTAL":>8} {totals["n_reference"]:>7} {tp:>7} {fp:>5} {fn:>5} '
          f'{100 * sensitivity:>8.2f} {100 * predictivity:>8.2f} '
          f'{100 * f1:>8.2f} {100 * der:>8.2f}')

    print(f'\ndetektor: {args.method}')
    print(f'band-pass wewnętrzny detektora: {"wyłączony" if args.no_clean else "włączony"}')
    print(f'tolerancja dopasowania: {args.tolerance_ms:.0f} ms')
    if failures:
        print(f'nie udało się wczytać: {", ".join(failures)}')

    if sensitivity < 0.99:
        print('\nOSTRZEŻENIE: czułość poniżej 99 procent na bazie referencyjnej. '
              'Sprawdź kanał, jednostki i częstotliwość próbkowania przed użyciem '
              'detektora do metryki F1.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
