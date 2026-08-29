"""
Validates the noise mixing implementation against the premixed NSTDB records.

Regenerates `118e06` and its siblings from MIT-BIH record 118 and the `em` noise
record, then compares the result with the official file published by PhysioNet. A close
match proves that the signal and noise power estimators and the gain calculation agree
with the `nst` tool of the WFDB package, which is what makes signal-to-noise ratios
quoted in this work comparable with the literature built on that database.

A constant offset has to be removed first. The premixed records store samples with the
baseline already subtracted while their headers still declare `200(1024)`, so WFDB
subtracts it a second time and `p_signal` comes out shifted by 1024/200 = 5.12 mV
relative to the original record. The shift is estimated on the noise-free stretches,
where the two recordings must agree exactly, and removed before the noise is measured.
That estimate doubles as an alignment check: a large spread there means the records are
not sample aligned and nothing further can be compared.

An exact match is not expected even after that. The published records are quantised,
the noise record is repeated from an unknown phase, and `nst` recomputes DC offsets at
every gain change. What matters is that the estimated gain matches the one implied by
the published file.

Requires network access to PhysioNet on the first run.

Usage
-----
    python scripts/validate_noise_mixing.py
    python scripts/validate_noise_mixing.py --record 119 --snr 6 0 -6
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preparing.noise_mixing import (
    mix_nstdb_protocol,
    noise_power,
    nstdb_protocol_mask,
    signal_power,
)

BEAT_SYMBOLS = set('NLRBAaJSVrFejnEf/Q?')
SNR_SUFFIX = {24.0: 'e24', 18.0: 'e18', 12.0: 'e12',
              6.0: 'e06', 0.0: 'e00', -6.0: 'e_6'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record', default='118', choices=['118', '119'])
    parser.add_argument('--snr', nargs='+', type=float,
                        default=[24.0, 18.0, 12.0, 6.0, 0.0, -6.0])
    parser.add_argument('--channel', type=int, default=0)
    return parser.parse_args(argv)


def beat_annotations(annotation) -> np.ndarray:
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

    clean_record = wfdb.rdrecord(args.record, pn_dir='mitdb')
    annotation = wfdb.rdann(args.record, 'atr', pn_dir='mitdb')
    noise_record = wfdb.rdrecord('em', pn_dir='nstdb')

    fs = clean_record.fs
    clean = np.asarray(clean_record.p_signal)[:, args.channel]
    noise = np.asarray(noise_record.p_signal)[:, args.channel]
    r_peaks = beat_annotations(annotation)

    power_clean = signal_power(clean, r_peaks, fs)
    power_noise = noise_power(noise, fs)

    print(f'rekord {args.record}, kanał {args.channel}, fs = {fs} Hz')
    print(f'moc sygnału  S = {power_clean:.6g}   (amplituda QRS do kwadratu przez 8)')
    print(f'moc szumu    N = {power_noise:.6g}   (przycięta wartość skuteczna do kwadratu)')
    print()

    header = f'{"SNR":>6} {"rekord":>10} {"wzm. wlasne":>13} {"wzm. z pliku":>14} ' \
             f'{"blad wzm.":>11} {"offset":>9} {"rozrzut":>10}'
    print(header)
    print('-' * len(header))

    worst = 0.0
    for snr_db in args.snr:
        suffix = SNR_SUFFIX.get(float(snr_db))
        if suffix is None:
            print(f'{snr_db:>6.0f}  brak rekordu referencyjnego dla tego SNR')
            continue

        reference_id = f'{args.record}{suffix}'
        try:
            reference = wfdb.rdrecord(reference_id, pn_dir='nstdb')
        except Exception as error:
            print(f'{snr_db:>6.0f} {reference_id:>10}  {type(error).__name__}')
            continue

        published = np.asarray(reference.p_signal)[:, args.channel]
        n = min(published.size, clean.size)
        generated, meta = mix_nstdb_protocol(clean[:n], noise, r_peaks, fs, snr_db)

        mask = nstdb_protocol_mask(n, fs)
        difference = published[:n] - clean[:n]

        # na odcinkach bez szumu oba przebiegi musza byc identyczne,
        # wiec to co tam zostaje jest stalym przesunieciem poziomu odniesienia
        quiet = ~mask
        quiet[:int(1.0 * fs)] = False
        offset = float(np.median(difference[quiet]))
        offset_spread = float(np.std(difference[quiet]))

        injected = difference - offset
        published_gain = np.sqrt(np.mean(injected[mask] ** 2) / power_noise)
        gain_error = abs(meta['gain'] - published_gain) / published_gain
        worst = max(worst, gain_error)

        print(f'{snr_db:>6.0f} {reference_id:>10} {meta["gain"]:>13.5f} '
              f'{published_gain:>14.5f} {100 * gain_error:>10.2f}% '
              f'{offset:>9.3f} {offset_spread:>10.4f}')

    print()
    print('wzm. wlasne  - wzmocnienie wyliczone przez ten kod')
    print('wzm. z pliku - wzmocnienie odtworzone z roznicy, po usunieciu przesuniecia')
    print('offset       - stale przesuniecie poziomu odniesienia [mV], oczekiwane -5.12')
    print('rozrzut      - odchylenie tego przesuniecia na odcinkach bez szumu;')
    print('               wartosc bliska zeru potwierdza, ze przebiegi sa wyrownane')
    print()
    if worst < 0.05:
        print(f'WYNIK: kalibracja zgodna z nst, najwiekszy blad wzmocnienia {100 * worst:.2f}%')
        return 0
    print(f'WYNIK: rozbieznosc wzmocnienia do {100 * worst:.1f}%. Sprawdz estymatory '
          f'mocy sygnalu i szumu przed generowaniem zbioru.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
