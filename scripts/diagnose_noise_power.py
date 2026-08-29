"""
Locates the source of the constant factor between this implementation and `nst`.

Validation against the premixed NSTDB records showed the recovered gain to be larger
than the computed one by the same factor, close to 4.9, at every signal-to-noise ratio.
A factor that does not vary with the ratio cannot originate in the gain formula or in
the mixing protocol, so it has to sit in one of the two power estimates.

The diagnosis rests on an identity that holds whatever the estimators are. Writing the
injected noise as a times the unscaled noise, the definition of the ratio gives

    SNR = 10 log10( S / (a^2 N) )    hence    a^2 N = S / 10^(SNR/10)

The left side is the power of the noise actually present in a published record, which is
measurable. The right side contains only the signal power. Comparing the two therefore
tests the signal power estimate on its own, with the noise power estimate cancelled out.

If the measured injected power matches S / 10^(SNR/10), the signal side agrees with
`nst` and the discrepancy lives entirely in N. If it does not, the implied signal power
is reported so the size of the disagreement is visible directly.

Usage
-----
    python scripts/diagnose_noise_power.py
    python scripts/diagnose_noise_power.py --record 119 --channel 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preparing.noise_mixing import noise_power, nstdb_protocol_mask, signal_power

BEAT_SYMBOLS = set('NLRBAaJSVrFejnEf/Q?')
SNR_LEVELS = [(24.0, 'e24'), (18.0, 'e18'), (12.0, 'e12'),
              (6.0, 'e06'), (0.0, 'e00'), (-6.0, 'e_6')]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record', default='118', choices=['118', '119'])
    parser.add_argument('--channel', type=int, default=0)
    return parser.parse_args(argv)


def beat_annotations(annotation) -> np.ndarray:
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample)
    return samples[np.isin(symbols, list(BEAT_SYMBOLS))]


def highpass(signal: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)


def measure_injected_power(published: np.ndarray, clean: np.ndarray, fs: float) -> float:
    """Power of the noise present in a published record, with the baseline shift removed."""
    n = min(published.size, clean.size)
    mask = nstdb_protocol_mask(n, fs)
    difference = published[:n] - clean[:n]

    quiet = ~mask
    quiet[:int(fs)] = False
    injected = difference - np.median(difference[quiet])
    return float(np.mean(injected[mask] ** 2))


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
    amplitude = np.sqrt(8.0 * power_clean)

    print(f'rekord {args.record}, kanał {args.channel}, fs = {fs} Hz')
    print(f'  amplituda QRS  A_pp = {amplitude:.4f} mV')
    print(f'  moc sygnalu    S    = {power_clean:.6g}')
    print()
    print('naglowek rekordu szumu em:')
    print(f'  wzmocnienie:        {noise_record.adc_gain}')
    print(f'  poziom odniesienia: {noise_record.baseline}')
    print(f'  rozdzielczosc:      {noise_record.adc_res} bit')
    print(f'  jednostki:          {noise_record.units}')
    print()

    print('=== TEST 1: czy moc wstrzyknietego szumu zgadza sie z definicja SNR ===')
    print('    tozsamosc a^2 N = S / 10^(SNR/10) nie zawiera N, wiec sprawdza samo S')
    print()
    print(f'{"SNR":>6} {"zmierzona":>14} {"S/10^(SNR/10)":>16} {"stosunek":>10} {"S wynikajace":>14}')
    print('-' * 64)

    ratios = []
    for snr_db, suffix in SNR_LEVELS:
        try:
            reference = wfdb.rdrecord(f'{args.record}{suffix}', pn_dir='nstdb')
        except Exception as error:
            print(f'{snr_db:>6.0f}  {type(error).__name__}')
            continue
        published = np.asarray(reference.p_signal)[:, args.channel]
        measured = measure_injected_power(published, clean, fs)
        expected = power_clean / 10.0 ** (snr_db / 10.0)
        implied_s = measured * 10.0 ** (snr_db / 10.0)
        ratios.append(measured / expected)
        print(f'{snr_db:>6.0f} {measured:>14.6g} {expected:>16.6g} '
              f'{measured / expected:>10.3f} {implied_s:>14.6g}')

    if not ratios:
        print('nie udalo sie wczytac zadnego rekordu referencyjnego', file=sys.stderr)
        return 1

    mean_ratio = float(np.mean(ratios))
    print()
    print(f'sredni stosunek: {mean_ratio:.3f}')
    if abs(mean_ratio - 1.0) < 0.10:
        print('  -> S zgodne z nst. Rozbieznosc siedzi wylacznie w estymacie N.')
    else:
        print(f'  -> S rozni sie od nst {mean_ratio:.2f}-krotnie, czyli amplituda QRS '
              f'{np.sqrt(mean_ratio):.2f}-krotnie.')
        print(f'     nst przyjmuje A_pp = {amplitude * np.sqrt(mean_ratio):.3f} mV '
              f'zamiast {amplitude:.3f} mV.')
    print()

    print('=== TEST 2: kandydaci na definicje mocy szumu ===')
    current = noise_power(noise, fs)
    first300 = noise[:int(300 * fs)]
    candidates = {
        'obecna: okna 1 s, srednia odjeta, przyciete 5%': current,
        'wariancja pierwszych 300 s': float(np.var(first300)),
        'wariancja calego rekordu': float(np.var(noise)),
        'po filtrze gornoprzepustowym 0.5 Hz': float(np.var(highpass(first300, fs, 0.5))),
        'po filtrze gornoprzepustowym 1 Hz': float(np.var(highpass(first300, fs, 1.0))),
        'po filtrze gornoprzepustowym 5 Hz': float(np.var(highpass(first300, fs, 5.0))),
        'srednia wartosc bezwzgledna do kwadratu': float(
            np.mean(np.abs(first300 - first300.mean())) ** 2),
    }
    print(f'{"definicja":>48} {"N":>13} {"wzgl. obecnej":>15}')
    print('-' * 78)
    for name, value in candidates.items():
        print(f'{name:>48} {value:>13.6g} {value / current:>15.4f}')

    print()
    print('=== TEST 3: rozklad amplitudy szumu w czasie ===')
    print('    jesli pierwsze 5 minut rozni sie od reszty, wybor okna pomiarowego ma znaczenie')
    chunk = int(300 * fs)
    print(f'{"odcinek":>16} {"wariancja":>14}')
    print('-' * 32)
    for index in range(0, noise.size - chunk + 1, chunk):
        part = noise[index:index + chunk]
        print(f'{index / fs / 60:>6.1f}-{(index + chunk) / fs / 60:>5.1f} min '
              f'{float(np.var(part)):>14.6g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
