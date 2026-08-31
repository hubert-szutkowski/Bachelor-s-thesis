'''
Command line entry point for the unified training loop.

Examples
--------
    python src/train/cli.py --list

    python src/train/cli.py --model wavelet_cnn --data data/nstdb_synthetic.npz \\
        --epochs 120 --batch-size 64

    python src/train/cli.py --model sced_net --data data/nstdb_scc.npz \\
        --batch-size 8 --grad-accum-steps 4

Dataset format
--------------
A single .npz as written by `scripts/export_dataset.py`, holding:

    noisy           (n_segments, segment_length) float
    clean           (n_segments, segment_length) float
    fs              scalar, sampling frequency in Hz, required
    patient         (n_segments,) patient label, optional but strongly preferred
    r_peaks         concatenated beat indices, relative to the start of their segment
    r_peaks_offset  (n_segments + 1,) where each segment begins in `r_peaks`

The training and validation parts are separated by patient whenever `patient` is present,
so no recording contributes to both. Without those labels the split falls back to a
contiguous cut on segment index, which is better than a shuffle but still lands inside
whichever patient straddles it; the run says so on standard output when that happens.

`fs` is required rather than defaulted. A wrong sampling frequency shifts every quantity
expressed in hertz, and a default that happens to be wrong does so without raising.
'''

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset_io import load_dataset, read_r_peaks, split_by_patient, split_indices
from train.signal_selection import available_models, describe, select_signal, build_model
from train.training import (
    ECGDenoisingDataset,
    Trainer,
    TrainerConfig,
    build_criterion,
    build_dataloaders,
    set_seed,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Unified ECG denoising trainer')

    parser.add_argument('--list', action='store_true', help='print the model registry and exit')
    parser.add_argument('--model', choices=available_models(), help='architecture to train')
    parser.add_argument('--data', help='path to the .npz dataset')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--limit', type=int, default=0, help='use only the first N segments')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--loss', default=None, help='mse, l1, huber or improved_mse')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--early-stopping-patience', type=int, default=15)

    parser.add_argument('--amp', default='auto', choices=['auto', 'bf16', 'fp16', 'off'])
    parser.add_argument('--no-channels-last', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--non-deterministic', action='store_true')
    parser.add_argument('--checkpoint-dir', default='artifacts/runs')
    parser.add_argument('--log-every', type=int, default=0)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        print('registered models:')
        for name in available_models():
            print('  ' + describe(name))
        return 0

    if not args.model or not args.data:
        print('--model and --data are required (or use --list)', file=sys.stderr)
        return 2

    set_seed(args.seed, deterministic=not args.non_deterministic)

    payload = load_dataset(args.data, limit=args.limit)
    spec = select_signal(args.model, fs=payload['fs'])

    if spec.requires_r_peaks and payload['r_peaks'] is None:
        print(f"model '{args.model}' needs r_peaks in the dataset", file=sys.stderr)
        return 2
    if spec.segment_length is not None and payload['noisy'].shape[1] != spec.segment_length:
        print(f"segment length mismatch: dataset has {payload['noisy'].shape[1]}, "
              f"'{args.model}' expects {spec.segment_length}", file=sys.stderr)
        return 2

    if payload['patient'] is not None:
        train_idx, val_idx = split_by_patient(payload['patient'], args.val_fraction)
        held = sorted(set(payload['patient'][val_idx].tolist()))
        print(f'podzial po pacjentach | walidacja: {", ".join(held)} '
              f'({len(val_idx)} z {len(payload["noisy"])} okien)')
    else:
        train_idx, val_idx = split_indices(len(payload['noisy']), args.val_fraction)
        print('UWAGA: archiwum nie zawiera etykiet pacjenta, uzyto ciaglego podzialu '
              'po indeksie segmentu; granica moze wypasc wewnatrz jednego nagrania')

    take = lambda arr, idx: None if arr is None else [arr[i] for i in idx]

    train_dataset = ECGDenoisingDataset(
        take(payload['noisy'], train_idx), take(payload['clean'], train_idx),
        spec, r_peaks=take(payload['r_peaks'], train_idx))
    val_dataset = ECGDenoisingDataset(
        take(payload['noisy'], val_idx), take(payload['clean'], val_idx),
        spec, r_peaks=take(payload['r_peaks'], val_idx))

    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        amp=args.amp,
        channels_last=not args.no_channels_last,
        early_stopping_patience=args.early_stopping_patience,
        scheduler=args.scheduler,
        log_every=args.log_every,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.model),
        seed=args.seed,
        deterministic=not args.non_deterministic,
    )

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config,
                                                 num_workers=args.num_workers)

    trainer = Trainer(
        model=build_model(args.model),
        spec=spec,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=build_criterion(args.loss or spec.default_loss),
        device=args.device,
    )

    print(f'segments   {len(train_dataset)} train / {len(val_dataset)} val, fs={payload["fs"]} Hz')
    trainer.fit(train_dataset, val_dataset)

    summary = trainer.evaluate_waveforms(val_loader, val_dataset)
    print('\nfinal time-domain metrics on the validation set:')
    print(f'  SNR  {summary["snr"]:8.3f} +/- {summary["snr_std"]:.3f} dB')
    print(f'  MSE  {summary["mse"]:8.6f} +/- {summary["mse_std"]:.6f}')
    print(f'  RMSE {summary["rmse"]:8.6f} +/- {summary["rmse_std"]:.6f}')
    print(f'  PRD  {summary["prd"]:8.3f} +/- {summary["prd_std"]:.3f} %')
    print(f'  n    {summary["n"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
