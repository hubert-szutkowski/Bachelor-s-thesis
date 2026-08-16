'''
Unified training and validation loop shared by every deep architecture in this project.

The loop is representation-agnostic: it receives a `SignalSpec` from `signal_selection`
and never needs to know whether the batch holds a 1-D window, a stacked cardiac cycle
tensor or a time-frequency map.

Domain policy
-------------
The loss is computed in the native domain of the model. Hard metrics (SNR, MSE, RMSE, PRD)
are computed in the time domain after `SignalSpec.decode`, because a mean squared error
measured on an STFT magnitude and one measured on a waveform are not comparable quantities
and reporting them side by side in the thesis would be indefensible.

Memory policy
-------------
    * encoded tensors live in one contiguous float32 array, not in a list of per-sample arrays
    * metric accumulators stay on the device until the epoch ends, so no per-batch device
      synchronisation is triggered by a `.item()` call
    * `channels_last` is enabled only for 4-D batches, where it actually changes the kernel
      selection
    * gradients are released with `set_to_none=True`
    * automatic mixed precision defaults to bfloat16 when the device supports it, which
      removes the need for loss scaling
'''

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train.signal_selection import SignalSpec, select_signal, build_model


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    '''
    Seeds every RNG that can influence a run and optionally forces deterministic kernels.

    Deterministic cuDNN kernels cost roughly 10-20 percent throughput on convolutional
    workloads. That price is worth paying for results quoted in a thesis.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(device: str = 'auto') -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def resolve_amp_dtype(device: torch.device, requested: str = 'auto') -> Optional[torch.dtype]:
    '''
    Returns the autocast dtype, or None when mixed precision is disabled.

    bfloat16 is preferred over float16 because its exponent range matches float32, so no
    GradScaler is needed and the loss cannot silently overflow.
    '''
    if requested == 'off':
        return None
    if device.type != 'cuda':
        return None
    if requested == 'bf16':
        return torch.bfloat16
    if requested == 'fp16':
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class ECGDenoisingDataset(Dataset):
    '''
    Pairs of encoded noisy inputs and encoded clean targets.

    Parameters
    ----------
    noisy : sequence of np.ndarray
        Raw noisy segments in the time domain.
    clean : sequence of np.ndarray
        Matching clean segments. Must be the same length as `noisy`.
    spec : SignalSpec
        Representation to apply.
    r_peaks : sequence of np.ndarray, optional
        R-peak indices per segment. Mandatory when `spec.requires_r_peaks` is True.
    precompute : bool
        Encode the whole set up front. Keep this enabled: the MIEMD representation runs an
        Empirical Mode Decomposition that takes about a second per segment and would stall
        every DataLoader worker on every epoch.

    Notes
    -----
    The clean target is always encoded with the normalization statistics of the matching
    noisy input. Encoding them independently would let the network learn an implicit
    rescaling and would inflate the reported SNR.
    '''

    def __init__(self, noisy: Sequence[np.ndarray], clean: Sequence[np.ndarray],
                 spec: SignalSpec, r_peaks: Sequence[np.ndarray] = None,
                 precompute: bool = True):
        if len(noisy) != len(clean):
            raise ValueError(f'noisy and clean differ in length: {len(noisy)} vs {len(clean)}')
        if spec.requires_r_peaks and r_peaks is None:
            raise ValueError(f"representation '{spec.domain}' requires r_peaks")
        if r_peaks is not None and len(r_peaks) != len(noisy):
            raise ValueError('r_peaks must have one entry per segment')

        self.spec = spec
        self.noisy = noisy
        self.clean = clean
        self.r_peaks = r_peaks
        self.meta: list = [None] * len(noisy)

        self._x = None
        self._y = None
        if precompute:
            self._encode_all()

    def _encode_one(self, index: int) -> tuple:
        peaks = self.r_peaks[index] if self.r_peaks is not None else None
        x, meta = self.spec.encode(self.noisy[index], r_peaks=peaks)
        y, _ = self.spec.encode(self.clean[index], r_peaks=peaks, stats=meta['stats'])
        return x, y, meta

    def _encode_all(self) -> None:
        n = len(self.noisy)
        shape = (n,) + tuple(self.spec.tensor_shape)
        self._x = np.empty(shape, dtype=np.float32)
        self._y = np.empty(shape, dtype=np.float32)

        for i in range(n):
            x, y, meta = self._encode_one(i)
            self._x[i] = x
            self._y[i] = y
            self.meta[i] = meta

    @property
    def nbytes(self) -> int:
        if self._x is None:
            return 0
        return self._x.nbytes + self._y.nbytes

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, index: int) -> dict:
        if self._x is not None:
            x, y = self._x[index], self._y[index]
        else:
            x, y, self.meta[index] = self._encode_one(index)
        return {
            'x': torch.from_numpy(np.ascontiguousarray(x)),
            'y': torch.from_numpy(np.ascontiguousarray(y)),
            'index': index,
        }


def build_criterion(name: str = 'improved_mse', **kwargs) -> nn.Module:
    '''
    Loss factory.

    'improved_mse' is the peak-weighted MSE of [8], which penalises errors in proportion to
    the target amplitude and therefore protects the QRS complex from being smoothed away.
    '''
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'l1':
        return nn.L1Loss()
    if name == 'huber':
        return nn.HuberLoss(delta=kwargs.get('delta', 1.0))
    if name == 'improved_mse':
        from filters.wavelet_transformers import ImprovedMSELoss
        return ImprovedMSELoss(peak_weight=kwargs.get('peak_weight', 2.0))
    raise KeyError(f"unknown loss '{name}'")


def time_domain_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    '''
    Hard metrics on the time-domain waveform.

    SNR  = 10 log10( sum(x^2) / sum((x - x_hat)^2) )      [dB]
    PRD  = 100 sqrt( sum((x - x_hat)^2) / sum(x^2) )      [percent]
    '''
    reference = np.asarray(reference, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()

    n = min(reference.size, estimate.size)
    reference, estimate = reference[:n], estimate[:n]

    residual = reference - estimate
    signal_power = float(np.sum(reference ** 2))
    residual_power = float(np.sum(residual ** 2))

    mse = residual_power / max(n, 1)
    return {
        'snr': 10.0 * math.log10(signal_power / residual_power) if residual_power > 0 and signal_power > 0 else float('nan'),
        'mse': mse,
        'rmse': math.sqrt(mse),
        'prd': 100.0 * math.sqrt(residual_power / signal_power) if signal_power > 0 else float('nan'),
    }


class EarlyStopping:
    '''Stops training when the monitored value has not improved for `patience` epochs.'''

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = math.inf if mode == 'min' else -math.inf
        self.counter = 0
        self.should_stop = False

    def improved(self, value: float) -> bool:
        if self.mode == 'min':
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        if math.isnan(value):
            self.counter += 1
            self.should_stop = self.counter >= self.patience
            return False
        if self.improved(value):
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    amp: str = 'auto'
    channels_last: bool = True
    early_stopping_patience: int = 15
    scheduler: str = 'cosine'
    warmup_epochs: int = 0
    log_every: int = 0
    eval_waveform_batches: int = 4
    checkpoint_dir: str = 'artifacts/runs'
    seed: int = 42
    deterministic: bool = True


class Trainer:
    '''
    Representation-agnostic training loop.

    Parameters
    ----------
    model : nn.Module
        Any architecture from the registry.
    spec : SignalSpec
        Must match the representation the datasets were encoded with.
    train_loader, val_loader : DataLoader
        Loaders over `ECGDenoisingDataset`.
    config : TrainerConfig
    criterion : nn.Module, optional
        Defaults to `spec.default_loss`.
    device : str
        'auto', 'cpu', 'cuda' or an explicit index.
    '''

    def __init__(self, model: nn.Module, spec: SignalSpec, train_loader: DataLoader,
                 val_loader: DataLoader, config: TrainerConfig = None,
                 criterion: nn.Module = None, optimizer=None, device: str = 'auto'):
        self.config = config or TrainerConfig()
        self.spec = spec
        self.device = resolve_device(device)
        self.amp_dtype = resolve_amp_dtype(self.device, self.config.amp)

        self.model = model.to(self.device)
        self.use_channels_last = self.config.channels_last and spec.is_spatial and self.device.type == 'cuda'
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or build_criterion(spec.default_loss)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.amp_dtype == torch.float16))

        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        self.history: list = []
        self.best_state = None
        self.best_epoch = -1

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _build_scheduler(self):
        if self.config.scheduler == 'none':
            return None
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.config.epochs - self.config.warmup_epochs, 1))
        if self.config.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5)
        raise KeyError(f"unknown scheduler '{self.config.scheduler}'")

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, non_blocking=True)
        if self.use_channels_last and tensor.dim() == 4:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor

    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        loss_sum = torch.zeros((), device=self.device)
        sample_count = 0
        accum = max(self.config.grad_accum_steps, 1)
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            x = self._to_device(batch['x'])
            y = self._to_device(batch['y'])

            with self._autocast():
                prediction = self.model(x)
                loss = self.criterion(prediction, y)

            self.scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.detach() * x.size(0)
            sample_count += x.size(0)

            if self.config.log_every and (step + 1) % self.config.log_every == 0:
                print(f'    epoch {epoch:3d} step {step + 1:5d}/{len(self.train_loader)} '
                      f'loss {loss.item():.6f}')

        return {'train_loss': (loss_sum / max(sample_count, 1)).item()}

    @torch.inference_mode()
    def validate(self, dataset: ECGDenoisingDataset = None) -> dict:
        '''
        Native-domain loss over the whole validation set plus time-domain metrics over the
        first `config.eval_waveform_batches` batches.

        The waveform metrics require `decode`, which runs on the CPU in NumPy, so they are
        deliberately restricted to a subset during training. Run `evaluate_waveforms` on the
        full set once, at the end.
        '''
        self.model.eval()

        loss_sum = torch.zeros((), device=self.device)
        sample_count = 0
        waveform_metrics: list = []

        for step, batch in enumerate(self.val_loader):
            x = self._to_device(batch['x'])
            y = self._to_device(batch['y'])

            with self._autocast():
                prediction = self.model(x)
                loss = self.criterion(prediction, y)

            loss_sum += loss.detach() * x.size(0)
            sample_count += x.size(0)

            if dataset is not None and step < self.config.eval_waveform_batches:
                waveform_metrics.extend(
                    self._waveform_metrics(prediction, batch['index'], dataset))

        result = {'val_loss': (loss_sum / max(sample_count, 1)).item()}
        if waveform_metrics:
            for key in ('snr', 'mse', 'rmse', 'prd'):
                values = [m[key] for m in waveform_metrics if not math.isnan(m[key])]
                result[f'val_{key}'] = float(np.mean(values)) if values else float('nan')
        return result

    def _waveform_metrics(self, prediction: torch.Tensor, indices: torch.Tensor,
                          dataset: ECGDenoisingDataset) -> list:
        prediction = prediction.detach().float().cpu().numpy()
        metrics = []
        for row, index in enumerate(indices.tolist()):
            meta = dataset.meta[index]
            if meta is None:
                continue
            estimate = self.spec.decode(prediction[row], meta)
            reference = np.asarray(dataset.clean[index], dtype=np.float64).ravel()
            metrics.append(time_domain_metrics(reference, estimate))
        return metrics

    def fit(self, train_dataset: ECGDenoisingDataset = None,
            val_dataset: ECGDenoisingDataset = None) -> list:
        '''Runs the full schedule and returns the per-epoch history.'''
        set_seed(self.config.seed, self.config.deterministic)

        print(f'model      {type(self.model).__name__}')
        print(f'device     {self.device} amp={self.amp_dtype} channels_last={self.use_channels_last}')
        print(f'tensor     {self.spec.tensor_shape} domain={self.spec.domain}')
        print(f'parameters {sum(p.numel() for p in self.model.parameters()) / 1e6:.3f} M')
        if train_dataset is not None:
            print(f'cache      {train_dataset.nbytes / 1024 ** 2:.1f} MiB train')

        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()

            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(val_dataset)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_stats['val_loss'])
            elif self.scheduler is not None:
                self.scheduler.step()

            record = {'epoch': epoch,
                      'lr': self.optimizer.param_groups[0]['lr'],
                      'seconds': time.perf_counter() - started,
                      **train_stats, **val_stats}
            self.history.append(record)
            self._log_epoch(record)

            if self.early_stopping.step(val_stats['val_loss']):
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint('best.pt', epoch, record)

            if self.early_stopping.should_stop:
                print(f'early stopping at epoch {epoch}, best epoch {self.best_epoch}')
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.save_history()
        return self.history

    def _log_epoch(self, record: dict) -> None:
        parts = [f"epoch {record['epoch']:3d}",
                 f"train {record['train_loss']:.6f}",
                 f"val {record['val_loss']:.6f}"]
        if 'val_snr' in record:
            parts.append(f"snr {record['val_snr']:6.2f} dB")
            parts.append(f"prd {record['val_prd']:6.2f} %")
        parts.append(f"lr {record['lr']:.2e}")
        parts.append(f"{record['seconds']:.1f} s")
        print('  ' + '  '.join(parts))

    def save_checkpoint(self, filename: str, epoch: int, record: dict) -> str:
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_name': self.spec.model_name,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'record': record,
        }, path)
        return path

    def save_history(self) -> str:
        path = os.path.join(self.config.checkpoint_dir, 'history.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.history, handle, indent=2)
        return path

    @torch.inference_mode()
    def evaluate_waveforms(self, loader: DataLoader, dataset: ECGDenoisingDataset) -> dict:
        '''Time-domain metrics over an entire loader, for the final report.'''
        self.model.eval()
        metrics: list = []
        for batch in loader:
            x = self._to_device(batch['x'])
            with self._autocast():
                prediction = self.model(x)
            metrics.extend(self._waveform_metrics(prediction, batch['index'], dataset))

        summary = {}
        for key in ('snr', 'mse', 'rmse', 'prd'):
            values = [m[key] for m in metrics if not math.isnan(m[key])]
            summary[key] = float(np.mean(values)) if values else float('nan')
            summary[f'{key}_std'] = float(np.std(values)) if values else float('nan')
        summary['n'] = len(metrics)
        return summary


def build_dataloaders(train_dataset: ECGDenoisingDataset, val_dataset: ECGDenoisingDataset,
                      config: TrainerConfig, num_workers: int = 0) -> tuple:
    '''
    Loaders with pinned memory when a GPU is present.

    `num_workers=0` is the right default here because the datasets are pre-encoded and held
    in RAM; spawning workers would only add inter-process copies.
    '''
    pin = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True,
                              generator=generator,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, drop_last=False,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader
