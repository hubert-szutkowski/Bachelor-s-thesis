"""
One calling convention for every filtering method in this project.

Seventeen methods belong to three families whose signatures have nothing in common. A
static filter is a pure function of the waveform. An adaptive filter additionally needs a
reference derived from the accelerometers. A network needs trained weights, a
representation and a device. Written out directly, every evaluation loop and every figure
would branch on the family, and with seventeen methods against six signal to noise levels
in two environments that is where an error hides which nothing reports.

The registry replaces the branching with a declaration. Each method states what it needs;
`apply_filter` checks that the context supplies it and refuses clearly when it does not,
rather than failing several frames deep inside numpy with a message about shapes.

Adaptive methods declare `requires_reference`. On the MIT-BIH database there is no
accelerometer, so they are simply not applicable there and the registry says so instead of
inventing a substitute: a reference derived from the noise that was added would measure
the best a perfect sensor could do, which is a different question from the one this work
asks. The synthetic environment therefore compares seven static methods against five
networks, and only the recordings made with the wearable platform compare all seventeen.

Parameters come from `configs/filters.yaml`, version controlled so that `git log` on one
file is the history of how the methods were tuned. Values resolved for a call are carried
in the result rather than left implicit, since a number in a table nobody can reproduce is
not a result. Tune on development material only; a parameter chosen against the held out
patients turns them into training data.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Sequence

import numpy as np

FAMILIES = ('static', 'adaptive', 'deep')
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'filters.yaml'


@dataclass(frozen=True)
class FilterContext:
    """
    Everything a method may need beyond the waveform itself.

    Frozen and typed rather than a dictionary: a misspelled key in a dictionary passes
    silently and shows up as a method that performs worse than it should, which is
    indistinguishable from a method that is worse.
    """

    fs: float
    reference: Optional[np.ndarray] = None
    reference_names: Optional[Sequence[str]] = None
    r_peaks: Optional[np.ndarray] = None
    powerline_hz: float = 50.0
    checkpoint: Optional[Path] = None
    record: Optional[str] = None
    patient: Optional[str] = None

    def __post_init__(self):
        if self.fs <= 0:
            raise ValueError(f'fs must be positive, got {self.fs}')
        if self.reference is not None:
            reference = np.atleast_2d(np.asarray(self.reference, dtype=np.float64))
            object.__setattr__(self, 'reference', reference)


@dataclass(frozen=True)
class FilterResult:
    """
    A filtered waveform together with what it took to produce it.

    `params` holds the values actually used, not the ones requested, and `code_version`
    the commit they were produced at. Between them a row of a results table can be traced
    back to the state of the repository that made it.
    """

    signal: np.ndarray
    method: str
    family: str
    params: dict
    elapsed_s: float
    code_version: str
    n_reference_channels: int = 0


@dataclass(frozen=True)
class FilterSpec:
    """Declaration of one method: what it is, what it needs and how it is called."""

    name: str
    family: str
    fn: Callable
    defaults: dict = field(default_factory=dict)
    requires_reference: bool = False
    requires_r_peaks: bool = False
    requires_checkpoint: bool = False
    description: str = ''

    def missing(self, context: 'FilterContext') -> list:
        """Names of the pieces of context this method needs and did not receive."""
        absent = []
        if self.requires_reference and context.reference is None:
            absent.append('reference')
        if self.requires_r_peaks and context.r_peaks is None:
            absent.append('r_peaks')
        if self.requires_checkpoint and context.checkpoint is None:
            absent.append('checkpoint')
        return absent


_REGISTRY: dict = {}
_CONFIG: dict = {}
_CODE_VERSION: Optional[str] = None


def register(name: str, family: str, fn: Callable, defaults: Optional[dict] = None,
             requires_reference: bool = False, requires_r_peaks: bool = False,
             requires_checkpoint: bool = False, description: str = '') -> FilterSpec:
    """Adds one method to the registry, refusing to overwrite an existing name."""
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; available: {FAMILIES}')
    if name in _REGISTRY:
        raise ValueError(f'{name!r} is already registered')

    spec = FilterSpec(name=name, family=family, fn=fn, defaults=dict(defaults or {}),
                      requires_reference=requires_reference,
                      requires_r_peaks=requires_r_peaks,
                      requires_checkpoint=requires_checkpoint,
                      description=description)
    _REGISTRY[name] = spec
    return spec


def unregister(name: str) -> None:
    """Removes a method. Exists for tests; production code registers once at import."""
    _REGISTRY.pop(name, None)


def available_filters(family: Optional[str] = None) -> list:
    """Registered method names, optionally of one family, in a stable order."""
    if family is not None and family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; available: {FAMILIES}')
    return sorted(name for name, spec in _REGISTRY.items()
                  if family is None or spec.family == family)


def get_spec(name: str) -> FilterSpec:
    if name not in _REGISTRY:
        raise KeyError(f'unknown method {name!r}; available: {available_filters()}')
    return _REGISTRY[name]


def applicable_filters(context: FilterContext, family: Optional[str] = None) -> dict:
    """
    Which methods the given context can run, and why the others cannot.

    Answers in one call what the synthetic environment looks like: with no accelerometer
    the adaptive family is unavailable, and the reason is recorded rather than the methods
    silently vanishing from a table.
    """
    runnable, blocked = [], {}
    for name in available_filters(family):
        absent = get_spec(name).missing(context)
        if absent:
            blocked[name] = absent
        else:
            runnable.append(name)
    return {'runnable': runnable, 'blocked': blocked}


# --- configuration -------------------------------------------------------

def load_config(path=None) -> dict:
    """
    Reads the parameter file and keeps it for subsequent calls.

    Missing file is not an error: every method carries defaults, and the file exists to
    override them and to record the override in version control.
    """
    global _CONFIG

    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.exists():
        _CONFIG = {}
        return _CONFIG

    import yaml

    with path.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}

    flattened: dict = {}
    for family, methods in loaded.items():
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r} in {path}; available: {FAMILIES}')
        for method, params in (methods or {}).items():
            flattened[method] = dict(params or {})

    _CONFIG = flattened
    return _CONFIG


def configured_params(name: str) -> dict:
    """Parameters of one method as they stand: registered defaults under the file."""
    spec = get_spec(name)
    return {**spec.defaults, **_CONFIG.get(name, {})}


def reset_config() -> None:
    global _CONFIG
    _CONFIG = {}


# --- provenance ----------------------------------------------------------

def code_version() -> str:
    """
    Short commit hash, marked when the working tree carries uncommitted changes.

    A result produced from a modified tree cannot be reproduced from the hash alone, and
    saying so in the record is cheaper than discovering it later.
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return _CODE_VERSION

    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                                capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            _CODE_VERSION = 'unknown'
            return _CODE_VERSION
        version = commit.stdout.strip()

        status = subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                                capture_output=True, text=True, timeout=5)
        if status.returncode == 0 and status.stdout.strip():
            version += '-dirty'
        _CODE_VERSION = version
    except (OSError, subprocess.SubprocessError):
        _CODE_VERSION = 'unknown'
    return _CODE_VERSION


# --- the one entry point -------------------------------------------------

def apply_filter(name: str, signal: np.ndarray,
                 context: Optional[FilterContext] = None, **overrides) -> FilterResult:
    """
    Runs one registered method over one waveform.

    Parameters resolve in three layers, each overriding the one before: the defaults
    declared at registration, the configuration file, and the arguments of this call.

    The input is copied before it is handed on, so a method that writes into its argument
    cannot reach the caller's array. The output is checked for length and finiteness: a
    waveform holding a not-a-number would otherwise travel into the metrics and turn a
    whole aggregate into one, which reads as a broken pipeline rather than as a broken
    filter.
    """
    spec = get_spec(name)

    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError('the signal is empty')
    if not np.all(np.isfinite(signal)):
        raise ValueError(f'{name}: the input holds non-finite samples')

    if context is None:
        raise ValueError(f'{name} needs a FilterContext, at least for the sampling frequency')

    absent = spec.missing(context)
    if absent:
        raise ValueError(
            f'{name} ({spec.family}) needs {", ".join(absent)}, which this context does '
            f'not supply; adaptive methods are not applicable where no accelerometer was '
            f'recorded')

    if context.reference is not None and context.reference.shape[1] != signal.size:
        raise ValueError(f'{name}: reference holds {context.reference.shape[1]} samples '
                         f'against {signal.size} in the signal')

    params = {**configured_params(name), **overrides}

    started = perf_counter()
    output = spec.fn(signal.copy(), context, **params)
    elapsed = perf_counter() - started

    output = np.asarray(output, dtype=np.float64).ravel()
    if output.size != signal.size:
        raise ValueError(f'{name} returned {output.size} samples against {signal.size} '
                         f'in the input')
    if not np.all(np.isfinite(output)):
        raise ValueError(f'{name} returned non-finite samples; check the step size or the '
                         f'conditioning of the reference')

    return FilterResult(
        signal=output,
        method=name,
        family=spec.family,
        params=params,
        elapsed_s=elapsed,
        code_version=code_version(),
        n_reference_channels=0 if context.reference is None else context.reference.shape[0],
    )


def describe(name: str) -> str:
    """One line about a method, for `--list` and for the console."""
    spec = get_spec(name)
    needs = [label for label, flag in (('reference', spec.requires_reference),
                                       ('r_peaks', spec.requires_r_peaks),
                                       ('checkpoint', spec.requires_checkpoint)) if flag]
    return (f'{spec.name:24s} {spec.family:9s} '
            f'wymaga: {", ".join(needs) if needs else "-":24s} {spec.description}')
