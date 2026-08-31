"""
Preparation and diagnosis of the reference channels of an adaptive filter.

An ECG is a differential measurement: the potential recorded is the difference between
two electrodes, and a motion artefact arises at the skin-electrode interface of each of
them. One accelerometer can therefore explain one of the two terms. Two accelerometers,
one on each electrode, match the structure of the measurement, and the adaptive filters
in this project take the reference as a matrix of shape (channels, samples) so that any
number of them may be supplied.

More references are not free. The weight vector grows as channels times filter order, so
convergence takes longer and the steady state misadjustment rises roughly in proportion.
Whether a second accelerometer helps is therefore a question to be measured rather than
assumed, and running the same recording with each sensor alone and with both together
answers it.

Redundancy is a separate matter from usefulness. Two sensors rigidly coupled through the
rib cage see nearly the same motion, and a reference matrix built from both is then close
to rank deficient. Its autocorrelation matrix becomes ill conditioned, the recursive least
squares update inverts something almost singular, and the filter can diverge for numerical
reasons that have nothing to do with the signal. Dropping a redundant channel is a
remedy for that, and `select_channels` implements it.

Two conditions make that selection sound, and both are the caller's responsibility:

    the threshold is fixed before the data is seen, not tuned until the result improves;
    the decision is taken on the development material and then frozen, never re-taken per
    recording on the held out material.

The criterion is deliberately blind to the outcome. Channels are ranked by their own
variance, a property of the reference alone, so nothing about the filtered result enters
the choice. Ranking them by the score they eventually produce would be selection on the
outcome and would invalidate the comparison it was meant to inform.
"""

from typing import Optional, Sequence

import numpy as np

DEFAULT_REDUNDANCY_THRESHOLD = 0.95
CONDITION_WARNING = 1e6


def reference_matrix(*channels, svm: bool = False) -> np.ndarray:
    """
    Stacks reference channels into a matrix of shape (channels, samples).

    Accepts individual one dimensional channels, a single two dimensional array already in
    that layout, or any mixture. `svm` collapses each group of three consecutive channels
    into its signal vector magnitude, which reduces one accelerometer to a single channel
    that carries the amount of motion but not its direction.
    """
    stacked = []
    for channel in channels:
        if channel is None:
            continue
        array = np.asarray(channel, dtype=np.float64)
        if array.ndim == 1:
            stacked.append(array[None, :])
        elif array.ndim == 2:
            stacked.append(array)
        else:
            raise ValueError(f'a reference channel must be one or two dimensional, '
                             f'got shape {array.shape}')

    if not stacked:
        raise ValueError('at least one reference channel is required')

    lengths = {part.shape[1] for part in stacked}
    if len(lengths) != 1:
        raise ValueError(f'reference channels differ in length: {sorted(lengths)}')

    reference = np.concatenate(stacked, axis=0)

    if svm:
        if reference.shape[0] % 3:
            raise ValueError(f'the signal vector magnitude needs channels in groups of '
                             f'three, got {reference.shape[0]}')
        groups = reference.reshape(-1, 3, reference.shape[1])
        return np.sqrt(np.sum(groups ** 2, axis=1))

    return reference


def standardise(reference: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Centres each channel and scales it to unit variance.

    The stability bound of the least mean squares update depends on the power of the
    reference, so a step size chosen for an accelerometer reported in units of gravity
    diverges for the same sensor reported in raw converter counts. Fixing the power makes
    the step size a property of the algorithm rather than of the recording equipment,
    which is what lets one value carry across devices.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    centred = reference - reference.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    return centred / np.maximum(scale, eps)


def principal_channel(reference: np.ndarray) -> np.ndarray:
    """
    The single channel that carries most of the motion, as a linear combination.

    Methods built around one reference, such as the Laguerre lattice, need the several
    channels reduced to one. Three reductions are available and only one of them is sound.

    Picking an axis is a choice made on the data unless the axis is fixed in advance, and
    fixed in advance it is arbitrary. The magnitude of the acceleration looks principled
    and is not: rectifying the channels destroys the linear relationship between the
    reference and the artefact, which is the only relationship a linear canceller can act
    on. Measured on synthetic material, the magnitude correlates with the channel that
    generated the artefact at 0.015 and the lattice removes nothing, while the first
    principal component correlates at 0.80 and recovers most of what the multichannel
    methods recover.

    The projection is linear, deterministic and computed from the reference alone, so it
    keeps the choice on the reference side of the problem.
    """
    matrix = standardise(reference)
    centred = matrix - matrix.mean(axis=1, keepdims=True)

    # rozklad wedlug wartosci osobliwych macierzy (kanaly, probki): kolumny U to wagi
    # przypisane kanalom, wiersze Vt to odpowiadajace im przebiegi w czasie. Pierwszy
    # wiersz Vt jest wiec juz gotowa kombinacja liniowa kanalow, rownowazna U[:, 0] @ X.
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return standardise(components[0][None, :])


def channel_correlations(reference: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of reference channels."""
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    if reference.shape[0] == 1:
        return np.ones((1, 1))

    spread = reference.std(axis=1)
    if np.any(spread == 0):
        raise ValueError('a constant reference channel has no correlation with anything')
    return np.corrcoef(reference)


def conditioning(reference: np.ndarray) -> dict:
    """
    How close the reference set is to being rank deficient.

    `condition` is the ratio of the largest to the smallest singular value of the
    standardised reference; a large value means one channel is nearly a linear combination
    of the others and the recursive least squares update is inverting something almost
    singular. `effective_rank` is the exponential of the entropy of the normalised singular
    values, and reads as the number of channels that genuinely carry distinct information.
    """
    matrix = standardise(reference)
    values = np.linalg.svd(matrix, compute_uv=False)
    values = values[values > 0]

    if values.size == 0:
        raise ValueError('the reference matrix has no non-zero singular values')

    weights = values / values.sum()
    entropy = float(-np.sum(weights * np.log(weights)))

    return {
        'n_channels': int(np.atleast_2d(reference).shape[0]),
        'condition': float(values[0] / values[-1]),
        'effective_rank': float(np.exp(entropy)),
        'ill_conditioned': bool(values[0] / values[-1] > CONDITION_WARNING),
    }


def select_channels(reference: np.ndarray,
                    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
                    names: Optional[Sequence[str]] = None) -> dict:
    """
    Drops channels that duplicate one already kept.

    Channels are visited in decreasing order of their own variance and a channel is
    discarded when its absolute correlation with any already kept channel exceeds
    `threshold`. Ordering by variance keeps the decision on the reference side of the
    problem: nothing about the filtered result takes part in it.

    Returns the kept indices, the discarded ones with the channel each duplicates, and the
    conditioning before and after, so the whole decision can be reported rather than
    merely applied.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f'threshold must lie in (0, 1], got {threshold}')

    n_channels = reference.shape[0]
    labels = list(names) if names is not None else [f'ch{index}' for index in range(n_channels)]
    if len(labels) != n_channels:
        raise ValueError(f'{len(labels)} names for {n_channels} channels')

    correlation = np.abs(channel_correlations(reference))
    order = np.argsort(-reference.std(axis=1))

    kept, dropped = [], []
    for candidate in order:
        duplicate = next((other for other in kept
                          if correlation[candidate, other] > threshold), None)
        if duplicate is None:
            kept.append(int(candidate))
        else:
            dropped.append({'channel': int(candidate),
                            'name': labels[candidate],
                            'duplicates': labels[duplicate],
                            'correlation': float(correlation[candidate, duplicate])})

    kept.sort()
    return {
        'kept': kept,
        'kept_names': [labels[index] for index in kept],
        'dropped': dropped,
        'threshold': float(threshold),
        'max_correlation': float(np.max(correlation - np.eye(n_channels))) if n_channels > 1 else 0.0,
        'before': conditioning(reference),
        'after': conditioning(reference[kept]),
    }


def apply_selection(reference: np.ndarray, selection: dict) -> np.ndarray:
    """
    Applies a selection taken elsewhere.

    Separate from `select_channels` on purpose. The decision is meant to be taken once on
    the development material and then carried unchanged to the held out material; deciding
    again on each recording would be a choice made on the data being reported.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))
    kept = list(selection['kept'])
    if not kept:
        raise ValueError('the selection keeps no channel')
    if max(kept) >= reference.shape[0]:
        raise ValueError(f'the selection refers to channel {max(kept)} of a reference '
                         f'holding {reference.shape[0]}')
    return reference[kept]


def describe_selection(selection: dict) -> str:
    """Readable summary of a selection, for the console and for the thesis appendix."""
    lines = [
        f'kanaly zachowane : {", ".join(selection["kept_names"])}',
        f'prog redundancji : {selection["threshold"]:.2f}',
        f'maks. korelacja  : {selection["max_correlation"]:.4f}',
        f'uwarunkowanie    : {selection["before"]["condition"]:.3g} '
        f'-> {selection["after"]["condition"]:.3g}',
        f'rank efektywny   : {selection["before"]["effective_rank"]:.2f} '
        f'-> {selection["after"]["effective_rank"]:.2f}',
    ]
    for entry in selection['dropped']:
        lines.append(f'  odrzucono {entry["name"]}: powiela {entry["duplicates"]} '
                     f'(r = {entry["correlation"]:.4f})')
    if not selection['dropped']:
        lines.append('  nie odrzucono zadnego kanalu')
    return '\n'.join(lines)
