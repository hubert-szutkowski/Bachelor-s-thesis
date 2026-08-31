"""
Adaptive method tests.

The material is synthetic and built so that the answer is known: a clean beat train plus a
weighted sum of the reference channels. A canceller with access to those channels should
recover most of the beat train, and one that does not is either broken or reading a
reference that no longer carries the artefact.

The signal to noise ratio is used as the measure throughout. It is the quantity these
methods exist to improve, and a filter that runs, returns the right length and improves
nothing is a failure that no shape check would catch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.methods_adaptive import prepare_reference, register_adaptive_filters
from filters.methods_static import register_static_filters
from filters.registry import (
    FilterContext,
    applicable_filters,
    apply_filter,
    available_filters,
    configured_params,
    load_config,
    reset_config,
    unregister,
)

FS = 360.0
N = 4096
CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'filters.yaml'


@pytest.fixture(autouse=True)
def registry():
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    names = register_adaptive_filters()
    load_config(CONFIG)
    yield names
    for name in list(available_filters()):
        unregister(name)
    reset_config()


def accelerometer(base, seed, n=N):
    rng = np.random.default_rng(seed)
    return np.stack([base + 0.4 * rng.standard_normal(n),
                     0.5 * base + 0.8 * rng.standard_normal(n),
                     0.2 * base + 1.0 * rng.standard_normal(n)])


def scene(n=N, n_sensors=2, seed=0):
    """A beat train plus a weighted sum of reference channels, so the answer is known."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    clean = np.zeros(n)
    for peak in range(int(FS), n - int(FS), 288):
        clean[peak - 5:peak + 5] += np.hanning(10) * 1.2
        clean[peak + 20:peak + 60] += np.hanning(40) * 0.25

    first = np.sin(2 * np.pi * 1.7 * t) + 0.3 * rng.standard_normal(n)
    second = 0.7 * np.roll(first, 5) + 0.6 * np.sin(2 * np.pi * 0.5 * t)
    sensors = [accelerometer(first, 1, n)]
    if n_sensors > 1:
        sensors.append(accelerometer(second, 2, n))

    reference = np.vstack(sensors)
    noisy = clean + 0.30 * reference[0]
    if n_sensors > 1:
        noisy = noisy + 0.20 * reference[3]
    return clean, noisy, reference


def snr(clean, estimate):
    return float(10.0 * np.log10(np.sum(clean ** 2) / np.sum((estimate - clean) ** 2)))


@pytest.fixture
def context():
    _, _, reference = scene()
    return FilterContext(fs=FS, reference=reference)


# --- the whole family ----------------------------------------------------

def test_exactly_five_adaptive_methods_are_registered(registry):
    assert len(registry) == 5
    assert available_filters('adaptive') == sorted(registry)


def test_every_method_declares_that_it_needs_a_reference(registry):
    """
    Which is why none of them appears in the synthetic comparison.

    MIT-BIH carries no accelerometer, so the adaptive family is inapplicable there and the
    registry has to say so rather than let them run on something improvised.
    """
    report = applicable_filters(FilterContext(fs=FS), family='adaptive')
    assert report['runnable'] == []
    assert set(report['blocked']) == set(registry)


def test_every_method_runs_and_returns_the_input_length(registry, context):
    _, noisy, _ = scene()
    for name in registry:
        result = apply_filter(name, noisy, context)
        assert result.signal.shape == noisy.shape, name
        assert np.all(np.isfinite(result.signal)), name


def test_every_method_improves_the_ratio(registry, context):
    """A filter that runs and improves nothing passes every shape check."""
    clean, noisy, _ = scene()
    before = snr(clean, noisy)
    for name in registry:
        after = snr(clean, apply_filter(name, noisy, context).signal)
        assert after > before + 1.0, f'{name}: {before:.2f} -> {after:.2f} dB'


def test_every_method_belongs_to_the_adaptive_family(registry, context):
    _, noisy, _ = scene()
    for name in registry:
        assert apply_filter(name, noisy, context).family == 'adaptive'


def test_every_method_leaves_its_input_untouched(registry, context):
    _, noisy, _ = scene()
    original = noisy.copy()
    for name in registry:
        apply_filter(name, noisy, context)
    assert np.array_equal(noisy, original)


def test_the_channel_count_is_recorded(registry, context):
    _, noisy, _ = scene()
    assert apply_filter('rls_anc', noisy, context).n_reference_channels == 6


# --- one accelerometer against two ---------------------------------------

@pytest.mark.parametrize('name', ['lms_anc', 'rls_anc', 'blms_anc', 'gall_kalman'])
def test_the_multichannel_methods_use_every_channel(registry, name):
    """
    An identical result would mean the extra channels never reached the weight vector.

    Whether the second accelerometer helps is a question for the experiment; that it is
    used at all is a question for the code.
    """
    clean, noisy, reference = scene(n_sensors=2)
    one = apply_filter(name, noisy, FilterContext(fs=FS, reference=reference[:3])).signal
    both = apply_filter(name, noisy, FilterContext(fs=FS, reference=reference)).signal
    assert not np.allclose(one, both)


def test_a_single_accelerometer_still_works(registry):
    clean, noisy, reference = scene(n_sensors=1)
    context = FilterContext(fs=FS, reference=reference)
    before = snr(clean, noisy)
    for name in registry:
        assert snr(clean, apply_filter(name, noisy, context).signal) > before


# --- standardisation of the reference ------------------------------------

def test_the_reference_reaches_the_filters_with_unit_variance(registry):
    _, _, reference = scene()
    prepared = prepare_reference(FilterContext(fs=FS, reference=reference))
    assert np.allclose(prepared.std(axis=1), 1.0)


def test_the_unit_of_the_reference_does_not_change_the_result(registry):
    """
    The stability bound of the step size depends on the power of the reference.

    Without standardisation, a step size chosen for a sensor in units of gravity would
    diverge for the same sensor reported in converter counts.
    """
    clean, noisy, reference = scene()
    in_g = apply_filter('lms_anc', noisy, FilterContext(fs=FS, reference=reference)).signal
    in_counts = apply_filter('lms_anc', noisy,
                             FilterContext(fs=FS, reference=4096.0 * reference)).signal
    assert np.allclose(in_g, in_counts)


def test_without_standardisation_the_same_step_size_diverges(registry):
    """
    The demonstration is stronger than expected: the filter does not merely do worse.

    A reference reported in converter counts carries about ten million times the power of
    the same reference in units of gravity, which puts the configured step size far above
    the stability bound. The weights grow without limit, and the guard in the registry
    catches the result before it reaches a metric and turns a whole aggregate into a
    not-a-number.
    """
    clean, noisy, reference = scene()
    context = FilterContext(fs=FS, reference=4096.0 * reference)

    standardised = apply_filter('lms_anc', noisy, context).signal
    assert snr(clean, standardised) > snr(clean, noisy)

    with pytest.raises(ValueError, match='non-finite'):
        apply_filter('lms_anc', noisy, context, standardise_reference=False)


# --- how the lattice reduces the channels --------------------------------

def test_the_magnitude_of_the_acceleration_cancels_nothing(registry, context):
    """
    Rectifying the channels destroys the linear relationship a canceller acts on.

    The magnitude looked like the principled reduction and measures as the useless one:
    it correlates with the channel that generated the artefact at about 0.01.
    """
    clean, noisy, _ = scene()
    before = snr(clean, noisy)
    magnitude = apply_filter('gall_anc', noisy, context, reduction='svm').signal
    assert snr(clean, magnitude) < before + 0.5


def test_the_principal_component_cancels_most_of_it(registry, context):
    clean, noisy, _ = scene()
    before = snr(clean, noisy)
    projected = apply_filter('gall_anc', noisy, context, reduction='pca').signal
    assert snr(clean, projected) > before + 5.0


def test_the_projection_beats_the_magnitude_by_a_wide_margin(registry, context):
    clean, noisy, _ = scene()
    magnitude = snr(clean, apply_filter('gall_anc', noisy, context, reduction='svm').signal)
    projected = snr(clean, apply_filter('gall_anc', noisy, context, reduction='pca').signal)
    assert projected > magnitude + 5.0


def test_the_projection_is_computed_from_the_reference_alone(registry):
    """
    Deterministic and blind to the outcome, which is what makes it a sound reduction.

    Picking whichever channel scored best would be a choice made on the result.
    """
    from filters.reference import principal_channel

    _, _, reference = scene()
    first = principal_channel(reference)
    second = principal_channel(reference)
    assert np.array_equal(first, second)
    assert first.shape == (1, N)


def test_an_unknown_reduction_is_rejected(registry, context):
    _, noisy, _ = scene()
    with pytest.raises(ValueError, match='unknown reduction'):
        apply_filter('gall_anc', noisy, context, reduction='median')


# --- the low pass guide of the modified least mean squares ---------------

def test_the_guide_signal_is_computed_not_supplied(registry):
    """
    Taking it from the caller would make the result depend on what the caller happened
    to pass, and the method would stop being reproducible from its own record.
    """
    assert 'lowpass_hz' in configured_params('lms_anc')
    assert 'lowpass_order' in configured_params('lms_anc')


def test_the_guide_cutoff_changes_the_result(registry, context):
    _, noisy, _ = scene()
    narrow = apply_filter('lms_anc', noisy, context, lowpass_hz=10.0).signal
    wide = apply_filter('lms_anc', noisy, context, lowpass_hz=40.0).signal
    assert not np.allclose(narrow, wide)


# --- parameters and provenance -------------------------------------------

def test_the_shipped_configuration_covers_every_adaptive_method(registry):
    import yaml

    with CONFIG.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    assert set(loaded['adaptive']) == set(registry)


def test_a_call_argument_changes_the_result(registry, context):
    _, noisy, _ = scene()
    low = apply_filter('rls_anc', noisy, context, filter_order=3).signal
    high = apply_filter('rls_anc', noisy, context, filter_order=12).signal
    assert not np.allclose(low, high)


def test_the_resolved_parameters_are_recorded(registry, context):
    _, noisy, _ = scene()
    result = apply_filter('rls_anc', noisy, context, lam=0.95)
    assert result.params['lam'] == 0.95
    assert result.params['filter_order'] == 5


def test_the_returned_tuples_are_reduced_to_a_waveform(registry, context):
    """
    Two implementations return a weight history alongside the signal.

    At order 32 over a window of 4096 that history is a megabyte per window, so the
    adapters keep the waveform and drop the rest.
    """
    _, noisy, _ = scene()
    for name in ('gall_anc', 'gall_kalman'):
        result = apply_filter(name, noisy, context)
        assert result.signal.ndim == 1


def test_every_method_reports_a_positive_runtime(registry, context):
    _, noisy, _ = scene()
    for name in registry:
        assert apply_filter(name, noisy, context).elapsed_s > 0.0


# --- the two families together -------------------------------------------

def test_the_wearable_context_runs_twelve_of_the_seventeen(registry, context):
    """Seven static and five adaptive; the networks are registered elsewhere."""
    register_static_filters()
    report = applicable_filters(context)
    assert len(report['runnable']) == 12
    assert report['blocked'] == {}


def test_the_synthetic_context_runs_only_the_static_seven(registry):
    register_static_filters()
    report = applicable_filters(FilterContext(fs=FS))
    assert len(report['runnable']) == 7
    assert len(report['blocked']) == 5
