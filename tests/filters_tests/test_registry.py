"""
Registry contract tests.

No real method takes part here. What is under test is the machinery that will carry all
seventeen of them: that a method declaring a need is refused when the need is unmet, that
parameters resolve in the intended order, that a filter cannot reach into its caller's
array, and that a waveform holding a not-a-number is stopped before it reaches a metric
and turns a whole aggregate into one.

The refusal is the point. Adaptive methods are not applicable on a database recorded
without an accelerometer, and the registry has to say so rather than fail deep inside
numpy with a message about shapes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.registry import (
    FAMILIES,
    FilterContext,
    FilterResult,
    apply_filter,
    applicable_filters,
    available_filters,
    code_version,
    configured_params,
    describe,
    get_spec,
    load_config,
    register,
    reset_config,
    unregister,
)

FS = 360.0
N = 1024


@pytest.fixture
def signal():
    waveform = np.zeros(N)
    for peak in range(100, N - 100, 288):
        waveform[peak - 5:peak + 5] += np.hanning(10) * 1.2
    return waveform


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return rng.standard_normal((6, N))


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts from an empty registry and an empty configuration."""
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    yield
    for name in list(available_filters()):
        unregister(name)
    reset_config()


def gain_filter(signal, context, gain=1.0, **_):
    return gain * signal


def reference_filter(signal, context, weight=1.0, **_):
    return signal - weight * context.reference[0]


# --- registration --------------------------------------------------------

def test_a_registered_method_becomes_available():
    register('demo', 'static', gain_filter)
    assert available_filters() == ['demo']


def test_families_are_listed_separately():
    register('a', 'static', gain_filter)
    register('b', 'adaptive', reference_filter, requires_reference=True)
    register('c', 'deep', gain_filter, requires_checkpoint=True)
    assert available_filters('static') == ['a']
    assert available_filters('adaptive') == ['b']
    assert available_filters('deep') == ['c']


def test_the_listing_order_is_stable():
    for name in ('zeta', 'alpha', 'mu'):
        register(name, 'static', gain_filter)
    assert available_filters() == ['alpha', 'mu', 'zeta']


def test_an_unknown_family_is_rejected():
    with pytest.raises(ValueError, match='unknown family'):
        register('demo', 'wavelet', gain_filter)


def test_a_name_cannot_be_registered_twice():
    register('demo', 'static', gain_filter)
    with pytest.raises(ValueError, match='already registered'):
        register('demo', 'static', gain_filter)


def test_an_unknown_method_is_reported_with_the_available_ones():
    register('demo', 'static', gain_filter)
    with pytest.raises(KeyError, match='demo'):
        get_spec('missing')


# --- declared requirements -----------------------------------------------

def test_a_method_needing_a_reference_is_refused_without_one(signal):
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    with pytest.raises(ValueError, match='needs reference'):
        apply_filter('anc', signal, FilterContext(fs=FS))


def test_the_refusal_names_the_reason(signal):
    """
    On MIT-BIH there is no accelerometer, so the adaptive family is inapplicable.

    The message has to say that rather than surface as a shape error several frames deep.
    """
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    with pytest.raises(ValueError, match='no accelerometer'):
        apply_filter('anc', signal, FilterContext(fs=FS))


def test_a_method_needing_beats_is_refused_without_them(signal):
    register('beatwise', 'static', gain_filter, requires_r_peaks=True)
    with pytest.raises(ValueError, match='needs r_peaks'):
        apply_filter('beatwise', signal, FilterContext(fs=FS))


def test_a_network_is_refused_without_weights(signal):
    register('net', 'deep', gain_filter, requires_checkpoint=True)
    with pytest.raises(ValueError, match='needs checkpoint'):
        apply_filter('net', signal, FilterContext(fs=FS))


def test_every_unmet_need_is_listed_at_once(signal):
    register('greedy', 'deep', gain_filter, requires_reference=True,
             requires_r_peaks=True, requires_checkpoint=True)
    with pytest.raises(ValueError) as raised:
        apply_filter('greedy', signal, FilterContext(fs=FS))
    for need in ('reference', 'r_peaks', 'checkpoint'):
        assert need in str(raised.value)


def test_a_satisfied_requirement_runs(signal, reference):
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    result = apply_filter('anc', signal, FilterContext(fs=FS, reference=reference))
    assert result.signal.shape == signal.shape


# --- what a context can run ----------------------------------------------

def test_the_synthetic_environment_excludes_the_adaptive_family(signal):
    register('static_one', 'static', gain_filter)
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    register('net', 'deep', gain_filter, requires_checkpoint=True)

    report = applicable_filters(FilterContext(fs=FS, checkpoint=Path('x.pt')))
    assert report['runnable'] == ['net', 'static_one']
    assert report['blocked'] == {'anc': ['reference']}


def test_the_wearable_environment_runs_everything(signal, reference):
    register('static_one', 'static', gain_filter)
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    register('net', 'deep', gain_filter, requires_checkpoint=True)

    report = applicable_filters(FilterContext(fs=FS, reference=reference,
                                              checkpoint=Path('x.pt')))
    assert len(report['runnable']) == 3
    assert report['blocked'] == {}


# --- parameters ----------------------------------------------------------

def test_registered_defaults_are_used(signal):
    register('demo', 'static', gain_filter, defaults={'gain': 2.0})
    result = apply_filter('demo', signal, FilterContext(fs=FS))
    assert result.params['gain'] == 2.0
    assert np.allclose(result.signal, 2.0 * signal)


def test_a_call_argument_overrides_the_default(signal):
    register('demo', 'static', gain_filter, defaults={'gain': 2.0})
    result = apply_filter('demo', signal, FilterContext(fs=FS), gain=5.0)
    assert result.params['gain'] == 5.0


def test_the_configuration_file_overrides_the_default(signal, tmp_path):
    register('demo', 'static', gain_filter, defaults={'gain': 2.0})
    path = tmp_path / 'filters.yaml'
    path.write_text('static:\n  demo:\n    gain: 3.0\n', encoding='utf-8')
    load_config(path)
    assert configured_params('demo')['gain'] == 3.0


def test_a_call_argument_overrides_the_configuration_file(signal, tmp_path):
    """Three layers, each overriding the one before."""
    register('demo', 'static', gain_filter, defaults={'gain': 2.0})
    path = tmp_path / 'filters.yaml'
    path.write_text('static:\n  demo:\n    gain: 3.0\n', encoding='utf-8')
    load_config(path)
    result = apply_filter('demo', signal, FilterContext(fs=FS), gain=7.0)
    assert result.params['gain'] == 7.0


def test_a_missing_configuration_file_is_not_an_error(tmp_path):
    register('demo', 'static', gain_filter, defaults={'gain': 2.0})
    load_config(tmp_path / 'absent.yaml')
    assert configured_params('demo') == {'gain': 2.0}


def test_an_unknown_family_in_the_file_is_rejected(tmp_path):
    path = tmp_path / 'filters.yaml'
    path.write_text('wavelets:\n  demo:\n    gain: 3.0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unknown family'):
        load_config(path)


def test_the_resolved_parameters_travel_with_the_result(signal):
    """A number in a table nobody can reproduce is not a result."""
    register('demo', 'static', gain_filter, defaults={'gain': 2.0, 'unused': 'x'})
    result = apply_filter('demo', signal, FilterContext(fs=FS), gain=4.0)
    assert result.params == {'gain': 4.0, 'unused': 'x'}


# --- guards on the waveform ----------------------------------------------

def test_the_caller_array_is_never_written_into(signal):
    def vandal(data, context, **_):
        data *= 0.0
        return data + 1.0

    register('vandal', 'static', vandal)
    original = signal.copy()
    apply_filter('vandal', signal, FilterContext(fs=FS))
    assert np.array_equal(signal, original)


def test_a_method_returning_the_wrong_length_is_caught(signal):
    register('shrink', 'static', lambda data, context, **_: data[:100])
    with pytest.raises(ValueError, match='returned 100 samples'):
        apply_filter('shrink', signal, FilterContext(fs=FS))


def test_a_method_returning_non_finite_samples_is_caught(signal):
    """
    Otherwise one window turns a whole aggregate into a not-a-number.

    That reads as a broken pipeline rather than as a diverging filter, which is the more
    expensive of the two to diagnose.
    """
    def diverging(data, context, **_):
        out = data.copy()
        out[10] = np.inf
        return out

    register('diverging', 'static', diverging)
    with pytest.raises(ValueError, match='non-finite'):
        apply_filter('diverging', signal, FilterContext(fs=FS))


def test_a_non_finite_input_is_refused_before_the_method_runs(signal):
    register('demo', 'static', gain_filter)
    broken = signal.copy()
    broken[5] = np.nan
    with pytest.raises(ValueError, match='input holds non-finite'):
        apply_filter('demo', broken, FilterContext(fs=FS))


def test_an_empty_signal_is_refused():
    register('demo', 'static', gain_filter)
    with pytest.raises(ValueError, match='empty'):
        apply_filter('demo', np.array([]), FilterContext(fs=FS))


def test_a_reference_of_the_wrong_length_is_refused(signal):
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    context = FilterContext(fs=FS, reference=np.zeros((3, N // 2)))
    with pytest.raises(ValueError, match='against 1024'):
        apply_filter('anc', signal, context)


def test_a_missing_context_is_refused(signal):
    register('demo', 'static', gain_filter)
    with pytest.raises(ValueError, match='FilterContext'):
        apply_filter('demo', signal)


# --- context -------------------------------------------------------------

def test_a_one_dimensional_reference_becomes_a_single_channel():
    context = FilterContext(fs=FS, reference=np.zeros(N))
    assert context.reference.shape == (1, N)


def test_a_nonpositive_sampling_frequency_is_refused():
    with pytest.raises(ValueError, match='fs must be positive'):
        FilterContext(fs=0.0)


def test_the_context_cannot_be_modified_after_construction():
    context = FilterContext(fs=FS)
    with pytest.raises(Exception):
        context.fs = 250.0


def test_the_powerline_frequency_is_a_parameter_not_a_constant():
    """MIT-BIH was recorded in the United States at 60 Hz; Poland runs at 50."""
    assert FilterContext(fs=FS).powerline_hz == 50.0
    assert FilterContext(fs=FS, powerline_hz=60.0).powerline_hz == 60.0


# --- result --------------------------------------------------------------

def test_the_result_carries_its_provenance(signal, reference):
    register('anc', 'adaptive', reference_filter, requires_reference=True)
    result = apply_filter('anc', signal, FilterContext(fs=FS, reference=reference))

    assert isinstance(result, FilterResult)
    assert result.method == 'anc'
    assert result.family == 'adaptive'
    assert result.n_reference_channels == 6
    assert result.elapsed_s >= 0.0
    assert isinstance(result.code_version, str) and result.code_version


def test_the_result_cannot_be_modified(signal):
    register('demo', 'static', gain_filter)
    result = apply_filter('demo', signal, FilterContext(fs=FS))
    with pytest.raises(Exception):
        result.method = 'other'


def test_the_code_version_is_stable_within_a_run():
    assert code_version() == code_version()


def test_the_description_names_the_family_and_the_needs():
    register('anc', 'adaptive', reference_filter, requires_reference=True,
             description='wielokanalowy ANC')
    text = describe('anc')
    assert 'adaptive' in text and 'reference' in text and 'wielokanalowy' in text


def test_every_family_name_is_covered_by_the_tuple():
    for family in FAMILIES:
        register(f'demo_{family}', family, gain_filter)
    assert len(available_filters()) == len(FAMILIES)
