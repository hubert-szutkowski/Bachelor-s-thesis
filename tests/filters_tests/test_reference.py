"""
Reference channel tests.

Two accelerometers turn the reference from three channels into six, and every adaptive
filter in this project has to accept both without its weight vector, its regressor or its
covariance quietly keeping the old size. The first section pins that. The second covers
the diagnosis of redundancy, whose purpose is numerical rather than cosmetic: two sensors
rigidly coupled through the rib cage make the reference nearly rank deficient, and the
recursive least squares update then inverts something almost singular.

The selection rule is blind to the filtered result on purpose, and one test asserts that
directly, because a rule that ranked channels by the score they eventually produce would
be a choice made on the outcome.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.adaptive_filters import (
    blms_ecg_filter,
    hybrid_gall_kalman_ecg_filter,
    modified_lms_anc,
    rls_anc,
)
from filters.reference import (
    apply_selection,
    channel_correlations,
    conditioning,
    describe_selection,
    reference_matrix,
    select_channels,
    standardise,
)

FS = 360.0
N = 2000


def motion(seed=0, n=N):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 1.7 * t) + 0.3 * rng.standard_normal(n)


def accelerometer(base, seed=0, n=N):
    """
    Three axes sharing a common motion but measuring different projections of it.

    Axes correlated at 0.99 with one another would be a sensor reporting the same number
    three times, and the redundancy rule would rightly discard two of them.
    """
    rng = np.random.default_rng(seed)
    return np.stack([base + 0.4 * rng.standard_normal(n),
                     0.5 * base + 0.8 * rng.standard_normal(n),
                     0.2 * base + 1.0 * rng.standard_normal(n)])


def ecg(n=N):
    signal = np.zeros(n)
    for peak in range(180, n - 180, 288):
        signal[peak - 5:peak + 5] += np.hanning(10) * 1.2
    return signal


@pytest.fixture
def one_sensor():
    return accelerometer(motion(0), seed=1)


@pytest.fixture
def two_sensors():
    first = accelerometer(motion(0), seed=1)
    second = accelerometer(0.8 * np.roll(motion(0), 3) + 0.5 * motion(5), seed=2)
    return np.vstack([first, second])


# --- assembly ------------------------------------------------------------

def test_separate_axes_stack_into_a_matrix():
    axes = [motion(index) for index in range(3)]
    assert reference_matrix(*axes).shape == (3, N)


def test_two_accelerometers_stack_into_six_channels(one_sensor):
    assert reference_matrix(one_sensor, one_sensor).shape == (6, N)


def test_a_matrix_and_loose_axes_may_be_mixed(one_sensor):
    assert reference_matrix(one_sensor, motion(9)).shape == (4, N)


def test_the_signal_vector_magnitude_collapses_each_triple(two_sensors):
    collapsed = reference_matrix(two_sensors, svm=True)
    assert collapsed.shape == (2, N)
    assert np.all(collapsed >= 0)


def test_the_magnitude_needs_whole_triples():
    with pytest.raises(ValueError, match='groups of'):
        reference_matrix(motion(0), motion(1), svm=True)


def test_channels_of_unequal_length_are_rejected():
    with pytest.raises(ValueError, match='differ in length'):
        reference_matrix(motion(0), motion(1, n=500))


def test_an_empty_reference_is_rejected():
    with pytest.raises(ValueError, match='at least one'):
        reference_matrix(None)


# --- standardisation -----------------------------------------------------

def test_standardising_gives_every_channel_unit_variance(two_sensors):
    """
    The stability bound of the least mean squares update depends on reference power.

    A step size chosen for a sensor reported in units of gravity diverges for the same
    sensor reported in converter counts; fixing the power makes it carry across devices.
    """
    scaled = standardise(two_sensors)
    assert np.allclose(scaled.std(axis=1), 1.0)
    assert np.allclose(scaled.mean(axis=1), 0.0, atol=1e-12)


def test_standardising_removes_the_effect_of_the_unit(two_sensors):
    assert np.allclose(standardise(two_sensors), standardise(4096.0 * two_sensors))


def test_a_constant_channel_does_not_divide_by_zero():
    reference = np.vstack([motion(0), np.zeros(N)])
    assert np.all(np.isfinite(standardise(reference)))


# --- diagnosis -----------------------------------------------------------

def test_correlation_of_a_channel_with_itself_is_one(two_sensors):
    assert np.allclose(np.diag(channel_correlations(two_sensors)), 1.0)


def test_duplicated_sensors_correlate_almost_perfectly(one_sensor):
    reference = np.vstack([one_sensor, one_sensor + 1e-6 * motion(3)])
    correlation = np.abs(channel_correlations(reference))
    assert correlation[0, 3] > 0.99


def test_independent_sensors_do_not(two_sensors):
    correlation = np.abs(channel_correlations(two_sensors))
    assert correlation[0, 3] < 0.95


def test_a_duplicated_sensor_is_ill_conditioned(one_sensor):
    """Rigid coupling through the rib cage produces exactly this."""
    doubled = conditioning(np.vstack([one_sensor, one_sensor]))
    single = conditioning(one_sensor)
    assert doubled['condition'] > 100 * single['condition']


def test_effective_rank_counts_the_distinct_channels(one_sensor):
    doubled = conditioning(np.vstack([one_sensor, one_sensor]))
    assert doubled['n_channels'] == 6
    assert doubled['effective_rank'] < 4.0


# --- selection -----------------------------------------------------------

def test_a_duplicated_sensor_is_dropped(one_sensor):
    selection = select_channels(np.vstack([one_sensor, one_sensor]), threshold=0.95)
    assert len(selection['kept']) == 3
    assert len(selection['dropped']) == 3


def test_two_distinct_sensors_are_both_kept(two_sensors):
    selection = select_channels(two_sensors, threshold=0.95)
    assert len(selection['kept']) == 6
    assert selection['dropped'] == []


def test_selection_improves_the_conditioning(one_sensor):
    selection = select_channels(np.vstack([one_sensor, one_sensor]))
    assert selection['after']['condition'] < selection['before']['condition']


def test_the_rule_never_looks_at_the_filtered_result(two_sensors):
    """
    Ranking channels by the score they produce would be selection on the outcome.

    The same reference must yield the same selection whatever the ECG it will be used to
    clean, so the choice cannot depend on the signal being filtered.
    """
    first = select_channels(two_sensors)
    second = select_channels(two_sensors)
    assert first['kept'] == second['kept']
    assert 'signal' not in select_channels.__code__.co_varnames


def test_channels_are_visited_by_their_own_variance():
    """Ordering is a property of the reference alone, so it is reproducible."""
    strong = 5.0 * motion(0)
    weak = 0.1 * motion(0) + 1e-9 * motion(4)
    selection = select_channels(np.vstack([weak, strong]), threshold=0.9)
    assert selection['kept'] == [1]


def test_a_selection_can_be_carried_to_other_material(two_sensors):
    """
    The decision is taken once on development material and then frozen.

    Re-deciding on each recording would be a choice made on the data being reported.
    """
    selection = select_channels(two_sensors[:, :1000], threshold=0.95)
    applied = apply_selection(two_sensors[:, 1000:], selection)
    assert applied.shape == (len(selection['kept']), 1000)


def test_a_selection_referring_to_a_missing_channel_is_rejected(one_sensor):
    selection = {'kept': [0, 9]}
    with pytest.raises(ValueError, match='channel 9'):
        apply_selection(one_sensor, selection)


def test_an_impossible_threshold_is_rejected(two_sensors):
    with pytest.raises(ValueError, match='threshold'):
        select_channels(two_sensors, threshold=0.0)


def test_the_summary_names_what_was_dropped(one_sensor):
    text = describe_selection(select_channels(
        np.vstack([one_sensor, one_sensor]),
        names=['ax', 'ay', 'az', 'bx', 'by', 'bz']))
    assert 'powiela' in text and 'uwarunkowanie' in text


# --- the filters themselves ----------------------------------------------

@pytest.mark.parametrize('n_channels', [1, 3, 6, 9])
def test_every_adaptive_filter_accepts_any_channel_count(n_channels):
    """
    The weight vector is sized as channels times filter order in each of them.

    A filter that kept the old size would still run and still return a waveform, and the
    only symptom would be a worse result.
    """
    signal = ecg()
    reference = standardise(np.vstack([motion(index) for index in range(n_channels)]))
    noisy = signal + 0.3 * reference[0]

    assert rls_anc(noisy, reference=reference).shape == signal.shape
    assert blms_ecg_filter(noisy, reference=reference).shape == signal.shape
    assert modified_lms_anc(noisy, noisy, reference=reference).shape == signal.shape

    clean, total, weights = hybrid_gall_kalman_ecg_filter(noisy, reference=reference)
    assert clean.shape == signal.shape
    assert total.shape == signal.shape
    assert weights.shape == (signal.size, n_channels)


def test_the_hybrid_carries_one_weight_per_channel(two_sensors):
    """It used to hard-code three, which silently ignored the second accelerometer."""
    noisy = ecg() + 0.3 * two_sensors[0]
    _, _, weights = hybrid_gall_kalman_ecg_filter(noisy, reference=standardise(two_sensors))
    assert weights.shape[1] == 6


def test_the_separate_axis_convention_still_works(one_sensor):
    noisy = ecg() + 0.3 * one_sensor[0]
    by_axes = rls_anc(noisy, one_sensor[0], one_sensor[1], one_sensor[2])
    by_matrix = rls_anc(noisy, reference=one_sensor)
    assert np.allclose(by_axes, by_matrix)


def test_mixing_the_two_conventions_is_rejected(one_sensor):
    noisy = ecg() + 0.3 * one_sensor[0]
    with pytest.raises(ValueError, match='not both'):
        rls_anc(noisy, one_sensor[0], one_sensor[1], one_sensor[2], reference=one_sensor)


def test_a_second_sensor_changes_the_result(two_sensors):
    """
    Whether it helps is a question for the experiment; that it is used is a question here.

    An identical result would mean the extra channels never reached the weight vector.
    """
    noisy = ecg() + 0.3 * two_sensors[0] + 0.2 * two_sensors[3]
    one = rls_anc(noisy, reference=standardise(two_sensors[:3]))
    both = rls_anc(noisy, reference=standardise(two_sensors))
    assert not np.allclose(one, both)


def test_the_filters_leave_their_input_untouched(two_sensors):
    noisy = ecg() + 0.3 * two_sensors[0]
    original = noisy.copy()
    rls_anc(noisy, reference=standardise(two_sensors))
    assert np.array_equal(noisy, original)
