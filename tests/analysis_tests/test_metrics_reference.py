"""
Reference metric tests.

Every value here is checked against one computed by hand, because a metric that is wrong
by a constant factor moves every result in the same direction and no comparison between
methods would reveal it. The four conventions settled from the literature each get a test
that fails if the convention is quietly changed back.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.metrics_reference import (
    METRIC_FIELDS,
    aggregate,
    correlation,
    mean_absolute_error,
    mean_squared_error,
    prd,
    reference_metrics,
    root_mean_squared_error,
    snr,
    snr_improvement,
)

FS = 360.0
N = 4096


def beats(n=N, amplitude=1.2):
    signal = np.zeros(n)
    for peak in range(int(FS), n - int(FS), 288):
        signal[peak - 5:peak + 5] += np.hanning(10) * amplitude
        signal[peak + 20:peak + 60] += np.hanning(40) * 0.25 * amplitude
    return signal


# --- values computed by hand ---------------------------------------------

def test_the_ratio_matches_a_hand_computation():
    clean = np.array([3.0, 4.0, 0.0, 0.0])
    estimate = np.array([3.0, 4.0, 1.0, 1.0])
    assert snr(clean, estimate) == pytest.approx(10.0 * math.log10(25.0 / 2.0))


def test_the_difference_matches_a_hand_computation():
    clean = np.array([3.0, 4.0, 0.0, 0.0])
    estimate = np.array([3.0, 4.0, 1.0, 1.0])
    assert prd(clean, estimate) == pytest.approx(100.0 * math.sqrt(2.0 / 25.0))


def test_the_squared_and_absolute_errors_match_a_hand_computation():
    clean = np.array([0.0, 0.0, 0.0, 0.0])
    estimate = np.array([1.0, -1.0, 2.0, -2.0])
    assert mean_squared_error(clean, estimate) == pytest.approx(2.5)
    assert root_mean_squared_error(clean, estimate) == pytest.approx(math.sqrt(2.5))
    assert mean_absolute_error(clean, estimate) == pytest.approx(1.5)


def test_halving_the_residual_adds_six_decibels():
    clean = beats()
    rng = np.random.default_rng(0)
    noise = 0.2 * rng.standard_normal(N)
    assert snr(clean, clean + 0.5 * noise) - snr(clean, clean + noise) == \
           pytest.approx(20.0 * math.log10(2.0))


# --- the conventions settled from the literature -------------------------

def test_the_clean_signal_is_in_the_denominator():
    """
    Blanco-Velasco et al. 2005, Chiang et al. 2019, Bing et al. 2020.

    Equation (14) of Wang et al. 2023 puts the denoised signal there instead, which is a
    minority reading; the two disagree whenever the estimate differs in energy from the
    clean signal, and this test fixes which one is used here.
    """
    clean = np.array([2.0, 2.0, 2.0, 2.0])
    estimate = np.array([1.0, 1.0, 1.0, 1.0])

    with_clean = 100.0 * math.sqrt(4.0 / 16.0)
    with_estimate = 100.0 * math.sqrt(4.0 / 4.0)
    assert prd(clean, estimate) == pytest.approx(with_clean)
    assert prd(clean, estimate) != pytest.approx(with_estimate)


def test_the_raw_difference_moves_with_a_constant_offset():
    """
    Blanco-Velasco et al.: the raw form depends on the direct current level.

    Which is the reason it is not reported alone.
    """
    clean = beats()
    estimate = clean + 0.05 * np.random.default_rng(0).standard_normal(N)
    assert prd(clean + 5.0, estimate + 5.0) != pytest.approx(prd(clean, estimate), rel=0.1)


def test_the_mean_removed_difference_does_not():
    clean = beats()
    estimate = clean + 0.05 * np.random.default_rng(0).standard_normal(N)
    shifted = prd(clean + 5.0, estimate + 5.0, remove_mean=True)
    assert shifted == pytest.approx(prd(clean, estimate, remove_mean=True), rel=1e-9)


def test_the_improvement_falls_as_the_input_ratio_rises():
    """
    Chiang et al. 2019: the improvement is inversely related to the input ratio.

    The same method scores a larger improvement at a low ratio, so a table of improvements
    alone would rank methods by the ratio they were tested at. This is why the output
    ratio is reported next to it.
    """
    clean = beats()
    noise = np.random.default_rng(0).standard_normal(N)

    # metoda o stalym poziomie residuum: czysci do wlasnej podlogi i nizej nie zejdzie,
    # co jest zachowaniem kazdego realnego filtru
    estimate = clean + 0.2 * noise

    gains, inputs = [], []
    for level in (2.0, 0.5, 0.1):
        noisy = clean + level * noise
        gains.append(snr_improvement(clean, noisy, estimate))
        inputs.append(snr(clean, noisy))

    assert inputs == sorted(inputs)
    assert gains == sorted(gains, reverse=True)
    assert gains[-1] < 0.0


def test_the_output_ratio_separates_what_the_improvement_cannot():
    """Two methods with the same gain from different starting points."""
    clean = beats()
    noise = np.random.default_rng(0).standard_normal(N)

    weak = snr(clean, clean + 0.5 * 2.0 * noise)
    strong = snr(clean, clean + 0.5 * 0.1 * noise)
    assert strong > weak + 10.0


# --- edge cases ----------------------------------------------------------

def test_a_perfect_reconstruction_gives_an_infinite_ratio():
    clean = beats()
    assert snr(clean, clean) == float('inf')
    assert prd(clean, clean) == 0.0


def test_a_reference_of_zeros_has_no_ratio():
    """Nothing to be a ratio of; reported rather than silently divided by zero."""
    zeros = np.zeros(N)
    assert math.isnan(snr(zeros, zeros + 0.1))
    assert math.isnan(prd(zeros, zeros + 0.1))


def test_a_constant_reference_has_a_ratio_but_no_mean_removed_one():
    """
    A constant signal carries power, so the raw form is defined and the other is not.

    Exactly the distinction the two forms exist to make: after the mean is removed there
    is nothing left to compare against.
    """
    constant = np.full(N, 2.0)
    assert np.isfinite(snr(constant, constant + 0.1))
    assert np.isfinite(prd(constant, constant + 0.1))
    assert math.isnan(prd(constant, constant + 0.1, remove_mean=True))


def test_a_constant_waveform_has_no_correlation():
    assert math.isnan(correlation(np.full(N, 2.0), beats()))


def test_an_empty_waveform_is_rejected():
    with pytest.raises(ValueError, match='empty'):
        snr(np.array([]), np.array([]))


def test_waveforms_of_unequal_length_are_trimmed():
    clean = beats()
    assert np.isfinite(snr(clean, clean[:1000] + 0.01))


# --- the span ------------------------------------------------------------

def test_the_span_restricts_the_comparison():
    """
    Which is what the cardiac cycle architecture needs.

    It reconstructs from the first cycle to the last and returns the input unchanged
    outside them, so a comparison over the whole window mixes filtered and unfiltered.
    """
    clean = beats()
    estimate = clean.copy()
    estimate[:500] += 5.0

    assert snr(clean, estimate, span=(500, N)) == float('inf')
    assert np.isfinite(snr(clean, estimate))


def test_a_span_outside_the_waveform_is_rejected():
    clean = beats()
    with pytest.raises(ValueError, match='outside'):
        snr(clean, clean, span=(0, N + 10))


# --- the whole set -------------------------------------------------------

def test_every_declared_field_is_produced():
    clean = beats()
    noise = 0.2 * np.random.default_rng(0).standard_normal(N)
    metrics = reference_metrics(clean, clean + noise, clean + 0.4 * noise)
    assert set(metrics) == set(METRIC_FIELDS)


def test_the_set_agrees_with_the_individual_functions():
    clean = beats()
    noise = 0.2 * np.random.default_rng(0).standard_normal(N)
    noisy, estimate = clean + noise, clean + 0.4 * noise

    metrics = reference_metrics(clean, noisy, estimate)
    assert metrics['snr_out'] == pytest.approx(snr(clean, estimate))
    assert metrics['snr_in'] == pytest.approx(snr(clean, noisy))
    assert metrics['prd1'] == pytest.approx(prd(clean, estimate, remove_mean=True))
    assert metrics['mae'] == pytest.approx(mean_absolute_error(clean, estimate))


def test_a_better_estimate_scores_better_on_every_metric():
    clean = beats()
    noise = 0.3 * np.random.default_rng(0).standard_normal(N)
    good = reference_metrics(clean, clean + noise, clean + 0.2 * noise)
    poor = reference_metrics(clean, clean + noise, clean + 0.8 * noise)

    assert good['snr_out'] > poor['snr_out']
    assert good['prd'] < poor['prd']
    assert good['rmse'] < poor['rmse']
    assert good['mae'] < poor['mae']
    assert good['correlation'] > poor['correlation']


def test_correlation_sees_a_lost_shape_where_the_ratios_see_a_lost_amplitude():
    """
    Halving the amplitude and destroying the shape are different failures.

    The ratio metrics score them similarly; the correlation separates them.
    """
    clean = beats()
    scaled = 0.5 * clean
    shuffled = np.random.default_rng(0).permutation(clean)

    assert correlation(clean, scaled) == pytest.approx(1.0)
    assert abs(correlation(clean, shuffled)) < 0.2


# --- aggregation ---------------------------------------------------------

def test_the_aggregate_ignores_what_is_not_finite():
    """
    One perfect window would otherwise carry an infinite ratio into the average.

    A single one of those turns a whole column into infinity, which reads as a broken
    pipeline rather than as a method that worked.
    """
    rows = [{'snr_out': 10.0}, {'snr_out': 20.0}, {'snr_out': float('inf')},
            {'snr_out': float('nan')}]
    summary = aggregate(rows, fields=('snr_out',))['snr_out']

    assert summary['mean'] == pytest.approx(15.0)
    assert summary['n'] == 2
    assert summary['n_dropped'] == 2


def test_the_aggregate_reports_the_spread_and_the_median():
    rows = [{'snr_out': value} for value in (1.0, 2.0, 3.0, 4.0)]
    summary = aggregate(rows, fields=('snr_out',))['snr_out']
    assert summary['median'] == pytest.approx(2.5)
    assert summary['std'] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_a_method_that_produced_nothing_usable_cannot_hide():
    rows = [{'snr_out': float('nan')} for _ in range(10)] + [{'snr_out': 30.0}]
    summary = aggregate(rows, fields=('snr_out',))['snr_out']
    assert summary['mean'] == pytest.approx(30.0)
    assert summary['n'] == 1
    assert summary['n_dropped'] == 10
