"""
Statistics tests.

The first test is the one the module exists for: the same data tested over windows and
tested over patients give opposite answers, and only the second is defensible. Everything
else checks that the corrections and effect sizes behave as their definitions say, using
cases whose answer can be worked out by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.stats import (
    MIN_PATIENTS_FOR_TESTING,
    aggregate_by_patient,
    compare_methods,
    confidence_interval,
    format_comparison,
    friedman,
    holm,
    paired_matrix,
    paired_wilcoxon,
    rank_biserial,
)


def windows(n_patients=6, n_windows=300, effect=0.05, seed=0):
    """
    Two methods differing by a small constant, measured on many windows per patient.

    The between-patient spread is large and the difference small, which is the situation
    where the two units of analysis disagree.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_patients):
        level = rng.normal(10.0, 3.0)
        for window in range(n_windows):
            rows.append({'method': 'a', 'patient': f'p{index}',
                         'snr_out': level + rng.normal(0.0, 1.0)})
            rows.append({'method': 'b', 'patient': f'p{index}',
                         'snr_out': level + effect + rng.normal(0.0, 1.0)})
    return rows


# --- the reason this module exists ---------------------------------------

def test_the_two_units_of_analysis_disagree():
    """
    Over windows the difference is significant; over patients it is not.

    Windows from one recording share a patient, an electrode placement and often the same
    samples, so the effective size of the sample is the number of patients. Testing over
    windows is pseudo-replication and produces significance for a difference of no
    consequence.
    """
    from scipy.stats import wilcoxon

    rows = windows(effect=0.05)
    first = [row['snr_out'] for row in rows if row['method'] == 'a']
    second = [row['snr_out'] for row in rows if row['method'] == 'b']
    by_window = wilcoxon(first, second).pvalue

    aggregated = aggregate_by_patient(rows, 'snr_out')
    matrix, _, _, _ = paired_matrix(aggregated)
    by_patient = paired_wilcoxon(matrix[:, 0], matrix[:, 1])['p_value']

    assert by_window < 0.01
    assert by_patient > 0.05


def test_the_effective_sample_is_the_number_of_patients():
    rows = windows(n_patients=6, n_windows=500)
    aggregated = aggregate_by_patient(rows, 'snr_out')
    matrix, methods, patients, _ = paired_matrix(aggregated)

    assert matrix.shape == (6, 2)
    assert len(patients) == 6
    assert methods == ['a', 'b']


# --- aggregation ---------------------------------------------------------

def test_the_aggregate_is_the_mean_over_the_windows_of_a_patient():
    rows = [{'method': 'a', 'patient': 'p0', 'snr_out': value} for value in (1.0, 2.0, 6.0)]
    assert aggregate_by_patient(rows, 'snr_out')['a']['p0'] == pytest.approx(3.0)


def test_the_median_is_available_for_a_skewed_metric():
    rows = [{'method': 'a', 'patient': 'p0', 'snr_out': value}
            for value in (1.0, 2.0, 100.0)]
    aggregated = aggregate_by_patient(rows, 'snr_out', statistic='median')
    assert aggregated['a']['p0'] == pytest.approx(2.0)


def test_values_that_are_not_finite_are_left_out():
    """A perfect window gives an infinite ratio and would swallow the patient's mean."""
    rows = [{'method': 'a', 'patient': 'p0', 'snr_out': value}
            for value in (1.0, 3.0, float('inf'), float('nan'))]
    assert aggregate_by_patient(rows, 'snr_out')['a']['p0'] == pytest.approx(2.0)


def test_an_unknown_statistic_is_rejected():
    with pytest.raises(ValueError, match='mean'):
        aggregate_by_patient([], 'snr_out', statistic='geometric')


def test_a_patient_missing_from_one_method_is_dropped_from_all():
    """A repeated measures test needs every method on every patient."""
    aggregated = {'a': {'p0': 1.0, 'p1': 2.0, 'p2': 3.0}, 'b': {'p0': 1.5, 'p1': 2.5}}
    matrix, methods, patients, dropped = paired_matrix(aggregated)

    assert patients == ['p0', 'p1']
    assert dropped == ['p2']
    assert matrix.shape == (2, 2)


def test_a_comparison_with_no_common_patient_is_rejected():
    with pytest.raises(ValueError, match='every method'):
        paired_matrix({'a': {'p0': 1.0}, 'b': {'p1': 2.0}})


# --- omnibus -------------------------------------------------------------

def test_the_omnibus_finds_a_difference_that_is_there():
    rng = np.random.default_rng(0)
    base = rng.normal(10.0, 2.0, size=(12, 1))
    matrix = base + np.array([0.0, 1.5, 3.0]) + 0.1 * rng.normal(size=(12, 3))
    assert friedman(matrix)['p_value'] < 0.01


def test_the_omnibus_finds_nothing_where_there_is_nothing():
    rng = np.random.default_rng(0)
    matrix = rng.normal(10.0, 2.0, size=(12, 3))
    assert friedman(matrix)['p_value'] > 0.05


def test_the_concordance_effect_size_lies_between_zero_and_one():
    rng = np.random.default_rng(0)
    base = rng.normal(10.0, 2.0, size=(12, 1))
    perfect = friedman(base + np.array([0.0, 1.5, 3.0]))
    none = friedman(rng.normal(10.0, 2.0, size=(12, 3)))

    assert 0.0 <= none['kendalls_w'] <= 1.0
    assert perfect['kendalls_w'] > none['kendalls_w']
    assert perfect['kendalls_w'] == pytest.approx(1.0, abs=1e-9)


def test_too_few_patients_raises_a_warning_rather_than_a_number_alone():
    """
    Five held out patients against seventeen methods leaves the test almost no power.

    The warning is what turns that into a decision to read the effect sizes instead.
    """
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(MIN_PATIENTS_FOR_TESTING - 3, 4))
    with pytest.warns(UserWarning, match='little power'):
        friedman(matrix)


def test_an_omnibus_needs_three_methods():
    with pytest.raises(ValueError, match='three methods'):
        friedman(np.random.default_rng(0).normal(size=(10, 2)))


# --- paired contrasts and their effect size ------------------------------

def test_the_signed_rank_test_finds_a_consistent_difference():
    rng = np.random.default_rng(0)
    first = rng.normal(10.0, 2.0, size=15)
    outcome = paired_wilcoxon(first + 1.0, first)
    assert outcome['p_value'] < 0.01
    assert outcome['median_difference'] == pytest.approx(1.0)


def test_the_effect_size_is_one_when_every_pair_agrees():
    first = np.arange(10, dtype=float)
    assert rank_biserial(first + 1.0, first) == pytest.approx(1.0)
    assert rank_biserial(first - 1.0, first) == pytest.approx(-1.0)


def test_the_effect_size_is_zero_for_interchangeable_methods():
    values = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    assert rank_biserial(values, np.zeros_like(values)) == pytest.approx(0.0)


def test_identical_methods_give_no_difference_and_no_effect():
    values = np.arange(10, dtype=float)
    outcome = paired_wilcoxon(values, values)
    assert outcome['p_value'] == 1.0
    assert outcome['rank_biserial'] == 0.0


def test_unpaired_samples_are_rejected():
    with pytest.raises(ValueError, match='unpaired'):
        paired_wilcoxon(np.zeros(5), np.zeros(6))


# --- correction ----------------------------------------------------------

def test_holm_leaves_the_smallest_p_value_multiplied_by_the_family_size():
    adjusted = holm([0.01, 0.02, 0.03, 0.04])['adjusted']
    assert adjusted[0] == pytest.approx(0.04)


def test_holm_weights_each_p_value_by_what_remains():
    """Which is what makes it reject more than Bonferroni at the same guarantee."""
    p_values = [0.01, 0.02, 0.03, 0.04]
    adjusted = holm(p_values)['adjusted']
    bonferroni = np.minimum(1.0, np.asarray(p_values) * len(p_values))

    assert np.all(adjusted <= bonferroni + 1e-12)
    assert np.any(adjusted < bonferroni)


def test_holm_never_lowers_an_adjusted_value_below_an_earlier_one():
    """Monotone by construction, so the ordering of the contrasts is preserved."""
    p_values = np.array([0.001, 0.9, 0.002, 0.5])
    adjusted = holm(p_values)['adjusted']
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_holm_never_returns_more_than_one():
    assert np.all(holm([0.5, 0.6, 0.7, 0.8])['adjusted'] <= 1.0)


def test_holm_rejects_at_the_stated_level():
    outcome = holm([0.001, 0.20, 0.30], alpha=0.05)
    assert outcome['rejected'][0]
    assert not outcome['rejected'][1]


def test_an_impossible_p_value_is_rejected():
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        holm([0.5, 1.5])


def test_an_empty_family_needs_no_correction():
    assert holm([])['adjusted'].size == 0


# --- intervals -----------------------------------------------------------

def test_the_interval_brackets_the_mean():
    values = np.random.default_rng(0).normal(5.0, 1.0, size=40)
    interval = confidence_interval(values)
    assert interval['low'] < interval['mean'] < interval['high']


def test_a_larger_sample_gives_a_narrower_interval():
    rng = np.random.default_rng(0)
    small = confidence_interval(rng.normal(5.0, 1.0, size=8))
    large = confidence_interval(rng.normal(5.0, 1.0, size=200))
    assert (large['high'] - large['low']) < (small['high'] - small['low'])


def test_the_decibel_route_differs_from_the_linear_one():
    """
    The logarithm is not linear, so the two are not the same interval.

    The literature surveyed does not settle which is correct, so the route is an argument
    and the choice is recorded rather than assumed.
    """
    values = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    on_decibels = confidence_interval(values, scale='linear', n_bootstrap=2000)
    through_linear = confidence_interval(values, scale='decibel', n_bootstrap=2000)

    assert through_linear['mean'] != pytest.approx(on_decibels['mean'], rel=1e-3)
    assert through_linear['scale'] == 'decibel'


def test_the_decibel_route_averages_powers_not_logarithms():
    """Ten decibels is ten times the power of zero; their mean is 10 log10(5.5)."""
    interval = confidence_interval(np.array([0.0, 10.0]), scale='decibel',
                                   n_bootstrap=100)
    assert interval['mean'] == pytest.approx(10.0 * np.log10(5.5), abs=1e-9)


def test_an_unknown_scale_is_rejected():
    with pytest.raises(ValueError, match='linear'):
        confidence_interval([1.0, 2.0], scale='logarithmic')


def test_a_single_value_has_no_interval():
    interval = confidence_interval([3.0])
    assert interval['mean'] == 3.0
    assert np.isnan(interval['low'])


# --- the whole workflow --------------------------------------------------

def test_the_workflow_runs_from_rows_to_a_corrected_table():
    rng = np.random.default_rng(0)
    rows = []
    for index in range(12):
        level = rng.normal(10.0, 2.0)
        for method, gain in (('poor', 0.0), ('fair', 2.0), ('good', 4.0)):
            for _ in range(50):
                rows.append({'method': method, 'patient': f'p{index}',
                             'snr_out': level + gain + rng.normal(0.0, 0.5)})

    comparison = compare_methods(rows, 'snr_out')

    assert comparison['methods'] == ['fair', 'good', 'poor']
    assert len(comparison['patients']) == 12
    assert comparison['omnibus']['p_value'] < 0.01
    assert len(comparison['contrasts']) == 3
    assert all('p_adjusted' in contrast for contrast in comparison['contrasts'])
    assert set(comparison['intervals']) == set(comparison['methods'])


def test_the_correction_is_applied_across_the_whole_family():
    rng = np.random.default_rng(0)
    rows = []
    for index in range(12):
        level = rng.normal(10.0, 2.0)
        for method, gain in (('a', 0.0), ('b', 1.0), ('c', 2.0), ('d', 3.0)):
            for _ in range(30):
                rows.append({'method': method, 'patient': f'p{index}',
                             'snr_out': level + gain + rng.normal(0.0, 0.4)})

    comparison = compare_methods(rows, 'snr_out')
    assert len(comparison['contrasts']) == 6
    for contrast in comparison['contrasts']:
        assert contrast['p_adjusted'] >= contrast['p_value']


def test_the_summary_names_the_methods_and_the_correction():
    rng = np.random.default_rng(0)
    rows = []
    for index in range(10):
        level = rng.normal(10.0, 2.0)
        for method, gain in (('poor', 0.0), ('fair', 2.0), ('good', 4.0)):
            for _ in range(20):
                rows.append({'method': method, 'patient': f'p{index}',
                             'snr_out': level + gain + rng.normal(0.0, 0.5)})

    text = format_comparison(compare_methods(rows, 'snr_out'))
    for token in ('Friedman', 'holm', 'poor', 'fair', 'good', 'W Kendalla'):
        assert token in text
