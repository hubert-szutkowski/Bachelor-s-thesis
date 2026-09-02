"""
Edge cases of the metric panel.

A metric is a measuring instrument, and an instrument is characterised by what it does at
the ends of its range, not in the middle. Three inputs decide it here: a residual of
exactly zero, a reference with no alternating component, and a reference of no energy at
all. Each has one defensible answer and several plausible wrong ones, and the wrong ones
are quiet: they return a finite number that reads like an ordinary result.

The convention follows Berntsen & Brandt 2021 and Moshrefi et al. 2021, who settle the
ratio logic and the treatment of a vanishing difference; the electrocardiogram literature
does not legislate these cases, so the rule is declared rather than cited.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.metrics_reference import (
    DEGENERATE_TOLERANCE,
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
N = 1024


def wave(n=N, fs=FS, hz=1.2):
    return np.sin(2 * np.pi * hz * np.arange(n) / fs)


def constant(value=0.7, n=N):
    return np.full(n, value)


# --- residual of exactly zero --------------------------------------------

def test_a_perfect_reconstruction_gives_an_infinite_ratio():
    """
    Not a number would be wrong: the answer is known, it is simply unbounded.

    An aggregate that meets this value has a problem, and the aggregate is where that
    should surface, not here.
    """
    clean = wave()
    assert snr(clean, clean) == float('inf')


def test_a_perfect_reconstruction_gives_zero_percent_difference():
    """Zero error, not an undefined one. The difference from the reference vanishes."""
    clean = wave()
    assert prd(clean, clean) == 0.0
    assert prd(clean, clean, remove_mean=True) == 0.0


def test_a_perfect_reconstruction_gives_zero_on_every_error():
    clean = wave()
    assert mean_squared_error(clean, clean) == 0.0
    assert root_mean_squared_error(clean, clean) == 0.0
    assert mean_absolute_error(clean, clean) == 0.0
    assert correlation(clean, clean) == pytest.approx(1.0)


def test_the_zero_of_the_difference_is_not_a_rounding_artefact():
    """A residual below the scale of the signal still is not exactly zero."""
    clean = wave()
    almost = clean + 1e-9
    assert prd(clean, almost) > 0.0


# --- a reference with no alternating component ---------------------------

def test_a_constant_reference_leaves_the_mean_removed_form_undefined():
    """
    The case the module has to special case.

    Subtracting the mean from a constant leaves a residue of the order of machine epsilon.
    It is not zero, so a test against zero passes it through, and the reciprocal of that
    residue is a finite number with no meaning. Before the guard this returned 1.4e31.
    """
    reference = constant()
    estimate = reference + 0.1 * np.random.default_rng(0).standard_normal(N)

    assert math.isnan(prd(reference, estimate, remove_mean=True))


def test_the_raw_form_survives_a_constant_reference_because_it_uses_total_energy():
    """The two conventions disagree here, which is the reason both are reported."""
    reference = constant()
    estimate = reference + 0.1 * np.random.default_rng(0).standard_normal(N)

    value = prd(reference, estimate)
    assert math.isfinite(value) and value > 0.0


def test_the_ratio_on_a_constant_reference_is_finite_and_declared_as_such():
    """
    Computed on total energy, so it does not collapse.

    The number means nothing, since a constant is not an electrocardiogram, but the
    convention is stated in the docstring rather than left for the reader to infer from a
    value that looks ordinary.
    """
    reference = constant()
    estimate = reference + 0.1 * np.random.default_rng(0).standard_normal(N)
    assert math.isfinite(snr(reference, estimate))


def test_a_constant_reference_matched_exactly_counts_as_perfect():
    """Indeterminate in the mathematics, perfect agreement by convention."""
    reference = constant()
    assert snr(reference, reference) == float('inf')
    assert prd(reference, reference) == 0.0


def test_a_constant_reference_has_no_correlation_to_measure():
    reference = constant()
    estimate = reference + 0.1 * np.random.default_rng(0).standard_normal(N)
    assert math.isnan(correlation(reference, estimate))


@pytest.mark.parametrize('level', [0.0, 1e-6, 0.7, 1e3])
def test_the_guard_holds_at_any_offset(level):
    reference = constant(level)
    estimate = reference + 0.1 * np.random.default_rng(0).standard_normal(N)
    assert math.isnan(prd(reference, estimate, remove_mean=True))


def test_the_guard_is_relative_so_the_unit_does_not_change_the_verdict():
    """
    Millivolts or microvolts must give the same answer.

    An absolute threshold would call a genuine microvolt signal degenerate, which is the
    failure mode of writing the tolerance as a fixed number of units.
    """
    clean = wave()
    for scale in (1.0, 1e-3, 1e-6):
        assert prd(clean * scale, clean * scale, remove_mean=True) == 0.0
        assert math.isfinite(prd(clean * scale, clean * scale * 0.9, remove_mean=True))


def test_the_declared_tolerance_is_far_below_any_physical_signal():
    assert DEGENERATE_TOLERANCE == 1e-12


# --- a reference of no energy at all -------------------------------------

def test_a_reference_of_zeros_leaves_every_ratio_undefined():
    """No signal means no signal to noise ratio, whatever the estimate did."""
    zeros = np.zeros(N)
    noise = 0.1 * np.random.default_rng(0).standard_normal(N)

    assert math.isnan(snr(zeros, noise))
    assert math.isnan(prd(zeros, noise))
    assert math.isnan(snr(zeros, zeros))


def test_the_errors_stay_defined_on_a_reference_of_zeros():
    """A difference needs no denominator, so it is still a number."""
    zeros = np.zeros(N)
    noise = 0.1 * np.random.default_rng(0).standard_normal(N)

    assert mean_squared_error(zeros, noise) > 0.0
    assert mean_squared_error(zeros, zeros) == 0.0


# --- estimates that failed in an informative way -------------------------

def test_an_inverted_estimate_is_scored_as_the_failure_it_is():
    """Two hundred percent, and a correlation of minus one that says what happened."""
    clean = wave()
    assert prd(clean, -clean) == pytest.approx(200.0)
    assert correlation(clean, -clean) == pytest.approx(-1.0)
    assert snr(clean, -clean) == pytest.approx(-6.021, abs=0.01)


def test_an_estimate_that_output_a_flat_line_loses_the_whole_signal():
    clean = wave()
    flat = np.full(N, clean.mean())
    assert prd(clean, flat, remove_mean=True) == pytest.approx(100.0)
    assert math.isnan(correlation(clean, flat))


def test_an_estimate_worse_than_the_input_gives_a_negative_improvement():
    """Which must not be clipped: a method that made things worse has to say so."""
    clean = wave()
    rng = np.random.default_rng(0)
    noisy = clean + 0.1 * rng.standard_normal(N)
    worse = clean + 0.5 * rng.standard_normal(N)

    assert snr_improvement(clean, noisy, worse) < 0.0


# --- lengths and spans ---------------------------------------------------

def test_an_empty_waveform_is_refused_rather_than_scored():
    with pytest.raises(ValueError, match='empty waveform'):
        snr(np.array([]), np.array([]))


def test_waveforms_of_different_lengths_are_compared_over_the_shorter_one():
    clean = wave()
    assert snr(clean, clean[:512]) == float('inf')


def test_a_single_sample_is_still_a_waveform():
    assert snr(np.array([1.0]), np.array([1.0])) == float('inf')
    assert math.isnan(correlation(np.array([1.0]), np.array([1.0])))


def test_a_span_outside_the_waveform_is_refused():
    clean = wave()
    with pytest.raises(ValueError, match='lies outside'):
        snr(clean, clean, span=(0, N + 1))


# --- what the aggregate does with them -----------------------------------

def test_a_non_finite_value_is_dropped_from_the_mean_and_counted():
    """
    Dropping it silently would be the dangerous version.

    A perfect reconstruction among ordinary results is a fact about the run, and the count
    of what was excluded travels with the aggregate so that it cannot be read as if every
    window contributed.
    """
    rows = [{'snr_out': 5.0}, {'snr_out': float('inf')}, {'snr_out': 7.0}]
    summary = aggregate(rows, fields=('snr_out',))['snr_out']

    assert summary['mean'] == pytest.approx(6.0)
    assert summary['n'] == 2
    assert summary['n_dropped'] == 1


def test_an_aggregate_of_nothing_finite_reports_no_value_rather_than_zero():
    rows = [{'snr_out': float('nan')}, {'snr_out': float('inf')}]
    summary = aggregate(rows, fields=('snr_out',))['snr_out']

    assert summary['n'] == 0
    assert math.isnan(summary['mean'])


def test_the_panel_returns_every_declared_field_on_a_degenerate_input():
    """
    A method that failed must still occupy its row in the table.

    Raising here would drop the worst method from the comparison instead of showing it,
    which is the same failure the detection score was fixed for.
    """
    from analysis.metrics_reference import METRIC_FIELDS

    reference = constant()
    noisy = reference + 0.1 * np.random.default_rng(0).standard_normal(N)
    row = reference_metrics(reference, noisy, noisy)

    assert set(METRIC_FIELDS) <= set(row)
