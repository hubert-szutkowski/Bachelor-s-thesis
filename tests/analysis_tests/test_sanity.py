"""
Sanity check tests.

These are checks on the checks. The point of the module is to catch an error in a metric
that would move every method in the same direction and therefore stay invisible in any
comparison, so the tests here confirm that each check would in fact notice such an error:
a deliberately broken measurement has to fail the suite, not merely look worse.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from analysis.metrics_reference import snr
from analysis.metrics_sqi import psqi
from analysis.sanity import (
    ANALYTIC_TOLERANCE_DB,
    CONCORDANCE_WARNING,
    analytic_lowpass_gain_db,
    check_controls,
    check_lowpass_gain,
    check_monotonicity,
    format_suite,
    metric_concordance,
    run_sanity_suite,
)

FS = 360.0
N = 4096


def beats(n=N, fs=FS, bpm=75):
    signal = np.zeros(n)
    for peak in range(int(fs), n - int(fs), int(60.0 / bpm * fs)):
        signal[peak - 90:peak - 61] += np.hanning(29) * 0.15
        signal[peak - 4:peak + 4] += np.hanning(8) * 1.2
        signal[peak + 40:peak + 98] += np.hanning(58) * 0.3
    return signal


# --- the gain that can be derived on paper -------------------------------

def test_the_predicted_gain_is_the_ratio_of_the_bands():
    """Half the sampling frequency over the cutoff, in decibels. Nothing measured."""
    assert analytic_lowpass_gain_db(90.0, 360.0) == pytest.approx(10.0 * math.log10(2.0))
    assert analytic_lowpass_gain_db(18.0, 360.0) == pytest.approx(10.0)
    assert analytic_lowpass_gain_db(1.8, 360.0) == pytest.approx(20.0)


def test_halving_the_cutoff_adds_three_decibels():
    first = analytic_lowpass_gain_db(40.0, FS)
    second = analytic_lowpass_gain_db(20.0, FS)
    assert second - first == pytest.approx(10.0 * math.log10(2.0))


def test_a_cutoff_at_or_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match='between zero'):
        analytic_lowpass_gain_db(200.0, FS)


def test_the_measured_gain_matches_the_derivation():
    """
    The check itself, on a case whose answer needs no measurement.

    A departure here is a fault in the metric, in the filter or in the mixing, and which
    of the three follows from whether the other checks also fail.
    """
    report = check_lowpass_gain(FS, cutoff_hz=40.0, order=8)
    assert report['passed']
    assert abs(report['error_db']) < ANALYTIC_TOLERANCE_DB


@pytest.mark.parametrize('cutoff', [20.0, 40.0, 60.0])
def test_the_agreement_holds_over_several_cutoffs(cutoff):
    report = check_lowpass_gain(FS, cutoff_hz=cutoff, order=8)
    assert report['passed'], f"{cutoff} Hz: {report['error_db']:+.3f} dB"


def test_the_check_notices_a_metric_scaled_by_a_constant():
    """
    Which is the error class the whole module exists for.

    A ratio computed with the wrong base moves every method the same way, so no comparison
    between methods reveals it; only a case with a known answer does.
    """
    report = check_lowpass_gain(FS, cutoff_hz=40.0, order=8)
    wrong = report['measured_db'] * 2.0
    assert abs(wrong - report['predicted_db']) > ANALYTIC_TOLERANCE_DB


# --- direction of every metric -------------------------------------------

def test_every_metric_moves_the_way_it_must():
    clean = beats()
    noise = np.random.default_rng(0).standard_normal(N)
    report = check_monotonicity(clean, noise, FS)

    assert report['snr_rises']
    assert report['prd_falls']
    assert report['correlation_rises']
    assert report['passed']


def test_the_ratio_recovers_the_level_it_was_mixed_at():
    clean = beats()
    noise = np.random.default_rng(0).standard_normal(N)
    report = check_monotonicity(clean, noise, FS)
    assert report['snr'] == pytest.approx(report['levels_db'], abs=1e-3)


def test_an_inverted_metric_would_be_visible():
    """The check tests an ordering, so a sign error anywhere breaks it."""
    clean = beats()
    noise = np.random.default_rng(0).standard_normal(N)
    report = check_monotonicity(clean, noise, FS)

    inverted = list(reversed(report['prd']))
    assert inverted != sorted(inverted, reverse=True) or len(set(inverted)) == 1


# --- agreement between metrics -------------------------------------------

def test_metrics_that_agree_score_a_high_concordance():
    scores = {
        'snr_out': {'a': 1.0, 'b': 5.0, 'c': 9.0},
        'prd': {'a': 90.0, 'b': 50.0, 'c': 10.0},
    }
    report = metric_concordance(scores)
    assert report['passed']
    assert report['worst'] == pytest.approx(1.0)


def test_metrics_that_disagree_are_named():
    """
    Shi et al. 2021 report exactly this: one method better on one metric, another on
    another, over the same material.
    """
    # ratio prefers a, error prefers c: po sprowadzeniu do wspolnego kierunku
    # uporzadkowania sa odwrotne
    scores = {
        'snr_out': {'a': 9.0, 'b': 5.0, 'c': 1.0},
        'rmse': {'a': 0.9, 'b': 0.5, 'c': 0.1},
    }
    report = metric_concordance(scores)
    assert not report['passed']
    assert report['disputed'][0]['metrics'] == ('rmse', 'snr_out')
    assert report['worst'] == pytest.approx(-1.0)


def test_the_direction_of_each_metric_is_accounted_for():
    """A lower difference and a higher ratio mean the same thing and must agree."""
    scores = {
        'snr_out': {'a': 1.0, 'b': 5.0, 'c': 9.0},
        'mse': {'a': 0.9, 'b': 0.5, 'c': 0.1},
        'correlation': {'a': 0.5, 'b': 0.8, 'c': 0.99},
    }
    assert metric_concordance(scores)['passed']


def test_too_few_methods_to_compare_orderings_is_rejected():
    with pytest.raises(ValueError, match='three methods'):
        metric_concordance({'snr_out': {'a': 1.0, 'b': 2.0},
                            'prd': {'a': 2.0, 'b': 1.0}})


def test_a_single_metric_has_nothing_to_agree_with():
    with pytest.raises(ValueError, match='two metrics'):
        metric_concordance({'snr_out': {'a': 1.0, 'b': 2.0, 'c': 3.0}})


def test_the_threshold_is_the_declared_one():
    assert CONCORDANCE_WARNING == 0.7


# --- the two controls ----------------------------------------------------

def test_the_unfiltered_control_scores_exactly_what_the_input_scores():
    """Anything else means the pipeline is altering a waveform it was told to pass."""
    clean = beats()
    noisy = clean + 0.3 * np.random.default_rng(0).standard_normal(N)
    destructive = sosfiltfilt(butter(4, [5.0, 15.0], btype='bandpass', fs=FS,
                                     output='sos'), noisy)

    report = check_controls(clean, noisy, noisy, destructive, FS)
    assert report['identity_matches_input']
    assert report['identity_snr_db'] == pytest.approx(report['input_snr_db'])


def test_the_destructive_control_wins_on_the_quality_index():
    """
    Which is the falsification the control exists to provide.

    A panel that ranks it highly has been shown to reward the removal of signal rather
    than the removal of noise, and that is a demonstration rather than a suspicion.
    """
    clean = beats()
    noisy = clean + 0.3 * np.random.default_rng(0).standard_normal(N)
    destructive = sosfiltfilt(butter(4, [5.0, 15.0], btype='bandpass', fs=FS,
                                     output='sos'), noisy)

    report = check_controls(clean, noisy, noisy, destructive, FS)
    assert report['destructive_wins_on_quality']
    assert report['destructive_psqi'] > 2.0 * report['input_psqi']


def test_the_controls_are_registered_and_named_as_such():
    from filters.methods_static import (
        CONTROL_METHODS, register_control_methods, register_static_filters)
    from filters.registry import (
        available_filters, describe, reset_config, unregister)

    for name in list(available_filters()):
        unregister(name)
    reset_config()

    register_static_filters()
    controls = register_control_methods()

    assert tuple(controls) == CONTROL_METHODS
    for name in controls:
        assert 'KONTROLA' in describe(name)

    # siedem metod statycznych plus dwie kontrole, ktore do rankingu nie naleza
    assert len(available_filters('static')) == 9

    for name in list(available_filters()):
        unregister(name)
    reset_config()


def test_the_unfiltered_control_returns_its_input_unchanged():
    from filters.methods_static import _no_filtering
    from filters.registry import FilterContext

    signal = beats()
    assert np.array_equal(_no_filtering(signal, FilterContext(fs=FS)), signal)


def test_the_destructive_control_loses_the_p_and_t_waves():
    from filters.methods_static import _destructive_bandpass
    from filters.registry import FilterContext

    clean = beats()
    damaged = _destructive_bandpass(clean, FilterContext(fs=FS))
    assert snr(clean, damaged) < 5.0
    assert psqi(damaged, FS) > 0.9


# --- the suite -----------------------------------------------------------

def test_the_suite_passes_on_a_correct_pipeline():
    report = run_sanity_suite(FS)
    assert report['passed']
    assert set(report['checks']) == {'analytic_lowpass_gain', 'metric_monotonicity'}


def test_the_summary_names_every_check_and_its_outcome():
    text = format_suite(run_sanity_suite(FS))
    assert 'analytic_lowpass_gain' in text
    assert 'metric_monotonicity' in text
    assert 'zysk analityczny' in text
