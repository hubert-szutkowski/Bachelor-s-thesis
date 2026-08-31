"""
Static method tests.

Each method is exercised through `apply_filter`, which is how the evaluation will call it,
so what is tested is the pair of adapter and implementation rather than either alone.

Two properties get more attention than the rest. Window lengths configured in
milliseconds must resolve the same band at any rate, which is what keeps the parameters
independent of the recording equipment. And wavelet baseline removal must refuse a window
too short to reach below the baseline band: the decomposition depth a window admits is bounded by its length,
and a shallow decomposition removes the P and T waves along with the drift while returning
a waveform that still looks like an ECG.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from filters.methods_static import baseline_cut_hz, register_static_filters
from filters.registry import (
    FilterContext,
    apply_filter,
    available_filters,
    configured_params,
    load_config,
    reset_config,
    unregister,
)

FS = 360.0
WIDTH = 4096
CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'filters.yaml'


@pytest.fixture(autouse=True)
def registry():
    for name in list(available_filters()):
        unregister(name)
    reset_config()
    names = register_static_filters()
    load_config(CONFIG)
    yield names
    for name in list(available_filters()):
        unregister(name)
    reset_config()


def ecg(n=WIDTH, fs=FS, bpm=75, seed=0):
    """A beat train with baseline drift, mains interference and broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    signal = np.zeros(n)
    for peak in range(int(fs), n - int(fs), int(60.0 / bpm * fs)):
        signal[peak - 5:peak + 5] += np.hanning(10) * 1.2
        signal[peak + 20:peak + 60] += np.hanning(40) * 0.25
    drift = 0.4 * np.sin(2 * np.pi * 0.25 * t)
    mains = 0.15 * np.sin(2 * np.pi * 50.0 * t)
    return signal + drift + mains + 0.05 * rng.standard_normal(n), signal


@pytest.fixture
def context():
    return FilterContext(fs=FS)


# --- the whole family ----------------------------------------------------

def test_exactly_seven_static_methods_are_registered(registry):
    assert len(registry) == 7
    assert available_filters('static') == sorted(registry)


def test_every_method_runs_and_returns_the_input_length(registry, context):
    noisy, _ = ecg()
    for name in registry:
        result = apply_filter(name, noisy, context)
        assert result.signal.shape == noisy.shape, name
        assert np.all(np.isfinite(result.signal)), name


def test_every_method_belongs_to_the_static_family(registry, context):
    noisy, _ = ecg()
    for name in registry:
        assert apply_filter(name, noisy, context).family == 'static'


def test_no_method_needs_a_reference(registry, context):
    """
    Static methods must run on MIT-BIH, where no accelerometer exists.

    A static method declaring a need for one would vanish from the synthetic comparison.
    """
    from filters.registry import applicable_filters

    assert applicable_filters(FilterContext(fs=FS))['blocked'] == {}


def test_every_method_leaves_its_input_untouched(registry, context):
    noisy, _ = ecg()
    original = noisy.copy()
    for name in registry:
        apply_filter(name, noisy, context)
    assert np.array_equal(noisy, original)


def test_every_method_reads_its_parameters_from_the_file(registry):
    for name in registry:
        assert configured_params(name), name


# --- the filters do what their names say ---------------------------------

@pytest.mark.parametrize('name', ['fir_bandpass', 'iir_bandpass'])
def test_the_bandpass_filters_remove_the_drift(registry, context, name):
    noisy, clean = ecg()
    filtered = apply_filter(name, noisy, context).signal
    before = float(np.std(noisy - clean))
    after = float(np.std(filtered - clean))
    assert after < 0.5 * before


@pytest.mark.parametrize('name', ['fir_bandpass', 'iir_bandpass'])
def test_the_bandpass_filters_remove_the_mains(registry, context, name):
    from scipy.signal import welch

    noisy, _ = ecg()
    filtered = apply_filter(name, noisy, context).signal
    frequency, before = welch(noisy, fs=FS, nperseg=1024)
    _, after = welch(filtered, fs=FS, nperseg=1024)
    band = (frequency > 45) & (frequency < 55)
    assert after[band].sum() < 0.05 * before[band].sum()


@pytest.mark.parametrize('name', ['fir_bandpass', 'iir_bandpass'])
def test_the_bandpass_filters_do_not_shift_the_beat(registry, context, name):
    """
    Zero phase filtering is what keeps the time domain metrics meaningful.

    A phase shift raises the mean square error even when the noise was removed perfectly.
    """
    noisy, clean = ecg()
    filtered = apply_filter(name, noisy, context).signal
    lags = np.arange(-20, 21)
    scores = [float(np.corrcoef(clean, np.roll(filtered, lag))[0, 1]) for lag in lags]
    assert lags[int(np.argmax(scores))] == 0


def test_the_moving_average_reduces_broadband_noise(registry, context):
    noisy, clean = ecg()
    filtered = apply_filter('moving_average', noisy, context).signal
    assert np.std(np.diff(filtered)) < np.std(np.diff(noisy))


def test_the_moving_median_removes_a_spike(registry, context):
    """A median survives an impulse that would drag a mean with it."""
    noisy, _ = ecg()
    spiked = noisy.copy()
    spiked[2000] += 40.0
    filtered = apply_filter('moving_median', spiked, context).signal
    assert abs(filtered[2000]) < 5.0


def test_wavelet_denoising_lowers_the_broadband_noise(registry, context):
    noisy, clean = ecg()
    filtered = apply_filter('wavelet_denoising', noisy, context).signal
    assert np.std(np.diff(filtered)) < np.std(np.diff(noisy))


def test_wavelet_baseline_removal_flattens_the_drift(registry, context):
    noisy, _ = ecg()
    filtered = apply_filter('wavelet_baseline', noisy, context).signal
    from scipy.signal import welch

    frequency, before = welch(noisy, fs=FS, nperseg=2048)
    _, after = welch(filtered, fs=FS, nperseg=2048)
    band = frequency < 0.5
    assert after[band].sum() < 0.2 * before[band].sum()


def test_emd_returns_a_waveform_of_the_same_length(registry, context):
    noisy, _ = ecg()
    assert apply_filter('emd_denoising', noisy, context).signal.shape == noisy.shape


# --- lengths in milliseconds ---------------------------------------------

def test_a_window_in_milliseconds_is_the_same_filter_at_any_rate(registry):
    """
    Nine samples smooth 25 ms at 360 Hz and 36 ms at 250 Hz.

    Everything here runs at 360 Hz, so this is a guard rather than a present need: a
    length in samples would change the filter at a different rate without saying so.
    """
    from filters.methods_static import _odd_samples

    assert _odd_samples(25.0, 360.0) == 9
    assert _odd_samples(25.0, 250.0) == 7
    assert _odd_samples(25.0, 1000.0) == 25


def test_window_lengths_are_forced_odd(registry):
    from filters.methods_static import _odd_samples

    for milliseconds in (10.0, 25.0, 40.0, 60.0, 100.0):
        assert _odd_samples(milliseconds, FS) % 2 == 1


def test_the_sampling_frequency_comes_from_the_context_not_the_file(registry):
    """A rate in the parameter file could disagree with the recording it is applied to."""
    assert 'fs' not in configured_params('fir_bandpass')
    assert 'fs' not in configured_params('moving_average')


def test_a_lower_rate_still_produces_a_working_filter(registry):
    noisy, _ = ecg(n=2048, fs=250.0)
    result = apply_filter('iir_bandpass', noisy, FilterContext(fs=250.0))
    assert np.all(np.isfinite(result.signal))


# --- the guard on baseline removal ---------------------------------------

def test_the_achievable_cut_is_reported():
    report = baseline_cut_hz(WIDTH, FS, 'db8', 9)
    assert report['max_level'] == 8
    assert report['effective_level'] == 8
    assert report['cut_hz'] == pytest.approx(360.0 / 2 ** 9)


def test_a_short_window_admits_too_few_levels():
    """1024 samples at 360 Hz reach only 2.8 Hz, which is inside the P and T waves."""
    report = baseline_cut_hz(1024, FS, 'db8', 9)
    assert report['max_level'] == 6
    assert report['cut_hz'] > 2.0


def test_baseline_removal_refuses_a_window_that_is_too_short(registry):
    noisy, _ = ecg(n=1024)
    with pytest.raises(ValueError, match='P and T waves'):
        apply_filter('wavelet_baseline', noisy, FilterContext(fs=FS))


def test_the_refusal_can_be_overridden_deliberately(registry):
    """Never by accident: the caller has to name the frequency it accepts."""
    noisy, _ = ecg(n=1024)
    result = apply_filter('wavelet_baseline', noisy, FilterContext(fs=FS), max_cut_hz=5.0)
    assert result.signal.shape == noisy.shape


def test_the_wider_window_is_what_makes_the_method_usable(registry, context):
    """This is why the static filters are measured on 4096 samples and not on 1024."""
    assert baseline_cut_hz(4096, FS, 'db8', 9)['cut_hz'] < 1.0
    assert baseline_cut_hz(1024, FS, 'db8', 9)['cut_hz'] > 1.0


def test_wavelet_denoising_clamps_the_level_instead_of_failing(registry):
    noisy, _ = ecg(n=512)
    result = apply_filter('wavelet_denoising', noisy, FilterContext(fs=FS))
    assert result.signal.shape == noisy.shape


# --- parameters ----------------------------------------------------------

def test_the_shipped_configuration_covers_every_static_method(registry):
    import yaml

    with CONFIG.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    assert set(loaded['static']) == set(registry)


def test_a_call_argument_changes_the_result(registry, context):
    noisy, _ = ecg()
    narrow = apply_filter('iir_bandpass', noisy, context, high_hz=15.0).signal
    wide = apply_filter('iir_bandpass', noisy, context, high_hz=40.0).signal
    assert not np.allclose(narrow, wide)


def test_the_resolved_parameters_are_recorded(registry, context):
    noisy, _ = ecg()
    result = apply_filter('iir_bandpass', noisy, context, order=6)
    assert result.params['order'] == 6
    assert result.params['low_hz'] == 0.5


def test_every_method_reports_a_positive_runtime(registry, context):
    """The cost table of the results chapter is built from these."""
    noisy, _ = ecg()
    for name in registry:
        assert apply_filter(name, noisy, context).elapsed_s > 0.0


# --- what a FIR filter of a given length can actually do -----------------

def test_the_transition_width_follows_the_length():
    """A band edge below this width has no stopband beneath it."""
    from filters.methods_static import fir_transition_hz

    short = fir_transition_hz(101, FS, 'hamming')
    long = fir_transition_hz(613, FS, 'hamming')
    assert short['transition_hz'] == pytest.approx(3.3 * FS / 101, rel=1e-6)
    assert long['transition_hz'] < 2.0
    assert short['transition_hz'] > 10.0


def test_a_length_in_milliseconds_fixes_the_transition_across_rates():
    """
    This is the reason the length is configured in milliseconds.

    A length in coefficients resolves a different band at every rate; a length in
    milliseconds resolves the same one.
    """
    from filters.methods_static import _odd_samples, fir_transition_hz

    widths = [fir_transition_hz(_odd_samples(1700.0, fs), fs)['transition_hz']
              for fs in (250.0, 360.0, 500.0)]
    assert max(widths) - min(widths) < 0.05


def test_a_short_filter_cannot_remove_the_drift(registry, context):
    """
    Measured, not assumed: 101 coefficients leave two thirds of the drift in place.

    The frequency response still shows the requested band, which is what makes this
    failure quiet, and it is the reason FIR is a poor choice for a low band edge.
    """
    noisy, clean = ecg()
    residual = {}
    for milliseconds in (280.0, 1700.0):
        filtered = apply_filter('fir_bandpass', noisy, context,
                                numtaps_ms=milliseconds).signal
        residual[milliseconds] = float(np.std(filtered - clean))

    assert residual[280.0] > 3 * residual[1700.0]


def test_the_configured_length_approaches_the_recursive_filter(registry, context):
    noisy, clean = ecg()
    fir = float(np.std(apply_filter('fir_bandpass', noisy, context).signal - clean))
    iir = float(np.std(apply_filter('iir_bandpass', noisy, context).signal - clean))
    assert fir < 1.5 * iir


def test_a_filter_too_long_for_the_window_is_refused(registry):
    """
    Zero phase filtering pads by three times the length, so the window bounds it.

    Without the guard this surfaces as a scipy message about padlen several frames deep.
    """
    noisy, _ = ecg(n=1024)
    with pytest.raises(ValueError, match='samples for zero phase'):
        apply_filter('fir_bandpass', noisy, FilterContext(fs=FS))


def test_a_shorter_filter_fits_the_shorter_window(registry):
    noisy, _ = ecg(n=1024)
    result = apply_filter('fir_bandpass', noisy, FilterContext(fs=FS), numtaps_ms=500.0)
    assert result.signal.shape == noisy.shape
