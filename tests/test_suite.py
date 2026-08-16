"""
Single ordered entry point for the project's manual-style check suites.

`smoke_test.py` and `test_signal_selection.py` predate assert-based testing in this
project: each check prints its own diagnostics and returns a bool instead of
asserting, so pytest reports them as passing regardless of the returned value. This
file is the one place that actually asserts on their result, running every check in
a fixed order. Both source files are excluded from pytest's own collection (see
`pytest.ini`) so each check runs exactly once.

To register a new suite: import its module below and append `(label, check)` tuples
to `CHECKS`, in the order they should run.
"""

import smoke_test
import test_signal_selection

CHECKS = [
    ('smoke_test.test_models', smoke_test.test_models),
    ('smoke_test.test_scc_roundtrip', smoke_test.test_scc_roundtrip),
    ('smoke_test.test_stft_roundtrip', smoke_test.test_stft_roundtrip),
    ('test_signal_selection.test_registry', test_signal_selection.test_registry),
    ('test_signal_selection.test_shapes_and_roundtrip', test_signal_selection.test_shapes_and_roundtrip),
    ('test_signal_selection.test_shared_statistics', test_signal_selection.test_shared_statistics),
    ('test_signal_selection.test_cache_budget', test_signal_selection.test_cache_budget),
]


def test_full_suite() -> None:
    """Runs every registered check in order and asserts once all of them ran."""
    failed = []
    for label, check in CHECKS:
        print(f'\n>>> {label}')
        if not check():
            failed.append(label)

    assert not failed, f'failed checks: {failed}'
