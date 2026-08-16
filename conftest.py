"""Shared pytest configuration: network/slow opt-in markers and repo root on sys.path."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--run-network', action='store_true', default=False,
                      help='run tests marked "network" (requires PhysioNet access)')
    parser.addoption('--run-slow', action='store_true', default=False,
                      help='run tests marked "slow"')


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    skip_network = pytest.mark.skip(reason='needs --run-network to run')
    skip_slow = pytest.mark.skip(reason='needs --run-slow to run')
    run_network = config.getoption('--run-network')
    run_slow = config.getoption('--run-slow')

    for item in items:
        if 'network' in item.keywords and not run_network:
            item.add_marker(skip_network)
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)
