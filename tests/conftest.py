"""Shared pytest configuration for the assignment2 test suite.

Live tests (real paid/remote services) are opt-in: set RUN_LIVE=1.
"""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `ingest` (and later `app`) importable
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test; set RUN_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls real remote services")


@pytest.fixture(scope="session")
def root() -> pathlib.Path:
    return ROOT
