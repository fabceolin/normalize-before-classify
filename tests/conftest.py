"""Makes the offline claim a mechanism rather than a promise.

The guard installs ONCE, at `pytest_configure`, before collection imports a single test module
-- because a function-scoped autouse fixture, which is what this was, leaves collection-time
imports and every module- or session-scoped fixture running unguarded. It is lifted per test for
`smoke`, which is excluded from the default run by `addopts` in `pyproject.toml`.

It also reaches child processes. See `offline_guard` for why that is not optional here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import offline_guard

REPO_ROOT = Path(__file__).resolve().parent.parent
_CHILD_GUARD = Path(__file__).resolve().parent / "_childguard"


def pytest_configure(config: pytest.Config) -> None:
    """Install before collection, and arrange for children to install it too."""
    offline_guard.install()

    # `sitecustomize` is imported automatically at interpreter start, so a child that inherits
    # this PYTHONPATH re-installs the guard on itself. The tests directory comes along because
    # that is where `offline_guard` lives.
    tests_dir = str(Path(__file__).resolve().parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_CHILD_GUARD), tests_dir] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    os.environ["NBC_OFFLINE_GUARD"] = "1"


def pytest_unconfigure(config: pytest.Config) -> None:
    offline_guard.uninstall()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """`smoke` is the one escape hatch, and it is lifted for that test only."""
    if item.get_closest_marker("smoke") is not None:
        offline_guard.uninstall()
    else:
        offline_guard.install()


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    offline_guard.install()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so tests can read `pyproject.toml` and `uv.lock` as data."""
    return REPO_ROOT
