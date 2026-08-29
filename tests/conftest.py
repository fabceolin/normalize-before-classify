"""Makes the offline claim a mechanism rather than a promise.

"The unit suite runs with no network and no model download" is a claim a reader has to be
able to trust without unplugging their machine. So the suite unplugs itself: every test that
is not marked `smoke` runs with outbound sockets and DNS refused, and a test that reaches for
the network fails loudly here instead of passing on the maintainer's laptop and failing on a
CI runner with no egress.

`smoke` is the escape hatch, and it is excluded from the default run by `addopts` in
`pyproject.toml`. Nothing in this project's unit suite is expected to use it.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Iterator

import pytest

from offline_guard import NetworkAccessInUnitSuite

REPO_ROOT = Path(__file__).resolve().parent.parent

_LOCAL_FAMILIES = frozenset(
    getattr(socket, name) for name in ("AF_UNIX",) if hasattr(socket, name)
)

_HF_OFFLINE_VARS = (
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Refuse outbound sockets and DNS for the duration of every non-`smoke` test."""
    if request.node.get_closest_marker("smoke") is not None:
        yield
        return

    for var in _HF_OFFLINE_VARS:
        monkeypatch.setenv(var, "1")

    def _refuse(what: str, target: object) -> NetworkAccessInUnitSuite:
        return NetworkAccessInUnitSuite(
            f"{what} to {target!r} was blocked: the unit suite runs with no network and no "
            f"model download. Mark the test `@pytest.mark.smoke` if it genuinely needs one."
        )

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family in _LOCAL_FAMILIES:
            return real_connect(self, address)
        raise _refuse("socket.connect", address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in _LOCAL_FAMILIES:
            return real_connect_ex(self, address)
        raise _refuse("socket.connect_ex", address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        raise _refuse("socket.create_connection", address)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        raise _refuse("socket.getaddrinfo", host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so tests can read `pyproject.toml` and `uv.lock` as data."""
    return REPO_ROOT
