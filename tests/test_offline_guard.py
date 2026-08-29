"""The offline guard is armed. Without this, `conftest.py` could silently stop working."""

from __future__ import annotations

import os
import socket

import pytest

from offline_guard import NetworkAccessInUnitSuite


def test_outbound_tcp_connect_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessInUnitSuite):
            sock.connect(("example.invalid", 80))


def test_connect_ex_is_refused_rather_than_returning_an_errno() -> None:
    # `connect_ex` reports failure by return value, so an unguarded version would let a
    # test "succeed" at reaching the network and never raise.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessInUnitSuite):
            sock.connect_ex(("example.invalid", 80))


def test_create_connection_is_refused() -> None:
    with pytest.raises(NetworkAccessInUnitSuite):
        socket.create_connection(("example.invalid", 80), timeout=0.01)


def test_name_resolution_is_refused() -> None:
    with pytest.raises(NetworkAccessInUnitSuite):
        socket.getaddrinfo("example.invalid", 80)


def test_hugging_face_offline_variables_are_set() -> None:
    # A model download is the other half of the claim, and the hub client honours these
    # before it opens a socket at all.
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert os.environ.get(var) == "1", f"{var} is not set for the unit suite"


def test_local_sockets_still_work() -> None:
    # The guard blocks egress, not the loopback machinery pytest and its plugins may use.
    if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - platform dependent
        pytest.skip("AF_UNIX not available on this platform")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError) as caught:
            sock.connect("/nonexistent/nbc-offline-guard.sock")
    assert not isinstance(caught.value, NetworkAccessInUnitSuite)
