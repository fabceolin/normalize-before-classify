"""The offline guard is armed. Without this, `conftest.py` could silently stop working."""

from __future__ import annotations

import os
import socket

import subprocess
import sys
from typing import Any
import offline_guard
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


# --- Pass 6: the holes the review found ---------------------------------------------------------
#
# Two blocking findings. Both were reproduced before the fix: a child process reached the network
# and printed CONNECTED, and a module-scoped fixture opened a socket while the suite reported
# green.


def test_the_guard_survives_a_subprocess() -> None:
    """Patching this process says nothing about its children, and this suite has many.

    Seven test modules shell out to `sys.executable`, one of them driving `nbc.pins`, which has
    a path that opens `https://huggingface.co/api/...`. Before this, the child connected.
    """
    probe = (
        "import socket\n"
        "s = socket.socket(); s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53)); print('CONNECTED')\n"
        "except Exception as error:\n"
        "    print(type(error).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )

    assert result.stdout.strip() == "NetworkAccessInUnitSuite", result.stdout + result.stderr


def test_the_guard_is_installed_before_collection_not_per_test() -> None:
    """A module- or session-scoped fixture runs outside a function-scoped autouse fixture.

    The canonical way a later story loads a pinned artifact once is a session-scoped fixture, and
    under the old guard that fixture would have downloaded a model with the suite still green.
    """
    assert offline_guard.is_installed()


@pytest.mark.parametrize(
    "call",
    [
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr("93.184.216.34"),
    ],
)
def test_every_resolution_path_is_refused_not_only_getaddrinfo(call: Any) -> None:
    """`getaddrinfo` was patched and its three siblings were not."""
    with pytest.raises(NetworkAccessInUnitSuite):
        call()


def test_an_unconnected_udp_send_is_refused() -> None:
    """`sendto` needs no `connect`, so patching connect left the whole path open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkAccessInUnitSuite):
            sock.sendto(b"x", ("1.1.1.1", 53))
    finally:
        sock.close()


def test_loopback_is_permitted() -> None:
    """Widened only after the enforcement above held, per decision D-E.

    A test that binds a local server and talks to it is doing nothing this guard exists to stop,
    and refusing it pushed tests toward mocking what they could have exercised.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.socket()
        try:
            client.settimeout(5)
            client.connect(server.getsockname())
        finally:
            client.close()
    finally:
        server.close()


def test_uninstall_lifts_the_offline_environment_too(monkeypatch: Any) -> None:
    """`smoke` is the one escape hatch, and it stopped being one for anything but sockets.

    The guard sets HF_HUB_OFFLINE and its siblings process-wide. `uninstall()` restored the
    socket API and left those set, so a smoke test -- the only tier that touches a real pinned
    artifact, and the only one CI runs against the hub -- got its sockets back while the
    huggingface libraries went on refusing to fetch. Introduced by the fix for the guard's own
    holes, which is the shape this repository keeps finding.
    """
    offline_guard.uninstall()
    try:
        for var in offline_guard.HF_OFFLINE_VARS:
            assert os.environ.get(var) != "1", f"{var} survived uninstall()"
    finally:
        offline_guard.install()

    for var in offline_guard.HF_OFFLINE_VARS:
        assert os.environ[var] == "1", "install() must set them back"
