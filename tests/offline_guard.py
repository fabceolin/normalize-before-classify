"""The offline claim, as a mechanism a child process cannot walk out of.

"The unit suite runs with no network and no model download" is a claim a reader has to be able
to trust without unplugging their machine, so the suite unplugs itself. Three properties this
module exists to hold, each of which the first version did not:

**It installs once, globally.** The guard used to be a function-scoped autouse fixture, so
collection-time imports and every module- or session-scoped fixture ran outside it. A later
story writing `@pytest.fixture(scope="session")` around a pinned-artifact load -- the canonical
way to load one once -- would have downloaded a model while the suite reported green.

**It survives `subprocess`.** Patching this process says nothing about its children, and this
project's tests shell out to `sys.executable` in seven modules. `nbc.pins` has a `--verify` path
that opens `https://huggingface.co/api/...`, and one default flip inside it would have made the
offline claim false with no signal. The guard is re-installed in every child through
`PYTHONPATH` and a `sitecustomize` module.

**It refuses more than `connect`.** `gethostbyname`, `sendto` on an unconnected UDP socket, and
`create_connection` are all outbound paths that the original three patches left open.

Loopback is permitted, and the ORDER of those two changes was deliberate: widening the guard
before its enforcement held would have shipped something both leakier and more permissive than
what it replaced.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Final

HF_OFFLINE_VARS: Final[tuple[str, ...]] = (
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

_LOOPBACK: Final[frozenset[str]] = frozenset({"127.0.0.1", "::1", "localhost"})

_LOCAL_FAMILIES: Final[frozenset[int]] = frozenset(
    getattr(socket, name) for name in ("AF_UNIX",) if hasattr(socket, name)
)

_SAVED: dict[str, Any] = {}
_SAVED_ENVIRONMENT: dict[str, str | None] = {}


class NetworkAccessInUnitSuite(RuntimeError):
    """A unit test tried to reach the network. It must not.

    Deliberately *not* an `nbc.errors.NbcError`: this is a defect in a test, not one of the
    project's declared aborts, and it must never be mistaken for one by an exit code.
    """


def _target_host(address: Any) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


def _is_local(family: Any, address: Any) -> bool:
    if family in _LOCAL_FAMILIES:
        return True
    host = _target_host(address)
    return host in _LOOPBACK if host is not None else False


def _refuse(what: str, target: object) -> NetworkAccessInUnitSuite:
    return NetworkAccessInUnitSuite(
        f"{what} to {target!r} was blocked: the unit suite runs with no network and no model "
        f"download. Mark the test `@pytest.mark.smoke` if it genuinely needs one."
    )


def install() -> None:
    """Refuse every outbound path. Idempotent, so a child that inherits it does not double-patch."""
    if _SAVED:
        return

    _SAVED["connect"] = socket.socket.connect
    _SAVED["connect_ex"] = socket.socket.connect_ex
    _SAVED["sendto"] = socket.socket.sendto
    _SAVED["create_connection"] = socket.create_connection
    _SAVED["getaddrinfo"] = socket.getaddrinfo
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        _SAVED[name] = getattr(socket, name, None)

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if _is_local(self.family, address):
            return _SAVED["connect"](self, address)
        raise _refuse("socket.connect", address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if _is_local(self.family, address):
            return _SAVED["connect_ex"](self, address)
        raise _refuse("socket.connect_ex", address)

    def guarded_sendto(self: socket.socket, data: Any, *args: Any) -> int:
        address = args[-1] if args else None
        if _is_local(self.family, address):
            return _SAVED["sendto"](self, data, *args)
        raise _refuse("socket.sendto", address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if _is_local(socket.AF_INET, address):
            return _SAVED["create_connection"](address, *args, **kwargs)
        raise _refuse("socket.create_connection", address)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        if str(host) in _LOOPBACK:
            return _SAVED["getaddrinfo"](host, port, *args, **kwargs)
        raise _refuse("socket.getaddrinfo", host)

    def guarded_resolution(name: str) -> Any:
        def guard(host: Any, *args: Any, **kwargs: Any) -> Any:
            if str(host) in _LOOPBACK:
                return _SAVED[name](host, *args, **kwargs)
            raise _refuse(f"socket.{name}", host)

        return guard

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = guarded_sendto  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        if _SAVED[name] is not None:
            setattr(socket, name, guarded_resolution(name))

    for var in HF_OFFLINE_VARS:
        # Saved so `uninstall()` can put them back. Setting them process-wide and never lifting
        # them made `smoke` -- the one documented escape hatch, and the only tier CI runs against
        # real artifacts -- an escape hatch for in-process sockets and nothing else: the
        # huggingface libraries would still refuse to fetch.
        _SAVED_ENVIRONMENT[var] = os.environ.get(var)
        os.environ[var] = "1"


def uninstall() -> None:
    """Restore the real socket API, for a test marked `smoke`."""
    if not _SAVED:
        return
    socket.socket.connect = _SAVED["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _SAVED["connect_ex"]  # type: ignore[method-assign]
    socket.socket.sendto = _SAVED["sendto"]  # type: ignore[method-assign]
    socket.create_connection = _SAVED["create_connection"]  # type: ignore[assignment]
    socket.getaddrinfo = _SAVED["getaddrinfo"]  # type: ignore[assignment]
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        if _SAVED[name] is not None:
            setattr(socket, name, _SAVED[name])
    for var, previous in _SAVED_ENVIRONMENT.items():
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous
    _SAVED_ENVIRONMENT.clear()
    _SAVED.clear()


def is_installed() -> bool:
    return bool(_SAVED)
