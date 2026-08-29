"""The failure a unit test sees when it reaches for the network.

Lives in its own module rather than in `conftest.py` so tests can import it by a stable name
once the test tree grows nested packages, each with a `conftest` of its own.
"""

from __future__ import annotations


class NetworkAccessInUnitSuite(RuntimeError):
    """A unit test tried to reach the network. It must not.

    Deliberately *not* an `nbc.errors.NbcError`: this is a defect in a test, not one of the
    project's declared aborts, and it must never be mistaken for one by an exit code.
    """
