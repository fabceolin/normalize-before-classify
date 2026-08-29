"""The one exception base every abort in this project raises from.

A failure that would change the meaning of a published number aborts the run: a pin that no
longer resolves to the same commit, a label mapping with zero matches or with two, a partial
corpus, a missing caveats section, a machine below the platform floor. Each of those needs to
be distinguishable from the others by an automated caller, which means a distinct exit code
per abort and no two aborts sharing one.

Distinctness is enforced at class-definition time rather than by a test that someone might
not run. Declaring a duplicate code raises `ValueError` while the defining module is being
imported, so the ambiguous code cannot ship.

Adding an abort in a later story::

    class PinMismatch(NbcError, exit_code=10):
        '''A pinned revision no longer resolves to the recorded commit.'''

Codes 0 and 1 are reserved: 0 is success and 1 is an unexpected, unclassified failure — the
shape of every uncaught Python exception. Codes above 125 are reserved by POSIX shells
(126 not-executable, 127 not-found, 128+N killed by signal N), so a declared abort must not
land there or a caller cannot tell our abort from the shell's.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Final, Mapping

__all__ = [
    "EXIT_OK",
    "EXIT_UNEXPECTED",
    "MAX_EXIT_CODE",
    "MIN_EXIT_CODE",
    "NbcError",
    "declared_exit_codes",
    "exit_code_for",
]

EXIT_OK: Final[int] = 0
"""The run completed and every published number means what it says."""

EXIT_UNEXPECTED: Final[int] = 1
"""An unclassified failure. Never declared by an `NbcError` subclass."""

MIN_EXIT_CODE: Final[int] = 2
MAX_EXIT_CODE: Final[int] = 125

_RESERVED: Final[frozenset[int]] = frozenset({EXIT_OK, EXIT_UNEXPECTED})

_REGISTRY: Final[dict[int, type["NbcError"]]] = {}


class NbcError(Exception):
    """Base class for every abort. Not raisable on its own — subclass it with an exit code.

    Subclasses declare their code in the class header::

        class MissingCaveats(NbcError, exit_code=11): ...

    The code is then available as `MissingCaveats.exit_code` and on any instance.
    """

    exit_code: ClassVar[int]

    def __init_subclass__(cls, /, exit_code: int | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if exit_code is None:
            # An intermediate subclass that groups aborts without being raised itself
            # inherits its parent's code; a leaf that declares none and has no ancestor
            # with one would be an abort with no code at all, which is the bug this
            # module exists to prevent.
            if not hasattr(cls, "exit_code"):
                raise ValueError(
                    f"{cls.__qualname__} declares no exit_code and inherits none; "
                    f"every abort must carry one, e.g. "
                    f"`class {cls.__name__}(NbcError, exit_code=NN)`"
                )
            return

        if type(exit_code) is not int:
            raise ValueError(
                f"{cls.__qualname__} declares a non-integer exit_code "
                f"{exit_code!r} ({type(exit_code).__name__})"
            )

        if exit_code in _RESERVED:
            raise ValueError(
                f"{cls.__qualname__} declares reserved exit code {exit_code}; "
                f"{EXIT_OK} means success and {EXIT_UNEXPECTED} means an unclassified failure"
            )

        if not MIN_EXIT_CODE <= exit_code <= MAX_EXIT_CODE:
            raise ValueError(
                f"{cls.__qualname__} declares exit code {exit_code}, outside the usable "
                f"range {MIN_EXIT_CODE}..{MAX_EXIT_CODE}; codes above {MAX_EXIT_CODE} are "
                f"reserved by POSIX shells for their own failures"
            )

        already = _REGISTRY.get(exit_code)
        if already is not None:
            raise ValueError(
                f"{cls.__qualname__} declares exit code {exit_code}, already declared by "
                f"{already.__qualname__}; every abort must be distinguishable by its code"
            )

        _REGISTRY[exit_code] = cls
        cls.exit_code = exit_code

    def __init__(self, *args: object) -> None:
        if type(self) is NbcError:
            raise TypeError(
                "NbcError is the abort base and carries no exit code; raise a subclass "
                "that declares one"
            )
        super().__init__(*args)


def declared_exit_codes() -> Mapping[int, type[NbcError]]:
    """Every exit code declared so far, mapped to the class that declared it.

    Only reflects classes whose defining module has been imported, so a caller wanting the
    full set must import the modules it cares about first.
    """
    return MappingProxyType(dict(_REGISTRY))


def exit_code_for(error: BaseException) -> int:
    """The process exit code for `error`: its declared code, or `EXIT_UNEXPECTED`.

    An exception that is not one of our declared aborts is by definition unclassified, and
    reporting it as anything other than `EXIT_UNEXPECTED` would make an unhandled crash
    look like a checked failure.
    """
    if isinstance(error, NbcError):
        code = getattr(type(error), "exit_code", None)
        if type(code) is int:
            return code
    return EXIT_UNEXPECTED
