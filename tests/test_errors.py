"""Every abort raises from one base and carries an exit code no other abort can share."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import pytest

import nbc
from nbc import errors as errors_module
from nbc.errors import (
    EXIT_OK,
    EXIT_UNEXPECTED,
    MAX_EXIT_CODE,
    MIN_EXIT_CODE,
    NbcError,
    declared_exit_codes,
    exit_code_for,
)


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the exit-code registry.

    These tests declare throwaway aborts. Left in the registry they would collide with a
    real code a later story declares, and this file would start failing for a reason that
    has nothing to do with the code under test.
    """
    snapshot = dict(errors_module._REGISTRY)
    try:
        yield
    finally:
        errors_module._REGISTRY.clear()
        errors_module._REGISTRY.update(snapshot)


# --- the whole source tree, read statically ------------------------------------------------
#
# Read rather than imported: importing every module to collect its aborts would drag the
# inference runtime into a unit suite that must stay offline and cheap, and would only ever
# see the codes that a particular import order happened to register.


def _error_declarations() -> list[tuple[str, str, int | None]]:
    """Every `NbcError` subclass in `src/nbc/`, as (location, class name, declared code)."""
    package_root = Path(nbc.__file__).resolve().parent
    known_bases = {"NbcError"}
    found: list[tuple[str, str, int | None]] = []

    sources = sorted(package_root.rglob("*.py"))
    # Two passes: a subclass may be declared before the class it extends is seen.
    for _ in range(2):
        for path in sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
                if not (base_names & known_bases):
                    continue
                known_bases.add(node.name)
                code: int | None = None
                for keyword in node.keywords:
                    if keyword.arg == "exit_code":
                        value = keyword.value
                        if not (isinstance(value, ast.Constant) and type(value.value) is int):
                            pytest.fail(
                                f"{path}:{node.lineno} {node.name} declares exit_code as an "
                                f"expression; it must be an integer literal so the declared "
                                f"codes can be read without running the project"
                            )
                        code = value.value
                entry = (f"{path.name}:{node.lineno}", node.name, code)
                if entry not in found:
                    found.append(entry)
    return found


def test_every_declared_abort_has_an_exit_code_distinct_from_every_other() -> None:
    by_code: dict[int, str] = {}
    collisions: list[str] = []

    for location, name, code in _error_declarations():
        if code is None:
            continue  # groups an existing abort under its parent's code
        owner = f"{name} ({location})"
        if code in by_code:
            collisions.append(f"exit code {code}: {by_code[code]} and {owner}")
        else:
            by_code[code] = owner

    assert not collisions, "; ".join(collisions)


def test_no_declared_abort_uses_a_reserved_or_out_of_range_code() -> None:
    offenders = [
        f"{name} ({location}) declares {code}"
        for location, name, code in _error_declarations()
        if code is not None and not MIN_EXIT_CODE <= code <= MAX_EXIT_CODE
    ]
    assert not offenders, "; ".join(offenders)


# --- the mechanism that makes a collision impossible rather than merely detected ----------


def test_a_duplicate_exit_code_fails_at_class_definition_time() -> None:
    class First(NbcError, exit_code=90):
        pass

    with pytest.raises(ValueError, match="already declared by"):

        class Second(NbcError, exit_code=90):
            pass

    assert First.exit_code == 90


@pytest.mark.parametrize("reserved", [EXIT_OK, EXIT_UNEXPECTED])
def test_reserved_exit_codes_are_refused(reserved: int) -> None:
    with pytest.raises(ValueError, match="reserved exit code"):

        class Reserved(NbcError, exit_code=reserved):
            pass


@pytest.mark.parametrize("out_of_range", [MIN_EXIT_CODE - 1, MAX_EXIT_CODE + 1, 128, 255])
def test_exit_codes_outside_the_usable_range_are_refused(out_of_range: int) -> None:
    # 126, 127 and 128+N belong to the shell; an abort landing there is indistinguishable
    # from "not executable", "not found", or "killed by a signal".
    with pytest.raises(ValueError):

        class OutOfRange(NbcError, exit_code=out_of_range):
            pass


def test_a_non_integer_exit_code_is_refused() -> None:
    with pytest.raises(ValueError, match="non-integer"):

        class Wrong(NbcError, exit_code="12"):  # type: ignore[arg-type]
            pass


def test_a_subclass_declaring_no_code_and_inheriting_none_is_refused() -> None:
    with pytest.raises(ValueError, match="declares no exit_code"):

        class NoCode(NbcError):
            pass


def test_a_subclass_may_group_aborts_under_an_inherited_code() -> None:
    class Group(NbcError, exit_code=91):
        pass

    class Specific(Group):
        pass

    assert Specific.exit_code == 91


def test_the_base_itself_cannot_be_raised() -> None:
    with pytest.raises(TypeError, match="carries no exit code"):
        raise NbcError("this should never be the shape of an abort")


def test_declared_aborts_report_their_own_code() -> None:
    class Declared(NbcError, exit_code=92):
        pass

    assert exit_code_for(Declared("boom")) == 92
    assert declared_exit_codes()[92] is Declared


def test_an_undeclared_exception_is_unclassified_rather_than_a_checked_failure() -> None:
    assert exit_code_for(RuntimeError("boom")) == EXIT_UNEXPECTED
    assert exit_code_for(KeyboardInterrupt()) == EXIT_UNEXPECTED


def test_the_registry_is_a_read_only_snapshot() -> None:
    codes = declared_exit_codes()
    with pytest.raises(TypeError):
        codes[99] = NbcError  # type: ignore[index]
