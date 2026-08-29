"""`schema.py` is a leaf, and `Score` carries the spine's fields verbatim."""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from nbc import schema
from nbc.schema import Score


def _schema_source() -> tuple[Path, ast.Module]:
    path = Path(schema.__file__)
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_schema_imports_nothing_from_nbc() -> None:
    """The one rule that lets `canon/` depend on this module without losing its isolation.

    Checked against the parsed source, so an import hidden inside a function body or a
    `TYPE_CHECKING` block is caught just the same.
    """
    path, tree = _schema_source()
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nbc" or alias.name.startswith("nbc."):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # any relative import is an import from inside nbc
                offenders.append(
                    f"{path.name}:{node.lineno} from {'.' * node.level}{node.module or ''} import ..."
                )
            elif node.module == "nbc" or (node.module or "").startswith("nbc."):
                offenders.append(f"{path.name}:{node.lineno} from {node.module} import ...")

    assert not offenders, "schema.py must import nothing from nbc: " + "; ".join(offenders)


def test_importing_schema_pulls_in_no_other_nbc_module() -> None:
    """The same rule observed at runtime, in a fresh interpreter.

    `nbc` itself is loaded because `nbc.schema` is inside it; nothing else may be.
    """
    code = (
        "import sys, nbc.schema; "
        "print(sorted(m for m in sys.modules if m == 'nbc' or m.startswith('nbc.')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "['nbc', 'nbc.schema']", completed.stdout


def test_score_carries_exactly_the_declared_fields() -> None:
    assert [f.name for f in dataclasses.fields(Score)] == ["p_injection", "n_windows"]


def test_score_is_frozen() -> None:
    score = Score(p_injection=0.5, n_windows=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.p_injection = 0.9  # type: ignore[misc]


def test_score_accepts_the_full_probability_range() -> None:
    assert Score(p_injection=0.0, n_windows=1).p_injection == 0.0
    assert Score(p_injection=1.0, n_windows=7).p_injection == 1.0


def test_score_stores_integral_probabilities_as_floats() -> None:
    # So the serialized form does not depend on how the caller spelled 0 or 1.
    stored = Score(p_injection=1, n_windows=1).p_injection  # type: ignore[arg-type]
    assert isinstance(stored, float)


@pytest.mark.parametrize("bad", [1.5, -0.001, float("nan"), float("inf"), "0.5", None, True])
def test_score_rejects_a_p_injection_that_is_not_a_probability(bad: object) -> None:
    with pytest.raises(ValueError):
        Score(p_injection=bad, n_windows=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 1.0, "1", None, True])
def test_score_rejects_a_window_count_below_one(bad: object) -> None:
    # A scored document occupies at least one window; zero would silently mean "not scored".
    with pytest.raises(ValueError):
        Score(p_injection=0.5, n_windows=bad)  # type: ignore[arg-type]


def test_schema_declares_only_the_types_this_epic_uses() -> None:
    """Later stories add their own types here; none of them has run yet.

    This is not a freeze on `schema.py`. It is a reminder that a type belongs to the story
    that first needs it, so the next story updates this list in the same commit that adds
    its type — and a type that appears with no consumer is caught.
    """
    assert schema.__all__ == ["Score"]


def test_every_record_type_defined_here_is_exported() -> None:
    """A type present in the module but missing from `__all__` is a type with no contract."""
    defined = {
        name
        for name, obj in vars(schema).items()
        if not name.startswith("_")
        and dataclasses.is_dataclass(obj)
        and getattr(obj, "__module__", None) == schema.__name__
    }
    assert defined == set(schema.__all__)
