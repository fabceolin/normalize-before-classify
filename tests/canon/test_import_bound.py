"""`canon/` imports only the standard library, its vendored data, and two stdlib-only leaves.

This is the bound that keeps the layer usable in front of any classifier: no model, no third-party
package, no reach back into the harness. It is a test rather than a convention because a
convention is what an import statement quietly breaks.

`nbc.errors` is in the allowance alongside `nbc.schema` for a stated reason — every abort in this
project raises from there with a distinct exit code — and it is not merely allow-listed: both are
scanned and must **stay** leaves that import nothing but the standard library. The scanner itself
is checked against synthetic modules that violate each rule, because a scanner nobody has seen
report anything is not a scanner.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
CANON = SRC / "nbc" / "canon"

ALLOWED_LEAVES = frozenset({"nbc.schema", "nbc.errors"})
"""The only `nbc` modules `canon/` may import, and both must be stdlib-only leaves themselves."""


def imported_modules(path: Path) -> set[str]:
    """Every module name `path` imports, fully qualified, with relative imports resolved.

    Reads the syntax tree rather than the text: a name inside a docstring or a string literal is
    not an import, and a scan that could not tell the difference would be pattern-matching where
    structure is available.
    """
    # The package the file lives in, which is the anchor a relative import resolves against.
    # `pkg/__init__.py` and `pkg/module.py` both anchor at `pkg`.
    package = path.relative_to(SRC).with_suffix("").parts[:-1]

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                prefix = ".".join(base)
                names.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                names.add(node.module)
    return names


def offending_imports(path: Path, *, allowed_prefixes: tuple[str, ...]) -> list[str]:
    """The imports in `path` that are neither standard library nor allowed by name or prefix."""
    offenders = []
    for name in sorted(imported_modules(path)):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names:
            continue
        if name in ALLOWED_LEAVES:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in allowed_prefixes):
            continue
        offenders.append(name)
    return offenders


CANON_MODULES = sorted(CANON.rglob("*.py"))
LEAF_MODULES = [SRC / "nbc" / f"{leaf.split('.')[1]}.py" for leaf in sorted(ALLOWED_LEAVES)]


def test_the_scan_found_the_modules_it_is_supposed_to_scan() -> None:
    # A scan over an empty file list passes vacuously. This is what makes the suite below mean
    # something: the pipeline, the four stages and the shared helper are all in it.
    found = {path.relative_to(CANON).as_posix() for path in CANON_MODULES}
    assert {
        "__init__.py",
        "pipeline.py",
        "edits.py",
        "confusables_table.py",
        "stages/__init__.py",
        "stages/invisible.py",
        "stages/confusables.py",
        "stages/nfkc.py",
        "stages/decode.py",
    } <= found


@pytest.mark.parametrize("path", CANON_MODULES, ids=lambda p: p.name)
def test_every_canon_module_imports_only_what_the_bound_allows(path: Path) -> None:
    assert offending_imports(path, allowed_prefixes=("nbc.canon",)) == []


@pytest.mark.parametrize("path", LEAF_MODULES, ids=lambda p: p.name)
def test_the_allowed_leaves_are_still_leaves(path: Path) -> None:
    """Not allow-listed by name: scanned, and required to import nothing but the standard library.

    The day `nbc.schema` or `nbc.errors` grows an import of `nbc.pins`, the layer's isolation is
    gone and only this test notices.
    """
    assert offending_imports(path, allowed_prefixes=()) == []


def test_only_the_pipeline_imports_the_stage_modules() -> None:
    """AD-4: no caller may invoke a stage out of band. An import is how a caller would get one.

    Scoped to `src/`; the tests import the stages directly on purpose, one test file per stage.
    """
    importers = {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if any(name.startswith("nbc.canon.stages") for name in imported_modules(path))
    }
    assert importers == {"nbc/canon/pipeline.py"}


# --- the scanner, checked against modules that break each rule --------------------------------


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "src" / "nbc" / "canon" / "probe.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")

    def run(body: str, *, allowed_prefixes: tuple[str, ...] = ("nbc.canon",)) -> list[str]:
        return offending_imports(write(tmp_path, body), allowed_prefixes=allowed_prefixes)

    return run


def test_the_scanner_reports_a_third_party_import(scan) -> None:
    assert scan("import requests\n") == ["requests"]


def test_the_scanner_reports_a_third_party_from_import(scan) -> None:
    assert scan("from onnxruntime import InferenceSession\n") == ["onnxruntime"]


def test_the_scanner_reports_a_reach_back_into_the_harness(scan) -> None:
    assert scan("from nbc.pins import load\n") == ["nbc.pins"]


def test_the_leaf_scan_reports_a_leaf_that_stopped_being_one(scan) -> None:
    # The exact input the leaf check exists for: `nbc.schema` or `nbc.errors` growing an import of
    # a module that is not standard library. Scanned with no prefix allowance, as the leaves are.
    assert scan("from nbc.pins import load\n", allowed_prefixes=()) == ["nbc.pins"]


def test_the_scanner_reports_a_relative_import_that_leaves_the_package(scan) -> None:
    assert scan("from ..pins import load\n") == ["nbc.pins"]


def test_the_scanner_admits_the_two_declared_leaves(scan) -> None:
    assert scan("from nbc.schema import Edit\nimport nbc.errors\n") == []


def test_the_scanner_admits_siblings_and_the_standard_library(scan) -> None:
    assert scan("import json\nfrom nbc.canon.edits import build_edits\nfrom . import edits\n") == []


def test_the_scanner_refuses_the_leaves_when_the_allowance_is_not_in_play(scan) -> None:
    # The leaf scan runs with no prefix allowance, and `ALLOWED_LEAVES` must not smuggle a leaf
    # into its own check. A leaf importing the other one is still a leaf importing `nbc`.
    assert scan("import nbc.canon.edits\n", allowed_prefixes=()) == ["nbc.canon.edits"]


def test_the_scanner_ignores_module_names_that_only_appear_in_text(scan) -> None:
    assert scan('"""import requests, and then from nbc.pins import load."""\nimport json\n') == []


# --- what actually gets imported at runtime ----------------------------------------------------


def test_importing_the_layer_pulls_in_nothing_outside_the_bound() -> None:
    """The static scan reads what the files say; this reads what the interpreter did.

    A module reached through some path the AST scan did not model shows up here and nowhere else.
    """
    code = (
        "import nbc.canon.pipeline, sys;"
        "print(' '.join(sorted(m for m in sys.modules if m.split('.')[0] == 'nbc')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(completed.stdout.split())
    assert loaded == {
        "nbc",
        "nbc.canon",
        "nbc.canon.confusables_table",
        "nbc.canon.edits",
        "nbc.canon.pipeline",
        "nbc.canon.stages",
        "nbc.canon.stages.confusables",
        "nbc.canon.stages.decode",
        "nbc.canon.stages.invisible",
        "nbc.canon.stages.nfkc",
        "nbc.errors",
        "nbc.schema",
    }, completed.stdout


@pytest.mark.parametrize("forbidden", ["onnxruntime", "tokenizers", "numpy", "huggingface_hub"])
def test_importing_the_layer_does_not_pull_in_a_model_runtime(forbidden: str) -> None:
    code = f"import nbc.canon.pipeline, sys; print({forbidden!r} in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout
