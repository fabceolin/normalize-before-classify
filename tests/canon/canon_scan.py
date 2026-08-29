"""Structural facts about `canon/`, read from the syntax tree, shared by the tests that need them.

Not a test module. It answers three questions the layer's claims are checked against: which files
the layer is, which of them run in front of a classifier, and what each one binds at module scope.

Every answer comes from `ast`, never from a regex over source text. Story 2.1's review found
`"Tokenizer(" in "WindowedTokenizer("` in this repository, and the lesson stuck: where a parsed
structure is available, matching text is a guess that happens to be right.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nbc.report.size_budget import EXCLUDED_MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
CANON = SRC / "nbc" / "canon"
TESTS = REPO_ROOT / "tests" / "canon"


def canon_modules() -> tuple[Path, ...]:
    """Every Python module under `canon/`, sorted, derivation script included."""
    return tuple(sorted(CANON.rglob("*.py")))


def is_runtime(path: Path) -> bool:
    """Whether `path` is part of the layer that runs, rather than the build-time derivation.

    The split is `nbc.report.size_budget.EXCLUDED_MODULES`, which is one home for the notion and
    is itself checked against a real import in `tests/report/test_size_budget.py`. A second list
    here would be a second answer to the same question.
    """
    return path.relative_to(CANON).as_posix() not in EXCLUDED_MODULES


def runtime_modules() -> tuple[Path, ...]:
    return tuple(path for path in canon_modules() if is_runtime(path))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_scope_bindings(tree: ast.Module) -> tuple[str, ...]:
    """Every name bound by an assignment at module scope, in source order.

    Assignments only. Names that arrive by `import` are not this module's state — they are
    somebody else's object, and holding `canon/` responsible for `typing.Final` being a
    `_SpecialForm` would make the check about the standard library rather than about the layer.
    """
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                names.append(node.target.id)
    return tuple(names)


def top_level_imports(tree: ast.Module) -> frozenset[str]:
    """The top-level package name of every import anywhere in the tree.

    `from urllib.request import urlopen` and `import urllib.request` both answer `urllib`, which
    is the granularity the clock-and-randomness vocabulary is written at. Relative imports never
    reach the standard library, so they contribute nothing and are skipped.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    return frozenset(names)


def global_statements(tree: ast.Module) -> tuple[str, ...]:
    """Every name any function in this module declares `global`, which is how module state moves."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.extend(node.names)
    return tuple(sorted(names))


def imported_dotted(tree: ast.Module) -> frozenset[str]:
    """Every absolute module name the tree imports, including the ones a `from` import names.

    `from nbc.canon.stages import decode` answers `nbc.canon.stages` **and**
    `nbc.canon.stages.decode`, because the second is the module that was actually reached and the
    caller asking "does this file import that module" means the second. Relative imports are
    skipped: nothing under `tests/` uses one, and resolving them would need the anchor package a
    test directory does not have.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return frozenset(names)
