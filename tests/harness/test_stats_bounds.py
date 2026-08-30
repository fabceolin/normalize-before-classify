"""Two structural bounds on `harness/stats.py`, read from the syntax tree rather than from text.

*It has no numerical dependency.* The whole argument for writing four estimators out by hand is
that a reader does not have to ask which variant somebody's library gave us. An `import numpy` in
this module would end that argument without failing anything, which is why it is a test.

*Nothing published calls the rejected method.* `rejected_hanley_mcneil_variance` exists so the
rejection is a comparison rather than a paragraph, and a rejected estimator sitting in the same
module as the accepted one is one autocomplete away from being used. The scan refuses a reference
to that name from anywhere under `src/nbc/` but the module that defines it.

Both scans are checked against synthetic sources that violate them, because a scanner nobody has
seen report anything is not a scanner. The shape is `tests/canon/test_import_bound.py`'s, and the
reason it is not imported from there is that it answers a different question over a different tree.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
STATS = SRC / "nbc" / "harness" / "stats.py"

ALLOWED_NBC_IMPORTS = frozenset({"nbc.errors", "nbc.schema"})
"""The only `nbc` modules `stats.py` may reach, and both are stdlib-only leaves already."""

REJECTED = "rejected_hanley_mcneil_variance"


def imported_modules(source: str, package: tuple[str, ...]) -> set[str]:
    """Every module `source` imports, fully qualified, with relative imports resolved.

    From the syntax tree: the string `"import numpy"` inside a docstring is not an import, and this
    module's docstring contains exactly that phrase.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    names.add(node.module)
            else:
                anchor = package[: len(package) - node.level + 1]
                tail = (node.module,) if node.module else ()
                names.add(".".join((*anchor, *tail)))
    return names


def referenced_names(source: str) -> set[str]:
    """Every bare name and attribute tail the source mentions, from the tree.

    Attribute tails are included so `stats.rejected_hanley_mcneil_variance` is caught as readily as
    a direct import of it. A substring search over the text would report this test file's own
    docstring, which is the failure mode that made this repository stop searching text.

    The limit, declared: a reference assembled at runtime -- `getattr(stats, name)` where `name` is
    built from parts -- is invisible to any static scan and to this one. The scan is not a sandbox;
    it stops the reference somebody writes by reaching for the nearest function, which is the way
    a rejected estimator sitting beside the accepted one actually gets used.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def test_stats_imports_only_the_standard_library_and_two_leaves() -> None:
    package = STATS.relative_to(SRC).with_suffix("").parts[:-1]
    imported = imported_modules(STATS.read_text(encoding="utf-8"), package)

    for name in imported:
        root = name.split(".")[0]
        if root == "nbc":
            assert name in ALLOWED_NBC_IMPORTS, (
                f"stats.py imports {name!r}; only {sorted(ALLOWED_NBC_IMPORTS)} are allowed, and "
                f"both are stdlib-only leaves"
            )
        else:
            assert root in sys.stdlib_module_names, (
                f"stats.py imports {name!r}, which is not in the standard library; every interval "
                f"this project publishes is computed here with no numerical dependency"
            )


def test_the_import_scan_sees_a_numerical_dependency_when_there_is_one() -> None:
    """The input that turns the previous test red."""
    imported = imported_modules("import numpy as np\nfrom scipy import stats\n", ("nbc", "harness"))
    assert imported == {"numpy", "scipy"}
    assert {name for name in imported if name not in sys.stdlib_module_names} == {"numpy", "scipy"}


def test_the_import_scan_resolves_a_relative_import() -> None:
    imported = imported_modules("from ..schema import Interval\n", ("nbc", "harness"))
    assert imported == {"nbc.schema"}


def test_the_import_scan_does_not_read_a_docstring_as_an_import() -> None:
    """`"import numpy"` in prose is prose. Story 2.1's review found `"Tokenizer("` inside
    `"WindowedTokenizer("` in this repository and the lesson stuck."""
    assert imported_modules('"""Never import numpy here."""\nimport math\n', ("nbc",)) == {"math"}


def test_the_rejected_method_is_defined_in_stats_and_referenced_nowhere_else() -> None:
    defining = []
    referencing = []
    for path in sorted(SRC.rglob("*.py")):
        if REJECTED in referenced_names(path.read_text(encoding="utf-8")):
            (defining if path == STATS else referencing).append(path)

    assert defining == [STATS], f"{REJECTED} is not defined where it is supposed to be"
    assert referencing == [], (
        f"{REJECTED} is referenced by {[str(p) for p in referencing]}; Hanley-McNeil's variance is "
        f"rejected, and it ships only so the rejection is a comparison a test can run"
    )


def test_the_reference_scan_sees_a_call_through_a_module_attribute() -> None:
    """The shape the previous test exists to catch, and the one an import scan alone would miss."""
    source = "from nbc.harness import stats\nv = stats.rejected_hanley_mcneil_variance(0.9, 5, 5)\n"
    assert REJECTED in referenced_names(source)


def test_the_reference_scan_sees_a_direct_import() -> None:
    source = "from nbc.harness.stats import rejected_hanley_mcneil_variance\n"
    assert REJECTED in referenced_names(source)


def test_the_reference_scan_does_not_report_an_unrelated_module() -> None:
    assert REJECTED not in referenced_names("from nbc.harness.stats import roc_auc\nroc_auc(x)\n")


def test_numpy_is_installed_and_that_is_why_the_scan_is_the_enforcement() -> None:
    """The obvious stronger assertion -- that no numerical library is installed at all -- is false
    here, and writing it would have made the suite fail for the wrong reason.

    `onnxruntime==1.29.0` declares `numpy>=1.21.6` as a hard requirement, so numpy is in every
    environment this project measures in and always will be. "No numerical dependency" is therefore
    a claim about what `stats.py` imports and can only be checked by reading `stats.py`. This test
    exists to record that, so nobody later strengthens the bound into an absence check that
    `uv sync` will contradict.

    `scipy` is genuinely absent, and nothing here relies on it staying that way.
    """
    import numpy  # noqa: F401  -- imported to demonstrate it is present, not to use it

    assert "numpy" not in imported_modules(STATS.read_text(encoding="utf-8"), ("nbc", "harness"))
