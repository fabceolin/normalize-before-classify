"""The Structural Seed exists, and importing it costs nothing.

The layout is not decoration: each namespace is one side of a declared seam, and a module
that lands in the wrong one breaks a rule the diagram makes look obvious.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import nbc

SEED_NAMESPACES = ("canon", "corpus", "baselines", "harness", "report")
LEAF_MODULES = ("schema", "errors", "platform", "pins")


@pytest.mark.parametrize("namespace", SEED_NAMESPACES)
def test_each_seed_namespace_is_an_importable_package(namespace: str) -> None:
    module = importlib.import_module(f"nbc.{namespace}")
    assert module.__file__ is not None
    assert Path(module.__file__).name == "__init__.py"
    assert Path(module.__file__).parent.name == namespace


@pytest.mark.parametrize("leaf", LEAF_MODULES)
def test_each_leaf_module_is_importable(leaf: str) -> None:
    module = importlib.import_module(f"nbc.{leaf}")
    assert module.__file__ is not None
    assert Path(module.__file__).name == f"{leaf}.py"


def test_the_package_lives_under_a_src_layout() -> None:
    assert Path(nbc.__file__).resolve().parent.name == "nbc"


def test_importing_nbc_does_not_import_the_inference_runtime() -> None:
    """The platform preflight runs before `onnxruntime` is imported, or it is no floor at all.

    If `import nbc` dragged the runtime in, the preflight would be checking a floor the
    import had already crashed through — with the wheel's own error message, which is the
    exact failure the floor exists to replace.
    """
    code = "import sys, nbc; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout


def test_the_test_tree_mirrors_the_source_tree(repo_root: Path) -> None:
    """Tests mirror the source tree, so a module without a test is visible as a gap.

    Only the packages that already hold code are checked: an empty seed namespace has
    nothing to mirror yet, and demanding a test directory for it would be demanding a
    placeholder.
    """
    package_root = repo_root / "src" / "nbc"
    tests_root = repo_root / "tests"

    missing: list[str] = []
    for namespace in SEED_NAMESPACES:
        source_dir = package_root / namespace
        has_code = any(
            path.name != "__init__.py" for path in source_dir.glob("*.py")
        )
        if has_code and not (tests_root / namespace).is_dir():
            missing.append(f"src/nbc/{namespace}/ has modules but tests/{namespace}/ does not exist")

    assert not missing, "; ".join(missing)
