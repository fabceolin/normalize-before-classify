"""The tests mirror the source tree, and the mirror is well defined.

Story 2.5 asks for `tests/canon/test_<stage>.py` per stage. That convention is worth checking rather
than remembering, for one reason: a module added to `canon/` with no test file beside it does not
fail anything. Nothing goes red. The suite stays green and the layer grows a piece nobody exercised,
which is the quietest way a checked artifact stops being one.

Two rules, and the second is what makes the first meaningful:

- **Every module has its mirror.** `canon/stages/decode.py` requires `tests/canon/test_decode.py`.
- **No two modules share a stem.** The mirror is flat — `tests/canon/` has no `stages/`
  subdirectory — so `canon/decode.py` alongside `canon/stages/decode.py` would make one test file
  the mirror of two modules, and the first rule would pass while one of them went untested. That is
  the input this rule exists for, and it is written as a test below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import canon_scan

MODULES = [path for path in canon_scan.canon_modules() if path.name != "__init__.py"]
"""`__init__.py` is excluded: `canon/__init__.py` and `canon/stages/__init__.py` carry the package
docstrings and no logic, they share a stem by construction, and the bound they do state is checked
by `test_import_bound.py`."""


def mirror_of(path: Path) -> Path:
    """The test file a module requires: flat under `tests/canon/`, named for the module's stem."""
    return canon_scan.TESTS / f"test_{path.stem}.py"


def test_the_scan_found_the_modules_it_is_supposed_to_mirror() -> None:
    # Eight modules, and this list is what stops the parametrized check below from being a loop
    # over an empty sequence.
    assert {path.name for path in MODULES} == {
        "pipeline.py",
        "edits.py",
        "confusables_table.py",
        "vendor_confusables.py",
        "invisible.py",
        "confusables.py",
        "nfkc.py",
        "decode.py",
    }


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_canon_module_has_its_mirrored_test_file(path: Path) -> None:
    mirror = mirror_of(path)
    assert mirror.is_file(), (
        f"{path.relative_to(canon_scan.REPO_ROOT)} has no mirror at "
        f"{mirror.relative_to(canon_scan.REPO_ROOT)}"
    )
    assert mirror.stat().st_size > 0


def test_no_two_canon_modules_share_a_stem() -> None:
    """The rule that makes a flat mirror well defined, and the failure it prevents.

    Adding `canon/decode.py` beside `canon/stages/decode.py` would leave both pointing at
    `tests/canon/test_decode.py`. Every module would still have "its" mirror and one of them would
    have no tests at all.
    """
    stems = [path.stem for path in MODULES]
    duplicated = sorted({stem for stem in stems if stems.count(stem) > 1})
    assert duplicated == [], duplicated


def test_the_mirror_rule_maps_a_nested_module_to_a_flat_test_file() -> None:
    # The rule itself, stated as an example, so the convention is readable without inferring it
    # from the directory listing.
    assert mirror_of(canon_scan.CANON / "stages" / "decode.py").name == "test_decode.py"
    assert mirror_of(canon_scan.CANON / "pipeline.py").name == "test_pipeline.py"


def test_the_stem_rule_reports_a_collision(tmp_path: Path) -> None:
    """The input that makes the collision rule fail, since the real tree must never contain it."""
    stems = [path.stem for path in [tmp_path / "decode.py", tmp_path / "stages" / "decode.py"]]
    duplicated = sorted({stem for stem in stems if stems.count(stem) > 1})
    assert duplicated == ["decode"]


def test_every_mirrored_test_file_imports_the_module_it_mirrors() -> None:
    """A mirror that never imports its module is a file with the right name and the wrong content.

    Read from the import graph, not from the text: a module name inside a docstring is not an
    import, and this repository has already paid once for a check that could not tell the
    difference. Any spelling counts, because they all reach the same module --
    `from nbc.canon.stages.decode import run`, `from nbc.canon.stages import decode`, or
    `import nbc.canon.pipeline`.
    """
    missing = []
    for path in MODULES:
        dotted = ".".join(path.relative_to(canon_scan.SRC).with_suffix("").parts)
        imported = canon_scan.imported_dotted(canon_scan.parse(mirror_of(path)))
        if dotted not in imported:
            missing.append(f"{mirror_of(path).name} does not import {dotted}")
    assert missing == []


def test_the_import_check_reads_every_spelling_of_an_import() -> None:
    import ast

    tree = ast.parse(
        "import nbc.canon.pipeline\n"
        "from nbc.canon.stages import decode\n"
        "from nbc.canon.stages.nfkc import run\n"
    )
    imported = canon_scan.imported_dotted(tree)
    assert {
        "nbc.canon.pipeline",
        "nbc.canon.stages.decode",
        "nbc.canon.stages.nfkc",
    } <= imported
    assert "nbc.canon.edits" not in imported
