"""The builder's import rule, and the two claims about the hub that only the hub can settle.

Three separate things are under test:

1. `datasets` is imported inside `corpus/build.py` and nowhere else, and importing the module does
   not drag it in. Both halves matter: an AST scan cannot see that a top-level import happened,
   and a `sys.modules` check cannot see a second module that would import it under some branch.
2. `datasets` is declared as an optional dependency group, so the measurement runtime never
   acquires it.
3. `smoke` only: one small exclusion source really loads at its pinned revision, and the
   access-restricted one really answers 401. Those are facts about the world, and `pins.toml`
   records both as declarations -- this is where the declaration is compared to the thing it
   describes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import nbc
from nbc.corpus import build
from nbc.corpus.exclusion import Observation, normalized_texts, plan, verify_observations
from nbc.pins import EXCLUSION_UNREACHABLE, HTTP_OK, load_pins

BUILDER = Path(build.__file__).resolve()
DEPENDENCY = "datasets"
OPTIONAL_GROUP = "build"


def _source_files() -> list[Path]:
    """Every Python file that ships, the same scope the pin-literal scans use."""
    root = Path(nbc.__file__).resolve().parents[2]
    return sorted(
        path
        for directory in ("src", "spikes")
        for path in (root / directory).rglob("*.py")
    )


def _imported_names(path: Path) -> set[str]:
    """Top-level module names imported anywhere in the file, function bodies included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_builder_imports_the_dataset_library() -> None:
    """A second importer is a second module the offline runtime would have to carry."""
    importers = [
        path.name for path in _source_files() if DEPENDENCY in _imported_names(path)
    ]

    assert importers == [BUILDER.name], importers


def test_the_builder_imports_it_inside_a_function_rather_than_at_module_scope() -> None:
    """The AST answer. A top-level import would satisfy the scan above and still be wrong."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top_level.add(node.module.split(".")[0])

    assert DEPENDENCY not in top_level


def test_importing_the_builder_leaves_the_dataset_library_out_of_sys_modules() -> None:
    """The runtime answer, in a child process, because this one may already have imported it."""
    code = f"import sys, nbc.corpus.build; print({DEPENDENCY!r} in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "False", completed.stdout


def test_the_dataset_library_is_declared_as_an_optional_group(repo_root: Path) -> None:
    """Read from `pyproject.toml` as data, so a move into the runtime deps fails here."""
    document = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]

    optional = project["optional-dependencies"][OPTIONAL_GROUP]
    assert any(requirement.startswith(DEPENDENCY) for requirement in optional), optional
    assert not any(
        requirement.startswith(DEPENDENCY) for requirement in project["dependencies"]
    ), project["dependencies"]


def test_the_probe_url_names_the_revision_where_there_is_one() -> None:
    """The unreadable source has no sha, so the only URL it has is the bare repository."""
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    by_repository = {source.repository: source for source in pins.exclusion_sources}

    for source in by_repository.values():
        if source.revision:
            assert source.probe_url.endswith(f"/revision/{source.revision}")
        else:
            assert source.availability == EXCLUSION_UNREACHABLE
            assert source.probe_url.endswith(f"/{source.repository}")


def test_a_row_is_walked_for_every_string_it_holds() -> None:
    """One pinned source keeps its text inside nested role/content records, not a column."""
    row = {
        "prompt": "top level",
        "messages": [{"role": "user", "content": "nested"}],
        "score": 3,
        "tags": ["a", "b"],
    }

    # Values, never keys, and every depth. `"user"` is in there and that is the documented cost:
    # a short label value enters the index, so a corpus row that *is* that word would be removed.
    assert sorted(build._strings_in(row)) == ["a", "b", "nested", "top level", "user"]


# --- the world, once, in the smoke tier -------------------------------------------------------


@pytest.mark.smoke
def test_every_source_really_answers_the_status_the_pins_declare() -> None:
    """`pins.toml` records a status per source and caveat 3d publishes one of them.

    This is the comparison behind the claim, over every declared source rather than the
    interesting one: a status recorded and never asked about is the defect this repository keeps
    finding in itself.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])

    observed = {source.repository: build.probe(source) for source in pins.exclusion_sources}
    declared = {
        source.repository: source.http_status for source in pins.exclusion_sources
    }

    assert observed == declared


@pytest.mark.smoke
def test_the_source_the_pins_call_unreadable_really_will_not_load() -> None:
    """One source resolves at its sha and still refuses to hand over rows. Checked, not asserted.

    Without this, `unreadable` would be a word in a file. With it, a repository that stopped
    being a loading script fails here and the pins get re-read -- which is the direction that
    matters, because that source is one of the four seeds and every row of it that this filter
    cannot see is a row that stays in the corpus.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    unreadable = [
        source
        for source in pins.exclusion_sources
        if source.revision and not source.loadable
    ]

    assert unreadable, "the pins declare no unreadable source; this test has lost its subject"
    for source in unreadable:
        assert build.probe(source) == HTTP_OK
        with pytest.raises(Exception) as refusal:
            next(build.iter_exclusion_texts(source))
        assert not isinstance(refusal.value, StopIteration), (
            f"{source.repository} loaded and produced no rows, which is a different fault"
        )


@pytest.mark.smoke
def test_one_small_source_really_loads_at_its_pinned_revision() -> None:
    """The loader against the real hub, on one small declared source, once.

    Small on purpose: the point is that `get_dataset_config_names` plus `load_dataset` at a
    pinned revision returns text this filter can index, not that a hundred megabytes downloads.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    smallest = min(
        (source for source in pins.exclusion_sources if source.loadable),
        key=lambda source: source.repository,
    )

    assert build.probe(smallest) == HTTP_OK
    keys = normalized_texts(build.iter_exclusion_texts(smallest))

    assert keys
    verify_observations(
        [entry for entry in plan(pins) if entry.key == smallest.key],
        {smallest.key: Observation(HTTP_OK, loadable=True, texts_loaded=len(keys))},
    )
