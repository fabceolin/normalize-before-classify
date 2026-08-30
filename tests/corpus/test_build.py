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
import dataclasses
import io
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import nbc
from nbc.corpus import attack, build, crosscheck
from nbc.corpus.exclusion import Observation, normalized_texts, plan, verify_observations
from nbc.corpus.matrix import CHAINS, HELDOUT_CHAINS, render_chain
from nbc.errors import exit_code_for
from nbc.pins import (
    EXCLUSION_UNREACHABLE,
    HTTP_OK,
    PoolRowRef,
    WithdrawnText,
    load_pins,
)
from nbc.schema import FAMILY_ATTACK

BUILDER = Path(build.__file__).resolve()
DEPENDENCY = "datasets"
OPTIONAL_GROUP = "build"

CONTRADICTIONS_IN_THE_PINNED_POOL = 2
"""How many texts the pinned attack pool carries under both labels, read once by a human.

Two: a DAN jailbreak and a "Caveat Emptor" docker prompt, each present as one attack row and one
benign row. The number is here rather than derived from the pool so that the smoke test compares
two different things -- what a human found at the pinned revision, and what the loader finds now.
Deriving it from the pool would make the assertion true by construction on any pool at all.
"""


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


# --- AD-1: one writer, and a label nobody typed -----------------------------------------------

# The two modules allowed to write a file at all, and what each one writes. A pair rather than a
# bare list: `vendor_confusables` writes the vendored table into the package and is the reason the
# rule cannot be "nothing but the builder writes", so its subject is named here instead of being
# an unexplained exemption.
WRITERS: dict[str, str] = {
    "build.py": "data/*.jsonl, data/manifest.json and data/ATTRIBUTION.md, per AD-1 and AD-34",
    "vendor_confusables.py": "the vendored confusables table under canon/data/",
    "run.py": (
        "story 4.2's scoring pass writes results/scores-<n>-<i>.jsonl and results/scores.jsonl. "
        "It writes nothing under data/ and cannot: the locator scan in tests/corpus/test_manifest "
        "refuses any module outside the two above that can even name a corpus file, so AD-1's "
        "one-writer rule is untouched by this entry"
    ),
}

_WRITE_ATTRIBUTES = frozenset({"write_text", "write_bytes", "writelines"})


def _write_primitive_lines(source: str, filename: str) -> list[int]:
    """Lines at which `source` calls something that puts bytes on disk.

    Structural, not textual: an `ast.Call` whose function is one of the path write methods, or
    `open(...)` with a mode that is not read-only. A grep for `write` would fire on `write_fields`
    and miss `getattr(path, "write_" + kind)`; this fires on the call shape.
    """
    tree = ast.parse(source, filename=filename)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _WRITE_ATTRIBUTES:
            lines.append(node.lineno)
        elif isinstance(function, ast.Name) and function.id == "open":
            modes = [
                argument.value
                for argument in list(node.args[1:2])
                + [keyword.value for keyword in node.keywords if keyword.arg == "mode"]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            ]
            if any(set(mode) & set("wax+") for mode in modes):
                lines.append(node.lineno)
    return sorted(lines)


def test_the_write_scan_fires_on_a_file_that_writes() -> None:
    """The scan's own failing input, so it cannot pass by failing to look.

    Three shapes, because a scan that caught only the first would let the other two through: a
    `Path.write_text`, an `open(..., "w")`, and a mode passed by keyword.
    """
    offender = (
        "from pathlib import Path\n"
        "def go(p):\n"
        "    Path(p).write_text('')\n"
        "    with open(p, 'w') as handle:\n"
        "        handle.write('x')\n"
        "    open(p, mode='a').close()\n"
    )
    assert _write_primitive_lines(offender, "<offender>") == [3, 4, 6]
    assert _write_primitive_lines("x = 1\nprint(open('p').read())\n", "<clean>") == []


def test_only_the_declared_writers_put_bytes_on_disk() -> None:
    """AD-1: `corpus/build.py` is the only writer of `data/*.jsonl`.

    Enforced as the stronger property that is actually checkable from the source tree -- which
    files write **anything** -- because "writes a path that ends in .jsonl" is a claim about a
    runtime value and this is a claim about the code. A second module acquiring a write is caught
    here whether or not the reviewer can tell where its path came from.
    """
    writing = {
        path.name
        for path in _source_files()
        if _write_primitive_lines(path.read_text(encoding="utf-8"), str(path))
    }

    assert writing == set(WRITERS), sorted(writing.symmetric_difference(WRITERS))


def _corpus_item_label_arguments(source: str, filename: str) -> list[ast.expr]:
    """The `label=` argument of every `CorpusItem(...)` call in `source`."""
    tree = ast.parse(source, filename=filename)
    labels: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CorpusItem"
        ):
            found = [
                keyword.value for keyword in node.keywords if keyword.arg == "label"
            ]
            assert found, f"{filename}:{node.lineno} builds a CorpusItem with no label"
            labels.extend(found)
    return labels


def test_the_label_scan_rejects_a_label_read_off_the_source_row() -> None:
    """The failing input for the rule below, before the rule is applied to the real tree."""
    offender = "CorpusItem(text=t, label=row.label)\n"
    (argument,) = _corpus_item_label_arguments(offender, "<offender>")
    assert not isinstance(argument, ast.Name)


def test_every_gold_label_in_the_tree_names_a_schema_constant() -> None:
    """FR4, made structural: no gold label is a value read from somebody else's annotation.

    A literal `1` would be just as wrong as `row.label` and is refused the same way: the constant
    has to be named, so a reader following the name arrives at `schema.py` and at the paragraph
    saying why the dataset's `attack_label` is a different fact.
    """
    admissible = {"ATTACK", "BENIGN"}
    seen = 0
    for path in _source_files():
        for argument in _corpus_item_label_arguments(
            path.read_text(encoding="utf-8"), str(path)
        ):
            seen += 1
            assert isinstance(argument, ast.Name), (
                f"{path}: label must name a schema constant, got {ast.dump(argument)}"
            )
            assert argument.id in admissible, f"{path}: label names {argument.id}"

    assert seen, "no CorpusItem is constructed anywhere, so this scan passed vacuously"


# --- the CLI --------------------------------------------------------------------------------


def test_rebuilding_is_its_own_subcommand_and_not_a_flag_on_another() -> None:
    """AD-1: a rebuild is an explicit act, so it cannot be reached by running the ordinary build."""
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.corpus.build", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for subcommand in ("exclusion-report", "build-attack", "rebuild-attack"):
        assert subcommand in completed.stdout, completed.stdout


def test_the_cli_refuses_to_run_with_no_subcommand() -> None:
    """Exit 2 is argparse's usage error. A default subcommand would make one of them a side effect."""
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.corpus.build"], capture_output=True, text=True
    )
    assert completed.returncode == 2, completed.stderr


# --- the world, once, in the smoke tier -------------------------------------------------------


@pytest.mark.smoke
def test_the_committed_withdrawals_describe_the_pinned_pool_exactly() -> None:
    """The one claim in `pins.toml` that only the artifact can settle.

    Five row positions, five labels and two SHA-256 digests, asserted against the dataset the pins
    name rather than against a fixture. Everything else about the withdrawal mechanism is covered
    offline; this is the half that goes stale silently when somebody re-pins the pool, and it is
    the same check the CI `attack-pool-report` step runs.

    Both directions are covered by the abort this test does not see: `withdraw` reports a declared
    row the pool does not carry and a row the pool carries that no declaration names.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    dataset = pins.attack_datasets[0]
    assert dataset.withdrawn, "the pins declare no withdrawal; this test has lost its subject"

    pool, observed_splits, withdrawn = build.read_attack_pool(dataset)
    assert attack.verify_splits(dataset.splits, observed_splits) == ()

    # Reaching this line at all is most of the check: `read_attack_pool` aborts when the
    # declaration and the pool disagree in either direction. What is left is that the rows it
    # removed are the rows the declaration names, position for position and label for label.
    assert {(row.split, row.index, row.label) for row in withdrawn} == {
        (row.split, row.index, row.label)
        for entry in dataset.withdrawn
        for row in entry.rows
    }
    # And nothing else in the pool contradicts itself: if it did, FR4 would stop the build and
    # this file's declaration would be describing a question that is no longer the only one.
    assert attack.contradictions(pool) == ()


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


@pytest.mark.smoke
def test_the_pinned_pool_yields_exactly_the_splits_the_pins_declare() -> None:
    """The declaration compared against the thing it describes, rather than against itself.

    `read_attack_pool` returns the splits it observed; `pins.toml` declares the splits the counts
    are taken over. The comparison is the point: a dataset that grew a `validation` split, or lost
    one, changes what "over every split" means and nothing else would notice.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    dataset = pins.attack_datasets[0]

    rows, observed, _withdrawn = build.read_attack_pool(dataset)

    assert attack.verify_splits(dataset.splits, observed) == ()
    assert rows
    assert {row.label for row in rows} >= {dataset.attack_label}


@pytest.mark.smoke
def test_the_pinned_pool_still_carries_the_two_texts_a_human_ruled_on_and_no_others() -> None:
    """FR4's second case, on the real pool. It is not hypothetical and this is where that is shown.

    The count is asserted against a reviewed constant rather than against whatever the pool
    happens to hold: a pool that grew a third contradiction, or lost these two, is a different
    gold-label situation and a human has to look at it again. `pins.toml` pins the revision, so
    the only way this number moves is a pin edit.

    That the build now completes is because two texts were withdrawn by a person, never because
    the gate was loosened, and this is the pair of assertions that says so against the real
    dataset rather than against a fixture.
    """
    pins = load_pins(Path(nbc.__file__).resolve().parents[2])
    dataset = pins.attack_datasets[0]

    pool, observed, withdrawn = build.read_attack_pool(dataset)
    # The artifact as published, reconstructed from the two halves the one door returns: this
    # asks about the dataset, not about the dataset after this repository's own decisions.
    as_published = pool + withdrawn
    problems = attack.contradictions(as_published)

    assert len(problems) == CONTRADICTIONS_IN_THE_PINNED_POOL, problems
    # And the pool the build actually uses carries none of them, because both were withdrawn by a
    # person on 2026-08-30. The pair is the point: the build completes because of a decision, not
    # because the gate stopped seeing anything.
    assert attack.contradictions(pool) == ()

    # The gate is reached from the top of the pipeline, before anything is downloaded: the thunk
    # would raise if it were called, so the abort proves the order as well as the check. The
    # declaration is dropped for this call rather than the rows being put back, because an
    # unfiltered pool beside a live declaration trips the withdrawal precondition first -- which
    # is a different abort, and this test is about this one.
    def must_not_be_called() -> object:
        raise AssertionError("the exclusion index was built for a pool that already failed")

    unruled = dataclasses.replace(dataset, withdrawn=())
    with pytest.raises(attack.LabelContradiction) as caught:
        attack.draw_attack_items(as_published, observed, unruled, must_not_be_called)
    assert exit_code_for(caught.value) == attack.LabelContradiction.exit_code


# --- the build end to end, with the two remote reads replaced ------------------------------------


def _fake_pool() -> tuple[
    tuple[attack.PoolRow, ...], tuple[str, ...], tuple[attack.PoolRow, ...]
]:
    """A pool with the shape of the real one -- two splits, mixed labels -- and no contradiction.

    The third element is `read_attack_pool`'s withdrawn half, empty here: these fixtures declare
    no withdrawal, so there is nothing for the door to have removed.
    """
    rows = [
        attack.PoolRow(split="train", index=index, text=f"payload {index}", label=1)
        for index in range(6)
    ] + [
        attack.PoolRow(split="test", index=index, text=f"benign {index}", label=0)
        for index in range(3)
    ]
    return tuple(rows), ("train", "test"), ()


@pytest.fixture()
def offline_build(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> Path:
    """`build_attack_corpus` with its two network reads replaced, so the wiring runs offline.

    Only the two reads are replaced. The gates, the draw, the ordering, the report assembly, the
    refusal and the writer are the shipped ones -- replacing more would test the fake.
    """
    monkeypatch.setattr(build, "read_attack_pool", lambda dataset: _fake_pool())
    monkeypatch.setattr(
        build,
        "read_exclusion_index",
        lambda pins: (
            build.build_index({}),
            build.plan(pins),
            {entry.key: build.Observation(HTTP_OK, loadable=True, texts_loaded=1)
             for entry in build.plan(pins)},
        ),
    )
    return repo_root


def _over_a_fake_pool(dataset):
    """The committed attack pin with its withdrawals dropped, for a fixture that fakes the pool.

    The licence is **not** touched, and that is the point of this helper existing in this shape.
    It used to answer the licence question too, because the committed pool declared none and
    AD-34's gate aborted every build; that question was answered for real on 2026-08-30, so the
    fixture stopped needing to pretend and now exercises the committed declaration itself.

    What it still has to drop is the withdrawals. `corpus/attack.py::withdraw` compares the
    declared digests and rows against the pool as read and aborts when they disagree in either
    direction -- deliberately, so a withdrawal cannot quietly describe nothing -- and every pool
    below is six invented payloads that hash to nothing the declaration names. Dropping them here
    is honest about that; keeping them would mean loosening the comparison in production code so a
    test could pass, which is the trade this project does not make.

    The withdrawals are exercised against the real texts in `tests/corpus/test_attack.py`, and
    against the real dataset by the CI step that runs `attack-pool-report`.
    """
    return dataclasses.replace(dataset, withdrawn=())


def _pins_with_a_small_draw(root: Path) -> object:
    """The committed pins, with the draw shrunk to what the fake pool can satisfy.

    The declared size is 1200 and the fake pool holds six positives: shrinking it here keeps the
    fixture from being a second copy of the committed number, which the no-literals scan forbids
    anyway.
    """
    pins = load_pins(root)
    dataset = pins.attack_datasets[0]
    small = dataclasses.replace(
        _over_a_fake_pool(dataset),
        draw=dataclasses.replace(dataset.draw, sample_size_positives=4),
    )
    return dataclasses.replace(pins, attack_datasets=(small,))


def test_the_build_writes_the_corpus_and_reports_what_it_drew(
    offline_build: Path, tmp_path: Path
) -> None:
    pins = _pins_with_a_small_draw(offline_build)
    draw_report, exclusion_report, path, written = build.build_attack_corpus(
        pins, root=str(tmp_path)
    )

    assert path == tmp_path / build.DATA_DIRNAME / build.ATTACK_CORPUS_FILENAME
    # The corpus, its generated credits, and nothing else. CI refuses a dirty tree, and a builder
    # that dropped a cache or a temporary file beside the corpus would fail there with no
    # diagnosis; here it fails naming the file. `data/manifest.json` is story 3.6's and is
    # deliberately not here -- this half carries no manifest, which is what keeps it unreadable
    # through the guarded door. `ATTRIBUTION.md` IS here, because AD-34's rule is about rows on
    # disk and these rows are on disk.
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        f"{build.DATA_DIRNAME}/{build.ATTRIBUTION_FILENAME}",
        f"{build.DATA_DIRNAME}/{build.ATTACK_CORPUS_FILENAME}",
    ]
    assert written == len(path.read_bytes())

    fields = draw_report.as_run_fields()["attack_draw"]
    assert fields["rows_by_split"] == {"test": 3, "train": 6}
    assert fields["positives_by_split"] == {"train": 6}
    assert fields["unique_positives"] == 6
    assert fields["drawn_positives"] == 4
    # Three units, and the report carries all three so a reader can multiply them out: rows of
    # the pool, positives drawn from it, and one corpus row per drawn positive per declared
    # chain. `chains` is the dressing axis of the headline table, published beside the corpus.
    assert fields["chains"] == [render_chain(chain) for chain in CHAINS[FAMILY_ATTACK]]
    # AD-28's block travels as its own axis and is never merged into the bound one: `chain_class`
    # is part of the cell key (AD-2) and no function aggregates across it (AD-11).
    assert fields["held_out_chains"] == [
        render_chain(chain) for chain in HELDOUT_CHAINS[FAMILY_ATTACK]
    ]
    assert fields["items_written"] == 4 * (
        len(CHAINS[FAMILY_ATTACK]) + len(HELDOUT_CHAINS[FAMILY_ATTACK])
    )
    # And the report is self-consistent read on its own terms, which is how a reader of
    # `results.json` will read it -- without the constant in front of them.
    assert fields["items_written"] == fields["drawn_positives"] * (
        len(fields["chains"]) + len(fields["held_out_chains"])
    )
    assert len(path.read_text(encoding="utf-8").splitlines()) == fields["items_written"]
    # The exclusion accounting travels with the draw: FR3.3's counts are published beside the
    # corpus they shaped, not in a separate run.
    assert exclusion_report.as_run_fields()["exclusion"]["rows_in"] == 6


def test_a_second_build_refuses_and_an_explicit_rebuild_is_byte_identical(
    offline_build: Path, tmp_path: Path
) -> None:
    """AD-1's two halves: never a side effect, and the same pins produce the same bytes."""
    pins = _pins_with_a_small_draw(offline_build)
    _report, _exclusion, path, _written = build.build_attack_corpus(pins, root=str(tmp_path))
    before = path.read_bytes()

    with pytest.raises(build.CorpusWriteRefused) as caught:
        build.build_attack_corpus(pins, root=str(tmp_path))
    assert exit_code_for(caught.value) == build.CorpusWriteRefused.exit_code

    build.build_attack_corpus(pins, root=str(tmp_path), rebuild=True)
    assert path.read_bytes() == before


# --- the benign-code archive reader (story 3.6) ------------------------------------------------
#
# Offline: the archive is built in the test and handed to the reader through a stub `urlopen`, so
# the whole streaming path -- the leading component, the size band, the non-UTF-8 member, the entry
# limit -- is covered by a suite that never opens a socket.


def _archive(members: dict[str, bytes], root: str = "repo-abc") -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serving(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload)
    )


def _repository():
    from nbc.pins import BenignCodeRepository, Licence

    return BenignCodeRepository(
        key="example-code",
        repository="example/code",
        revision="c" * 40,
        licence=Licence(
            identifier="MIT", source="fixture", attribution="fixture", redistributed=True
        ),
    )


def test_the_archive_reader_drops_the_leading_component_and_applies_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path a reader can open on the web, and only the members the frame could ever accept."""
    _serving(
        monkeypatch,
        _archive(
            {
                "src/a.js": b"x" * 300,
                "src/tiny.js": b"x" * 10,
                "src/huge.js": b"x" * 5000,
            }
        ),
    )
    files = list(build.read_repository_files(_repository(), 100, 1000))
    assert [entry.path for entry in files] == ["src/a.js"]


def test_a_member_that_is_not_utf8_is_skipped_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`UnicodeDecodeError` is a `ValueError`, and the narrow name alone would let it escape."""
    _serving(
        monkeypatch,
        _archive({"src/a.js": b"\xff\xfe" + b"x" * 300, "src/b.js": b"y" * 300}),
    )
    files = list(build.read_repository_files(_repository(), 100, 1000))
    assert [entry.path for entry in files] == ["src/b.js"]


def test_an_archive_with_too_many_entries_aborts_rather_than_being_read_in_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository read halfway contributes a candidate set its sha does not describe."""
    monkeypatch.setattr(build, "ARCHIVE_MEMBER_LIMIT", 2)
    _serving(
        monkeypatch,
        _archive({f"src/f{index}.js": b"x" * 300 for index in range(5)}),
    )
    with pytest.raises(build.RepositoryUnreadable, match="entries"):
        list(build.read_repository_files(_repository(), 100, 1000))


def test_a_sha_the_host_no_longer_serves_aborts_naming_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error
    import urllib.request

    def refuse(*args, **kwargs):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(build.RepositoryUnreadable, match="example/code"):
        list(build.read_repository_files(_repository(), 100, 1000))


def test_an_unreadable_archive_aborts_rather_than_contributing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository silently contributing zero is a frame drawing from fewer sources than declared."""
    _serving(monkeypatch, b"this is not a gzip stream")
    with pytest.raises(build.RepositoryUnreadable, match="readable archive"):
        list(build.read_repository_files(_repository(), 100, 1000))


def test_the_repository_abort_has_an_exit_code_distinct_from_every_other() -> None:
    from nbc.errors import declared_exit_codes

    codes = declared_exit_codes()
    assert codes[build.RepositoryUnreadable.exit_code] is build.RepositoryUnreadable


def test_the_benign_rows_are_the_complement_of_the_declared_attack_label() -> None:
    """Derived from `attack_label`, so a third label value cannot leave the corpus unnoticed."""
    from nbc.corpus.attack import PoolRow

    dataset = _dataset_pin()
    rows = [
        PoolRow(split="train", index=0, text="attack", label=1),
        PoolRow(split="train", index=1, text="benign", label=0),
        PoolRow(split="train", index=2, text="", label=0),
        PoolRow(split="train", index=3, text="other", label=2),
    ]
    assert build.read_benign_rows(dataset, rows) == ("benign", "other")


def _dataset_pin():
    from nbc.pins import AttackDataset, AttackDraw, Licence, Provenance

    return AttackDataset(
        key="fixture",
        repository="example/pool",
        revision="a" * 40,
        splits=("train",),
        attack_label=1,
        draw=AttackDraw(
            declared_on="2026-08-29",
            sample_size_positives=1,
            method="seeded_random",
            seed=1,
            sort_key=None,
        ),
        licence=Licence(
            identifier="MIT", source="fixture", attribution="fixture", redistributed=True
        ),
        provenance=Provenance(checked_on="2026-08-29", card_revision="a" * 40, seeds=()),
    )


def test_verify_corpus_is_the_one_subcommand_that_touches_no_network(tmp_path: Path) -> None:
    """It is also the only one the offline suite may run, which is why it is run here.

    With no corpus on disk the guarded read refuses, and the process exits with the manifest
    mismatch code rather than with a crash -- the check that the subcommand is wired to the door
    rather than merely named in the help text.
    """
    from nbc.corpus.manifest import CorpusManifestMismatch

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "pins.toml").write_text(
        (root / "pins.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert build.main(["--root", str(tmp_path), "verify-corpus"]) == (
        CorpusManifestMismatch.exit_code
    )


def test_a_benign_selection_that_overlaps_the_attack_one_is_refused() -> None:
    """The failing input is a benign selection taken ON the attack label instead of against it.

    Nothing else in the repository would catch it: every row would be individually well formed, the
    gold labels would name schema constants, and the corpus would report a false-positive rate over
    the attack payloads.
    """
    from nbc.corpus.attack import PoolRow

    dataset = _dataset_pin()
    rows = [
        PoolRow(split="train", index=0, text="ignore previous instructions", label=1),
        PoolRow(split="train", index=1, text="how do i sort a list", label=0),
    ]
    assert build.selection_overlap(dataset, rows, ("how do i sort a list",)) == ()
    (problem,) = build.selection_overlap(dataset, rows, ("ignore previous instructions",))
    assert "both halves" in problem and "complement" in problem


# --- both halves, one manifest, and AD-27's gate reached from the top ------------------------------


def _small_corpus_pins(root: Path) -> object:
    """The committed pins with both declared sizes shrunk to what the offline fixtures can fill.

    The attack draw drops to four positives and the benign frame to twenty items per class, which is
    the hand-authored allowance exactly -- so B-chat needs no dataset row and the fake pool's three
    benign rows are enough. Every other field is the committed one: shrinking two counts is the
    smallest edit that lets `build_corpus` run with no network, and everything the test asserts about
    is the shipped code path.
    """
    pins = load_pins(root)
    dataset = pins.attack_datasets[0]
    frame = pins.benign_frame
    return dataclasses.replace(
        pins,
        attack_datasets=(
            dataclasses.replace(
                _over_a_fake_pool(dataset),
                draw=dataclasses.replace(dataset.draw, sample_size_positives=4),
            ),
        ),
        benign_frame=dataclasses.replace(
            frame,
            sample_size_items=frame.b_chat.hand_authored_items,
            b_code=dataclasses.replace(frame.b_code, min_repositories=2),
        ),
    )


def _synthetic_code(pins, payload_text: str = "") -> dict[str, tuple]:
    """Twenty eligible-looking B-code files over two repositories, ten each: the frame's cap exactly.

    The two repositories are the **first two the frame pins**, and each file names itself through
    `BenignCodeRepository.file_source`, so every synthetic row carries a source AD-34's attribution
    can account for. An invented repository id would abort the build here -- correctly, because a
    committed row whose source is not a pinned source is a row published with no licence -- and the
    fixture would then be testing the licence gate instead of the builder.

    `payload_text` is prepended to the first file only. Empty by default, so the happy path and the
    abort path differ in exactly the one thing under test.
    """
    from nbc.corpus.benign import CodeFile

    files: dict[str, tuple] = {}
    for index, repository in enumerate(pins.benign_frame.b_code.repositories[:2]):
        entries = []
        for number in range(10):
            marker = f"{repository.key}-{number}"
            head = f"// {payload_text}\n" if (index, number) == (0, 0) and payload_text else ""
            path = f"src/{marker}.js"
            entries.append(
                CodeFile(
                    repository_key=repository.key,
                    source=repository.file_source(path),
                    path=path,
                    text=(
                        f"{head}// {marker}\n"
                        "const TOKEN = 'aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Q=';\n"
                        + f"let n{number} = {number};\n" * 8
                    ),
                )
            )
        files[repository.key] = tuple(entries)
    return files


@pytest.fixture()
def offline_full_build(offline_build: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`offline_build` plus the third remote read, so `build_corpus` runs end to end offline."""
    monkeypatch.setattr(build, "read_benign_code", _synthetic_code)
    return offline_build


def test_the_full_build_writes_both_halves_and_reports_the_cross_check(
    offline_full_build: Path, tmp_path: Path
) -> None:
    """`build_corpus` end to end, which also proves AD-27's payloads reach the benign draw.

    Without the threading the benign draw would raise `BenignItemMislabelled` for having no payloads
    at all, so a build that completes is a build in which the drawn attack payloads arrived.
    """
    pins = _small_corpus_pins(offline_full_build)
    manifest, fields = build.build_corpus(pins, root=str(tmp_path))

    assert manifest.frame_id == pins.benign_frame.frame_id
    cross_check = fields["benign_draw"]["cross_check"]
    assert cross_check["payloads_checked"] == 4
    assert cross_check["sources_checked"] == pins.benign_frame.sample_size_items * 2
    assert cross_check["metric"] == crosscheck.CROSS_CHECK_METRIC

    # And it is recorded *with the corpus*, not only returned to the caller: M-02's whole point is
    # that a rebuild at a different threshold must be visible in the committed tree.
    written = json.loads(
        (tmp_path / build.DATA_DIRNAME / build.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    recorded = written["reports"]["benign_draw"]["cross_check"]
    assert recorded == cross_check
    assert recorded["similarity_threshold"] == crosscheck.SIMILARITY_THRESHOLD
    assert recorded["shingle_width"] == crosscheck.SHINGLE_WIDTH


def test_a_withdrawn_text_reaches_neither_half_of_the_corpus(
    offline_full_build: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression the first full build of 2026-08-30 found, pinned so it stays found.

    `withdraw` used to run inside `draw_attack_items`. The attack half got the filtered pool and
    `read_benign_rows` -- reading the same tuple one call later -- got the unfiltered one, so both
    withdrawn texts went straight back into the corpus under a benign gold label. Nothing caught
    it offline. The live build stopped only because `selection_overlap` fired on a text that was
    in both halves, which is a report about a symptom two steps downstream of the cause.

    The pool now has one reader and `read_attack_pool` filters at that door, so the check is that
    the text is in **neither** half. Reverting the placement turns this red.
    """
    contradicted = "the text a human ruled on"
    withdrawal = WithdrawnText(
        text_sha256=attack.text_digest(contradicted),
        rows=(
            PoolRowRef(split="train", index=6, label=1),
            PoolRowRef(split="test", index=3, label=0),
        ),
        on="2026-08-30",
        by="a named human",
        reason="fixture: contradicted, so neither row is usable",
    )

    def pool_with_the_contradiction(dataset):
        """`read_attack_pool`'s shipped tail over a fake read: the same `withdraw`, same abort."""
        rows, observed, _empty = _fake_pool()
        rows = rows + (
            attack.PoolRow(split="train", index=6, text=contradicted, label=1),
            attack.PoolRow(split="test", index=3, text=contradicted, label=0),
        )
        surviving, problems = attack.withdraw(rows, dataset.withdrawn)
        assert problems == (), problems
        kept = set(surviving)
        return surviving, observed, tuple(row for row in rows if row not in kept)

    monkeypatch.setattr(build, "read_attack_pool", pool_with_the_contradiction)

    # The wiring itself, not only its outcome. The bug was that the benign half read a *different*
    # tuple from the one the attack half was handed, so what this records is which pool
    # `read_benign_rows` was actually given -- and the assertion below is that it is the one the
    # door returned. Moving the withdrawal back inside `draw_attack_items` makes this the
    # unfiltered pool again, and it goes red here rather than two steps downstream.
    handed: list[object] = []
    real_read_benign_rows = build.read_benign_rows

    def recording_read_benign_rows(dataset, rows):
        handed.append(tuple(rows))
        return real_read_benign_rows(dataset, rows)

    monkeypatch.setattr(build, "read_benign_rows", recording_read_benign_rows)

    pins = _small_corpus_pins(offline_full_build)
    dataset = pins.attack_datasets[0]
    pins = dataclasses.replace(
        pins,
        attack_datasets=(dataclasses.replace(dataset, withdrawn=(withdrawal,)),),
    )

    _manifest, fields = build.build_corpus(pins, root=str(tmp_path))

    directory = tmp_path / build.DATA_DIRNAME
    written = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.jsonl"))
    )
    assert written, "the fixture wrote no corpus; this test has lost its subject"
    assert contradicted not in written

    (benign_pool,) = handed
    assert benign_pool == pool_with_the_contradiction(pins.attack_datasets[0])[0]
    assert all(row.text != contradicted for row in benign_pool)

    # And the count travels: a build that dropped the rows without saying so would satisfy the
    # assertions above and publish a corpus nobody could reconcile against the pinned dataset.
    assert fields["attack_draw"]["withdrawn_rows"] == len(withdrawal.rows)


def test_the_attack_half_refuses_to_overwrite_credits_that_are_already_there(
    offline_build: Path, tmp_path: Path
) -> None:
    """The rebuild rule covers the credits too, or a rebuild could leave stale ones behind.

    The failing input is a credits file with no corpus beside it, which is the state a half-deleted
    rebuild leaves: the corpus write would succeed and the credits would silently keep describing
    the previous draw.
    """
    directory = tmp_path / build.DATA_DIRNAME
    directory.mkdir(parents=True)
    (directory / build.ATTRIBUTION_FILENAME).write_text("stale\n", encoding="utf-8")

    with pytest.raises(build.CorpusWriteRefused):
        build.build_attack_corpus(_pins_with_a_small_draw(offline_build), root=str(tmp_path))

    assert (directory / build.ATTRIBUTION_FILENAME).read_text(encoding="utf-8") == "stale\n"
    assert not (directory / build.ATTACK_CORPUS_FILENAME).exists()


def test_the_full_build_generates_the_attribution_file_and_verify_accepts_it(
    offline_full_build: Path, tmp_path: Path
) -> None:
    """AD-34 end to end: the credits are written beside the corpus and they describe it.

    The row counts are asserted against the rows on disk rather than against the draw, which is the
    whole reason they are tallied from `CorpusItem.source`: a credits file that agreed with the
    declaration and disagreed with the corpus would be the drift this file exists to prevent.
    """
    from nbc.corpus.attribution import ATTRIBUTION_FILENAME, counts_by_key

    pins = _small_corpus_pins(offline_full_build)
    _manifest, fields = build.build_corpus(pins, root=str(tmp_path))

    path = tmp_path / build.DATA_DIRNAME / ATTRIBUTION_FILENAME
    assert fields["written"]["attribution"] == ATTRIBUTION_FILENAME
    credits = path.read_text(encoding="utf-8")

    rows = [
        json.loads(line)
        for name in (build.ATTACK_CORPUS_FILENAME, build.BENIGN_CORPUS_FILENAME)
        for line in (tmp_path / build.DATA_DIRNAME / name).read_text(encoding="utf-8").splitlines()
        if line
    ]
    from nbc.schema import CorpusItem

    items = tuple(
        CorpusItem(
            id=row["id"],
            source=row["source"],
            family=row["family"],
            benign_class=row["benign_class"],
            dressing=tuple(row["dressing"]),
            text=row["text"],
            label=row["label"],
        )
        for row in rows
    )
    counts, problems = counts_by_key(items, pins)
    assert problems == ()
    assert sum(counts.values()) == len(items)
    for repository in pins.benign_frame.b_code.repositories[:2]:
        assert f"| {counts[repository.key]} |" in credits
        assert repository.licence.attribution in credits

    # And what `verify-corpus` would regenerate from those rows is byte-identical to what was
    # written. The CLI path itself is exercised in `tests/corpus/test_attribution.py`, over the
    # committed declaration -- this fixture shrinks two sizes, so its build_id is deliberately not
    # the committed one and the guarded read would refuse it for that reason instead.
    assert build.attribution_text(pins, items, _manifest.build_id) == credits


def test_a_synthetic_b_code_file_carrying_a_pool_payload_stops_the_full_build(
    offline_build: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FR3.2 from the entrypoint down, and nothing is written when it fires.

    The poisoned file carries every positive the fake pool holds, so which four the draw selects
    cannot decide whether the gate fires.
    """
    carried = " ".join(f"payload {index}" for index in range(6))
    monkeypatch.setattr(
        build, "read_benign_code", lambda pins: _synthetic_code(pins, carried)
    )

    corpus_pins = _small_corpus_pins(offline_build)
    with pytest.raises(crosscheck.BenignItemMislabelled) as raised:
        build.build_corpus(corpus_pins, root=str(tmp_path))

    assert exit_code_for(raised.value) == 24
    # The poisoned file is the first one of the first repository the frame pins, and the abort
    # names it by the source that file carries.
    offender = corpus_pins.benign_frame.b_code.repositories[0]
    assert any(offender.repository in problem for problem in raised.value.problems)
    assert list(tmp_path.rglob("*.jsonl")) == []
