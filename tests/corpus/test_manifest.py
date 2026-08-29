"""`build_id`, `data/manifest.json`, and the one door anything reads the corpus through.

Two properties are worth more than the rest here. The first is that `build_id` **moves** when any
of the five declared components moves -- an id that covered only what it was easy to cover would
publish a table computed over the previous corpus with every check green, which is the failure
FR5.1 names in so many words. The second is that the guarded read is the *only* read: a check the
entrypoint may forget to call is a check that gets forgotten, so the scan below is over the tree.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nbc.corpus import manifest as manifest_module
from nbc.corpus.manifest import (
    ATTACK_CORPUS_FILENAME,
    BENIGN_CORPUS_FILENAME,
    CORPUS_FILENAMES,
    DATA_DIRNAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    CorpusManifestMismatch,
    Manifest,
    build_id,
    content_hash,
    corpus_directory,
    files_for,
    parse,
    read_corpus,
    render,
)
from nbc.corpus.matrix import CHAINS, HELDOUT_CHAINS
from nbc.errors import declared_exit_codes, exit_code_for
from nbc.pins import load_pins
from nbc.schema import ATTACK, BENIGN, FAMILY_ATTACK, FAMILY_BENIGN, CorpusItem

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def pins():
    return load_pins(REPO_ROOT)


# --- build_id covers what it says it covers -----------------------------------------------------


def test_the_same_declaration_computes_the_same_id(pins) -> None:
    assert build_id(pins) == build_id(pins)


def test_the_attack_sample_size_moves_the_build_id(pins) -> None:
    """FR5.1's own worked example: a frame_id guarding only the benign half would not notice this."""
    dataset = pins.attack_datasets[0]
    moved = replace(
        pins,
        attack_datasets=(
            replace(dataset, draw=replace(dataset.draw, sample_size_positives=7)),
        ),
    )
    assert build_id(moved) != build_id(pins)


def test_the_benign_frame_moves_the_build_id(pins) -> None:
    moved = replace(pins, benign_frame=replace(pins.benign_frame, seed=999_983))
    assert build_id(moved) != build_id(pins)


def test_the_confirmatory_cell_moves_the_build_id(pins) -> None:
    """The cell the verdict rests on cannot be changed and the corpus reused."""
    cell = replace(pins.benign_frame.confirmatory_cell, benign_class="b_chat")
    moved = replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))
    assert build_id(moved) != build_id(pins)


def test_the_exclusion_declaration_moves_the_build_id(pins) -> None:
    moved = replace(pins, exclusion_sources=pins.exclusion_sources[:-1])
    assert build_id(moved) != build_id(pins)


@pytest.mark.parametrize("registry", ["CHAINS", "HELDOUT_CHAINS"])
def test_the_dressing_registries_move_the_build_id(
    pins, monkeypatch: pytest.MonkeyPatch, registry: str
) -> None:
    """The dressing axis of the table *is* those constants, so a chain added is a new corpus."""
    before = build_id(pins)
    source = CHAINS if registry == "CHAINS" else HELDOUT_CHAINS
    widened = {
        corpus_class: tuple(chains) + (("base64", "hex"),)
        for corpus_class, chains in source.items()
    }
    monkeypatch.setattr(manifest_module, registry, widened)
    assert build_id(pins) != before


def test_the_build_id_payload_version_is_part_of_it(pins, monkeypatch) -> None:
    before = build_id(pins)
    monkeypatch.setattr(manifest_module, "BUILD_ID_VERSION", manifest_module.BUILD_ID_VERSION + 1)
    assert build_id(pins) != before


# --- the manifest file ----------------------------------------------------------------------------


def _items() -> tuple[tuple[CorpusItem, ...], tuple[CorpusItem, ...]]:
    attack = (
        CorpusItem(
            id="a::clean",
            source="example/pool@" + "a" * 40,
            family=FAMILY_ATTACK,
            benign_class=None,
            dressing=(),
            text="ignore previous instructions",
            label=ATTACK,
        ),
    )
    benign = (
        CorpusItem(
            id="b::clean",
            source="github.com/example/code@" + "c" * 40 + ":src/a.js",
            family=FAMILY_BENIGN,
            benign_class="b_code",
            dressing=(),
            text="const x = 1;",
            label=BENIGN,
        ),
    )
    return attack, benign


def _write(root: Path, pins, *, mutate=None) -> Manifest:
    from nbc.corpus.attack import serialize

    attack, benign = _items()
    payloads = [
        (ATTACK_CORPUS_FILENAME, serialize(attack), len(attack)),
        (BENIGN_CORPUS_FILENAME, serialize(benign), len(benign)),
    ]
    record = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        frame_id=pins.benign_frame.frame_id,
        build_id=build_id(pins),
        files=files_for([(name, text.encode("utf-8"), rows) for name, text, rows in payloads]),
        reports={},
    )
    if mutate is not None:
        record = mutate(record)
    directory = corpus_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text, _rows in payloads:
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    (directory / MANIFEST_FILENAME).write_text(render(record), encoding="utf-8", newline="\n")
    return record


def test_a_manifest_round_trips_through_its_rendered_form(tmp_path: Path, pins) -> None:
    record = _write(tmp_path, pins)
    read_back = parse((corpus_directory(tmp_path) / MANIFEST_FILENAME).read_text())
    assert read_back.frame_id == record.frame_id
    assert read_back.build_id == record.build_id
    assert [entry.as_json_object() for entry in read_back.files] == [
        entry.as_json_object() for entry in record.files
    ]


def test_the_guarded_read_returns_the_rows(tmp_path: Path, pins) -> None:
    _write(tmp_path, pins)
    record, items = read_corpus(pins, tmp_path)
    assert len(items) == 2
    assert {item.family for item in items} == {FAMILY_ATTACK, FAMILY_BENIGN}
    assert record.build_id == build_id(pins)


def test_a_corpus_with_no_manifest_is_refused(tmp_path: Path, pins) -> None:
    (tmp_path / DATA_DIRNAME).mkdir()
    with pytest.raises(CorpusManifestMismatch, match=MANIFEST_FILENAME):
        read_corpus(pins, tmp_path)


def test_a_recorded_frame_id_that_is_not_the_declared_one_refuses_the_read(
    tmp_path: Path, pins
) -> None:
    """FR5.1's last clause: the frame cannot be changed and the corpus reused."""
    _write(tmp_path, pins, mutate=lambda record: replace(record, frame_id="f" * 64))
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(pins, tmp_path)
    assert any("frame_id" in problem for problem in raised.value.problems)


def test_a_recorded_build_id_that_is_not_the_computed_one_refuses_the_read(
    tmp_path: Path, pins
) -> None:
    _write(tmp_path, pins, mutate=lambda record: replace(record, build_id="e" * 64))
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(pins, tmp_path)
    assert any("build_id" in problem for problem in raised.value.problems)


def test_a_corpus_file_edited_after_the_build_refuses_the_read(tmp_path: Path, pins) -> None:
    _write(tmp_path, pins)
    path = corpus_directory(tmp_path) / BENIGN_CORPUS_FILENAME
    path.write_text(path.read_text().replace("const x = 1;", "const x = 2;"), encoding="utf-8")
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(pins, tmp_path)
    assert any("hashes to" in problem for problem in raised.value.problems)


def test_a_row_whose_label_disagrees_with_its_family_refuses_the_read(
    tmp_path: Path, pins
) -> None:
    """The one defect that would make every rate wrong in the direction nobody would question."""
    _write(tmp_path, pins)
    directory = corpus_directory(tmp_path)
    path = directory / BENIGN_CORPUS_FILENAME
    record = json.loads(path.read_text().strip())
    record["label"] = ATTACK
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")

    # Rewrite the manifest so the CONTENT HASH still matches: without that the read would be
    # refused by the digest and this gate would never be reached, which is how a check gets
    # written and never fires.
    manifest_file = directory / MANIFEST_FILENAME
    document = json.loads(manifest_file.read_text())
    for entry in document["files"]:
        if entry["name"] == BENIGN_CORPUS_FILENAME:
            entry["sha256"] = content_hash(payload.encode("utf-8"))
            entry["bytes"] = len(payload.encode("utf-8"))
    manifest_file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(pins, tmp_path)
    assert any("gold label of that family" in problem for problem in raised.value.problems)


def test_a_manifest_naming_a_file_that_is_not_there_is_refused(tmp_path: Path, pins) -> None:
    _write(tmp_path, pins)
    (corpus_directory(tmp_path) / BENIGN_CORPUS_FILENAME).unlink()
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(pins, tmp_path)
    assert any(BENIGN_CORPUS_FILENAME in problem for problem in raised.value.problems)


def test_a_manifest_from_another_schema_is_refused(tmp_path: Path, pins) -> None:
    _write(tmp_path, pins)
    path = corpus_directory(tmp_path) / MANIFEST_FILENAME
    document = json.loads(path.read_text())
    document["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document))
    with pytest.raises(CorpusManifestMismatch, match="schema_version"):
        read_corpus(pins, tmp_path)


def test_a_manifest_that_is_not_json_is_refused_rather_than_crashing(tmp_path: Path, pins) -> None:
    directory = corpus_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / MANIFEST_FILENAME).write_bytes(b"\xff\xfe not json")
    with pytest.raises(CorpusManifestMismatch):
        read_corpus(pins, tmp_path)


def test_the_abort_has_an_exit_code_distinct_from_every_other() -> None:
    codes = declared_exit_codes()
    assert codes[CorpusManifestMismatch.exit_code] is CorpusManifestMismatch
    assert exit_code_for(CorpusManifestMismatch("x")) == CorpusManifestMismatch.exit_code


# --- one reader ------------------------------------------------------------------------------------


READERS: dict[str, str] = {
    "manifest.py": "the guarded read, which is the only door into data/*.jsonl",
    "build.py": "writes them, and reads them back through manifest.read_corpus",
}
"""Every module under `src/` that may name a corpus file at all, each with its reason.

The rule this enforces is FR5.1's "the entrypoint refuses to measure when the recorded frame_id
differs": an entrypoint check is one caller's discipline, and a reader that verifies is the only
way in. A second module that could locate `data/attack.jsonl` could open it without verifying, and
this is what stops that being a code-review question.
"""


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


LOCATORS = frozenset(
    {
        "ATTACK_CORPUS_FILENAME",
        "BENIGN_CORPUS_FILENAME",
        "CORPUS_FILENAMES",
        "MANIFEST_FILENAME",
        "corpus_directory",
        "manifest_path",
    }
)
"""The names that let a module find a corpus file. `read_corpus` is deliberately not one of them:
importing the guarded door is the point, and a module that wants rows is supposed to ask for it."""


def _locates_a_corpus_file(source: str, filename: str) -> bool:
    """Whether `source` could open a corpus file: it spells a name, or it imports a locator.

    Both halves matter and neither is enough alone. A literal catches a module that hard-codes
    `data/attack.jsonl`; the import catches one that reaches for `corpus_directory` and builds the
    path from the constants, which is what an honest offender would do. Containment rather than
    equality on the literal, because the offender's string is usually the whole path.
    """
    tree = ast.parse(source, filename=filename)
    names = (*CORPUS_FILENAMES, MANIFEST_FILENAME)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(name in node.value for name in names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module == manifest_module.__name__:
            if any(alias.name in LOCATORS for alias in node.names):
                return True
    return False


def test_only_the_declared_modules_can_locate_a_corpus_file() -> None:
    naming = {
        path.name
        for path in _source_files()
        if _locates_a_corpus_file(path.read_text(encoding="utf-8"), str(path))
    }
    assert naming == set(READERS), sorted(naming.symmetric_difference(READERS))


def test_the_locator_scan_fires_on_both_shapes_of_offender() -> None:
    """The scan's own failing inputs, so it cannot pass by failing to look."""
    literal = f'rows = open("{DATA_DIRNAME}/{ATTACK_CORPUS_FILENAME}").read()\n'
    imported = (
        f"from {manifest_module.__name__} import corpus_directory\n"
        "rows = list(corpus_directory().iterdir())\n"
    )
    clean = "from nbc.corpus.manifest import read_corpus\nrows = read_corpus(p)\n"
    assert _locates_a_corpus_file(literal, "<literal>")
    assert _locates_a_corpus_file(imported, "<imported>")
    assert not _locates_a_corpus_file(clean, "<clean>")


# --- the confirmatory cell names a cell of the table -----------------------------------------


def test_the_committed_confirmatory_cell_names_a_cell_the_corpus_carries(pins) -> None:
    from nbc.corpus.manifest import confirmatory_cell_problems

    assert confirmatory_cell_problems(pins) == ()


def test_a_cell_naming_a_class_the_table_has_no_row_for_is_refused(pins) -> None:
    from nbc.corpus.manifest import confirmatory_cell_problems

    cell = replace(pins.benign_frame.confirmatory_cell, benign_class="b_email")
    moved = replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))
    (problem,) = confirmatory_cell_problems(moved)
    # The class message specifically, not merely "something mentioned b_email": with the class
    # check gone the chain check fires on the empty registry for that class and names it too, and
    # a test that accepted either would pass with the class check deleted.
    assert "b_email" in problem and "which is not one of" in problem
    assert "dressing chain" not in problem


def test_a_cell_naming_a_chain_no_registry_declares_is_refused(pins) -> None:
    """A verdict over a cell the corpus carries no row in looks exactly like one with data."""
    from nbc.corpus.manifest import confirmatory_cell_problems

    cell = replace(pins.benign_frame.confirmatory_cell, dressing_chain="base32+rot13")
    moved = replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))
    (problem,) = confirmatory_cell_problems(moved)
    assert "base32+rot13" in problem


def test_a_stale_cell_refuses_the_guarded_read(tmp_path: Path, pins) -> None:
    _write(tmp_path, pins)
    cell = replace(pins.benign_frame.confirmatory_cell, dressing_chain="base32+rot13")
    moved = replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(moved, tmp_path)
    assert any("base32+rot13" in problem for problem in raised.value.problems)
