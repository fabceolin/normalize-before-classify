"""AD-34 / FR5.2: the licence gate, the row attribution and the generated credits file.

Every gate here is exercised through the input that makes it fail, built by `replace()`-ing the
real pins so the failing input is the real declaration with one field moved. A check whose two
sides come from the same place is not a check (P3), and the committed `pins.toml` is the one side
that is not under this file's control.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import nbc
from nbc.corpus import build as build_module
from nbc.corpus.attribution import (
    ATTRIBUTION_FILENAME,
    COMPATIBLE,
    KIND_ATTACK_DATASET,
    KIND_BENIGN_CODE,
    KIND_HAND_AUTHORED,
    LOCAL_LICENCE,
    REFUSED,
    RedistributionRefused,
    attribution_problems,
    counts_by_key,
    licence_problems,
    pinned_sources,
    render,
)
from nbc.corpus.benign import HAND_AUTHORED_SOURCE
from nbc.corpus.manifest import (
    ATTACK_CORPUS_FILENAME,
    BENIGN_CORPUS_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    build_id,
    corpus_directory,
    files_for,
    render as render_manifest,
)
from nbc.errors import declared_exit_codes, exit_code_for
from nbc.pins import NOT_DECLARED, load_pins
from nbc.schema import ATTACK, BENIGN, FAMILY_ATTACK, FAMILY_BENIGN, CorpusItem

REPO_ROOT = Path(nbc.__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pins():
    return load_pins(REPO_ROOT)


# --- the vocabulary -------------------------------------------------------------------------------


def test_the_two_licence_sets_are_disjoint() -> None:
    """An identifier in both would make the verdict depend on which branch ran first."""
    assert not COMPATIBLE & set(REFUSED)


def test_every_identifier_is_the_case_normalized_spelling() -> None:
    """Both sets are consulted with `casefold()`, so an upper-case member could never match."""
    for identifier in COMPATIBLE | set(REFUSED):
        assert identifier == identifier.casefold(), identifier


def test_every_refused_identifier_carries_a_reason() -> None:
    assert all(reason.strip() for reason in REFUSED.values())


def test_the_local_licence_is_the_one_the_repository_actually_offers() -> None:
    """P1: `LOCAL_LICENCE` claims MIT for the hand-authored rows. The LICENSE file is the evidence.

    Read rather than asserted, so relicensing this repository fails here instead of publishing
    twenty rows under a licence the file no longer grants.
    """
    licence_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence_text.splitlines()[0].strip() == "MIT License"
    assert LOCAL_LICENCE.identifier.casefold() == "mit"
    assert LOCAL_LICENCE.identifier.casefold() in COMPATIBLE


# --- the enumeration ------------------------------------------------------------------------------


def test_every_pinned_source_is_enumerated_exactly_once(pins) -> None:
    """The gate reads what `pins.toml` declares, and a kind it forgot is a kind it never checks."""
    records = pinned_sources(pins)
    expected = (
        len(pins.baselines)
        + len(pins.attack_datasets)
        + len(pins.exclusion_sources)
        + len(pins.benign_frame.b_code.repositories)
        + 1  # the hand-authored B-chat items
    )
    assert len(records) == expected
    assert len({record.key for record in records}) == expected


def test_every_pinned_source_declares_a_licence_and_an_attribution(pins) -> None:
    """FR5.2's first clause, over the committed file rather than over a fixture."""
    for record in pinned_sources(pins):
        assert record.licence.identifier
        assert record.licence.source
        assert record.licence.attribution
        assert record.repository in record.licence.attribution


# --- the gate -------------------------------------------------------------------------------------


def _dataset_licence(pins, **fields):
    """The committed pins with the attack pool's licence fields moved.

    `accepted` is cleared unless a caller asks for it. Every caller here that declares an
    identifier is describing a source whose question the *publisher* answered, and an acceptance
    answers the absence of an answer -- the loader refuses the combination outright, and this
    helper builds records directly, so nothing else would stop a fixture from being incoherent.
    """
    fields.setdefault("accepted", None)
    dataset = pins.attack_datasets[0]
    return replace(
        pins,
        attack_datasets=(
            replace(dataset, licence=replace(dataset.licence, **fields)),
        ),
    )


def test_the_committed_pins_pass_only_because_a_human_accepted_the_exposure(pins) -> None:
    """The committed state, and the one edit that takes it back to an abort.

    `xTRam1/safe-guard-prompt-injection` still declares **no licence** at the pinned revision and
    its rows are still redistributed into `data/*.jsonl`. What changed on 2026-08-30 is that a
    named person recorded a decision to publish anyway, with the reasoning in `README.md`. So the
    two halves are asserted separately: the fact is unchanged, and the build proceeds only because
    of the decision.

    Removing the acceptance is the input that turns this red, and it is the whole check: a gate
    that passed because the *fact* had been edited away would be a different gate.
    """
    dataset = pins.attack_datasets[0]
    # The fact, unchanged, read from the pin rather than typed here: `pins.toml` is the only home
    # for a repository id, and `tests/test_pins.py` refuses a copy under `tests/`.
    assert dataset.licence.identifier == NOT_DECLARED
    assert dataset.licence.redistributed
    assert dataset.licence.blocks_redistribution

    # The decision, and where it points.
    accepted = dataset.licence.accepted
    assert accepted is not None
    assert accepted.by and accepted.on and accepted.reasoning
    assert not dataset.licence.refuses_publication
    assert licence_problems(pins) == ()

    # Take the decision away and the same file aborts again, naming the same source.
    withdrawn = _dataset_licence(pins, accepted=None)
    (problem,) = licence_problems(withdrawn)
    assert dataset.repository in problem
    assert dataset.revision in problem
    assert "declares no licence" in problem


def test_the_acceptance_reasoning_is_published_where_a_reader_meets_it(pins) -> None:
    """It travels into `results.json` and into the generated credits, not just into `pins.toml`.

    A decision recorded only in the file that permits it is a decision nobody downstream sees.
    """
    dataset = pins.attack_datasets[0]
    accepted = dataset.licence.accepted
    assert accepted is not None

    fields = dataset.licence.as_run_fields()
    assert fields["identifier"] == NOT_DECLARED, "the run fields must not launder the gap"
    assert fields["accepted"] == accepted.as_run_fields()

    counts, _unaccounted = counts_by_key((), pins)
    credits = render(pins, counts, build_id="fixture")
    assert "Published without a licence, by decision" in credits
    assert accepted.by in credits
    assert accepted.on in credits
    assert accepted.position in credits
    # Rendered on one line: a raw newline from the TOML multi-line string would end the list item
    # and drop the rest of the reasoning without failing anything.
    assert " ".join(accepted.reasoning.split()) in credits


def test_the_position_names_a_heading_the_readme_actually_has(pins) -> None:
    """P1: the position is the evidence for the decision, so it is resolved against the prose.

    A free-text "see the README" cannot be resolved and would be evidence recorded beside a value
    that nothing compares it to. This reads `README.md`, derives the anchors its headings produce,
    and refuses a position naming one that is not there.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    anchors = {
        "README.md#"
        + re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in readme.splitlines()
        if line.startswith("#")
    }
    assert "README.md#redistribution-of-undeclared-material" in anchors, (
        "the fixture this test rests on is gone"
    )
    for record in pinned_sources(pins):
        accepted = record.licence.accepted
        if accepted is not None:
            assert accepted.position in anchors, record.key


def test_an_acceptance_does_not_excuse_a_licence_that_refuses_redistribution(pins) -> None:
    """The acceptance answers the ABSENCE of a licence and nothing else.

    A signature on a GPL source would be a human overruling terms somebody actually wrote, which
    is a different act from publishing where nobody wrote any. The gate refuses it; the loader
    refuses the same combination one layer earlier.
    """
    dataset = pins.attack_datasets[0]
    copyleft = _dataset_licence(
        pins,
        identifier="gpl-3.0",
        attribution=f"{dataset.repository}, GPL-3.0, revision {dataset.revision}",
        accepted=dataset.licence.accepted,
    )
    (problem,) = licence_problems(copyleft)
    assert REFUSED["gpl-3.0"] in problem


def test_an_acceptance_does_not_excuse_an_empty_identifier(pins) -> None:
    """`not-declared` is a reading of the card; `""` is a field nobody filled in.

    There is nothing to accept about a reading that was never made, and the empty limb is checked
    before the acceptance so a signature can never stand in for one.
    """
    dataset = pins.attack_datasets[0]
    blank = _dataset_licence(pins, identifier="", accepted=dataset.licence.accepted)
    (problem,) = licence_problems(blank)
    assert "empty licence identifier" in problem


def test_a_declared_compatible_licence_passes(pins) -> None:
    """The other side of the same check: the gate is not simply always-refuse."""
    licensed = _dataset_licence(
        pins,
        identifier="MIT",
        attribution=(
            f"{pins.attack_datasets[0].repository}, MIT, "
            f"revision {pins.attack_datasets[0].revision}"
        ),
        unresolved="",
    )
    assert licence_problems(licensed) == ()


def test_an_unrecognized_licence_is_refused(pins) -> None:
    licensed = _dataset_licence(
        pins,
        identifier="Weird-1.0",
        attribution=(
            f"{pins.attack_datasets[0].repository}, Weird-1.0, "
            f"revision {pins.attack_datasets[0].revision}"
        ),
        unresolved="",
    )
    (problem,) = licence_problems(licensed)
    assert "not in this project's compatible set" in problem
    assert "Weird-1.0" in problem


@pytest.mark.parametrize("identifier", sorted(REFUSED))
def test_a_refused_licence_is_refused_with_its_reason(pins, identifier: str) -> None:
    licensed = _dataset_licence(
        pins,
        identifier=identifier,
        attribution=(
            f"{pins.attack_datasets[0].repository}, {identifier}, "
            f"revision {pins.attack_datasets[0].revision}"
        ),
        unresolved="",
    )
    (problem,) = licence_problems(licensed)
    assert REFUSED[identifier] in problem


def test_an_attribution_that_does_not_name_the_repository_is_refused(pins) -> None:
    licensed = _dataset_licence(
        pins,
        identifier="MIT",
        attribution="some dataset, MIT",
        unresolved="",
    )
    problems = licence_problems(licensed)
    assert any("does not name the repository" in problem for problem in problems)


def test_an_attribution_that_does_not_name_the_revision_is_refused(pins) -> None:
    """P1: the attribution is the evidence for the credit, so it is compared against the pin."""
    repository = pins.benign_frame.b_code.repositories[0]
    moved = replace(
        repository,
        licence=replace(
            repository.licence,
            attribution=f"{repository.repository}, MIT, revision {'0' * 40}",
        ),
    )
    b_code = replace(
        pins.benign_frame.b_code,
        repositories=(moved,) + pins.benign_frame.b_code.repositories[1:],
    )
    licensed = replace(pins, benign_frame=replace(pins.benign_frame, b_code=b_code))
    problems = licence_problems(licensed)
    assert any("does not name the pinned revision" in problem for problem in problems)


def test_a_source_nobody_redistributes_may_declare_nothing(pins) -> None:
    """The second baseline declares no licence and ships no byte into `data/`. Not an abort.

    This is the recorded reading, not a loophole invented here: `[baseline.licence]` in
    `pins.toml` has said since Epic 1 that the abort is about rows that ARE redistributed.
    """
    undeclared = [
        record
        for record in pinned_sources(pins)
        if record.licence.identifier == NOT_DECLARED and not record.licence.redistributed
    ]
    assert undeclared, "the fixture this test rests on is gone"
    for record in undeclared:
        assert not any(record.key in problem for problem in licence_problems(pins))


def test_the_gate_agrees_with_the_pin_layers_own_property(pins) -> None:
    """P1 / decision D-C: `Licence.refuses_publication` is consumed, not merely published.

    The gate is compared against `refuses_publication` -- the fact **and** the absence of a
    decision about it -- rather than against `blocks_redistribution`, which is the fact alone.
    Comparing it against the fact would now be wrong in a way that matters: it would demand an
    abort for a source somebody accepted, and the two properties exist separately precisely so
    that `results.json` can keep saying the licence was never established.

    Asserted over the committed file and over two mutated copies -- the decision removed, and a
    real licence declared -- so the property and the gate cannot part company without this
    failing.
    """
    for record in pinned_sources(pins):
        refused = record.licence.refuses_publication
        named = any(record.repository in problem for problem in licence_problems(pins))
        assert refused == named, record.key

    without_the_decision = _dataset_licence(pins, accepted=None)
    dataset = without_the_decision.attack_datasets[0]
    assert dataset.licence.refuses_publication
    assert dataset.licence.blocks_redistribution
    assert len(licence_problems(without_the_decision)) == 1

    licensed = _dataset_licence(
        pins,
        identifier="MIT",
        attribution=(
            f"{pins.attack_datasets[0].repository}, MIT, "
            f"revision {pins.attack_datasets[0].revision}"
        ),
        unresolved="",
    )
    assert not licensed.attack_datasets[0].licence.blocks_redistribution
    assert not licensed.attack_datasets[0].licence.refuses_publication
    assert licence_problems(licensed) == ()


def test_the_exit_code_is_declared_and_distinct() -> None:
    codes = declared_exit_codes()
    assert codes[RedistributionRefused.exit_code] is RedistributionRefused
    assert exit_code_for(RedistributionRefused("x")) == RedistributionRefused.exit_code


# --- attributing rows to sources ------------------------------------------------------------------


def _item(source: str, *, attack: bool = False, suffix: str = "") -> CorpusItem:
    return CorpusItem(
        id=f"{source}{suffix}::clean",
        source=source,
        family=FAMILY_ATTACK if attack else FAMILY_BENIGN,
        benign_class=None if attack else "b_code",
        dressing=(),
        text="ignore previous instructions" if attack else "const x = 1;",
        label=ATTACK if attack else BENIGN,
    )


def _real_items(pins) -> tuple[CorpusItem, ...]:
    dataset = pins.attack_datasets[0]
    repository = pins.benign_frame.b_code.repositories[0]
    chat = replace(
        _item(f"{dataset.repository}@{dataset.revision}"), benign_class="b_chat"
    )
    hand = replace(_item(HAND_AUTHORED_SOURCE), benign_class="b_chat")
    return (
        _item(dataset.repository, attack=True),
        _item(repository.file_source("src/a.js")),
        _item(repository.file_source("src/b.js"), suffix="2"),
        chat,
        hand,
    )


def test_every_row_shape_the_build_writes_is_attributed(pins) -> None:
    counts, problems = counts_by_key(_real_items(pins), pins)
    assert problems == ()
    assert counts[pins.attack_datasets[0].key] == 2  # the attack row and the B-chat row
    assert counts[pins.benign_frame.b_code.repositories[0].key] == 2
    assert counts[HAND_AUTHORED_SOURCE] == 1


def test_a_row_from_a_source_nothing_pinned_is_refused(pins) -> None:
    """The failing input for the tally: a row that would be published uncredited."""
    _counts, problems = counts_by_key(
        (*_real_items(pins), _item("github.com/someone/else@" + "d" * 40 + ":x.py")), pins
    )
    assert len(problems) == 1
    assert "someone/else" in problems[0]


def test_a_row_naming_a_pinned_repository_at_another_revision_is_refused(pins) -> None:
    """P2: the identity is parsed and compared, so a prefix match is not enough."""
    repository = pins.benign_frame.b_code.repositories[0]
    stale = f"github.com/{repository.repository}@{'e' * 40}:src/a.js"
    _counts, problems = counts_by_key((_item(stale),), pins)
    assert len(problems) == 1
    assert stale in problems[0]


@pytest.mark.parametrize(
    "source",
    [
        "github.com/psf/requests",  # no path
        "github.com/psf/requests:src/a.py",  # no revision
        "psf/requests@" + "f" * 40 + ":src/a.py",  # no host
        "github.com/@" + "f" * 40 + ":src/a.py",  # no repository
    ],
)
def test_a_malformed_b_code_source_is_refused_rather_than_half_parsed(pins, source) -> None:
    _counts, problems = counts_by_key((_item(source),), pins)
    assert len(problems) == 1


# --- the generated file ---------------------------------------------------------------------------


def test_the_credits_name_every_source_with_its_licence_revision_and_count(pins) -> None:
    counts, _problems = counts_by_key(_real_items(pins), pins)
    text = render(pins, counts, build_id="b" * 64)

    for record in pinned_sources(pins):
        assert record.repository in text, record.repository
        if record.revision:
            assert record.revision in text, record.key
        assert record.licence.identifier in text
        if record.licence.redistributed:
            assert record.licence.attribution in text

    # The counts are rendered, not merely computed: two rows from the dataset (one attack row and
    # one B-chat row) and two from the first B-code repository.
    assert "| 2 |" in text
    assert "| 1 |" in text
    assert "b" * 64 in text


def test_the_credits_separate_what_is_redistributed_from_what_is_only_consulted(pins) -> None:
    counts, _problems = counts_by_key(_real_items(pins), pins)
    text = render(pins, counts, build_id="b" * 64)
    redistributed, _marker, consulted = text.partition("## Consulted, not redistributed")
    assert _marker

    for record in pinned_sources(pins):
        target = redistributed if record.licence.redistributed else consulted
        other = consulted if record.licence.redistributed else redistributed
        assert f"`{record.repository}`" in target, record.key
        assert f"`{record.repository}`" not in other, record.key


def test_the_rendered_credits_are_a_function_of_the_declaration_alone(pins) -> None:
    """Byte-identical across renders, which is what makes the regeneration check a check."""
    counts, _problems = counts_by_key(_real_items(pins), pins)
    assert render(pins, counts, build_id="b" * 64) == render(
        pins, counts, build_id="b" * 64
    )


def test_a_drifted_credits_file_is_named_and_a_matching_one_is_not() -> None:
    expected = "# Attribution\n"
    assert attribution_problems(expected, expected) == ()
    (problem,) = attribution_problems("# Attribution, edited by hand\n", expected)
    assert ATTRIBUTION_FILENAME in problem
    (missing,) = attribution_problems(None, expected)
    assert "is not beside the corpus" in missing


# --- verify-corpus, end to end over a corpus on disk ----------------------------------------------


def _committed_corpus(root: Path, pins, *, attribution: str | None) -> None:
    """A corpus in `root` whose rows credit real pinned sources, with a chosen credits file."""
    from nbc.corpus.attack import serialize

    items = _real_items(pins)
    attack = tuple(item for item in items if item.family == FAMILY_ATTACK)
    benign = tuple(item for item in items if item.family == FAMILY_BENIGN)
    payloads = [
        (ATTACK_CORPUS_FILENAME, serialize(attack), len(attack)),
        (BENIGN_CORPUS_FILENAME, serialize(benign), len(benign)),
    ]
    record = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        frame_id=pins.benign_frame.frame_id,
        build_id=build_id(pins),
        files=files_for(
            [(name, text.encode("utf-8"), rows) for name, text, rows in payloads]
        ),
        reports={},
    )
    directory = corpus_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text, _rows in payloads:
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    (directory / MANIFEST_FILENAME).write_text(
        render_manifest(record), encoding="utf-8", newline="\n"
    )
    if attribution is not None:
        (directory / ATTRIBUTION_FILENAME).write_text(
            attribution, encoding="utf-8", newline="\n"
        )
    shutil.copy(REPO_ROOT / "pins.toml", root / "pins.toml")


def _generated(pins) -> str:
    counts, _problems = counts_by_key(_real_items(pins), pins)
    return render(pins, counts, build_id=build_id(pins))


def test_verify_corpus_accepts_the_credits_it_would_generate(tmp_path: Path, pins) -> None:
    _committed_corpus(tmp_path, pins, attribution=_generated(pins))
    assert build_module.main(["--root", str(tmp_path), "verify-corpus"]) == 0


def test_verify_corpus_refuses_credits_edited_by_hand(
    tmp_path: Path, pins, capsys
) -> None:
    _committed_corpus(
        tmp_path, pins, attribution=_generated(pins) + "\n- some other project\n"
    )
    assert (
        build_module.main(["--root", str(tmp_path), "verify-corpus"])
        == RedistributionRefused.exit_code
    )
    assert ATTRIBUTION_FILENAME in capsys.readouterr().err


def test_verify_corpus_refuses_credits_that_are_not_utf8_text(
    tmp_path: Path, pins, capsys
) -> None:
    """P5: `UnicodeDecodeError` is a `ValueError`, not an `OSError`, and it must not escape raw.

    A credits file re-encoded by an editor is a classified abort with a diagnosis, not a traceback
    a caller reads as an unexpected crash.
    """
    _committed_corpus(tmp_path, pins, attribution=_generated(pins))
    (corpus_directory(tmp_path) / ATTRIBUTION_FILENAME).write_bytes(b"# Attribution\n\xff\xfe")

    assert (
        build_module.main(["--root", str(tmp_path), "verify-corpus"])
        == RedistributionRefused.exit_code
    )
    assert "cannot be read as UTF-8" in capsys.readouterr().err


def test_verify_corpus_refuses_a_corpus_with_no_credits(
    tmp_path: Path, pins, capsys
) -> None:
    _committed_corpus(tmp_path, pins, attribution=None)
    assert (
        build_module.main(["--root", str(tmp_path), "verify-corpus"])
        == RedistributionRefused.exit_code
    )
    assert "is not beside the corpus" in capsys.readouterr().err


def _pins_with_the_question_reopened(destination: Path) -> Path:
    """The committed `pins.toml` with the acceptance cut out and the open question put back.

    Text surgery on the real file rather than a hand-written fixture, so the input that turns the
    gate red is the committed declaration with one block moved -- P3, a check whose two sides come
    from the same place is not a check. The surgery **asserts what it removed**, so a file whose
    shape changes fails here loudly instead of quietly cutting nothing and passing.

    The block is replaced rather than deleted, and that is not decoration: `_read_licence` refuses
    a redistributed source that declares neither an identifier, an open question nor a decision,
    so a plain deletion would abort at load with exit 4 and this test would stop exercising the
    build-time gate it is named after.
    """
    lines = (REPO_ROOT / "pins.toml").read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("[attack_dataset.licence.accepted]")
    ]
    assert len(starts) == 1, "the acceptance block this test rewrites is not where it was"
    begin = starts[0]
    end = next(index for index in range(begin + 1, len(lines)) if lines[index].startswith("["))
    removed = "".join(lines[begin:end])
    assert "position =" in removed and "reasoning =" in removed, removed

    path = destination / "pins.toml"
    path.write_text(
        "".join(lines[:begin])
        + 'unresolved = "fixture: the decision withdrawn, the question open again"\n\n'
        + "".join(lines[end:]),
        encoding="utf-8",
    )
    return path


def test_the_gate_still_aborts_the_build_before_the_network_when_nobody_has_decided(
    tmp_path: Path, capsys
) -> None:
    """The story's outcome, through the CLI, under the offline guard.

    The suite installs a guard that raises on any socket, so a build that reached the pool before
    the licence gate would fail here with a network error rather than with exit 25. The exit code
    is therefore evidence for *both* halves of the claim: the gate fires, and it fires first.

    What changed on 2026-08-30 is the committed file, not the gate. The acceptance is cut out
    here, and everything this test asserted before still holds.
    """
    _pins_with_the_question_reopened(tmp_path)

    assert (
        build_module.main(["--root", str(tmp_path), "build-corpus"])
        == RedistributionRefused.exit_code
    )

    stderr = capsys.readouterr().err
    committed = load_pins(REPO_ROOT).attack_datasets[0]
    assert committed.repository in stderr
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not (tmp_path / "data" / ATTRIBUTION_FILENAME).exists()


def test_build_attack_is_refused_by_the_same_gate(tmp_path: Path, capsys) -> None:
    """`build-attack` writes redistributed rows too, so it is gated identically."""
    _pins_with_the_question_reopened(tmp_path)

    assert (
        build_module.main(["--root", str(tmp_path), "build-attack"])
        == RedistributionRefused.exit_code
    )
    assert load_pins(REPO_ROOT).attack_datasets[0].repository in capsys.readouterr().err


def test_the_committed_declaration_now_reaches_the_pool_instead_of_aborting(
    tmp_path: Path,
) -> None:
    """The other side of the two tests above, and the reason they had to be rewritten.

    With the acceptance in place the licence gate no longer stops anything, so under the offline
    guard the build gets as far as the first socket and dies there. Asserting that it is **not**
    `RedistributionRefused` is what keeps the pair honest: without this, both tests above would
    keep passing over a tampered file while nobody noticed that the committed one no longer does
    what the story says it does.
    """
    shutil.copy(REPO_ROOT / "pins.toml", tmp_path / "pins.toml")

    with pytest.raises(BaseException) as caught:
        build_module.build_attack_corpus(load_pins(tmp_path), root=str(tmp_path))
    assert not isinstance(caught.value, RedistributionRefused)
    assert not list(tmp_path.rglob("*.jsonl"))


# --- the three sides that have to agree ---------------------------------------------------------
#
# P6, the failure the Epic 1 remediation itself committed: a decision written into a document,
# cited later in a CI comment as fact, and never implemented. The README describes this gate and
# CI asserts an exit code for it, so both are compared against the code here.


def test_the_readme_describes_the_file_this_build_actually_writes(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert f"data/{ATTRIBUTION_FILENAME}" in readme
    assert "python -m nbc.corpus.build verify-corpus" in readme


def test_the_ci_step_asserts_what_the_committed_declarations_actually_do(
    repo_root: Path,
) -> None:
    """P6: the CI step must expect what the code does, and must name the codes the code declares.

    Split on the workflow's own step boundaries rather than grepped for a number: the file has
    several `-ne` comparisons and matching the wrong one would make this test agree with itself.

    Two claims, because the step makes two. It asserts a success code, and it explains in prose
    which aborts it would have caught -- and that prose is the thing P6 is about, a decision
    recorded in a comment and cited later as fact. So the numbers it names are compared against
    the exit codes the classes declare, not against literals typed here.
    """
    from nbc.corpus.attack import LabelContradiction, WithdrawalDoesNotMatchPool
    from nbc.errors import EXIT_OK

    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    steps = workflow.split("- name:")
    matching = [
        step for step in steps if "describe the real pool" in step.splitlines()[0].lower()
    ]
    assert len(matching) == 1, [step.splitlines()[0] for step in steps]
    step = matching[0]

    expected = re.findall(r'\$status" -ne (\d+)', step)
    assert expected == [str(EXIT_OK)]
    assert "attack-pool-report" in step

    for abort in (RedistributionRefused, WithdrawalDoesNotMatchPool, LabelContradiction):
        assert f"{abort.exit_code} is" in step, abort.__name__
