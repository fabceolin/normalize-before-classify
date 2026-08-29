"""The benign draw, gate by gate, each with the input that makes it fail.

`corpus/benign.py` is pure, so the whole decision procedure -- which files are eligible, how many
each repository may contribute, what happens when a class cannot be filled -- is here rather than
behind a network call. The frames are built in code and their numbers are deliberately not
`pins.toml`'s, so a gate that stopped firing shows up.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nbc.corpus import benign
from nbc.corpus.benign import (
    BENIGN_CHAT,
    BENIGN_CODE,
    DECODE_STAGE_NAMES,
    HAND_AUTHORED_SOURCE,
    BenignDrawUnsatisfiable,
    SourceFile,
    chains_for,
    draw_benign_items,
    draw_chat_texts,
    eligible,
    offers_decode_candidate,
    render_benign_item,
    repository_seed,
    select_repository_files,
)
from nbc.corpus.matrix import CHAINS, HELDOUT_CHAINS, item_id, payload_id
from nbc.corpus.sources.encoded_messages import KIND_JWT, MESSAGES, EncodedMessage
from nbc.errors import declared_exit_codes, exit_code_for
from nbc.canon.stages import decode
from nbc.pins import AttackDataset, AttackDraw, Licence, Provenance
from nbc.schema import BENIGN, BENIGN_CLASSES, FAMILY_BENIGN

from tests.corpus.benign_fixtures import (
    ELIGIBLE_TEXT,
    PLAIN_TEXT,
    code_file,
    frame,
    repository,
    source_file,
    unique_eligible,
)


@pytest.fixture(scope="module")
def ctx():
    return benign.default_eligibility_context()


DATASET = AttackDataset(
    key="fixture",
    repository="example/pool",
    revision="a" * 40,
    splits=("train",),
    attack_label=1,
    draw=AttackDraw(
        declared_on="2026-08-29",
        sample_size_positives=2,
        method="seeded_random",
        seed=1,
        sort_key=None,
    ),
    licence=Licence(
        identifier="MIT", source="fixture", attribution="fixture", redistributed=True
    ),
    provenance=Provenance(checked_on="2026-08-29", card_revision="a" * 40, seeds=()),
)


def chat_pool(count: int) -> tuple[str, ...]:
    return tuple(f"a perfectly ordinary conversational message number {index}" for index in range(count))


# --- the vocabulary is read from its home -------------------------------------------------------


def test_the_benign_classes_are_schema_s_and_not_a_second_spelling() -> None:
    assert (BENIGN_CODE, BENIGN_CHAT) == BENIGN_CLASSES


def test_the_eligibility_rule_reads_the_stage_names_off_the_stage() -> None:
    """Spelled from `decode`'s own constants, so a renamed stage moves the rule with it."""
    assert DECODE_STAGE_NAMES == frozenset({decode.NAME, decode.CEILING_NAME})


# --- eligibility --------------------------------------------------------------------------------


def test_a_file_the_layer_would_examine_is_eligible_and_a_plain_one_is_not(ctx) -> None:
    """The two halves of the declared rule, on two files of the same size and extension."""
    assert offers_decode_candidate(ELIGIBLE_TEXT, ctx)
    assert not offers_decode_candidate(PLAIN_TEXT, ctx)
    assert eligible(SourceFile("a.js", ELIGIBLE_TEXT), frame(), ctx)
    assert not eligible(SourceFile("a.js", PLAIN_TEXT), frame(), ctx)


def test_the_extension_is_the_path_s_own_suffix_and_not_a_substring(ctx) -> None:
    """`assets.js.map` is not a source file, and a substring test would admit it."""
    assert eligible(SourceFile("a/b.js", ELIGIBLE_TEXT), frame(), ctx)
    assert not eligible(SourceFile("a/b.js.map", ELIGIBLE_TEXT), frame(), ctx)
    assert not eligible(SourceFile("Makefile", ELIGIBLE_TEXT), frame(), ctx)


def test_the_size_band_is_applied_at_both_ends(ctx) -> None:
    """Both ends, on the same eligible file, so neither bound can be the one that never fires."""
    size = len(ELIGIBLE_TEXT.encode("utf-8"))
    inside = frame(b_code=replace(frame().b_code, min_file_bytes=size, max_file_bytes=size))
    too_small = frame(b_code=replace(frame().b_code, min_file_bytes=size + 1))
    too_large = frame(b_code=replace(frame().b_code, max_file_bytes=size - 1))
    assert eligible(SourceFile("a.js", ELIGIBLE_TEXT), inside, ctx)
    assert not eligible(SourceFile("a.js", ELIGIBLE_TEXT), too_small, ctx)
    assert not eligible(SourceFile("a.js", ELIGIBLE_TEXT), too_large, ctx)


def test_a_context_with_tracing_off_is_refused_rather_than_judging_every_file_ineligible(
    ctx,
) -> None:
    """With no trace the predicate would answer False for everything and empty the draw in silence."""
    with pytest.raises(ValueError, match="tracing off"):
        offers_decode_candidate(ELIGIBLE_TEXT, replace(ctx, trace_enabled=False))


# --- one repository's contribution --------------------------------------------------------------


def test_a_repository_contributes_at_most_the_declared_cap(ctx) -> None:
    files = [source_file(f"file{index}") for index in range(10)]
    selected = select_repository_files(repository(0), files, frame(), ctx)
    assert len(selected) == frame().b_code.max_files_per_repository


def test_a_repository_s_selection_does_not_depend_on_the_order_it_was_read(ctx) -> None:
    files = [source_file(f"file{index}") for index in range(6)]
    forward = select_repository_files(repository(0), files, frame(), ctx)
    backward = select_repository_files(repository(0), list(reversed(files)), frame(), ctx)
    assert [entry.path for entry in forward] == [entry.path for entry in backward]


def test_two_repositories_draw_different_files_from_the_same_candidate_set(ctx) -> None:
    """The per-repository seed is derived from the frame's seed and the repository's own id."""
    assert repository_seed(frame(), repository(0)) != repository_seed(frame(), repository(1))
    assert repository_seed(frame(), repository(0)) == repository_seed(frame(), repository(0))


def test_a_repository_s_duplicate_files_become_one_candidate(ctx) -> None:
    one = source_file("same")
    other = SourceFile(path="vendor/same.js", text=one.text)
    selected = select_repository_files(repository(0), [one, other], frame(), ctx)
    assert len(selected) == 1


def test_ineligible_files_are_simply_absent(ctx) -> None:
    selected = select_repository_files(
        repository(0), [SourceFile("a.js", PLAIN_TEXT)], frame(), ctx
    )
    assert selected == ()


def test_an_eligibility_rule_nothing_implements_aborts(ctx) -> None:
    broken = frame(b_code=replace(frame().b_code, eligibility="whatever_looks_encoded"))
    with pytest.raises(BenignDrawUnsatisfiable, match="whatever_looks_encoded"):
        select_repository_files(repository(0), [source_file("a")], broken, ctx)


# --- the chat draw --------------------------------------------------------------------------------


def test_the_chat_draw_takes_the_class_minus_its_hand_authored_allowance() -> None:
    hand, drawn = draw_chat_texts(chat_pool(50), frame(), MESSAGES[:1])
    assert len(hand) == 1
    assert len(drawn) == frame().sample_size_items - 1


def test_a_short_surviving_pool_aborts_rather_than_topping_up() -> None:
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        draw_chat_texts(chat_pool(2), frame(), MESSAGES[:1])
    (problem,) = raised.value.problems
    assert "3 rows" in problem and "2 benign rows survive" in problem
    assert "topping up" in problem


def test_a_hand_authored_count_the_frame_does_not_allow_aborts() -> None:
    """The allowance is compared against what `corpus/sources/` holds, never trusted."""
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        draw_chat_texts(chat_pool(50), frame(), MESSAGES[:3])
    assert any("allows 1 hand-authored" in problem for problem in raised.value.problems)


def test_a_hand_authored_item_that_is_not_its_declared_kind_aborts() -> None:
    forged = EncodedMessage(key="x", kind=KIND_JWT, text="no token in here at all")
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        draw_chat_texts(chat_pool(50), frame(), [forged])
    assert any(KIND_JWT in problem for problem in raised.value.problems)


def test_a_hand_authored_text_that_the_dataset_also_carries_aborts() -> None:
    """One payload under two sources would be one item id carrying two `source` values."""
    pool = chat_pool(50) + (MESSAGES[0].text,)
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        draw_chat_texts(pool, frame(), MESSAGES[:1])
    assert any("two sources" in problem for problem in raised.value.problems)


def test_a_method_nothing_implements_aborts_with_this_half_s_diagnosis() -> None:
    with pytest.raises(BenignDrawUnsatisfiable, match="every_other_row"):
        draw_chat_texts(chat_pool(50), frame(method="every_other_row"), MESSAGES[:1])


# --- the whole draw -------------------------------------------------------------------------------


def _code(repositories: int, per_repository: int) -> dict[str, tuple]:
    return {
        f"example-code-{index}": tuple(
            code_file(f"example-code-{index}", f"file{number}")
            for number in range(per_repository)
        )
        for index in range(repositories)
    }


def _draw(code=None, chat=None, the_frame=None, messages=MESSAGES[:1]):
    return draw_benign_items(
        frame=the_frame or frame(),
        code_by_repository=code if code is not None else _code(3, 2),
        chat_surviving=chat if chat is not None else chat_pool(50),
        dataset=DATASET,
        chat_rows_in=50,
        chat_rows_removed=6,
        messages=messages,
    )


def test_both_classes_are_filled_exactly_and_every_chain_is_built() -> None:
    items, report = _draw()
    per_class = len(chains_for(BENIGN_CODE))
    assert per_class == len(CHAINS[BENIGN_CODE]) + len(HELDOUT_CHAINS[BENIGN_CODE])
    for benign_class in BENIGN_CLASSES:
        rows = [item for item in items if item.benign_class == benign_class]
        assert len(rows) == frame().sample_size_items * per_class
        assert len({item.text for item in rows}) == len(rows)
    assert report.items_written == len(items)
    assert report.sample_size_items == frame().sample_size_items


def test_every_row_is_benign_by_construction() -> None:
    items, _report = _draw()
    assert {item.family for item in items} == {FAMILY_BENIGN}
    assert {item.label for item in items} == {BENIGN}
    assert {item.benign_class for item in items} == set(BENIGN_CLASSES)


def test_one_payload_s_rows_share_a_stem_so_the_dressing_axis_can_be_paired() -> None:
    item = render_benign_item(
        ELIGIBLE_TEXT, source="x", benign_class=BENIGN_CODE, chain=("base64",)
    )
    assert item.id == item_id(payload_id(ELIGIBLE_TEXT), ("base64",))
    assert payload_id(item.text) != payload_id(ELIGIBLE_TEXT)


def test_hand_authored_rows_name_the_module_they_were_written_in() -> None:
    items, report = _draw()
    sources = {item.source for item in items if item.benign_class == BENIGN_CHAT}
    assert HAND_AUTHORED_SOURCE in sources
    assert f"{DATASET.repository}@{DATASET.revision}" in sources
    assert report.chat_hand_authored == 1


def test_the_realized_repository_count_and_the_per_repository_counts_are_recorded() -> None:
    items, report = _draw()
    assert report.code_repositories_realized == len(report.files_by_repository)
    assert sum(report.files_by_repository.values()) == frame().sample_size_items
    assert report.code_repositories_pinned == len(frame().b_code.repositories)
    assert max(report.files_by_repository.values()) <= frame().b_code.max_files_per_repository
    assert "b_code" in report.as_run_fields()["benign_draw"]


def test_too_few_eligible_files_abort_rather_than_lowering_the_declared_count() -> None:
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        _draw(code=_code(2, 1))
    (problem,) = raised.value.problems
    assert "needs 4 files" in problem and "2 eligible" in problem


def test_too_few_realized_repositories_abort_even_when_the_count_is_reachable() -> None:
    """The floor is about the design effect, and a corpus can fill its count while failing it."""
    tall = frame(b_code=replace(frame().b_code, max_files_per_repository=4))
    with pytest.raises(BenignDrawUnsatisfiable, match="realized 1 repositories"):
        _draw(code=_code(1, 4), the_frame=tall)


def test_more_files_than_the_cap_from_one_repository_abort() -> None:
    """`select_repository_files` applies the cap; this is the check that it was applied."""
    with pytest.raises(BenignDrawUnsatisfiable, match="more than 2 files"):
        _draw(code={"example-code-0": tuple(code_file("example-code-0", f"f{n}") for n in range(4)),
                    "example-code-1": tuple(code_file("example-code-1", f"f{n}") for n in range(4))})


def test_the_same_file_vendored_in_two_repositories_becomes_one_candidate() -> None:
    shared = code_file("example-code-0", "shared")
    duplicated = {
        "example-code-0": (shared, code_file("example-code-0", "a")),
        "example-code-1": (shared, code_file("example-code-1", "b")),
        "example-code-2": (code_file("example-code-2", "c"), code_file("example-code-2", "d")),
    }
    items, report = _draw(code=duplicated)
    assert report.code_candidates == 5
    assert len({item.id for item in items}) == len(items)


def test_the_draw_is_a_function_of_the_frame_and_not_of_the_reading_order() -> None:
    code = _code(3, 2)
    first, _ = _draw(code=code)
    reversed_code = {key: tuple(reversed(value)) for key, value in reversed(list(code.items()))}
    second, _ = _draw(code=reversed_code)
    assert sorted(item.id for item in first) == sorted(item.id for item in second)


def test_the_abort_has_an_exit_code_distinct_from_every_other() -> None:
    codes = declared_exit_codes()
    assert codes[BenignDrawUnsatisfiable.exit_code] is BenignDrawUnsatisfiable
    assert exit_code_for(BenignDrawUnsatisfiable("x")) == BenignDrawUnsatisfiable.exit_code


def test_a_draw_that_returns_the_wrong_count_is_refused_even_when_every_other_gate_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FR5.1's "exactly, never at least", with the one input that can still reach it.

    The shortfall gates above fire on the material; this one fires on the draw itself. Its failing
    input is a selection helper that returns fewer items than it was asked for -- which no declared
    method does, and which is exactly why the equality is checked rather than assumed. Without a
    reachable input this assertion was unreachable code wearing the name of the requirement.
    """
    real = benign.take

    def short(pool, size, **declared):
        drawn = real(pool, size, **declared)
        return drawn if drawn is None else drawn[:-1]

    monkeypatch.setattr(benign, "take", short)
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        _draw()
    assert any("realized 3 items against a declared 4" in p for p in raised.value.problems)


def test_one_payload_drawn_into_both_classes_is_refused() -> None:
    """The constructible id collision: same text, two classes, two rows under one item id.

    `id_collisions` cannot catch this -- it asks whether two *different* payloads landed on one id,
    which is a SHA-256 prefix collision nobody can construct. This is the other question, and the
    classes are deduplicated separately, so it is a real one.
    """
    from nbc.corpus.benign import CodeFile

    shared = MESSAGES[0].text
    code = dict(_code(3, 2))
    code["example-code-0"] = (
        CodeFile(
            repository_key="example-code-0",
            source="github.com/example/code@" + "c" * 40 + ":src/shared.js",
            path="src/shared.js",
            text=shared,
        ),
        code["example-code-0"][1],
    )
    with pytest.raises(BenignDrawUnsatisfiable) as raised:
        _draw(code=code)
    assert any("two rows share item id" in problem for problem in raised.value.problems)
