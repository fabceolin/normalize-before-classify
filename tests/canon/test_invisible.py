"""Step 1: what it removes, why that set and no other, and what the interpreter says about it.

The declared families are not taken on trust. Every property this module claims about them —
they are format characters, they are the complete set of explicit directional controls, they
survive NFKC untouched, they are disjoint from the confusables domain — is read off the
interpreter's own tables and compared, not restated.
"""

from __future__ import annotations

import unicodedata

import pytest

from nbc.canon import confusables_table
from nbc.canon.stages.invisible import (
    BIDI_CONTROL,
    BIDI_CONTROL_CLASSES,
    NAME,
    REMOVED,
    ZERO_WIDTH,
    run,
)
from nbc.schema import CanonContext, Edit, MAX_ASCII

DECLARED = ZERO_WIDTH + BIDI_CONTROL

ZWSP = "​"
ZWNJ = "‌"
RLO = "‮"


@pytest.fixture(scope="module")
def ctx() -> CanonContext:
    return CanonContext(confusables={0x0430: "a"}, ceiling=0)


# --- behaviour ------------------------------------------------------------------------------


def test_a_document_with_nothing_to_remove_is_untouched(ctx: CanonContext) -> None:
    result = run("hello world", ctx)
    assert result.text == "hello world"
    assert result.edits == ()


def test_one_zero_width_character_is_one_edit(ctx: CanonContext) -> None:
    result = run(f"a{ZWSP}b", ctx)
    assert result.text == "ab"
    assert result.edits == (Edit(stage=NAME, span=(1, 2), before=ZWSP, after=""),)


def test_adjacent_removals_coalesce_into_one_span(ctx: CanonContext) -> None:
    result = run(f"a{ZWSP}{ZWNJ}{RLO}b", ctx)
    assert result.text == "ab"
    assert result.edits == (
        Edit(stage=NAME, span=(1, 4), before=f"{ZWSP}{ZWNJ}{RLO}", after=""),
    )


def test_separated_removals_stay_two_edits(ctx: CanonContext) -> None:
    result = run(f"a{ZWSP}b{ZWSP}c", ctx)
    assert result.text == "abc"
    assert [edit.span for edit in result.edits] == [(1, 2), (3, 4)]


def test_a_document_of_nothing_but_invisibles_becomes_empty(ctx: CanonContext) -> None:
    result = run(ZWSP + ZWSP, ctx)
    assert result.text == ""
    assert result.edits == (Edit(stage=NAME, span=(0, 2), before=ZWSP + ZWSP, after=""),)


@pytest.mark.parametrize("code_point", [cp for cp, _ in DECLARED])
def test_every_declared_character_is_actually_removed(code_point: int, ctx: CanonContext) -> None:
    assert run(f"a{chr(code_point)}b", ctx).text == "ab"


def test_tracing_off_removes_the_same_characters_and_records_nothing() -> None:
    quiet = CanonContext(confusables={0x0430: "a"}, ceiling=0, trace_enabled=False)
    result = run(f"a{ZWSP}{RLO}b", quiet)
    assert result.text == "ab"
    assert result.edits == ()


# --- the declared set, checked against the interpreter ----------------------------------------


@pytest.mark.parametrize(("code_point", "name"), DECLARED)
def test_every_declared_character_is_a_format_character(code_point: int, name: str) -> None:
    assert unicodedata.category(chr(code_point)) == "Cf"


@pytest.mark.parametrize(("code_point", "name"), DECLARED)
def test_every_declared_name_is_the_name_the_interpreter_gives_it(
    code_point: int, name: str
) -> None:
    # The name is the human-readable half of the declaration. If it drifts from the code point,
    # the list stops being auditable by reading it, which is the only reason it is a list.
    assert unicodedata.name(chr(code_point)) == name


@pytest.mark.parametrize(("code_point", "name"), DECLARED)
def test_no_declared_character_is_ascii(code_point: int, name: str) -> None:
    assert code_point > MAX_ASCII


@pytest.mark.parametrize(("code_point", "name"), DECLARED)
def test_no_declared_character_changes_under_nfkc(code_point: int, name: str) -> None:
    # Step 1 runs before step 3. If NFKC would have altered one of these, removing it first would
    # be a different transformation from removing it after, and the order would be load-bearing in
    # a way nothing states.
    char = chr(code_point)
    assert unicodedata.normalize("NFKC", char) == char


@pytest.mark.parametrize(("code_point", "name"), BIDI_CONTROL)
def test_every_bidi_control_has_an_explicit_directional_class(code_point: int, name: str) -> None:
    assert unicodedata.bidirectional(chr(code_point)) in BIDI_CONTROL_CLASSES


@pytest.mark.parametrize(("code_point", "name"), ZERO_WIDTH)
def test_no_zero_width_character_has_an_explicit_directional_class(
    code_point: int, name: str
) -> None:
    # The two families are disjoint by construction, and the completeness check below depends on
    # that: a zero-width entry with a control class would make that check pass while the tuple it
    # checks is wrong.
    assert unicodedata.bidirectional(chr(code_point)) not in BIDI_CONTROL_CLASSES


def test_the_bidi_control_family_is_complete_over_the_whole_code_point_range() -> None:
    """The one family that can be derived is compared to the interpreter, not merely listed.

    A Unicode revision that adds a directional control fails here rather than silently leaving the
    layer blind to it — the same coupling the vendored confusables revision has to the interpreter.
    """
    found = {
        code_point
        for code_point in range(0x110000)
        if unicodedata.bidirectional(chr(code_point)) in BIDI_CONTROL_CLASSES
    }
    assert found == {code_point for code_point, _ in BIDI_CONTROL}


def test_the_declared_classes_are_the_nine_explicit_directional_formatting_classes() -> None:
    assert BIDI_CONTROL_CLASSES == frozenset(
        {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
    )


def test_the_two_families_do_not_overlap() -> None:
    assert not {cp for cp, _ in ZERO_WIDTH} & {cp for cp, _ in BIDI_CONTROL}


def test_no_code_point_is_declared_twice() -> None:
    assert len({cp for cp, _ in DECLARED}) == len(DECLARED)
    assert len(REMOVED) == len(DECLARED)


def test_the_removed_set_is_disjoint_from_the_confusables_domain() -> None:
    """Step 1 and step 2 may not contend for the same character.

    If one of these were also a confusables key, whichever step ran first would decide the output
    and the order would be silently load-bearing. This is the input that makes that visible.
    """
    table = confusables_table.load()
    assert not REMOVED & set(table.mapping)


def test_the_excluded_neighbours_are_excluded_on_purpose() -> None:
    """Soft hyphen and the invisible operators are format characters this step does not remove.

    Named here so the boundary is executable rather than only prose: widening it later changes a
    published number, and this test is what makes that a deliberate edit.
    """
    excluded = ["­", "⁡", "⁢", "⁣", "⁤", "\U000e0001"]
    for char in excluded:
        assert unicodedata.category(char) == "Cf"
        assert char not in REMOVED
