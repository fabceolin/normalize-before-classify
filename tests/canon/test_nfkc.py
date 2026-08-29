"""Step 3: the output is exactly what NFKC says, and the trace locates the change where it can."""

from __future__ import annotations

import unicodedata

import pytest

from nbc.canon.stages.nfkc import NAME, run
from nbc.schema import MAX_ASCII, CanonContext, Edit

# A battery wide enough that "the stage always agrees with the standard library" is a claim about
# more than the three examples the story names.
BATTERY = [
    "",
    "hello world",
    "ﬁ",
    "①②③",
    "ＡＢＣ",
    "é",
    "é",
    "á̧b",
    "가",  # conjoining jamo: the input that takes the fallback
    "ᄀ",
    "x가y",
    "①ﬁ",
    "a①b②c",
    "½",
    "ﬁﬂ",
    "café ﬁle",
    "no change at all",
    "①" * 40,
    "µ",
    "Ω",
]


@pytest.fixture(scope="module")
def ctx() -> CanonContext:
    return CanonContext(confusables={0x0430: "a"}, ceiling=0)


def test_a_document_already_in_nfkc_is_untouched(ctx: CanonContext) -> None:
    result = run("hello world", ctx)
    assert result.text == "hello world"
    assert result.edits == ()


def test_compatibility_characters_are_normalized_and_located(ctx: CanonContext) -> None:
    result = run("a ﬁ b", ctx)
    assert result.text == "a fi b"
    assert result.edits == (Edit(stage=NAME, span=(2, 3), before="ﬁ", after="fi"),)


def test_adjacent_changed_segments_coalesce_into_one_span(ctx: CanonContext) -> None:
    result = run("ﬁ①", ctx)
    assert result.text == "fi1"
    assert result.edits == (Edit(stage=NAME, span=(0, 2), before="ﬁ①", after="fi1"),)


def test_separated_changes_stay_two_edits(ctx: CanonContext) -> None:
    result = run("ﬁx①", ctx)
    assert result.text == "fix1"
    assert [edit.span for edit in result.edits] == [(0, 1), (2, 3)]


def test_a_composition_across_a_combining_mark_is_one_edit(ctx: CanonContext) -> None:
    result = run("é", ctx)
    assert result.text == "é"
    assert result.edits == (Edit(stage=NAME, span=(0, 2), before="é", after="é"),)


def test_two_separated_compositions_stay_two_edits(ctx: CanonContext) -> None:
    """The combining-class-0 boundary is what earns the fine-grained trace.

    Splitting at every code point instead would tear both compositions, the segments would not
    join back to the NFKC form, and the fallback would report one span covering the whole
    document. That is the input which makes the boundary rule pay its way.
    """
    result = run("e\u0301 middle e\u0301", ctx)
    assert result.text == "\u00e9 middle \u00e9"
    assert [edit.span for edit in result.edits] == [(0, 2), (10, 12)]


def test_conjoining_hangul_jamo_take_the_coalesced_fallback(ctx: CanonContext) -> None:
    """The constructible input where per-segment attribution does not join back.

    `U+1100` and `U+1161` are both starters, so the combining-class-0 split separates a
    composition and the segments normalize to themselves. The stage must still emit the composed
    text, and one edit rather than a wrong two.
    """
    result = run("가", ctx)
    assert result.text == "가"
    assert result.edits == (
        Edit(stage=NAME, span=(0, 2), before="가", after="가"),
    )


def test_the_fallback_trims_what_both_sides_agree_on(ctx: CanonContext) -> None:
    result = run("head가tail", ctx)
    assert result.text == "head가tail"
    assert result.edits == (
        Edit(stage=NAME, span=(4, 6), before="가", after="가"),
    )


@pytest.mark.parametrize("text", BATTERY)
def test_the_stage_never_disagrees_with_the_standard_library(text: str, ctx: CanonContext) -> None:
    assert run(text, ctx).text == unicodedata.normalize("NFKC", text)


@pytest.mark.parametrize("text", BATTERY)
def test_replaying_the_edits_reproduces_the_output(text: str, ctx: CanonContext) -> None:
    """The trace is checked against the transformation, on every input in the battery.

    The runner does this too, on every document. Doing it here as well is the difference between
    trusting the runner's check and having one that can fail on its own.
    """
    result = run(text, ctx)
    rebuilt: list[str] = []
    position = 0
    for edit in result.edits:
        start, end = edit.span
        assert start >= position
        assert text[start:end] == edit.before
        rebuilt.append(text[position:start])
        rebuilt.append(edit.after)
        position = end
    rebuilt.append(text[position:])
    assert "".join(rebuilt) == result.text


@pytest.mark.parametrize("text", BATTERY)
def test_tracing_off_produces_the_same_text_and_no_edits(text: str) -> None:
    quiet = CanonContext(confusables={0x0430: "a"}, ceiling=0, trace_enabled=False)
    loud = CanonContext(confusables={0x0430: "a"}, ceiling=0)
    assert run(text, quiet).text == run(text, loud).text
    assert run(text, quiet).edits == ()


def test_every_ascii_code_point_is_its_own_nfkc_form() -> None:
    """The fact the single-ASCII-character shortcut in `_cover` rests on.

    Asserted over the whole range rather than assumed, because the shortcut skips the call that
    would otherwise prove it one character at a time.
    """
    for code_point in range(MAX_ASCII + 1):
        char = chr(code_point)
        assert unicodedata.normalize("NFKC", char) == char


def test_ascii_text_is_untouched(ctx: CanonContext) -> None:
    ascii_text = "".join(chr(cp) for cp in range(MAX_ASCII + 1))
    result = run(ascii_text, ctx)
    assert result.text == ascii_text
    assert result.edits == ()
