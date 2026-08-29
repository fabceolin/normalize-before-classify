"""The granularity rule, and the covers that are not covers.

Every rule `build_edits` enforces is exercised here with the input that makes it fail. A rule with
no such input is not a check, and would not be described as one.
"""

from __future__ import annotations

import pytest

from nbc.canon.edits import build_edits, build_reported_edits, map_code_points
from nbc.schema import Edit

STAGE = "probe"


def apply(text: str, replace, *, trace: bool = True) -> tuple[str, tuple[Edit, ...]]:
    return build_edits(text, map_code_points(text, replace), stage=STAGE, trace=trace)


def drop(chars: str):
    return lambda char: "" if char in chars else None


def test_an_untouched_document_costs_one_segment_and_no_edits() -> None:
    text, edits = apply("hello world", lambda char: None)
    assert text == "hello world"
    assert edits == ()


def test_the_cover_of_an_untouched_document_is_a_single_run() -> None:
    # The per-code-point walk must not emit one segment per character, or a 100 KB source file
    # allocates 100 000 tuples per stage on the path whose p50 the run publishes.
    assert list(map_code_points("hello", lambda char: None)) == [(0, 5, "hello")]


def test_one_changed_character_is_one_edit_holding_only_that_span() -> None:
    text, edits = apply("axb", drop("x"))
    assert text == "ab"
    assert edits == (Edit(stage=STAGE, span=(1, 2), before="x", after=""),)


def test_adjacent_changes_coalesce_into_one_span() -> None:
    text, edits = apply("axyzb", drop("xyz"))
    assert text == "ab"
    assert edits == (Edit(stage=STAGE, span=(1, 4), before="xyz", after=""),)


def test_changes_separated_by_untouched_text_stay_separate_edits() -> None:
    text, edits = apply("axbxc", drop("x"))
    assert text == "abc"
    assert [edit.span for edit in edits] == [(1, 2), (3, 4)]


def test_a_document_that_is_entirely_removed_is_one_edit() -> None:
    text, edits = apply("xx", drop("x"))
    assert text == ""
    assert edits == (Edit(stage=STAGE, span=(0, 2), before="xx", after=""),)


def test_a_replacement_longer_than_its_source_keeps_the_span_on_the_source() -> None:
    text, edits = apply("aXb", lambda char: "long" if char == "X" else None)
    assert text == "alongb"
    assert edits == (Edit(stage=STAGE, span=(1, 2), before="X", after="long"),)


def test_a_replacement_equal_to_its_source_is_not_a_change() -> None:
    # `map_code_points` may hand back the same text for a code point it looked at; that is not a
    # changed span, and recording it as one would inflate every per-stage edit count.
    text, edits = apply("abc", lambda char: char if char == "b" else None)
    assert text == "abc"
    assert edits == ()


def test_the_empty_document_produces_nothing() -> None:
    assert apply("", drop("x")) == ("", ())


def test_tracing_off_still_transforms_and_records_nothing() -> None:
    text, edits = apply("axb", drop("x"), trace=False)
    assert text == "ab"
    assert edits == ()


@pytest.mark.parametrize(
    ("segments", "needle"),
    [
        ([(0, 1, "a"), (2, 3, "c")], "had reached"),  # a gap
        ([(0, 2, "ab"), (1, 3, "bc")], "had reached"),  # an overlap
        ([(0, 2, "ab"), (2, 1, "")], "reversed"),  # end before start
        ([(0, 9, "abc")], "past the"),  # past the end of the text
        ([(0, 2, "ab")], "covered 2 of 3"),  # stops short
        ([(1, 3, "bc")], "had reached 0"),  # starts late
    ],
)
def test_a_cover_that_is_not_a_partition_is_refused(segments, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        build_edits("abc", segments, stage=STAGE, trace=True)


def test_a_broken_cover_is_refused_even_with_tracing_off() -> None:
    # The partition check guards the output text, not the trace, so switching the trace off must
    # not switch it off. Without this, the timing pass would silently accept a truncating stage.
    with pytest.raises(ValueError, match="covered 2 of 3"):
        build_edits("abc", [(0, 2, "ab")], stage=STAGE, trace=False)


# --- reported spans: the edit a stage makes about a span it left alone --------------------------


def report(text: str, spans) -> tuple[str, tuple[Edit, ...]]:
    """Every span reported under the same stage name; the two-name case has its own tests below."""
    return build_reported_edits(text, [(a, b, r, STAGE) for a, b, r in spans])


def test_a_reported_span_that_did_not_change_is_still_an_edit() -> None:
    """The whole reason this function exists beside `build_edits`, which drops exactly this."""
    text, edits = report("see abcdef now", [(4, 10, "abcdef")])
    assert text == "see abcdef now"
    assert edits == (Edit(stage=STAGE, span=(4, 10), before="abcdef", after="abcdef"),)

    dropped = build_edits(
        "see abcdef now",
        [(0, 4, "see "), (4, 10, "abcdef"), (10, 14, " now")],
        stage=STAGE,
        trace=True,
    )
    assert dropped == ("see abcdef now", ())


def test_a_reported_span_that_changed_is_replaced_in_place() -> None:
    text, edits = report("see abcdef now", [(4, 10, "X")])
    assert text == "see X now"
    assert edits == (Edit(stage=STAGE, span=(4, 10), before="abcdef", after="X"),)


def test_the_text_between_reported_spans_is_copied_through_unreported() -> None:
    text, edits = report("a bb c dd e", [(2, 4, "B"), (7, 9, "D")])
    assert text == "a B c D e"
    assert [edit.span for edit in edits] == [(2, 4), (7, 9)]


def test_no_reported_spans_leaves_the_text_alone() -> None:
    assert report("hello", []) == ("hello", ())


def test_reported_spans_are_not_coalesced_even_when_adjacent() -> None:
    # Two decisions about two candidates stay two entries. Coalescing them would report two
    # refusals as one, and the trace's job is to say how many candidates were examined.
    text, edits = report("abcd", [(0, 2, "X"), (2, 4, "Y")])
    assert text == "XY"
    assert len(edits) == 2


def test_overlapping_reported_spans_are_refused() -> None:
    with pytest.raises(ValueError, match="ordered and never overlap"):
        report("abcdefgh", [(0, 5, "X"), (3, 8, "Y")])


def test_reported_spans_out_of_order_are_refused() -> None:
    with pytest.raises(ValueError, match="ordered and never overlap"):
        report("abcdefgh", [(4, 6, "X"), (0, 2, "Y")])


def test_a_reversed_reported_span_is_refused() -> None:
    with pytest.raises(ValueError, match="reversed span"):
        report("abcdefgh", [(5, 2, "X")])


def test_a_reported_span_past_the_end_is_refused() -> None:
    with pytest.raises(ValueError, match="past the 8 code points"):
        report("abcdefgh", [(4, 12, "X")])


def test_a_reported_span_carries_the_stage_name_of_its_own_decision() -> None:
    """Two kinds of decision, one ordered pass, two names.

    A ceiling refusal and an AD-18 rejection are the same shape — `before == after` — and can be
    interleaved in one document at one depth. Without a name per span they would be one entry kind
    in the trace, which is exactly what AD-6 asks not to happen.
    """
    text, edits = build_reported_edits(
        "a bbbb c dddd e",
        [(2, 6, "bbbb", "decode"), (9, 13, "dddd", "decode-ceiling")],
    )
    assert text == "a bbbb c dddd e"
    assert [(edit.stage, edit.span) for edit in edits] == [
        ("decode", (2, 6)),
        ("decode-ceiling", (9, 13)),
    ]


def test_reported_edits_are_built_whether_or_not_anyone_will_read_them() -> None:
    """There is no trace switch here, and that is the point.

    These edits are how the runner learns which spans were accepted and must be canonicalized one
    level deeper. A switch that could suppress them would change the canonical text of the timing
    pass, not just its trace. Whether they reach the document's trace is the runner's decision, and
    `tests/canon/test_recursion.py` is where that is checked.
    """
    with pytest.raises(TypeError):
        build_reported_edits("abc", [(0, 1, "X", STAGE)], trace=False)  # type: ignore[call-arg]
