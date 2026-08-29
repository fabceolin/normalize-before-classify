"""Step 2: the vendored table applied per code point, and the ASCII promise it must keep."""

from __future__ import annotations

import pytest

from nbc.canon import confusables_table
from nbc.canon.pipeline import default_context
from nbc.canon.stages.confusables import NAME, run
from nbc.schema import MAX_ASCII, CanonContext, Edit


@pytest.fixture(scope="module")
def ctx() -> CanonContext:
    return default_context()


def test_a_document_with_no_confusable_is_untouched(ctx: CanonContext) -> None:
    result = run("paypal login", ctx)
    assert result.text == "paypal login"
    assert result.edits == ()


def test_a_run_of_confusables_is_one_edit(ctx: CanonContext) -> None:
    result = run("раypal", ctx)  # Cyrillic er and a
    assert result.text == "paypal"
    assert result.edits == (Edit(stage=NAME, span=(0, 2), before="ра", after="pa"),)


def test_separated_confusables_stay_two_edits(ctx: CanonContext) -> None:
    result = run("рaypаl", ctx)  # Cyrillic er at 0, Cyrillic a at 4
    assert result.text == "paypal"
    assert [edit.span for edit in result.edits] == [(0, 1), (4, 5)]


def test_a_confusable_whose_latin_form_is_two_characters_keeps_a_one_code_point_span(
    ctx: CanonContext,
) -> None:
    result = run("Ы", ctx)
    assert result.text == "bl"
    assert result.edits == (Edit(stage=NAME, span=(0, 1), before="Ы", after="bl"),)


def test_the_stage_is_the_identity_on_every_ascii_code_point(ctx: CanonContext) -> None:
    """Asserted over the whole range, not sampled.

    This is the property that keeps the benign-code counter-metric a number about the layer rather
    than a number about ASCII folding, and that leaves base64 and hex runs intact for step 4.
    """
    ascii_text = "".join(chr(cp) for cp in range(MAX_ASCII + 1))
    result = run(ascii_text, ctx)
    assert result.text == ascii_text
    assert result.edits == ()


def test_every_entry_in_the_vendored_table_is_applied(ctx: CanonContext) -> None:
    table = confusables_table.load()
    for key, value in table.mapping.items():
        assert run(f"[{key}]", ctx).text == f"[{value}]"


def test_the_stage_reads_the_table_off_the_context_and_not_off_disk() -> None:
    """A hand-built context with one entry maps that entry and nothing else.

    The failing input for "the stage loads its own table" is exactly this: if it did, `ф` would
    map through the vendored table's entry and `"ф"` would not survive.
    """
    narrow = CanonContext(confusables={0x0430: "Z"}, ceiling=0)
    result = run("аф", narrow)
    assert result.text == "Zф"


def test_tracing_off_maps_the_same_characters_and_records_nothing() -> None:
    quiet = default_context(trace_enabled=False)
    result = run("раypal", quiet)
    assert result.text == "paypal"
    assert result.edits == ()


def test_a_context_that_folds_ascii_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="ASCII code point"):
        CanonContext(confusables={0x61: "b"}, ceiling=0)
