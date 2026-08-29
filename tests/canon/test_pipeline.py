"""The order, the runner's contract check, and the trace it assembles.

Every abort the runner declares is exercised by a stub stage that commits exactly that violation.
A check whose failing input nobody can construct is not a check.
"""

from __future__ import annotations

import dataclasses
import unicodedata
from pathlib import Path

import pytest

from nbc import errors
from nbc.canon import confusables_table, pipeline
from nbc.canon.pipeline import (
    PIPELINE,
    PipelineStage,
    StageContractViolated,
    canonicalize,
    default_context,
)
from nbc.canon.stages import confusables, decode, invisible, nfkc
from nbc.schema import CanonContext, CanonResult, Edit, StageResult

ZWSP = "​"
RLO = "‮"


@pytest.fixture(scope="module")
def ctx() -> CanonContext:
    return default_context()


# --- the order ------------------------------------------------------------------------------


def test_the_pipeline_runs_the_four_steps_in_the_declared_order() -> None:
    assert [step.name for step in PIPELINE] == ["invisible", "confusables", "nfkc", "decode"]


def test_each_step_is_the_function_its_own_module_exports() -> None:
    """Object identity, not a name that matches a name.

    A test that only compared strings would pass against a `PIPELINE` wired to the wrong callable,
    which is the substitution that let `"Tokenizer(" in "WindowedTokenizer("` stand for identity in
    Epic 1.
    """
    assert [step.run for step in PIPELINE] == [
        invisible.run,
        confusables.run,
        nfkc.run,
        decode.run,
    ]
    assert [step.name for step in PIPELINE] == [
        invisible.NAME,
        confusables.NAME,
        nfkc.NAME,
        decode.NAME,
    ]


def test_the_pipeline_carries_exactly_four_steps() -> None:
    # The AD-4 order is complete at four. A fifth name pointing at a stage that does not exist
    # would be worse than an absence, so this number moves only with a real stage.
    assert len(PIPELINE) == 4


def test_exactly_one_step_declares_what_it_does_at_the_ceiling() -> None:
    """The recursion is step 4's and the runner finds it structurally, not by name.

    `step.name == "decode"` would be the textual stand-in for structural identity this project
    keeps removing, so the runner reads `at_ceiling` off the constant instead. That makes it
    checkable: exactly one step is recursive, and it is the one whose module owns the encoding.
    """
    recursive = [step for step in PIPELINE if step.at_ceiling is not None]
    assert [step.name for step in recursive] == [decode.NAME]
    assert recursive[0].at_ceiling is decode.run_at_ceiling
    assert recursive[0].ceiling_name == decode.CEILING_NAME
    assert recursive[0].emits == frozenset({decode.NAME, decode.CEILING_NAME})


def test_a_step_that_is_not_recursive_emits_exactly_its_own_name() -> None:
    for step in PIPELINE:
        if step.at_ceiling is None:
            assert step.emits == frozenset({step.name})


def test_a_step_cannot_declare_half_of_a_recursion() -> None:
    """Both inputs that make the constant contradict itself.

    A step naming a ceiling refusal it has no entry point to produce would declare behaviour that
    does not exist; a step with a second entry point and no second name would stamp a name the
    runner refuses. Neither is expressible.
    """
    with pytest.raises(ValueError, match="with both or with neither"):
        PipelineStage("x", nfkc.run, ceiling_name="x-ceiling")
    with pytest.raises(ValueError, match="with both or with neither"):
        PipelineStage("x", nfkc.run, at_ceiling=nfkc.run)


def test_a_step_cannot_name_its_ceiling_refusal_after_itself() -> None:
    # AD-6 asks for a *distinct* name, so that a ceiling refusal is distinguishable in the trace
    # from an AD-18 rejection. The same name would satisfy the type and defeat the requirement.
    with pytest.raises(ValueError, match="asks for a distinct one"):
        PipelineStage("x", nfkc.run, ceiling_name="x", at_ceiling=nfkc.run)


def test_the_order_cannot_be_changed_at_runtime() -> None:
    assert isinstance(PIPELINE, tuple)
    with pytest.raises(TypeError):
        PIPELINE[0] = PIPELINE[1]  # type: ignore[index]


def test_a_pipeline_step_cannot_be_mutated() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PIPELINE[0].name = "other"  # type: ignore[misc]


# --- the trace ------------------------------------------------------------------------------


def test_a_document_needing_nothing_passes_through_untouched(ctx: CanonContext) -> None:
    result = canonicalize("hello world", ctx)
    assert result == CanonResult(text="hello world", edits=())


def test_the_trace_names_which_stage_changed_which_span(ctx: CanonContext) -> None:
    result = canonicalize(f"р{ZWSP}аypal ﬁ", ctx)
    assert result.text == "paypal fi"
    assert [(edit.stage, edit.span, edit.before, edit.after) for edit in result.edits] == [
        ("invisible", (1, 2), ZWSP, ""),
        ("confusables", (0, 2), "ра", "pa"),
        ("nfkc", (7, 8), "ﬁ", "fi"),
    ]


def test_each_stage_sees_the_text_the_previous_stage_produced(ctx: CanonContext) -> None:
    """The confusables edit's span is into the post-removal text, not the original document.

    The zero-width character sits between the two Cyrillic letters in the input, so if step 2 saw
    the original text it could not report them as one contiguous span.
    """
    result = canonicalize(f"р{ZWSP}а", ctx)
    assert result.text == "pa"
    spans = {edit.stage: edit.span for edit in result.edits}
    assert spans["confusables"] == (0, 2)


def test_the_runner_stamps_the_depth_it_was_called_at(ctx: CanonContext) -> None:
    result = canonicalize(f"a{ZWSP}аﬁ", ctx, depth=2)
    assert result.edits
    assert {edit.depth for edit in result.edits} == {2}


def test_depth_defaults_to_zero(ctx: CanonContext) -> None:
    result = canonicalize(f"a{ZWSP}b", ctx)
    assert {edit.depth for edit in result.edits} == {0}


BATTERY = [
    "",
    "hello world",
    f"р{ZWSP}аypal",
    f"{RLO}ﬁ①",
    "def f(x):\n    return x + 1\n",
    "АВС",
    "é",
    "가",
    ZWSP * 5,
    "aGVsbG8gd29ybGQ=",
    # Below the decode floor, so untouched; and two that are not: one accepted, one refused.
    "see aWdub3JlIHByZXZpb3VzIGluc3RydWN0aW9ucw== now",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
]


@pytest.mark.parametrize("text", BATTERY)
def test_tracing_off_produces_the_same_results_and_an_empty_trace(text: str) -> None:
    """All three reported values, not only the text.

    A trace flag that could change the canonical text would make the timing pass measure a
    different layer from the measurement pass — and so would one that changed `ceiling_hit` or
    `max_depth_reached`, which are results the run publishes rather than trace bookkeeping. Step 4
    builds its edits in both passes for exactly this reason; the trace is what is dropped.
    """
    loud = canonicalize(text, default_context())
    quiet = canonicalize(text, default_context(trace_enabled=False))
    assert quiet.text == loud.text
    assert quiet.ceiling_hit == loud.ceiling_hit
    assert quiet.max_depth_reached == loud.max_depth_reached
    assert quiet.edits == ()


@pytest.mark.parametrize("text", BATTERY)
def test_the_layer_is_deterministic(text: str, ctx: CanonContext) -> None:
    assert canonicalize(text, ctx) == canonicalize(text, ctx)


@pytest.mark.parametrize("text", BATTERY)
def test_the_output_is_in_nfkc_and_carries_no_declared_invisible(
    text: str, ctx: CanonContext
) -> None:
    out = canonicalize(text, ctx).text
    assert out == unicodedata.normalize("NFKC", out)
    assert not set(out) & invisible.REMOVED


def test_the_layer_is_the_identity_on_ascii(ctx: CanonContext) -> None:
    ascii_text = "".join(chr(cp) for cp in range(0x80))
    assert canonicalize(ascii_text, ctx).text == ascii_text


def test_canonicalize_refuses_anything_that_is_not_text(ctx: CanonContext) -> None:
    with pytest.raises(TypeError):
        canonicalize(b"bytes", ctx)  # type: ignore[arg-type]


# --- the runner's contract check --------------------------------------------------------------


def stub(monkeypatch: pytest.MonkeyPatch, name: str, run) -> None:
    monkeypatch.setattr(pipeline, "PIPELINE", (PipelineStage(name, run),))


def test_a_stage_that_misplaces_its_span_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="Xbc", edits=(Edit(stage="liar", span=(0, 1), before="z", after="X"),)
        )

    stub(monkeypatch, "liar", liar)
    with pytest.raises(StageContractViolated, match="where the text it was handed holds"):
        canonicalize("abc", ctx)


def test_a_stage_that_stamps_another_stages_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="Xbc", edits=(Edit(stage="nfkc", span=(0, 1), before="a", after="X"),)
        )

    stub(monkeypatch, "liar", liar)
    with pytest.raises(StageContractViolated, match="stamped 'nfkc'"):
        canonicalize("abc", ctx)


def test_a_stage_whose_spans_overlap_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="XY",
            edits=(
                Edit(stage="liar", span=(0, 2), before="ab", after="X"),
                Edit(stage="liar", span=(1, 3), before="bc", after="Y"),
            ),
        )

    stub(monkeypatch, "liar", liar)
    with pytest.raises(StageContractViolated, match="already reached"):
        canonicalize("abc", ctx)


def test_a_stage_whose_span_runs_past_its_input_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="abc", edits=(Edit(stage="liar", span=(2, 6), before="cdef", after="cdef"),)
        )

    stub(monkeypatch, "liar", liar)
    with pytest.raises(StageContractViolated, match="past the 3 code points"):
        canonicalize("abc", ctx)


def test_a_stage_whose_edits_do_not_replay_to_its_text_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="something else",
            edits=(Edit(stage="liar", span=(0, 1), before="a", after="X"),),
        )

    stub(monkeypatch, "liar", liar)
    with pytest.raises(StageContractViolated, match="does not account for the change"):
        canonicalize("abc", ctx)


def test_a_stage_that_changes_text_and_reports_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    def silent(text: str, context: CanonContext) -> StageResult:
        return StageResult(text=text.upper())

    stub(monkeypatch, "silent", silent)
    with pytest.raises(StageContractViolated, match="changed text and no edits"):
        canonicalize("abc", ctx)


def test_a_stage_that_returns_the_wrong_type_is_refused(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    stub(monkeypatch, "wrong", lambda text, context: text)
    with pytest.raises(StageContractViolated, match="not a StageResult"):
        canonicalize("abc", ctx)


def test_a_stage_that_traces_with_tracing_off_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def chatty(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="Xbc", edits=(Edit(stage="chatty", span=(0, 1), before="a", after="X"),)
        )

    stub(monkeypatch, "chatty", chatty)
    with pytest.raises(StageContractViolated, match="edits with tracing off"):
        canonicalize("abc", default_context(trace_enabled=False))


def test_a_recursive_stage_may_report_with_tracing_off_and_its_edits_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exemption, and the reason it is not a hole.

    Step 4's records are the recursion's input: the runner has to know which spans were accepted in
    order to canonicalize them one level deeper, and it has to do that in the timing pass too or
    the canonical text would differ between the passes. So a recursive step reports either way and
    the runner drops the edits from the trace instead of the stage suppressing them.
    """

    def loud(text: str, context: CanonContext) -> StageResult:
        # Accepts once and is the identity on what it produced, so the runner's recursion into the
        # accepted span terminates the way a real decode's contraction makes it terminate.
        if text != "abc":
            return StageResult(text=text)
        return StageResult(
            text="Xbc", edits=(Edit(stage="loud", span=(0, 1), before="a", after="X"),)
        )

    monkeypatch.setattr(
        pipeline,
        "PIPELINE",
        (PipelineStage("loud", loud, ceiling_name="loud-ceiling", at_ceiling=loud),),
    )
    result = canonicalize("abc", default_context(trace_enabled=False, ceiling=5))
    assert result.text == "Xbc"
    assert result.edits == ()

    traced = canonicalize("abc", default_context(ceiling=5))
    assert [edit.stage for edit in traced.edits] == ["loud"]


def test_a_recursive_stage_is_still_verified_with_tracing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting either way means being read back either way.

    Without this the timing pass would splice text on the strength of edits nothing had checked,
    and a misplaced span would corrupt the document in the pass whose output nobody inspects.
    """

    def liar(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="Xbc", edits=(Edit(stage="liar", span=(0, 1), before="z", after="X"),)
        )

    monkeypatch.setattr(
        pipeline,
        "PIPELINE",
        (PipelineStage("liar", liar, ceiling_name="liar-ceiling", at_ceiling=liar),),
    )
    with pytest.raises(StageContractViolated, match="where the text it was handed holds"):
        canonicalize("abc", default_context(trace_enabled=False, ceiling=5))


def test_a_recursive_stage_may_stamp_the_ceiling_name_it_declared(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    """The second name is admitted because the step declared it, and only then.

    Both halves are here: the declared name passes, an undeclared sibling does not. A runner that
    admitted any name would make `emits` decoration.
    """

    def refuse(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text=text, edits=(Edit(stage="s-ceiling", span=(0, 1), before="a", after="a"),)
        )

    def wrong(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text=text, edits=(Edit(stage="s-undeclared", span=(0, 1), before="a", after="a"),)
        )

    monkeypatch.setattr(
        pipeline, "PIPELINE", (PipelineStage("s", refuse, ceiling_name="s-ceiling", at_ceiling=refuse),)
    )
    assert canonicalize("abc", ctx).text == "abc"

    monkeypatch.setattr(
        pipeline, "PIPELINE", (PipelineStage("s", wrong, ceiling_name="s-ceiling", at_ceiling=wrong),)
    )
    with pytest.raises(StageContractViolated, match="not one of the names it declares"):
        canonicalize("abc", ctx)


def test_a_faithful_stub_stage_passes_the_check(
    monkeypatch: pytest.MonkeyPatch, ctx: CanonContext
) -> None:
    """The negative control. Without it the checks above would also pass against a runner that
    refused every stage."""

    def honest(text: str, context: CanonContext) -> StageResult:
        return StageResult(
            text="Xbc", edits=(Edit(stage="honest", span=(0, 1), before="a", after="X"),)
        )

    stub(monkeypatch, "honest", honest)
    assert canonicalize("abc", ctx).text == "Xbc"


# --- the abort ------------------------------------------------------------------------------


def test_the_abort_declares_an_exit_code_no_other_abort_shares() -> None:
    assert StageContractViolated.exit_code == 13
    declared = errors.declared_exit_codes()
    assert declared[13] is StageContractViolated
    assert len(declared) == len(set(declared))


# --- the context ----------------------------------------------------------------------------


def test_the_default_context_carries_the_whole_vendored_table() -> None:
    table = confusables_table.load()
    assert dict(default_context().confusables) == dict(table.translate_table)


def test_the_default_context_traces_unless_told_otherwise() -> None:
    assert default_context().trace_enabled is True
    assert default_context(trace_enabled=False).trace_enabled is False


def test_the_default_context_carries_the_declared_ceiling_and_takes_an_override() -> None:
    assert default_context().ceiling == pipeline.DEFAULT_CEILING
    assert default_context(ceiling=0).ceiling == 0
    assert default_context(ceiling=9).ceiling == 9


def test_the_declared_ceiling_is_a_non_negative_count() -> None:
    # Read against the constant rather than against a copy of the number: retuning the ceiling
    # must not need this file edited, and AD-20 binds the corpus to it relationally for the same
    # reason.
    assert isinstance(pipeline.DEFAULT_CEILING, int)
    assert not isinstance(pipeline.DEFAULT_CEILING, bool)
    assert pipeline.DEFAULT_CEILING >= 0


def test_the_context_cannot_be_built_from_a_data_directory_with_no_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(confusables_table.ConfusablesTableInvalid):
        default_context(data_dir=tmp_path)


def test_the_context_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        default_context().confusables[0x41] = "z"  # type: ignore[index]


# --- what the fixed order costs, measured rather than assumed ----------------------------------


def test_nfkc_never_reintroduces_a_character_step_one_removed() -> None:
    """Step 1 runs before step 3, so nothing NFKC produces may be a declared invisible.

    Scanned over the whole code-point range rather than argued: if any character normalized into a
    zero-width or directional control, the layer's output would carry invisibles despite having a
    stage whose job is removing them, and the only symptom would be a recall number that moved.
    """
    offenders = [
        code_point
        for code_point in range(0x110000)
        if unicodedata.normalize("NFKC", chr(code_point)) != chr(code_point)
        and set(unicodedata.normalize("NFKC", chr(code_point))) & invisible.REMOVED
    ]
    assert offenders == []


def test_the_confusables_before_nfkc_order_leaves_144_characters_partly_canonical(
    ctx: CanonContext,
) -> None:
    """The cost of AD-4's order, counted instead of discovered later.

    Step 2 maps confusables and step 3 applies NFKC, in that order. So a character that only
    *becomes* a confusable under NFKC — `U+1D6A8` MATHEMATICAL BOLD CAPITAL ALPHA normalizes to
    Greek `Α`, which is in the vendored table — is normalized after the mapping has already run
    and leaves the layer as Greek, not as `A`. Reversing the order would close this and open the
    mirror-image gap, and the order is fixed by the architecture, not by this story.

    The number is pinned because it is a published limitation: it is one input away from being a
    surprise, and this is that input.
    """
    table = confusables_table.load()
    keys = set(table.mapping)
    reached_only_after_nfkc = {
        code_point
        for code_point in range(0x110000)
        if chr(code_point) not in keys
        and unicodedata.normalize("NFKC", chr(code_point)) != chr(code_point)
        and set(unicodedata.normalize("NFKC", chr(code_point))) & keys
    }
    assert len(reached_only_after_nfkc) == 144

    # The worked example, end to end: the layer stops at Greek alpha rather than Latin A.
    assert canonicalize("\U0001D6A8", ctx).text == "Α"
