"""The canonicalization order, the runner, and the trace the runner owns.

**One constant.** `PIPELINE` is the order, and it is the only place the order exists. AD-4's whole
point is that NFKC before or after zero-width removal is a different transformation, and a project
where each caller assembles its own sequence resolves that question differently in every module. A
reorder here is a one-line diff a reviewer sees; a test asserts the order, so it is also a failing
test until someone changes the assertion too.

**Four steps.** Steps 1 to 3 rewrite characters; step 4 decodes embedded encodings. Step 4 is last
because the three before it are what make a run visible as a run: a zero-width joiner in the middle
of a base64 blob leaves no base64 blob for step 4 to find. The order is not a preference and is not
this module's to change.

**The runner checks what the stages report.** Each stage hands back the text it produced and the
edits that account for it. The runner replays those edits over the text it handed *in* and compares
the result to the text the stage handed *out*. The two sides come from different places, so the
trace is verified against the transformation rather than recorded beside it — and a stage that
misattributes a change fails a test instead of quietly publishing a wrong attribution into
`traces.jsonl`. That check is the difference between a trace and a claim about a trace.

**What the fixed order costs, stated rather than discovered.** Step 2 maps confusables and step 3
applies NFKC, in that order, so a character that only *becomes* a confusable under NFKC is
normalized after the mapping has already run. `U+1D6A8` MATHEMATICAL BOLD CAPITAL ALPHA leaves the
layer as Greek `\u0391`, not as `A`. There are 144 such code points at this Unicode revision, and
the number is pinned by a test so it stays a published limitation instead of a surprise. Reversing
the order would close this gap and open its mirror image; the order is the architecture's, not this
module's, and this is the price of having one.

**Depth belongs to the runner.** `Stage(text, ctx) -> StageResult` has no depth in it, by AD-5, and
a stage genuinely does not know at what recursion depth it is being run. The runner stamps the
depth it was called at onto every edit it collects, and it is the only thing that decides whether
the ceiling has been reached. A step that behaves differently at the ceiling says so by declaring
a second entry point in `PIPELINE` — not by being recognized by name, which would be the textual
stand-in for structural identity this project keeps finding and removing.

**The recursion, in one place.** Below the ceiling, every span step 4 accepted is canonicalized as
an **independent document** through all four steps at `depth + 1`, unconditionally, including when
it holds no further candidate; the result replaces the source span, and the host document is not
re-scanned afterwards, so a candidate that only appears across the seam of a replacement is not
decoded. Sibling breadth is unbounded: there is no per-document decode budget, only the per-branch
depth, because a benign file with fifty base64 spans is not a fifty-level nest and must not be
truncated as though it were.

**The trace is flat, and the order is the reading rule.** `Edit` has five fields and no parent
pointer, so two sibling decodes both emit depth-`d+1` edits whose spans are into their own
segments. The rule is positional and it is tested: an accepted decode's edit is immediately
followed by every edit of the sub-document it produced.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from nbc.canon import confusables_table
from nbc.canon.stages import confusables, decode, invisible, nfkc
from nbc.errors import NbcError
from nbc.schema import CanonContext, CanonResult, Edit, StageResult

__all__ = (
    "DEFAULT_CEILING",
    "PIPELINE",
    "PipelineStage",
    "Stage",
    "StageContractViolated",
    "canonicalize",
    "default_context",
    "trace_stage_labels",
)

DEFAULT_CEILING: Final[int] = 3
"""How many levels of nested decoding the layer opens, unless a run declares otherwise.

FR10 asks for an explicit parameter with a **declared** default, so this is the one place the
number exists: `default_context` reads it, `CanonContext.ceiling` carries it, and no stage holds a
literal. A test asserts this module is the only one in `src/` that reads it.

**Why three.** Three levels recovers deeper nesting than any attack in the pinned corpora needs —
double-encoding is already unusual — while bounding the work per branch at three decodes plus the
one the ceiling check itself spends. It also has to be cheap to *exceed*: AD-20 requires the corpus
to carry a chain nested past this number, so N4 has data on what the ceiling costs. Base64 expands
by `4/3` per level, so a chain one level past this ceiling costs a 100-character payload about 330
characters — `tests/canon/test_recursion.py` measures that rather than taking the arithmetic on
trust. AD-20 asserts the relation against this constant rather than hard-coding a depth, so
retuning the ceiling does not silently invalidate the corpus.

**What it costs, stated rather than discovered.** A payload nested four deep is published as a
ceiling hit instead of being recovered. That is the measurement, not a failure: `ceiling_hit`
answers "would a higher ceiling have recovered more of this document", and a reader who thinks
three is the wrong number can see the consequence in the table rather than infer it.
"""


class StageContractViolated(NbcError, exit_code=13):
    """A stage's returned edits do not account for the text it returned.

    An abort rather than a warning, and rated with the pin mismatches rather than with the
    programming errors: the trace is the evidence for every per-transformation attribution the
    results document makes. A layer whose trace disagrees with its own output produces recall
    numbers that are still correct and attributions that are not, and nothing downstream can tell.
    """


class Stage(Protocol):
    """The one shape every stage has. A stage is pure: it returns its edits and writes nowhere."""

    def __call__(self, text: str, ctx: CanonContext, /) -> StageResult: ...


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One step: the names that may appear in the trace, bound to the functions that do the work.

    The names are not spelled here. They are read from the stage module that emits them, so the
    trace and the pipeline cannot drift apart — and the runner still compares them on every
    document, because a stage stamping some other step's name is a misattribution no type would
    catch.

    `at_ceiling` is what makes a step **recursive**, structurally rather than by name. A step that
    declares it behaves differently once `depth >= ctx.ceiling`: the runner calls `at_ceiling`
    instead of `run` there, and below the ceiling it canonicalizes whatever `run` accepted as an
    independent document one level deeper. `ceiling_name` is the extra stage name that entry point
    is allowed to stamp. The two are declared together or not at all — a step naming a ceiling
    refusal it has no way to produce, or producing one it never declared, is a contradiction the
    constant should not be able to express.
    """

    name: str
    run: Stage
    ceiling_name: str | None = None
    at_ceiling: Stage | None = None

    def __post_init__(self) -> None:
        if (self.ceiling_name is None) != (self.at_ceiling is None):
            raise ValueError(
                f"step {self.name!r} declares ceiling_name={self.ceiling_name!r} and "
                f"at_ceiling={self.at_ceiling!r}; a step is recursive with both or with neither"
            )
        if self.ceiling_name == self.name:
            raise ValueError(
                f"step {self.name!r} names its ceiling refusal {self.ceiling_name!r}, the same "
                f"name it stamps on an ordinary decision; AD-6 asks for a distinct one"
            )

    @property
    def emits(self) -> frozenset[str]:
        """Every stage name this step is allowed to stamp on an edit."""
        if self.ceiling_name is None:
            return frozenset({self.name})
        return frozenset({self.name, self.ceiling_name})


PIPELINE: Final[tuple[PipelineStage, ...]] = (
    PipelineStage(invisible.NAME, invisible.run),
    PipelineStage(confusables.NAME, confusables.run),
    PipelineStage(nfkc.NAME, nfkc.run),
    PipelineStage(
        decode.NAME,
        decode.run,
        ceiling_name=decode.CEILING_NAME,
        at_ceiling=decode.run_at_ceiling,
    ),
)
"""The canonical order, and the only place it exists."""


def trace_stage_labels() -> tuple[str, ...]:
    """Every label an edit in a trace can carry, derived from `PIPELINE` and not spelled twice.

    **Wider than `schema.PIPELINE_STAGES`, and the difference is the point.** A stage that runs at
    the ceiling decides exactly as it would below it and then replaces nothing, reporting the
    candidate it WOULD have decoded under its `ceiling_name`. That is not a fifth stage: it is a
    reason on a decode-stage report, and it is the fact `ceiling_hit` is derived from, which already
    has its own census. So the census axis is the four stage names and the trace vocabulary is those
    four plus whichever stages declare a ceiling entry point -- two sets, two questions.

    Read off `PIPELINE` rather than listed, because listing it is what broke. The first real run
    aborted here: `aggregate.read_traces` validated a trace against `PIPELINE_STAGES` and refused
    `decode-ceiling` as "a stage nobody ran", on a corpus that carries a chain nested past the
    ceiling BY REQUIREMENT (AD-20) -- so the abort was certain the first time anything real was
    measured. The agreement test that was meant to make the second spelling safe compared
    `stage.name` and never `stage.ceiling_name`: it agreed with half the declaration and passed.

    Generated from the stages that actually declare one, not from all four, so a label no stage can
    stamp is still refused. `invisible-ceiling` is not a name this returns.
    """
    labels = [stage.name for stage in PIPELINE]
    labels.extend(stage.ceiling_name for stage in PIPELINE if stage.ceiling_name)
    return tuple(labels)


def default_context(
    *,
    ceiling: int = DEFAULT_CEILING,
    trace_enabled: bool = True,
    data_dir: Path = confusables_table.DATA_DIR,
) -> CanonContext:
    """Build the context an entrypoint hands to every pass: the table loaded exactly once.

    The loader validates on every call and holds no state, so this is where "once" is decided.
    Aborts with `ConfusablesTableInvalid` if the vendored artifact is absent, ambiguous, or at a
    Unicode revision the interpreter does not share.

    This is also the **only** place `DEFAULT_CEILING` is read. AD-6 requires the ceiling to be
    overridable only through a context the entrypoint constructs once, and a second function
    applying the same default would be a second place a run parameter could be set.
    """
    table = confusables_table.load(data_dir)
    return CanonContext(
        confusables=table.translate_table, ceiling=ceiling, trace_enabled=trace_enabled
    )


def _verify(step: PipelineStage, text_in: str, result: StageResult) -> None:
    """Read the stage's edits back against the text it was handed, and against what it returned."""
    if not result.edits:
        # No edits means the stage claims it changed nothing. Compared directly rather than
        # through the replay below, which would otherwise copy the whole document to say so.
        if result.text != text_in:
            raise StageContractViolated(
                f"stage {step.name!r} returned changed text and no edits; a change with no entry "
                f"in the trace is a change nothing can attribute"
            )
        return

    pieces: list[str] = []
    position = 0

    for edit in result.edits:
        if edit.stage not in step.emits:
            raise StageContractViolated(
                f"stage {step.name!r} returned an edit stamped {edit.stage!r}, which is not one "
                f"of the names it declares ({sorted(step.emits)}); a change attributed to the "
                f"wrong stage is worse than an unattributed one"
            )
        start, end = edit.span
        if start < position:
            raise StageContractViolated(
                f"stage {step.name!r} returned a span starting at {start} after a span that "
                f"already reached {position}; edits are ordered and never overlap"
            )
        if end > len(text_in):
            raise StageContractViolated(
                f"stage {step.name!r} returned a span ending at {end}, past the {len(text_in)} "
                f"code points it was handed"
            )
        if text_in[start:end] != edit.before:
            raise StageContractViolated(
                f"stage {step.name!r} recorded {edit.before!r} at {edit.span}, where the text it "
                f"was handed holds {text_in[start:end]!r}"
            )
        pieces.append(text_in[position:start])
        pieces.append(edit.after)
        position = end

    pieces.append(text_in[position:])
    replayed = "".join(pieces)
    if replayed != result.text:
        raise StageContractViolated(
            f"replaying stage {step.name!r}'s {len(result.edits)} edit(s) over its input produces "
            f"text that is not the text it returned; the trace does not account for the change"
        )


def _stage_result(step: PipelineStage, run: Stage, text: str, ctx: CanonContext) -> StageResult:
    """Run one stage entry point and check what it returned against the text it was handed."""
    result = run(text, ctx)
    if not isinstance(result, StageResult):
        raise StageContractViolated(
            f"stage {step.name!r} returned a {type(result).__name__}, not a StageResult"
        )
    if ctx.trace_enabled or step.at_ceiling is not None:
        # A recursive step reports whether or not the trace survives, so it is verified whether or
        # not the trace survives. Skipping the check with tracing off would leave the timing pass
        # splicing text on the strength of edits nothing had read back.
        _verify(step, text, result)
    elif result.edits:
        raise StageContractViolated(
            f"stage {step.name!r} returned edits with tracing off; the timing pass would then "
            f"be measuring a different layer than the measurement pass"
        )
    return result


def canonicalize(text: str, ctx: CanonContext, *, depth: int = 0) -> CanonResult:
    """Run every step of `PIPELINE` in order and return the canonical text with its trace.

    The trace is the concatenation of the stages' edits, in stage order, each stamped with `depth`.
    Every stage's result is verified against the text it was handed before the next stage sees it,
    so a broken attribution stops at the stage that produced it.

    **Depth is per-branch.** A document handed in at `depth` is decoded only while
    `depth < ctx.ceiling`; at or above the ceiling the recursive step runs through its `at_ceiling`
    entry point, which replaces nothing and reports a would-have-decoded candidate under a distinct
    name. Below it, every accepted span is canonicalized as an independent document at `depth + 1`,
    **unconditionally** — including when it holds no further candidate, which is why a decode of
    plain ASCII still reports `max_depth_reached == 1` — and the resulting text replaces the source
    span. The host document is not re-scanned. Siblings are unbounded.

    Accepted or refused is read from AD-18's own marker: a refusal is an edit whose `before` equals
    its `after`. That is not an inference about text. Story 2.3 proved a decode strictly contracts,
    so an accepted decode cannot return the run it was given, and `tests/canon/test_recursion.py`
    recomputes the accepted set with `decode.decide` and requires the recursion to have entered
    exactly those spans.

    With `ctx.trace_enabled` off the character stages return no edits, and the recursive step's
    edits are dropped here instead of at the stage — they are the recursion's input, not trace
    bookkeeping. `text`, `ceiling_hit` and `max_depth_reached` are identical in both passes, which
    is asserted from the outside over a battery rather than assumed here.
    """
    if not isinstance(text, str):
        raise TypeError(f"canonicalize takes text, got {type(text).__name__}")

    edits: list[Edit] = []
    ceiling_hit = False
    max_depth_reached = depth

    def keep(edit: Edit) -> Edit:
        return edit if edit.depth == depth else dataclasses.replace(edit, depth=depth)

    for step in PIPELINE:
        if step.at_ceiling is None:
            result = _stage_result(step, step.run, text, ctx)
            edits.extend(keep(edit) for edit in result.edits)
            text = result.text
            continue

        if depth >= ctx.ceiling:
            result = _stage_result(step, step.at_ceiling, text, ctx)
            if any(edit.stage == step.ceiling_name for edit in result.edits):
                ceiling_hit = True
            if ctx.trace_enabled:
                edits.extend(keep(edit) for edit in result.edits)
            text = result.text
            continue

        result = _stage_result(step, step.run, text, ctx)
        pieces: list[str] = []
        position = 0
        for edit in result.edits:
            start, end = edit.span
            pieces.append(text[position:start])
            position = end
            if ctx.trace_enabled:
                edits.append(keep(edit))
            if edit.before == edit.after:
                pieces.append(edit.after)
                continue
            sub = canonicalize(edit.after, ctx, depth=depth + 1)
            pieces.append(sub.text)
            if ctx.trace_enabled:
                edits.extend(sub.edits)
            ceiling_hit = ceiling_hit or sub.ceiling_hit
            max_depth_reached = max(max_depth_reached, sub.max_depth_reached)
        pieces.append(text[position:])
        text = "".join(pieces)

    return CanonResult(
        text=text,
        edits=tuple(edits),
        ceiling_hit=ceiling_hit,
        max_depth_reached=max_depth_reached,
    )
