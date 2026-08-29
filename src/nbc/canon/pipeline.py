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
depth it was called at onto every edit it collects. Nothing calls it with a depth other than `0`
yet; Story 2.4, which canonicalizes a decoded segment as an independent document at `depth + 1`, is
what will.
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

__all__ = [
    "PIPELINE",
    "PipelineStage",
    "Stage",
    "StageContractViolated",
    "canonicalize",
    "default_context",
]


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
    """One step: the name that appears in the trace, bound to the function that does the work.

    The name is not spelled here. It is read from the stage module that emits it, so the trace and
    the pipeline cannot drift apart — and the runner still compares the two on every document,
    because a stage stamping some other step's name is a misattribution no type would catch.
    """

    name: str
    run: Stage


PIPELINE: Final[tuple[PipelineStage, ...]] = (
    PipelineStage(invisible.NAME, invisible.run),
    PipelineStage(confusables.NAME, confusables.run),
    PipelineStage(nfkc.NAME, nfkc.run),
    PipelineStage(decode.NAME, decode.run),
)
"""The canonical order, and the only place it exists."""


def default_context(
    *, trace_enabled: bool = True, data_dir: Path = confusables_table.DATA_DIR
) -> CanonContext:
    """Build the context an entrypoint hands to every pass: the table loaded exactly once.

    The loader validates on every call and holds no state, so this is where "once" is decided.
    Aborts with `ConfusablesTableInvalid` if the vendored artifact is absent, ambiguous, or at a
    Unicode revision the interpreter does not share.
    """
    table = confusables_table.load(data_dir)
    return CanonContext(confusables=table.translate_table, trace_enabled=trace_enabled)


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
        if edit.stage != step.name:
            raise StageContractViolated(
                f"stage {step.name!r} returned an edit stamped {edit.stage!r}; a change "
                f"attributed to the wrong stage is worse than an unattributed one"
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


def canonicalize(text: str, ctx: CanonContext, *, depth: int = 0) -> CanonResult:
    """Run every step of `PIPELINE` in order and return the canonical text with its trace.

    The trace is the concatenation of the stages' edits, in stage order, each stamped with `depth`.
    Every stage's result is verified against the text it was handed before the next stage sees it,
    so a broken attribution stops at the stage that produced it.

    With `ctx.trace_enabled` off, stages return no edits and there is nothing to verify; the text
    is identical either way, which is asserted from the outside rather than assumed here.
    """
    if not isinstance(text, str):
        raise TypeError(f"canonicalize takes text, got {type(text).__name__}")

    edits: list[Edit] = []
    for step in PIPELINE:
        result = step.run(text, ctx)
        if not isinstance(result, StageResult):
            raise StageContractViolated(
                f"stage {step.name!r} returned a {type(result).__name__}, not a StageResult"
            )
        if ctx.trace_enabled:
            _verify(step, text, result)
        elif result.edits:
            raise StageContractViolated(
                f"stage {step.name!r} returned edits with tracing off; the timing pass would then "
                f"be measuring a different layer than the measurement pass"
            )
        edits.extend(
            edit if edit.depth == depth else dataclasses.replace(edit, depth=depth)
            for edit in result.edits
        )
        text = result.text

    return CanonResult(text=text, edits=tuple(edits))
