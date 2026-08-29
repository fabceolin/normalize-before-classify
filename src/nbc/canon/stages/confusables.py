"""Step 2: map Unicode confusables to their Latin form, one code point at a time.

This step **applies** the vendored table; it does not own it, load it, or cache it. The table
arrives on the context, already validated by `nbc.canon.confusables_table.load()`, because the
loader reads and validates on every call by design and a stage that called it per document would
put disk I/O inside the path whose p50 and p95 the run publishes.

**Why after step 1 and before step 3.** After step 1 because a zero-width character between two
homoglyphs is noise the mapping should not have to reason about. Before step 3 because NFKC is the
interpreter's own table at the same Unicode revision this one was vendored at: doing the mapping
second and NFKC third means every code point is seen by both, in a fixed order, at one revision.

The mapping is not the identity on any ASCII code point — that is enforced twice, by the artifact
loader and by `CanonContext`, because folding `1` to `l` and `0` to `O` across ordinary source code
would turn the benign-code counter-metric into a number about ASCII folding.
"""

from __future__ import annotations

from typing import Final

from nbc.canon.edits import build_edits, map_code_points
from nbc.schema import CanonContext, StageResult

__all__ = ("NAME", "run")

NAME: Final[str] = "confusables"


def run(text: str, ctx: CanonContext) -> StageResult:
    """Replace each confusable code point with its ASCII form; adjacent ones become one span."""
    table = ctx.confusables
    new_text, edits = build_edits(
        text,
        map_code_points(text, lambda char: table.get(ord(char))),
        stage=NAME,
        trace=ctx.trace_enabled,
    )
    return StageResult(text=new_text, edits=edits)
