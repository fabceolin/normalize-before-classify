"""The one implementation of the layer's edit-granularity rule.

AD-5 declares the granularity rather than leaving it to each stage: **one `Edit` per contiguous
changed span**, `before` and `after` holding only that span, and adjacent changed characters
coalescing into one span. Three stages implementing that rule three times would produce three
granularities, and the published layer-cost number is sensitive to exactly that.

A stage therefore does not build `Edit` records. It produces a **cover** of its input — an ordered
sequence of `(start, end, replacement)` segments that partitions the text with no gap and no
overlap — and this module turns the cover into the output text and the coalesced edits. The cover
is validated on the way through, so a stage that produces a broken partition is caught here rather
than emitting a trace that quietly locates a change in the wrong place.

Step 4 reports one thing the character stages never do: a span it **examined and left alone**. AD-18
requires a refused decode candidate to be visible in the trace as an `Edit` whose `before` equals
its `after`, and `build_edits` is built to drop exactly that segment. `build_reported_edits` is the
second entry point for that case — the caller names the spans it is reporting **and the stage name
each decision belongs to**, each becomes exactly one edit changed or not, and the unexamined text
between them is copied through unreported.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from nbc.schema import Edit

__all__ = ("Report", "Segment", "build_edits", "build_reported_edits", "map_code_points")

Segment = tuple[int, int, str]
"""`(start, end, replacement)`: the half-open span `text[start:end]` becomes `replacement`."""

Report = tuple[int, int, str, str]
"""`(start, end, replacement, stage)`: a reported span, and the stage name that decided it.

The stage travels **per span** rather than per call because one pass over a document can carry two
kinds of decision that must not be confused: a candidate the decode stage refused on its own merits
and a candidate it would have decoded but for the recursion ceiling. AD-6 requires the second to be
distinguishable in the trace from the first, and both can occur in the same document at the same
depth, interleaved in document order.
"""


def map_code_points(text: str, replace: Callable[[str], str | None]) -> Iterable[Segment]:
    """Cover `text` by applying `replace` per code point, coalescing the untouched runs.

    `replace` returns `None` for a code point it leaves alone and the replacement string
    otherwise — including `""` for a removal. Untouched code points are emitted as runs rather
    than one segment each, so a document the stage does not change costs one segment, not one
    per character.
    """
    start = 0
    for index, char in enumerate(text):
        replacement = replace(char)
        if replacement is None:
            continue
        if index > start:
            yield (start, index, text[start:index])
        yield (index, index + 1, replacement)
        start = index + 1
    if start < len(text):
        yield (start, len(text), text[start:])


def build_edits(
    text: str,
    segments: Iterable[Segment],
    *,
    stage: str,
    trace: bool,
) -> tuple[str, tuple[Edit, ...]]:
    """Apply a cover of `text` and return the new text with one edit per changed span.

    Raises `ValueError` if `segments` is not an ordered partition of `text`: a gap, an overlap, a
    segment out of order, or a cover that stops short of the end. That is a programming error in a
    stage, not a condition of the data, which is why it is a `ValueError` and not an abort.
    """
    pieces: list[str] = []
    edits: list[Edit] = []

    position = 0
    # The changed run currently open, as (start, before-pieces, after-pieces).
    run_start = -1
    run_before: list[str] = []
    run_after: list[str] = []

    def close_run() -> None:
        nonlocal run_start
        if run_start < 0:
            return
        if trace:
            before = "".join(run_before)
            edits.append(
                Edit(
                    stage=stage,
                    span=(run_start, run_start + len(before)),
                    before=before,
                    after="".join(run_after),
                )
            )
        run_start = -1
        run_before.clear()
        run_after.clear()

    for start, end, replacement in segments:
        if start != position:
            raise ValueError(
                f"stage {stage!r} produced a segment starting at {start} where the cover of the "
                f"text had reached {position}; the segments must partition the text in order"
            )
        if end < start:
            raise ValueError(f"stage {stage!r} produced the reversed segment ({start}, {end})")
        if end > len(text):
            raise ValueError(
                f"stage {stage!r} produced a segment ending at {end}, past the {len(text)} code "
                f"points it was handed"
            )

        original = text[start:end]
        pieces.append(replacement)
        if not trace:
            # The timing pass runs with the trace off, so it does not pay for bookkeeping it
            # will throw away. The partition check below still runs: it guards the output text,
            # not the trace.
            pass
        elif replacement == original:
            close_run()
        else:
            if run_start < 0:
                run_start = start
            run_before.append(original)
            run_after.append(replacement)
        position = end

    close_run()

    if position != len(text):
        raise ValueError(
            f"stage {stage!r} covered {position} of {len(text)} code points; the segments must "
            f"partition the whole text"
        )

    return "".join(pieces), tuple(edits)


def build_reported_edits(
    text: str,
    reports: Iterable[Report],
) -> tuple[str, tuple[Edit, ...]]:
    """Apply an ordered, non-overlapping set of **reported** spans and emit one Edit per span.

    The difference from `build_edits` is the whole reason this exists: `build_edits` drops a
    segment whose replacement equals the original, because for a character stage an unchanged
    character is not news. Step 4 has news to report about a span it decided **not** to change —
    AD-18 requires a refused decode candidate to appear in the trace as an `Edit` whose `before`
    equals its `after` — so the caller states which spans are reported and every one of them
    becomes exactly one edit, under the stage name that span's decision belongs to.

    `reports` need not partition the text: the gaps between them are the parts the stage never
    examined, and they are copied through untouched and unreported. What `reports` must be is
    ordered, non-overlapping and inside the text, which is what the runner will independently
    demand of the edits that come out; a stage that breaks it raises `ValueError` here, at the
    place that can name what it produced, rather than as a contract violation two frames later.

    **There is no `trace` switch here**, unlike `build_edits`. Step 4's records are how the runner
    learns which spans were accepted, and it has to recurse into exactly those spans in the timing
    pass as well or the canonical text would differ between the two passes. So these edits are
    always built, and whether they reach the document's trace is the runner's decision, not this
    function's.

    No coalescing. Two reported spans are never adjacent in practice — a decode candidate is a
    *maximal* run over its alphabet, so any two are separated by at least one character that is
    not in it — and merging two independent decisions into one edit would report two candidates
    as one change.
    """
    pieces: list[str] = []
    edits: list[Edit] = []
    position = 0

    for start, end, replacement, stage in reports:
        if start < position:
            raise ValueError(
                f"stage {stage!r} reported a span starting at {start} after a span that already "
                f"reached {position}; reported spans are ordered and never overlap"
            )
        if end < start:
            raise ValueError(f"stage {stage!r} reported the reversed span ({start}, {end})")
        if end > len(text):
            raise ValueError(
                f"stage {stage!r} reported a span ending at {end}, past the {len(text)} code "
                f"points it was handed"
            )

        pieces.append(text[position:start])
        pieces.append(replacement)
        edits.append(
            Edit(stage=stage, span=(start, end), before=text[start:end], after=replacement)
        )
        position = end

    pieces.append(text[position:])
    return "".join(pieces), tuple(edits)
