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
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from nbc.schema import Edit

__all__ = ["Segment", "build_edits", "map_code_points"]

Segment = tuple[int, int, str]
"""`(start, end, replacement)`: the half-open span `text[start:end]` becomes `replacement`."""


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
