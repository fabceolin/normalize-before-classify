"""Step 3: apply NFKC, and attribute the change to spans without ever changing what NFKC said.

The output is `unicodedata.normalize("NFKC", text)`. Nothing here is allowed to produce anything
else, and a test asserts that over a battery of inputs, because the whole reason the confusables
table is pinned to `unicodedata.unidata_version` is that step 2 and step 3 must agree about the
same character. A hand-rolled normalization would break that pin from the other side.

Attribution is the hard half. NFKC is defined over the whole string — `"e" + U+0301` becomes one
code point, `"ﬁ"` becomes two — so "which span changed" is not something the standard library
answers. A general diff would answer it at quadratic cost, on documents that include whole source
files.

So: split the input at combining-class-0 boundaries, normalize each segment on its own, and accept
that per-segment alignment **only if the segments join back to the authoritative output**. When
they do, every changed segment is a located change and adjacent ones coalesce. When they do not,
the whole changed region becomes one edit, prefix- and suffix-trimmed.

The fallback is reachable and is exercised: conjoining Hangul jamo compose across the split, since
`U+1100` and `U+1161` are both starters, so `"가"` takes it. Both paths emit exactly the
same text; only the resolution of the trace differs, and the trace says which stage it came from
either way.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from typing import Final

from nbc.canon.edits import Segment, build_edits
from nbc.schema import MAX_ASCII, CanonContext, StageResult

__all__ = ("NAME", "run")

NAME: Final[str] = "nfkc"

_FORM: Final[str] = "NFKC"


def _cover(text: str) -> Iterator[Segment]:
    """Cover `text` with segments cut at combining-class-0 boundaries, each normalized alone."""
    start = 0
    for index in range(1, len(text) + 1):
        if index < len(text) and unicodedata.combining(text[index]) != 0:
            continue
        if index == start:
            continue
        chunk = text[start:index]
        if len(chunk) == 1 and ord(chunk) <= MAX_ASCII:
            # Every ASCII code point is its own NFKC form; asserted over the whole range by a
            # test, so this shortcut is a checked fact rather than a belief about ASCII.
            yield (start, index, chunk)
        else:
            yield (start, index, unicodedata.normalize(_FORM, chunk))
        start = index


def _whole_region(text: str, normalized: str) -> Iterator[Segment]:
    """Cover `text` with one changed span: the region left after trimming what both agree on."""
    limit = min(len(text), len(normalized))
    prefix = 0
    while prefix < limit and text[prefix] == normalized[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and text[-1 - suffix] == normalized[-1 - suffix]:
        suffix += 1

    end = len(text) - suffix
    if prefix:
        yield (0, prefix, text[:prefix])
    yield (prefix, end, normalized[prefix : len(normalized) - suffix])
    if suffix:
        yield (end, len(text), text[end:])


def run(text: str, ctx: CanonContext) -> StageResult:
    """Normalize to NFKC, reporting one edit per contiguous changed span where it can locate them."""
    normalized = unicodedata.normalize(_FORM, text)
    if normalized == text:
        return StageResult(text=text, edits=())

    segments = list(_cover(text))
    if "".join(replacement for _, _, replacement in segments) != normalized:
        segments = list(_whole_region(text, normalized))

    new_text, edits = build_edits(text, segments, stage=NAME, trace=ctx.trace_enabled)
    # No guard here that `new_text == normalized`: both covers are constructed to produce it, so
    # no input could make such a guard fire, and a branch nobody can reach is not a check. The
    # equality is asserted from the outside instead, over a battery of inputs, in
    # `tests/canon/test_nfkc.py`.
    return StageResult(text=new_text, edits=edits)
