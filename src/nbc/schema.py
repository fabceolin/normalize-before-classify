"""Every record type in this project, defined here and nowhere else.

This module is a **leaf**: it imports the standard library and nothing from `nbc`. That is
what lets `canon/` depend on it without breaking the canonicalization layer's isolation, and
it is enforced by a test that reads this file's source rather than by convention.

Two consequences of the leaf rule that look like omissions and are not:

- Validation here raises `ValueError`, not `nbc.errors.NbcError`. It cannot raise `NbcError`
  without importing it. That is the right outcome anyway: a malformed record is a
  programming error inside one process, not one of the aborts that change the meaning of a
  published number.
- No module adds a field to these types from outside. A field that six modules write into
  and no module declares is exactly the drift this module exists to prevent.

Types are added here **by the story that first needs them**, not all at once. The rule
governs *where* a record type lives, not *when* it appears: enumerating fields before
anything consumes them freezes names against requirements no one has met yet. `Score` is
here because the model boundary returns it, and the canonicalization layer's four shapes are
here because Story 2.2 is the first thing to produce one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

__all__ = ["CanonContext", "CanonResult", "Edit", "Score", "StageResult"]


@dataclass(frozen=True, slots=True)
class Score:
    """One document's score from one baseline, under one condition.

    `p_injection` is `softmax(logits)[positive_index]` over the model's full label axis,
    computed in one shared helper so an adapter's only freedom is producing the logits.

    `n_windows` rides on the score rather than on hidden module state: a document longer
    than one window is split into fixed non-overlapping windows, each scored
    independently, and the document takes the maximum. How many windows a document needed
    is part of what the run reports, so it travels with the number it produced.
    """

    p_injection: float
    n_windows: int

    def __post_init__(self) -> None:
        p = self.p_injection
        if isinstance(p, bool) or not isinstance(p, (float, int)):
            raise ValueError(f"p_injection must be a real number, got {p!r}")
        if not 0.0 <= p <= 1.0:
            # Also rejects NaN, which fails every comparison, and both infinities.
            raise ValueError(f"p_injection must lie in [0, 1], got {p!r}")
        if not isinstance(p, float):
            # Stored as a float so the serialized form does not depend on how the caller
            # happened to spell 0 or 1.
            object.__setattr__(self, "p_injection", float(p))

        n = self.n_windows
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"n_windows must be an int, got {n!r}")
        if n < 1:
            raise ValueError(
                f"n_windows must be at least 1: a scored document occupies at least one "
                f"window, got {n!r}"
            )


# --- the canonicalization layer's shapes -----------------------------------------------------

MAX_ASCII: Final[int] = 0x7F
"""The last code point the canonicalization layer promises to leave alone.

Spelled here as well as in `nbc.canon.confusables_table` because this module is a leaf and
cannot import that one. A test asserts the two are the same number, so the duplication is a
comparison rather than a second opinion.
"""

MAX_CODE_POINT: Final[int] = 0x10FFFF


@dataclass(frozen=True, slots=True)
class Edit:
    """One contiguous span that one stage changed, at one recursion depth.

    Granularity is declared rather than left to the implementation: **one `Edit` per contiguous
    changed span**, `before` and `after` holding only that span and never the whole document, and
    adjacent single-character changes coalescing into one span. Two compliant layers that disagreed
    about this would publish layer-cost numbers an order of magnitude apart.

    `span` is `(start, end)` into **the text handed to that stage at that depth** — not into the
    original document. Remapping onto the original offsets is the reader's job; the layer would have
    to carry a full offset history to do it, and no consumer asks for one.

    `before == after` is legal and load-bearing: a decode candidate that was examined and refused is
    recorded as a no-op edit, so a rejection is visible in the trace rather than invisible. Stories
    2.3 and 2.4 are the ones that emit those.
    """

    stage: str
    span: tuple[int, int]
    before: str
    after: str
    depth: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError(f"stage must be a non-empty name, got {self.stage!r}")

        span = self.span
        if isinstance(span, list):
            # A trace read back from JSONL arrives with lists; the record is a tuple.
            span = tuple(span)
            object.__setattr__(self, "span", span)
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError(f"span must be a (start, end) pair, got {self.span!r}")
        start, end = span
        for label, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"span {label} must be an int, got {value!r}")
        if start < 0:
            raise ValueError(f"span start must not be negative, got {start!r}")
        if end < start:
            raise ValueError(f"span end {end!r} is before its start {start!r}")

        for label, value in (("before", self.before), ("after", self.after)):
            if not isinstance(value, str):
                raise ValueError(f"{label} must be a str, got {value!r}")

        if len(self.before) != end - start:
            # The span is the evidence for `before`, so it is compared to it. A span that does
            # not measure its own text would let a reader locate the change in the wrong place.
            raise ValueError(
                f"span {span} covers {end - start} code points but before holds "
                f"{len(self.before)}"
            )

        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise ValueError(f"depth must be an int, got {self.depth!r}")
        if self.depth < 0:
            raise ValueError(f"depth must not be negative, got {self.depth!r}")


def _as_edit_tuple(edits: object, field: str) -> tuple[Edit, ...]:
    if isinstance(edits, Edit) or isinstance(edits, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of Edit, got {edits!r}")
    if not isinstance(edits, Sequence):
        raise ValueError(f"{field} must be a sequence of Edit, got {edits!r}")
    frozen = tuple(edits)
    for edit in frozen:
        if not isinstance(edit, Edit):
            raise ValueError(f"{field} must hold only Edit, got {edit!r}")
    return frozen


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one stage returns: the text it produced and the edits that account for it.

    A stage writes nowhere else. Everything it has to say about what it did is in this record,
    which is what lets the runner replay the edits over the stage's input and compare the result
    to `text` — the trace is checked against the transformation instead of being trusted beside it.
    """

    text: str
    edits: tuple[Edit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(f"text must be a str, got {self.text!r}")
        object.__setattr__(self, "edits", _as_edit_tuple(self.edits, "edits"))


@dataclass(frozen=True, slots=True)
class CanonResult:
    """What the pipeline returns for one document: its canonical text and the whole trace.

    Two fields, not four. `ceiling_hit` and `max_depth_reached` belong to the recursion contract,
    which Story 2.4 writes; nothing in the layer computes them yet, and a field shipped as a
    constant would be describing behaviour nobody implemented.
    """

    text: str
    edits: tuple[Edit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(f"text must be a str, got {self.text!r}")
        object.__setattr__(self, "edits", _as_edit_tuple(self.edits, "edits"))


@dataclass(frozen=True, slots=True)
class CanonContext:
    """Everything a stage is allowed to depend on besides its input text.

    `confusables` is the vendored table, already loaded, as code point to ASCII string. It rides
    on the context and not in the stage because the two alternatives are both barred: the loader
    reads and validates on every call by design, so calling it per document would put disk I/O
    inside the path whose p50 and p95 the run publishes, and caching it in module state is
    forbidden outright — no module-level mutable state anywhere in `canon/`. The entrypoint builds
    this once, which is where a table should be loaded once.

    `trace_enabled` is real, not decoration: with it off the stages skip building `Edit` records
    entirely. The text they produce is identical either way, and a test asserts that over a
    battery of inputs, because a trace flag that could change the canonical text would make the
    timing pass measure a different layer than the measurement pass.

    `ceiling` is **not** here. It is Story 2.4's, nothing in this story reads it, and a field no
    code consumes is the shape that let a published obligation go undischarged through all of
    Epic 1.
    """

    confusables: Mapping[int, str]
    trace_enabled: bool = True

    def __post_init__(self) -> None:
        table = self.confusables
        if not isinstance(table, Mapping):
            raise ValueError(f"confusables must be a mapping, got {table!r}")
        for key, value in table.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise ValueError(f"confusables keys are code points; got {key!r}")
            if not 0 <= key <= MAX_CODE_POINT:
                raise ValueError(f"confusables key {key!r} is not a code point")
            if key <= MAX_ASCII:
                # The layer's promise is the identity on U+0000..U+007F. Enforcing it only in
                # the artifact loader would leave it unenforced for any context built by hand,
                # and folding `1` to `l` across source code is the specific damage.
                raise ValueError(
                    f"confusables maps the ASCII code point U+{key:04X}; the layer is the "
                    f"identity on U+0000..U+{MAX_ASCII:04X}"
                )
            if not isinstance(value, str) or not value:
                raise ValueError(f"confusables maps U+{key:04X} to {value!r}, not a non-empty str")
        object.__setattr__(self, "confusables", MappingProxyType(dict(table)))

        if not isinstance(self.trace_enabled, bool):
            raise ValueError(f"trace_enabled must be a bool, got {self.trace_enabled!r}")
