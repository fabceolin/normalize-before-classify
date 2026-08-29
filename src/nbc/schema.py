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
    """What the pipeline returns for one document: its canonical text, its trace, and its depth.

    `max_depth_reached` is the greatest depth at which a document was canonicalized during this
    run: `0` for a document with no accepted decode, and `d + 1` for a segment accepted at depth
    `d`. Only **accepted** decodes raise it — a candidate refused on its own merits and a candidate
    refused because the ceiling was reached both leave it where it was.

    `ceiling_hit` is true if and only if at least one candidate would have been decoded and was
    refused **solely** because the recursion ceiling had been reached. A candidate the decode
    stage would have refused anyway is not a ceiling hit, and the difference is the whole content
    of the flag: `ceiling_hit` answers "would a higher ceiling have recovered more of this
    document", which is a published outcome rather than a warning.

    The two are compared here rather than merely carried together: an edit stamped at depth `k`
    is evidence that a document was canonicalized at depth `k`, so `max_depth_reached` cannot be
    smaller than the deepest edit in the trace. The check is one-directional on purpose. With
    tracing off the trace is empty and bounds nothing, and a sub-document that needed no change
    produces no edit at all, so equality would be wrong in both directions.
    """

    text: str
    edits: tuple[Edit, ...] = ()
    ceiling_hit: bool = False
    max_depth_reached: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(f"text must be a str, got {self.text!r}")
        object.__setattr__(self, "edits", _as_edit_tuple(self.edits, "edits"))

        if not isinstance(self.ceiling_hit, bool):
            raise ValueError(f"ceiling_hit must be a bool, got {self.ceiling_hit!r}")

        depth = self.max_depth_reached
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"max_depth_reached must be an int, got {depth!r}")
        if depth < 0:
            raise ValueError(f"max_depth_reached must not be negative, got {depth!r}")

        deepest = max((edit.depth for edit in self.edits), default=0)
        if depth < deepest:
            raise ValueError(
                f"max_depth_reached is {depth} but the trace holds an edit at depth {deepest}; "
                f"an edit at a depth is evidence a document was canonicalized there"
            )


@dataclass(frozen=True, slots=True)
class CanonContext:
    """Everything a stage is allowed to depend on besides its input text.

    `confusables` is the vendored table, already loaded, as code point to ASCII string. It rides
    on the context and not in the stage because the two alternatives are both barred: the loader
    reads and validates on every call by design, so calling it per document would put disk I/O
    inside the path whose p50 and p95 the run publishes, and caching it in module state is
    forbidden outright — no module-level mutable state anywhere in `canon/`. The entrypoint builds
    this once, which is where a table should be loaded once.

    `ceiling` is the per-branch recursion ceiling: decoding is attempted only while
    `depth < ceiling`, so `0` decodes nothing at all and every candidate that would have decoded
    is reported as a ceiling hit. It has **no default here**, and that is the requirement rather
    than an oversight — FR10 asks for an explicit parameter with a declared default and never an
    implicit one, so the default lives in exactly one place, `nbc.canon.pipeline.DEFAULT_CEILING`,
    applied by `default_context`. This module is a leaf and cannot import it, which is what makes
    the single home enforceable instead of merely intended.

    `trace_enabled` is real, not decoration: with it off the character stages skip building `Edit`
    records entirely. It does **not** reach step 4, whose records are the recursion's input rather
    than trace bookkeeping; the runner drops those from the trace instead. The text, the ceiling
    hit and the depth are identical either way, and a test asserts all three over a battery of
    inputs, because a trace flag that could change any of them would make the timing pass measure
    a different layer than the measurement pass.
    """

    confusables: Mapping[int, str]
    ceiling: int
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

        ceiling = self.ceiling
        if isinstance(ceiling, bool) or not isinstance(ceiling, int):
            # A bool is an int in Python and `ceiling=True` would silently mean "one level".
            raise ValueError(f"ceiling counts recursion levels and must be an int, got {ceiling!r}")
        if ceiling < 0:
            raise ValueError(
                f"ceiling must not be negative, got {ceiling!r}; zero already means "
                f"'decode nothing, report every candidate as a ceiling hit'"
            )

        if not isinstance(self.trace_enabled, bool):
            raise ValueError(f"trace_enabled must be a bool, got {self.trace_enabled!r}")
