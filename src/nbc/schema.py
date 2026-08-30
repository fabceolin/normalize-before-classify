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

__all__ = [
    "ATTACK",
    "AUC_STRUCTURAL",
    "AXIS_BASELINE",
    "AXIS_BENIGN_CLASS",
    "AXIS_CANON_ON",
    "AXIS_CHAIN_CLASS",
    "AXIS_DRESSING_CHAIN",
    "AXIS_FAMILY",
    "AXIS_WINDOW_POLICY",
    "Auc",
    "BENIGN",
    "BENIGN_CLASSES",
    "CANONICAL",
    "CELL_AXES",
    "CHAIN_CLASSES_FOR_KEYS",
    "CONDITIONS",
    "CONTRAST_ARGUMENT_REQUIRED",
    "CONTRAST_ATTACKS_VS_BENIGN_CLASS",
    "CONTRAST_BOUND_VS_HELD_OUT",
    "CONTRAST_CANON_ON_VS_OFF",
    "CONTRAST_CLEAN_VS_CHAIN",
    "CONTRAST_KINDS",
    "CanonContext",
    "CanonResult",
    "CellKey",
    "Contrast",
    "CorpusItem",
    "Count",
    "DELTA_AUC_STRUCTURAL",
    "Delta",
    "Edit",
    "FAMILIES",
    "FAMILY_ATTACK",
    "FAMILY_BENIGN",
    "INTERVAL_METHODS",
    "Interval",
    "ItemScore",
    "LABELS",
    "NEWCOMBE_PAIRED",
    "PERMITTED_SPANS",
    "PROTECTED_AXES",
    "PairedCount",
    "RAW",
    "REQUIRED_SPANS",
    "Rate",
    "Score",
    "StageResult",
    "WILSON_SCORE",
]


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


# --- the corpus ------------------------------------------------------------------------------

ATTACK: Final[int] = 1
"""The gold label of an attack item. Named here so no builder writes the integer.

This is the *corpus* vocabulary, and it is not the same thing as a pinned dataset's
`attack_label` even where the two integers coincide. The dataset's value decides which of its
rows are drawn; this one is the label the builder asserts by construction over what it rendered.
Reading the corpus label off the source row would make the gold label a copy of somebody else's
annotation, which is the one thing FR4 says this repository does not do -- and while the two
integers agree, nothing would show it.
"""

BENIGN: Final[int] = 0
"""The gold label of a benign item."""

LABELS: Final[tuple[int, ...]] = (BENIGN, ATTACK)
"""The only two gold labels. A third would make every rate in the project ill-defined."""

FAMILY_ATTACK: Final[str] = "attack"
FAMILY_BENIGN: Final[str] = "benign"

FAMILIES: Final[tuple[str, ...]] = (FAMILY_ATTACK, FAMILY_BENIGN)
"""The two halves of the table. Reported separately and never pooled."""

BENIGN_CLASSES: Final[tuple[str, ...]] = ("b_code", "b_chat")
"""The benign classes, named here because `ItemScore` carries the distinction rather than
inferring it: `label == 0` separates benign from attack, but nothing separates code from chat,
and that is the one distinction FR3.1 exists to protect. The corpus for these classes is drawn by
story 3.6; the vocabulary lives here because the field does.
"""


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One line of `data/*.jsonl`: the text as rendered, and the label as asserted.

    The two travel together by construction. `corpus/build.py` is the only writer, it produces
    both in one constructor call, and an AST scan refuses a `label=` argument that is anything
    other than one of the two constants above -- so a gold label can never be a value read off a
    source row.

    `benign_class` is always present as a key and is `None` for an attack. Absent-versus-null is
    the difference between "this item has no benign class" and "somebody forgot to write one",
    and a reader of the file cannot tell those apart if the key comes and goes.

    `dressing` is the ordered chain the text was rendered through, empty for the clean chain.
    The fold that applies it and the chain registry are story 3.3's; this type carries the field
    because the id and the ordering already depend on it.
    """

    id: str
    source: str
    family: str
    benign_class: str | None
    dressing: tuple[str, ...]
    text: str
    label: int

    def __post_init__(self) -> None:
        for name in ("id", "source", "text"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {value!r}")
        if not self.id:
            raise ValueError("id is empty; every corpus row is addressable by id")
        if not self.source:
            raise ValueError("source is empty; every corpus row names where it came from")

        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if isinstance(self.label, bool) or self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")

        # The pair is checked, not each side alone. An attack row carrying label 0 is exactly the
        # drift this type exists to make impossible, and each field is individually valid.
        expected = ATTACK if self.family == FAMILY_ATTACK else BENIGN
        if self.label != expected:
            raise ValueError(
                f"family {self.family!r} carries label {expected}, got {self.label!r}"
            )

        if self.family == FAMILY_ATTACK:
            if self.benign_class is not None:
                raise ValueError(
                    f"an attack item has no benign class, got {self.benign_class!r}"
                )
        elif self.benign_class not in BENIGN_CLASSES:
            raise ValueError(
                f"a benign item must name one of {BENIGN_CLASSES}, got {self.benign_class!r}"
            )

        if not isinstance(self.dressing, tuple) or not all(
            isinstance(link, str) and link for link in self.dressing
        ):
            raise ValueError(
                f"dressing must be a tuple of non-empty dressing names, got {self.dressing!r}"
            )

    def as_json_object(self) -> dict[str, object]:
        """The serialized form, with every key present and in a declared order.

        Written by one module and read by another, so the key set is fixed here rather than in
        whichever of the two is edited first.
        """
        return {
            "id": self.id,
            "source": self.source,
            "family": self.family,
            "benign_class": self.benign_class,
            "dressing": list(self.dressing),
            "text": self.text,
            "label": self.label,
        }


# --- the scoring pass --------------------------------------------------------------------------

RAW: Final[str] = "raw"
"""The item's text exactly as `data/*.jsonl` carries it, handed to the baseline unchanged."""

CANONICAL: Final[str] = "canonical"
"""The same item after the canonicalization layer. The other half of every comparison."""

CONDITIONS: Final[tuple[str, ...]] = (RAW, CANONICAL)
"""The two conditions every item is scored under, and the whole of them.

A closed vocabulary rather than a free string, for the reason `FAMILIES` is one: the headline
number is a difference between two conditions, and a third condition appearing in the scores file
-- a typo, a variant somebody tried once -- would make "the difference" ill-defined while every
individual record still looked valid.
"""


@dataclass(frozen=True, slots=True)
class ItemScore:
    """One corpus item, one baseline, one condition: the number the table is computed from.

    `Score` is what the model boundary returns and says nothing about *what* was scored.  This is
    the committed record, and it carries its own coordinates so that a line of the scores file
    means something without the corpus beside it.  The three that identify it -- `item_id`,
    `baseline_key`, `condition` -- are the key the shard algebra partitions on and the key the
    coverage check counts; the rest travel with it because 4-1 and 4-3 group by them and a group
    key resolved by joining against another file is a group key that can silently join wrongly.

    **No class, and no threshold.** `p_injection` is committed as it was computed.  Turning it
    into a class is story 4-3's job, at the per-baseline threshold `pins.toml` declares, in one
    place -- because a threshold applied at write time is a threshold that cannot be changed
    without re-running eighty-five hours of inference, and one applied in two places is one that
    will eventually differ between them.

    **`max_depth_reached` and `ceiling_hit` are present only under `CANONICAL`, and `None` under
    `RAW`.** They are outcomes of the canonicalization layer, and the raw condition does not run
    it: a depth of `0` on a raw record would be indistinguishable from a canonical record whose
    document needed no decoding, so a reader tallying FR10's ceiling hits would be tallying over
    twice the population.  `None` rather than an absent key, for the reason `CorpusItem` gives
    about `benign_class`: absent-versus-null is the difference between "this record has no depth"
    and "somebody forgot to write one".

    `family`, `benign_class` and `label` are copied from the `CorpusItem` that was scored and
    checked against each other here exactly as `CorpusItem` checks them, so a record that took a
    wrong turn between the corpus and the scores file is refused at the point it is built.
    """

    item_id: str
    family: str
    benign_class: str | None
    label: int
    baseline_key: str
    condition: str
    p_injection: float
    n_windows: int
    max_depth_reached: int | None = None
    ceiling_hit: bool | None = None

    def __post_init__(self) -> None:
        for name in ("item_id", "baseline_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if isinstance(self.label, bool) or self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        # The same pairing `CorpusItem` enforces, enforced again on the way out. The corpus is
        # verified on read, but the copy into this record is a second chance to attach the wrong
        # label to a text, and it is the copy the published rates are computed over.
        expected = ATTACK if self.family == FAMILY_ATTACK else BENIGN
        if self.label != expected:
            raise ValueError(
                f"family {self.family!r} carries label {expected}, got {self.label!r}"
            )
        if self.family == FAMILY_ATTACK:
            if self.benign_class is not None:
                raise ValueError(
                    f"an attack item has no benign class, got {self.benign_class!r}"
                )
        elif self.benign_class not in BENIGN_CLASSES:
            raise ValueError(
                f"a benign item must name one of {BENIGN_CLASSES}, got {self.benign_class!r}"
            )

        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")

        # Delegated rather than restated: `Score` already owns what a probability and a window
        # count are, including that NaN fails every comparison and that a bool is not an int.
        # Restating the rules here is how the two would come to disagree.
        checked = Score(p_injection=self.p_injection, n_windows=self.n_windows)
        object.__setattr__(self, "p_injection", checked.p_injection)

        canonical = self.condition == CANONICAL
        depth, hit = self.max_depth_reached, self.ceiling_hit
        if canonical:
            if isinstance(depth, bool) or not isinstance(depth, int):
                raise ValueError(
                    f"a {CANONICAL} score reports the depth its document reached, got {depth!r}"
                )
            if depth < 0:
                raise ValueError(f"max_depth_reached must not be negative, got {depth!r}")
            if not isinstance(hit, bool):
                raise ValueError(
                    f"a {CANONICAL} score reports whether the ceiling was hit, got {hit!r}"
                )
        else:
            if depth is not None:
                raise ValueError(
                    f"a {self.condition!r} score reports no depth: the canonicalization layer "
                    f"did not run over it, and a 0 here is indistinguishable from a canonical "
                    f"document that needed no decoding, got {depth!r}"
                )
            if hit is not None:
                raise ValueError(
                    f"a {self.condition!r} score reports no ceiling hit: there was no recursion "
                    f"to hit a ceiling, got {hit!r}"
                )

    def as_json_object(self) -> dict[str, object]:
        """The serialized form, every key present, in a declared order.

        Written by the shard walk and read by the merge, in different processes and possibly on
        different machines, so the key set is fixed here rather than in whichever of the two is
        edited first.
        """
        return {
            "item_id": self.item_id,
            "family": self.family,
            "benign_class": self.benign_class,
            "label": self.label,
            "baseline_key": self.baseline_key,
            "condition": self.condition,
            "p_injection": self.p_injection,
            "n_windows": self.n_windows,
            "max_depth_reached": self.max_depth_reached,
            "ceiling_hit": self.ceiling_hit,
        }


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


# --- the intervals every published number carries ------------------------------------------------

WILSON_SCORE: Final[str] = "wilson-score"
"""The plain Wilson score interval, not continuity-corrected. Every single-proportion rate."""

NEWCOMBE_PAIRED: Final[str] = "newcombe-paired-score"
"""Newcombe's method 10 for the difference of two proportions measured on the same items."""

AUC_STRUCTURAL: Final[str] = "auc-structural-components"
"""The empirical (Sen/DeLong) structural-component variance for one ROC AUC."""

DELTA_AUC_STRUCTURAL: Final[str] = "delta-auc-structural-components"
"""The same structural components for two conditions over one item set, with their covariance."""

INTERVAL_METHODS: Final[tuple[str, ...]] = (
    WILSON_SCORE,
    NEWCOMBE_PAIRED,
    AUC_STRUCTURAL,
    DELTA_AUC_STRUCTURAL,
)
"""The whole of the methods this project publishes an interval under.

Closed rather than free, and validated at construction, for the reason `CONDITIONS` is closed. An
interval is a credibility claim, and which method produced it is the difference between two claims
that print identically. A free string lets `"wilson"`, `"Wilson"` and `"wilson-score-cc"` all reach
the results file, where a reader comparing two runs cannot tell a rename from a method change.
"""


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval and the name of the method that produced it.

    The name rides on the value rather than sitting in a `methods` block beside it. The epic asks
    that a reader know which interval they are reading; a side mapping discharges that only while
    somebody keeps it in step, and when it drifts, nothing fails -- which is defect pattern 1 in
    its exact shape, evidence recorded beside a value and never compared to it. Here the two are
    inseparable: an interval cannot be constructed without a method from the closed vocabulary, so
    no number can reach a serializer with its method unstated, and swapping Wilson for its
    continuity-corrected variant changes a string a golden test reads.

    `lo` and `hi` are fractions, never percentages. Formatting is a rendering decision and a record
    that is sometimes 0.83 and sometimes 83.0 is a record with two units.
    """

    lo: float
    hi: float
    method: str

    def __post_init__(self) -> None:
        for name in ("lo", "hi"):
            value = getattr(self, name)
            # A bool is a float's subtype's sibling here only by accident of `int`; reject it
            # explicitly so `Interval(True, ...)` is not silently 1.0.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} is not finite: {value!r}")
            object.__setattr__(self, name, value)

        if self.lo > self.hi:
            raise ValueError(f"interval is inverted: lo={self.lo!r} exceeds hi={self.hi!r}")

        if self.method not in INTERVAL_METHODS:
            raise ValueError(
                f"method must be one of {INTERVAL_METHODS}, got {self.method!r}"
            )

    @property
    def width(self) -> float:
        """`hi - lo`. Named because three tests compare widths and each spelling it itself is one
        more place the subtraction can be written backwards."""
        return self.hi - self.lo

    def as_json_object(self) -> dict[str, object]:
        """The serialized form. The method travels with the numbers or neither travels."""
        return {"lo": self.lo, "hi": self.hi, "method": self.method}


@dataclass(frozen=True, slots=True)
class PairedCount:
    """The full 2x2 table of two binary conditions measured on the same items.

    Laid out as the pairing reads it: `a` items positive under both, `b` positive under the first
    only, `c` positive under the second only, `d` negative under both.

    **All four cells, and not the discordant pair alone.** `b` and `c` are everything McNemar's
    test needs and everything Tango's score interval needs. Newcombe's method builds a Wilson
    interval for *each marginal* and joins them with a correlation term that reads `a` and `d`, so
    a record shaped `(b, c, n)` does not make a Newcombe implementation fail -- it makes one
    impossible to write, and the natural repair is to reach for the method whose inputs happen to
    be present. That is how a repository publishes Tango's interval under Newcombe's name, and the
    record shape is the only place to stop it.
    """

    a: int
    b: int
    c: int
    d: int

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} counts items and must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value!r}")

    @property
    def n(self) -> int:
        """The items the pair was measured on. Every cell is one of them, counted once."""
        return self.a + self.b + self.c + self.d

    @property
    def first_positive(self) -> int:
        """The first condition's marginal: `a + b`."""
        return self.a + self.b

    @property
    def second_positive(self) -> int:
        """The second condition's marginal: `a + c`."""
        return self.a + self.c

    @property
    def theta(self) -> float:
        """The paired difference `(b - c) / n`, which is also `p1 - p2`.

        One definition with three callers. Written twice it is written backwards once.
        """
        if self.n == 0:
            raise ValueError("a paired difference over no items is undefined")
        return (self.b - self.c) / self.n

    def as_json_object(self) -> dict[str, object]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}


# --- the cells of the table -----------------------------------------------------------------------

AXIS_BASELINE: Final[str] = "baseline"
AXIS_DRESSING_CHAIN: Final[str] = "dressing_chain"
AXIS_CHAIN_CLASS: Final[str] = "chain_class"
AXIS_WINDOW_POLICY: Final[str] = "window_policy"
AXIS_CANON_ON: Final[str] = "canon_on"
AXIS_FAMILY: Final[str] = "family"
AXIS_BENIGN_CLASS: Final[str] = "benign_class"

CELL_AXES: Final[tuple[str, ...]] = (
    AXIS_BASELINE,
    AXIS_DRESSING_CHAIN,
    AXIS_CHAIN_CLASS,
    AXIS_WINDOW_POLICY,
    AXIS_CANON_ON,
    AXIS_FAMILY,
    AXIS_BENIGN_CLASS,
)
"""Every coordinate a published cell has, and the whole of them."""

PROTECTED_AXES: Final[tuple[str, ...]] = (
    AXIS_BENIGN_CLASS,
    AXIS_CHAIN_CLASS,
    AXIS_WINDOW_POLICY,
)
"""AD-11's three, and the three reasons are different rather than one rule with three names.

`benign_class`: a pooled benign number hides the failure the reader came for -- a layer that is
safe on chat and destructive on code reads as a small average.

`chain_class`: a held-out result averaged into bound ones destroys the only evidence of
generalization this artifact has, and destroys it by producing a number that looks fine.

`window_policy`: a sensitivity check averaged into the headline makes the headline neither policy.

Aggregating across one of these is refused where the value would be written down, in `CellKey`,
rather than checked afterwards by a function somebody has to remember to call.
"""

CONTRAST_CANON_ON_VS_OFF: Final[str] = "canon_on_vs_off"
CONTRAST_CLEAN_VS_CHAIN: Final[str] = "clean_vs"
CONTRAST_BOUND_VS_HELD_OUT: Final[str] = "bound_vs_held_out"
CONTRAST_ATTACKS_VS_BENIGN_CLASS: Final[str] = "attacks_vs"

CONTRAST_KINDS: Final[tuple[str, ...]] = (
    CONTRAST_CANON_ON_VS_OFF,
    CONTRAST_CLEAN_VS_CHAIN,
    CONTRAST_BOUND_VS_HELD_OUT,
    CONTRAST_ATTACKS_VS_BENIGN_CLASS,
)
"""The whole of the comparisons this table makes.

Aggregating across an axis is forbidden; comparing across one is required, and the difference
between the two is this field. Without it a `Delta` between canon-on and canon-off has two equally
defensible homes -- filed under `canon_on = true` or under `canon_on = false` -- and two compliant
implementations put it in different cells. The contrast is authoritative and the axis it spans
reads `null`.
"""

CONTRAST_ARGUMENT_REQUIRED: Final[Mapping[str, bool]] = MappingProxyType(
    {
        CONTRAST_CANON_ON_VS_OFF: False,
        CONTRAST_CLEAN_VS_CHAIN: True,
        CONTRAST_BOUND_VS_HELD_OUT: False,
        CONTRAST_ATTACKS_VS_BENIGN_CLASS: True,
    }
)
"""Which kinds name the other side. `clean_vs_<chain>` and `attacks_vs_<benign_class>` do."""

REQUIRED_SPANS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        CONTRAST_CANON_ON_VS_OFF: frozenset({AXIS_CANON_ON}),
        CONTRAST_CLEAN_VS_CHAIN: frozenset({AXIS_DRESSING_CHAIN}),
        CONTRAST_BOUND_VS_HELD_OUT: frozenset({AXIS_CHAIN_CLASS, AXIS_DRESSING_CHAIN}),
        CONTRAST_ATTACKS_VS_BENIGN_CLASS: frozenset({AXIS_FAMILY}),
    }
)
"""The axes a comparison of that kind must span, because its two sides differ on them.

`attacks_vs_<benign_class>` spans `family` and **not** `benign_class`, and the distinction is the
one story 4.4 corrected. Its two sides are attacks and benign items -- they differ on `family`. The
benign class is a *coordinate*: the comparison is against one class, and which one is part of what
the cell is about. Spanning it would leave the axis null, and the only way to recover the class
would be to read the part of `attacks_vs_b_code` after the underscore -- a substring where a field
belongs, in the module whose whole subject is that the two classes must never be confused.
"""

PERMITTED_SPANS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        # `family` is permitted here and not required, because the same contrast keys two
        # different quantities: a paired-proportion difference, which is about one family, and a
        # difference of two AUCs, which is about neither -- an AUC draws its two sides from both
        # halves of the corpus, so `family` is null for what the quantity IS rather than because
        # anything was averaged. Widening it cannot pool anything: `family` is not one of
        # `PROTECTED_AXES`, and `benign_class` stays named on both. What tells the two apart on
        # the record is the interval's method, `newcombe-paired-score` against
        # `delta-auc-structural-components`, which is a field rather than an inference.
        CONTRAST_CANON_ON_VS_OFF: frozenset({AXIS_CANON_ON, AXIS_FAMILY}),
        # A held-out chain compared against `clean` also straddles `chain_class`, because `clean`
        # is bound. Whether it does is a fact about the registries and is decided by
        # `matrix.chain_class` at the call site, not by a literal here -- this module is a leaf and
        # cannot see the registries. So the axis is permitted and not required.
        CONTRAST_CLEAN_VS_CHAIN: frozenset({AXIS_DRESSING_CHAIN, AXIS_CHAIN_CLASS}),
        CONTRAST_BOUND_VS_HELD_OUT: frozenset({AXIS_CHAIN_CLASS, AXIS_DRESSING_CHAIN}),
        CONTRAST_ATTACKS_VS_BENIGN_CLASS: frozenset({AXIS_FAMILY}),
    }
)
"""The most a comparison of that kind may span. Anything beyond it is pooling wearing a label."""


@dataclass(frozen=True, slots=True)
class Contrast:
    """A declared comparison, and the axes its two sides differ on.

    `spans` is carried rather than looked up, because one kind's answer depends on data this module
    cannot see: `clean_vs_<chain>` spans `chain_class` when the chain is held out and not when it is
    bound, and which one it is comes from the chain registries. Carrying it and validating it
    against `REQUIRED_SPANS` and `PERMITTED_SPANS` keeps the decision at the call site that has the
    registry while keeping the record unable to claim a span nobody allows.
    """

    kind: str
    argument: str | None = None
    spans: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.kind not in CONTRAST_KINDS:
            raise ValueError(f"contrast kind must be one of {CONTRAST_KINDS}, got {self.kind!r}")

        needs_argument = CONTRAST_ARGUMENT_REQUIRED[self.kind]
        if needs_argument and not (isinstance(self.argument, str) and self.argument):
            raise ValueError(
                f"contrast {self.kind!r} names the other side and must carry an argument, "
                f"got {self.argument!r}"
            )
        if not needs_argument and self.argument is not None:
            raise ValueError(
                f"contrast {self.kind!r} takes no argument, got {self.argument!r}; a name that "
                f"means nothing to a reader is a name that will come to mean two things"
            )
        if self.kind == CONTRAST_ATTACKS_VS_BENIGN_CLASS and self.argument not in BENIGN_CLASSES:
            raise ValueError(
                f"contrast {self.kind!r} names {self.argument!r}, which is not one of "
                f"{BENIGN_CLASSES}"
            )

        spans = frozenset(self.spans)
        if not spans <= frozenset(CELL_AXES):
            raise ValueError(f"contrast spans {sorted(spans - set(CELL_AXES))}, which are not axes")
        required = REQUIRED_SPANS[self.kind]
        if not required <= spans:
            raise ValueError(
                f"contrast {self.kind!r} must span {sorted(required)}; it declares {sorted(spans)}"
            )
        permitted = PERMITTED_SPANS[self.kind]
        if not spans <= permitted:
            raise ValueError(
                f"contrast {self.kind!r} may span at most {sorted(permitted)}; it declares "
                f"{sorted(spans - permitted)} as well, which is aggregation wearing a label"
            )
        object.__setattr__(self, "spans", spans)

    @property
    def name(self) -> str:
        """How the contrast appears in a results file: `canon_on_vs_off`, `clean_vs_base64`."""
        return self.kind if self.argument is None else f"{self.kind}_{self.argument}"

    def as_json_object(self) -> dict[str, object]:
        return {"kind": self.kind, "argument": self.argument, "spans": sorted(self.spans)}


CHAIN_CLASSES_FOR_KEYS: Final[tuple[str, ...]] = ("bound", "held_out")
"""The two halves of the table, spelled here because this module is a leaf.

`corpus/matrix.py` declares the same pair as `CHAIN_CLASSES` and decides which one a chain belongs
to; it imports this module, so this module cannot import it back. `tests/harness/test_aggregate.py`
asserts the two tuples are equal, which is the comparison that makes a second spelling safe.
"""


@dataclass(frozen=True, slots=True)
class CellKey:
    """What a published number is about. The pooling rule is a precondition of it existing.

    A protected axis is `None` on this key **only** when a `Contrast` declares the cell spans it.
    That is the whole enforcement: `benign_class = None` on a benign cell with no contrast is not a
    cell that gets flagged later, it is a value that cannot be written down.

    `family` is on the key for a reason worth stating, because it looks redundant beside
    `benign_class`. Without it, `benign_class = None` means both "this cell is about attacks, which
    have no benign class" and "somebody averaged b_code into b_chat" -- the two readings this whole
    story exists to keep apart. With it, the pair is checked exactly as `CorpusItem` and `ItemScore`
    already check it: the third copy of one rule, deliberately the same rule.
    """

    baseline: str | None
    dressing_chain: str | None
    chain_class: str | None
    window_policy: str | None
    canon_on: bool | None
    family: str | None
    benign_class: str | None = None
    contrast: Contrast | None = None

    def __post_init__(self) -> None:
        if self.contrast is not None and not isinstance(self.contrast, Contrast):
            raise ValueError(f"contrast must be a Contrast, got {self.contrast!r}")
        spans = self.contrast.spans if self.contrast is not None else frozenset()

        for axis in CELL_AXES:
            value = getattr(self, axis)
            if axis in spans:
                if value is not None:
                    raise ValueError(
                        f"{axis} is spanned by contrast {self.contrast.name!r} and must be null "
                        f"on that cell, got {value!r}; the contrast is authoritative"
                    )
                continue
            if axis == AXIS_BENIGN_CLASS:
                continue  # decided below, as a pair with family
            if axis == AXIS_CANON_ON:
                if not isinstance(value, bool):
                    raise ValueError(f"canon_on must be a bool, got {value!r}")
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{axis} must be a non-empty string, got {value!r}"
                    + (
                        f"; {axis} is one of AD-11's protected axes and a cell that leaves it "
                        f"unnamed is a cell aggregated across it"
                        if axis in PROTECTED_AXES
                        else ""
                    )
                )

        # `benign_class` is decided on every path, and the four are written out rather than
        # collapsed. An earlier version checked it only when `family` was named, which left a cell
        # whose contrast spanned `family` free to carry a null class with nothing claiming it --
        # a pooled AUC, reachable, in the module that exists to make one unreachable.
        if AXIS_BENIGN_CLASS in spans:
            pass  # the axis is null by the check above, and the contrast says why
        elif AXIS_FAMILY in spans:
            # A comparison across the two halves still names the class it was scored against.
            if self.benign_class not in BENIGN_CLASSES:
                raise ValueError(
                    f"a cell whose contrast spans family still names the benign class it was "
                    f"measured against, got {self.benign_class!r}; without it a reader would have "
                    f"to parse the contrast's name to learn which class this is about"
                )
        elif self.family == FAMILY_ATTACK:
            if self.benign_class is not None:
                raise ValueError(f"an attack cell has no benign class, got {self.benign_class!r}")
        elif self.benign_class not in BENIGN_CLASSES:
            raise ValueError(
                f"a benign cell must name one of {BENIGN_CLASSES}, got {self.benign_class!r}; "
                f"a false-positive rate over both classes at once is the number FR3.1 exists "
                f"to prevent"
            )

        if AXIS_FAMILY not in spans and self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")

        if self.chain_class is not None and self.chain_class not in CHAIN_CLASSES_FOR_KEYS:
            raise ValueError(
                f"chain_class must be one of {CHAIN_CLASSES_FOR_KEYS}, got {self.chain_class!r}"
            )

    def as_json_object(self) -> dict[str, object]:
        payload: dict[str, object] = {axis: getattr(self, axis) for axis in CELL_AXES}
        payload["contrast"] = None if self.contrast is None else self.contrast.name
        return payload


@dataclass(frozen=True, slots=True)
class Rate:
    """An estimate: `k` of `n`, with the interval the method named in it produced.

    The epic writes this type as `Rate(k, n, lo, hi)`. It carries an `Interval` instead, with `lo`
    and `hi` as properties, because story 4.1 made an interval unable to exist without the name of
    the method that produced it -- and flattening it to two floats here would undo that in the first
    module to consume it, invisibly, since two floats look exactly like an interval.

    `lo` and `hi` are fractions in [0,1]. Percentages are a rendering decision and a record that is
    sometimes 0.83 and sometimes 83.0 is a record with two units.
    """

    k: int
    n: int
    interval: Interval
    key: CellKey

    def __post_init__(self) -> None:
        _refuse_a_bad_tally(self.k, self.n)
        if not isinstance(self.interval, Interval):
            raise ValueError(f"interval must be an Interval, got {self.interval!r}")
        if not 0.0 <= self.interval.lo <= self.interval.hi <= 1.0:
            raise ValueError(
                f"a rate's interval lies in [0,1] as a fraction, got {self.interval!r}"
            )

    @property
    def lo(self) -> float:
        return self.interval.lo

    @property
    def hi(self) -> float:
        return self.interval.hi

    @property
    def value(self) -> float:
        return self.k / self.n

    def as_json_object(self) -> dict[str, object]:
        return {
            "kind": "rate",
            "key": self.key.as_json_object(),
            "k": self.k,
            "n": self.n,
            "value": self.value,
            "interval": self.interval.as_json_object(),
        }


@dataclass(frozen=True, slots=True)
class Count:
    """A census: `k` of `n`, and **no interval, with no field to put one in**.

    A ceiling hit is not a sample from a population of possible ceiling hits. It is a property of
    the corpus as built, complete, and the corpus is not a draw this run could have made
    differently. An interval beside it would invite a reader to generalise something that has no
    population to generalise to.

    Saying that in a docstring while leaving an `interval` field defaulting to `None` would last
    until the first person who found the gap untidy. There is no field.
    """

    k: int
    n: int
    key: CellKey

    def __post_init__(self) -> None:
        _refuse_a_bad_tally(self.k, self.n)

    def as_json_object(self) -> dict[str, object]:
        return {"kind": "count", "key": self.key.as_json_object(), "k": self.k, "n": self.n}


@dataclass(frozen=True, slots=True)
class Auc:
    """A threshold-free estimate: rank separation, with its structural-component interval.

    `tied_pairs` of `total_pairs` says how many of the comparisons behind this number were ties
    rather than orderings. It is on the record rather than derivable later because story 4.4's
    saturation limit is a claim about *why* an AUC moved: a layer that pushes every score into the
    ceiling at `p = 1` manufactures ties and drives AUC **down** without changing any ordering, and
    this corpus is exactly where that happens. A reader who can see "8 of 16 were ties" can tell
    that case from a real loss of separation; a reader given only the number cannot.
    """

    value: float
    interval: Interval
    n_positive: int
    n_negative: int
    key: CellKey
    tied_pairs: int = 0
    total_pairs: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError(f"an AUC is a number, got {self.value!r}")
        value = float(self.value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"an AUC lies in [0,1], got {value!r}")
        object.__setattr__(self, "value", value)
        if not isinstance(self.interval, Interval):
            raise ValueError(f"interval must be an Interval, got {self.interval!r}")
        for name in ("n_positive", "n_negative"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} counts items and must be a positive int, got {count!r}")
        for name in ("tied_pairs", "total_pairs"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} counts comparisons and must be an int, got {count!r}")
        if self.tied_pairs > self.total_pairs:
            raise ValueError(
                f"tied_pairs {self.tied_pairs} exceeds total_pairs {self.total_pairs}"
            )
        if self.total_pairs not in (0, self.n_positive * self.n_negative):
            raise ValueError(
                f"total_pairs is every positive against every negative, "
                f"{self.n_positive} x {self.n_negative}, got {self.total_pairs}"
            )

    def as_json_object(self) -> dict[str, object]:
        return {
            "kind": "auc",
            "key": self.key.as_json_object(),
            "value": self.value,
            "interval": self.interval.as_json_object(),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "tied_pairs": self.tied_pairs,
            "total_pairs": self.total_pairs,
        }


@dataclass(frozen=True, slots=True)
class Delta:
    """A difference between two conditions, with an interval the difference produced.

    **Never a subtraction of two `Rate`s.** Two rates measured on the same items are correlated, and
    an interval assembled from theirs is wrong in the anti-conservative direction on exactly the
    paired comparison N1 reads. The interval here comes from `stats.newcombe_paired_interval` over
    the full 2x2, or from `stats.delta_auc` over the structural components -- and the method name
    riding on the `Interval` is what tells a reader which.

    The `key`'s spanned axis is `null` and its `contrast` says why. A `Delta` with no contrast is
    refused here rather than left to mean whatever the reader assumes.
    """

    value: float
    interval: Interval
    key: CellKey

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError(f"a delta is a number, got {self.value!r}")
        object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.interval, Interval):
            raise ValueError(f"interval must be an Interval, got {self.interval!r}")
        if self.interval.method == WILSON_SCORE:
            raise ValueError(
                "a delta's interval is not a Wilson interval; Wilson estimates one proportion, and "
                "a difference built from two of them inherits neither their pairing nor their "
                "covariance"
            )
        if self.key.contrast is None:
            raise ValueError(
                "a Delta carries a contrast: without it, a difference between canon-on and "
                "canon-off has two equally defensible homes and lands in different cells in two "
                "compliant implementations"
            )

    @property
    def contrast(self) -> Contrast:
        """The comparison this cell is. Authoritative over the axis it spans, which reads null."""
        assert self.key.contrast is not None  # refused at construction
        return self.key.contrast

    def as_json_object(self) -> dict[str, object]:
        return {
            "kind": "delta",
            "key": self.key.as_json_object(),
            "contrast": self.contrast.as_json_object(),
            "value": self.value,
            "interval": self.interval.as_json_object(),
        }


def _refuse_a_bad_tally(k: object, n: object) -> None:
    for name, value in (("k", k), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} counts items and must be an int, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value!r}")
    assert isinstance(k, int) and isinstance(n, int)
    if n == 0:
        raise ValueError("a cell over no items is not a cell; there is nothing it is about")
    if k > n:
        raise ValueError(f"k must not exceed n, got k={k} and n={n}")
