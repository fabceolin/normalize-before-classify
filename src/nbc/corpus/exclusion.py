"""FR3.3's third tooth: no corpus row is text a baseline was taught to call safe.

The first two teeth are declaration teeth and live in `pins.py`: what the cards say, recorded as
data so a rule can read it. They refuse a baseline trained on the pinned attack dataset, and a
baseline that reaches it at one hop through a seed. Neither removes a single row. This module is
the tooth that removes rows, and it is the one the counter-metric rests on: recall looks better on
attacks a model has seen, and the **false-positive rate looks better on benign text it was taught
to call safe**, which is the number the whole artifact is built to defend.

**The obligation is consumed, not re-derived.** `Pins.required_exclusion_sources()` names the seeds
a declared `seeded-from-declared-training-source` bought, and until this module existed nothing
outside `pins.py` read it -- the evidence recorded beside the value and never compared to it. So
`plan()` reads that function and marks those sources **required**, and a required source that the
run could not actually download and intersect aborts. That is the difference between an exemption
granted and an exemption discharged.

**Two sets, and they are not the same size.** The pins declare an `[[exclusion_source]]` for every
source either lineage block implies -- every source a baseline declares `trained-on`, plus every
seed a pinned dataset's card names, twelve today. `required_exclusion_sources()` is the strictly
smaller set the declared hops oblige, two today. The wide set is what the build removes against;
the small set is what it may not proceed without. A source in the wide set that cannot be read is
**named** in the report; a source in the small set that cannot be read stops the run.

**Two of the twelve cannot be read, and they fail differently.** One answers HTTP 401 and hands
back no commit (`unreachable`). One resolves fine and is published as a loading script the pinned
`datasets` refuses (`unreadable`) -- its sha is pinned and verified like any other, and its rows
are still out of reach. Both are declared, both are compared against what the run actually
observes, and neither is treated as contributing zero. Neither is one of the two required seeds,
which is the only reason the build proceeds at all.

**The declared normalization, and why its order is a gate.** Two texts are the same row when
`normalize` maps them to the same string: NFKC, then lowercased, then whitespace collapsed. That
sentence is published in `pins.toml` and in the README, so it is implemented as written --
`str.lower()` and not `str.casefold()`, because a normalization that differs from its published
description is the defect this repository keeps finding in itself. The order is load-bearing rather
than incidental: NFKC maps U+00A0 to an ordinary space, so collapsing whitespace before NFKC leaves
`a b` and `a b` as different rows. `tests/corpus/test_exclusion.py` holds each step and that
ordering with the input that fails without it.

**Why this removes where the benign cross-check will abort.** The benign cross-check story 3.7
specifies -- a benign source item that literally contains a pinned attack payload -- is designed to
abort the build and make a human look, because it is a **gold-label error**: the builder labels
benign material benign by construction, so one of the two labels is wrong and nothing here can say
which. (That gate is not written yet; this paragraph states why the two differ, not that both
exist.) Training overlap is not that. The row is
correctly labelled and merely uninformative *for one baseline*, so removing it costs sample size
and nothing else. What made silent exclusion unacceptable in the other case was the silence, not
the exclusion -- which is why every removal here is counted per source and published rather than
quietly applied.

This module is pure and offline. It imports the standard library and `nbc.errors` and `nbc.pins`,
never `datasets` and never a socket: everything that touches the hub is in `corpus/build.py`, so
the whole decision procedure is covered by a suite that runs with no network at all.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Callable, Final, Generic, Iterable, Mapping, Sequence, TypeVar

from nbc.errors import NbcError
from nbc.pins import HTTP_OK, ExclusionSource, Pins, canonical_source_id

__all__ = [
    "NORMALIZATION",
    "NO_ANSWER",
    "ExclusionIndex",
    "ExclusionReport",
    "ExclusionSetUnusable",
    "Filtered",
    "Observation",
    "PlannedSource",
    "SourceOutcome",
    "build_index",
    "declaration_digest",
    "filter_rows",
    "normalize",
    "normalized_texts",
    "outcomes_of",
    "plan",
    "verify_observations",
]

NORMALIZATION: Final[str] = "nfkc-lower-collapse-whitespace"
"""The name of the rule two texts are compared under, carried into the report.

A count without its rule is a number nobody can reproduce: the same reach measures 3071 rows under
this normalization and 3073 on exact text, and `pins.toml` records both because the pair was once
published as one number.
"""

NO_ANSWER: Final[int] = 0
"""The observed status for a probe that got no HTTP answer at all -- a DNS failure, a timeout.

Not `None` and not 200: it has to compare unequal to every declared status, so that "the hub could
not be reached" fails the same gate as "the hub said 404" rather than passing as available.
"""


class ExclusionSetUnusable(NbcError, exit_code=15):
    """The exclusion set the pins declare is not the one this machine can actually apply.

    Code 15 because 3 through 14 are taken. It is a sibling of `PinMismatch` rather than of
    `PinsFileInvalid`: in the ordinary case nothing in the repository is wrong and the world moved
    -- a source went private, a schema changed, a gated repository opened up. The remedy is to
    re-read the sources, record what they now say in `pins.toml`, and re-run. The run must not
    proceed in the meantime, because a corpus filtered against a different set of sources than the
    published one is a corpus whose false-positive rate means something else.

    Every problem is collected before aborting, so a build that is wrong in three places says all
    three in one run.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "the declared exclusion set could not be applied:\n  - "
            + "\n  - ".join(problems)
        )
        self.problems = tuple(problems)


def normalize(text: str) -> str:
    """The declared comparison form: NFKC, lowercased, whitespace collapsed, in that order.

    Never the form anything is published in -- this is only ever the key two texts are compared
    under, exactly as `canonical_source_id` is in `pins.py`.

    `split()` with no argument splits on every Unicode whitespace run and drops leading and
    trailing whitespace, so the collapse also strips. It runs last because NFKC produces
    whitespace: U+00A0 becomes an ordinary space under NFKC and would survive an earlier collapse
    as a non-space character.
    """
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


@dataclass(frozen=True, slots=True)
class PlannedSource:
    """One exclusion source, with the one fact `pins.py` cannot carry: whether it is required.

    `required` is read from `Pins.required_exclusion_sources()` and from nowhere else. That
    function is the whole content of decision D-C: the pins grant a one-hop exemption on the
    promise that the coincident rows are removed, and this flag is where the promise becomes an
    obligation something checks.
    """

    source: ExclusionSource
    required: bool

    @property
    def key(self) -> str:
        return self.source.key

    @property
    def repository(self) -> str:
        return self.source.repository


@dataclass(frozen=True, slots=True)
class Observation:
    """What the hub and the loader actually answered about one source, this run.

    Three facts, because the pins declare three and each is compared:

    - `http_status` -- what the hub answered, against the pinned status;
    - `loadable` -- whether the pinned reader returned rows, against `ExclusionSource.loadable`.
      A source can resolve and still be unreadable, which is exactly what one pinned source does;
    - `texts_loaded` -- the count of distinct normalized texts contributed. A source that loaded
      and yielded nothing has changed shape under its pin, and reporting that as "zero overlap"
      would be indistinguishable from a source with no overlap.

    `load_error` is why the load failed, in the reader's own words, carried into the report so a
    reader of the results learns what the gap is rather than only that there is one.
    """

    http_status: int
    loadable: bool = False
    texts_loaded: int = 0
    load_error: str = ""


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    """One row of the published accounting: what was declared, what was seen, what it removed."""

    key: str
    repository: str
    revision: str
    required: bool
    declared_availability: str
    declared_http_status: int
    observed_http_status: int
    observed_loadable: bool
    texts_loaded: int
    matched_rows: int
    load_error: str

    @property
    def read(self) -> bool:
        """Whether this source was actually downloaded and intersected against the corpus."""
        return self.observed_http_status == HTTP_OK and self.observed_loadable

    def as_run_fields(self) -> dict[str, object]:
        return {
            "key": self.key,
            "repository": self.repository,
            "revision": self.revision,
            "required": self.required,
            "declared_availability": self.declared_availability,
            "declared_http_status": self.declared_http_status,
            "observed_http_status": self.observed_http_status,
            "observed_loadable": self.observed_loadable,
            "read": self.read,
            "texts_loaded": self.texts_loaded,
            "load_error": self.load_error,
            # Absent rather than 0 for a source that was never read. Zero is a measurement and
            # this is not one, and the whole point of naming an unreachable source is that a
            # reader must not be able to read its contribution as nothing.
            "matched_rows": self.matched_rows if self.read else None,
        }


@dataclass(frozen=True, slots=True)
class ExclusionReport:
    """What the filter did, in the shape `results.json` and the README publish.

    `rows_removed` and the per-source counts do not add up, on purpose: a row appearing in three
    exclusion sources is removed once and counted by all three. Publishing only the sum would
    overstate the removal; publishing only the total would hide which source carried it.
    """

    normalization: str
    declaration_digest: str
    rows_in: int
    rows_removed: int
    outcomes: tuple[SourceOutcome, ...]

    @property
    def rows_kept(self) -> int:
        return self.rows_in - self.rows_removed

    @property
    def unread(self) -> tuple[SourceOutcome, ...]:
        """The sources that were declared and could not be read. Never treated as zero."""
        return tuple(outcome for outcome in self.outcomes if not outcome.read)

    def as_run_fields(self) -> dict[str, object]:
        return {
            "exclusion": {
                "normalization": self.normalization,
                "declaration_digest": self.declaration_digest,
                "rows_in": self.rows_in,
                "rows_removed": self.rows_removed,
                "rows_kept": self.rows_kept,
                "sources": [outcome.as_run_fields() for outcome in self.outcomes],
                # Named, so a reader cannot mistake an unreachable source for one that
                # contributed nothing.
                "unread_sources": [outcome.repository for outcome in self.unread],
            }
        }


def plan(pins: Pins) -> tuple[PlannedSource, ...]:
    """The sources to process this run, in pinned order, with the required ones marked.

    Aborts when a source `required_exclusion_sources()` names has no `[[exclusion_source]]` entry.
    `load_pins` already refuses such a file, and this is not that check: it holds for a `Pins`
    assembled in code rather than read from disk, which is how every consumer of this module in a
    test receives one.
    """
    by_canonical = {
        canonical_source_id(source.repository): source for source in pins.exclusion_sources
    }
    required = {canonical_source_id(name): name for name in pins.required_exclusion_sources()}

    missing = sorted(
        raw for canonical, raw in required.items() if canonical not in by_canonical
    )
    if missing:
        raise ExclusionSetUnusable(
            *(
                f"{name} is a required exclusion source -- a baseline declares training on it "
                f"and a pinned dataset's card names it as a seed -- and no [[exclusion_source]] "
                f"pins it, so the build has nothing to download and nothing to remove"
                for name in missing
            )
        )

    return tuple(
        PlannedSource(source=source, required=canonical_source_id(source.repository) in required)
        for source in pins.exclusion_sources
    )


def verify_observations(
    planned: Sequence[PlannedSource], observations: Mapping[str, Observation]
) -> None:
    """Compare what the pins declare about each source against what this run saw. Abort on drift.

    Five ways this fails, and each has an input that produces it:

    - a planned source with no observation -- the loop skipped it;
    - a declared status the hub did not answer, **in either direction**: a source declared
      reachable that now answers 404, and a gated source that opened up, are both moves in the
      exclusion set, and a moving exclusion set silently changes which rows survive;
    - a source the pins call readable that the pinned reader refused, or one the pins call
      unreadable that loaded fine -- the same comparison, one axis down. A repository can resolve
      at its sha and still refuse to hand over rows, so the status alone does not answer it;
    - a **required** source that was not read, even where the pins honestly say so, because the
      published recall is a ceiling until exactly those rows are removed;
    - a source that loaded and yielded no text at all, which is a schema change wearing the face
      of a source with no overlap.
    """
    problems: list[str] = []

    for entry in planned:
        observed = observations.get(entry.key)
        if observed is None:
            problems.append(
                f"{entry.repository} is a declared exclusion source and this run recorded no "
                f"observation for it; a source nobody probed is a source silently treated as "
                f"contributing zero"
            )
            continue

        if observed.http_status != entry.source.http_status:
            problems.append(
                f"{entry.repository} is pinned as {entry.source.availability!r} answering HTTP "
                f"{entry.source.http_status} (checked {entry.source.checked_on}) and this run "
                f"observed HTTP {observed.http_status}; the exclusion set has moved, and a "
                f"moving exclusion set changes which rows survive into the corpus"
            )
        elif observed.http_status == HTTP_OK and observed.loadable != entry.source.loadable:
            problems.append(
                f"{entry.repository} is pinned as {entry.source.availability!r} and the hub "
                f"answered {HTTP_OK} as declared, but the pinned reader "
                + (
                    f"refused it: {observed.load_error or 'no reason recorded'}"
                    if entry.source.loadable
                    else "loaded it, so the pins now understate the exclusion set and rows this "
                    "run could have removed were left in"
                )
            )
        elif observed.loadable and observed.texts_loaded == 0:
            problems.append(
                f"{entry.repository} was read at its pinned revision and yielded no text at "
                f"all; a source whose rows stopped being readable looks exactly like a source "
                f"with no overlap, and the two publish very different numbers"
            )

        if entry.required and not (
            observed.http_status == HTTP_OK and observed.loadable
        ):
            problems.append(
                f"{entry.repository} is a required exclusion source and its rows were not read "
                f"(HTTP {observed.http_status}, loadable={observed.loadable}). The published "
                f"clean recall is an upper bound until the rows reaching it are removed, so a "
                f"run that skipped it would republish that ceiling as if it were the value"
            )

    if problems:
        raise ExclusionSetUnusable(*problems)


@dataclass(frozen=True, slots=True)
class ExclusionIndex:
    """Normalized text -> the source keys that carry it.

    One mapping rather than one set per source: the filter asks about every corpus row, and a
    per-source loop would walk twelve sets per row where one lookup answers both questions -- is
    this row excluded, and by which sources.
    """

    sources_by_text: Mapping[str, tuple[str, ...]]

    def __len__(self) -> int:
        return len(self.sources_by_text)

    def sources_for(self, text: str) -> tuple[str, ...]:
        return self.sources_by_text.get(normalize(text), ())


def normalized_texts(texts: Iterable[str]) -> set[str]:
    """The distinct non-blank comparison keys a stream of raw texts contributes.

    Consumes an iterator, so the caller never has to hold a whole training source in memory at
    once: the pinned sources run to hundreds of thousands of rows and several strings per row,
    and the set of distinct keys is a fraction of that.

    Blank texts are dropped rather than indexed. A source that carries an empty cell -- and most
    of them do -- would otherwise remove every corpus row whose text normalizes to nothing, and
    every such removal would be attributed to whichever sources happened to have a blank column.
    """
    return {key for key in (normalize(text) for text in texts) if key}


def build_index(texts_by_source: Mapping[str, Iterable[str]]) -> ExclusionIndex:
    """Normalize every text each source carries and record which sources carry it.

    Idempotent under `normalized_texts`: handing it already-normalized keys produces the same
    index, which is what lets the build normalize once, as it loads, and still pass the result
    through here.
    """
    sources_by_text: dict[str, list[str]] = {}
    for key in sorted(texts_by_source):
        for normalized in sorted(normalized_texts(texts_by_source[key])):
            sources_by_text.setdefault(normalized, []).append(key)
    return ExclusionIndex(
        sources_by_text={text: tuple(keys) for text, keys in sources_by_text.items()}
    )


_Row = TypeVar("_Row")


@dataclass(frozen=True, slots=True)
class Filtered(Generic[_Row]):
    """The surviving rows, the removed ones, and who removed them."""

    kept: tuple[_Row, ...]
    removed: tuple[_Row, ...]
    matches_by_source: Mapping[str, int]


def filter_rows(
    rows: Iterable[_Row], index: ExclusionIndex, text_of: Callable[[_Row], str]
) -> Filtered[_Row]:
    """Remove every row whose text appears in an exclusion source, counting per source.

    Deterministic and order-preserving: the same rows and the same index produce the same three
    outputs, which is what lets the filtered corpus be part of a build identity rather than a
    property of when it ran. `text_of` is passed rather than assumed because this module is
    written before the corpus row it will be handed.
    """
    kept: list[_Row] = []
    removed: list[_Row] = []
    matches: dict[str, int] = {}

    for row in rows:
        carriers = index.sources_for(text_of(row))
        if not carriers:
            kept.append(row)
            continue
        removed.append(row)
        for key in carriers:
            matches[key] = matches.get(key, 0) + 1

    return Filtered(kept=tuple(kept), removed=tuple(removed), matches_by_source=matches)


def declaration_digest(pins: Pins) -> str:
    """A hash of the exclusion declaration: the normalization, and every source's identity.

    Offered to the corpus `build_id` (story 3.6), which covers the whole build declaration. It is
    a hash of what was *declared*, never of what was observed: two runs of the same pins must
    agree even though one of them fetched over the network and the other read a cache.

    Order-independent by construction -- the entries are sorted -- so re-ordering the
    `[[exclusion_source]]` array does not invent a new corpus, while changing any revision does.
    """
    payload = {
        "normalization": NORMALIZATION,
        "sources": sorted(
            [
                source.repository,
                source.revision,
                source.availability,
                str(source.http_status),
            ]
            for source in pins.exclusion_sources
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def outcomes_of(
    planned: Sequence[PlannedSource],
    observations: Mapping[str, Observation],
    matches_by_source: Mapping[str, int],
) -> tuple[SourceOutcome, ...]:
    """Join the three halves into the published accounting, in pinned order."""
    return tuple(
        SourceOutcome(
            key=entry.key,
            repository=entry.repository,
            revision=entry.source.revision,
            required=entry.required,
            declared_availability=entry.source.availability,
            declared_http_status=entry.source.http_status,
            observed_http_status=observed.http_status,
            observed_loadable=observed.loadable,
            texts_loaded=observed.texts_loaded,
            load_error=observed.load_error,
            matched_rows=matches_by_source.get(entry.key, 0),
        )
        for entry, observed in (
            (entry, observations.get(entry.key, Observation(NO_ANSWER)))
            for entry in planned
        )
    )
