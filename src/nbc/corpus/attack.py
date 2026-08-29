"""The attack half of the corpus: drawn by a declared rule, labelled as it is rendered.

This module is **pure and offline**. It imports the standard library, `nbc.errors`, `nbc.pins`,
`nbc.schema`, story 3.1's `nbc.corpus.exclusion` and story 3.3's `nbc.corpus.matrix` and
`nbc.corpus.dressings`; it never imports `datasets` and never opens a socket. Everything that
reaches the hub is `corpus/build.py`, which hands the pool in as data. That split is what lets the
whole decision procedure -- what is a contradiction, what is a positive, which rows survive, which
are drawn, how each one is dressed, what the file looks like byte for byte -- be covered by a
suite that runs with no network at all.

**Where the labels come from, and where they do not.** The pinned dataset's `attack_label` is a
*selection* input: it says which of its rows this build considers attack payloads. The gold label
written into `data/*.jsonl` is `schema.ATTACK`, asserted by the builder over text it rendered
itself. The two integers coincide today and that is a coincidence: reading the corpus label off
the source row would make the gold label a copy of somebody else's annotation, which is precisely
what FR4 says this repository does not do. `tests/corpus/test_build.py` walks this file's AST and
refuses any `CorpusItem(...)` whose `label=` is not one of the two schema constants.

**The order of the pipeline is load-bearing.** Rows -> contradiction gate -> positives ->
exclusion filter -> draw -> render, once per declared chain. Two of those steps are where they are
for a reason:

- the contradiction gate runs over **every row**, not over the positives. Each of the two texts
  the pinned pool carries at both labels appears once as a positive and once as a benign row, so
  a gate that looked only at rows carrying `attack_label` would see neither and would pass
  vacuously on the exact pool that motivated the requirement.
- the exclusion filter runs **before** the draw, per AD-20. Drawing first would make the realized
  corpus smaller than the declared size by an amount nobody declared; filtering first makes the
  declared size exact, and turns a short pool into an abort instead of a silent shortfall.

**Why two builds are byte-identical.** Every ordering input is content-derived. The payload id is
a truncated SHA-256 of the exact payload text, rows are sorted by `(source, payload_id, chain)`,
and the draw sorts the pool before it shuffles. Nothing reads a row index, a split order, a set
iteration order or a process hash seed. `tests/corpus/test_attack.py` asserts that by building
the same pool twice under different `PYTHONHASHSEED` values in a subprocess, from a shuffled row
order, and comparing bytes.

**Every declared chain, once per drawn payload.** The chains come from
`corpus/matrix.py::CHAINS`, keyed on the attack family, and the text from `dressings.dress`, which
is AD-3's `reduce(apply, chain, payload)`. So one drawn payload becomes as many corpus rows as
there are attack chains, each with its own item id, and the dressing axis of the headline table is
that constant rather than a list any caller assembled. `matrix.validate()` runs before the first
row is rendered, so a chain naming a dressing nothing implements aborts instead of producing a
column that silently does not exist.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Callable, Final, Iterable, Mapping, Sequence

from nbc.corpus.dressings import dress
from nbc.corpus.exclusion import ExclusionIndex, filter_rows
from nbc.corpus.matrix import (
    CHAINS,
    CLEAN_CHAIN,
    render_chain,
    validate as validate_matrix,
)
from nbc.errors import NbcError
from nbc.pins import DRAW_HEAD, DRAW_SEEDED_RANDOM, AttackDataset, AttackDraw
from nbc.schema import ATTACK, FAMILY_ATTACK, CorpusItem

__all__ = [
    "ID_SEPARATOR",
    "PAYLOAD_ID_HEX",
    "AttackDrawReport",
    "AttackDrawUnsatisfiable",
    "LabelContradiction",
    "PoolRow",
    "contradictions",
    "draw_attack_items",
    "id_collisions",
    "item_id",
    "payload_id",
    "render_attack_item",
    "select_payloads",
    "serialize",
    "sort_key_for",
    "verify_splits",
]

PAYLOAD_ID_HEX: Final[int] = 16
"""Hex characters of SHA-256 kept in a payload id.

64 bits. Over a pool of ten thousand payloads the birthday probability of a collision is about
3e-12, and a collision would be caught rather than silently merging two payloads: the builder
refuses a pool in which two distinct texts produce one id.
"""

ID_SEPARATOR: Final[str] = "::"
"""What separates the payload id from the chain in an item id.

The chain half of AD-3's rule -- the names joined by `+`, or the literal `clean` -- is
`matrix.render_chain`, because the benign builders and the held-out registry need the same
spelling and may not reach into the attack module for it.
"""


class LabelContradiction(NbcError, exit_code=16):
    """The pinned dataset carries one text under both labels, and nothing here can say which wins.

    Code 16 because 3 through 15 are taken. This is a **gold-label error in the source**, and it
    is the reason the build stops rather than choosing: exactly one of the two rows is wrong, the
    builder has no evidence about which, and a builder that silently picks one has an unreviewed
    annotation policy -- the precise thing FR4 claims this repository does not have. Two rows in
    ten thousand would not move a rate, and that is not the argument.

    It is a sibling of story 3.7's benign cross-check rather than of `ExclusionSetUnusable`.
    Training overlap removes a row that is correctly labelled and merely uninformative for one
    baseline; this removes nothing, because there is no correct answer to remove toward. An abort
    forces a human to look.

    Every contradictory text is collected before aborting, and each is reported with **both** of
    its rows by split and index, so a human can open the dataset at those two positions.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "the pinned dataset labels the same text both ways:\n  - " + "\n  - ".join(problems)
        )
        self.problems = tuple(problems)


class AttackDrawUnsatisfiable(NbcError, exit_code=17):
    """The declared draw cannot be taken from the pool this run actually read.

    Code 17. Four inputs produce it, and they are different diagnoses rather than one:

    - the splits the reader yielded are not the splits `pins.toml` declares, in either direction.
      A missing split is a count taken over part of the dataset; an extra one is a count taken
      over rows nobody declared;
    - no row carries the declared `attack_label`, so the draw would be over an empty pool and the
      published recall would be a rate over nothing;
    - two distinct payload texts collide on one payload id;
    - fewer positives survive the exclusion filter than the declared size. That is FR5.1's rule
      applied to the attack half: the build fails rather than topping up, because a frame that
      quietly substitutes rows is not a frame.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "the declared attack draw could not be taken:\n  - " + "\n  - ".join(problems)
        )
        self.problems = tuple(problems)


@dataclass(frozen=True, slots=True)
class PoolRow:
    """One row of the pinned dataset, with enough identity to name it in an abort.

    `split` and `index` exist for exactly one consumer: the contradiction message, which has to
    point a human at **both** offending rows. Nothing downstream reads them, and in particular
    nothing about the corpus depends on them -- an id or an ordering derived from a row position
    would make the build a function of how the pool happened to be iterated.
    """

    split: str
    index: int
    text: str
    label: int


def payload_id(text: str) -> str:
    """A stable, content-derived id for one payload.

    Content-derived rather than positional because AD-1's stable order is `(source, payload_id,
    chain)`: an id that carried a row number would make the file's order a property of the read,
    which is the one thing the byte-identical claim cannot depend on.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:PAYLOAD_ID_HEX]


def item_id(payload: str, chain: Sequence[str] = CLEAN_CHAIN) -> str:
    """AD-3's item id: `<payload_id>::<chain>`, the chain joined by `+`, or the literal `clean`.

    The full chain, never the last link: a reported dressing axis that named only the outermost
    dressing would make `base64+base64` and `base64` the same cell.
    """
    return f"{payload}{ID_SEPARATOR}{render_chain(chain)}"


def contradictions(rows: Iterable[PoolRow]) -> tuple[str, ...]:
    """One message per text the pool carries under more than one label. Empty when there are none.

    Runs over every row, both labels. See the module docstring for why looking only at positives
    would pass vacuously on the pinned pool.

    Blank texts are excluded: a dataset carrying two empty cells at two labels is not a
    contradiction about any payload, and the builder drops blank rows anyway.
    """
    seen: dict[str, list[PoolRow]] = {}
    for row in rows:
        if row.text:
            seen.setdefault(row.text, []).append(row)

    problems: list[str] = []
    for text in sorted(seen):
        carried = seen[text]
        labels = {row.label for row in carried}
        if len(labels) < 2:
            continue
        where = ", ".join(
            f"{row.split}[{row.index}] label={row.label}"
            for row in sorted(carried, key=lambda row: (row.split, row.index))
        )
        problems.append(
            f"{text!r} is carried as {where}; exactly one of those labels is wrong and nothing "
            f"here can say which"
        )
    return tuple(problems)


def verify_splits(declared: Sequence[str], observed: Sequence[str]) -> tuple[str, ...]:
    """One message per split that is declared and unread, or read and undeclared. Empty when equal.

    Both directions, because they are different errors with the same consequence. A declared split
    the reader did not yield means the counts are over part of the dataset -- the error this
    project has already made twice, once as rows-for-positives and once as one-split-for-two. A
    split the reader yielded that the pins do not declare means the corpus holds rows from a
    source nobody pinned.
    """
    missing = sorted(set(declared) - set(observed))
    extra = sorted(set(observed) - set(declared))
    problems: list[str] = []
    if missing:
        problems.append(
            f"pins.toml declares split(s) {missing} that the pinned revision did not yield; a "
            f"count taken over part of a dataset is the error this list exists to prevent"
        )
    if extra:
        problems.append(
            f"the pinned revision yielded split(s) {extra} that pins.toml does not declare; the "
            f"draw would be taken over rows no pin describes"
        )
    return tuple(problems)


def sort_key_for(name: str):
    """The declared sort key for a `head` draw, as a function of the payload text.

    A closed mapping rather than a lookup by attribute name: both keys are pure functions of the
    payload, and neither can be made to read a row position, a split or a file order.
    """
    keys = {"text": lambda text: text, "payload_id": payload_id}
    return keys[name]


def select_payloads(payloads: Iterable[str], draw: AttackDraw) -> tuple[str, ...]:
    """The declared draw, taken from a pool that is sorted before anything else happens.

    Sorting first is what makes the result a function of the seed and of nothing else: not of
    parquet row order, not of which split was read first, not of the process hash seed. The
    result is sorted again on the way out so the returned order is content-derived too.

    `random.Random(seed).shuffle` is the draw. The interpreter is pinned to CPython 3.13 exactly
    by `pyproject.toml`, and the Mersenne Twister stream for a given seed is stable within it, so
    the same seed reproduces the same sample wherever this project is allowed to run at all.

    A pool at or below the declared size is taken whole -- but nothing here decides whether that
    is acceptable. The floor is `AttackDrawUnsatisfiable`'s, raised by `draw_attack_items`, so the
    rule lives in one place instead of being half-enforced by a silent truncation here.
    """
    unique = sorted(set(payloads))
    size = draw.sample_size_positives
    if size >= len(unique):
        return tuple(unique)

    if draw.method == DRAW_HEAD:
        ordered = sorted(unique, key=sort_key_for(str(draw.sort_key)))
        return tuple(sorted(ordered[:size]))
    if draw.method == DRAW_SEEDED_RANDOM:
        pool = list(unique)
        random.Random(draw.seed).shuffle(pool)
        return tuple(sorted(pool[:size]))
    # `load_pins` refuses any other value, so reaching here means a `Pins` was assembled in code
    # with a method nothing implements. Silently drawing something would publish a corpus under a
    # rule that does not exist.
    raise AttackDrawUnsatisfiable(
        f"the draw declares selection method {draw.method!r}, which nothing implements; the "
        f"admissible methods are {DRAW_HEAD!r} and {DRAW_SEEDED_RANDOM!r}"
    )


def id_collisions(pairs: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """One message per item id that two different payloads produced. Empty when there are none.

    A separate function so it has a failing input a test can supply: producing a real SHA-256
    prefix collision is not something a test can do, and a check nobody has seen fire is a check
    nobody knows fires. `tests/corpus/test_attack.py` hands it two payloads under one id directly.

    The consequence it prevents is silent: two distinct payloads under one id merge into one
    corpus row, the count drops by one, and every rate computed from it is over a pool that is not
    the pool the report describes.
    """
    seen: dict[str, str] = {}
    problems: list[str] = []
    for identifier, payload in pairs:
        first = seen.setdefault(identifier, payload)
        if first != payload:
            problems.append(
                f"payloads {first!r} and {payload!r} both produce item id {identifier}"
            )
    return tuple(problems)


def render_attack_item(
    payload: str, *, source: str, chain: Sequence[str] = CLEAN_CHAIN
) -> CorpusItem:
    """One corpus row: the rendered text and the gold label, produced by one constructor call.

    The text is `dressings.dress(payload, chain)`, which is AD-3's `reduce(apply, chain, payload)`
    -- so `("base64", "homoglyph")` produces `homoglyph(to_base64(payload))`, the later link
    wrapping the earlier. The empty chain returns the payload itself, which is what makes `clean`
    the identity element of the fold rather than a dressing that happens to change nothing.

    The **payload id is the id of the payload**, never of the dressed text: the whole point of
    AD-3's `<payload_id>::<chain>` is that one payload's ten rows share a stem, so the harness can
    pair the same payload across the dressing axis. Hashing the rendered text instead would make
    every cell an unrelated item and N2's paired comparison unrepresentable.

    `label=ATTACK` is a named schema constant on purpose, and a test enforces it: a label read
    from the source row would be somebody else's annotation wearing this builder's name.
    """
    return CorpusItem(
        id=item_id(payload_id(payload), chain),
        source=source,
        family=FAMILY_ATTACK,
        benign_class=None,
        dressing=tuple(chain),
        text=dress(payload, chain),
        label=ATTACK,
    )


@dataclass(frozen=True, slots=True)
class AttackDrawReport:
    """What the draw did, in numbers a reader can check against the pinned declaration.

    Every count is in **attack positives** except `rows_by_split` and `items_written`, which are
    in rows and are named so. Both units are published because the pair is the error this project
    keeps making: the pool holds three times as many rows as positives, and a size read in the
    wrong unit is off by that much. A third unit now sits under those two -- one drawn positive
    becomes one row **per chain** -- so `chains` travels beside the counts and
    `items_written == drawn_positives * len(chains)` is checkable by a reader of the report rather
    than only by a reader of the code.
    """

    repository: str
    revision: str
    declared_splits: tuple[str, ...]
    chains: tuple[str, ...]
    rows_by_split: Mapping[str, int]
    positives_by_split: Mapping[str, int]
    blank_positive_rows: int
    unique_positives: int
    removed_by_exclusion: int
    surviving_positives: int
    drawn_positives: int
    items_written: int
    draw: AttackDraw

    def as_run_fields(self) -> dict[str, object]:
        return {
            "attack_draw": {
                "repository": self.repository,
                "revision": self.revision,
                "declared_splits": list(self.declared_splits),
                # The dressing axis of the headline table, published with the corpus that carries
                # it: `results.json` reports per chain, and a chain in the results that is not in
                # this list is a cell computed over rows this build did not write.
                "chains": list(self.chains),
                "rows_by_split": dict(sorted(self.rows_by_split.items())),
                "positives_by_split": dict(sorted(self.positives_by_split.items())),
                # Counted and published rather than dropped in silence. The 3071-versus-3073
                # discrepancy this project already shipped came from a truncation nobody
                # reported, and it was found by rederiving a number rather than by reading one.
                "blank_positive_rows": self.blank_positive_rows,
                "unique_positives": self.unique_positives,
                "removed_by_exclusion": self.removed_by_exclusion,
                "surviving_positives": self.surviving_positives,
                "drawn_positives": self.drawn_positives,
                "items_written": self.items_written,
                "declared": self.draw.as_run_fields(),
            }
        }


def draw_attack_items(
    rows: Sequence[PoolRow],
    observed_splits: Sequence[str],
    dataset: AttackDataset,
    index_of: Callable[[], ExclusionIndex],
) -> tuple[tuple[CorpusItem, ...], AttackDrawReport, Mapping[str, int]]:
    """The whole offline pipeline: gate, filter, draw, render. Returns items, report, per-source.

    Aborts before returning anything, so no caller can write a corpus assembled from a pool that
    failed a gate.

    **The index arrives as a thunk, and the order that produces is the point.** Building it is the
    largest download this project makes -- twelve training sources, hundreds of thousands of rows
    -- and the two gates above it cost nothing. Calling it lazily means a pool that contradicts
    itself, or that yielded the wrong splits, aborts in seconds instead of after a gigabyte. The
    ordering is enforced here, once, rather than by asking each caller to check the cheap things
    first.
    """
    # Before anything is rendered, and before the pool is even read for contradictions: a chain
    # naming a dressing nothing implements is a defect in a constant, and finding it after a
    # gigabyte of downloads would be finding it late for no reason.
    validate_matrix()
    chains = CHAINS[FAMILY_ATTACK]

    split_problems = verify_splits(dataset.splits, observed_splits)
    if split_problems:
        raise AttackDrawUnsatisfiable(*split_problems)

    contradiction_problems = contradictions(rows)
    if contradiction_problems:
        raise LabelContradiction(*contradiction_problems)

    positive_rows = [row for row in rows if row.label == dataset.attack_label]
    if not positive_rows:
        observed_labels = sorted({row.label for row in rows})
        raise AttackDrawUnsatisfiable(
            f"{dataset.repository} carries no row at the declared attack_label "
            f"{dataset.attack_label}; the labels this run observed are {observed_labels}, so the "
            f"draw would be taken over an empty pool and the recall published over nothing"
        )

    blank = sum(1 for row in positive_rows if not row.text)
    unique_positives = sorted({row.text for row in positive_rows if row.text})

    filtered = filter_rows(unique_positives, index_of(), lambda text: text)
    survivors = filtered.kept

    if len(survivors) < dataset.draw.sample_size_positives:
        raise AttackDrawUnsatisfiable(
            f"the draw declares {dataset.draw.sample_size_positives} attack positives and "
            f"{len(survivors)} survive the training-overlap filter "
            f"({len(unique_positives)} unique positives, {len(filtered.removed)} removed). "
            f"The build fails rather than topping up from elsewhere: a declared size that "
            f"quietly becomes whatever survived is not a declared size"
        )

    drawn = select_payloads(survivors, dataset.draw)

    # The cross product, and it is a product rather than a choice: AD-20 makes `CHAINS` the
    # dressing axis of the table, so every drawn payload appears once per declared chain or the
    # axis has a hole nothing reports.
    rendered = tuple(
        (render_attack_item(payload, source=dataset.repository, chain=chain), payload)
        for payload in drawn
        for chain in chains
    )
    items = tuple(item for item, _payload in rendered)
    collisions = id_collisions((item.id, payload) for item, payload in rendered)
    if collisions:
        raise AttackDrawUnsatisfiable(*collisions)

    report = AttackDrawReport(
        repository=dataset.repository,
        revision=dataset.revision,
        declared_splits=tuple(dataset.splits),
        chains=tuple(render_chain(chain) for chain in chains),
        rows_by_split=_count_by_split(rows),
        positives_by_split=_count_by_split(positive_rows),
        blank_positive_rows=blank,
        unique_positives=len(unique_positives),
        removed_by_exclusion=len(filtered.removed),
        surviving_positives=len(survivors),
        drawn_positives=len(drawn),
        items_written=len(items),
        draw=dataset.draw,
    )
    return items, report, dict(filtered.matches_by_source)


def _count_by_split(rows: Iterable[PoolRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
    return dict(sorted(counts.items()))


def serialize(items: Iterable[CorpusItem]) -> str:
    """The exact file content: one JSON object per line, in AD-1's declared order, LF-terminated.

    Sorted by `(source, payload id, chain)`, all three content-derived, so the text depends on
    what was drawn and never on how it was read. The key is `(source, id)` rather than a
    three-tuple because a payload id is a fixed-width hex string followed by `::`, so ordering the
    id lexicographically **is** ordering by payload id and then by chain -- one comparison instead
    of a parse that could disagree with the id it parsed.

    `ensure_ascii=False` keeps the file reviewable -- FR5's whole point is that a reader opens
    `data/` and checks the rows without running anything, and `\\u0301` escapes would defeat that
    for exactly the homoglyph and zero-width payloads this corpus is about. The escaping that
    matters for JSONL is unaffected: `json.dumps` still escapes the line terminators, so no
    payload can inject a second line.
    """
    lines = [
        json.dumps(item.as_json_object(), ensure_ascii=False, separators=(",", ":"))
        for item in sorted(items, key=lambda item: (item.source, item.id))
    ]
    return "".join(f"{line}\n" for line in lines)
