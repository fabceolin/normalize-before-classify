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

**Every declared chain, once per drawn payload, over both registries.** The chains come from
`corpus/matrix.py::CHAINS` and `corpus/matrix.py::HELDOUT_CHAINS`, keyed on the attack family, and
the text from `dressings.dress_declared`, which is AD-3's `reduce(apply, chain, payload)` over the
union of the two dressing registries. So one drawn payload becomes as many corpus rows as there are
bound chains plus held-out chains, each with its own item id, and the dressing axis of the headline
table is those constants rather than a list any caller assembled.

The held-out half is **built here rather than only declared**: AD-28's abort list names an empty
held-out block as the specific failure it exists to prevent, and a registry nothing renders from
leaves that block empty while satisfying every word about registries. The two axes travel
separately on the report (`chains`, `held_out_chains`) because `chain_class` is part of the cell key
(AD-2) and no function aggregates across it (AD-11).

`matrix.validate(CHAINS, DRESSINGS)` and `heldout.validate_heldout()` both run before the first row
is rendered, so a chain naming a dressing nothing implements aborts instead of producing a column
that silently does not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from nbc.corpus.draw import take
from nbc.corpus.dressings import DRESSINGS, dress_declared
from nbc.corpus.exclusion import ExclusionIndex, filter_rows
from nbc.corpus.heldout import validate_heldout
from nbc.corpus.roundtrip import payloads_below_decode_floor
from nbc.corpus.matrix import (
    CHAINS,
    CLEAN_CHAIN,
    HELDOUT_CHAINS,
    id_collisions,
    item_id,
    payload_id,
    render_chain,
    validate as validate_matrix,
)
from nbc.errors import NbcError
from nbc.pins import DRAW_HEAD, DRAW_SEEDED_RANDOM, AttackDataset, AttackDraw
from nbc.schema import ATTACK, FAMILY_ATTACK, CorpusItem

__all__ = [
    "AttackDrawReport",
    "AttackDrawUnsatisfiable",
    "LabelContradiction",
    "PoolRow",
    "contradictions",
    "draw_attack_items",
    "render_attack_item",
    "select_payloads",
    "serialize",
    "verify_splits",
]

class LabelContradiction(NbcError, exit_code=16):
    """The pinned dataset carries one text under both labels, and nothing here can say which wins.

    Code 16 because 3 through 15 are taken. This is a **gold-label error in the source**, and it
    is the reason the build stops rather than choosing: exactly one of the two rows is wrong, the
    builder has no evidence about which, and a builder that silently picks one has an unreviewed
    annotation policy -- the precise thing FR4 claims this repository does not have. Two rows in
    ten thousand would not move a rate, and that is not the argument.

    It is a sibling of `crosscheck.BenignItemMislabelled` rather than of `ExclusionSetUnusable`.
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


def select_payloads(payloads: Iterable[str], draw: AttackDraw) -> tuple[str, ...]:
    """The declared draw, taken by `corpus/draw.py`, which the benign frame draws through too.

    Both halves of the corpus take their draw with one implementation, so FR5.1's "the same
    selection-method vocabulary as the attack draw" is a shared function rather than two that agree
    today. What stays here is the abort: a method nothing implements is this half's diagnosis, with
    this half's exit code.

    A pool at or below the declared size is taken whole -- but nothing there decides whether that
    is acceptable. The floor is `AttackDrawUnsatisfiable`'s, raised by `draw_attack_items`, so the
    rule lives in one place instead of being half-enforced by a silent truncation.
    """
    drawn = take(
        payloads,
        draw.sample_size_positives,
        method=draw.method,
        seed=draw.seed,
        sort_key=draw.sort_key,
    )
    if drawn is None:
        # `load_pins` refuses any other value, so reaching here means a `Pins` was assembled in
        # code with a method nothing implements. Silently drawing something would publish a corpus
        # under a rule that does not exist.
        raise AttackDrawUnsatisfiable(
            f"the draw declares selection method {draw.method!r}, which nothing implements; the "
            f"admissible methods are {DRAW_HEAD!r} and {DRAW_SEEDED_RANDOM!r}"
        )
    return drawn


def render_attack_item(
    payload: str, *, source: str, chain: Sequence[str] = CLEAN_CHAIN
) -> CorpusItem:
    """One corpus row: the rendered text and the gold label, produced by one constructor call.

    The text is `dressings.dress_declared(payload, chain)`, which is AD-3's
    `reduce(apply, chain, payload)` over **both** registries -- so `("base64", "homoglyph")`
    produces `homoglyph(to_base64(payload))`, the later link wrapping the earlier, and
    `("base32",)` produces a held-out row. The empty chain returns the payload itself, which is
    what makes `clean` the identity element of the fold rather than a dressing that happens to
    change nothing.

    The union rather than `DRESSINGS`, because AD-28 requires the held-out block to be **built**:
    a registry nothing renders from leaves the held-out column of the table empty, which is the
    specific failure AD-28's abort list names. `chain_class` (AD-2) tells the two halves apart
    downstream, and it reads the registries rather than this call.

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
        text=dress_declared(payload, chain),
        label=ATTACK,
    )


@dataclass(frozen=True, slots=True)
class AttackDrawReport:
    """What the draw did, in numbers a reader can check against the pinned declaration.

    Every count is in **attack positives** except `rows_by_split` and `items_written`, which are
    in rows and are named so. Both units are published because the pair is the error this project
    keeps making: the pool holds three times as many rows as positives, and a size read in the
    wrong unit is off by that much. A third unit now sits under those two -- one drawn positive
    becomes one row **per chain**, over both registries -- so `chains` and `held_out_chains` travel
    beside the counts and `items_written == drawn_positives * (len(chains) + len(held_out_chains))`
    is checkable by a reader of the report rather than only by a reader of the code.

    The two chain lists are **separate fields rather than one concatenation**, which is AD-2 and
    AD-11: `chain_class` is part of the cell key and no function aggregates across it, so a report
    that published one merged axis would have handed Epic 4 a list it could only split by looking
    the names up again somewhere else.
    """

    repository: str
    revision: str
    declared_splits: tuple[str, ...]
    chains: tuple[str, ...]
    held_out_chains: tuple[str, ...]
    rows_by_split: Mapping[str, int]
    positives_by_split: Mapping[str, int]
    blank_positive_rows: int
    payloads_below_decode_floor: int
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
                # AD-28's block, published as its own axis: `report/` renders it under its own
                # heading and N4 quantifies over it, so a held-out chain that reached the results
                # without appearing here would be a cell computed over rows nobody declared.
                "held_out_chains": list(self.held_out_chains),
                "rows_by_split": dict(sorted(self.rows_by_split.items())),
                "positives_by_split": dict(sorted(self.positives_by_split.items())),
                # Counted and published rather than dropped in silence. The 3071-versus-3073
                # discrepancy this project already shipped came from a truncation nobody
                # reported, and it was found by rederiving a number rather than by reading one.
                "blank_positive_rows": self.blank_positive_rows,
                # Story 3.4's round-trip contract exempts a payload the layer declines by its own
                # published candidate floor -- too short, or too repetitive, for `decode.py` to
                # open. The exemption is structural and narrow, and it is counted here because an
                # exemption nobody counts is an exemption nobody can size: these payloads carry
                # rows on every encoded chain that no ceiling and no character mapping will
                # recover, and a reader dividing this by `drawn_positives` knows how much of the
                # encoded columns is unrecoverable before any classifier ran.
                "payloads_below_decode_floor": self.payloads_below_decode_floor,
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
) -> tuple[
    tuple[CorpusItem, ...], tuple[str, ...], AttackDrawReport, Mapping[str, int]
]:
    """The whole offline pipeline: gate, filter, draw, render.

    Returns items, **the drawn payloads in their undressed clean form**, the report, and the
    per-source removal counts.

    Aborts before returning anything, so no caller can write a corpus assembled from a pool that
    failed a gate.

    **Why the payloads are returned rather than recovered downstream.** AD-27's benign cross-check
    compares undressed benign sources against undressed attack payloads, and
    `benign.draw_benign_items` therefore needs exactly this tuple. The alternative -- reading the
    payloads back off the rendered items whose chain is `clean` -- would make the gate depend on
    `CLEAN_CHAIN` staying in `CHAINS[FAMILY_ATTACK]`, so removing that one entry from a constant
    would empty the cross-check's payload set without failing anything that names it. They are also
    deliberately **not** on `AttackDrawReport`: that record is counts, and it is serialized into
    the corpus manifest, where twelve hundred payload texts do not belong.

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
    validate_matrix(CHAINS, DRESSINGS)
    validate_heldout()
    chains = tuple(tuple(chain) for chain in CHAINS[FAMILY_ATTACK])
    held_out = tuple(tuple(chain) for chain in HELDOUT_CHAINS[FAMILY_ATTACK])
    # Both registries, in a declared order: bound first, held out after, so the file's row order
    # stays a property of the sort in `serialize` rather than of this concatenation.
    every_chain = chains + held_out

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
        for chain in every_chain
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
        held_out_chains=tuple(render_chain(chain) for chain in held_out),
        rows_by_split=_count_by_split(rows),
        positives_by_split=_count_by_split(positive_rows),
        blank_positive_rows=blank,
        # Over the **bound** chains only, and deliberately: the exemption is story 3.4's, whose
        # contract is scoped to the bound registry. A held-out chain is not exempt from the round
        # trip -- it is outside it -- so counting it here would inflate a number whose whole job
        # is to size how much of the *bound* encoded columns is structurally unrecoverable.
        payloads_below_decode_floor=len(payloads_below_decode_floor(drawn, list(chains))),
        unique_positives=len(unique_positives),
        removed_by_exclusion=len(filtered.removed),
        surviving_positives=len(survivors),
        drawn_positives=len(drawn),
        items_written=len(items),
        draw=dataset.draw,
    )
    return items, tuple(drawn), report, dict(filtered.matches_by_source)


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
