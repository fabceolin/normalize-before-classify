"""The one reader of `results/scores.jsonl`, and the one place a cell of the table is made.

Between the scores file and the arithmetic there was nothing that said **what a number is about**,
and that gap is where this table's central failure lives. A false-positive rate that pools `b_code`
and `b_chat` reads as a small number and hides a layer that is safe on chat and destructive on
code. FR3.1 exists for that one distinction; until this module, nothing enforced it.

**The enforcement is a constructor, not a checker.** `schema.CellKey` refuses a protected axis left
unnamed unless a `Contrast` declares the cell spans it, so a pooled value is not something that
gets flagged after it exists -- it is something that cannot be written down. `pooling_problems`
still ships, because it answers a different question: whether a *set* of cells covers what it
should. The rule itself lives where the value is made.

**One reader, one producer, and the reason is not tidiness.** Every threshold-dependent rate, every
AUC and every discordant pair in the published table is computed here, from the file that is
committed. That is what makes "the committed scores are the scores that produced the numbers" a
property of the code rather than a promise: there is no second path from a model to a cell.
`tests/harness/test_aggregate_bounds.py` refuses any other module under `src/nbc/` from naming the
scores file, and refuses a `.threshold` read anywhere but `pins.py` and here.

**The threshold is applied in exactly one place, and `>=` is the declared convention.** Story 4.2
deliberately committed `p_injection` unrounded and unclassified, because a threshold applied at
write time cannot be changed without re-running eighty-five hours of inference, and one applied in
two places will eventually differ between them. `classify` is that one place.

**Where the dressing axis comes from.** `ItemScore` carries no chain, so it is recovered from the
item id through `matrix.parse_item_id` -- the declared inverse of `matrix.item_id`, which validates
the payload's shape and checks every link against the registries. Not a split on `::` and `+`:
recovering a structure by matching text is a mistake this repository has found in its own history
often enough to have a name for.

**What this module deliberately is not.** It writes nothing -- `results/results.json` is 4-7's, and
`tests/corpus/test_build.py`'s writer scan would fail if this file acquired a write. It computes no
variance and spells no `z`: every interval comes from `harness/stats.py`. It does not argue for ROC
over PR, does not document AUC's saturation and resolution limits, and does not report a sign
disagreement between a delta and the threshold table -- 4-4 owns the summary and its caveats, and
inherits from here only the constraint that a pooled AUC is unconstructible. It evaluates no
falsification condition (4-6), measures no time (4-5), and formats no percentage (5-1).

**Three of the four declared contrasts are emitted here, and the fourth deliberately is not.**
`canon_on_vs_off` keys both a paired-proportion `Delta` and a ΔAUC one, `clean_vs_<chain>` keys a
`Delta`, and `attacks_vs_<benign_class>` keys every `Auc`. `bound_vs_held_out` is declared in `schema.CONTRAST_KINDS` and no cell in this module
carries it: N4 compares a bound recovery interval against a held-out one, which is a comparison
between two `Delta`s that already exist rather than a fifth cell, and inventing one here would put
a column in the table that no requirement asks a reader to read.

**Census counts, and the one that is not available yet.** The scores file supports two: ceiling
hits and window overflow. Per-stage edit counts need `results/traces.jsonl`, which does not exist;
they arrive with 4-7 and are not quietly approximated here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from nbc.corpus.matrix import chain_class, parse_item_id, render_chain
from nbc.errors import NbcError
from nbc.harness.stats import (
    AucSample,
    delta_auc,
    newcombe_paired_interval,
    roc_auc,
    wilson_interval,
)
from nbc.pins import Baseline, Pins
from nbc.schema import (
    AXIS_BENIGN_CLASS,
    CENSUS_CEILING_HIT,
    CENSUS_KINDS,
    CENSUS_WINDOW_OVERFLOW,
    POPULATION_SINGLE_WINDOW,
    AXIS_CANON_ON,
    AXIS_CHAIN_CLASS,
    AXIS_DRESSING_CHAIN,
    AXIS_FAMILY,
    BENIGN_CLASSES,
    CANONICAL,
    CONTRAST_ATTACKS_VS_BENIGN_CLASS,
    CONTRAST_CANON_ON_VS_OFF,
    CONTRAST_CLEAN_VS_CHAIN,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    PROTECTED_AXES,
    Auc,
    CellKey,
    Contrast,
    Count,
    Delta,
    ItemScore,
    PairedCount,
    Rate,
)

__all__ = [
    "CENSUS_CEILING_HIT",
    "CENSUS_KINDS",
    "CENSUS_WINDOW_OVERFLOW",
    "Cell",
    "CellsInvalid",
    "auc_cells",
    "auc_delta_cells",
    "canon_delta_cells",
    "cells",
    "census_cells",
    "chain_delta_cells",
    "classify",
    "false_positive_cells",
    "key_for",
    "pooling_problems",
    "read_scores",
    "recall_cells",
    "windows_matched_delta_cells",
]

Cell = Rate | Count | Auc | Delta
"""The four kinds a published column may be, and the whole of them."""

"""The census counts the scores file supports are `schema.CENSUS_KINDS`, and they are named there
rather than here because `Count` now carries which census it is -- a field, not a convention.

Per-stage edit counts belong in that vocabulary too and are not: they are read off
`results/traces.jsonl`, which story 4-7 writes. Emitting an approximation of them from this file
would put a number in a census column that no trace produced.
"""


class CellsInvalid(NbcError, exit_code=30):
    """The scores file cannot be turned into cells, so no cell is produced.

    Code 30 because 3 through 29 are taken. The inputs that produce it, each with the test that
    fires it:

    - an empty scores file, because a table over no scores is not a table;
    - a line that is not a JSON object, or that is missing a field `ItemScore` requires;
    - two lines carrying one (item, baseline, condition), which the merge promised was a partition;
    - a `baseline_key` that `pins.toml` does not declare, because a column whose axis value came
      out of a file rather than out of a declaration is what AD-20 exists to prevent;
    - a ceiling-hit census asked for under `RAW`, where `ceiling_hit` is `None` by construction and
      a count over nulls would silently be a count over half the population;
    - a paired delta whose two conditions do not cover the same items.
    """


THRESHOLD_COMPARISON: Final[str] = ">="
"""Recorded rather than left to the reader of one line of code.

A score exactly at the threshold is positive. Which way the tie falls changes recall on any item
whose score lands on the boundary, and a run that decided it the other way would publish a
different number under the same name.
"""


def classify(p_injection: float, threshold: float) -> bool:
    """Whether one score counts as a detection. **The only place a threshold is applied.**

    A scan in `tests/harness/test_aggregate_bounds.py` refuses a `.threshold` attribute read from
    any module under `src/nbc/` but `pins.py` and this one, because a second site is a second
    answer waiting to differ from this one on the borderline items the threshold decides.
    """
    return p_injection >= threshold


# --- reading the file ------------------------------------------------------------------------------


def read_scores(path: Path) -> tuple[ItemScore, ...]:
    """Every record of the merged scores file, in file order. **The only reader.**

    Refuses a duplicated (item, baseline, condition) rather than letting the last line win: the
    merge in `harness/score.py` already proved the shard set was a partition, so a duplicate here
    means the file was edited or concatenated after that proof, and picking a winner would publish
    a number chosen by file order.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CellsInvalid(f"the scores file at {path} could not be read: {error}") from error

    scores: list[ItemScore] = []
    seen: dict[tuple[str, str, str], int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CellsInvalid(f"{path}:{number} is blank; the scores file is one record per line")
        try:
            payload = json.loads(line)
        except ValueError as error:
            # `json.JSONDecodeError` is a `ValueError`, and so is what a truncated record raises.
            raise CellsInvalid(f"{path}:{number} is not a JSON object: {error}") from error
        if not isinstance(payload, dict):
            raise CellsInvalid(f"{path}:{number} is a {type(payload).__name__}, not an object")
        try:
            score = ItemScore(**payload)
        except (TypeError, ValueError) as error:
            # TypeError is the missing-or-extra-field shape; ValueError is the bad-value shape.
            raise CellsInvalid(f"{path}:{number} is not an ItemScore: {error}") from error

        key = (score.item_id, score.baseline_key, score.condition)
        if key in seen:
            raise CellsInvalid(
                f"{path}:{number} repeats {key!r}, first seen at line {seen[key]}; the merge "
                f"proved the shard set was a partition, so a duplicate here means the file changed "
                f"after that proof and picking a winner would publish a number chosen by file order"
            )
        seen[key] = number
        scores.append(score)

    if not scores:
        raise CellsInvalid(
            f"the scores file at {path} holds no records; a rate over nothing is not a rate and a "
            f"table over no scores is not a table"
        )
    return tuple(scores)


# --- the cell key -----------------------------------------------------------------------------------


def _declared_baselines(pins: Pins) -> dict[str, Baseline]:
    return {baseline.key: baseline for baseline in pins.baselines}


def _baseline_for(score: ItemScore, pins: Pins) -> Baseline:
    declared = _declared_baselines(pins)
    found = declared.get(score.baseline_key)
    if found is None:
        raise CellsInvalid(
            f"the scores file names baseline {score.baseline_key!r}, which pins.toml does not "
            f"declare; the declared keys are {sorted(declared)}. A column keyed on a baseline "
            f"nobody pinned is a column that came out of a file"
        )
    return found


def key_for(score: ItemScore, pins: Pins) -> CellKey:
    """The cell a score belongs to, with every axis derived rather than assumed.

    `dressing_chain` and `chain_class` come from the id through `parse_item_id` and
    `matrix.chain_class`; `window_policy` and the threshold come from the pin; `canon_on`,
    `family` and `benign_class` are on the record.
    """
    baseline = _baseline_for(score, pins)
    _, chain = parse_item_id(score.item_id)
    return CellKey(
        baseline=baseline.key,
        dressing_chain=render_chain(chain),
        chain_class=chain_class(chain),
        window_policy=baseline.window_policy,
        canon_on=score.condition == CANONICAL,
        family=score.family,
        benign_class=score.benign_class,
    )


def _grouped(scores: Iterable[ItemScore], pins: Pins) -> dict[CellKey, list[ItemScore]]:
    groups: dict[CellKey, list[ItemScore]] = {}
    for score in scores:
        groups.setdefault(key_for(score, pins), []).append(score)
    return groups


def _threshold_for(key: CellKey, pins: Pins) -> float:
    declared = _declared_baselines(pins)
    found = declared.get(str(key.baseline))
    if found is None:
        raise CellsInvalid(f"no pinned baseline named {key.baseline!r}")
    return found.threshold


# --- the four kinds ------------------------------------------------------------------------------------


def recall_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Rate, ...]:
    """Recall on attacks, one `Rate` per key. Never pooled with anything benign."""
    produced: list[Rate] = []
    for key, members in _grouped(scores, pins).items():
        if key.family != FAMILY_ATTACK:
            continue
        threshold = _threshold_for(key, pins)
        detected = sum(1 for score in members if classify(score.p_injection, threshold))
        produced.append(
            Rate(detected, len(members), wilson_interval(detected, len(members)), key)
        )
    return tuple(produced)


def false_positive_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Rate, ...]:
    """The false-positive rate, **one `Rate` per benign class**.

    There is no pooled variant and there is no way to ask for one: the key of a benign cell must
    name its class, so the two classes cannot arrive in one group. The grouping does the enforcing
    -- this function contains no branch that would have to be remembered.
    """
    produced: list[Rate] = []
    for key, members in _grouped(scores, pins).items():
        if key.family != FAMILY_BENIGN:
            continue
        threshold = _threshold_for(key, pins)
        flagged = sum(1 for score in members if classify(score.p_injection, threshold))
        produced.append(Rate(flagged, len(members), wilson_interval(flagged, len(members)), key))
    return tuple(produced)


def census_cells(scores: Sequence[ItemScore], pins: Pins, kind: str) -> tuple[Count, ...]:
    """A census over one key: how many items of it hit the ceiling, or needed a second window.

    Ceiling hits are refused under `RAW`. `ceiling_hit` is `None` there by construction -- the raw
    condition does not run the layer -- and counting nulls as false would report FR10's ceiling
    hits over twice the population they can occur in, which is a wrong number that looks reasonable.
    """
    if kind not in CENSUS_KINDS:
        raise CellsInvalid(f"census kind must be one of {CENSUS_KINDS}, got {kind!r}")

    produced: list[Count] = []
    for key, members in _grouped(scores, pins).items():
        if kind == CENSUS_CEILING_HIT:
            if key.canon_on is not True:
                if any(score.ceiling_hit is not None for score in members):
                    raise CellsInvalid(
                        f"a ceiling-hit census was asked for on a raw cell that carries a "
                        f"ceiling_hit; {key.as_json_object()}"
                    )
                continue
            missing = [score.item_id for score in members if score.ceiling_hit is None]
            if missing:
                raise CellsInvalid(
                    f"a ceiling-hit census over {key.as_json_object()} found {len(missing)} "
                    f"records with no ceiling_hit, first {missing[0]!r}; counting a null as false "
                    f"reports the census over a population it cannot occur in"
                )
            hits = sum(1 for score in members if score.ceiling_hit)
        else:
            hits = sum(1 for score in members if score.n_windows > 1)
        produced.append(Count(hits, len(members), key, census=kind))
    return tuple(produced)


def _contrast_for_chain(chain: Sequence[str]) -> Contrast:
    """`clean_vs_<chain>`, spanning `chain_class` too when the two sides straddle it.

    Whether they do is decided by `matrix.chain_class` against the registries, never by a literal:
    `clean` is bound, so a held-out chain compared against it spans both axes and a bound one spans
    only the dressing.
    """
    spans = {AXIS_DRESSING_CHAIN}
    if chain_class(chain) != chain_class(()):
        spans.add(AXIS_CHAIN_CLASS)
    return Contrast(CONTRAST_CLEAN_VS_CHAIN, render_chain(chain), frozenset(spans))


def auc_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Auc, ...]:
    """Rank separation of attacks against **one** benign class, in the same dressing.

    Pooling is not refused by a check here; it is unreachable. The cell would need a key whose
    `benign_class` is null with nothing spanning it, and `CellKey` refuses that.

    The contrast is `attacks_vs_<benign_class>` and it spans **`family` alone**. The two sides are
    drawn from different halves of the corpus, which is what `family` being null says; the benign
    class is not spanned, because the comparison is against *one* class and which one is part of
    what the cell is about. Story 4.4 corrected that: with the axis null, recovering the class
    meant reading the part of `attacks_vs_b_code` after the underscore, which is a substring where
    a field belongs.
    """
    groups = _grouped(scores, pins)
    produced: list[Auc] = []

    for key, attacks in sorted(groups.items(), key=lambda pair: _sort_key(pair[0])):
        if key.family != FAMILY_ATTACK:
            continue
        for benign_class in BENIGN_CLASSES:
            benign_key = CellKey(
                baseline=key.baseline,
                dressing_chain=key.dressing_chain,
                chain_class=key.chain_class,
                window_policy=key.window_policy,
                canon_on=key.canon_on,
                family=FAMILY_BENIGN,
                benign_class=benign_class,
            )
            benign = groups.get(benign_key)
            if not benign:
                continue
            sample = AucSample(
                positive_ids=tuple(score.item_id for score in attacks),
                positive_scores=tuple(score.p_injection for score in attacks),
                negative_ids=tuple(score.item_id for score in benign),
                negative_scores=tuple(score.p_injection for score in benign),
            )
            estimate = roc_auc(sample)
            contrast = Contrast(
                CONTRAST_ATTACKS_VS_BENIGN_CLASS,
                benign_class,
                frozenset({AXIS_FAMILY}),
            )
            produced.append(
                Auc(
                    value=estimate.auc,
                    interval=estimate.interval,
                    n_positive=estimate.n_positive,
                    n_negative=estimate.n_negative,
                    tied_pairs=estimate.tied_pairs,
                    total_pairs=estimate.total_pairs,
                    key=CellKey(
                        baseline=key.baseline,
                        dressing_chain=key.dressing_chain,
                        chain_class=key.chain_class,
                        window_policy=key.window_policy,
                        canon_on=key.canon_on,
                        family=None,
                        benign_class=benign_class,
                        contrast=contrast,
                    ),
                )
            )
    return tuple(produced)


def canon_delta_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Delta, ...]:
    """The `canon_on_vs_off` difference, from the full 2x2 through Newcombe's paired interval.

    Never a subtraction of the two `Rate`s. The two conditions are measured on the same items, and
    an interval assembled from two independent ones is too wide -- which makes N1 harder to trigger
    and errs toward never declaring a negative result. The 2x2 is built by walking the items, so
    `a` and `d` are present and the method that needs them is the one that runs.
    """
    groups = _grouped(scores, pins)
    produced: list[Delta] = []

    for key, members in sorted(groups.items(), key=lambda pair: _sort_key(pair[0])):
        if key.canon_on is not True:
            continue
        raw_key = CellKey(
            baseline=key.baseline,
            dressing_chain=key.dressing_chain,
            chain_class=key.chain_class,
            window_policy=key.window_policy,
            canon_on=False,
            family=key.family,
            benign_class=key.benign_class,
        )
        raw_members = groups.get(raw_key)
        if raw_members is None:
            continue

        threshold = _threshold_for(key, pins)
        canonical_by_item = {score.item_id: score for score in members}
        raw_by_item = {score.item_id: score for score in raw_members}
        if canonical_by_item.keys() != raw_by_item.keys():
            only = sorted(canonical_by_item.keys() ^ raw_by_item.keys())
            raise CellsInvalid(
                f"the canon-on and canon-off halves of {key.as_json_object()} cover different "
                f"items; {len(only)} differ, first {only[0]!r}. A paired difference over two "
                f"different item sets is a different quantity wearing the same name"
            )

        a = b = c = d = 0
        for item_id, canonical in canonical_by_item.items():
            on = classify(canonical.p_injection, threshold)
            off = classify(raw_by_item[item_id].p_injection, threshold)
            if on and off:
                a += 1
            elif on:
                b += 1
            elif off:
                c += 1
            else:
                d += 1

        counts = PairedCount(a, b, c, d)
        contrast = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))
        produced.append(
            Delta(
                value=counts.theta,
                interval=newcombe_paired_interval(counts),
                key=CellKey(
                    baseline=key.baseline,
                    dressing_chain=key.dressing_chain,
                    chain_class=key.chain_class,
                    window_policy=key.window_policy,
                    canon_on=None,
                    family=key.family,
                    benign_class=key.benign_class,
                    contrast=contrast,
                ),
            )
        )
    return tuple(produced)


def auc_delta_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Delta, ...]:
    """ΔAUC between canon-on and canon-off, over one item set, per benign class.

    Emitted here for the reason `clean_vs_<chain>` is: this module is the only producer of cells,
    so a summary or a verdict that computed its own comparison would become a second one.

    The variance is `stats.delta_auc`'s -- `Var(A) + Var(B) - 2Cov(A, B)` from the structural
    components -- and never the difference of two independent AUC intervals, which would be too
    wide on two conditions measured over the same items and would err toward never declaring
    anything.

    Both conditions must cover the same attacks **and** the same benign items, in the same order;
    `delta_auc` refuses anything else by comparing id tuples, and the ordering here is the file's
    for both sides so the comparison is item-for-item.
    """
    groups = _grouped(scores, pins)
    produced: list[Delta] = []

    for key, attacks_on in sorted(groups.items(), key=lambda pair: _sort_key(pair[0])):
        if key.family != FAMILY_ATTACK or key.canon_on is not True:
            continue
        attacks_off = groups.get(_with(key, canon_on=False))
        if not attacks_off:
            continue

        for benign_class in BENIGN_CLASSES:
            benign_on = groups.get(_benign_key(key, benign_class, canon_on=True))
            benign_off = groups.get(_benign_key(key, benign_class, canon_on=False))
            if not benign_on or not benign_off:
                continue

            on = _sample(attacks_on, benign_on)
            off = _sample(attacks_off, benign_off)
            if on.positive_ids != off.positive_ids or on.negative_ids != off.negative_ids:
                raise CellsInvalid(
                    f"the canon-on and canon-off AUC samples for {key.as_json_object()} against "
                    f"{benign_class!r} do not cover the same items in the same order; a paired "
                    f"difference over two different item sets is a different quantity wearing the "
                    f"same name"
                )

            result = delta_auc(on, off)
            produced.append(
                Delta(
                    value=result.delta,
                    interval=result.interval,
                    key=CellKey(
                        baseline=key.baseline,
                        dressing_chain=key.dressing_chain,
                        chain_class=key.chain_class,
                        window_policy=key.window_policy,
                        canon_on=None,
                        family=None,
                        benign_class=benign_class,
                        contrast=Contrast(
                            CONTRAST_CANON_ON_VS_OFF,
                            None,
                            frozenset({AXIS_CANON_ON, AXIS_FAMILY}),
                        ),
                    ),
                )
            )
    return tuple(produced)


def windows_matched_delta_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Delta, ...]:
    """The canon-on-versus-off delta again, over items occupying **one window under both states**.

    The windowing artifact is a confound with a specific shape: a document over one window is
    scored as the maximum over its windows, so the layer can change a cell's number by changing how
    many windows a document needs rather than by changing what the classifier sees in any of them.
    Restricting to items that occupy exactly one window under *both* canon states removes that
    channel, and the difference between the two cells is how much of the effect ran through it.

    Under **both**, not under either: an item that needs one window raw and two canonical is
    exactly the item the artifact acts through, so admitting it on the strength of one condition
    would leave the confound in the companion that exists to remove it.

    It is the same contrast over a different population, so it carries `population =
    single_window` rather than a new contrast kind -- the two sides do not differ on any axis.
    """
    groups = _grouped(scores, pins)
    produced: list[Delta] = []

    for key, members in sorted(groups.items(), key=lambda pair: _sort_key(pair[0])):
        if key.canon_on is not True:
            continue
        raw_members = groups.get(_with(key, canon_on=False))
        if raw_members is None:
            continue

        threshold = _threshold_for(key, pins)
        canonical_by_item = {score.item_id: score for score in members}
        raw_by_item = {score.item_id: score for score in raw_members}
        eligible = [
            item_id
            for item_id in canonical_by_item
            if item_id in raw_by_item
            and canonical_by_item[item_id].n_windows == 1
            and raw_by_item[item_id].n_windows == 1
        ]
        if not eligible:
            continue

        a = b = c = d = 0
        for item_id in eligible:
            on = classify(canonical_by_item[item_id].p_injection, threshold)
            off = classify(raw_by_item[item_id].p_injection, threshold)
            if on and off:
                a += 1
            elif on:
                b += 1
            elif off:
                c += 1
            else:
                d += 1

        counts = PairedCount(a, b, c, d)
        produced.append(
            Delta(
                value=counts.theta,
                interval=newcombe_paired_interval(counts),
                key=_with(
                    key,
                    canon_on=None,
                    contrast=Contrast(
                        CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON})
                    ),
                    population=POPULATION_SINGLE_WINDOW,
                ),
            )
        )
    return tuple(produced)


def _with(key: CellKey, **overrides: object) -> CellKey:
    """The same key with some axes replaced. One place, because six call sites rebuilt it."""
    fields: dict[str, object] = {
        "baseline": key.baseline,
        "dressing_chain": key.dressing_chain,
        "chain_class": key.chain_class,
        "window_policy": key.window_policy,
        "canon_on": key.canon_on,
        "family": key.family,
        "benign_class": key.benign_class,
        "contrast": key.contrast,
        "population": key.population,
    }
    fields.update(overrides)
    return CellKey(**fields)  # type: ignore[arg-type]


def _benign_key(key: CellKey, benign_class: str, *, canon_on: bool) -> CellKey:
    return _with(key, family=FAMILY_BENIGN, benign_class=benign_class, canon_on=canon_on)


def _sample(attacks: Sequence[ItemScore], benign: Sequence[ItemScore]) -> AucSample:
    return AucSample(
        positive_ids=tuple(score.item_id for score in attacks),
        positive_scores=tuple(score.p_injection for score in attacks),
        negative_ids=tuple(score.item_id for score in benign),
        negative_scores=tuple(score.p_injection for score in benign),
    )


def chain_delta_cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Delta, ...]:
    """`clean_vs_<chain>`: what a dressing costs, paired by payload.

    Emitted here rather than left to 4-6, and the reason is this module's own rule: it is the only
    producer of cells, so a comparison nobody emits is a comparison the condition that reads it
    would have to compute for itself -- which would make the reader of the table a second producer.

    **Paired by payload id, not by item id.** A dressed item and its clean twin are the same
    payload rendered two ways, so they have the same payload id and different chains; pairing on
    the whole item id would find no overlap at all and quietly produce an empty comparison. The
    payload is what `matrix.parse_item_id` returns first, for this reason.

    The contrast spans `chain_class` when the two sides straddle it, decided by
    `matrix.chain_class` against the registries: `clean` is bound, so comparing a held-out chain
    against it crosses the axis, and comparing a bound one does not.
    """
    groups = _grouped(scores, pins)
    clean_name = render_chain(())
    produced: list[Delta] = []

    for key, members in sorted(groups.items(), key=lambda pair: _sort_key(pair[0])):
        if key.dressing_chain == clean_name:
            continue
        clean_key = CellKey(
            baseline=key.baseline,
            dressing_chain=clean_name,
            chain_class=chain_class(()),
            window_policy=key.window_policy,
            canon_on=key.canon_on,
            family=key.family,
            benign_class=key.benign_class,
        )
        clean_members = groups.get(clean_key)
        if not clean_members:
            continue

        threshold = _threshold_for(key, pins)
        dressed_by_payload = {parse_item_id(s.item_id)[0]: s for s in members}
        clean_by_payload = {parse_item_id(s.item_id)[0]: s for s in clean_members}
        shared = dressed_by_payload.keys() & clean_by_payload.keys()
        if not shared:
            continue

        a = b = c = d = 0
        for payload in shared:
            on_clean = classify(clean_by_payload[payload].p_injection, threshold)
            on_dressed = classify(dressed_by_payload[payload].p_injection, threshold)
            if on_clean and on_dressed:
                a += 1
            elif on_clean:
                b += 1
            elif on_dressed:
                c += 1
            else:
                d += 1

        counts = PairedCount(a, b, c, d)
        chain = parse_item_id(next(iter(members)).item_id)[1]
        contrast = _contrast_for_chain(chain)
        produced.append(
            Delta(
                value=counts.theta,
                interval=newcombe_paired_interval(counts),
                key=CellKey(
                    baseline=key.baseline,
                    dressing_chain=None,
                    chain_class=None if AXIS_CHAIN_CLASS in contrast.spans else key.chain_class,
                    window_policy=key.window_policy,
                    canon_on=key.canon_on,
                    family=key.family,
                    benign_class=key.benign_class,
                    contrast=contrast,
                ),
            )
        )
    return tuple(produced)


def cells(scores: Sequence[ItemScore], pins: Pins) -> tuple[Cell, ...]:
    """Every cell the scores file supports. **The only producer.**

    Ordered by key so two runs over one file emit the same sequence: the results file 4-7 writes is
    compared byte for byte by a reader recomputing the table, and an order that depended on dict
    insertion would make that comparison fail for no reason anybody could see.
    """
    produced: list[Cell] = [
        *recall_cells(scores, pins),
        *false_positive_cells(scores, pins),
        *census_cells(scores, pins, CENSUS_CEILING_HIT),
        *census_cells(scores, pins, CENSUS_WINDOW_OVERFLOW),
        *auc_cells(scores, pins),
        *canon_delta_cells(scores, pins),
        *chain_delta_cells(scores, pins),
        *auc_delta_cells(scores, pins),
        *windows_matched_delta_cells(scores, pins),
    ]
    return tuple(sorted(produced, key=lambda cell: (_kind_of(cell), _sort_key(cell.key))))


def _kind_of(cell: Cell) -> str:
    return str(cell.as_json_object()["kind"])


def _sort_key(key: CellKey) -> tuple[str, ...]:
    contrast = "" if key.contrast is None else key.contrast.name
    return (
        str(key.population),
        str(key.baseline),
        str(key.dressing_chain),
        str(key.chain_class),
        str(key.window_policy),
        str(key.canon_on),
        str(key.family),
        str(key.benign_class),
        contrast,
    )


# --- the set-level check ---------------------------------------------------------------------------


def pooling_problems(produced: Iterable[Cell]) -> tuple[str, ...]:
    """What is wrong with a *set* of cells, which the constructor cannot see.

    `CellKey` already makes a pooled value unwritable. This answers the other question: given the
    cells that exist, does any protected axis read null without a contrast that spans it, and does
    any benign key appear for one class and not the other -- which is how a class silently vanishes
    from the table rather than being averaged into it.
    """
    problems: list[str] = []
    benign_seen: dict[tuple[str, ...], set[str]] = {}

    for cell in produced:
        key = cell.key
        spans = frozenset() if key.contrast is None else key.contrast.spans
        for axis in PROTECTED_AXES:
            if getattr(key, axis) is not None or axis in spans:
                continue
            if axis == AXIS_BENIGN_CLASS and key.family != FAMILY_BENIGN:
                # An attack cell has no benign class and a family-spanned cell has none to name.
                # This is the pair `CellKey` checks at construction, and a checker that read the
                # axis alone would report every recall cell in the table as pooled -- which is the
                # exact ambiguity `family` was put on the key to resolve.
                continue
            problems.append(
                f"{_kind_of(cell)} cell {key.as_json_object()} leaves {axis} null with no "
                f"contrast spanning it, which is aggregation across a protected axis"
            )
        if key.family == FAMILY_BENIGN and key.benign_class is not None:
            without_class = _sort_key(key)[:6]
            benign_seen.setdefault(without_class, set()).add(key.benign_class)

    for without_class, classes in sorted(benign_seen.items()):
        missing = set(BENIGN_CLASSES) - classes
        if missing:
            problems.append(
                f"the benign cells at {without_class} cover {sorted(classes)} and not "
                f"{sorted(missing)}; a class that is absent is as unreadable as one that was pooled"
            )

    return tuple(problems)
