"""The four falsification conditions, evaluated by the run rather than by the author.

This artifact's whole claim is that it could have failed. Four conditions say what failure would
look like, and a README section listing conditions the author checked by eye is decoration: checked
when the author feels like it, against numbers they have already seen, and a reader has no way to
tell a condition that did not fire from one nobody evaluated.

**The asymmetry between the two failure modes is the design.** A **triggered** condition never
aborts -- a negative result is publishable and this artifact commits to publishing it. A
**`not_evaluable`** condition *does* abort, naming the missing input, because a condition that
silently reads as un-triggered when its input is absent is exactly how the section becomes
decorative. `verdicts` produces all four and aborts on none; `refuse_an_unevaluable_run` is the
gate, separate so both paths are reachable and both are tests.

**Every verdict carries a minimum detectable effect**, because `not_triggered` is the outcome most
likely to be misread: "N3 did not trigger" reads as "the layer costs nothing" when it may mean
"this corpus could not have detected a cost below X". It is computed from the deciding interval's
half-width rather than asserted.

**N1 is decided on one pre-registered cell, and the pre-registration is real.** `pins.toml` declares
`[benign_frame.confirmatory_cell]` and it is hashed into `frame_id` and thence into `build_id`, so
the cell cannot have been chosen after the numbers existed. The cell is *read* from there and never
selected from the cells that exist; a run whose cells do not contain it is `not_evaluable`, which is
the honest outcome and not a reason to pick another.

Every other cell is **exploratory**: reported, never the verdict, and counted. The count `m` is
recorded rather than corrected for, because applying a multiplicity correction would mean choosing
one and the choice would be ours rather than the reader's.

**This module computes no comparison.** `harness/aggregate.py` is the only producer of cells, so
the two things this story needed and did not have -- the windows-matched companion, and the
difference of two independent estimates -- were added there and in `harness/stats.py` rather than
here. It also does not re-derive `harness/summary.py`'s limits: saturation, resolution, sign
disagreement, the bound-chain definitional effect and the windowing divergence are findings, and a
verdict that restated one would give the results file two ways to say the same thing.

**It writes no file.** `results.json` is 4-7's.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from nbc.corpus.heldout import PROBE_NONE, probes_for
from nbc.corpus.matrix import (
    CHAIN_CLASS_BOUND,
    CHAIN_CLASS_HELD_OUT,
    encoding_depth,
    parse_chain,
)
from nbc.errors import NbcError
from nbc.harness.stats import mover_difference_interval
from nbc.schema import (
    CENSUS_CEILING_HIT,
    CONTRAST_CANON_ON_VS_OFF,
    CONTRAST_CLEAN_VS_CHAIN,
    FALSIFICATION_CONDITIONS,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    NEWCOMBE_PAIRED,
    OUTCOME_NOT_EVALUABLE,
    OUTCOME_NOT_TRIGGERED,
    OUTCOME_TRIGGERED,
    POPULATION_ALL,
    CellKey,
    Count,
    Delta,
    Verdict,
)

__all__ = [
    "N3_ABSOLUTE_CEILING_NS",
    "N3_INFERENCE_SHARE",
    "VerdictNotEvaluable",
    "evaluate_n1",
    "evaluate_n2",
    "evaluate_n3",
    "evaluate_n4",
    "refuse_an_unevaluable_run",
    "verdicts",
]

N3_INFERENCE_SHARE: Final[float] = 0.10
N3_ABSOLUTE_CEILING_NS: Final[int] = 1_000_000
"""N3's two ceilings, and `min` is the operator between them rather than "whichever comes first".

The share says the layer must not be a material fraction of what it sits in front of. The absolute
says it must not be slow in human terms even if inference is slower still. `min` means a baseline
being slow cannot buy the layer more budget, which "whichever comes first" would leave ambiguous.
"""


class VerdictNotEvaluable(NbcError, exit_code=33):
    """A falsification condition had no input, so the run stops rather than publishing a table.

    Code 33 because 3 through 32 are taken.

    It is the **only** abort this module raises, and a `triggered` condition deliberately does not
    produce one: a negative result is the artifact working and gets published. The danger this
    guards is the opposite one -- an unevaluable condition reading as an un-triggered condition, so
    that a reader sees four conditions and cannot tell "checked and did not fire" from "never
    checked". The message names the missing input rather than the condition alone.
    """


def _includes_zero(delta: Delta) -> bool:
    return delta.interval.lo <= 0.0 <= delta.interval.hi


def _half_width(delta: Delta) -> float:
    """The minimum detectable effect for a cell, taken from its own interval.

    The smallest effect whose interval would have excluded zero is one about a half-width away from
    it, so the half-width is what "an effect this size would have been visible" means here. Derived
    from the cell rather than declared, so it stays true when `n` changes.
    """
    return delta.interval.width / 2


def _canon_deltas(cells: Iterable[object]) -> tuple[Delta, ...]:
    """Every published canon-on-versus-off proportion delta.

    `population == all` excludes the windows-matched companion, which shares a column and is a
    different item set: admitting it would let a companion stand in for the cell it companions.
    The method excludes the AUC delta, which is a different quantity.
    """
    return tuple(
        cell
        for cell in cells
        if isinstance(cell, Delta)
        and cell.contrast.kind == CONTRAST_CANON_ON_VS_OFF
        and cell.interval.method == NEWCOMBE_PAIRED
        and cell.key.population == POPULATION_ALL
    )


def _matches(key: CellKey, *, baseline: str, chain: str) -> bool:
    return key.baseline == baseline and key.dressing_chain == chain


def evaluate_n1(cells: Sequence[object], confirmatory) -> Verdict:  # type: ignore[no-untyped-def]
    """N1, on the pre-registered cell alone: does the cost exceed the recovery?

    `D = ΔFPR − Δrecall`, and it triggers iff `D`'s interval lies **wholly above zero** -- the layer
    costs more in false positives on the declared benign class than it recovers in recall on the
    same chain. Recovery is the canon-on-minus-canon-off delta on **that same chain**, never a
    comparison against `clean`: comparing a dressed chain against clean answers "what did the
    dressing cost", which is N2's question.

    The two deltas are combined through `stats.mover_difference_interval`, and the combination is
    valid because they are computed over **disjoint item sets** -- benign items of one class and
    attack items. That disjointness is what independence means here and it is why this is the one
    place in the project that combines two intervals as if uncorrelated.
    """
    wanted = (
        f"baseline={confirmatory.baseline!r} chain={confirmatory.dressing_chain!r} "
        f"benign_class={confirmatory.benign_class!r}"
    )
    canon = _canon_deltas(cells)

    recall = [
        delta
        for delta in canon
        if delta.key.family == FAMILY_ATTACK
        and _matches(
            delta.key, baseline=confirmatory.baseline, chain=confirmatory.dressing_chain
        )
    ]
    false_positive = [
        delta
        for delta in canon
        if delta.key.family == FAMILY_BENIGN
        and delta.key.benign_class == confirmatory.benign_class
        and _matches(
            delta.key, baseline=confirmatory.baseline, chain=confirmatory.dressing_chain
        )
    ]

    # Every canon delta that is not one of the two the declaration names. Counted, not corrected:
    # a reader applies whichever multiplicity correction they trust to a number they can see.
    exploratory = tuple(
        delta.key for delta in canon if delta not in recall and delta not in false_positive
    )

    if not recall or not false_positive:
        missing = []
        if not recall:
            missing.append("the attack recall delta")
        if not false_positive:
            missing.append("the false-positive delta")
        return Verdict(
            condition="N1",
            outcome=OUTCOME_NOT_EVALUABLE,
            keys=exploratory,
            reason=(
                f"the confirmatory cell declared in pins.toml on {confirmatory.declared_on} "
                f"({wanted}) is missing {' and '.join(missing)} from this run's cells. N1 is "
                f"decided on that cell alone and is not re-pointed at another"
            ),
            computed={
                "confirmatory_cell": confirmatory.as_run_fields(),
                "exploratory_cells_scanned": len(exploratory),
                "minimum_detectable_effect": None,
            },
        )

    (recall_delta,) = recall
    (fpr_delta,) = false_positive
    interval = mover_difference_interval(
        fpr_delta.value, fpr_delta.interval, recall_delta.value, recall_delta.interval
    )
    triggered = interval.lo > 0.0

    return Verdict(
        condition="N1",
        outcome=OUTCOME_TRIGGERED if triggered else OUTCOME_NOT_TRIGGERED,
        keys=(fpr_delta.key, recall_delta.key),
        reason=(
            f"D = ΔFPR − Δrecall on the pre-registered cell ({wanted}) is "
            f"{fpr_delta.value:+.6f} − {recall_delta.value:+.6f} = "
            f"{fpr_delta.value - recall_delta.value:+.6f}, interval "
            f"[{interval.lo:+.6f}, {interval.hi:+.6f}]. "
            + (
                "It lies wholly above zero: the layer costs more on this benign class than it "
                "recovers on this chain."
                if triggered
                else "It includes zero, so this cell does not show a cost exceeding the recovery."
            )
            + f" {len(exploratory)} other canon deltas were scanned and are exploratory: a "
            f"triggered one among them is reported as exploratory and is never the verdict."
        ),
        computed={
            "confirmatory_cell": confirmatory.as_run_fields(),
            "delta_false_positive": fpr_delta.value,
            "delta_false_positive_interval": fpr_delta.interval.as_json_object(),
            "delta_recall": recall_delta.value,
            "delta_recall_interval": recall_delta.interval.as_json_object(),
            "difference": fpr_delta.value - recall_delta.value,
            "difference_interval": interval.as_json_object(),
            "exploratory_cells_scanned": len(exploratory),
            "minimum_detectable_effect": interval.width / 2,
        },
    )


def evaluate_n2(cells: Sequence[object]) -> Verdict:
    """N2: are the dressings actually defeating the classifier before the layer runs?

    Triggers iff, for **every** baseline and **every** encoded bound chain, the `clean_vs_<chain>`
    recall delta at canon-off has an interval including zero -- which would mean the dressings never
    hurt in the first place and the experiment has no degradation to recover. The verdict **names
    the pair** that kept it from triggering, because "not every pair" is unactionable.

    "Encoded" is `matrix.encoding_depth(chain) > 0`, derived rather than listed: a chain of
    homoglyphs costs the layer no decode budget and is not what this condition is about.
    """
    pairs = [
        cell
        for cell in cells
        if isinstance(cell, Delta)
        and cell.contrast.kind == CONTRAST_CLEAN_VS_CHAIN
        and cell.key.family == FAMILY_ATTACK
        and cell.key.canon_on is False
        and cell.key.chain_class == CHAIN_CLASS_BOUND
        and encoding_depth(_links(cell.contrast.argument)) > 0
    ]

    if not pairs:
        return Verdict(
            condition="N2",
            outcome=OUTCOME_NOT_EVALUABLE,
            keys=(),
            reason=(
                "this run produced no clean-versus-chain recall delta at canon-off on an encoded "
                "bound chain, so there is nothing to ask whether the dressings degraded"
            ),
            computed={"pairs_examined": 0, "minimum_detectable_effect": None},
        )

    kept_out = [cell for cell in pairs if not _includes_zero(cell)]
    triggered = not kept_out
    widest = max(_half_width(cell) for cell in pairs)

    if triggered:
        reason = (
            f"every one of the {len(pairs)} (baseline, encoded bound chain) pairs has a "
            f"clean-versus-chain recall interval including zero at canon-off: the dressings did "
            f"not degrade the classifier, so there is no degradation for the layer to recover."
        )
    else:
        named = ", ".join(
            f"{cell.key.baseline}/{cell.contrast.argument} "
            f"[{cell.interval.lo:+.6f}, {cell.interval.hi:+.6f}]"
            for cell in sorted(kept_out, key=lambda c: (str(c.key.baseline), str(c.contrast.argument)))
        )
        reason = (
            f"{len(kept_out)} of {len(pairs)} pairs have an interval excluding zero, so the "
            f"dressings did degrade the classifier somewhere. The pairs that kept this from "
            f"triggering: {named}"
        )

    return Verdict(
        condition="N2",
        outcome=OUTCOME_TRIGGERED if triggered else OUTCOME_NOT_TRIGGERED,
        keys=tuple(cell.key for cell in pairs),
        reason=reason,
        computed={
            "pairs_examined": len(pairs),
            "pairs_excluding_zero": [
                {
                    "baseline": cell.key.baseline,
                    "chain": cell.contrast.argument,
                    "value": cell.value,
                    "interval": cell.interval.as_json_object(),
                }
                for cell in kept_out
            ],
            "minimum_detectable_effect": widest,
        },
    )


def _chains_that_hit_the_ceiling(cells: Iterable[object]) -> frozenset[str]:
    """The chains whose documents ran past the recursion budget, from the run's own census.

    `census == ceiling_hit` is the filter, and it is load-bearing. Story 4.3 emitted two censuses
    with the same type and the same key, and this function found that they were indistinguishable
    on the record: without the field, a chain with one multi-window document would have been read
    as over-ceiling and counted into N4's generalization set. `Count.census` was added for this.

    The ceiling census is emitted only for canon-on keys, where `ceiling_hit` is not `None`. A
    chain appearing with `k > 0` reached the ceiling under the ceiling this run used.
    """
    return frozenset(
        str(cell.key.dressing_chain)
        for cell in cells
        if isinstance(cell, Count)
        and cell.census == CENSUS_CEILING_HIT
        and cell.k > 0
        and cell.key.dressing_chain is not None
    )


def _links(chain_name: str | None) -> tuple[str, ...]:
    """The links of a rendered chain name, through `matrix.parse_chain`.

    Not `name.split("+")`: the parse validates against the declared registries, so a chain nobody
    declared is refused here rather than silently classified as bound and counted into N2 or N4.
    """
    if not chain_name:
        return ()
    return parse_chain(chain_name)


def evaluate_n3(timing) -> Verdict:  # type: ignore[no-untyped-def]
    """N3: is the layer a material share of what it sits in front of?

    Triggers iff `layer_p95 > min(0.10 × the fastest baseline's p95 inference, 1_000_000 ns)`.
    `min` is the operator: a slow baseline cannot buy the layer more budget.

    "Fastest" is the smallest inference p95, because that is the baseline against which the layer's
    share is largest -- the conservative reading, and the one the criterion names.
    """
    if timing is None:
        return Verdict(
            condition="N3",
            outcome=OUTCOME_NOT_EVALUABLE,
            keys=(),
            reason=(
                "no timing report was produced, so N3 has no right-hand side. The dedicated pass "
                "in harness/timing.py is what supplies it and it is mandatory for this reason"
            ),
            computed={"minimum_detectable_effect": None},
        )

    inference = timing.inference.by_baseline
    if not inference:
        return Verdict(
            condition="N3",
            outcome=OUTCOME_NOT_EVALUABLE,
            keys=(),
            reason="the timing report carries no baseline latency, so N3 has no right-hand side",
            computed={"minimum_detectable_effect": None},
        )

    fastest_key = min(inference, key=lambda key: inference[key].p95)
    fastest_p95 = inference[fastest_key].p95
    share_ceiling = N3_INFERENCE_SHARE * fastest_p95
    ceiling = min(share_ceiling, float(N3_ABSOLUTE_CEILING_NS))
    layer_p95 = timing.layer.overall.p95
    triggered = layer_p95 > ceiling

    binding = "the share of inference" if share_ceiling <= N3_ABSOLUTE_CEILING_NS else "the absolute"

    return Verdict(
        condition="N3",
        outcome=OUTCOME_TRIGGERED if triggered else OUTCOME_NOT_TRIGGERED,
        keys=(),
        reason=(
            f"the layer's p95 is {layer_p95} ns against a ceiling of {ceiling:.1f} ns, which is "
            f"min({N3_INFERENCE_SHARE:g} x {fastest_p95} ns for the fastest baseline "
            f"{fastest_key!r}, {N3_ABSOLUTE_CEILING_NS} ns) -- {binding} ceiling is the binding "
            f"one here. "
            + (
                "The layer exceeds it."
                if triggered
                else "The layer is within it; a cost above this ceiling would have been detected."
            )
        ),
        computed={
            "layer_p95_ns": layer_p95,
            "fastest_baseline": fastest_key,
            "fastest_baseline_p95_ns": fastest_p95,
            "share_ceiling_ns": share_ceiling,
            "absolute_ceiling_ns": N3_ABSOLUTE_CEILING_NS,
            "ceiling_ns": ceiling,
            "binding_ceiling": binding,
            # The smallest layer p95 that would have triggered: one nanosecond past the ceiling.
            "minimum_detectable_effect": max(0.0, ceiling - layer_p95),
        },
    )


def evaluate_n4(cells: Sequence[object]) -> Verdict:
    """N4: does the layer only recover what it was built against?

    Triggers iff, for **every** held-out chain the layer can engage (`probes != none`) **and** the
    over-ceiling chain, the recovery interval includes zero, **while** the bound-chain recovery
    interval excludes zero and lies above it. That pairing is the whole condition: a layer that
    recovers nothing anywhere has not generalized badly, it has not worked, and the bound half is
    what tells the two apart.

    A `probes: none` chain is **excluded from the trigger and reported anyway**. `rot13` gives the
    layer nothing to engage -- no alphabet marker, no entropy signature -- so no recovery is
    expected there and counting it would make N4 trigger for a reason that is not generalization
    failure.

    The over-ceiling chain is **read off the run's own ceiling-hit census**, not re-derived from
    `matrix.encoding_depth` against the declared default. Two reasons, and the second is the one
    that matters. `tests/canon/test_recursion.py` holds `DEFAULT_CEILING` to a single reader,
    because a second module applying it is a second place it can be set -- and this module reading
    it would have been exactly that. More importantly, the census records the ceiling the run
    ACTUALLY applied to those documents, so a run under a non-default ceiling classifies its chains
    the way it scored them rather than the way the default says it should have.

    `census_cells(..., CENSUS_CEILING_HIT)` is a `Count` per key, and `k > 0` is the test. Since
    hitting the ceiling is a property of the chain rather than of a document, every item of a chain
    agrees and `k` is 0 or `n`; `k > 0` is written rather than `k == n` because the weaker form is
    the one that stays correct if that ever stops holding.
    """
    canon = _canon_deltas(cells)
    recovery = [delta for delta in canon if delta.key.family == FAMILY_ATTACK]
    over_ceiling_chains = _chains_that_hit_the_ceiling(cells)

    held_out: list[Delta] = []
    excluded: list[Delta] = []
    over_ceiling: list[Delta] = []
    bound: list[Delta] = []
    for delta in recovery:
        links = _links(delta.key.dressing_chain)
        if delta.key.chain_class == CHAIN_CLASS_HELD_OUT:
            if probes_for(links) == PROBE_NONE:
                excluded.append(delta)
            else:
                held_out.append(delta)
        elif delta.key.dressing_chain in over_ceiling_chains:
            over_ceiling.append(delta)
        elif encoding_depth(links) > 0:
            bound.append(delta)

    generalization_set = held_out + over_ceiling
    if not generalization_set or not bound:
        missing = []
        if not generalization_set:
            missing.append("a held-out or over-ceiling recovery delta")
        if not bound:
            missing.append("a bound-chain recovery delta")
        return Verdict(
            condition="N4",
            outcome=OUTCOME_NOT_EVALUABLE,
            keys=tuple(delta.key for delta in recovery),
            reason=(
                f"N4 compares recovery on chains the layer was not built against with recovery on "
                f"chains it was, and this run has no {' and no '.join(missing)}"
            ),
            computed={
                "excluded_probes_none": [d.key.dressing_chain for d in excluded],
                "minimum_detectable_effect": None,
            },
        )

    recovering = [delta for delta in generalization_set if not _includes_zero(delta)]
    bound_recovers = [
        delta for delta in bound if not _includes_zero(delta) and delta.interval.lo > 0.0
    ]
    triggered = not recovering and bool(bound_recovers)

    if triggered:
        reason = (
            f"every one of the {len(generalization_set)} held-out and over-ceiling chains has a "
            f"recovery interval including zero, while {len(bound_recovers)} of {len(bound)} bound "
            f"chains recover with an interval above zero. The layer recovers what it was built "
            f"against and does not generalize."
        )
    elif recovering:
        named = ", ".join(
            sorted(str(delta.key.dressing_chain) for delta in recovering)
        )
        reason = (
            f"{len(recovering)} of {len(generalization_set)} held-out or over-ceiling chains "
            f"recover with an interval excluding zero, so the layer is not confined to what it was "
            f"built against. The chains that kept this from triggering: {named}"
        )
    else:
        reason = (
            f"no bound chain recovers with an interval above zero, so there is nothing for the "
            f"held-out result to be worse than: a layer that recovers nowhere has not generalized "
            f"badly, it has not worked."
        )

    return Verdict(
        condition="N4",
        outcome=OUTCOME_TRIGGERED if triggered else OUTCOME_NOT_TRIGGERED,
        keys=tuple(delta.key for delta in generalization_set + bound),
        reason=reason
        + (
            f" Excluded from the trigger and reported anyway: "
            f"{sorted(str(d.key.dressing_chain) for d in excluded)}, whose held-out encoding "
            f"declares probes = {PROBE_NONE!r} -- the layer has nothing to engage there, so no "
            f"recovery is expected and counting it would make N4 trigger for a reason that is not "
            f"generalization failure."
            if excluded
            else ""
        ),
        computed={
            "generalization_chains": sorted(
                str(d.key.dressing_chain) for d in generalization_set
            ),
            "bound_chains": sorted(str(d.key.dressing_chain) for d in bound),
            "chains_recovering_off_distribution": sorted(
                str(d.key.dressing_chain) for d in recovering
            ),
            "excluded_probes_none": sorted(str(d.key.dressing_chain) for d in excluded),
            "minimum_detectable_effect": (
                max(_half_width(delta) for delta in generalization_set)
            ),
        },
    )


def verdicts(cells: Sequence[object], timing, confirmatory) -> tuple[Verdict, ...]:  # type: ignore[no-untyped-def]
    """All four, in declared order, and **no abort** whatever they say.

    Producing and gating are separate on purpose: a caller sees every verdict before any of them
    stops the run, and both paths are reachable from a test. `refuse_an_unevaluable_run` is the
    gate.
    """
    produced = (
        evaluate_n1(cells, confirmatory),
        evaluate_n2(cells),
        evaluate_n3(timing),
        evaluate_n4(cells),
    )
    assert tuple(v.condition for v in produced) == FALSIFICATION_CONDITIONS
    return produced


def refuse_an_unevaluable_run(produced: Sequence[Verdict]) -> None:
    """Abort if any condition had no input. A triggered condition passes through.

    The two are opposite failures and only one of them is a failure of the artifact. A triggered
    condition is the artifact working -- the result is negative and gets published. An unevaluable
    one is a condition nobody could check, and the specific danger is that it reads as the first.
    """
    unevaluable = [v for v in produced if v.outcome == OUTCOME_NOT_EVALUABLE]
    if not unevaluable:
        return
    detail = "; ".join(f"{v.condition}: {v.reason}" for v in unevaluable)
    raise VerdictNotEvaluable(
        f"{len(unevaluable)} of {len(produced)} falsification conditions could not be evaluated, "
        f"so this run publishes no table. A condition that silently reads as un-triggered when its "
        f"input is absent is how the whole section becomes decorative. {detail}"
    )
