"""The four falsification conditions, and the asymmetry between the two ways they can fail.

Cells are built by hand rather than aggregated, because the shapes that decide these conditions --
a held-out chain that recovers, a bound chain that does not, a pair whose interval excludes zero --
are ones a realistic corpus makes hard to arrange and harder to read.

The one thing not built by hand is N1's declared cell: it is read from `pins.toml`, which is where
it was hashed before anything was measured, and a test that supplied its own would be testing a
declaration this repository does not use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from nbc.corpus.matrix import CHAIN_CLASS_BOUND, CHAIN_CLASS_HELD_OUT, CLEAN_CHAIN_NAME
from nbc.errors import declared_exit_codes
from nbc.harness.stats import mover_difference_interval
from nbc.harness.timing import (
    BATCH_SIZE_ONE,
    CORPUS_CLASSES,
    InferenceTiming,
    LayerTiming,
    Percentiles,
    TimingReport,
)
from nbc.harness.verdict import (
    N3_ABSOLUTE_CEILING_NS,
    N3_INFERENCE_SHARE,
    VerdictNotEvaluable,
    evaluate_n1,
    evaluate_n2,
    evaluate_n3,
    evaluate_n4,
    refuse_an_unevaluable_run,
    verdicts,
)
from nbc.pins import load_pins
from nbc.schema import (
    AXIS_CANON_ON,
    AXIS_DRESSING_CHAIN,
    CENSUS_CEILING_HIT,
    CENSUS_WINDOW_OVERFLOW,
    CONTRAST_CANON_ON_VS_OFF,
    CONTRAST_CLEAN_VS_CHAIN,
    FALSIFICATION_CONDITIONS,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    MOVER_DIFFERENCE,
    NEWCOMBE_PAIRED,
    OUTCOME_NOT_EVALUABLE,
    OUTCOME_NOT_TRIGGERED,
    OUTCOME_TRIGGERED,
    POPULATION_SINGLE_WINDOW,
    CellKey,
    Contrast,
    Count,
    Delta,
    Interval,
)

POLICY = "shared"
BOUND_CHAIN = "base64"
HELD_OUT_CHAIN = "base32"
PROBES_NONE_CHAIN = "rot13"

CANON = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))


@pytest.fixture(scope="module")
def declared():
    """The confirmatory cell as `pins.toml` declares it, hashed into `frame_id` before measuring."""
    return load_pins(None).benign_frame.confirmatory_cell


def key(
    *,
    baseline: str,
    chain: str,
    family: str = FAMILY_ATTACK,
    benign_class: str | None = None,
    chain_class: str = CHAIN_CLASS_BOUND,
    contrast: Contrast | None = CANON,
    canon_on: bool | None = None,
    population: str = "all",
) -> CellKey:
    return CellKey(
        baseline=baseline,
        dressing_chain=chain,
        chain_class=chain_class,
        window_policy=POLICY,
        canon_on=canon_on,
        family=family,
        benign_class=benign_class,
        contrast=contrast,
        population=population,
    )


def canon_delta(value: float, half: float, **kwargs: object) -> Delta:
    return Delta(
        value=value,
        interval=Interval(value - half, value + half, NEWCOMBE_PAIRED),
        key=key(**kwargs),  # type: ignore[arg-type]
    )


def clean_vs(value: float, half: float, *, baseline: str, chain: str, chain_class: str = CHAIN_CLASS_BOUND) -> Delta:
    spans = {AXIS_DRESSING_CHAIN}
    if chain_class != CHAIN_CLASS_BOUND:
        spans.add("chain_class")
    return Delta(
        value=value,
        interval=Interval(value - half, value + half, NEWCOMBE_PAIRED),
        key=CellKey(
            baseline=baseline,
            dressing_chain=None,
            chain_class=None if "chain_class" in spans else chain_class,
            window_policy=POLICY,
            canon_on=False,
            family=FAMILY_ATTACK,
            benign_class=None,
            contrast=Contrast(CONTRAST_CLEAN_VS_CHAIN, chain, frozenset(spans)),
        ),
    )


def a_timing(layer_p95: int, inference: dict[str, int]) -> TimingReport:
    one = Percentiles(layer_p95, layer_p95, 3)
    return TimingReport(
        layer=LayerTiming(
            overall=one,
            by_class={name: one for name in CORPUS_CLASSES},
            trace_enabled=True,
        ),
        inference=InferenceTiming(
            by_baseline={k: Percentiles(v, v, 3) for k, v in inference.items()},
            batch_size=BATCH_SIZE_ONE,
        ),
        elapsed_ns=1,
    )


# --- the arithmetic N1 needed and did not have -----------------------------------------------------


def test_mover_reduces_to_the_variance_sum_on_symmetric_intervals() -> None:
    """The criterion writes `Var(D) = Var(A) + Var(B)`. These deltas carry Newcombe intervals,
    which are asymmetric and are not variances, so there is nothing to add -- MOVER-R is the
    interval-form of the same argument, and this is the input where the two coincide."""
    a, b = 0.30, 0.10
    result = mover_difference_interval(
        a, Interval(a - 0.08, a + 0.08, NEWCOMBE_PAIRED),
        b, Interval(b - 0.06, b + 0.06, NEWCOMBE_PAIRED),
    )
    assert result.method == MOVER_DIFFERENCE
    assert (result.lo + result.hi) / 2 == pytest.approx(a - b)
    assert result.width / 2 == pytest.approx(math.hypot(0.08, 0.06))


def test_mover_combines_the_two_sides_of_an_asymmetric_interval_separately() -> None:
    """The input the symmetric case cannot distinguish: the lower bound of the difference uses the
    first estimate's lower reach and the second's upper one, and the upper bound uses the other
    two. Getting the pairing backwards is invisible on symmetric inputs."""
    a, b = 0.30, 0.10
    result = mover_difference_interval(
        a, Interval(a - 0.20, a + 0.01, NEWCOMBE_PAIRED),
        b, Interval(b - 0.02, b + 0.30, NEWCOMBE_PAIRED),
    )
    assert result.lo == pytest.approx((a - b) - math.hypot(0.20, 0.30))
    assert result.hi == pytest.approx((a - b) + math.hypot(0.01, 0.02))


def test_mover_refuses_an_estimate_outside_its_own_interval() -> None:
    from nbc.harness.stats import StatisticUndefined

    with pytest.raises(StatisticUndefined):
        mover_difference_interval(
            0.9, Interval(0.0, 0.2, NEWCOMBE_PAIRED),
            0.1, Interval(0.0, 0.2, NEWCOMBE_PAIRED),
        )


# --- N1 ----------------------------------------------------------------------------------------------


def n1_cells(declared, *, fpr: float, recall: float, half: float = 0.02) -> list[Delta]:
    return [
        canon_delta(
            fpr, half,
            baseline=declared.baseline, chain=declared.dressing_chain,
            family=FAMILY_BENIGN, benign_class=declared.benign_class,
        ),
        canon_delta(
            recall, half,
            baseline=declared.baseline, chain=declared.dressing_chain,
        ),
    ]


def test_the_confirmatory_cell_is_declared_in_pins_and_hashed(declared) -> None:
    """N1's pre-registration is real rather than promised: the cell is in `pins.toml` and travels
    into `frame_id`, so it cannot have been chosen after the numbers existed."""
    assert declared.baseline and declared.dressing_chain and declared.benign_class
    assert declared.declared_on


def test_n1_triggers_when_the_cost_exceeds_the_recovery(declared) -> None:
    verdict = evaluate_n1(n1_cells(declared, fpr=0.30, recall=0.05), declared)
    assert verdict.outcome == OUTCOME_TRIGGERED
    assert verdict.computed["difference"] == pytest.approx(0.25)
    assert verdict.computed["difference_interval"]["lo"] > 0  # type: ignore[index]


def test_n1_does_not_trigger_when_the_interval_includes_zero(declared) -> None:
    verdict = evaluate_n1(n1_cells(declared, fpr=0.10, recall=0.09), declared)
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED
    assert verdict.computed["minimum_detectable_effect"] > 0


def test_n1_is_not_evaluable_when_its_declared_cell_is_absent(declared) -> None:
    """It is never re-pointed at another cell. A run whose cells do not contain the pre-registered
    one has not tested the pre-registered hypothesis."""
    elsewhere = canon_delta(0.5, 0.01, baseline="somebody-else", chain=CLEAN_CHAIN_NAME)
    verdict = evaluate_n1([elsewhere], declared)
    assert verdict.outcome == OUTCOME_NOT_EVALUABLE
    assert declared.dressing_chain in verdict.reason


def test_n1_counts_the_exploratory_cells_it_scanned(declared) -> None:
    """`m` is recorded rather than corrected for: applying a multiplicity correction would mean
    choosing one, and the choice would be ours rather than the reader's."""
    others = [
        canon_delta(0.4, 0.01, baseline=declared.baseline, chain=CLEAN_CHAIN_NAME),
        canon_delta(0.4, 0.01, baseline="another", chain=BOUND_CHAIN),
    ]
    verdict = evaluate_n1(n1_cells(declared, fpr=0.30, recall=0.05) + others, declared)
    assert verdict.computed["exploratory_cells_scanned"] == 2
    assert "exploratory" in verdict.reason


def test_a_triggered_exploratory_cell_is_never_the_verdict(declared) -> None:
    """The exploratory cell here has a far larger gap than the confirmatory one, and N1 still reads
    the confirmatory one."""
    exploratory = [
        canon_delta(0.9, 0.01, baseline=declared.baseline, chain=CLEAN_CHAIN_NAME,
                    family=FAMILY_BENIGN, benign_class=declared.benign_class),
        canon_delta(0.0, 0.01, baseline=declared.baseline, chain=CLEAN_CHAIN_NAME),
    ]
    verdict = evaluate_n1(n1_cells(declared, fpr=0.10, recall=0.09) + exploratory, declared)
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED
    assert verdict.computed["difference"] == pytest.approx(0.01)


def test_n1_ignores_the_windows_matched_companion(declared) -> None:
    """The companion shares the confirmatory cell's column and is a different item set. Admitting
    it would give N1 two candidates for one side and make the verdict depend on dict order."""
    companion = canon_delta(
        0.99, 0.01,
        baseline=declared.baseline, chain=declared.dressing_chain,
        family=FAMILY_BENIGN, benign_class=declared.benign_class,
        population=POPULATION_SINGLE_WINDOW,
    )
    verdict = evaluate_n1(n1_cells(declared, fpr=0.10, recall=0.09) + [companion], declared)
    assert verdict.computed["delta_false_positive"] == pytest.approx(0.10)


# --- N2 ----------------------------------------------------------------------------------------------


def test_n2_triggers_when_no_dressing_degraded_anything() -> None:
    cells = [
        clean_vs(0.0, 0.05, baseline="primary", chain=BOUND_CHAIN),
        clean_vs(0.01, 0.05, baseline="secondary", chain=BOUND_CHAIN),
    ]
    verdict = evaluate_n2(cells)
    assert verdict.outcome == OUTCOME_TRIGGERED
    assert verdict.computed["pairs_examined"] == 2


def test_n2_names_the_pair_that_kept_it_from_triggering() -> None:
    """"Not every pair" is unactionable; the pair is what a reader needs."""
    cells = [
        clean_vs(0.0, 0.05, baseline="primary", chain=BOUND_CHAIN),
        clean_vs(0.40, 0.05, baseline="secondary", chain=BOUND_CHAIN),
    ]
    verdict = evaluate_n2(cells)
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED
    assert "secondary" in verdict.reason and BOUND_CHAIN in verdict.reason
    assert len(verdict.computed["pairs_excluding_zero"]) == 1  # type: ignore[arg-type]


def test_n2_ignores_an_unencoded_chain() -> None:
    """A homoglyph chain costs the layer no decode budget and is not what this condition asks
    about. `matrix.encoding_depth` decides, rather than a list of names."""
    assert evaluate_n2(
        [clean_vs(0.9, 0.01, baseline="primary", chain="homoglyph")]
    ).outcome == OUTCOME_NOT_EVALUABLE


def test_n2_is_not_evaluable_with_no_bound_encoded_pair() -> None:
    verdict = evaluate_n2([])
    assert verdict.outcome == OUTCOME_NOT_EVALUABLE
    assert verdict.computed["pairs_examined"] == 0


# --- N3 ----------------------------------------------------------------------------------------------


def test_n3_uses_the_share_when_it_is_the_smaller_ceiling() -> None:
    """A fast baseline: 10% of 2 ms is 200 us, under the absolute millisecond, so the share binds."""
    verdict = evaluate_n3(a_timing(layer_p95=300_000, inference={"fast": 2_000_000}))
    assert verdict.computed["ceiling_ns"] == pytest.approx(200_000.0)
    assert verdict.computed["binding_ceiling"] == "the share of inference"
    assert verdict.outcome == OUTCOME_TRIGGERED


def test_n3_uses_the_absolute_when_it_is_the_smaller_ceiling() -> None:
    """A slow baseline: 10% of 50 ms is 5 ms, above the absolute, so the millisecond binds. `min`
    is the operator, so a slow baseline cannot buy the layer more budget."""
    verdict = evaluate_n3(a_timing(layer_p95=300_000, inference={"slow": 50_000_000}))
    assert verdict.computed["ceiling_ns"] == pytest.approx(float(N3_ABSOLUTE_CEILING_NS))
    assert verdict.computed["binding_ceiling"] == "the absolute"
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED


def test_n3_takes_the_fastest_baseline() -> None:
    """The smallest inference p95 is the one the layer's share is largest against, which is the
    conservative reading and the one the criterion names."""
    verdict = evaluate_n3(
        a_timing(layer_p95=1, inference={"slow": 50_000_000, "fast": 2_000_000})
    )
    assert verdict.computed["fastest_baseline"] == "fast"


def test_n3_ceilings_are_the_declared_constants() -> None:
    verdict = evaluate_n3(a_timing(layer_p95=1, inference={"one": 10_000_000}))
    assert verdict.computed["share_ceiling_ns"] == pytest.approx(
        N3_INFERENCE_SHARE * 10_000_000
    )
    assert verdict.computed["absolute_ceiling_ns"] == N3_ABSOLUTE_CEILING_NS


def test_n3_is_not_evaluable_without_a_timing_report() -> None:
    """Without it the condition has no right-hand side, which is why the dedicated pass is
    mandatory rather than optional."""
    assert evaluate_n3(None).outcome == OUTCOME_NOT_EVALUABLE


# --- N4 ----------------------------------------------------------------------------------------------


def ceiling_census(chain: str, *, hits: int = 3) -> Count:
    return Count(
        hits, 3, key(baseline="primary", chain=chain, contrast=None, canon_on=True),
        census=CENSUS_CEILING_HIT,
    )


def window_census(chain: str, *, hits: int = 3) -> Count:
    """The other census, which shares a type and a key with the one above and is not it."""
    return Count(
        hits, 3, key(baseline="primary", chain=chain, contrast=None, canon_on=True),
        census=CENSUS_WINDOW_OVERFLOW,
    )


def test_n4_triggers_when_recovery_is_confined_to_the_bound_chains() -> None:
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        canon_delta(0.00, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert verdict.outcome == OUTCOME_TRIGGERED
    assert HELD_OUT_CHAIN in verdict.computed["generalization_chains"]  # type: ignore[operator]


def test_n4_names_the_held_out_chain_that_kept_it_from_triggering() -> None:
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        canon_delta(0.40, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED
    assert HELD_OUT_CHAIN in verdict.reason


def test_n4_does_not_trigger_when_no_bound_chain_recovers() -> None:
    """A layer that recovers nowhere has not generalized badly, it has not worked. The bound half
    is what tells the two apart."""
    cells = [
        canon_delta(0.00, 0.05, baseline="primary", chain=BOUND_CHAIN),
        canon_delta(0.00, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert verdict.outcome == OUTCOME_NOT_TRIGGERED
    assert "has not worked" in verdict.reason


def test_n4_excludes_a_probes_none_chain_and_reports_it_anyway() -> None:
    """`rot13` gives the layer nothing to engage, so no recovery is expected there and counting it
    would make N4 trigger for a reason that is not generalization failure."""
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        canon_delta(0.00, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
        canon_delta(0.00, 0.05, baseline="primary", chain=PROBES_NONE_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert verdict.computed["excluded_probes_none"] == [PROBES_NONE_CHAIN]
    assert PROBES_NONE_CHAIN not in verdict.computed["generalization_chains"]  # type: ignore[operator]
    assert "probes" in verdict.reason


def test_n4_reads_the_over_ceiling_chain_off_the_runs_own_census() -> None:
    """Not re-derived from the declared default: `tests/canon/test_recursion.py` holds
    `DEFAULT_CEILING` to one reader, and the census records the ceiling the run ACTUALLY applied."""
    over = "base64+base64+base64+base64"
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        canon_delta(0.00, 0.05, baseline="primary", chain=over),
        ceiling_census(over),
    ]
    verdict = evaluate_n4(cells)
    assert over in verdict.computed["generalization_chains"]  # type: ignore[operator]
    assert verdict.outcome == OUTCOME_TRIGGERED


def test_a_window_overflow_census_is_not_read_as_a_ceiling_hit() -> None:
    """The two censuses share a type and a key, and story 4.3 gave them no field to tell them
    apart. Without `Count.census` this chain -- which has multi-window documents and no ceiling hit
    -- would have been classified as over-ceiling and counted into the generalization set."""
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        window_census(BOUND_CHAIN),
        canon_delta(0.00, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert BOUND_CHAIN in verdict.computed["bound_chains"]  # type: ignore[operator]
    assert BOUND_CHAIN not in verdict.computed["generalization_chains"]  # type: ignore[operator]


def test_a_count_refuses_a_census_nobody_declared() -> None:
    with pytest.raises(ValueError):
        Count(1, 2, key(baseline="primary", chain=BOUND_CHAIN, contrast=None, canon_on=True),
              census="edits_by_stage")


def test_a_chain_with_no_ceiling_hits_stays_a_bound_chain() -> None:
    """The input that keeps the census read from classifying everything as over-ceiling."""
    cells = [
        canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
        ceiling_census(BOUND_CHAIN, hits=0),
        canon_delta(0.00, 0.05, baseline="primary", chain=HELD_OUT_CHAIN,
                    chain_class=CHAIN_CLASS_HELD_OUT),
    ]
    verdict = evaluate_n4(cells)
    assert BOUND_CHAIN in verdict.computed["bound_chains"]  # type: ignore[operator]


def test_n4_is_not_evaluable_without_both_halves() -> None:
    only_bound = [canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN)]
    assert evaluate_n4(only_bound).outcome == OUTCOME_NOT_EVALUABLE


# --- the gate ------------------------------------------------------------------------------------------


def a_complete_run(declared):  # type: ignore[no-untyped-def]
    over = "base64+base64+base64+base64"
    return (
        n1_cells(declared, fpr=0.10, recall=0.09)
        + [
            clean_vs(0.0, 0.05, baseline="primary", chain=BOUND_CHAIN),
            canon_delta(0.40, 0.05, baseline="primary", chain=BOUND_CHAIN),
            canon_delta(0.00, 0.05, baseline="primary", chain=over),
            ceiling_census(over),
        ],
        a_timing(layer_p95=300_000, inference={"fast": 2_000_000}),
    )


def test_verdicts_produces_all_four_in_declared_order(declared) -> None:
    cells, timing = a_complete_run(declared)
    produced = verdicts(cells, timing, declared)
    assert tuple(v.condition for v in produced) == FALSIFICATION_CONDITIONS
    for verdict in produced:
        assert verdict.reason.strip()
        assert "minimum_detectable_effect" in verdict.computed


def test_a_triggered_condition_never_aborts(declared) -> None:
    """A negative result is publishable and this artifact commits to publishing it."""
    cells, timing = a_complete_run(declared)
    produced = verdicts(cells, timing, declared)
    assert any(v.triggered for v in produced), "this run triggers at least one"
    refuse_an_unevaluable_run(produced)  # does not raise


def test_an_unevaluable_condition_aborts_naming_the_missing_input(declared) -> None:
    """The opposite failure, and the one that matters: a condition that reads as un-triggered when
    nobody could check it is how the whole section becomes decorative."""
    cells, _ = a_complete_run(declared)
    produced = verdicts(cells, None, declared)
    assert [v.outcome for v in produced].count(OUTCOME_NOT_EVALUABLE) == 1
    with pytest.raises(VerdictNotEvaluable) as caught:
        refuse_an_unevaluable_run(produced)
    assert "N3" in str(caught.value)
    assert "right-hand side" in str(caught.value)


def test_the_new_abort_declares_exit_code_33_and_declares_it_once() -> None:
    assert declared_exit_codes()[33] is VerdictNotEvaluable
    assert VerdictNotEvaluable.exit_code == 33
