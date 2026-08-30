"""Every estimator in `harness/stats.py`, against values produced somewhere other than by it.

**Where the goldens come from.** Each one was computed on 2026-08-30 in a separate evaluation using
`fractions.Fraction` for everything exact and 60-digit `decimal.Decimal` for the square roots, in
the textbook `p-hat` arrangement rather than the `(k + z^2/2)/(n + z^2)` one the implementation
uses. That is what makes this a comparison: the two sides differ in arithmetic *and* in algebraic
arrangement, so a transcription error, a mis-parenthesised half-width or a lost digit at small `k`
shows up as a disagreement rather than as two identical mistakes.

**Why `rel_tol` and not equality.** The goldens are exact and the implementation is float64, so they
agree to rounding and not to the bit. The measured worst case across every golden here is a
relative deviation of **1.8e-15**, on Newcombe's lower bound for `(40, 10, 2, 48)`; everything else
is at or below 1e-15. `GOLDEN_TOLERANCE` is 1e-14 -- an order of magnitude above what float64
rounding produces and many orders below anything a different method would move. Swapping Wilson for
its continuity-corrected variant moves `k=0, n=10` from 0.2775 to 0.3050.

`Z_95` is used everywhere except the one test that pins `z = 1.96` and checks three published
textbook values, which is a genuinely external source rather than a second arrangement of ours.

**Where that independence is weaker, said rather than left to be discovered.** Wilson has an
external anchor and AUC has two arrangements of one definition that part company on ties. Newcombe
has neither: its goldens were produced from the same algebraic formula the implementation uses,
transcribed into exact arithmetic, so they check the transcription and the precision and **not the
choice of method**. What carries the method choice there is the property tests, and they are named
so a reviewer can weigh them: the paired and unpaired widths reverse order between a
positively- and a negatively-associated table, a table with no discordance gives an interval
symmetric about exactly zero, a wholly discordant table drives the correlation term to -1, and a
table where one condition never fires degenerates to plain square-and-add. A `(b, c, n)` record --
which is what selects Tango's method by accident -- cannot produce three of those four.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from nbc.errors import declared_exit_codes
from nbc.harness.stats import (
    P50,
    P95,
    Z_95,
    AucSample,
    StatisticUndefined,
    delta_auc,
    nearest_rank_percentile,
    newcombe_paired_interval,
    rejected_hanley_mcneil_variance,
    roc_auc,
    unpaired_square_and_add_interval,
    wilson_interval,
)
from nbc.schema import (
    AUC_STRUCTURAL,
    DELTA_AUC_STRUCTURAL,
    NEWCOMBE_PAIRED,
    WILSON_SCORE,
    Interval,
    PairedCount,
)

GOLDEN_TOLERANCE = 1e-14
"""Ten times the worst float64 deviation measured against the exact goldens. See the docstring."""


def assert_close(actual: float, expected: float, what: str) -> None:
    assert math.isclose(actual, expected, rel_tol=GOLDEN_TOLERANCE, abs_tol=GOLDEN_TOLERANCE), (
        f"{what}: got {actual!r}, golden {expected!r}, "
        f"relative deviation {abs(actual - expected) / max(abs(expected), 1e-300):.3e}"
    )


# --- Wilson -------------------------------------------------------------------------------------

WILSON_GOLDENS = (
    # (k, n, lo, hi) -- exact arithmetic, 2026-08-30, p-hat arrangement.
    (0, 10, 0.0, 0.2775327998628892045260309),
    (5, 10, 0.2365930905125640005632618, 0.7634069094874359994367382),
    (10, 10, 0.7224672001371107954739691, 1.0),
)


@pytest.mark.parametrize("k,n,lo,hi", WILSON_GOLDENS)
def test_wilson_matches_the_hand_checked_triples(k: int, n: int, lo: float, hi: float) -> None:
    interval = wilson_interval(k, n)
    assert_close(interval.lo, lo, f"wilson lo for {k}/{n}")
    assert_close(interval.hi, hi, f"wilson hi for {k}/{n}")
    assert interval.method == WILSON_SCORE


def test_wilson_at_the_shape_a_per_class_false_positive_rate_has() -> None:
    """A handful of events in a few hundred trials, which is where an arrangement that divides
    inside the square root loses digits.

    512 rather than the declared per-class benign sample size, deliberately. This golden is a fact
    about arithmetic and must not move when a sample size is re-declared, and `pins.toml` is the
    only home for that number -- `tests/test_pins.py` enforces both halves of that and caught this
    test using it.
    """
    interval = wilson_interval(3, 512)
    assert_close(interval.lo, 0.001994675792186092533941838, "wilson lo for 3/512")
    assert_close(interval.hi, 0.01708378068394411391104614, "wilson hi for 3/512")


def test_wilson_agrees_with_published_values_at_the_rounded_z() -> None:
    """The external anchor. At `z = 1.96` exactly these are the textbook figures for 0/10, 5/10 and
    10/10, and they come from outside this repository rather than from another arrangement of our
    own arithmetic."""
    assert round(wilson_interval(0, 10, 1.96).hi, 4) == 0.2775
    low = wilson_interval(5, 10, 1.96)
    assert (round(low.lo, 4), round(low.hi, 4)) == (0.2366, 0.7634)
    assert round(wilson_interval(10, 10, 1.96).lo, 4) == 0.7225


def test_wilson_never_leaves_the_unit_interval_at_the_boundaries() -> None:
    """At `k = 0` the centre and the half-width are equal in exact arithmetic, so their difference
    is zero -- and float64 lands on **either side of it depending on `n`**.

    Measured while writing this: `n = 10` gives a small negative and `n = 512` gives
    `+4.336808689942018e-19`. A clamp toward zero catches only the first, and the second is the one
    that publishes -- a "0 in a few hundred" false-positive rate would have carried a lower bound
    of 0.00000000000000000043. Both boundaries are therefore set from the algebra rather than
    clamped toward it, and the larger `n` is in this test because it is the input that was passing
    under the one-sided clamp.
    """
    assert wilson_interval(0, 10).lo == 0.0
    assert wilson_interval(0, 512).lo == 0.0
    assert wilson_interval(10, 10).hi == 1.0
    assert wilson_interval(512, 512).hi == 1.0


def test_wilson_refuses_a_rate_with_no_denominator() -> None:
    with pytest.raises(StatisticUndefined) as caught:
        wilson_interval(0, 0)
    assert "0 trials" in str(caught.value)


def test_wilson_refuses_more_events_than_trials() -> None:
    with pytest.raises(StatisticUndefined) as caught:
        wilson_interval(11, 10)
    assert "counted twice" in str(caught.value)


def test_wilson_refuses_a_bool_as_a_count() -> None:
    """`isinstance(True, int)` holds, so an unguarded implementation reads `True` as one success."""
    with pytest.raises(StatisticUndefined):
        wilson_interval(True, 10)


def test_wilson_refuses_a_negative_count() -> None:
    with pytest.raises(StatisticUndefined):
        wilson_interval(-1, 10)


# --- nearest-rank percentile ---------------------------------------------------------------------


def test_percentile_p50_is_the_fifth_of_ten() -> None:
    """`ceil(1/2 * 10) - 1 = 4`, and `sorted[4]` is 5. Hand-checkable in one line, which is the
    point of nearest rank."""
    assert nearest_rank_percentile(list(range(1, 11)), P50) == 5


def test_percentile_p95_is_the_largest_of_ten() -> None:
    assert nearest_rank_percentile(list(range(1, 11)), P95) == 10


def test_percentile_sorts_its_input() -> None:
    """The samples arrive in the order the timing pass produced them, not sorted."""
    assert nearest_rank_percentile([907, 101, 553, 302, 741], P50) == 553


def test_percentile_returns_an_observed_sample_and_never_an_interpolation() -> None:
    """With an even count the interpolating definitions return the mean of two neighbours. This one
    returns a sample, and 250 was never on any clock."""
    result = nearest_rank_percentile([100, 200, 300, 400], P50)
    assert result == 200
    assert isinstance(result, int)


def test_the_exact_quantile_and_the_float_one_disagree_and_the_exact_one_is_right() -> None:
    """The measured input, 2026-08-30: over every `j/100` and `j/1000` at `n` up to 20000, the
    smallest disagreement is `q = 7/100` at `n = 100`.

    Nearest rank is the 7th of 100 samples. `math.ceil(0.07 * 100)` is 8, because 0.07 as a double
    is a hair above 7/100. The test carries both sides so the reason for the `Fraction` is a
    measurement rather than a preference.

    Note what this does *not* claim: for `P50` and `P95`, the two percentiles this epic reports,
    the same search found no disagreement at any `n` up to 200000. This closes a class of bug
    before story 4-5 declares a third percentile; it does not repair a wrong number in the table.
    """
    samples = list(range(1, 101))
    assert math.ceil(0.07 * 100) == 8
    assert nearest_rank_percentile(samples, Fraction(7, 100)) == 7


def test_percentile_refuses_a_float_quantile() -> None:
    with pytest.raises(StatisticUndefined) as caught:
        nearest_rank_percentile([1, 2, 3], 0.95)  # type: ignore[arg-type]
    assert "Fraction" in str(caught.value)


def test_percentile_refuses_no_samples() -> None:
    with pytest.raises(StatisticUndefined) as caught:
        nearest_rank_percentile((), P50)
    assert "did not run" in str(caught.value)


def test_percentile_refuses_a_non_integer_sample() -> None:
    """`perf_counter_ns` returns integers. A float in the sample means somebody averaged first."""
    with pytest.raises(StatisticUndefined) as caught:
        nearest_rank_percentile([1, 2.5, 3], P50)  # type: ignore[list-item]
    assert "integer nanoseconds" in str(caught.value)


@pytest.mark.parametrize("q", [Fraction(0), Fraction(-1, 2), Fraction(3, 2)])
def test_percentile_refuses_a_quantile_outside_the_half_open_unit(q: Fraction) -> None:
    with pytest.raises(StatisticUndefined):
        nearest_rank_percentile([1, 2, 3], q)


# --- Newcombe ------------------------------------------------------------------------------------

POSITIVELY_CORRELATED = PairedCount(a=40, b=10, c=2, d=48)
"""The shape canon-on versus canon-off has: the two conditions agree on most items."""

NEGATIVELY_CORRELATED = PairedCount(a=5, b=45, c=45, d=5)
"""The shape that reverses the width argument, which is why it is in the suite."""


def test_newcombe_matches_the_hand_checked_table() -> None:
    interval = newcombe_paired_interval(POSITIVELY_CORRELATED)
    assert_close(interval.lo, 0.01130718880725064498743657, "newcombe lo")
    assert_close(interval.hi, 0.1466920129384579436782405, "newcombe hi")
    assert interval.method == NEWCOMBE_PAIRED
    assert POSITIVELY_CORRELATED.theta == 0.08


def test_newcombe_on_a_table_with_no_discordance() -> None:
    """`b = c = 0`: the two conditions classified every item the same way, so the difference is
    exactly zero and the interval is symmetric about it."""
    counts = PairedCount(a=30, b=0, c=0, d=70)
    interval = newcombe_paired_interval(counts)
    assert counts.theta == 0.0
    assert_close(interval.lo, -0.02426722084632472186980915, "newcombe lo, no discordance")
    assert_close(interval.hi, 0.02426722084632472186980915, "newcombe hi, no discordance")
    assert interval.lo < 0 < interval.hi


def test_newcombe_on_a_wholly_discordant_table() -> None:
    """`a = d = 0`, so the correlation term is -1 and the marginal products vanish on one side.
    The interval is wide and stays above zero, which is the answer 15 items support."""
    interval = newcombe_paired_interval(PairedCount(a=0, b=12, c=3, d=0))
    assert_close(interval.lo, 0.09629102569661289589840373, "newcombe lo, discordant")
    assert_close(interval.hi, 0.8590490130603687974142600, "newcombe hi, discordant")
    assert interval.lo > 0


def test_pairing_narrows_the_interval_on_the_table_this_artifact_will_see() -> None:
    """The epic's stated reason for pairing, as a measurement.

    Treating canon-on and canon-off as two independent samples ignores that they agree on 88 of
    100 items. Measured: the unpaired interval is 0.2704 wide against the paired 0.1354, a factor
    of 1.997 -- and it straddles zero where the paired one does not. That is the whole of the
    epic's argument: the wider interval makes N1 harder to trigger and errs toward never declaring
    a negative result.
    """
    paired = newcombe_paired_interval(POSITIVELY_CORRELATED)
    unpaired = unpaired_square_and_add_interval(POSITIVELY_CORRELATED)
    assert paired.width < unpaired.width
    assert unpaired.width / paired.width == pytest.approx(1.997, abs=0.001)
    assert unpaired.lo < 0 < unpaired.hi
    assert paired.lo > 0


def test_pairing_widens_the_interval_when_the_two_conditions_disagree() -> None:
    """The input that keeps the previous test's claim from being stated as universal.

    On a negatively-associated table the pairing term adds rather than subtracts, and the paired
    interval comes out **wider** than the unpaired one. The epic's direction is a property of
    positively-correlated conditions, not of the method, and the docstring says so because this
    table says so.
    """
    paired = newcombe_paired_interval(NEGATIVELY_CORRELATED)
    unpaired = unpaired_square_and_add_interval(NEGATIVELY_CORRELATED)
    assert paired.width > unpaired.width


def test_newcombe_on_a_table_where_one_condition_never_fires() -> None:
    """The `phi = 0` branch, which no other table in this suite reaches.

    With `a = b = 0` the first condition is negative on every item, so its marginal has no
    variation and there is nothing for it to correlate with -- the product of marginals is zero and
    Newcombe's correlation term is undefined rather than large. The method degenerates to plain
    square-and-add, which is why the paired and unpaired intervals are identical here and are
    asserted to be. Without this test the branch is a division by zero nobody has tried.
    """
    counts = PairedCount(a=0, b=0, c=20, d=80)
    paired = newcombe_paired_interval(counts)
    unpaired = unpaired_square_and_add_interval(counts)
    assert counts.first_positive == 0
    assert (paired.lo, paired.hi) == (unpaired.lo, unpaired.hi)
    assert paired.hi < 0, "the second condition fires on 20 items and the first on none"


def test_newcombe_refuses_an_empty_table() -> None:
    with pytest.raises(StatisticUndefined) as caught:
        newcombe_paired_interval(PairedCount(0, 0, 0, 0))
    assert "no items were compared" in str(caught.value)


def test_the_paired_count_carries_all_four_cells_and_derives_the_rest() -> None:
    """The record shape is the gate: a `(b, c, n)` record cannot produce these marginals, and a
    Newcombe implementation handed one is not written at all -- it is replaced by Tango's."""
    counts = PairedCount(a=40, b=10, c=2, d=48)
    assert counts.n == 100
    assert counts.first_positive == 50
    assert counts.second_positive == 42
    assert counts.theta == pytest.approx(0.08)


@pytest.mark.parametrize("cells", [(-1, 0, 0, 0), (0, 0, 0, -3)])
def test_paired_count_refuses_a_negative_cell(cells: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        PairedCount(*cells)


def test_paired_count_refuses_a_bool_cell() -> None:
    with pytest.raises(ValueError):
        PairedCount(True, 0, 0, 0)


# --- ROC AUC ---------------------------------------------------------------------------------------


def sample(pos: tuple[float, ...], neg: tuple[float, ...]) -> AucSample:
    return AucSample(
        positive_ids=tuple(f"p{i}" for i in range(len(pos))),
        positive_scores=pos,
        negative_ids=tuple(f"n{i}" for i in range(len(neg))),
        negative_scores=neg,
    )


def midrank_auc(pos: tuple[float, ...], neg: tuple[float, ...]) -> Fraction:
    """The rank arrangement of the same statistic, computed here in exact rationals.

    `A = (sum of positive midranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`. The implementation
    averages the structural components instead. Two arrangements of one definition, and ties are
    where they would part company if the half were dropped from either.
    """
    combined = sorted(list(pos) + list(neg))
    midranks: dict[float, Fraction] = {}
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1] == combined[index]:
            end += 1
        midranks[combined[index]] = Fraction(index + end + 2, 2)
        index = end + 1
    rank_sum = sum(midranks[x] for x in pos)
    return (rank_sum - Fraction(len(pos) * (len(pos) + 1), 2)) / (len(pos) * len(neg))


def test_auc_matches_the_hand_checked_triple() -> None:
    estimate = roc_auc(sample((3.0, 1.0), (2.0, 0.0)))
    assert estimate.auc == 0.75
    assert_close(estimate.interval.lo, 0.05704808782516111063153207, "auc lo")
    assert estimate.interval.method == AUC_STRUCTURAL
    assert (estimate.n_positive, estimate.n_negative) == (2, 2)


def test_the_auc_upper_bound_is_clipped_at_one_and_the_uncipped_value_is_above_it() -> None:
    """The normal approximation puts this cell's upper bound at 1.4430. A published upper bound
    above 1 on a probability is not a wider claim, it is an unreadable one -- so it is clipped, and
    the test names the value that was clipped away so the clip is visible rather than assumed."""
    estimate = roc_auc(sample((3.0, 1.0), (2.0, 0.0)))
    unclipped = estimate.auc + Z_95 * math.sqrt(estimate.variance)
    assert unclipped > 1.0
    assert_close(unclipped, 1.442951912174838889368468, "unclipped auc hi")
    assert estimate.interval.hi == 1.0


def test_the_auc_lower_bound_is_clipped_at_zero() -> None:
    """The other side of the same normal approximation, and it had no input until this one.

    `pos=(0.1, 0.9)` against `neg=(0.5, 0.5)`: one positive loses both comparisons and one wins
    both, so `V10` is `[0, 1]` and the variance is as large as two items allow. The unclipped lower
    bound is -0.48, and a published lower bound of -48% on a rank separation is unreadable.
    """
    estimate = roc_auc(sample((0.1, 0.9), (0.5, 0.5)))
    assert estimate.auc == 0.5
    assert estimate.auc - Z_95 * math.sqrt(estimate.variance) < 0.0
    assert estimate.interval.lo == 0.0


def test_auc_refuses_a_non_finite_score() -> None:
    """`p_injection` is a probability, so an infinity here means something upstream divided by
    zero. It is refused at the record rather than propagated into a rank."""
    with pytest.raises(ValueError):
        AucSample(("p0", "p1"), (0.5, float("inf")), ("n0", "n1"), (0.1, 0.2))


def test_percentile_types_its_samples_before_it_sorts_them() -> None:
    """The ordering matters and the input that proves it is a string.

    `sorted([1, "a"])` raises `TypeError` -- exit 1, an unclassified failure -- so a type check
    placed after the sort can never see this input. `[1, 2.5, 3]` sorts fine and would have made
    the wrong order look correct.
    """
    with pytest.raises(StatisticUndefined) as caught:
        nearest_rank_percentile([1, "a", 3], P50)  # type: ignore[list-item]
    assert "integer nanoseconds" in str(caught.value)


def test_auc_from_ranks_and_from_structural_components_agree_including_on_a_tie() -> None:
    """The one input where a midrank implementation and a pairwise one diverge if either drops the
    half: `0.4` appears in both classes."""
    pos, neg = (0.9, 0.4, 0.4), (0.4, 0.1, 0.8, 0.2)
    estimate = roc_auc(sample(pos, neg))
    assert midrank_auc(pos, neg) == Fraction(3, 4)
    assert estimate.auc == float(midrank_auc(pos, neg))
    assert_close(estimate.interval.lo, 0.3527120468118526109615471, "auc lo with a tie")


def test_auc_of_a_wholly_tied_cell_is_one_half_with_a_zero_width_interval() -> None:
    """Both structural components are constant, so their sample variance is genuinely zero. The
    estimator is saying the observed comparisons carry no information about sampling variability.
    A floor invented here would be a number no method produced."""
    estimate = roc_auc(sample((1.0, 1.0), (1.0, 1.0)))
    assert estimate.auc == 0.5
    assert estimate.variance == 0.0
    assert (estimate.interval.lo, estimate.interval.hi) == (0.5, 0.5)


def test_auc_of_a_perfectly_separated_cell_is_one_with_a_zero_width_interval() -> None:
    """The same limit at the other end, and the one this corpus is most likely to reach: the
    layer's scores saturate at `p = 1`."""
    estimate = roc_auc(sample((5.0, 6.0, 7.0), (1.0, 2.0)))
    assert estimate.auc == 1.0
    assert (estimate.interval.lo, estimate.interval.hi) == (1.0, 1.0)


@pytest.mark.parametrize(
    "pos,neg",
    [((), (1.0, 2.0)), ((1.0, 2.0), ())],
)
def test_auc_refuses_an_empty_class(pos: tuple[float, ...], neg: tuple[float, ...]) -> None:
    with pytest.raises(StatisticUndefined) as caught:
        roc_auc(sample(pos, neg))
    assert "empty class" in str(caught.value)


def test_auc_refuses_a_single_item_class() -> None:
    """One structural component has no sample variance at all, and `0.0` there would read as
    'no variability' where the truth is 'not estimable'."""
    with pytest.raises(StatisticUndefined) as caught:
        roc_auc(sample((1.0,), (0.0, 0.5)))
    assert "not estimable" in str(caught.value)


def test_auc_sample_refuses_mismatched_ids_and_scores() -> None:
    with pytest.raises(ValueError):
        AucSample(("p0", "p1"), (0.5,), ("n0", "n1"), (0.1, 0.2))


def test_auc_sample_refuses_a_repeated_id() -> None:
    with pytest.raises(ValueError):
        AucSample(("p0", "p0"), (0.5, 0.6), ("n0", "n1"), (0.1, 0.2))


# --- delta AUC --------------------------------------------------------------------------------------


def test_delta_auc_subtracts_the_two_and_carries_the_paired_covariance() -> None:
    """The two conditions over one item set. `Var(D) = Var(A) + Var(B) - 2Cov(A, B)`, and the
    covariance is what makes this narrower than treating them as two independent AUCs.

    Both cells overlap -- one positive falls below a negative under each condition -- so neither
    AUC is degenerate and both structural variances are non-zero, which is what makes the
    covariance term observable at all.
    """
    canon_off = sample((0.30, 0.40, 0.35, 0.22), (0.20, 0.10, 0.25, 0.15))
    canon_on = sample((0.90, 0.85, 0.70, 0.18), (0.20, 0.12, 0.28, 0.14))

    result = delta_auc(canon_on, canon_off)
    assert (result.first.auc, result.second.auc) == (0.875, 0.9375)
    assert result.delta == pytest.approx(-0.0625)
    assert result.interval.method == DELTA_AUC_STRUCTURAL

    # What the pairing buys, measured: dropping the covariance would put the half-width at 0.3317
    # instead of 0.1732, so an interval that already covers zero would cover almost twice as much
    # of the range on either side of it.
    independent_half = Z_95 * math.sqrt(result.first.variance + result.second.variance)
    assert result.covariance == pytest.approx(0.010416666666666666)
    assert independent_half == pytest.approx(0.33172547254389)
    assert result.interval.width / 2 == pytest.approx(0.17323797804370974)
    assert result.interval.width / 2 < independent_half


def test_delta_auc_of_a_condition_against_itself_is_zero_with_a_zero_width_interval() -> None:
    """`Var(A) + Var(A) - 2Var(A)` is exactly zero, and the covariance term is what makes it so.
    Drop the covariance and this interval opens up around a difference that is identically zero."""
    one = sample((0.30, 0.40, 0.35, 0.50), (0.20, 0.10, 0.25, 0.15))
    result = delta_auc(one, one)
    assert result.delta == 0.0
    assert result.interval.lo == result.interval.hi == 0.0


def test_delta_auc_refuses_two_samples_over_different_items() -> None:
    first = AucSample(("p0", "p1"), (0.9, 0.8), ("n0", "n1"), (0.1, 0.2))
    second = AucSample(("p0", "p9"), (0.4, 0.3), ("n0", "n1"), (0.1, 0.2))
    with pytest.raises(StatisticUndefined) as caught:
        delta_auc(first, second)
    assert "positive item 1 is 'p1' in the first and 'p9' in the second" in str(caught.value)


def test_delta_auc_refuses_the_same_items_in_a_different_order() -> None:
    """Order is part of the pairing: the covariance walks the two vectors together, so the same
    ids shuffled pairs each item's canon-on score with a different item's canon-off score while
    every length still matches."""
    first = AucSample(("p0", "p1"), (0.9, 0.8), ("n0", "n1"), (0.1, 0.2))
    second = AucSample(("p1", "p0"), (0.8, 0.9), ("n0", "n1"), (0.1, 0.2))
    with pytest.raises(StatisticUndefined) as caught:
        delta_auc(first, second)
    assert "same order" in str(caught.value)


def test_delta_auc_refuses_a_negative_side_that_differs_in_length() -> None:
    first = AucSample(("p0", "p1"), (0.9, 0.8), ("n0", "n1"), (0.1, 0.2))
    second = AucSample(("p0", "p1"), (0.4, 0.3), ("n0", "n1", "n2"), (0.1, 0.2, 0.3))
    with pytest.raises(StatisticUndefined) as caught:
        delta_auc(first, second)
    assert "negative items" in str(caught.value)


# --- the interval record itself ---------------------------------------------------------------------


def test_an_interval_cannot_be_built_under_a_method_nobody_declared() -> None:
    """This is how the epic's last criterion is discharged: the method name is not a field somebody
    remembers to fill, it is a precondition of the value existing."""
    with pytest.raises(ValueError) as caught:
        Interval(0.1, 0.2, "wilson")
    assert "wilson-score" in str(caught.value)


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(ValueError):
        Interval(0.9, 0.1, WILSON_SCORE)


def test_an_interval_refuses_a_nan_bound() -> None:
    """A `nan` in a results file is a published number a reader cannot tell from a real one."""
    with pytest.raises(ValueError):
        Interval(float("nan"), 0.2, WILSON_SCORE)


def test_every_estimator_returns_a_method_from_the_closed_vocabulary() -> None:
    produced = {
        wilson_interval(5, 10).method,
        newcombe_paired_interval(POSITIVELY_CORRELATED).method,
        roc_auc(sample((3.0, 1.0), (2.0, 0.0))).interval.method,
        delta_auc(
            sample((0.9, 0.8, 0.7), (0.1, 0.2, 0.3)),
            sample((0.5, 0.4, 0.6), (0.1, 0.2, 0.3)),
        ).interval.method,
    }
    assert produced == {WILSON_SCORE, NEWCOMBE_PAIRED, AUC_STRUCTURAL, DELTA_AUC_STRUCTURAL}


def test_the_new_abort_declares_exit_code_29_and_declares_it_once() -> None:
    codes = declared_exit_codes()
    assert codes[29] is StatisticUndefined
    assert StatisticUndefined.exit_code == 29


# --- the rejected method, measured rather than asserted -----------------------------------------------


def saturating_scores(n_positive: int, n_negative: int) -> tuple[list[float], list[float]]:
    """Deterministic score vectors with the shape this corpus has: positives piled up against 1.0.

    No RNG, so the numbers below are reproducible from the source rather than from a seed.
    """
    positives = [min(1.0, 0.55 + (i % 37) / 40.0) for i in range(n_positive)]
    negatives = [(j % 23) / 30.0 for j in range(n_negative)]
    return positives, negatives


def test_hanley_mcneil_is_anti_conservative_on_the_input_the_epic_describes() -> None:
    """Measured 2026-08-30. At `n_pos = 200, n_neg = 20` the closed form gives a standard error
    0.81 times the empirical one -- an interval about a fifth too narrow, in the overclaiming
    direction, on scores that saturate at 1.0 the way this corpus's do."""
    positives, negatives = saturating_scores(200, 20)
    estimate = roc_auc(sample(tuple(positives), tuple(negatives)))
    closed_form = rejected_hanley_mcneil_variance(estimate.auc, 200, 20)

    ratio = math.sqrt(closed_form) / math.sqrt(estimate.variance)
    assert ratio < 1.0
    assert 0.80 < ratio < 0.83


def test_the_direction_reverses_and_the_docstring_says_so() -> None:
    """The input that keeps the rejection from resting on a claim about widths.

    The epic's criterion says the rejected intervals are 18 to 30% too narrow. On four of the five
    constructed cells measured on 2026-08-30 the closed form was **wider**, not narrower. That
    figure is a coverage result under a simulation, not a statement about any particular interval,
    and the rejection rests on the assumption `Q1` and `Q2` encode rather than on a width that
    changes sign with the cell.
    """
    wider = 0
    for n_positive, n_negative in ((400, 400), (1024, 400), (40, 400), (20, 200)):
        positives, negatives = saturating_scores(n_positive, n_negative)
        estimate = roc_auc(sample(tuple(positives), tuple(negatives)))
        closed_form = rejected_hanley_mcneil_variance(estimate.auc, n_positive, n_negative)
        if math.sqrt(closed_form) > math.sqrt(estimate.variance):
            wider += 1
    assert wider == 4
