"""Every interval this project publishes, written out by name, with no numerical dependency.

A wrong boundary in `harness/score.py` aborts. A wrong label aborts. **A wrong interval publishes.**
It produces a table that looks exactly like a correct one, states a sharper claim than the data
supports, and is found by the reader this artifact exists to convince. That asymmetry is why every
estimator here is one named method rather than a call into a library: "which variant of the Wilson
interval did that version give us" is not a question a reader should answer by reading somebody
else's changelog.

It matters more than usual here because Epic 4's four falsification conditions are all decided by
whether an interval covers zero. An interval that is too narrow makes a negative result **fail to
trigger** -- which is precisely the outcome this epic is constructed to be able to reach.

**The method name rides on the value.** Every function returns a `schema.Interval`, and an
`Interval` cannot be constructed without a method drawn from `schema.INTERVAL_METHODS`. The epic
asks that the method names reach the results file so a reader knows which interval they are
reading; a `methods` block beside the numbers discharges that only while somebody keeps it in step,
and when it drifts nothing fails. Here they are inseparable.

**This module is pure**: the standard library, `nbc.errors` and `nbc.schema`. No file, no socket,
no model, and no `numpy`. `Z_95` is a literal rather than a call into a quantile function, because
a quantile function is a second implementation of a constant, and two spellings of 1.96 in one
repository are two credibility claims.

**Four methods, and why each rather than its neighbour.**

*Wilson score, plain.* Not Wald, whose interval leaves [0,1] and collapses to zero width at `k = 0`
-- and `k = 0` is where a per-class false-positive rate is expected to land. Not the
continuity-corrected variant, which is wider and is a different published claim; the choice is the
recorded method name, so swapping it fails a golden test instead of moving numbers quietly.

*Nearest rank for percentiles, on exact rationals.* `p = sorted[ceil(q * n) - 1]`, so a reported
latency is always a sample some clock actually produced. `q` is a `fractions.Fraction` and a float
is refused: `math.ceil(0.07 * 100)` is 8 where the exact rank is 7, measured on 2026-08-30 over
every `j/100` and `j/1000` at `n` up to 20000. What the same search did *not* find is worth as much
-- for `q = 1/2` and `q = 19/20`, the two percentiles this epic reports, there is no disagreement at
any `n` up to 200000. So this is not a bug being repaired; it is a class of bug made unreachable
before story 4-5 declares a third percentile, and it costs one import.

*Newcombe's method 10 for paired proportions.* Canon-on and canon-off are measured on the same
items, and treating them as independent samples is the error that widens the interval, which makes
N1 harder to trigger and errs toward never declaring a negative result. Measured on the two tables
in the suite, that direction holds where it matters and reverses elsewhere: with `a=40 b=10 c=2
d=48` the unpaired interval is 0.2704 wide against the paired 0.1354, and with `a=5 b=45 c=45 d=5`
the unpaired is 0.2720 against the paired 0.3649. The conditions this artifact pairs are strongly
positively correlated, which is the first table; stating the direction as universal would be a
claim the second refutes.

*Empirical structural components for AUC, and Hanley-McNeil rejected.* `Q1 = A/(2-A)` and
`Q2 = 2A^2/(1+A)` are closed forms derived under a bi-negative-exponential score distribution. They
are not estimated from the data and they do not cancel under class imbalance. This corpus is
imbalanced by construction and its scores pile up at `p = 1`, about as far from that generative
model as a score distribution goes. `V10` and `V01` assume nothing: they are computed from the
comparisons actually observed.

**What did not reproduce, recorded rather than repeated.** The epic's criterion says the rejected
method delivers intervals "18 to 30% too narrow". Measured on 2026-08-30 across five constructed
score sets with saturation at `p = 1`, Hanley-McNeil was narrower once (`n_pos=200, n_neg=20`,
ratio 0.81) and **wider** on the other four (1.11 to 1.81). The epic's figure is a *coverage* result
under a simulation, not a statement about the width of any particular interval, and restating it as
the latter would be a claim recorded once and then cited as a fact about this repository's numbers.
The decision does not move: a variance that is a closed form fitted to a distribution this corpus
does not have is the wrong tool whichever way it errs on a given input, and for an artifact whose
value is surviving a skeptical reviewer an anti-conservative interval is the worst available error
-- naming it in the caveats does not cover it. `rejected_hanley_mcneil_variance` ships so the
comparison is a test rather than a paragraph, and an AST scan refuses any reference to it from
elsewhere under `src/nbc/`.

**What this module deliberately is not.** No cell, no contrast, no verdict -- 4-3 and 4-6. No file
-- 4-7 owns `results.json`. No clock -- 4-5 owns the timing pass and calls the percentile here. The
one thing this story hands forward is a constraint: after it, a number reaches the results file with
the name of the method that produced it, or it does not reach it at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from nbc.errors import NbcError
from nbc.schema import (
    AUC_STRUCTURAL,
    MOVER_DIFFERENCE,
    DELTA_AUC_STRUCTURAL,
    NEWCOMBE_PAIRED,
    WILSON_SCORE,
    Interval,
    PairedCount,
)

__all__ = [
    "AucDelta",
    "AucEstimate",
    "AucSample",
    "P50",
    "P95",
    "StatisticUndefined",
    "Z_95",
    "delta_auc",
    "mover_difference_interval",
    "nearest_rank_percentile",
    "newcombe_paired_interval",
    "rejected_hanley_mcneil_variance",
    "roc_auc",
    "unpaired_square_and_add_interval",
    "wilson_interval",
]


class StatisticUndefined(NbcError, exit_code=29):
    """A statistic was asked for on an input where it has no value, so nothing is returned.

    Code 29 because 3 through 28 are taken. One class for every undefined input, because the
    remedy is the same in each case -- the caller assembled a cell that has no data in it -- and
    because a caller that wants to distinguish them can read the message.

    It aborts rather than returning a wide interval or a `nan`. A `nan` that reaches a JSON file is
    a published number a reader cannot tell from a real one, and a very wide interval invented for
    an empty cell answers "we could not measure this" with "we measured this and learned nothing",
    which are different sentences and only one of them is true.

    The inputs that produce it, each with the test that fires it:

    - `wilson_interval(0, 0)`, and `wilson_interval(11, 10)` where more events than trials means
      the caller counted something twice;
    - `nearest_rank_percentile((), P50)`, a percentile over no samples;
    - a `float` quantile, refused at the door rather than converted (see the module docstring);
    - a non-integer timing sample, which is not something `perf_counter_ns` produces;
    - `newcombe_paired_interval` over an all-zero table;
    - `roc_auc` with an empty class, or with one item in a class, where the sample variance of one
      structural component does not exist;
    - `delta_auc` over two samples whose item ids are not the same tuple in the same order.
    """


Z_95: Final[float] = 1.959963984540054
"""The standard normal 97.5th percentile, to float64.

A literal rather than `statistics.NormalDist().inv_cdf(0.975)`, which would be a second
implementation of a constant and would let a Python version move every interval in the table.
Every estimator in this module that needs a normal quantile reads it from here.
"""

P50: Final[Fraction] = Fraction(1, 2)
P95: Final[Fraction] = Fraction(19, 20)
"""The two percentiles Epic 4 reports, as exact rationals. See the module docstring for why not
floats."""


# --- one proportion ------------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = Z_95) -> Interval:
    """The plain Wilson score interval for `k` successes in `n` trials.

    Written in the `(k + z^2/2) / (n + z^2)` arrangement rather than the `p-hat` one. The two are
    algebraically identical and this one has no division by `n` inside the square root, so it does
    not lose digits at the small `k` and large `n` this table's false-positive rates live at. The
    goldens are computed in the other arrangement in exact arithmetic, which is what makes the pair
    a comparison rather than a restatement.

    `z` is a parameter only so a test can pin it to 1.96 and check three published textbook values
    -- a second source rather than a second arrangement. Nothing in the published path passes it.
    """
    _refuse_a_count("k", k)
    _refuse_a_count("n", n)
    if n == 0:
        raise StatisticUndefined(
            "a Wilson interval over 0 trials is undefined; a rate with no denominator is not a "
            "rate this run may publish"
        )
    if k > n:
        raise StatisticUndefined(
            f"a Wilson interval needs k <= n, got k={k} and n={n}; more events than trials means "
            f"something was counted twice"
        )

    z2 = z * z
    denominator = n + z2
    centre = (k + z2 / 2) / denominator
    half = z / denominator * math.sqrt(k * (n - k) / n + z2 / 4)

    lo = centre - half
    hi = centre + half

    # The two boundaries are exact in the algebra and inexact in float64, in both directions. At
    # k = 0 the centre and the half-width are equal, so their difference is zero exactly -- but the
    # subtraction lands on either side of it: -1e-18 at n = 10 and +4.34e-19 at n = 500, measured.
    # A one-sided clamp catches only the first, and the second is the one that publishes, because
    # a false-positive rate of "0 in 500" would carry a lower bound of 0.00000000000000000043
    # rather than 0. Set the boundary rather than clamp toward it: at k = 0 the interval starts at
    # 0 and at k = n it ends at 1, and those are the algebra rather than a tidied number.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return Interval(max(0.0, lo), min(1.0, hi), WILSON_SCORE)


# --- one percentile ------------------------------------------------------------------------------


def nearest_rank_percentile(samples: Sequence[int], q: Fraction) -> int:
    """The nearest-rank percentile of `samples`: `sorted[ceil(q * n) - 1]`.

    Returns an observed sample, always, and an `int`. No interpolation and no mean of two
    neighbours: `perf_counter_ns` counts nanoseconds and a reported latency that no clock produced
    is a fabricated observation, however defensible the average.

    `q` is a `Fraction` so `ceil(q * n)` is exact integer arithmetic. A `float` is refused rather
    than converted -- see the module docstring for the measured input where the two disagree.
    """
    if isinstance(q, bool) or not isinstance(q, Fraction):
        raise StatisticUndefined(
            f"the quantile must be a fractions.Fraction, got {q!r} ({type(q).__name__}); a float "
            f"quantile makes ceil(q * n) approximate, and math.ceil(0.07 * 100) is 8 where the "
            f"nearest rank is 7"
        )
    if not 0 < q <= 1:
        raise StatisticUndefined(
            f"the quantile must satisfy 0 < q <= 1, got {q}; rank ceil(q * n) is 0 at q = 0, "
            f"which addresses no sample"
        )

    # Typed before sorting, not after. `sorted([1, "a"])` raises TypeError -- an undeclared abort
    # with exit 1 -- and a check placed after the sort could never reach the input that produces
    # it. `[1, 2.5, 3]` sorts fine and would have made the misordered version look correct.
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise StatisticUndefined(
                f"timing samples are integer nanoseconds, got {sample!r} "
                f"({type(sample).__name__})"
            )

    ordered = sorted(samples)
    if not ordered:
        raise StatisticUndefined(
            "a percentile over no samples is undefined; an empty timing pass is a pass that did "
            "not run"
        )

    # `-(-x // 1)` is ceil on a Fraction without leaving exact arithmetic; math.ceil would too,
    # but writing the fraction's own floor division makes it visible that nothing became a float.
    rank = int(-((-q * len(ordered)) // 1))
    return ordered[rank - 1]


# --- one paired difference -----------------------------------------------------------------------


def newcombe_paired_interval(counts: PairedCount, z: float = Z_95) -> Interval:
    """Newcombe's method 10 for the difference between two proportions on the same items.

    A Wilson interval for each marginal, joined by square-and-add with the correlation the pairing
    supplies:

        phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        lo  = theta - sqrt((p1-l1)^2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)^2)
        hi  = theta + sqrt((u1-p1)^2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)^2)

    The `- n/2` on a positive `ad - bc` is Newcombe's, and it is one-sided by design: it damps a
    positive association, which widens the interval, and leaves a negative one alone. Damping both
    would narrow the interval on negatively-associated tables, which is the direction that must
    never be taken for free.

    The result is not clamped to [-1, 1] beyond what the algebra already gives, and is never
    clamped away from zero: whether this interval covers zero is exactly what N1 reads.
    """
    if counts.n == 0:
        raise StatisticUndefined(
            "a paired interval over an empty 2x2 table is undefined; no items were compared"
        )

    n = counts.n
    p1 = counts.first_positive / n
    p2 = counts.second_positive / n
    theta = counts.theta

    first = wilson_interval(counts.first_positive, n, z)
    second = wilson_interval(counts.second_positive, n, z)

    marginal_product = (
        counts.first_positive
        * (counts.c + counts.d)
        * counts.second_positive
        * (counts.b + counts.d)
    )
    if marginal_product == 0:
        # One condition was positive on every item or on none, so its marginal has no variation
        # and no correlation with anything. Newcombe's own degenerate case; phi = 0 reduces the
        # method to plain square-and-add, which is the honest answer rather than a division by
        # zero dressed up.
        phi = 0.0
    else:
        numerator = float(counts.a * counts.d - counts.b * counts.c)
        if numerator > 0:
            numerator = max(numerator - n / 2, 0.0)
        phi = numerator / math.sqrt(marginal_product)

    down = p1 - first.lo
    up_other = second.hi - p2
    delta = math.sqrt(max(down * down - 2 * phi * down * up_other + up_other * up_other, 0.0))

    up = first.hi - p1
    down_other = p2 - second.lo
    epsilon = math.sqrt(
        max(up * up - 2 * phi * up * down_other + down_other * down_other, 0.0)
    )

    return Interval(theta - delta, theta + epsilon, NEWCOMBE_PAIRED)


def unpaired_square_and_add_interval(counts: PairedCount, z: float = Z_95) -> Interval:
    """The same square-and-add with `phi` forced to zero: what treating the two conditions as two
    independent samples of the same size would publish.

    It exists so the reason for pairing is a comparison rather than a sentence, and it is the one
    function here whose two sides come from genuinely different places -- one reads the association
    in the table, the other refuses to. Nothing in the published path calls it; the tests do, in
    both directions, because the width difference reverses sign on a negatively-associated table
    and a claim stated as universal would be false there.

    It returns a `newcombe-paired-score` interval because that is the method it is a variant of,
    and because inventing a fifth vocabulary entry for a diagnostic would put a name in the results
    file's method vocabulary that no published number ever carries.
    """
    if counts.n == 0:
        raise StatisticUndefined(
            "a paired interval over an empty 2x2 table is undefined; no items were compared"
        )

    n = counts.n
    p1 = counts.first_positive / n
    p2 = counts.second_positive / n
    first = wilson_interval(counts.first_positive, n, z)
    second = wilson_interval(counts.second_positive, n, z)

    delta = math.hypot(p1 - first.lo, second.hi - p2)
    epsilon = math.hypot(first.hi - p1, p2 - second.lo)
    return Interval(counts.theta - delta, counts.theta + epsilon, NEWCOMBE_PAIRED)


def mover_difference_interval(
    first: float,
    first_interval: Interval,
    second: float,
    second_interval: Interval,
) -> Interval:
    """The difference `first - second` of two **independent** estimates that arrive with intervals.

        lo = (a - b) - sqrt((a - l_a)^2 + (u_b - b)^2)
        hi = (a - b) + sqrt((u_a - a)^2 + (b - l_b)^2)

    **Why this and not a variance sum.** N1's criterion writes `Var(D) = Var(A) + Var(B)`, which is
    right for two independent estimates and assumes both come with variances. These do not: they
    are Newcombe intervals, asymmetric by construction, and the asymmetry is why that method was
    chosen -- it is what makes the interval behave near 0 and 1. There is no single variance to add.
    MOVER-R is the interval-form of the same argument: it combines the distance from each estimate
    to the bound that matters for each side of the difference, squared and added. It **reduces to
    the variance sum exactly when both intervals are symmetric**, which the test asserts.

    **Independence is the caller's to establish and is not checked here.** N1's two deltas are
    computed over disjoint item sets -- benign items and attack items -- which is what makes the
    combination valid. Handed two intervals over overlapping populations this would silently return
    an interval that is too wide, and too wide is the direction that makes a negative result harder
    to declare. The caller that uses it says why its inputs are disjoint.
    """
    for name, value in (("first", first), ("second", second)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StatisticUndefined(f"{name} must be a real number, got {value!r}")
    for name, value, interval in (
        ("first", first, first_interval),
        ("second", second, second_interval),
    ):
        if not isinstance(interval, Interval):
            raise StatisticUndefined(f"{name}_interval must be an Interval, got {interval!r}")
        if not interval.lo <= value <= interval.hi:
            raise StatisticUndefined(
                f"the {name} estimate {value!r} lies outside its own interval {interval!r}; a "
                f"point estimate its interval does not cover is not an estimate this can combine"
            )

    down = math.hypot(first - first_interval.lo, second_interval.hi - second)
    up = math.hypot(first_interval.hi - first, second - second_interval.lo)
    difference = first - second
    return Interval(difference - down, difference + up, MOVER_DIFFERENCE)


# --- one AUC, and the difference of two ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AucSample:
    """The scores of one cell, split by gold label, each score still attached to its item.

    The ids are carried rather than dropped because `delta_auc` pairs two conditions **by item**,
    and the only honest way to check that two samples describe the same items is to compare the id
    tuples. Two conditions over one set of 500 benign items and two conditions over two different
    sets of 500 are the same shape; without the ids they are indistinguishable, and the covariance
    term would silently be computed across items that have nothing to do with each other.
    """

    positive_ids: tuple[str, ...]
    positive_scores: tuple[float, ...]
    negative_ids: tuple[str, ...]
    negative_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        for side in ("positive", "negative"):
            ids = getattr(self, f"{side}_ids")
            scores = getattr(self, f"{side}_scores")
            if not isinstance(ids, tuple) or not all(
                isinstance(item, str) and item for item in ids
            ):
                raise ValueError(f"{side}_ids must be a tuple of non-empty ids, got {ids!r}")
            if not isinstance(scores, tuple):
                raise ValueError(f"{side}_scores must be a tuple, got {scores!r}")
            for score in scores:
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"{side}_scores holds a non-number: {score!r}")
                value = float(score)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError(f"{side}_scores holds a non-finite value: {score!r}")
            if len(ids) != len(scores):
                raise ValueError(
                    f"{side}_ids has {len(ids)} entries and {side}_scores has {len(scores)}"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(f"{side}_ids repeats an id; one item is one comparison")


@dataclass(frozen=True, slots=True)
class AucEstimate:
    """One ROC AUC with its interval, the counts it rests on, its variance, and its ties.

    `tied_pairs` and `total_pairs` ride out because the kernel already computes them and throwing
    them away costs a later story its evidence. Story 4.4's saturation limit is a claim about *why*
    an AUC moved, and the honest evidence for it is the share of comparisons that were ties rather
    than orderings: "8 of these 16 comparisons were decided by the ceiling" is checkable where
    "saturation may have affected this" is not. Recovering it afterwards would mean re-reading the
    score file, and `harness/aggregate.py` is its only reader.
    """

    auc: float
    interval: Interval
    n_positive: int
    n_negative: int
    variance: float
    tied_pairs: int = 0
    total_pairs: int = 0


@dataclass(frozen=True, slots=True)
class AucDelta:
    """The difference between two AUCs measured on the same items, with its interval."""

    delta: float
    interval: Interval
    first: AucEstimate
    second: AucEstimate
    covariance: float


def _psi(x: float, y: float) -> float:
    """The Mann-Whitney kernel: 1 if the positive outscores the negative, 1/2 on a tie.

    The half is what "midranks for ties" means at the pairwise level, and it is where the layer's
    saturation at `p = 1` shows up in the AUC: two items pinned at 1.0 contribute 1/2 rather than 1.
    """
    if x > y:
        return 1.0
    if x == y:
        return 0.5
    return 0.0


def _structural_components(
    positives: Sequence[float], negatives: Sequence[float]
) -> tuple[list[float], list[float]]:
    """`V10` per positive and `V01` per negative, each the mean kernel against the other class."""
    n_pos = len(positives)
    n_neg = len(negatives)
    v10 = [sum(_psi(x, y) for y in negatives) / n_neg for x in positives]
    v01 = [sum(_psi(x, y) for x in positives) / n_pos for y in negatives]
    return v10, v01


def _tied_pairs(positives: Sequence[float], negatives: Sequence[float]) -> int:
    """How many of the `n_pos * n_neg` comparisons were ties rather than orderings.

    Counted from the same equality the kernel uses, so a change to what counts as a tie moves both
    together. This is the input story 4.4's saturation limit reads.
    """
    return sum(1 for x in positives for y in negatives if x == y)


def _sample_variance(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _sample_covariance(left: Sequence[float], right: Sequence[float]) -> float:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    return sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _refuse_a_thin_sample(sample: AucSample) -> None:
    if not sample.positive_ids:
        raise StatisticUndefined(
            "an AUC needs at least one positive item; this cell has none, and a rank separation "
            "over an empty class is not a number"
        )
    if not sample.negative_ids:
        raise StatisticUndefined(
            "an AUC needs at least one negative item; this cell has none, and a rank separation "
            "over an empty class is not a number"
        )
    for side, ids in (("positive", sample.positive_ids), ("negative", sample.negative_ids)):
        if len(ids) < 2:
            raise StatisticUndefined(
                f"the structural-component variance needs at least 2 {side} items, got "
                f"{len(ids)}; the sample variance of one component does not exist, and 0.0 there "
                f"would read as 'no variability' rather than 'not estimable'"
            )


def roc_auc(sample: AucSample) -> AucEstimate:
    """The Mann-Whitney U statistic with midranks for ties, and its structural-component interval.

    `A` is `mean(V10)`, which is the U statistic over `n_pos * n_neg` and is identical to the
    midrank formula `(sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`. The test
    computes it both ways on a sample carrying a tie, because two arrangements of one definition is
    the cheapest place this project has to check that ties are handled the same in each.

    `Var(A) = var(V10)/n_pos + var(V01)/n_neg`, sample variance in both terms.

    **A zero-width interval is returned rather than widened.** With every comparison tied, or with
    perfect separation, both components are constant and their sample variance is genuinely zero.
    Some implementations substitute a floor; a floor invented here would be a number no method
    produced. The condition is visible in the output -- `lo` equal to `hi` -- and it is the
    estimator saying the observed comparisons carry no information about sampling variability,
    which is true and is what 4-4's resolution-limit criterion reads.

    **The clip to [0,1] applies to AUC and to nothing else here.** The interval is a normal
    approximation and reaches past 1 on a small sample; a published upper bound above 1 on a
    probability is not a wider claim, it is an unreadable one. A Wilson interval needs no clip, and
    a Newcombe difference is not clipped because it legitimately spans zero.
    """
    _refuse_a_thin_sample(sample)

    v10, v01 = _structural_components(sample.positive_scores, sample.negative_scores)
    auc = sum(v10) / len(v10)
    variance = _sample_variance(v10) / len(v10) + _sample_variance(v01) / len(v01)

    half = Z_95 * math.sqrt(variance)
    interval = Interval(max(0.0, auc - half), min(1.0, auc + half), AUC_STRUCTURAL)
    return AucEstimate(
        auc=auc,
        interval=interval,
        n_positive=len(v10),
        n_negative=len(v01),
        variance=variance,
        tied_pairs=_tied_pairs(sample.positive_scores, sample.negative_scores),
        total_pairs=len(v10) * len(v01),
    )


def delta_auc(first: AucSample, second: AucSample) -> AucDelta:
    """`A(first) - A(second)` over one item set, with the paired covariance in its variance.

    `Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)`, and the covariance comes from the same two
    vectors each AUC already produced:

        Cov(A, B) = cov(V10_A, V10_B)/n_pos + cov(V01_A, V01_B)/n_neg

    So a delta between two conditions is one more difference with one more interval, not a special
    case -- which is what lets 4-3 treat it as a `Delta` like any other.

    **The two samples must be over the same items in the same order**, and that is compared as an
    id tuple rather than as two lengths. Order is part of the pairing: the covariance walks the two
    vectors together, and the same 500 ids shuffled would pair each item's canon-on score with a
    different item's canon-off score while every length still matched.
    """
    _refuse_a_thin_sample(first)
    _refuse_a_thin_sample(second)

    for side, left, right in (
        ("positive", first.positive_ids, second.positive_ids),
        ("negative", first.negative_ids, second.negative_ids),
    ):
        if left != right:
            raise StatisticUndefined(
                f"a paired AUC difference needs both conditions over the same {side} items in the "
                f"same order; {_first_difference(side, left, right)}"
            )

    first_estimate = roc_auc(first)
    second_estimate = roc_auc(second)

    v10_a, v01_a = _structural_components(first.positive_scores, first.negative_scores)
    v10_b, v01_b = _structural_components(second.positive_scores, second.negative_scores)
    covariance = _sample_covariance(v10_a, v10_b) / len(v10_a) + _sample_covariance(
        v01_a, v01_b
    ) / len(v01_a)

    variance = first_estimate.variance + second_estimate.variance - 2 * covariance
    # Rounding can drive a variance that is zero in exact arithmetic a hair below it -- two
    # identical conditions give Var(A) + Var(B) - 2Cov(A,B) with all three terms equal. Below zero
    # there is no square root, and the honest value there is zero.
    half = Z_95 * math.sqrt(max(variance, 0.0))

    delta = first_estimate.auc - second_estimate.auc
    return AucDelta(
        delta=delta,
        interval=Interval(delta - half, delta + half, DELTA_AUC_STRUCTURAL),
        first=first_estimate,
        second=second_estimate,
        covariance=covariance,
    )


def _first_difference(side: str, left: tuple[str, ...], right: tuple[str, ...]) -> str:
    """Where two id tuples first part company, named rather than left for the reader to diff."""
    if len(left) != len(right):
        return f"the first has {len(left)} {side} items and the second has {len(right)}"
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return f"{side} item {index} is {a!r} in the first and {b!r} in the second"
    return "they differ, but not at any position"  # pragma: no cover - unreachable by construction


# --- the method this project does not use ----------------------------------------------------------


def rejected_hanley_mcneil_variance(auc: float, n_positive: int, n_negative: int) -> float:
    """Hanley and McNeil's closed-form AUC variance. **Rejected; nothing published may call it.**

    It ships so the rejection is checkable rather than asserted. A method rejected in prose is a
    method nobody has compared against, and the module docstring's account of what did and did not
    reproduce rests on this function being callable from a test.

        Q1 = A / (2 - A)          Q2 = 2A^2 / (1 + A)
        Var = [A(1-A) + (n_pos - 1)(Q1 - A^2) + (n_neg - 1)(Q2 - A^2)] / (n_pos * n_neg)

    `Q1` and `Q2` are the reason. They are derived under a bi-negative-exponential score
    distribution, they are not estimated from the data, and they do not cancel under class
    imbalance -- which this corpus has by construction, with scores piled up at `p = 1`.

    An AST scan over `src/nbc/` refuses a reference to this name from any module but this one, so
    the rejection is enforced where a comment would only be read.
    """
    _refuse_a_count("n_positive", n_positive)
    _refuse_a_count("n_negative", n_negative)
    if n_positive == 0 or n_negative == 0:
        raise StatisticUndefined("an AUC variance needs both classes non-empty")

    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    return (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    ) / (n_positive * n_negative)


def _refuse_a_count(name: str, value: object) -> None:
    """A count is a non-negative `int`, and `True` is not one.

    `isinstance(True, int)` holds, so `wilson_interval(True, 10)` would quietly mean one success.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatisticUndefined(f"{name} counts items and must be an int, got {value!r}")
    if value < 0:
        raise StatisticUndefined(f"{name} must not be negative, got {value!r}")
