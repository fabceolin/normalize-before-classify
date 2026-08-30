"""The threshold-free summary, the two rejections behind it, and the four limits it does not escape.

Every number in this table so far is defensible only at one operating point. The sharpest objection
this artifact faces is not "your recall is wrong" -- it is **"your layer did not make attacks
recognizable, it shifted every score upward, and you picked the threshold afterwards"**. Recall at
a threshold cannot answer that, because a uniform shift moves it exactly as a genuine improvement
would.

**ROC AUC answers it, and the reason has to be stated correctly.** It summarises **rank separation
over the whole score range** rather than one operating point, so a shift that moves every score by
the same amount leaves it where it was while moving recall at any fixed threshold.

**That is not the reason an earlier draft gave, and the rejected one is recorded rather than
quietly replaced.** The earlier justification was invariance under monotone transformation of the
scores. It is **wrong here**: the layer does not transform scores, it changes the *text* and
re-scores every item. Two items can swap order under it and no invariance theorem applies to a
re-scoring. That argument is worth recording precisely because it reads as rigorous -- it would
survive review by anyone who did not check what the layer actually does.

**PR AUC is rejected too, for a different reason.** A precision-recall summary depends on
prevalence, and this corpus's prevalence is **constructed**: the attack and benign halves are drawn
to declared sizes. A PR number would therefore report a substantial amount of the corpus recipe
back to the reader as if it were a property of the layer.

**And the summary does not escape the corpus's problems -- it inherits them.** Four limits, each
emitted as a record attached to the cells it applies to rather than as prose in a file the reader
reaches afterwards:

*Saturation.* A shift that runs into the ceiling at `p = 1` manufactures ties, and ties move AUC
**down**. Measured 2026-08-30: four attacks and four benign items separated perfectly at AUC 1.00,
shifted up by 0.7 with a cap at 1.0, come out at **0.75** with 8 of 16 comparisons tied; shifted by
0.9, at **0.50** with all 16 tied. No ordering changed in either. This corpus is exactly where
saturation happens, so the tied share rides on every `Auc` cell and the finding reads it.

*Resolution.* One benign item moves a rate by `1/n`, and the low-false-positive region resolves in
steps that size. The finding computes it from the cell's own `n` rather than restating a figure,
because the per-class sample size lives in `pins.toml` and a transcribed number goes stale the
moment it is re-declared.

*Sign disagreement.* When ΔAUC and the threshold-table delta disagree in sign, the finding carries
both and **concludes from neither**. Turning it into a rule -- trust the AUC, or trust the table --
would discard exactly the information the requirement was added to surface.

*The bound chains.* The round-trip contract applies to benign items exactly as to attacks, so on a
bound chain the layer-on AUC **is** the clean AUC and ΔAUC comes out large and positive for
definitional reasons, whatever the layer does. The threshold-free summary inherits the
by-construction problem there rather than solving it.

**This module is pure and reads cells, not scores.** `harness/aggregate.py` is the only reader of
the score file and the only producer of cells; a summary that computed its own comparison would
make it a second producer. It writes nothing -- 4-7 owns `results.json` -- and it decides nothing:
whether a finding *triggers* anything is 4-6's, and this module only says what the numbers can
support.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from nbc.corpus.matrix import CHAIN_CLASS_BOUND
from nbc.errors import NbcError
from nbc.schema import (
    CONTRAST_CANON_ON_VS_OFF,
    DELTA_AUC_STRUCTURAL,
    FAMILY_ATTACK,
    NEWCOMBE_PAIRED,
    Auc,
    CellKey,
    Delta,
)

__all__ = [
    "ACCEPTED_JUSTIFICATION",
    "FINDING_BOUND_CHAIN",
    "FINDING_KINDS",
    "FINDING_RESOLUTION",
    "FINDING_SATURATION",
    "FINDING_SIGN_DISAGREEMENT",
    "REJECTED_JUSTIFICATIONS",
    "REJECTED_SUMMARIES",
    "SATURATION_TIE_SHARE",
    "SUMMARY_CHOICE",
    "SummaryFinding",
    "SummaryUnsupported",
    "bound_chain_findings",
    "findings",
    "resolution_findings",
    "saturation_findings",
    "sign_disagreement_findings",
]

SUMMARY_CHOICE: Final[str] = "roc_auc"
"""The threshold-free summary this project publishes, named once."""

REJECTED_SUMMARIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "pr_auc": (
            "A precision-recall summary depends on prevalence, and this corpus's prevalence is "
            "CONSTRUCTED: both halves are drawn to declared sizes. A PR number would report a "
            "substantial amount of the corpus recipe back to the reader as a property of the layer."
        ),
    }
)
"""Summaries considered and refused, each with the reason as a value rather than as a comment."""

REJECTED_JUSTIFICATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "monotone_invariance": (
            "An earlier draft justified ROC AUC by invariance under monotone transformation of "
            "the scores. That is wrong here. The layer does not transform scores: it changes the "
            "TEXT and re-scores every item, so two items can swap order and no invariance theorem "
            "applies to a re-scoring. Recorded rather than replaced, because the argument reads as "
            "rigorous and would survive a review by anyone who did not check what the layer does."
        ),
    }
)
"""Arguments for the choice that were made and are wrong. Kept, because a quietly replaced
justification is one the next reader re-derives."""

ACCEPTED_JUSTIFICATION: Final[str] = (
    "ROC AUC summarises rank separation over the whole score range rather than at one operating "
    "point, so a shift that moves every score by the same amount leaves it unchanged while moving "
    "recall at any fixed threshold. That is what makes it an answer to 'you only shifted the "
    "scores'."
)

FINDING_SATURATION: Final[str] = "saturation"
FINDING_RESOLUTION: Final[str] = "resolution"
FINDING_SIGN_DISAGREEMENT: Final[str] = "sign_disagreement"
FINDING_BOUND_CHAIN: Final[str] = "bound_chain_definitional"

FINDING_KINDS: Final[tuple[str, ...]] = (
    FINDING_SATURATION,
    FINDING_RESOLUTION,
    FINDING_SIGN_DISAGREEMENT,
    FINDING_BOUND_CHAIN,
)
"""The four limits, closed. A fifth arriving as a free string is a caveat nobody declared."""

SATURATION_TIE_SHARE: Final[float] = 0.05
"""The tied share above which a cell carries a saturation finding.

Not a threshold on the AUC and not a rule about when a number is wrong: it is where "worth telling
the reader that ties were involved" starts. One tie in twenty comparisons is already enough to move
an AUC by more than the resolution limit at any `n` this table reports, and the finding carries the
exact count so a reader applies their own judgement rather than this constant's.
"""


class SummaryUnsupported(NbcError, exit_code=31):
    """The cell set cannot carry a threshold-free summary, so the run has not answered the objection.

    Code 31 because 3 through 30 are taken. The one input that produces it: a cell set with no
    `Auc` in it at all. A run that published a threshold table and no threshold-free summary has
    not answered "you only shifted the scores", and saying so at the end of the run is better than
    discovering it in review.
    """


@dataclass(frozen=True, slots=True)
class SummaryFinding:
    """One limit, the cells it applies to, and the numbers it was computed from.

    `computed` carries the inputs rather than only the conclusion, for the reason every abort in
    this project names its inputs: a finding a reader cannot recompute is a finding they have to
    take on trust, and this artifact's whole value is not needing to be trusted.
    """

    kind: str
    keys: tuple[CellKey, ...]
    statement: str
    computed: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.kind not in FINDING_KINDS:
            raise ValueError(f"finding kind must be one of {FINDING_KINDS}, got {self.kind!r}")
        if not self.keys:
            raise ValueError("a finding names the cells it applies to; this one names none")
        if not all(isinstance(key, CellKey) for key in self.keys):
            raise ValueError(f"keys must all be CellKeys, got {self.keys!r}")
        if not self.statement.strip():
            raise ValueError(f"finding {self.kind!r} carries an empty statement")
        object.__setattr__(self, "computed", MappingProxyType(dict(self.computed)))

    def as_json_object(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "keys": [key.as_json_object() for key in self.keys],
            "statement": self.statement,
            "computed": dict(self.computed),
        }


def _aucs(cells: Iterable[object]) -> tuple[Auc, ...]:
    return tuple(cell for cell in cells if isinstance(cell, Auc))


def _deltas(cells: Iterable[object], method: str) -> tuple[Delta, ...]:
    return tuple(
        cell for cell in cells if isinstance(cell, Delta) and cell.interval.method == method
    )


def saturation_findings(cells: Sequence[object]) -> tuple[SummaryFinding, ...]:
    """One finding per `Auc` whose comparisons were substantially decided by ties.

    Ties are what a ceiling at `p = 1` produces, and they move AUC **down** without any ordering
    changing. The finding carries the count rather than a verdict, because a reader looking at
    "8 of 16 tied, AUC 0.75" can tell that case from a real loss of separation and a reader given
    only 0.75 cannot.
    """
    produced: list[SummaryFinding] = []
    for cell in _aucs(cells):
        if cell.total_pairs == 0:
            continue
        share = cell.tied_pairs / cell.total_pairs
        if share <= SATURATION_TIE_SHARE:
            continue
        produced.append(
            SummaryFinding(
                kind=FINDING_SATURATION,
                keys=(cell.key,),
                statement=(
                    f"{cell.tied_pairs} of this cell's {cell.total_pairs} attack-versus-benign "
                    f"comparisons were ties rather than orderings. A layer that pushes scores into "
                    f"the ceiling at p = 1 manufactures ties, and ties move AUC downward without "
                    f"any ordering changing, so a low value here is not on its own evidence that "
                    f"separation was lost."
                ),
                computed={
                    "auc": cell.value,
                    "tied_pairs": cell.tied_pairs,
                    "total_pairs": cell.total_pairs,
                    "tied_share": share,
                    "tie_share_reported_above": SATURATION_TIE_SHARE,
                },
            )
        )
    return tuple(produced)


def resolution_findings(cells: Sequence[object]) -> tuple[SummaryFinding, ...]:
    """One finding per `Auc`, stating what a single benign item is worth in that cell.

    `1/n` computed from the cell, never a transcribed figure: the per-class sample size lives in
    `pins.toml`, `tests/test_pins.py` refuses that literal in a test, and a number written into a
    published finding would go stale the moment the size is re-declared while still reading as a
    measurement.
    """
    produced: list[SummaryFinding] = []
    for cell in _aucs(cells):
        step = 1.0 / cell.n_negative
        produced.append(
            SummaryFinding(
                kind=FINDING_RESOLUTION,
                keys=(cell.key,),
                statement=(
                    f"With {cell.n_negative} benign items in this cell, a single one moves the "
                    f"false-positive rate by {step:.4%} -- {step * 100:.2f} percentage points -- so "
                    f"the low-false-positive region resolves in steps that size and a difference "
                    f"smaller than one item is not a difference."
                ),
                computed={
                    "n_negative": cell.n_negative,
                    "n_positive": cell.n_positive,
                    "one_item_moves_the_rate_by": step,
                },
            )
        )
    return tuple(produced)


def _column(key: CellKey) -> tuple[str, ...]:
    """The coordinates that identify the column a delta belongs to, benign class excluded.

    `canon_on` and `family` are out because both kinds of delta span them. `benign_class` is out
    too, and that is the whole subtlety of the pairing below: a ΔAUC is per benign class, and the
    threshold-table delta it is compared against -- attack recall -- has no benign class at all. A
    key that included it would match nothing and the disagreement finding would silently never
    fire, which is the failure mode a check like this is most prone to.
    """
    return (
        str(key.baseline),
        str(key.dressing_chain),
        str(key.chain_class),
        str(key.window_policy),
    )


def _sortable(key: CellKey) -> tuple[str, ...]:
    return (*_column(key), str(key.benign_class))


def sign_disagreement_findings(cells: Sequence[object]) -> tuple[SummaryFinding, ...]:
    """Where ΔAUC and the threshold-table delta point in opposite directions.

    Both are canon-on-versus-off. They are told apart by the method riding on the interval --
    `delta-auc-structural-components` against `newcombe-paired-score` -- which is a field rather
    than an inference about which function produced which cell.

    The threshold-table side is **attack recall**, and the pairing therefore ignores `benign_class`:
    a ΔAUC is per benign class and a recall delta has none, so one recall delta pairs with the two
    ΔAUCs in its column. A false-positive delta is not the other side of this comparison -- it
    answers a different question, and two questions are allowed to disagree.

    The finding states both and concludes from neither. They measure different things: a
    re-ranking confined to a narrow region can move recall at the threshold while leaving AUC flat,
    or the reverse. Resolving it here by preferring one would discard the fact the requirement was
    added to surface.
    """
    recall_deltas: dict[tuple[str, ...], Delta] = {}
    for delta in _deltas(cells, NEWCOMBE_PAIRED):
        if delta.contrast.kind != CONTRAST_CANON_ON_VS_OFF:
            continue
        if delta.key.family != FAMILY_ATTACK:
            # The threshold-table side of this comparison is attack RECALL: "did the layer catch
            # more attacks at the operating point". A false-positive delta answers a different
            # question and pairing it here would compare two things that are allowed to disagree.
            continue
        recall_deltas[_column(delta.key)] = delta

    pairs: list[tuple[Delta, Delta]] = []
    for auc_delta in _deltas(cells, DELTA_AUC_STRUCTURAL):
        if auc_delta.contrast.kind != CONTRAST_CANON_ON_VS_OFF:
            continue
        threshold_delta = recall_deltas.get(_column(auc_delta.key))
        if threshold_delta is not None:
            pairs.append((auc_delta, threshold_delta))

    produced: list[SummaryFinding] = []
    for auc_delta, threshold_delta in sorted(
        pairs, key=lambda pair: _sortable(pair[0].key)
    ):
        if auc_delta.value == 0.0 or threshold_delta.value == 0.0:
            continue
        if (auc_delta.value > 0) == (threshold_delta.value > 0):
            continue
        produced.append(
            SummaryFinding(
                kind=FINDING_SIGN_DISAGREEMENT,
                keys=(auc_delta.key, threshold_delta.key),
                statement=(
                    f"The threshold-free summary and the threshold table disagree in SIGN on this "
                    f"cell: ΔAUC is {auc_delta.value:+.6f} and the canon-on-versus-off difference "
                    f"at the threshold is {threshold_delta.value:+.6f}. They measure different "
                    f"things -- a re-ranking confined to a narrow region moves one and not the "
                    f"other -- and this report concludes from neither alone."
                ),
                computed={
                    "delta_auc": auc_delta.value,
                    "delta_auc_interval": auc_delta.interval.as_json_object(),
                    "threshold_delta": threshold_delta.value,
                    "threshold_delta_interval": threshold_delta.interval.as_json_object(),
                },
            )
        )
    return tuple(produced)


def bound_chain_findings(cells: Sequence[object]) -> tuple[SummaryFinding, ...]:
    """Where a ΔAUC sits on a bound chain, and therefore says less than it looks like it says.

    The round-trip contract applies to benign items exactly as to attacks, so on a bound chain the
    layer recovers the clean text on both halves: the layer-on AUC *is* the clean AUC, and ΔAUC
    comes out large and positive whatever the layer does. The threshold-free summary inherits the
    by-construction problem on those chains rather than escaping it.

    `matrix.chain_class` decided the axis when the cell was keyed; this reads that field rather
    than looking at a chain's name.
    """
    produced: list[SummaryFinding] = []
    for delta in _deltas(cells, DELTA_AUC_STRUCTURAL):
        if delta.key.chain_class != CHAIN_CLASS_BOUND:
            continue
        produced.append(
            SummaryFinding(
                kind=FINDING_BOUND_CHAIN,
                keys=(delta.key,),
                statement=(
                    f"This ΔAUC of {delta.value:+.6f} is on a bound chain, where the round-trip "
                    f"contract applies to benign items exactly as it does to attacks. The layer "
                    f"recovers the clean text on both halves, so the layer-on AUC IS the clean AUC "
                    f"and this difference is large and positive for definitional reasons. The "
                    f"threshold-free summary does not escape the by-construction problem here; it "
                    f"inherits it."
                ),
                computed={
                    "delta_auc": delta.value,
                    "chain_class": delta.key.chain_class,
                    "dressing_chain": delta.key.dressing_chain,
                },
            )
        )
    return tuple(produced)


def findings(cells: Sequence[object]) -> tuple[SummaryFinding, ...]:
    """Every limit the cell set carries, ordered by kind so two runs emit the same sequence."""
    if not _aucs(cells):
        raise SummaryUnsupported(
            "the cell set carries no AUC, so this run published a threshold table and no "
            "threshold-free summary; it has not answered 'you only shifted the scores upward', "
            "which is the objection the summary exists for"
        )

    produced = [
        *saturation_findings(cells),
        *resolution_findings(cells),
        *sign_disagreement_findings(cells),
        *bound_chain_findings(cells),
    ]
    order = {kind: index for index, kind in enumerate(FINDING_KINDS)}
    return tuple(
        sorted(
            produced,
            key=lambda finding: (
                order[finding.kind],
                tuple(_sortable(key) for key in finding.keys),
            ),
        )
    )
