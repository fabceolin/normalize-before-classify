"""The threshold-free summary's limits, and the two rejections behind the choice.

The cells here are built by hand rather than aggregated from a score file, because the sharp cases
are shapes a realistic corpus makes hard to arrange: an AUC whose comparisons are mostly ties, a
pair of deltas that disagree in sign, a bound chain beside a held-out one.

The saturation numbers are not invented. They were measured on 2026-08-30 by running
`stats.roc_auc` over four attacks and four benign items, perfectly separated at AUC 1.00, with a
uniform shift applied and capped at 1.0. Those inputs are in this file and the AUCs they produce
are asserted, so the limit the module states is a measurement the suite reproduces rather than a
claim it repeats.
"""

from __future__ import annotations

import pytest

from nbc.corpus.matrix import CHAIN_CLASS_BOUND, CHAIN_CLASS_HELD_OUT, CLEAN_CHAIN_NAME
from nbc.errors import declared_exit_codes
from nbc.harness.stats import AucSample, roc_auc
from nbc.harness.summary import (
    ACCEPTED_JUSTIFICATION,
    FINDING_BOUND_CHAIN,
    FINDING_KINDS,
    FINDING_RESOLUTION,
    FINDING_SATURATION,
    FINDING_SIGN_DISAGREEMENT,
    FINDING_WINDOWS_MATCHED,
    REJECTED_JUSTIFICATIONS,
    REJECTED_SUMMARIES,
    SATURATION_TIE_SHARE,
    SUMMARY_CHOICE,
    SummaryFinding,
    SummaryUnsupported,
    bound_chain_findings,
    findings,
    resolution_findings,
    saturation_findings,
    sign_disagreement_findings,
    windows_matched_findings,
)
from nbc.schema import (
    AUC_STRUCTURAL,
    AXIS_CANON_ON,
    AXIS_FAMILY,
    CONTRAST_ATTACKS_VS_BENIGN_CLASS,
    CONTRAST_CANON_ON_VS_OFF,
    DELTA_AUC_STRUCTURAL,
    FAMILY_ATTACK,
    NEWCOMBE_PAIRED,
    POPULATION_SINGLE_WINDOW,
    WILSON_SCORE,
    Auc,
    CellKey,
    Contrast,
    Count,
    Delta,
    Interval,
    Rate,
)

BASELINE = "primary"
POLICY = "shared"

AUC_CONTRAST = Contrast(CONTRAST_ATTACKS_VS_BENIGN_CLASS, "b_code", frozenset({AXIS_FAMILY}))
CANON_CONTRAST = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))
CANON_AUC_CONTRAST = Contrast(
    CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON, AXIS_FAMILY})
)


def auc_key(
    *,
    benign_class: str = "b_code",
    chain: str = CLEAN_CHAIN_NAME,
    chain_class: str = CHAIN_CLASS_BOUND,
    canon_on: bool | None = True,
) -> CellKey:
    return CellKey(
        baseline=BASELINE,
        dressing_chain=chain,
        chain_class=chain_class,
        window_policy=POLICY,
        canon_on=canon_on,
        family=None,
        benign_class=benign_class,
        contrast=AUC_CONTRAST if canon_on is not None else CANON_AUC_CONTRAST,
    )


def auc(
    value: float,
    *,
    n_positive: int = 4,
    n_negative: int = 4,
    tied_pairs: int = 0,
    key: CellKey | None = None,
) -> Auc:
    return Auc(
        value=value,
        interval=Interval(max(0.0, value - 0.1), min(1.0, value + 0.1), AUC_STRUCTURAL),
        n_positive=n_positive,
        n_negative=n_negative,
        tied_pairs=tied_pairs,
        total_pairs=n_positive * n_negative,
        key=key or auc_key(),
    )


def auc_delta(value: float, *, chain_class: str = CHAIN_CLASS_BOUND, chain: str = CLEAN_CHAIN_NAME) -> Delta:
    return Delta(
        value=value,
        interval=Interval(value - 0.05, value + 0.05, DELTA_AUC_STRUCTURAL),
        key=auc_key(chain=chain, chain_class=chain_class, canon_on=None),
    )


def recall_delta(value: float, *, chain: str = CLEAN_CHAIN_NAME, chain_class: str = CHAIN_CLASS_BOUND) -> Delta:
    return Delta(
        value=value,
        interval=Interval(value - 0.05, value + 0.05, NEWCOMBE_PAIRED),
        key=CellKey(
            baseline=BASELINE,
            dressing_chain=chain,
            chain_class=chain_class,
            window_policy=POLICY,
            canon_on=None,
            family=FAMILY_ATTACK,
            benign_class=None,
            contrast=CANON_CONTRAST,
        ),
    )


# --- the choice, and the two rejections -------------------------------------------------------------


def test_the_summary_is_roc_auc_and_pr_auc_is_rejected_with_its_reason() -> None:
    assert SUMMARY_CHOICE == "roc_auc"
    assert "pr_auc" in REJECTED_SUMMARIES
    reason = REJECTED_SUMMARIES["pr_auc"]
    assert "prevalence" in reason and "CONSTRUCTED" in reason


def test_the_rejected_justification_is_kept_rather_than_quietly_replaced() -> None:
    """The monotone-invariance argument reads as rigorous and is wrong here, because the layer
    changes the text and re-scores rather than transforming a score. Keeping it is what stops the
    next reader re-deriving it."""
    assert "monotone_invariance" in REJECTED_JUSTIFICATIONS
    rejected = REJECTED_JUSTIFICATIONS["monotone_invariance"]
    assert "re-scor" in rejected
    assert "rank separation" in ACCEPTED_JUSTIFICATION
    assert "monotone" not in ACCEPTED_JUSTIFICATION, "the accepted reason is not the rejected one"


# --- saturation, measured ------------------------------------------------------------------------------


def sample(pos: tuple[float, ...], neg: tuple[float, ...]) -> AucSample:
    return AucSample(
        tuple(f"p{i}" for i in range(len(pos))),
        pos,
        tuple(f"n{i}" for i in range(len(neg))),
        neg,
    )


SEPARATED_POS = (0.80, 0.85, 0.90, 0.95)
SEPARATED_NEG = (0.10, 0.20, 0.30, 0.40)


def shifted(values: tuple[float, ...], by: float) -> tuple[float, ...]:
    """The layer's effect as the objection describes it: every score up, capped at the ceiling."""
    return tuple(min(1.0, value + by) for value in values)


def test_a_shift_into_the_ceiling_drives_auc_down_by_manufacturing_ties() -> None:
    """The measurement the saturation limit rests on, reproduced rather than restated.

    Nothing about the ordering changes across these three rows -- every attack still outscores or
    ties every benign item. The AUC falls because the ceiling turned orderings into ties.
    """
    before = roc_auc(sample(SEPARATED_POS, SEPARATED_NEG))
    assert (before.auc, before.tied_pairs) == (1.0, 0)

    middle = roc_auc(sample(shifted(SEPARATED_POS, 0.7), shifted(SEPARATED_NEG, 0.7)))
    assert (middle.auc, middle.tied_pairs, middle.total_pairs) == (0.75, 8, 16)

    total = roc_auc(sample(shifted(SEPARATED_POS, 0.9), shifted(SEPARATED_NEG, 0.9)))
    assert (total.auc, total.tied_pairs) == (0.5, 16)


def test_a_saturated_cell_carries_a_finding_naming_the_tied_share() -> None:
    (finding,) = saturation_findings([auc(0.75, tied_pairs=8)])
    assert finding.kind == FINDING_SATURATION
    assert finding.computed["tied_pairs"] == 8
    assert finding.computed["total_pairs"] == 16
    assert finding.computed["tied_share"] == 0.5
    assert "not on its own evidence that separation was lost" in finding.statement


def test_an_unsaturated_cell_carries_no_saturation_finding() -> None:
    """The same cell before the shift. Without this the finding could fire on everything and still
    look like it worked."""
    assert saturation_findings([auc(1.0, tied_pairs=0)]) == ()


def test_the_saturation_threshold_is_where_the_finding_starts_and_both_sides_are_shown() -> None:
    below = int(SATURATION_TIE_SHARE * 16)
    assert saturation_findings([auc(0.9, tied_pairs=below)]) == ()
    assert len(saturation_findings([auc(0.9, tied_pairs=below + 1)])) == 1


# --- resolution, computed ------------------------------------------------------------------------------


@pytest.mark.parametrize("n_negative,step", [(100, 0.01), (250, 0.004), (1000, 0.001)])
def test_the_resolution_finding_computes_one_over_n(n_negative: int, step: float) -> None:
    """Computed from the cell, never transcribed: the per-class sample size lives in `pins.toml`,
    and a figure written into a published finding would go stale the moment it is re-declared while
    still reading as a measurement."""
    (finding,) = resolution_findings([auc(0.9, n_negative=n_negative)])
    assert finding.kind == FINDING_RESOLUTION
    assert finding.computed["one_item_moves_the_rate_by"] == pytest.approx(step)
    assert finding.computed["n_negative"] == n_negative


def test_the_resolution_statement_carries_the_number_it_computed() -> None:
    (finding,) = resolution_findings([auc(0.9, n_negative=250)])
    assert "0.40 percentage points" in finding.statement


# --- sign disagreement --------------------------------------------------------------------------------


def test_a_sign_disagreement_reports_both_and_concludes_from_neither() -> None:
    produced = sign_disagreement_findings([auc_delta(0.12), recall_delta(-0.03)])
    (finding,) = produced
    assert finding.kind == FINDING_SIGN_DISAGREEMENT
    assert finding.computed["delta_auc"] == 0.12
    assert finding.computed["threshold_delta"] == -0.03
    assert "concludes from neither alone" in finding.statement
    assert len(finding.keys) == 2, "the finding names both cells"


def test_agreeing_signs_produce_no_finding() -> None:
    assert sign_disagreement_findings([auc_delta(0.12), recall_delta(0.03)]) == ()


def test_a_zero_on_either_side_is_not_a_disagreement() -> None:
    """Zero has no sign. Reporting it as a disagreement would put a caveat on every cell the layer
    left exactly unchanged."""
    assert sign_disagreement_findings([auc_delta(0.0), recall_delta(-0.03)]) == ()
    assert sign_disagreement_findings([auc_delta(0.12), recall_delta(0.0)]) == ()


def test_the_pairing_ignores_the_benign_class_and_that_is_load_bearing() -> None:
    """A ΔAUC is per benign class; the recall delta it is compared against has none. A pairing key
    that included the class would match nothing and the finding would silently never fire -- which
    is the failure mode a check like this is most prone to, so both classes are here."""
    b_chat_key = auc_key(benign_class="b_chat", canon_on=None)
    b_chat = Delta(
        value=0.12,
        interval=Interval(0.07, 0.17, DELTA_AUC_STRUCTURAL),
        key=b_chat_key,
    )
    produced = sign_disagreement_findings([auc_delta(0.12), b_chat, recall_delta(-0.03)])
    assert len(produced) == 2, "one recall delta pairs with both classes in its column"


def test_a_false_positive_delta_is_not_the_other_side_of_this_comparison() -> None:
    """The threshold-table side is attack recall. A false-positive delta answers a different
    question, and two different questions are allowed to disagree without it being a finding."""
    fpr = Delta(
        value=-0.03,
        interval=Interval(-0.08, 0.02, NEWCOMBE_PAIRED),
        key=CellKey(
            baseline=BASELINE,
            dressing_chain=CLEAN_CHAIN_NAME,
            chain_class=CHAIN_CLASS_BOUND,
            window_policy=POLICY,
            canon_on=None,
            family="benign",
            benign_class="b_code",
            contrast=CANON_CONTRAST,
        ),
    )
    assert sign_disagreement_findings([auc_delta(0.12), fpr]) == ()


def test_a_delta_in_another_column_is_not_paired() -> None:
    assert (
        sign_disagreement_findings(
            [auc_delta(0.12, chain="base64"), recall_delta(-0.03, chain=CLEAN_CHAIN_NAME)]
        )
        == ()
    )


# --- the bound-chain inheritance ------------------------------------------------------------------------


def test_a_bound_chain_delta_carries_the_definitional_finding() -> None:
    (finding,) = bound_chain_findings([auc_delta(0.4, chain_class=CHAIN_CLASS_BOUND)])
    assert finding.kind == FINDING_BOUND_CHAIN
    assert "for definitional reasons" in finding.statement
    assert "inherits it" in finding.statement


def test_a_held_out_chain_delta_carries_none() -> None:
    """The input that keeps the previous test from being a finding attached to everything."""
    assert bound_chain_findings([auc_delta(0.4, chain_class=CHAIN_CLASS_HELD_OUT)]) == ()


def test_the_bound_chain_finding_reads_the_field_and_not_a_chain_name() -> None:
    """`matrix.chain_class` decided the axis when the cell was keyed; a chain called
    `held_out_looking` on a bound key is still bound, and this reads the key."""
    (finding,) = bound_chain_findings(
        [auc_delta(0.4, chain="base64", chain_class=CHAIN_CLASS_BOUND)]
    )
    assert finding.computed["chain_class"] == CHAIN_CLASS_BOUND


# --- the whole set -----------------------------------------------------------------------------------


def test_findings_aborts_when_the_run_published_no_threshold_free_summary() -> None:
    """A run with a threshold table and no AUC has not answered "you only shifted the scores",
    which is the objection this whole story exists for."""
    rate = Rate(1, 2, Interval(0.0, 1.0, WILSON_SCORE), recall_delta(0.0).key)
    with pytest.raises(SummaryUnsupported) as caught:
        findings([rate])
    assert "has not answered" in str(caught.value)


def windows_matched_pair(whole: float, matched: float) -> list[Delta]:
    """A canon delta and its single-window companion, which is what the fifth finding reads."""
    unmatched = recall_delta(whole)
    companion = Delta(
        value=matched,
        interval=Interval(matched - 0.02, matched + 0.02, NEWCOMBE_PAIRED),
        key=CellKey(
            baseline=BASELINE,
            dressing_chain=CLEAN_CHAIN_NAME,
            chain_class=CHAIN_CLASS_BOUND,
            window_policy=POLICY,
            canon_on=None,
            family=FAMILY_ATTACK,
            benign_class=None,
            contrast=CANON_CONTRAST,
            population=POPULATION_SINGLE_WINDOW,
        ),
    )
    return [unmatched, companion]


def test_a_windows_matched_gap_wider_than_the_matched_half_width_is_reported() -> None:
    """The confound this companion exists to expose: a document over one window is scored as the
    maximum over its windows, so the layer can move a cell by changing how many windows a document
    needs rather than what the classifier sees in any of them."""
    (finding,) = windows_matched_findings(windows_matched_pair(0.30, 0.05))
    assert finding.kind == FINDING_WINDOWS_MATCHED
    assert finding.computed["gap"] == pytest.approx(0.25)
    assert finding.computed["matched_half_width"] == pytest.approx(0.02)
    assert len(finding.keys) == 2, "both cells are named"


def test_a_windows_matched_gap_within_the_half_width_is_not_reported() -> None:
    """The input that keeps the previous test from being a finding attached to every cell."""
    assert windows_matched_findings(windows_matched_pair(0.30, 0.29)) == ()


def test_a_companion_never_stands_in_for_the_published_cell_in_a_sign_comparison() -> None:
    """The companion shares a column with the cell it companions. It was overwriting it in the
    sign-disagreement pairing, which turned a real disagreement into no finding at all -- silently,
    on a check whose whole job is to surface one."""
    unmatched, companion = windows_matched_pair(-0.03, 0.30)
    produced = sign_disagreement_findings([auc_delta(0.12), unmatched, companion])
    assert len(produced) == 1
    assert produced[0].computed["threshold_delta"] == pytest.approx(-0.03)


def test_a_companion_with_no_unmatched_twin_produces_no_finding() -> None:
    _, companion = windows_matched_pair(0.30, 0.05)
    assert windows_matched_findings([companion]) == ()


def test_an_auc_delta_has_no_window_companion_and_is_not_paired_with_one() -> None:
    """The two are told apart by the interval's method, which is a field. Pairing a ΔAUC with a
    proportion companion would compare two different quantities and always report a gap."""
    _, companion = windows_matched_pair(0.30, 0.05)
    assert windows_matched_findings([auc_delta(0.12), companion]) == ()


def test_findings_returns_every_kind_in_a_stable_order() -> None:
    cells = [
        auc(0.75, tied_pairs=8),
        auc_delta(0.12),
        *windows_matched_pair(-0.03, 0.30),
        Count(1, 2, recall_delta(0.0).key),
    ]
    produced = findings(cells)
    kinds = [finding.kind for finding in produced]
    assert set(kinds) == set(FINDING_KINDS)
    assert kinds == sorted(kinds, key=FINDING_KINDS.index)
    assert [f.as_json_object() for f in findings(list(reversed(cells)))] == [
        f.as_json_object() for f in produced
    ]


def test_a_finding_names_the_cells_it_applies_to() -> None:
    with pytest.raises(ValueError) as caught:
        SummaryFinding(FINDING_SATURATION, (), "something", {})
    assert "names none" in str(caught.value)


def test_a_finding_kind_outside_the_closed_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError):
        SummaryFinding("a_caveat_nobody_declared", (auc_key(),), "x", {})


def test_a_finding_with_an_empty_statement_is_refused() -> None:
    with pytest.raises(ValueError):
        SummaryFinding(FINDING_SATURATION, (auc_key(),), "   ", {})


def test_the_new_abort_declares_exit_code_31_and_declares_it_once() -> None:
    assert declared_exit_codes()[31] is SummaryUnsupported
    assert SummaryUnsupported.exit_code == 31


# --- the rejection is enforced, not only recorded ---------------------------------------------------


def test_no_module_computes_a_rejected_summary() -> None:
    """PR AUC is rejected in `REJECTED_SUMMARIES`. A rejection nobody enforces is a rejection that
    survives exactly until somebody finds the gap useful, which is why story 4.1's rejected variance
    is scanned for by name too.

    The scan is on the tree rather than on the text: the words "pr_auc" and "precision" appear in
    this repository's prose on purpose, and a grep would report the very docstring that records the
    rejection.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    forbidden = {"pr_auc", "precision_recall_auc", "average_precision"}
    offenders: list[str] = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                offenders.append(f"{path.name}:{node.lineno} defines {node.name}")
            if isinstance(node, ast.Name) and node.id in forbidden:
                offenders.append(f"{path.name}:{node.lineno} names {node.id}")
    assert offenders == [], offenders
    assert set(REJECTED_SUMMARIES) <= forbidden, "every rejected summary is one the scan covers"


def test_the_rejected_summary_scan_fires_on_a_definition() -> None:
    """The scan's own red input, so it cannot pass by failing to look."""
    import ast

    tree = ast.parse("def pr_auc(pos, neg):\n    return 0.0\n")
    found = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"pr_auc"}
    ]
    assert found == ["pr_auc"]


# --- the two modules actually compose ------------------------------------------------------------------


def test_findings_run_over_cells_that_aggregate_actually_produced() -> None:
    """The end-to-end seam, and the only test here whose cells come from somewhere else.

    Every other test in this file builds a cell by hand, which checks the findings and not the
    joint. This one runs a score file through `aggregate.cells` and hands the result to `findings`,
    so a key shape the aggregator emits and the summary cannot read is caught here rather than in
    story 4-7.
    """
    from nbc.harness.aggregate import cells as aggregate_cells
    from tests.harness.test_aggregate import PINS, a_realistic_file

    produced = aggregate_cells(a_realistic_file(), PINS)
    assert any(isinstance(cell, Auc) for cell in produced), "the file supports an AUC"

    result = findings(produced)
    assert result, "a real cell set carries at least the resolution limit"
    assert {finding.kind for finding in result} <= set(FINDING_KINDS)
    for finding in result:
        assert finding.statement.strip()
        assert finding.keys
