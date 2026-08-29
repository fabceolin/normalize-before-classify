"""Every golden case runs, and every stage is required to have one — including a no-op.

Two halves, and the second is the one that stops the table from rotting. Running the cases proves
the layer still does what the literals say. Requiring every step of `PIPELINE` to appear as a key,
with at least one no-op and at least one changing case, proves the table still covers the layer: a
fifth stage added in a later epic fails here on the day it is added, rather than shipping untested
behind a green suite.
"""

from __future__ import annotations

import pytest

from golden_cases import GOLDEN, LAYER, Golden
from nbc.canon.pipeline import PIPELINE, canonicalize, default_context
from nbc.schema import CanonContext, Edit

CASES = [(key, case) for key, cases in GOLDEN.items() for case in cases]
"""Flattened once, at import, so a case that is never reached would be a missing id in the report."""


@pytest.fixture(scope="module")
def ctx() -> CanonContext:
    return default_context()


def run_case(key: str, case: Golden, ctx: CanonContext):
    """Run one case through the stage it belongs to, or through the whole layer."""
    if key == LAYER:
        return canonicalize(case.before, ctx)
    (step,) = [candidate for candidate in PIPELINE if candidate.name == key]
    return step.run(case.before, ctx)


@pytest.mark.parametrize(
    ("key", "case"), CASES, ids=[f"{key}: {case.note}" for key, case in CASES]
)
def test_the_golden_case_reproduces_its_recorded_output(
    key: str, case: Golden, ctx: CanonContext
) -> None:
    result = run_case(key, case, ctx)
    assert result.text == case.after
    assert result.edits == case.edits


@pytest.mark.parametrize(
    ("key", "case"),
    [(key, case) for key, case in CASES if key == LAYER],
    ids=[case.note for key, case in CASES if key == LAYER],
)
def test_the_whole_layer_reports_the_recorded_depth_and_ceiling(
    key: str, case: Golden, ctx: CanonContext
) -> None:
    result = canonicalize(case.before, ctx)
    assert result.ceiling_hit is case.ceiling_hit
    assert result.max_depth_reached == case.max_depth_reached


# --- the table is required to cover the layer -----------------------------------------------------


def test_every_pipeline_step_has_golden_cases() -> None:
    """The keys are exactly the four steps plus the whole-layer sentinel, and nothing else.

    Equality, not containment: a key that is not a step is a table nobody runs against anything,
    which is how a stage renamed in one place survives with its old cases still green.
    """
    assert set(GOLDEN) == {step.name for step in PIPELINE} | {LAYER}


@pytest.mark.parametrize("key", sorted(GOLDEN), ids=str)
def test_every_stage_ships_a_no_op_case(key: str) -> None:
    """A stage never shown leaving text alone has never been shown doing half of its job."""
    no_ops = [case for case in GOLDEN[key] if case.is_no_op]
    assert no_ops, f"{key} has no case where nothing changes"


@pytest.mark.parametrize("key", sorted(GOLDEN), ids=str)
def test_every_stage_ships_a_changing_case(key: str) -> None:
    """The mirror image: a table of nothing but no-ops passes without exercising the stage."""
    changing = [case for case in GOLDEN[key] if not case.is_no_op]
    assert changing, f"{key} has no case where something changes"


def test_a_no_op_case_is_no_op_in_both_halves() -> None:
    """`is_no_op` is what the two requirements above are written in terms of, so it is checked.

    Same text and an empty trace. A case that returned its input while recording an edit would be
    a stage misattributing a change, which is a different failure with a different name.
    """
    assert Golden(note="", before="x", after="x").is_no_op
    assert not Golden(note="", before="x", after="y").is_no_op
    assert not Golden(
        note="",
        before="x",
        after="x",
        edits=(Edit(stage="probe", span=(0, 1), before="x", after="x"),),
    ).is_no_op


def test_the_ceiling_case_is_the_only_one_that_reports_a_hit() -> None:
    """The battery holds a ceiling hit and holds documents that must not report one.

    A table where every case reported `False` would make the `ceiling_hit` assertion above true
    and empty, which is the shape of check this project has had to remove before.
    """
    hits = [case.note for case in GOLDEN[LAYER] if case.ceiling_hit]
    assert len(hits) == 1, hits
    assert max(case.max_depth_reached for case in GOLDEN[LAYER]) == 3
