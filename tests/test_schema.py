"""`schema.py` is a leaf, and `Score` carries the spine's fields verbatim."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nbc import schema
from nbc.schema import (
    CANONICAL,
    CONDITIONS,
    RAW,
    CanonContext,
    CanonResult,
    Edit,
    ItemScore,
    Score,
    StageResult,
)


def _schema_source() -> tuple[Path, ast.Module]:
    path = Path(schema.__file__)
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_schema_imports_nothing_from_nbc() -> None:
    """The one rule that lets `canon/` depend on this module without losing its isolation.

    Checked against the parsed source, so an import hidden inside a function body or a
    `TYPE_CHECKING` block is caught just the same.
    """
    path, tree = _schema_source()
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nbc" or alias.name.startswith("nbc."):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # any relative import is an import from inside nbc
                offenders.append(
                    f"{path.name}:{node.lineno} from {'.' * node.level}{node.module or ''} import ..."
                )
            elif node.module == "nbc" or (node.module or "").startswith("nbc."):
                offenders.append(f"{path.name}:{node.lineno} from {node.module} import ...")

    assert not offenders, "schema.py must import nothing from nbc: " + "; ".join(offenders)


def test_importing_schema_pulls_in_no_other_nbc_module() -> None:
    """The same rule observed at runtime, in a fresh interpreter.

    `nbc` itself is loaded because `nbc.schema` is inside it; nothing else may be.
    """
    code = (
        "import sys, nbc.schema; "
        "print(sorted(m for m in sys.modules if m == 'nbc' or m.startswith('nbc.')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "['nbc', 'nbc.schema']", completed.stdout


def test_score_carries_exactly_the_declared_fields() -> None:
    assert [f.name for f in dataclasses.fields(Score)] == ["p_injection", "n_windows"]


def test_score_is_frozen() -> None:
    score = Score(p_injection=0.5, n_windows=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.p_injection = 0.9  # type: ignore[misc]


def test_score_accepts_the_full_probability_range() -> None:
    assert Score(p_injection=0.0, n_windows=1).p_injection == 0.0
    assert Score(p_injection=1.0, n_windows=7).p_injection == 1.0


def test_score_stores_integral_probabilities_as_floats() -> None:
    # So the serialized form does not depend on how the caller spelled 0 or 1.
    stored = Score(p_injection=1, n_windows=1).p_injection  # type: ignore[arg-type]
    assert isinstance(stored, float)


@pytest.mark.parametrize("bad", [1.5, -0.001, float("nan"), float("inf"), "0.5", None, True])
def test_score_rejects_a_p_injection_that_is_not_a_probability(bad: object) -> None:
    with pytest.raises(ValueError):
        Score(p_injection=bad, n_windows=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 1.0, "1", None, True])
def test_score_rejects_a_window_count_below_one(bad: object) -> None:
    # A scored document occupies at least one window; zero would silently mean "not scored".
    with pytest.raises(ValueError):
        Score(p_injection=0.5, n_windows=bad)  # type: ignore[arg-type]


def test_schema_declares_only_the_types_this_epic_uses() -> None:
    """Later stories add their own types here; most of them have not run yet.

    This is not a freeze on `schema.py`. It is a reminder that a type belongs to the story
    that first needs it, so the next story updates this list in the same commit that adds
    its type — and a type that appears with no consumer is caught.
    """
    assert schema.__all__ == [
        "ATTACK",
        "AUC_STRUCTURAL",
        "AXIS_BASELINE",
        "AXIS_BENIGN_CLASS",
        "AXIS_CANON_ON",
        "AXIS_CHAIN_CLASS",
        "AXIS_DRESSING_CHAIN",
        "AXIS_FAMILY",
        "AXIS_WINDOW_POLICY",
        "Auc",
        "BENIGN",
        "BENIGN_CLASSES",
        "CANONICAL",
        "CELL_AXES",
        "CENSUS_CEILING_HIT",
        "CENSUS_KINDS",
        "CENSUS_WINDOW_OVERFLOW",
        "CHAIN_CLASSES_FOR_KEYS",
        "CONDITIONS",
        "CONTRAST_ARGUMENT_REQUIRED",
        "CONTRAST_ATTACKS_VS_BENIGN_CLASS",
        "CONTRAST_BOUND_VS_HELD_OUT",
        "CONTRAST_CANON_ON_VS_OFF",
        "CONTRAST_CLEAN_VS_CHAIN",
        "CONTRAST_KINDS",
        "CanonContext",
        "CanonResult",
        "CellKey",
        "Contrast",
        "CorpusItem",
        "Count",
        "DELTA_AUC_STRUCTURAL",
        "Delta",
        "EDIT_CENSUS_PREFIX",
        "Edit",
        "FALSIFICATION_CONDITIONS",
        "FAMILIES",
        "FAMILY_ATTACK",
        "FAMILY_BENIGN",
        "INTERVAL_METHODS",
        "Interval",
        "ItemScore",
        "LABELS",
        "MOVER_DIFFERENCE",
        "NEWCOMBE_PAIRED",
        "OUTCOME_NOT_EVALUABLE",
        "OUTCOME_NOT_TRIGGERED",
        "OUTCOME_TRIGGERED",
        "PERMITTED_SPANS",
        "PIPELINE_STAGES",
        "POPULATIONS",
        "POPULATION_ALL",
        "POPULATION_SINGLE_WINDOW",
        "PROTECTED_AXES",
        "PairedCount",
        "RAW",
        "REQUIRED_SPANS",
        "Rate",
        "Score",
        "StageResult",
        "VERDICT_OUTCOMES",
        "Verdict",
        "WILSON_SCORE",
        "edit_census_of",
    ]


def test_every_record_type_defined_here_is_exported() -> None:
    """A type present in the module but missing from `__all__` is a type with no contract."""
    defined = {
        name
        for name, obj in vars(schema).items()
        if not name.startswith("_")
        and dataclasses.is_dataclass(obj)
        and getattr(obj, "__module__", None) == schema.__name__
    }
    # Containment, not equality: `__all__` also carries the corpus vocabulary constants, which
    # are not dataclasses. The claim under test is one-directional -- a record type defined here
    # and not exported is a type with no contract -- so the failing input is a new dataclass whose
    # name nobody added to `__all__`.
    assert defined, "the scan found no record types, which would pass vacuously"
    assert defined <= set(schema.__all__)
    missing = [name for name in schema.__all__ if not hasattr(schema, name)]
    assert not missing, f"{missing} are exported and not defined"


# --- the scoring pass's record ------------------------------------------------------------------


def an_item_score(**overrides) -> ItemScore:
    fields = {
        "item_id": "0001aaaa::clean",
        "family": schema.FAMILY_ATTACK,
        "benign_class": None,
        "label": schema.ATTACK,
        "baseline_key": "protectai-deberta-v3",
        "condition": RAW,
        "p_injection": 0.5,
        "n_windows": 1,
    }
    return ItemScore(**{**fields, **overrides})


def test_an_item_score_carries_its_own_coordinates() -> None:
    """A line of the scores file means something without the corpus open beside it."""
    assert [field.name for field in dataclasses.fields(ItemScore)] == [
        "item_id",
        "family",
        "benign_class",
        "label",
        "baseline_key",
        "condition",
        "p_injection",
        "n_windows",
        "max_depth_reached",
        "ceiling_hit",
    ]


def test_the_two_conditions_are_a_closed_vocabulary() -> None:
    """A third condition would make "the difference between the two" ill-defined."""
    assert CONDITIONS == (RAW, CANONICAL)

    with pytest.raises(ValueError, match="condition must be one of"):
        an_item_score(condition="canonicalised")


def test_a_canonical_score_reports_the_layers_outcome() -> None:
    record = an_item_score(condition=CANONICAL, max_depth_reached=2, ceiling_hit=True)

    assert record.max_depth_reached == 2
    assert record.ceiling_hit is True


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"max_depth_reached": None, "ceiling_hit": False}, "reports the depth"),
        ({"max_depth_reached": 0, "ceiling_hit": None}, "reports whether the ceiling"),
        ({"max_depth_reached": -1, "ceiling_hit": False}, "must not be negative"),
        ({"max_depth_reached": True, "ceiling_hit": False}, "reports the depth"),
    ],
)
def test_a_canonical_score_that_does_not_report_it_is_refused(kwargs, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        an_item_score(condition=CANONICAL, **kwargs)


@pytest.mark.parametrize("kwargs", [{"max_depth_reached": 0}, {"ceiling_hit": False}])
def test_a_raw_score_that_reports_a_depth_or_a_ceiling_hit_is_refused(kwargs) -> None:
    """A `0` on a raw record is indistinguishable from a canonical document that needed no decode.

    That is the whole reason the fields are `None` here rather than zero: a reader tallying FR10's
    ceiling hits over the scores file would otherwise be tallying over twice the population, and
    every individual record would look valid.
    """
    with pytest.raises(ValueError, match="reports no"):
        an_item_score(condition=RAW, **kwargs)


def test_an_item_score_checks_its_family_against_its_label() -> None:
    """The copy out of the corpus is a second chance to attach the wrong label to a text."""
    with pytest.raises(ValueError, match="carries label"):
        an_item_score(family=schema.FAMILY_BENIGN, benign_class="b_chat", label=schema.ATTACK)


def test_an_attack_score_carries_no_benign_class_and_a_benign_one_must() -> None:
    with pytest.raises(ValueError, match="an attack item has no benign class"):
        an_item_score(benign_class="b_chat")

    with pytest.raises(ValueError, match="must name one of"):
        an_item_score(family=schema.FAMILY_BENIGN, label=schema.BENIGN, benign_class=None)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_an_item_score_refuses_a_p_injection_that_is_not_a_probability(bad: object) -> None:
    """Delegated to `Score` rather than restated, and checked here so the delegation is real."""
    with pytest.raises(ValueError, match="p_injection"):
        an_item_score(p_injection=bad)


def test_an_item_score_refuses_a_window_count_below_one() -> None:
    with pytest.raises(ValueError, match="n_windows"):
        an_item_score(n_windows=0)


def test_an_item_score_stores_an_integral_probability_as_a_float() -> None:
    """So the serialized form does not depend on how the caller happened to spell 1."""
    assert json.loads(json.dumps(an_item_score(p_injection=1).as_json_object()))[
        "p_injection"
    ] == 1.0


def test_an_item_score_that_names_nothing_is_refused() -> None:
    for blank in ("item_id", "baseline_key"):
        with pytest.raises(ValueError, match=f"{blank} must be a non-empty string"):
            an_item_score(**{blank: ""})


def test_the_serialized_form_carries_every_key_including_the_absent_ones() -> None:
    """Absent-versus-null is the difference between "no depth" and "somebody forgot to write one"."""
    payload = an_item_score().as_json_object()

    assert list(payload) == [field.name for field in dataclasses.fields(ItemScore)]
    assert payload["max_depth_reached"] is None
    assert payload["ceiling_hit"] is None
    assert payload["benign_class"] is None


def test_an_item_score_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        an_item_score().p_injection = 0.9  # type: ignore[misc]


# --- the canonicalization layer's shapes ------------------------------------------------------


def test_the_ascii_bound_is_the_same_number_the_confusables_contract_uses() -> None:
    """The one duplicated constant in the leaf, compared rather than trusted.

    `schema.py` cannot import `nbc.canon.confusables_table` — it is a leaf, and that is the whole
    point of it. So the bound is spelled twice, and this is the check that makes the second copy a
    comparison instead of a second opinion.
    """
    from nbc.canon import confusables_table

    assert schema.MAX_ASCII == confusables_table.ASCII_LAST


def test_an_edit_records_a_span_that_measures_its_own_before_text() -> None:
    edit = Edit(stage="invisible", span=(3, 5), before="ab", after="")
    assert edit.span == (3, 5)
    assert edit.depth == 0


def test_an_edit_accepts_a_span_that_arrived_as_a_list() -> None:
    # A trace read back from JSONL has lists where the record has a tuple.
    assert Edit(stage="s", span=[1, 2], before="a", after="b").span == (1, 2)  # type: ignore[arg-type]


def test_a_no_op_edit_is_legal() -> None:
    # Stories 2.3 and 2.4 record a refused candidate as an edit whose before equals its after.
    edit = Edit(stage="decode", span=(0, 4), before="dGVz", after="dGVz")
    assert edit.before == edit.after


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "", "span": (0, 1), "before": "a", "after": ""},
        {"stage": 1, "span": (0, 1), "before": "a", "after": ""},
        {"stage": "s", "span": (0, 1, 2), "before": "a", "after": ""},
        {"stage": "s", "span": (0,), "before": "", "after": ""},
        {"stage": "s", "span": "01", "before": "", "after": ""},
        {"stage": "s", "span": (-1, 1), "before": "aa", "after": ""},
        {"stage": "s", "span": (2, 1), "before": "", "after": ""},
        {"stage": "s", "span": (0, 1.0), "before": "a", "after": ""},
        {"stage": "s", "span": (0, True), "before": "a", "after": ""},
        {"stage": "s", "span": (0, 2), "before": "a", "after": ""},
        {"stage": "s", "span": (0, 1), "before": b"a", "after": ""},
        {"stage": "s", "span": (0, 1), "before": "a", "after": None},
        {"stage": "s", "span": (0, 1), "before": "a", "after": "", "depth": -1},
        {"stage": "s", "span": (0, 1), "before": "a", "after": "", "depth": 1.5},
        {"stage": "s", "span": (0, 1), "before": "a", "after": "", "depth": True},
    ],
)
def test_an_edit_that_does_not_hold_together_is_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Edit(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("record", [StageResult, CanonResult])
def test_a_result_freezes_its_edits_into_a_tuple(record) -> None:
    edit = Edit(stage="s", span=(0, 1), before="a", after="")
    assert record(text="x", edits=[edit]).edits == (edit,)
    assert record(text="x").edits == ()


@pytest.mark.parametrize("record", [StageResult, CanonResult])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": 1, "edits": ()},
        {"text": "x", "edits": "not a sequence of edits"},
        {"text": "x", "edits": ["not an edit"]},
        {"text": "x", "edits": 5},
    ],
)
def test_a_result_that_does_not_hold_together_is_refused(record, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        record(**kwargs)


# --- the recursion contract's two reported values ---------------------------------------------


def test_a_canon_result_reports_no_recursion_by_default() -> None:
    result = CanonResult(text="x")
    assert result.ceiling_hit is False
    assert result.max_depth_reached == 0


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"ceiling_hit": "yes"}, "must be a bool"),
        ({"ceiling_hit": 1}, "must be a bool"),
        ({"max_depth_reached": 1.0}, "must be an int"),
        ({"max_depth_reached": True}, "must be an int"),
        ({"max_depth_reached": -1}, "must not be negative"),
    ],
)
def test_a_recursion_report_that_does_not_hold_together_is_refused(
    kwargs: dict, needle: str
) -> None:
    with pytest.raises(ValueError, match=needle):
        CanonResult(text="x", **kwargs)  # type: ignore[arg-type]


def test_a_reported_depth_shallower_than_its_own_trace_is_refused() -> None:
    """The gate, with the input that makes it fail.

    An edit stamped at depth 2 is evidence that a document was canonicalized at depth 2. A result
    claiming it never went past depth 1 is contradicting the trace it is carrying, which is the
    shape this project keeps finding: a value recorded beside its evidence and never compared.
    """
    deep = Edit(stage="decode", span=(0, 1), before="a", after="b", depth=2)
    with pytest.raises(ValueError, match="the trace holds an edit at depth 2"):
        CanonResult(text="b", edits=(deep,), max_depth_reached=1)

    # The negative control: the same trace with a depth that does account for it.
    assert CanonResult(text="b", edits=(deep,), max_depth_reached=2).max_depth_reached == 2
    assert CanonResult(text="b", edits=(deep,), max_depth_reached=5).max_depth_reached == 5


def test_a_deeper_reported_depth_than_the_trace_shows_is_allowed() -> None:
    """One-directional on purpose, and the two inputs that show why.

    With tracing off there is no trace to bound anything; and a sub-document that needed no change
    produces no edit at all although it was canonicalized. Requiring equality would refuse both.
    """
    assert CanonResult(text="x", edits=(), max_depth_reached=3).max_depth_reached == 3


def test_a_context_freezes_its_table_and_defaults_to_tracing() -> None:
    ctx = CanonContext(confusables={0x0430: "a"}, ceiling=3)
    assert ctx.trace_enabled is True
    assert ctx.ceiling == 3
    assert ctx.confusables[0x0430] == "a"
    with pytest.raises(TypeError):
        ctx.confusables[0x0431] = "b"  # type: ignore[index]


def test_a_context_has_no_default_ceiling_of_its_own() -> None:
    """FR10 asks for a declared default and never an implicit one.

    This module is a leaf and cannot import `nbc.canon.pipeline.DEFAULT_CEILING`, which is exactly
    what makes "one home for the default" enforceable rather than merely intended: a context built
    without a ceiling is a `TypeError` here, so no second place can quietly supply one.
    """
    with pytest.raises(TypeError):
        CanonContext(confusables={})  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"confusables": [(0x0430, "a")]}, "must be a mapping"),
        ({"confusables": {"а": "a"}}, "code points"),
        ({"confusables": {True: "a"}}, "code points"),
        ({"confusables": {0x110000: "a"}}, "not a code point"),
        ({"confusables": {0x61: "b"}}, "ASCII code point"),
        ({"confusables": {0x00: "x"}}, "ASCII code point"),
        ({"confusables": {0x0430: ""}}, "non-empty str"),
        ({"confusables": {0x0430: 97}}, "non-empty str"),
        ({"confusables": {}, "trace_enabled": "yes"}, "must be a bool"),
        ({"confusables": {}, "ceiling": 1.5}, "must be an int"),
        ({"confusables": {}, "ceiling": True}, "must be an int"),
        ({"confusables": {}, "ceiling": "3"}, "must be an int"),
        ({"confusables": {}, "ceiling": -1}, "must not be negative"),
    ],
)
def test_a_context_that_does_not_hold_together_is_refused(kwargs: dict, needle: str) -> None:
    kwargs = {"ceiling": 3, **kwargs}
    with pytest.raises(ValueError, match=needle):
        CanonContext(**kwargs)  # type: ignore[arg-type]


def test_a_ceiling_of_zero_is_a_legitimate_setting() -> None:
    # Zero is not "unset": it means decode nothing and report every candidate as a ceiling hit.
    assert CanonContext(confusables={}, ceiling=0).ceiling == 0
