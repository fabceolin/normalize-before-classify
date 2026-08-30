"""Two structural bounds on the path from a score to a cell, read from the syntax tree.

*The threshold is applied in one place.* Story 4.2 committed `p_injection` unclassified on purpose:
a threshold applied at write time cannot be changed without re-running eighty-five hours of
inference, and one applied in two places will eventually differ between them on the borderline
items it decides. `harness/aggregate.classify` is that place, and this scan is what keeps it the
only one.

*Nothing else opens the scores file.* Every threshold-dependent rate, every AUC and every
discordant pair in the published table is computed from the committed file by one module, which is
what makes "the committed scores are the scores that produced the numbers" a property of the code
and not a promise. `harness/run.py` writes the file and `harness/aggregate.py` reads it; a third
module on that path is what this scan refuses.

Both scans are checked against synthetic sources that violate them, because a scan nobody has seen
report anything is not a scan. The shape is `tests/canon/test_import_bound.py`'s and
`tests/corpus/test_build.py`'s, and the reason it is not imported from either is that it answers a
different question over a different set of modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

THRESHOLD_READERS: dict[str, str] = {
    "pins.py": "declares Baseline.threshold and is where the value comes from",
    "aggregate.py": "holds `classify`, the one place a score becomes a class",
}

SCORE_RECORD_MODULES: dict[str, str] = {
    "schema.py": "declares ItemScore",
    "score.py": "the pure half of the scoring pass: the key, the merge, the serializer",
    "run.py": "the IO half: it writes results/scores-<n>-<i>.jsonl and results/scores.jsonl",
    "aggregate.py": "the one reader of the merged file, and the one producer of cells",
}
"""Every module that may touch an `ItemScore` at all.

The bound is on the record rather than on the filename, because `aggregate.py` takes the path as an
argument and never spells it -- so a scan for the name would pass while saying nothing. The limit,
declared: a module handed an already-read tuple of `ItemScore`s by one of these four is invisible
here. What stops that is that nothing else opens the file, which the filename scan below covers.
"""

SCORES_FILE_MODULES: dict[str, str] = {
    "run.py": "names RESULTS_DIRNAME and SCORES_FILENAME, and writes both",
}
"""Who may spell the scores file's name. `aggregate.py` takes a path and is deliberately absent."""


SUBTRACTION_EXEMPT: dict[str, str] = {
    "verdict.py": (
        "reports the difference between two published deltas inside a Verdict, whose interval "
        "comes from stats.mover_difference_interval -- and a Verdict has no interval field, so a "
        "difference cannot inherit one there"
    ),
    "summary.py": (
        "reports the gap between two published cells inside a SummaryFinding, which has no "
        "interval field -- so a difference cannot inherit one there"
    ),
}
"""Modules where subtracting two estimates is not the thing the rule forbids.

Checked rather than trusted: `test_the_exemption_holds_because_a_finding_carries_no_interval`
asserts the property the reason claims.
"""


def source_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def attribute_reads(source: str, attribute: str) -> list[int]:
    """Lines at which `source` reads `.<attribute>` off something, from the tree.

    Structural: a grep for `threshold` fires on `measured_at_threshold`, on the word in a docstring,
    and on a local variable named `threshold` -- three false positives in this repository alone.
    An `ast.Attribute` whose `attr` is the name is the thing that means "somebody reached for the
    pin".
    """
    return sorted(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr == attribute
    )


def mentions(source: str, name: str) -> bool:
    """Whether `source` names `name` as code -- a bare name, an attribute tail, or an import."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
            return True
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return True
    return False


def test_only_the_declared_modules_read_a_threshold() -> None:
    reading = {
        path.name
        for path in source_files()
        if attribute_reads(path.read_text(encoding="utf-8"), "threshold")
    }
    assert reading == set(THRESHOLD_READERS), sorted(
        reading.symmetric_difference(THRESHOLD_READERS)
    )


def test_the_threshold_scan_fires_on_a_second_reader() -> None:
    """The input that turns the rule above red, and the three shapes it must not fire on."""
    assert attribute_reads("k = pins.baselines[0].threshold\n", "threshold") == [1]
    assert attribute_reads("threshold = 0.5\nif p >= threshold: pass\n", "threshold") == []
    assert attribute_reads("x = row.measured_at_threshold\n", "threshold") == []
    assert attribute_reads('"""the threshold lives in pins."""\n', "threshold") == []


def test_only_the_declared_modules_touch_a_score_record() -> None:
    touching = {
        path.name
        for path in source_files()
        if mentions(path.read_text(encoding="utf-8"), "ItemScore")
    }
    assert touching == set(SCORE_RECORD_MODULES), sorted(
        touching.symmetric_difference(SCORE_RECORD_MODULES)
    )


def test_only_the_declared_modules_spell_the_scores_file_name() -> None:
    spelling = {
        path.name
        for path in source_files()
        if mentions(path.read_text(encoding="utf-8"), "SCORES_FILENAME")
    }
    assert spelling == set(SCORES_FILE_MODULES), sorted(
        spelling.symmetric_difference(SCORES_FILE_MODULES)
    )


def test_the_mention_scan_sees_each_shape_and_not_a_docstring() -> None:
    assert mentions("from nbc.schema import ItemScore\n", "ItemScore")
    assert mentions("x = schema.ItemScore(**payload)\n", "ItemScore")
    assert mentions("ItemScore(item_id='a')\n", "ItemScore")
    assert not mentions('"""Reads an ItemScore per line."""\nx = 1\n', "ItemScore")


def test_aggregate_writes_nothing() -> None:
    """`tests/corpus/test_build.py` owns the tree-wide writer scan and would fail if this module
    acquired a write; asserted here too, at the module this story added, so the reason is next to
    the code. `results/results.json` is 4-7's and this story does not open a file for writing."""
    tree = ast.parse((SRC / "nbc" / "harness" / "aggregate.py").read_text(encoding="utf-8"))
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_text", "write_bytes", "writelines"}
    ]
    assert writes == []


def test_no_module_builds_a_delta_out_of_two_rates() -> None:
    """"A difference is never obtained by subtracting two `Rate`s" as a check rather than a rule.

    The shape it looks for is a subtraction whose two sides are `.value`, `.lo` or `.hi` reads off
    **two different receivers** -- which is what subtracting two rates looks like once somebody has
    reached for the accessors.

    The receiver comparison is not a refinement for tidiness. `Interval.width` is `self.hi -
    self.lo`, one interval's own two bounds, and it is the shape this scan found on its first run.
    Flagging it would have meant either deleting a correct property or weakening the rule, and both
    sides being the same expression is exactly what distinguishes one interval's width from a
    difference between two estimates.

    The limit, declared: assigning each side to a local first hides it from this scan. What stops
    that is `Delta` refusing a Wilson interval, which is the only interval two `Rate`s could offer
    it.

    `SUBTRACTION_EXEMPT` is one entry and it is not a convenience. `summary.py` subtracts two
    published cells to report the **gap** between them, and the rule this scan enforces is about a
    difference that INHERITS an interval it has no right to. A `SummaryFinding` has no interval
    field at all, so the thing the rule prevents is unrepresentable there -- and the test below
    asserts that rather than taking the exemption's word for it.
    """
    accessors = {"value", "lo", "hi"}
    offenders: list[str] = []
    for path in source_files():
        if path.name in SUBTRACTION_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)):
                continue
            left, right = node.left, node.right
            if not all(
                isinstance(side, ast.Attribute) and side.attr in accessors
                for side in (left, right)
            ):
                continue
            assert isinstance(left, ast.Attribute) and isinstance(right, ast.Attribute)
            if ast.dump(left.value) == ast.dump(right.value):
                continue  # one object's own two bounds, not two estimates
            offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], offenders


def flagged(source: str) -> int:
    """The scan above, over one source, so its two inputs can be shown side by side."""
    total = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)):
            continue
        left, right = node.left, node.right
        if not all(
            isinstance(side, ast.Attribute) and side.attr in {"value", "lo", "hi"}
            for side in (left, right)
        ):
            continue
        assert isinstance(left, ast.Attribute) and isinstance(right, ast.Attribute)
        if ast.dump(left.value) != ast.dump(right.value):
            total += 1
    return total


def test_the_subtraction_scan_fires_on_two_estimates_and_not_on_one_width() -> None:
    """Both inputs, because the scan is only a scan if the pair separates them."""
    assert flagged("d = canon_on.value - canon_off.value\n") == 1
    assert flagged("w = self.hi - self.lo\n") == 0


def test_the_declared_tie_convention_is_the_one_classify_implements() -> None:
    """`THRESHOLD_COMPARISON` is a constant recorded beside a value, which is this repository's
    first defect pattern unless something compares the two. This is that comparison.

    The operator is read out of `classify`'s syntax tree rather than inferred from behaviour,
    because a behavioural check at the boundary (`classify(t, t) is True`) passes for `>=` and for
    a `>` with an epsilon, and those publish different recall on any item that lands on the
    threshold.
    """
    from nbc.harness.aggregate import THRESHOLD_COMPARISON, classify

    tree = ast.parse((SRC / "nbc" / "harness" / "aggregate.py").read_text(encoding="utf-8"))
    (function,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify"
    ]
    comparisons = [node for node in ast.walk(function) if isinstance(node, ast.Compare)]
    assert len(comparisons) == 1, "classify compares once"
    (operator,) = comparisons[0].ops
    spelling = {ast.GtE: ">=", ast.Gt: ">", ast.LtE: "<=", ast.Lt: "<"}[type(operator)]
    assert spelling == THRESHOLD_COMPARISON
    assert classify(0.5, 0.5) is True


def test_the_exemption_holds_because_a_finding_carries_no_interval() -> None:
    """The exemption above says a `SummaryFinding` cannot inherit an interval. This is that claim.

    Without it the entry would be an assertion in a docstring, which is the shape this repository
    keeps finding in its own history: a reason recorded beside a value and never compared to it.
    """
    from nbc.harness.summary import SummaryFinding

    from nbc.schema import Verdict

    assert set(SUBTRACTION_EXEMPT) == {"summary.py", "verdict.py"}
    for record in (SummaryFinding, Verdict):
        assert "interval" not in record.__slots__, record.__name__
    assert "interval" not in SummaryFinding(
        kind="saturation",
        keys=(_a_key(),),
        statement="x",
        computed={},
    ).as_json_object()
    assert "interval" not in Verdict(
        condition="N1",
        outcome="not_triggered",
        keys=(_a_key(),),
        reason="x",
        computed={"minimum_detectable_effect": 0.0},
    ).as_json_object()


def _a_key():  # type: ignore[no-untyped-def]
    from nbc.schema import CellKey

    return CellKey(
        baseline="b",
        dressing_chain="clean",
        chain_class="bound",
        window_policy="shared",
        canon_on=True,
        family="attack",
    )
