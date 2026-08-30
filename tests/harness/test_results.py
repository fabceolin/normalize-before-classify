"""The results file's shape, and the assertion that a half-built table cannot ship.

Cells are built by hand because the sharp cases are absences: a benign class with no rate, a key
with no AUC, a table with no held-out chain. A realistic corpus produces a complete cell set, which
is the one input that cannot show any of them.

The completeness assertion is pure for this reason -- it guards a command that opens models, and
these tests run in milliseconds with none.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Final

import pytest

from nbc.canon.pipeline import PIPELINE
from nbc.corpus.matrix import CHAIN_CLASS_BOUND, CHAIN_CLASS_HELD_OUT, CLEAN_CHAIN_NAME
from nbc.errors import declared_exit_codes
from nbc.harness.results import (
    PROFILE_FULL,
    PROFILE_SMOKE,
    PROFILES,
    RESULT_KEYS,
    SCHEMA_VERSION,
    STEPS,
    STEP_AGGREGATE,
    STEP_BUILD,
    STEP_MEASURE,
    STEP_PREFLIGHT,
    STEP_RENDER,
    STEP_TIME,
    STEP_VERIFY,
    ResultsFile,
    ResultsIncomplete,
    RunBlock,
    completeness_problems,
    refuse_an_incomplete_table,
    render_results,
    smoke_sample,
)
from nbc.schema import (
    AUC_STRUCTURAL,
    AXIS_FAMILY,
    BENIGN_CLASSES,
    CENSUS_KINDS,
    CONTRAST_ATTACKS_VS_BENIGN_CLASS,
    EDIT_CENSUS_PREFIX,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    PIPELINE_STAGES,
    WILSON_SCORE,
    Auc,
    CellKey,
    Contrast,
    Interval,
    Rate,
    Verdict,
    edit_census_of,
)

BASELINE = "primary"
POLICY = "shared"
SRC = Path(__file__).resolve().parents[2] / "src"


def a_key(
    *,
    baseline: str = BASELINE,
    chain: str = CLEAN_CHAIN_NAME,
    chain_class: str = CHAIN_CLASS_BOUND,
    canon_on: bool = True,
    family: str = FAMILY_ATTACK,
    benign_class: str | None = None,
    policy: str = POLICY,
    contrast: Contrast | None = None,
) -> CellKey:
    return CellKey(
        baseline=baseline,
        dressing_chain=chain,
        chain_class=chain_class,
        window_policy=policy,
        canon_on=canon_on,
        family=family,
        benign_class=benign_class,
        contrast=contrast,
    )


def rate(**kwargs: object) -> Rate:
    return Rate(1, 2, Interval(0.0, 1.0, WILSON_SCORE), a_key(**kwargs))  # type: ignore[arg-type]


def auc(benign_class: str, **kwargs: object) -> Auc:
    return Auc(
        value=0.9,
        interval=Interval(0.8, 1.0, AUC_STRUCTURAL),
        n_positive=2,
        n_negative=2,
        tied_pairs=0,
        total_pairs=4,
        key=a_key(
            family=None,  # type: ignore[arg-type]
            benign_class=benign_class,
            contrast=Contrast(
                CONTRAST_ATTACKS_VS_BENIGN_CLASS, benign_class, frozenset({AXIS_FAMILY})
            ),
            **kwargs,  # type: ignore[arg-type]
        ),
    )


def a_complete_column(**kwargs: object) -> list[object]:
    """Every cell one `baseline x chain x canon_on` column is required to carry."""
    produced: list[object] = [rate(family=FAMILY_ATTACK, **kwargs)]  # type: ignore[arg-type]
    for benign_class in BENIGN_CLASSES:
        produced.append(
            rate(family=FAMILY_BENIGN, benign_class=benign_class, **kwargs)  # type: ignore[arg-type]
        )
        produced.append(auc(benign_class, **kwargs))
    return produced


def a_complete_table() -> list[object]:
    """Both chain classes, which is the minimum the assertion accepts."""
    return a_complete_column() + a_complete_column(
        chain="base32", chain_class=CHAIN_CLASS_HELD_OUT
    )


ONE_POLICY = {BASELINE: [POLICY]}


# --- the four keys ---------------------------------------------------------------------------------


def a_results_file() -> ResultsFile:
    return ResultsFile(
        run=RunBlock({"build_id": "abc"}),
        cells=tuple(a_complete_table()),
        verdict=(
            Verdict("N1", "not_triggered", (a_key(),), "because",
                    {"minimum_detectable_effect": 0.01}),
        ),
    )


def test_the_top_level_is_exactly_four_keys() -> None:
    payload = a_results_file().as_json_object()
    assert tuple(payload) == RESULT_KEYS
    assert payload["schema_version"] == SCHEMA_VERSION


def test_the_file_round_trips_through_json() -> None:
    results = a_results_file()
    parsed = json.loads(render_results(results))
    assert tuple(parsed) == RESULT_KEYS
    assert len(parsed["cells"]) == len(results.cells)
    assert len(parsed["verdict"]) == len(results.verdict)


def test_a_results_file_with_no_cells_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        ResultsFile(run=RunBlock({"a": 1}), cells=(), verdict=())
    assert "not a table" in str(caught.value)


def test_an_empty_run_block_is_refused() -> None:
    """Every parameter any decision mandates recording goes here, so an empty one means none were."""
    with pytest.raises(ValueError):
        RunBlock({})


def test_a_wrong_schema_version_is_refused() -> None:
    with pytest.raises(ValueError):
        ResultsFile(
            run=RunBlock({"a": 1}), cells=(rate(),), verdict=(), schema_version=SCHEMA_VERSION + 1
        )


def test_the_run_block_is_where_a_fifth_key_would_have_gone() -> None:
    """Story 4.4's findings are not parameters and go in `run` anyway. The rule's purpose is that a
    reader knows where to look, and a fifth key earned by one story is how a file grows a sixth."""
    payload = ResultsFile(
        run=RunBlock({"summary": {"findings": [{"kind": "resolution"}]}}),
        cells=(rate(),),
        verdict=(),
    ).as_json_object()
    assert tuple(payload) == RESULT_KEYS
    assert "findings" in json.dumps(payload["run"])


# --- the step order ----------------------------------------------------------------------------------


def test_the_steps_are_in_the_declared_order() -> None:
    """Each boundary is invisible in a function body written in that sequence. As a tuple it is
    something a test asserts and a reordering is a diff."""
    assert STEPS == (
        STEP_PREFLIGHT,
        STEP_VERIFY,
        STEP_BUILD,
        STEP_MEASURE,
        STEP_TIME,
        STEP_AGGREGATE,
        STEP_RENDER,
    )
    assert STEPS.index(STEP_PREFLIGHT) < STEPS.index(STEP_VERIFY)
    assert STEPS.index(STEP_VERIFY) < STEPS.index(STEP_MEASURE)
    assert STEPS.index(STEP_TIME) < STEPS.index(STEP_AGGREGATE)


RUNTIME_BINDING: Final[str] = "onnx_adapter.py"
"""The one module that IS the runtime binding, and may import it at module scope.

The rule step 0 needs is not "nobody imports onnxruntime" -- something has to. It is that nothing
on the entrypoint's path pulls it in before `platform.preflight` has run, which means no module may
import onnxruntime at module scope except this one, and no module may import THIS one at module
scope either. `run.open_baselines` imports it inside the function, which is what makes the preflight
a check on a floor rather than a check after the crash.
"""


def module_scope_imports(source: str) -> set[str]:
    """Every module name imported at module scope. Nested imports are the point and are excluded."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_the_runtime_binding_imports_onnxruntime_at_module_scope() -> None:
    offenders = [
        path.name
        for path in sorted(SRC.rglob("*.py"))
        if path.name != RUNTIME_BINDING
        and any(
            name.split(".")[0] == "onnxruntime"
            for name in module_scope_imports(path.read_text(encoding="utf-8"))
        )
    ]
    assert offenders == [], offenders


def test_nothing_imports_the_runtime_binding_at_module_scope() -> None:
    """The other half, and the one that actually protects the preflight: importing the adapter
    imports onnxruntime, so a module-scope import of the adapter is a module-scope import of the
    runtime one indirection away."""
    binding = f"nbc.baselines.{RUNTIME_BINDING.removesuffix('.py')}"
    offenders = [
        path.name
        for path in sorted(SRC.rglob("*.py"))
        if binding in module_scope_imports(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], offenders


def test_the_module_scope_scan_sees_a_top_level_import_and_not_a_nested_one() -> None:
    """Both inputs, because the scan is only a scan if the pair separates them."""
    assert "onnxruntime" in module_scope_imports("import onnxruntime\n")
    assert "onnxruntime" not in module_scope_imports(
        "def go():\n    import onnxruntime\n    return onnxruntime\n"
    )


# --- the completeness assertion -------------------------------------------------------------------------


def test_a_complete_table_has_no_problems() -> None:
    assert completeness_problems(a_complete_table(), ONE_POLICY) == ()


def test_a_column_with_no_attack_recall_is_named() -> None:
    cells = [c for c in a_complete_table() if not (isinstance(c, Rate) and c.key.family == FAMILY_ATTACK
                                                   and c.key.dressing_chain == CLEAN_CHAIN_NAME)]
    problems = completeness_problems(cells, ONE_POLICY)
    assert any("no attack recall rate" in p for p in problems)


def test_a_missing_benign_class_is_named() -> None:
    """FR3.1's distinction: a class silently absent is as unreadable as one that was pooled."""
    dropped = BENIGN_CLASSES[1]
    cells = [
        c
        for c in a_complete_table()
        if not (isinstance(c, Rate) and c.key.benign_class == dropped
                and c.key.dressing_chain == CLEAN_CHAIN_NAME)
    ]
    problems = completeness_problems(cells, ONE_POLICY)
    assert any("no false-positive rate" in p and dropped in p for p in problems)


def test_a_missing_auc_is_named() -> None:
    """A table defensible only at the threshold has not answered "you only shifted the scores"."""
    dropped = BENIGN_CLASSES[0]
    cells = [
        c
        for c in a_complete_table()
        if not (isinstance(c, Auc) and c.key.benign_class == dropped
                and c.key.dressing_chain == CLEAN_CHAIN_NAME)
    ]
    problems = completeness_problems(cells, ONE_POLICY)
    assert any("no AUC against" in p and dropped in p for p in problems)


def test_an_empty_held_out_block_is_named() -> None:
    """A table that tested no held-out chain has not tested generalization, and N4's answer would
    be about nothing."""
    problems = completeness_problems(a_complete_column(), ONE_POLICY)
    assert any(CHAIN_CLASS_HELD_OUT in p for p in problems)


def test_a_missing_window_policy_is_named() -> None:
    """A baseline declaring a publisher protocol carries both policies, and the requirement is read
    off the pins rather than assumed."""
    problems = completeness_problems(a_complete_table(), {BASELINE: [POLICY, "publisher"]})
    assert any("publisher" in p for p in problems)


def test_a_baseline_declaring_one_policy_is_satisfied_by_one() -> None:
    """The input that keeps the previous test from demanding two columns of every baseline."""
    assert completeness_problems(a_complete_table(), ONE_POLICY) == ()


def test_every_missing_cell_is_named_at_once() -> None:
    """A reader who fixes one and re-runs an eighty-five-hour pass to find the second has been told
    the truth twice and helped once."""
    problems = completeness_problems([rate(family=FAMILY_ATTACK)], ONE_POLICY)
    assert len(problems) >= 3
    with pytest.raises(ResultsIncomplete) as caught:
        refuse_an_incomplete_table(problems)
    assert str(len(problems)) in str(caught.value)


def test_a_complete_table_passes_the_gate() -> None:
    refuse_an_incomplete_table(completeness_problems(a_complete_table(), ONE_POLICY))


def test_a_contrast_cell_is_not_counted_as_a_published_estimate() -> None:
    """A `Delta` and a companion share a column with the cells the assertion demands. Counting one
    as a recall rate would let a table satisfy the assertion without carrying the rate."""
    from nbc.schema import (
        AXIS_CANON_ON,
        CONTRAST_CANON_ON_VS_OFF,
        NEWCOMBE_PAIRED,
        Delta,
    )

    delta = Delta(
        value=0.1,
        interval=Interval(0.0, 0.2, NEWCOMBE_PAIRED),
        key=CellKey(
            baseline=BASELINE,
            dressing_chain=CLEAN_CHAIN_NAME,
            chain_class=CHAIN_CLASS_BOUND,
            window_policy=POLICY,
            canon_on=None,
            family=FAMILY_ATTACK,
            benign_class=None,
            contrast=Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON})),
        ),
    )
    problems = completeness_problems([delta], ONE_POLICY)
    assert all("no attack recall rate" not in p for p in problems)


# --- the stage vocabulary ---------------------------------------------------------------------------------


def test_the_two_spellings_of_the_pipeline_stages_agree() -> None:
    """`schema` is a leaf and cannot import the pipeline, so the names live in both. This is the
    comparison that makes a second spelling safe rather than a second source of truth, and its two
    sides come from two modules."""
    assert PIPELINE_STAGES == tuple(stage.name for stage in PIPELINE)


def test_every_stage_has_a_census_and_the_vocabulary_is_built_from_them() -> None:
    for stage in PIPELINE_STAGES:
        assert edit_census_of(stage) in CENSUS_KINDS
        assert edit_census_of(stage).startswith(EDIT_CENSUS_PREFIX)
    assert len(CENSUS_KINDS) == 2 + len(PIPELINE_STAGES)


def test_a_census_for_a_stage_the_pipeline_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError):
        edit_census_of("transliterate")


def test_the_new_abort_declares_exit_code_34_and_declares_it_once() -> None:
    assert declared_exit_codes()[34] is ResultsIncomplete
    assert ResultsIncomplete.exit_code == 34


# --- the entrypoint's two gates that need no model ------------------------------------------------------


def test_the_report_subcommand_aborts_rather_than_rendering_nothing() -> None:
    """An empty rendered block and a rendered block are indistinguishable to a reader who did not
    run the command, so the seam says what is missing instead of writing one."""
    from nbc.harness.run import main

    assert main(["report"]) == ResultsIncomplete.exit_code


def test_a_partially_present_corpus_aborts_rather_than_rebuilding(tmp_path: Path) -> None:
    """The state where a rebuild writes half-new rows against a manifest describing the old ones.
    Reached without a model, because the check runs at step 2 and inference starts at step 3."""
    from nbc.corpus.manifest import CORPUS_FILENAMES, corpus_directory
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    directory = corpus_directory(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CORPUS_FILENAMES[0]).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), root=tmp_path)
    assert "partially present" in str(caught.value)
    assert CORPUS_FILENAMES[1] in str(caught.value)


def test_a_wholly_absent_corpus_says_to_build_it_rather_than_building_it(tmp_path: Path) -> None:
    """A build is a decision about which rows exist and a measurement is not, so the measuring
    command does not make one silently."""
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), root=tmp_path)
    assert "build-corpus" in str(caught.value)


def test_the_trace_file_is_gitignored() -> None:
    """Its consumer is a person debugging one document; what the table needs is the per-stage edit
    counts, which are Count cells in the results file."""
    from nbc.harness.run import TRACES_FILENAME

    ignored = (Path(__file__).resolve().parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert TRACES_FILENAME in ignored


# --- the smoke profile ------------------------------------------------------------------------------


def a_corpus_item(index: int, *, benign_class: str | None = None, chain: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    from nbc.corpus.matrix import item_id
    from nbc.schema import ATTACK, BENIGN, CorpusItem

    attack = benign_class is None
    return CorpusItem(
        id=item_id(f"{index:016x}", chain),
        source="test",
        family=FAMILY_ATTACK if attack else FAMILY_BENIGN,
        benign_class=None if attack else benign_class,
        dressing=chain,
        text="x",
        label=ATTACK if attack else BENIGN,
    )


def a_corpus() -> list[object]:
    """Two families, both benign classes and two chains -- every group the assertion demands."""
    rows: list[object] = [a_corpus_item(i) for i in range(8)]
    rows += [a_corpus_item(100 + i, benign_class=BENIGN_CLASSES[0]) for i in range(8)]
    rows += [
        a_corpus_item(200 + i, benign_class=BENIGN_CLASSES[1], chain=("base64",))
        for i in range(8)
    ]
    return rows


def test_the_smoke_sample_covers_every_group() -> None:
    """Per cell and not a total, because the smoke run executes the same completeness assertion: a
    total drawn from the whole corpus can miss a class or a chain and abort at that assertion, which
    is a gate going red because of the sampling rather than because of the code."""
    sampled = smoke_sample(a_corpus(), 3)
    groups = {(row.family, row.benign_class, row.dressing) for row in sampled}  # type: ignore[attr-defined]
    assert len(groups) == 3
    assert len(sampled) == 9


def test_the_smoke_sample_does_not_depend_on_how_the_corpus_was_read() -> None:
    """Content-derived, the same argument story 4.2 made about shard membership: a sample taken by
    row position would score a different set on a re-read and two smoke runs would not compare."""
    forward = [row.id for row in smoke_sample(a_corpus(), 3)]  # type: ignore[attr-defined]
    backward = [row.id for row in smoke_sample(list(reversed(a_corpus())), 3)]  # type: ignore[attr-defined]
    assert forward == backward


def test_a_group_smaller_than_the_sample_contributes_all_of_it() -> None:
    """Not an error: raising would make a smoke run's success depend on the corpus being large
    enough in every cell, which is a different requirement from the one being checked."""
    sampled = smoke_sample(a_corpus(), 100)
    assert len(sampled) == len(a_corpus())


def test_a_sample_of_nothing_is_refused() -> None:
    with pytest.raises(ResultsIncomplete) as caught:
        smoke_sample(a_corpus(), 0)
    assert "not a sample" in str(caught.value)


def test_the_profile_vocabulary_is_the_two() -> None:
    assert PROFILES == (PROFILE_FULL, PROFILE_SMOKE)


def test_the_declared_smoke_size_is_a_positive_int() -> None:
    """Its own named key in the pins file, so a run that says it was a smoke run says how small a
    one -- checked against the committed declaration rather than a fixture."""
    from nbc.pins import load_pins

    smoke = load_pins(None).smoke
    assert isinstance(smoke.items_per_cell, int)
    assert smoke.items_per_cell >= 1
    assert smoke.as_run_fields() == {"items_per_cell": smoke.items_per_cell}


def test_a_smoke_run_refuses_the_published_results_directory() -> None:
    """A smoke table is structurally identical to the published one with a small n as the only
    tell, so the command refuses to write one over the other."""
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), profile=PROFILE_SMOKE)
    assert "may not write into the repository" in str(caught.value)


def test_a_profile_outside_the_vocabulary_is_refused() -> None:
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), profile="quick")
    assert "must be one of" in str(caught.value)


def test_a_full_run_into_the_published_root_is_not_refused_for_that_reason(tmp_path: Path) -> None:
    """The input that keeps the smoke guard from being a guard on every run: a full profile writing
    to the repository is exactly what a published run does."""
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), profile=PROFILE_FULL, root=tmp_path)
    assert "may not write into the repository" not in str(caught.value)


def test_all_three_corpus_reads_apply_the_profile() -> None:
    """The defect the tag workflow found: `--profile` reached `full_run` and not `score_shard`, so
    a smoke job would have scored the entire corpus -- about eighty-five hours on a runner -- and
    then failed the merge, which computes its demand set over whatever corpus it was handed.

    Read from the syntax tree: every call to `read_corpus` in the entrypoint is followed by the one
    helper that applies the profile, so a fourth pass added later cannot quietly read the whole
    corpus.
    """
    import ast

    source = (SRC / "nbc" / "harness" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "read_corpus"
    ]
    applies = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_items_for_profile"
    ]
    assert len(reads) >= 3, reads
    # The timing subcommand reads the corpus too and measures the whole of it on purpose, so the
    # count is not required to match; what is required is that no read is left without one nearby.
    for line in reads:
        assert any(abs(line - other) <= 3 for other in applies) or _is_the_timing_pass(source, line), (
            f"run.py:{line} reads the corpus without applying the profile"
        )


def _is_the_timing_pass(source: str, line: int) -> bool:
    """`timing_pass` measures the whole corpus deliberately: a cost per document is a property of
    the layer, not of how many documents a profile chose to score."""
    before = "\n".join(source.splitlines()[:line])
    return before.rfind("def timing_pass(") > before.rfind("def full_run(")
