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

from nbc.canon.pipeline import PIPELINE, trace_stage_labels
from nbc.corpus.matrix import CHAIN_CLASS_BOUND, CHAIN_CLASS_HELD_OUT, CLEAN_CHAIN_NAME
from nbc.errors import EXIT_OK, declared_exit_codes
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
from nbc.report.caveats import RESULTS_END, RESULTS_START
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

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
"""The repository this suite checks, and the one no test may write into."""

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
    )
    assert STEPS.index(STEP_PREFLIGHT) < STEPS.index(STEP_VERIFY)
    assert STEPS.index(STEP_VERIFY) < STEPS.index(STEP_MEASURE)
    assert STEPS.index(STEP_TIME) < STEPS.index(STEP_AGGREGATE)


def test_rendering_is_not_one_of_the_steps_of_the_measuring_run() -> None:
    """`render` sat in `STEPS` for two stories with no code path emitting it.

    A declared step nothing takes says the run ends at a published table when it ends at a file --
    and a reader of `run.steps` could not tell the difference. It is now declared where
    `reaggregate` is, beside the command that performs it, and the measuring run leaves the README
    alone. What refuses a stale published block is a test that renders the committed results file
    and compares it, not a step nobody took.
    """
    from nbc.harness.run import STEP_REAGGREGATE, STEP_RENDER

    assert STEP_RENDER == "render"
    assert STEP_RENDER not in STEPS
    assert STEP_REAGGREGATE not in STEPS


def test_the_report_subcommand_records_the_render_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """And the constant is emitted by a code path, which is what it was not."""
    from nbc.harness.run import STEP_RENDER, main

    root = _report_root(tmp_path)
    assert main(["--root", str(root), "report"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["steps"] == [STEP_RENDER]


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


def test_the_census_axis_is_narrower_than_what_a_trace_can_carry() -> None:
    """The half of the declaration the agreement above does not read, and what it cost.

    A `PipelineStage` carries a `name` AND, where it has a ceiling entry point, a `ceiling_name`.
    The trace stamps edits with both. Comparing only the names agreed with half the declaration and
    passed for the whole of epic 4; the first real run then aborted in `aggregate.read_traces`,
    which had validated the trace against `PIPELINE_STAGES` and refused `decode-ceiling` as "a stage
    nobody ran" -- on a corpus that carries a chain nested past the ceiling BY REQUIREMENT (AD-20),
    so the abort was certain the first time anything real was measured.

    Asserted as a strict containment rather than as an equality of two lists, because the point is
    that these are two different sets: `PIPELINE_STAGES` is the census axis, `trace_stage_labels()`
    is the trace vocabulary, and a change that collapsed them would pass an equality.
    """
    labels = trace_stage_labels()
    assert set(PIPELINE_STAGES) < set(labels), "the trace vocabulary must be the wider set"
    ceilings = {stage.ceiling_name for stage in PIPELINE if stage.ceiling_name}
    assert ceilings, "no stage declares a ceiling entry point; this test would be vacuous"
    assert set(labels) == set(PIPELINE_STAGES) | ceilings
    # And it is generated from the stages that HAVE one, not from all four: a label no stage can
    # stamp stays refusable. This is the distractor -- a wider rule would accept it.
    assert "invisible-ceiling" not in labels


def test_a_trace_carrying_a_ceiling_label_is_read_and_a_fabricated_one_is_refused(
    tmp_path: Path,
) -> None:
    """The two inputs the abort turned on, through the reader that aborted.

    The first is the line the real run produced. The second is a label shaped like a ceiling label
    for a stage that declares no ceiling entry point, which must still be refused -- otherwise the
    fix would have been "accept anything ending in -ceiling", which is a check that stopped
    checking.
    """
    from nbc.harness.aggregate import CellsInvalid, read_traces

    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"item_id": "a::clean", "stages": ["decode", "decode-ceiling"]}) + "\n",
        encoding="utf-8",
    )
    assert read_traces(path)["a::clean"] == ("decode", "decode-ceiling")

    path.write_text(
        json.dumps({"item_id": "a::clean", "stages": ["invisible-ceiling"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CellsInvalid, match="invisible-ceiling"):
        read_traces(path)


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


def _report_root(tmp_path: Path) -> Path:
    """A root holding everything `report` reads: the pins, the results file and a README.

    Every byte is copied out of the repository into `tmp_path`. The earlier version of this test
    invoked `main(["report"])` at the repository root, which was safe only while the subcommand
    aborted -- the day it started writing, the suite would have published a table into the real
    README from inside a test run.
    """
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    (root / "pins.toml").write_text(
        (REPO_ROOT / "pins.toml").read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    (root / "results" / "results.json").write_text(
        (REPO_ROOT / "results" / "results.json").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    # **The block is emptied**, so the success path has something to do. Copied whole, the README
    # and the results file are already in sync: `render_into` finds nothing to write, returns
    # `readme_changed: False`, and a test asserting the markers are non-empty passes on the bytes
    # the fixture arrived with without ever watching an injection.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index(RESULTS_START) + len(RESULTS_START)
    (root / "README.md").write_text(
        readme[:start] + "\n" + readme[readme.index(RESULTS_END) :],
        encoding="utf-8",
        newline="\n",
    )
    return root


def test_the_report_subcommand_injects_the_block_into_the_readme_at_its_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The success path, and the assertion that it is the README **at the root** that changed.

    Swapping the injected path for anything else turns this red: the block has to land in the file
    `--root` names, and the repository's own README has to be untouched by the run.
    """
    from nbc.report.caveats import RESULTS_END, RESULTS_START
    from nbc.harness.run import main

    from nbc.report.caveats import RESULTS_END, RESULTS_START

    root = _report_root(tmp_path)
    before = (REPO_ROOT / "README.md").read_bytes()
    stale = (root / "README.md").read_text(encoding="utf-8")
    assert not stale[
        stale.index(RESULTS_START) + len(RESULTS_START) : stale.index(RESULTS_END)
    ].strip(), "the fixture is already in sync, so this test would watch no injection"

    assert main(["--root", str(root), "report"]) == EXIT_OK

    written = (root / "README.md").read_text(encoding="utf-8")
    body = written[written.index(RESULTS_START) + len(RESULTS_START) : written.index(RESULTS_END)]
    assert body.strip(), "the markers at the root are still empty"
    assert written != stale
    assert written[: written.index(RESULTS_START)] == stale[: stale.index(RESULTS_START)]
    assert (REPO_ROOT / "README.md").read_bytes() == before

    report = json.loads(capsys.readouterr().out)
    assert report["readme"] == str(root / "README.md")
    assert report["readme_changed"] is True
    assert report["cells_rendered"] > 0


def test_the_report_subcommand_aborts_rather_than_rendering_nothing(tmp_path: Path) -> None:
    """A root with no results file writes no block. An empty rendered block and a rendered one are
    indistinguishable to a reader who did not run the command, so the seam says what is missing."""
    from nbc.report.readme import ReportNotRenderable
    from nbc.harness.run import main

    root = _report_root(tmp_path)
    before = (root / "README.md").read_bytes()
    (root / "results" / "results.json").unlink()

    assert main(["--root", str(root), "report"]) == ReportNotRenderable.exit_code == 36
    assert (root / "README.md").read_bytes() == before


def _root_with_readme(tmp_path: Path) -> Path:
    """A temporary root carrying the honesty section, which every entrypoint verifies at step 1.

    `full_run` and `reaggregate` read the README **at their own root** rather than the one in the
    working directory, so a root without one aborts at the caveats check before reaching whatever
    the caller is testing. Copied rather than faked: the check is against the section this
    repository actually ships.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    return tmp_path


def test_the_run_verifies_the_readme_at_its_own_root_and_not_the_working_directory(
    tmp_path: Path,
) -> None:
    """`verify_caveats_file()` with no argument resolves `README.md` against the working directory.

    Under `--root` that made the honesty check read one file while `report` injected into another,
    which is exactly the split `readme_path` exists to close. The gutted section is at the root, so
    a run that still read the repository's own README would pass here and this would go green for
    the wrong reason.
    """
    from nbc.harness.run import full_run, reaggregate
    from nbc.pins import load_pins
    from nbc.report.caveats import CaveatsSectionMissing

    root = _root_with_readme(tmp_path)
    gutted = (root / "README.md").read_text(encoding="utf-8")
    (root / "README.md").write_text(
        gutted.replace("## What this does not show", "## What this deliberately omits"),
        encoding="utf-8",
        newline="\n",
    )
    assert "What this does not show" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for command in (full_run, reaggregate):
        with pytest.raises(CaveatsSectionMissing) as caught:
            command(load_pins(None), root=root)
        assert "no '## What this does not show' heading" in str(caught.value)


def test_a_partially_present_corpus_aborts_rather_than_rebuilding(tmp_path: Path) -> None:
    """The state where a rebuild writes half-new rows against a manifest describing the old ones.
    Reached without a model, because the check runs at step 2 and inference starts at step 3."""
    from nbc.corpus.manifest import CORPUS_FILENAMES, corpus_directory
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    directory = corpus_directory(_root_with_readme(tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CORPUS_FILENAMES[0]).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), root=_root_with_readme(tmp_path))
    assert "partially present" in str(caught.value)
    assert CORPUS_FILENAMES[1] in str(caught.value)


def test_a_wholly_absent_corpus_says_to_build_it_rather_than_building_it(tmp_path: Path) -> None:
    """A build is a decision about which rows exist and a measurement is not, so the measuring
    command does not make one silently."""
    from nbc.harness.run import full_run
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        full_run(load_pins(None), root=_root_with_readme(tmp_path))
    assert "build-corpus" in str(caught.value)


# --- re-deriving the table over a run that already happened ---------------------------------------------


def a_previous_results_file(root: Path, **overrides: object) -> Path:
    """A results file shaped like one `all` wrote, with only the fields `reaggregate` reads.

    The cells and the verdict are empty on purpose: this command never looks at them. It reads the
    run block, re-derives everything else from the scores, and a fixture carrying a plausible table
    would invite a test to assert against numbers nobody measured.
    """
    from nbc.harness.run import RESULTS_FILENAME, results_directory
    from nbc.harness.timing import BATCH_SIZE_ONE, CORPUS_CLASSES

    percentiles = {"p50_ns": 11, "p95_ns": 22, "n": 3}
    run: dict[str, object] = {
        "build_id": "a-build-nobody-here-has",
        "timing": {
            "layer_ns": {
                "overall": dict(percentiles),
                "by_class": {name: dict(percentiles) for name in CORPUS_CLASSES},
                "trace_enabled": True,
            },
            "inference_ns": {
                "by_baseline": {"primary": dict(percentiles)},
                "batch_size": BATCH_SIZE_ONE,
            },
            "elapsed_ns": 5,
        },
        "declared_path": {"note": "carried forward"},
        "profile": "full",
        "profile_items": 28600,
        "steps": ["preflight", "verify", "build", "measure", "time", "aggregate"],
        "total_wall_ns": 99,
    }
    run.update(overrides)

    directory = results_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESULTS_FILENAME
    path.write_text(
        json.dumps({"schema_version": 1, "run": run, "cells": [], "verdict": []}),
        encoding="utf-8",
    )
    return path


def test_reaggregating_with_no_previous_file_says_which_command_produces_one(
    tmp_path: Path,
) -> None:
    """This command re-derives a table; it does not measure one. Producing the first is `all`."""
    from nbc.harness.run import reaggregate
    from nbc.pins import load_pins

    with pytest.raises(ResultsIncomplete) as caught:
        reaggregate(load_pins(None), root=_root_with_readme(tmp_path))
    assert "results.json" in str(caught.value) and "`all`" in str(caught.value)


def test_reaggregating_without_a_timing_block_refuses_before_doing_the_work(
    tmp_path: Path,
) -> None:
    """N3 has no right-hand side without it, so the condition would be `not_evaluable` and abort --
    after the aggregation. Refused up front instead, naming the field."""
    from nbc.harness.run import reaggregate
    from nbc.pins import load_pins

    path = a_previous_results_file(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["run"]["timing"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResultsIncomplete) as caught:
        reaggregate(load_pins(None), root=_root_with_readme(tmp_path))
    assert "timing" in str(caught.value)


def test_reaggregating_without_the_trace_file_refuses_rather_than_emitting_no_census(
    tmp_path: Path,
) -> None:
    """The quiet failure this command is most exposed to.

    `aggregate.read_traces` reads a missing file as a run with no traces and returns an empty
    mapping, so `edit_census_cells` emits nothing and the table comes out missing a whole family of
    cells with no error anywhere. A crash is better than a table a reader has to count to distrust.
    """
    from nbc.harness.run import TRACES_FILENAME, reaggregate
    from nbc.pins import load_pins

    a_previous_results_file(tmp_path)

    with pytest.raises(ResultsIncomplete) as caught:
        reaggregate(load_pins(None), root=_root_with_readme(tmp_path))
    assert TRACES_FILENAME in str(caught.value)
    assert "census" in str(caught.value)


def test_a_run_block_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    from nbc.harness.run import RESULTS_FILENAME, reaggregate, results_directory
    from nbc.pins import load_pins

    directory = results_directory(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULTS_FILENAME).write_text(
        json.dumps({"schema_version": 1, "run": [], "cells": [], "verdict": []}),
        encoding="utf-8",
    )
    with pytest.raises(ResultsIncomplete) as caught:
        reaggregate(load_pins(None), root=_root_with_readme(tmp_path))
    assert "no run block" in str(caught.value)


def test_scores_from_another_corpus_are_refused() -> None:
    """The one input this cheap path can be handed that produces a plausible and wrong table.

    Fired directly rather than through the command, because reaching it needs a built corpus in a
    temporary root and building one is a decision this suite does not make.
    """
    from nbc.harness.run import refuse_a_corpus_the_scores_do_not_describe

    refuse_a_corpus_the_scores_do_not_describe("same", "same")  # does not raise

    with pytest.raises(ResultsIncomplete) as caught:
        refuse_a_corpus_the_scores_do_not_describe("this-corpus", "the-one-scored")
    assert "this-corpus" in str(caught.value) and "the-one-scored" in str(caught.value)
    assert "wrong and looks right" in str(caught.value)


def test_the_aggregate_subcommand_is_wired_to_the_reaggregating_path(tmp_path: Path) -> None:
    """The seam between the CLI and the function, checked where it costs nothing: a root with the
    committed pins and no previous results file reaches the first refusal and reports its code."""
    from nbc.harness.run import main
    from nbc.pins import PINS_FILENAME

    root = _root_with_readme(tmp_path)
    (root / PINS_FILENAME).write_bytes((REPO_ROOT / PINS_FILENAME).read_bytes())

    assert main(["--root", str(root), "aggregate"]) == ResultsIncomplete.exit_code


def test_reaggregating_records_that_it_measured_nothing() -> None:
    """The honesty requirement, asserted on the code rather than on a run.

    A results file whose latencies were produced by another process and does not say so is the
    artifact this project is not. `run.reaggregated` is where it says so, and the step name is
    `reaggregate` rather than `aggregate` so a reader diffing two files can tell the cheap act from
    the whole run.
    """
    from nbc.harness.run import STEP_REAGGREGATE
    from nbc.harness.results import STEP_AGGREGATE

    source = (Path(__file__).resolve().parents[2] / "src/nbc/harness/run.py").read_text(
        encoding="utf-8"
    )
    assert STEP_REAGGREGATE != STEP_AGGREGATE
    body = source.split("def reaggregate(")[1].split("\ndef ")[0]
    assert '"reaggregated"' in body
    assert "run_timing_pass" not in body, "this command measures no cost"
    assert "merge_shards" not in body, "this command scores nothing"
    assert "open_baselines" not in body, "this command opens no model"


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
        full_run(load_pins(None), profile=PROFILE_FULL, root=_root_with_readme(tmp_path))
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
    discards = _reads_that_discard_the_items(tree)
    # The timing subcommand reads the corpus too and measures the whole of it on purpose, so the
    # count is not required to match; what is required is that no read is left without one nearby.
    for line in reads:
        exempt = _is_the_timing_pass(source, line) or line in discards
        assert any(abs(line - other) <= 3 for other in applies) or exempt, (
            f"run.py:{line} reads the corpus without applying the profile"
        )


def _reads_that_discard_the_items(tree: "ast.Module") -> set[int]:
    """`reaggregate` reads the corpus for the manifest alone -- it re-reads scores a previous run
    wrote and opens no model -- so it binds the items to `_`. A profile chooses how many documents
    to score, and a pass that never binds a document cannot score one; that is why the discard is
    the exemption rather than the name of the function, which would have to be widened again for
    the next such pass.
    """
    import ast

    discarding = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            continue
        if value.func.id != "read_corpus":
            continue
        for target in node.targets:
            if isinstance(target, ast.Tuple) and len(target.elts) == 2:
                items = target.elts[1]
                if isinstance(items, ast.Name) and items.id == "_":
                    discarding.add(value.lineno)
    return discarding


def _is_the_timing_pass(source: str, line: int) -> bool:
    """`timing_pass` measures the whole corpus deliberately: a cost per document is a property of
    the layer, not of how many documents a profile chose to score."""
    before = "\n".join(source.splitlines()[:line])
    return before.rfind("def timing_pass(") > before.rfind("def full_run(")
