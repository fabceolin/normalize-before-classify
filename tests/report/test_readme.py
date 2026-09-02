"""The generated block, proved against the file it is a function of and against every way it aborts.

The repository's own `results/results.json` is the happy path, and it is the only input that can
prove completeness at scale: 1157 cells, four verdicts and 197 findings, all of which have to reach
a reader or the render has to say which did not. Every failure is exercised against a payload built
from a fixture, because the interesting inputs are the ones the committed file must never be.

Nothing here writes into the repository's own `README.md`. Every injection goes to `tmp_path`,
which is what `--readme` exists for: the earlier version of the report seam was tested by invoking
the entrypoint at the repository root, and the day the subcommand stopped aborting that test would
have published a table from inside the suite.
"""

from __future__ import annotations

import ast
import copy
import re
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nbc.errors import EXIT_OK, declared_exit_codes, exit_code_for
from nbc.report import readme as renderer
from nbc.report.caveats import ABSTRACT_END, ABSTRACT_START, RESULTS_END, RESULTS_START
from nbc.report.readme import (
    AXES,
    DEFAULT_RESULTS,
    HEADLINE_WINDOW_POLICY,
    ReportNotRenderable,
    SCHEMA_VERSION,
    inject,
    inject_abstract,
    load_results,
    main,
    render,
    render_abstract,
    render_into,
)

SRC = Path(__file__).resolve().parents[2] / "src"
MODULE = SRC / "nbc" / "report" / "readme.py"


# --- fixtures ------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def published(repo_root: Path) -> dict[str, Any]:
    """The committed results file, parsed. Read once; every mutating test deep-copies it."""
    return json.loads((repo_root / DEFAULT_RESULTS).read_text(encoding="utf-8"))


@pytest.fixture
def payload(published: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(published)


def a_readme(before: str = "# a repository\n\nprose above.\n\n", after: str = "\nprose below.\n") -> str:
    return f"{before}{RESULTS_START}\n{RESULTS_END}\n{after}"


def write(directory: Path, payload: dict[str, Any], readme: str | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results.json"
    results.write_text(json.dumps(payload), encoding="utf-8")
    target = directory / "README.md"
    target.write_text(a_readme() if readme is None else readme, encoding="utf-8")
    return results, target


def failures_of(caught: pytest.ExceptionInfo[ReportNotRenderable]) -> str:
    return "\n".join(caught.value.failures)


# --- the import bound, with its own red input -----------------------------------------------------
#
# Modelled on tests/canon/test_import_bound.py: the scanner reads the syntax tree rather than the
# text, and is itself checked against a module that violates the rule, because a scanner nobody has
# seen report anything is not a scanner.

ALLOWED = frozenset({"nbc.errors", "nbc.report.caveats"})
"""The whole of what the renderer may import outside the standard library.

`nbc.harness.*` would drag `onnxruntime` in and would let a renderer recompute a figure it failed
to find; `nbc.pins` would drag `pins.toml` in. Either turns the block into a function of more than
the results file."""


def imported_modules(path: Path, src: Path = SRC) -> set[str]:
    package = path.relative_to(src).with_suffix("").parts[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                prefix = ".".join(base)
                names.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                names.add(node.module)
    return names


def offending_imports(path: Path, src: Path = SRC) -> list[str]:
    offenders = []
    for name in sorted(imported_modules(path, src)):
        if name.split(".")[0] in sys.stdlib_module_names:
            continue
        if name in ALLOWED:
            continue
        offenders.append(name)
    return offenders


def test_the_renderer_imports_only_what_the_bound_allows() -> None:
    assert offending_imports(MODULE) == []


def test_the_import_scan_reports_a_module_that_breaks_the_bound(tmp_path: Path) -> None:
    """The scanner's own red input, built under `tmp_path` and never inside the shipped package.

    The first version of this test wrote the offender into `src/nbc/report/` and unlinked it in a
    `finally`. An interrupted run would have left a module importing the harness inside the package
    whose whole point is that it does not -- in a suite whose rule is that every write goes to
    `tmp_path`.
    """
    src = tmp_path / "src"
    offender = src / "nbc" / "report" / "_breaks_the_bound.py"
    offender.parent.mkdir(parents=True)
    offender.write_text(
        "'import nbc.pins is fine inside a docstring'\n"
        "from nbc.harness import run\n"
        "import nbc.pins\n"
        "import json\n",
        encoding="utf-8",
    )
    assert offending_imports(offender, src) == ["nbc.harness", "nbc.pins"]
    assert not (SRC / "nbc" / "report" / "_breaks_the_bound.py").exists()


def test_the_renderer_does_not_import_the_inference_runtime() -> None:
    """A renderer that opened a model would be a renderer that could recompute what it read."""
    code = "import sys, nbc.report.readme; print('onnxruntime' in sys.modules, 'nbc.pins' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False False", completed.stdout


# --- the three restated vocabularies, bound to their owners in both directions -------------------
#
# The renderer may not import `nbc.harness.results`, `nbc.schema` or `nbc.pins` -- that is the whole
# point of it -- so each of these three constants is a second spelling of somebody else's
# declaration. A second spelling with nothing holding it to the first is a defect that surfaces at
# the end of the next eighty-five-minute run. This test file has no import bound, so the binding
# lives here, in the two-direction shape `src/nbc/baselines/tokenization.py:112-125` already uses:
# neither side may hold a value the other does not.


def test_the_schema_version_is_the_one_the_producer_writes() -> None:
    """Bump `results.SCHEMA_VERSION` and the renderer refuses every file the producer emits."""
    from nbc.harness.results import SCHEMA_VERSION as PRODUCED

    assert renderer.SCHEMA_VERSION == PRODUCED


def test_the_axes_are_the_fields_the_cell_key_serializes_in_that_order() -> None:
    """`AXES` is the key's field order, and the order is load-bearing: it is the identity tuple.

    Both directions at once, because it is an ordered comparison: an axis added to `CellKey`, one
    removed, and any reordering all fail here rather than in a render that quietly groups by a
    different coordinate.
    """
    import dataclasses

    from nbc.schema import CellKey

    assert renderer.AXES == tuple(field.name for field in dataclasses.fields(CellKey))


def test_every_pinnable_window_policy_is_either_the_headline_or_a_declared_sensitivity_pass() -> None:
    """Both directions: a policy `pins.toml` admits and this module places nowhere would abort every
    render, and a policy declared here that no pin can select is a table nothing can produce."""
    from nbc.pins import WINDOW_POLICIES

    placed = {renderer.HEADLINE_WINDOW_POLICY} | set(renderer.SENSITIVITY_WINDOW_POLICIES)

    unplaced = sorted(WINDOW_POLICIES - placed)
    assert not unplaced, (
        f"pins.toml admits window policies {unplaced} that the renderer places in no table; every "
        f"cell measured under one would abort the render"
    )
    unpinnable = sorted(placed - WINDOW_POLICIES)
    assert not unpinnable, (
        f"the renderer declares window policies {unpinnable} that no pin can select; a section "
        f"keyed on one is a table no run can fill"
    )


def test_the_headline_policy_is_the_one_the_committed_cells_were_measured_under(
    published: dict[str, Any]
) -> None:
    """The binding above is against a declaration; this is against the file that exists."""
    assert {cell["key"]["window_policy"] for cell in published["cells"]} == {
        renderer.HEADLINE_WINDOW_POLICY
    }


# --- the abort itself -----------------------------------------------------------------------------


def test_the_new_abort_declares_exit_code_36_and_declares_it_once() -> None:
    assert declared_exit_codes()[36] is ReportNotRenderable
    assert ReportNotRenderable.exit_code == 36
    assert exit_code_for(ReportNotRenderable("because")) == 36


def test_the_abort_carries_every_failure_it_found() -> None:
    abort = ReportNotRenderable("one", "two")
    assert abort.failures == ("one", "two")
    assert "one" in str(abort) and "two" in str(abort)


# --- the happy path, over the file this repository actually ships ----------------------------------


def test_the_committed_results_file_renders_every_cell_it_holds(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Completeness is enforced in the renderer; this asserts the committed file gets through it."""
    results, target = write(tmp_path, published)
    report = render_into(results, target)
    assert report["cells_rendered"] == len(published["cells"])
    assert report["verdicts_rendered"] == len(published["verdict"])
    assert report["findings_rendered"] == len(published["run"]["summary"]["findings"])
    assert report["readme_changed"] is True


def test_the_block_replaces_only_the_bytes_between_the_markers(
    published: dict[str, Any], tmp_path: Path
) -> None:
    before = a_readme()
    results, target = write(tmp_path, published, readme=before)
    render_into(results, target)
    after = target.read_text(encoding="utf-8")

    assert after.count(RESULTS_START) == 1 and after.count(RESULTS_END) == 1
    assert after[: after.index(RESULTS_START)] == before[: before.index(RESULTS_START)]
    tail = len(RESULTS_END)
    assert after[after.index(RESULTS_END) + tail :] == before[before.index(RESULTS_END) + tail :]
    body = after[after.index(RESULTS_START) + len(RESULTS_START) : after.index(RESULTS_END)]
    assert RESULTS_START not in body and RESULTS_END not in body
    assert "\r" not in after


def test_rendering_twice_leaves_the_file_byte_identical(
    published: dict[str, Any], tmp_path: Path
) -> None:
    results, target = write(tmp_path, published)
    render_into(results, target)
    first = target.read_bytes()
    report = render_into(results, target)
    assert target.read_bytes() == first
    assert report["readme_changed"] is False


def test_the_readme_keeps_its_mode_across_the_write(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """`mkstemp` creates 0600 and `os.replace` carries the temp file's mode onto the target."""
    results, target = write(tmp_path, published)
    os.chmod(target, 0o644)
    render_into(results, target)
    assert target.stat().st_mode & 0o777 == 0o644


def test_the_block_carries_no_heading_and_no_uv_command(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Two existing README guards the block must not break: heading anchors, and runnable commands."""
    results, target = write(tmp_path, published)
    render_into(results, target)
    after = target.read_text(encoding="utf-8")
    body = after[after.index(RESULTS_START) : after.index(RESULTS_END)]
    assert not [line for line in body.splitlines() if line.startswith("#")]
    assert not [line for line in body.splitlines() if line.strip().startswith("uv ")]
    assert "```" not in body


def bare_percentages(block: str) -> list[str]:
    """Every `%` in `block` that is not inside a table cell carrying an interval or a denominator.

    Scans the whole block, not the tables: the provenance bullets, the verdicts and the findings
    are prose this module also writes, and a percentage there would be one with no `n` beside it in
    a place no table guard looks.
    """
    loose: list[str] = []
    for line in block.splitlines():
        if "%" not in line:
            continue
        if not line.startswith("| ") or line.startswith("| ---"):
            loose.append(line)
            continue
        loose += [
            cell.strip()
            for cell in line.strip("|").split(" | ")
            if "%" in cell and "[" not in cell and "/" not in cell
        ]
    return loose


def test_the_bare_percentage_scan_reports_a_percentage_with_no_n() -> None:
    """The scanner's own red input, both ways: a scan nobody has seen report anything is not one."""
    assert bare_percentages("| `a` | 12.00% [1.00%, 2.00%] 3/4 |\n") == []
    assert bare_percentages("| `a` | 12.00% |\n") == ["12.00%"]
    assert bare_percentages("- the layer edited 12.00% of them\n") == [
        "- the layer edited 12.00% of them"
    ]


def test_no_percentage_reaches_the_block_without_its_n(
    published: dict[str, Any], tmp_path: Path
) -> None:
    results, target = write(tmp_path, published)
    render_into(results, target)
    assert bare_percentages(_block(target)) == []


def test_a_column_renderer_that_dropped_the_denominator_is_caught_by_that_scan(
    published: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan run against a renderer that breaks the rule, so it is a check and not a restatement."""
    monkeypatch.setitem(
        renderer._COLUMN_RENDERERS, "rate", lambda cell: renderer._percent(cell.value)
    )
    results, target = write(tmp_path, published)
    render_into(results, target)
    assert bare_percentages(_block(target)) != []


# --- the corpus the block is about ----------------------------------------------------------------


def test_the_published_results_credit_the_corpus_the_manifest_and_the_attribution_do(
    published: dict[str, Any], repo_root: Path
) -> None:
    """The README says the block, `data/manifest.json` and `data/ATTRIBUTION.md` name one corpus.

    Written because the prose was about to be written without it. A sentence in the hand-written
    README asserting something nothing verifies is the defect this repository is about, and the
    three files are produced by three different commands.
    """
    manifest = json.loads((repo_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert published["run"]["build_id"] == manifest["build_id"]

    credits = (repo_root / "data" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert f"`{manifest['build_id']}`" in credits

    by_name = {entry["name"]: entry for entry in manifest["files"]}
    assert {entry["name"] for entry in published["run"]["corpus_files"]} == set(by_name)
    for entry in published["run"]["corpus_files"]:
        assert entry["sha256"] == by_name[entry["name"]]["sha256"]
        assert entry["rows"] == by_name[entry["name"]]["rows"]


def test_the_readme_states_the_realized_repository_count_the_manifest_records(
    repo_root: Path,
) -> None:
    """The one figure the page states in prose that the generated block cannot carry.

    Every B-code interval in the block is narrowed by the fact that files from one repository
    resemble each other, and the count a reader needs in order to judge by how much is in
    `data/manifest.json` and in no cell of `results.json`. Prose is therefore the only place it can
    be said -- and a hand-transcribed figure is the one thing this repository does not leave
    unchecked, so the sentence is bound to the manifest here rather than trusted.
    """
    manifest = json.loads((repo_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    realized = manifest["reports"]["benign_draw"]["b_code"]["repositories_realized"]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    above = readme[: readme.index(RESULTS_START)]
    assert f"**{realized} repositories**" in above, (
        f"the manifest records {realized} realized B-code repositories and the page above the "
        f"block does not state that number"
    )


# --- the I/O matrix, one row at a time --------------------------------------------------------------


def test_an_unknown_schema_version_names_the_one_it_understands(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["schema_version"] = SCHEMA_VERSION + 1
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert str(SCHEMA_VERSION + 1) in str(caught.value)
    assert str(SCHEMA_VERSION) in str(caught.value)


def test_an_unknown_contrast_names_the_cell(payload: dict[str, Any], tmp_path: Path) -> None:
    payload["cells"][0]["key"]["contrast"] = "clean_vs_a_chain_no_cell_carries"
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "clean_vs_a_chain_no_cell_carries" in failures_of(caught)


def test_a_contrast_naming_a_chain_the_file_does_hold_is_accepted(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The vocabulary is derived from the file's own chains, not from a list typed here.

    Both directions in one test: retargeting a delta onto a chain the file holds is accepted, and
    renaming that chain out of every cell makes the same contrast unknown.
    """
    chain = sorted({cell["key"]["dressing_chain"] for cell in payload["cells"]} - {None})[0]
    delta = next(cell for cell in payload["cells"] if cell["kind"] == "delta")
    delta["key"]["contrast"] = f"clean_vs_{chain}"
    delta["contrast"] = {"kind": "clean_vs", "argument": chain, "spans": ["dressing_chain"]}
    results, target = write(tmp_path, payload)
    render_into(results, target)  # accepted: the chain is one the file carries

    for cell in payload["cells"]:
        if cell["key"]["dressing_chain"] == chain:
            cell["key"]["dressing_chain"] = "renamed_out_of_the_file"
    results, _ = write(tmp_path / "renamed", payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert f"clean_vs_{chain}" in failures_of(caught)


def test_an_unknown_window_policy_aborts_rather_than_dropping_the_cell(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["cells"][0]["key"]["window_policy"] = "publisher"
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "publisher" in failures_of(caught)
    assert HEADLINE_WINDOW_POLICY in failures_of(caught)


@pytest.mark.parametrize(
    "broken",
    [
        f"{RESULTS_START}\nno end marker\n",
        f"{RESULTS_START}\n{RESULTS_START}\n{RESULTS_END}\n",
        f"{RESULTS_END}\nthe block is inverted\n{RESULTS_START}\n",
        "no markers at all\n",
    ],
    ids=["no-end", "doubled-start", "inverted", "absent"],
)
def test_a_readme_whose_markers_cannot_be_located_is_left_alone(
    payload: dict[str, Any], tmp_path: Path, broken: str
) -> None:
    results, target = write(tmp_path, payload, readme=broken)
    with pytest.raises(ReportNotRenderable):
        render_into(results, target)
    assert target.read_text(encoding="utf-8") == broken


def test_a_write_that_fails_leaves_the_readme_wholly_old(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.replace` is the last step, so a failure before it cannot leave a half-written file."""
    results, target = write(tmp_path, payload)
    before = target.read_text(encoding="utf-8")

    def refuse(source: str, destination: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(renderer.os, "replace", refuse)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "No space left on device" in str(caught.value)
    assert target.read_text(encoding="utf-8") == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["README.md", "results.json"]


def test_a_readme_that_cannot_be_read_is_the_same_abort(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, tmp_path / "there-is-no-readme-here.md")
    assert "there-is-no-readme-here.md" in str(caught.value)


def test_per_baseline_duplicate_chain_lists_are_published_once(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """`excluded_probes_none == ["rot13", "rot13"]` is one excluded encoding, counted per baseline."""
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    line = next(line for line in block.splitlines() if "`held_out_chains`" in line)
    assert line.count("`base32`") == 1
    assert line.count("`url_percent`") == 1
    excluded = next(line for line in block.splitlines() if "`excluded_probes_none`" in line)
    assert excluded.count("`rot13`") == 1


def test_two_opposed_chain_lists_are_resolvable_rather_than_a_flat_contradiction(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """N4 holds one chain in both the recovering and the degrading list.

    It recovers for one baseline and degrades for the other, and the lists record only the chain.
    Deduplicating them publishes the same name twice with opposite meanings and no way to resolve
    it; attributing each to how many of the verdict's baselines it accounts for does resolve it.
    """
    verdict = next(
        v
        for v in published["verdict"]
        if set(v["computed"].get("chains_recovering_off_distribution", []))
        & set(v["computed"].get("chains_degrading_off_distribution", []))
    )
    baselines = len({key["baseline"] for key in verdict["keys"]})
    assert baselines > 1, "the fixture this test rests on is gone"

    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    for name in ("chains_recovering_off_distribution", "chains_degrading_off_distribution"):
        line = next(line for line in block.splitlines() if f"`{name}`" in line)
        assert f"on 1 of {baselines}" in line, line


def test_only_a_verdicts_own_chain_lists_are_attributed_per_baseline(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """`declared_path.providers` is a list of strings and is not one entry per baseline.

    Attributing it "per baseline" would be a sentence about a structure the value does not have --
    and it read as `2 baseline-chain pairs: CPUExecutionProvider on 1` until this test existed.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    line = next(l for l in _block(target).splitlines() if "declared execution path" in l)
    assert "baseline-chain pair" not in line
    for provider in published["run"]["declared_path"]["providers"]:
        assert f"`{provider}`" in line


def test_a_chain_list_counts_every_baseline_it_accounts_for(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The denominator is the verdict's own baseline count, read from its keys, not typed here."""
    verdict = next(v for v in payload["verdict"] if len({k["baseline"] for k in v["keys"]}) == 2)
    verdict["computed"]["held_out_chains"] = ["base32", "base32", "url_percent"]
    results, target = write(tmp_path, payload)
    render_into(results, target)
    line = next(l for l in _block(target).splitlines() if "`held_out_chains`" in l)
    assert "3 baseline-chain pairs" in line
    assert "`base32` on 2 of 2" in line
    assert "`url_percent` on 1 of 2" in line


# --- freshness, which is not determinism ----------------------------------------------------------


def published_block(readme: str) -> str:
    """The bytes between the markers of `readme`, exactly as `inject` writes them."""
    start = readme.index(RESULTS_START) + len(RESULTS_START)
    return readme[start : readme.index(RESULTS_END)]


def drift(readme: str, expected_body: str) -> str | None:
    """Why the published block is not what the results file renders, or `None` when it is.

    One function, used by the freshness check and by the red input that proves the freshness check
    can fail. The comparison itself has to be the thing under test, or the red input is a test of
    `str.replace` and stays green however weak the comparison becomes.
    """
    if published_block(readme) == "\n" + expected_body:
        return None
    return (
        "the published block is not what the committed results file renders today. Run "
        "`python -m nbc.report.readme --readme README.md` and commit the result"
    )


def test_the_shipped_readme_carries_the_block_the_committed_results_render(
    repo_root: Path,
) -> None:
    """The committed README's block, against the committed results file, rendered here and now.

    Byte-identical output across two runs proves the renderer is a function of its input; it proves
    nothing about whether the block on disk is that function's *current* output. Falsifying the
    corpus `build_id` inside the published block left the whole suite green, which is the exact
    state this test refuses: a README that presents itself as a pure function of `results.json`
    while describing a different file.

    Two ways to go red, both correct: `results/results.json` changed and nobody ran
    `python -m nbc.report.readme`, or somebody edited between the markers by hand.
    """
    expected = render(load_results(repo_root / DEFAULT_RESULTS))
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    reason = drift(readme, expected)
    assert reason is None, reason


def test_the_freshness_check_notices_a_block_edited_by_hand(repo_root: Path) -> None:
    """Its own red input, run through the comparison the check above runs -- three edits of it."""
    published = load_results(repo_root / DEFAULT_RESULTS)
    expected = render(published)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert drift(readme, expected) is None, "the fixture this test rests on is gone"

    block = published_block(readme)
    for tampered in (
        readme.replace(published.run["build_id"], "0" * 64, 1),
        readme.replace(block, block + "an added line\n", 1),
        readme.replace(block, "\n", 1),
    ):
        assert tampered != readme, "the fixture this test rests on is gone"
        assert drift(tampered, expected) is not None


def test_the_freshness_check_notices_a_results_file_nobody_re_rendered(
    repo_root: Path, tmp_path: Path
) -> None:
    """The other direction: the file moved and the block did not."""
    payload = json.loads((repo_root / DEFAULT_RESULTS).read_text(encoding="utf-8"))
    payload["run"]["profile_items"] = int(payload["run"]["profile_items"]) + 1
    moved = tmp_path / "results.json"
    moved.write_text(json.dumps(payload), encoding="utf-8")

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert drift(readme, render(load_results(moved))) is not None


def test_the_saturated_confirmatory_limb_renders_from_computed_not_from_the_reason(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """`cell_could_decide` exists so the block states it without parsing a sentence for it."""
    verdict = next(v for v in payload["verdict"] if "cell_could_decide" in v["computed"])
    verdict["reason"] = "a sentence that says nothing about saturation at all"
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert "`cell_could_decide`: no" in block
    assert "`pinned_rates`" in block


def test_a_reaggregated_run_says_the_latencies_came_from_another_invocation(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["run"]["reaggregated"] = {
        "from_steps": ["verify", "build", "reaggregate"],
        "inherited": ["timing"],
        "note": "carried forward unchanged.",
    }
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert "not measured by the invocation that produced the cells" in block
    for step in ("verify", "build", "reaggregate"):
        assert f"`{step}`" in block


def test_a_run_that_was_not_reaggregated_makes_no_such_claim(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["run"].pop("reaggregated", None)
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert "not measured by the invocation that produced the cells" not in _block(target)


def test_a_file_with_no_single_window_cell_loses_the_section_and_its_lead_in(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The lead-in travels with the table. A promise of a comparison over an empty table is worse
    than silence, and a guard counting lines against a builder that always emits a header, a rule
    and a methods line can never fire."""
    results, target = write(tmp_path, payload)
    render_into(results, target)
    lead_in = "over the items that occupy one window"
    assert lead_in in _block(target)

    payload["cells"] = [
        cell for cell in payload["cells"] if cell["key"]["population"] != "single_window"
    ]
    payload["run"]["summary"]["findings"] = [
        finding
        for finding in payload["run"]["summary"]["findings"]
        if all(key["population"] != "single_window" for key in finding["keys"])
    ]
    results, target = write(tmp_path / "second", payload)
    render_into(results, target)
    assert lead_in not in _block(target)


def test_the_sensitivity_section_is_absent_when_every_cell_is_the_headline_policy(
    published: dict[str, Any], tmp_path: Path
) -> None:
    assert {cell["key"]["window_policy"] for cell in published["cells"]} == {
        HEADLINE_WINDOW_POLICY
    }
    results, target = write(tmp_path, published)
    render_into(results, target)
    assert "sensitivity pass" not in _block(target)


def test_a_declared_sensitivity_pass_renders_beside_the_headline_and_never_into_it(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second half of the same rule: a declared pass renders, in its own table.

    `pins.toml` admits one policy today, so the declared set is empty and the section is
    unreachable on any file a run can produce. Declaring one here is what makes the branch
    something that has been seen to work rather than something that reads as though it would.
    """
    monkeypatch.setattr(renderer, "SENSITIVITY_WINDOW_POLICIES", frozenset({"publisher"}))
    extra = copy.deepcopy(next(cell for cell in payload["cells"] if cell["kind"] == "rate"))
    extra["key"]["window_policy"] = "publisher"
    payload["cells"].append(extra)
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert "sensitivity pass" in block
    headline = block[block.index("The rates, per benign class") : block.index("sensitivity pass")]
    assert "publisher" not in headline


# --- the conditions, above the tables rather than only under them --------------------------------


def test_the_quiet_outcome_is_the_one_the_producer_writes() -> None:
    """Declared in the renderer for the reason `HEADLINE_WINDOW_POLICY` is, and bound to the
    producer here, so an outcome vocabulary that moves cannot leave the headline keyed on a value
    no verdict can carry."""
    from nbc.schema import OUTCOME_NOT_TRIGGERED, VERDICT_OUTCOMES

    assert renderer.QUIET_VERDICT_OUTCOME == OUTCOME_NOT_TRIGGERED
    assert renderer.QUIET_VERDICT_OUTCOME in VERDICT_OUTCOMES


def test_a_triggering_verdict_is_named_above_the_first_table(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """The committed file's N3 triggered and the block said so once, under every table. A reader
    with five minutes read the evidence and left without the conclusion.

    The headline names the outcome and not only the identifier, and it says it in positive voice:
    "1 did not come out `not_triggered`: `N3`" made a reader compose two negations to learn that a
    condition fired, and then told them only its name.
    """
    triggered = [
        (verdict["condition"], verdict["outcome"])
        for verdict in published["verdict"]
        if verdict["outcome"] != renderer.QUIET_VERDICT_OUTCOME
    ]
    assert triggered, "the fixture this test rests on is gone"

    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    headline = block[: block.index("| ")]
    for condition, outcome in triggered:
        assert f"`{condition}` came out `{outcome}`" in headline
    assert "did not come out" not in headline


def test_a_file_where_nothing_triggered_still_gets_a_line(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Never silence. Rendering "nothing fired" as absence makes the presence of the line the
    message, and a reader cannot tell it from a block that forgot to say."""
    for verdict in payload["verdict"]:
        verdict["outcome"] = renderer.QUIET_VERDICT_OUTCOME
    results, target = write(tmp_path, payload)
    render_into(results, target)
    headline = _block(target)
    headline = headline[: headline.index("| ")]
    assert "What the pre-registered conditions came out as" in headline
    assert f"came out `{renderer.QUIET_VERDICT_OUTCOME}`: none of them fired" in headline


def test_an_outcome_the_vocabulary_grows_later_is_loud_by_default(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Keyed by exclusion, so `not_evaluable` -- the artifact not working -- cannot be hidden by a
    headline that only knows how to look for `triggered`."""
    payload["verdict"][0]["outcome"] = "not_evaluable"
    for verdict in payload["verdict"][1:]:
        verdict["outcome"] = renderer.QUIET_VERDICT_OUTCOME
    results, target = write(tmp_path, payload)
    render_into(results, target)
    headline = _block(target)
    headline = headline[: headline.index("| ")]
    assert f"`{payload['verdict'][0]['condition']}` came out `not_evaluable`" in headline


def test_a_file_where_every_condition_is_loud_names_them_all_and_grows_no_tail(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The tail exists to account for the conditions the headline did not name. With nothing left
    over there is nothing to account for, and "; 0 of the 4 came out `not_triggered`" would be the
    headline reporting the empty set as a result.

    The branch that decides this shipped uncovered: deleting the `if quiet` guard left the suite
    green, because every file any test rendered had at least one quiet verdict.
    """
    for index, verdict in enumerate(payload["verdict"]):
        verdict["outcome"] = "triggered" if index % 2 else "not_evaluable"
    results, target = write(tmp_path, payload)
    render_into(results, target)
    headline = _block(target)
    headline = headline[: headline.index("| ")]

    for verdict in payload["verdict"]:
        assert f"`{verdict['condition']}` came out `{verdict['outcome']}`" in headline
    assert renderer.QUIET_VERDICT_OUTCOME not in headline
    assert "of the" in headline, "the total is still stated"


def test_a_file_that_evaluated_no_condition_gets_no_headline(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Silence, and it is chosen rather than inherited -- the deliberate exception to "never
    silence" one paragraph up.

    "Never silence" is a rule about *outcomes*: a condition that came out `not_triggered` is a
    result, and rendering it as an absent line makes the presence of the line the message. An
    empty `verdict` list has no outcome to be silent about. The two alternatives are both worse.
    An abort would refuse a file that is legitimately partial -- a results file rendered before
    anything was evaluated -- and this module's job is to render what a file holds, not to decide
    what a file must hold; the completeness that *is* worth an abort, a cell no table claims, is
    enforced in `render` against the file's own cells. A line reading "0 conditions were
    evaluated" above the first table would be worse still: the block would be announcing a finding
    that the aggregator never raised, in the artifact whose whole rule is that every figure came
    from the file.

    The branch shipped uncovered: `if not verdicts: return []` was removable with the suite green,
    and without it this file publishes "All 0 came out `not_triggered`: none of them fired" over
    tables whose conditions nobody evaluated.
    """
    payload["verdict"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    headline = block[: block.index("| ")]

    assert "pre-registered" not in headline
    assert "What the pre-registered conditions came out as" not in block
    assert "`rate` cells" in headline, "the first table still arrives with its lead-in"


# --- the clock, labelled for what it actually timed -----------------------------------------------


def test_the_wall_time_names_the_steps_the_invocation_ran(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """`- wall time: 1.37 min` sat three lines under "28,600 items scored" and was the
    reaggregation's own clock: the invocation that wrote it opened no model and scored nothing.

    The original scoring run's clock is not recoverable -- every re-derivation overwrites the
    field -- so the fix is the label, and the label is the file's own `run.steps`.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    line = next(line for line in block.splitlines() if line.startswith("- wall time"))
    assert line.startswith("- wall time of the steps this invocation ran (")
    for step in published["run"]["steps"]:
        assert f"`{step}`" in line
    assert "\n- wall time: " not in block


def test_the_wall_time_survives_a_run_that_records_no_steps(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The figure is the file's and stays; only the parenthesis naming the steps goes away."""
    payload["run"].pop("steps", None)
    results, target = write(tmp_path, payload)
    render_into(results, target)
    line = next(
        line for line in _block(target).splitlines() if line.startswith("- wall time")
    )
    assert line == "- wall time of the steps this invocation ran: " + renderer._duration(
        payload["run"]["total_wall_ns"]
    )


def test_a_run_that_names_no_step_emits_no_dangling_step_label(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """`steps: []` is a list that is not `None`, and the two guards over it disagreed.

    The step line tested the field for *presence* and published `- steps: ` with nothing after it;
    the wall-time parenthetical tested the same list for *truth* and silently vanished. So the
    block carried a label whose whole claim is that it names something, next to a label that had
    stopped naming anything, and neither said the file recorded no step.
    """
    payload["run"]["steps"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)

    assert "- steps: " not in block
    line = next(line for line in block.splitlines() if line.startswith("- wall time"))
    assert line == "- wall time of the steps this invocation ran: " + renderer._duration(
        payload["run"]["total_wall_ns"]
    )


# --- the fold: rendered height comes off, no byte does -------------------------------------------


def _a_section(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "name": "a section made by a test",
        "lead_in": "**the sentence that says what is in the table.**",
        "claim": lambda cell: True,
        "row_axes": ("baseline",),
        "column_axes": ("family",),
    }
    fields.update(overrides)
    return renderer.Section(**fields)


def _two_cells(payload: dict[str, Any]) -> list[Any]:
    failures: list[str] = []
    first = renderer._read_cell(
        next(c for c in payload["cells"] if c["kind"] == "rate"), 0, failures
    )
    second = renderer._read_cell(
        next(
            c
            for c in payload["cells"]
            if c["kind"] == "rate" and c["key"]["family"] != first.key[AXES.index("family")]
        ),
        1,
        failures,
    )
    assert failures == []
    return [first, second]


def test_a_folded_section_puts_its_table_in_a_details_and_its_lead_in_outside_it(
    payload: dict[str, Any],
) -> None:
    """The lead-in is the one line a reader skimming past a closed fold still has to meet.

    Both shapes from one pair of cells, so the difference in the output is the flag and nothing
    else: the fold adds a `<details>`, a `<summary>` naming the section and counting its cells,
    and the two blank lines Markdown needs to keep rendering a table inside HTML. Every row is
    still there -- a fold removes rendered height, never a figure.
    """
    cells = _two_cells(payload)
    failures: list[str] = []
    plain, _ = renderer._build_section(_a_section(), cells, _NO_ANCHORS, failures)
    folded, placed = renderer._build_section(
        _a_section(folded=True), cells, _NO_ANCHORS, failures
    )
    assert failures == []
    assert placed == {cell.identity for cell in cells}

    assert "<details>" not in "\n".join(plain)
    assert folded[1] == "**the sentence that says what is in the table.**"
    assert folded[3] == "<details><summary>a section made by a test -- 2 cells</summary>"
    assert folded[-1] == "</details>"
    assert folded[4] == "" and folded[-2] == ""
    assert [line for line in folded if line.startswith("| ")] == [
        line for line in plain if line.startswith("| ")
    ]


def test_a_folded_section_holding_one_cell_says_one_cell(payload: dict[str, Any]) -> None:
    """The summary is the only thing a reader sees with the fold shut, so it is the one place in
    the block where a plural over a count of one is read by everybody.

    The singular limb shipped uncovered: `{'' if len(claimed) == 1 else 's'}` collapsed to a bare
    `'s'` with the suite still green, because every folded section any test built held two cells
    or more. Meanwhile the findings catalogue was publishing "1 of them repeat another's
    `computed`" in the same commit -- so this is the branch that was written and never proven,
    beside the one that was never written.
    """
    cells = _two_cells(payload)
    failures: list[str] = []
    one, placed = renderer._build_section(
        _a_section(folded=True, claim=lambda cell: cell.identity == cells[0].identity),
        cells,
        _NO_ANCHORS,
        failures,
    )
    assert failures == [] and placed == {cells[0].identity}
    assert one[3] == "<details><summary>a section made by a test -- 1 cell</summary>"

    two, _ = renderer._build_section(_a_section(folded=True), cells, _NO_ANCHORS, failures)
    assert failures == []
    assert two[3] == "<details><summary>a section made by a test -- 2 cells</summary>"


def test_an_empty_folded_section_emits_nothing_at_all(payload: dict[str, Any]) -> None:
    """Not even the summary. A fold offering to expand a table that is not there is the same
    broken promise as a lead-in over one, which `:656` already refuses for the unfolded case."""
    cells = _two_cells(payload)
    failures: list[str] = []
    lines, placed = renderer._build_section(
        _a_section(folded=True, claim=lambda cell: False),
        cells,
        _NO_ANCHORS,
        failures,
    )
    assert failures == [] and placed == set()
    assert lines == []


def test_the_committed_block_folds_the_evidence_and_never_the_rates_or_the_verdicts(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Which sections fold is an editorial judgement, so it is asserted against the declaration.

    The two things a reader must meet with the fold shut are the rates -- the claim -- and the
    verdicts, which are the claim being decided. Everything folded is evidence for one of those.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)

    loaded = load_results(results)
    folded = [section for section in renderer._sections(loaded.cells) if section.folded]
    assert {section.name for section in renderer._sections(loaded.cells)} - {
        section.name for section in folded
    } == {"rates"}

    rendered = [
        section.name
        for section in folded
        if renderer._build_section(section, loaded.cells, _NO_ANCHORS, [])[0]
    ]
    assert block.count("<details>") == len(rendered)
    assert block.count("</details>") == len(rendered)
    for name in rendered:
        assert f"<summary>{name} -- " in block

    headline = block[: block.index("<details>")]
    assert "The rates, per benign class" in headline
    assert "pre-registered falsification conditions" in headline


def test_a_results_file_that_is_not_utf8_aborts_rather_than_crashing(tmp_path: Path) -> None:
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, and would escape a narrower catch."""
    results = tmp_path / "results.json"
    results.write_bytes(b'{"schema_version": 1, "run": "\xff\xfe"}')
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "UTF-8" in str(caught.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_is_named_rather_than_published(tmp_path: Path, constant: str) -> None:
    """`json.loads` accepts all three by default, and each would reach a reader as `nan%`."""
    results = tmp_path / "results.json"
    results.write_text(
        '{"schema_version": 1, "run": {"build_id": "x", "corpus_files": [], "value": '
        + constant
        + "}, \"cells\": [], \"verdict\": []}",
        encoding="utf-8",
    )
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert constant.lstrip("-") in str(caught.value)


def test_a_float_literal_that_overflows_to_infinity_is_refused(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """`1e400` is finite text and an infinite float, so `parse_constant` never sees it."""
    results = tmp_path / "results.json"
    cell = copy.deepcopy(next(c for c in payload["cells"] if c["kind"] == "rate"))
    payload["cells"] = [cell]
    text = json.dumps(payload).replace(json.dumps(cell["value"]), "1e400", 1)
    results.write_text(text, encoding="utf-8")
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "not finite" in failures_of(caught)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda cell: cell["interval"].pop("method"), "carries no method"),
        (lambda cell: cell["interval"].pop("lo"), "carries no lo"),
        (lambda cell: cell.__setitem__("interval", 3), "not an object"),
        (lambda cell: cell["key"].pop("population"), "carries no population"),
        (lambda cell: cell.pop("n"), "carries no n"),
    ],
    ids=["no-method", "no-lo", "interval-scalar", "no-axis", "no-n"],
)
def test_a_malformed_cell_is_an_abort_and_never_a_key_error(
    payload: dict[str, Any], tmp_path: Path, mutate: Any, expected: str
) -> None:
    mutate(next(cell for cell in payload["cells"] if cell["kind"] == "rate"))
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert expected in failures_of(caught)


def test_a_timing_statistic_missing_a_percentile_is_an_abort(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["run"]["timing"]["layer_ns"]["overall"].pop("p50_ns")
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "carries no p50_ns" in failures_of(caught)


def test_a_corpus_entry_missing_its_digest_is_an_abort(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["run"]["corpus_files"][0].pop("sha256")
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "carries no sha256" in failures_of(caught)


def test_a_cell_no_table_claims_aborts_naming_it(payload: dict[str, Any], tmp_path: Path) -> None:
    """Completeness is enforced here, never asserted in a test against today's file.

    The claim predicates all require the headline population; a cell in a population none of them
    names is legal, is measured, and would be invisible to a reader. It must stop the render.
    """
    orphan = copy.deepcopy(next(cell for cell in payload["cells"] if cell["kind"] == "rate"))
    orphan["key"]["population"] = "a_population_no_section_claims"
    payload["cells"].append(orphan)
    results, target = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "in no table" in failures_of(caught)
    assert "a_population_no_section_claims" in failures_of(caught)


def test_two_cells_at_one_identity_abort_rather_than_the_second_being_dropped(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["cells"].append(copy.deepcopy(payload["cells"][0]))
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "one identity is one measurement" in failures_of(caught)


def test_a_rate_and_a_count_at_the_same_nine_axes_are_not_a_duplicate(
    published: dict[str, Any]
) -> None:
    """A rate and a window-overflow census share all nine axes and are different measurements.

    Identity is the nine axes plus the kind and the census; nine alone would report one of the two
    as a duplicate of the other and the committed file would not render at all.
    """
    rates = {
        tuple(cell["key"][axis] for axis in AXES)
        for cell in published["cells"]
        if cell["kind"] == "rate"
    }
    counts = {
        tuple(cell["key"][axis] for axis in AXES)
        for cell in published["cells"]
        if cell["kind"] == "count"
    }
    assert rates & counts


@pytest.mark.parametrize(
    ("kind", "value"),
    [("rate", -0.5), ("rate", 1.5), ("auc", -0.01), ("delta", 1.5), ("delta", -1.5)],
    ids=["rate-negative", "rate-above-one", "auc-negative", "delta-high", "delta-low"],
)
def test_a_value_outside_the_range_its_kind_can_take_is_refused(
    payload: dict[str, Any], tmp_path: Path, kind: str, value: float
) -> None:
    """`_fixed` returned `abs(value)`, so a malformed rate of `-0.5` published as `50.00%`.

    There is no honest rendering of an out-of-range value, so it is refused rather than made
    presentable.
    """
    next(cell for cell in payload["cells"] if cell["kind"] == kind)["value"] = value
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "no honest way to render it" in failures_of(caught)


def test_a_magnitude_is_rendered_with_its_own_sign() -> None:
    assert renderer._fixed(0.25, 4) == "0.2500"
    assert renderer._fixed(-0.25, 4) == "-0.2500"


def test_an_interval_bound_outside_the_estimands_range_is_still_rendered(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """A structural-components interval can legitimately reach past the estimand's range."""
    auc = next(cell for cell in payload["cells"] if cell["kind"] == "auc")
    auc["value"] = 0.02
    auc["interval"] = {"lo": -0.05, "hi": 0.09, "method": "auc-structural-components"}
    payload["run"]["summary"]["findings"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert "[-0.0500, 0.0900]" in _block(target)


@pytest.mark.parametrize(
    ("axis", "value"),
    [("family", ["attack"]), ("benign_class", {"a": 1})],
    ids=["list", "object"],
)
def test_a_non_scalar_axis_value_on_a_cell_key_aborts_rather_than_crashing(
    payload: dict[str, Any], tmp_path: Path, axis: str, value: Any
) -> None:
    """It reached `dict.get(cell.identity)` as an unhashable key: exit 1 with a traceback.

    An axis is what a measurement is stored and looked up under, so one that cannot be a dictionary
    key leaves the process as a `TypeError` rather than as a report -- which falsified the promise
    that every failure to render is one abort.
    """
    payload["cells"][0]["key"][axis] = value
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "non-scalar axis value" in failures_of(caught)
    assert axis in failures_of(caught)


def test_a_non_scalar_axis_value_on_a_finding_key_aborts_rather_than_crashing(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The same coordinate, on the other side: it reached the anchor lookup instead."""
    payload["run"]["summary"]["findings"][0]["keys"][0]["benign_class"] = {"a": 1}
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "non-scalar axis value" in failures_of(caught)


def test_a_finding_kind_carrying_a_pre_formatted_figure_is_refused(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """`finding["kind"]` was the one file-supplied string reaching the reader unguarded.

    It published raw nanoseconds at exit 0, two lines from the same figures this module renders as
    durations -- while `_stored_text`'s own docstring said every string that reaches the reader
    goes through it.
    """
    payload["run"]["summary"]["findings"][0]["kind"] = "a_kind_taking_18394582 ns"
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "pre-formatted figure" in failures_of(caught)


def test_a_computed_field_name_carrying_a_figure_is_refused(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """A field name reaches the reader exactly as a value does, so it is exempt from nothing."""
    payload["verdict"][0]["computed"]["took_18394582 ns"] = 1
    results, target = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "pre-formatted figure" in failures_of(caught)


def test_a_cell_of_a_kind_this_renderer_has_no_column_for_aborts(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Four kinds, four renderers. A fifth would be rendered by nothing and dropped in silence."""
    payload["cells"][0]["kind"] = "a_kind_with_no_column"
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "no column for" in failures_of(caught)


def test_an_inverted_interval_is_refused(payload: dict[str, Any], tmp_path: Path) -> None:
    rate = next(cell for cell in payload["cells"] if cell["kind"] == "rate")
    rate["interval"] = {"lo": 0.9, "hi": 0.1, "method": "wilson-score"}
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "inverted" in failures_of(caught)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('["not an object at the top level"]', "not an object"),
        ('{"schema_version": 1, "run": 3, "cells": [], "verdict": []}', "run is int"),
        (
            '{"schema_version": 1, "run": {"build_id": "x", "corpus_files": []}, '
            '"cells": 3, "verdict": []}',
            "cells is int",
        ),
    ],
    ids=["top-level-list", "run-scalar", "cells-scalar"],
)
def test_a_top_level_of_the_wrong_shape_is_an_abort(
    tmp_path: Path, text: str, expected: str
) -> None:
    results = tmp_path / "results.json"
    results.write_text(text, encoding="utf-8")
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert expected in failures_of(caught) or expected in str(caught.value)


def test_a_verdict_missing_a_field_is_an_abort(payload: dict[str, Any], tmp_path: Path) -> None:
    payload["verdict"][0].pop("computed")
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "carries no computed" in failures_of(caught)


def test_a_finding_key_missing_an_axis_is_an_abort(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["run"]["summary"]["findings"][0]["keys"][0].pop("population")
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "carries no population" in failures_of(caught)


def test_two_cells_that_would_share_one_slot_abort_rather_than_one_being_dropped(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Distinct identities, one table slot. The second would never be rendered, silently."""
    twin = copy.deepcopy(next(cell for cell in payload["cells"] if cell["kind"] == "rate"))
    twin["key"]["contrast"] = "canon_on_vs_off"
    payload["cells"].append(twin)
    results, target = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "in one slot" in failures_of(caught)


def test_a_readme_that_is_not_utf8_is_the_same_abort(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    results, target = write(tmp_path, payload)
    target.write_bytes(b"# a repository\n\xff\xfe\n")
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "UTF-8" in str(caught.value)


def test_column_labels_fall_back_to_axis_equals_value_when_the_short_form_collides() -> None:
    """Two coordinate tuples must never print one heading, whatever the file happens to hold."""
    columns = [(None, "x"), ("x", None)]
    labels = renderer._column_labels(columns, ("family", "benign_class"))
    assert len(set(labels)) == 2
    assert all("=" in label for label in labels)


@pytest.mark.parametrize(
    "figure",
    ["0.2000%", "18394582 ns", "1000000.0 ns", "18.39 ms", "1.50 min", "+16.58 pp", "82.35 s"],
    ids=["percent", "ns", "float-ns", "ms", "min", "pp", "s"],
)
def test_stored_prose_carrying_a_pre_formatted_figure_is_refused(
    payload: dict[str, Any], tmp_path: Path, figure: str
) -> None:
    """Every unit this module writes, not percentages alone.

    N3's stored reason published `18394582 ns` and `1000000.0 ns` two lines above the same figures
    rendered as `18.39 ms` and `1.00 ms`. A guard written against `%` saw none of it.
    """
    payload["run"]["reaggregated"] = {
        "from_steps": ["build"],
        "inherited": ["timing"],
        "note": f"the layer took {figure} on this corpus.",
    }
    results, target = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "pre-formatted figure" in failures_of(caught)


@pytest.mark.parametrize(
    "harmless",
    ["a device of compute capability 8.6", "declared on 2026-08-29", "500 of 500 items"],
    ids=["compute-capability", "date", "bare-count"],
)
def test_a_stored_number_in_no_unit_this_module_writes_is_left_alone(
    payload: dict[str, Any], tmp_path: Path, harmless: str
) -> None:
    """The guard is about figures a reader would take for this module's output, not about digits.

    A CUDA compute capability is `8.6` and would abort a guard written against every decimal, on
    the committed file, for a number nothing here renders.
    """
    payload["run"]["reaggregated"] = {
        "from_steps": ["build"],
        "inherited": ["timing"],
        "note": harmless,
    }
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert harmless in _block(target)


def test_a_string_inside_a_computed_block_goes_through_the_same_guard(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The guard is about what reaches a reader, not about which field it arrived in."""
    payload["verdict"][0]["computed"]["binding_ceiling"] = "the absolute, 1000000.0 ns"
    results, target = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "pre-formatted figure" in failures_of(caught)


def test_an_axis_value_carrying_a_formatted_figure_is_refused(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Axis values become row and column labels, which is a string reaching a reader."""
    payload["cells"][0]["key"]["dressing_chain"] = "a_chain_taking_18394582 ns"
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "pre-formatted figure" in failures_of(caught)


def test_no_verdict_reason_is_published(published: dict[str, Any], tmp_path: Path) -> None:
    """The evaluator's sentence stays in `results.json`.

    It carries figures the evaluator formatted -- N3's reads `18394582 ns` against `1000000.0 ns`,
    two lines above the same numbers rendered here as `18.39 ms` and `1.00 ms`. Three spellings of
    two figures in one document is the drift the block exists to end, so verdicts render from
    `computed` exactly as findings do.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    for verdict in published["verdict"]:
        sentence = " ".join(verdict["reason"].split())
        assert sentence not in block
        assert sentence.split(".")[0] not in block
        assert f"`{verdict['condition']}`" in block
        assert f"`{verdict['outcome']}`" in block


# --- the anchoring mechanism, which names no kind ---------------------------------------------------


def test_a_finding_kind_this_module_has_never_heard_of_renders_beside_its_cells(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Anchoring is by `keys` alone. No `kind` is named anywhere in the module, so a kind invented
    here renders without a line being changed there."""
    cell = next(c for c in payload["cells"] if c["kind"] == "auc")
    payload["run"]["summary"]["findings"] = [
        {
            "kind": "a_kind_invented_by_a_test",
            "keys": [dict(cell["key"])],
            "statement": "prose the block prefers not to print",
            "computed": {"an_invented_figure": 0.125},
        }
    ]
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert "`a_kind_invented_by_a_test`" in block
    assert "`an_invented_figure` 0.125000" in block

    anchored = [
        line
        for line in block.splitlines()
        if line.startswith("| ") and " [1]" in line and f"`{cell['key']['baseline']}`" in line
    ]
    assert anchored, "the finding did not render beside the cell it names"


def test_a_finding_never_marks_a_measurement_at_a_coordinate_it_shares(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The defect this replaced: a `rate_pinned` finding stamped on seven census counts, 21 times.

    Cells are stored under `(kind, census, nine axes)` and the anchor was keyed on the nine alone,
    so a finding about a false-positive rate marked every count that shares the rate's coordinates.
    A finding carries no `kind`, so a shared coordinate anchors nowhere and says so.
    """
    shared = next(
        cell
        for cell in payload["cells"]
        if cell["kind"] == "rate"
        and any(
            other["kind"] == "count" and other["key"] == cell["key"]
            for other in payload["cells"]
        )
    )
    payload["run"]["summary"]["findings"] = [
        {
            "kind": "a_kind_naming_a_shared_coordinate",
            "keys": [dict(shared["key"])],
            "statement": "",
            "computed": {"figure": 1},
        }
    ]
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert " [1]" not in block
    assert "not anchored" in block


def test_the_committed_block_puts_no_marker_on_a_census_count(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """The published symptom, asserted against the file that produced it.

    Re-run against the folded block, which moved this section behind a `<details>` without moving a
    byte of it. The slice is unchanged and still lands on the whole census table -- the fold's own
    tags are not table rows and the `\\n\\n**` it stops at is now the verdicts under the closing
    tag -- and the count below is what keeps it from quietly becoming a scan over nothing.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    censuses = block[block.index("What the layer did to the text") :]
    censuses = censuses[: censuses.index("\n\n**")]
    rows = [line for line in censuses.splitlines() if line.startswith("| ")]
    assert len(rows) > 2, "the slice no longer covers the census table it is scanning"
    marked = [line for line in rows if "[" in line]
    assert marked == []


def test_every_marker_in_the_block_names_a_finding_the_block_lists(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """The block's own generated sentence, held to account: a bracket a reader cannot resolve is
    the sentence being false.

    **Restated for the collapse, not relaxed by it.** Findings of one kind whose `computed` blocks
    are the same block are now stated in one entry, so a finding's number is no longer always the
    first token of a bullet of its own. The invariant that mattered never was about the shape of a
    bullet: it is that every number a marker carries resolves inside the findings list, and that
    every number the file's findings were given is reachable there. So `listed` is now scanned out
    of the findings list itself -- which is stricter than the old scan, because a number appearing
    only in a table or only in the prose above no longer counts as listed.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    used = {
        int(number)
        for line in block.splitlines()
        if line.startswith("| ")
        for number in re.findall(r"\[(\d+)\]", line)
    }
    catalogue = block[block.index("findings the aggregator raised") :]
    listed = {int(number) for number in re.findall(r"\*\*\[(\d+)\]\*\*", catalogue)}
    assert used <= listed
    assert listed == set(range(1, len(published["run"]["summary"]["findings"]) + 1))


def test_findings_are_numbered_in_the_order_they_are_printed(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """File order ran the printed list [151]-[190], [19]-[21], [22]-[125]: a reader seeing [19]
    beside a figure had nowhere to look.

    Read across the whole findings list rather than off the first token of each bullet, because a
    collapsed entry carries several numbers on one line. The property is the same and it is the
    stronger reading of it: the numbers ascend from 1 in the order a reader meets them, whether
    they meet them one to a line or several. It is what makes `_finding_notes` and
    `_finding_lines` share one grouping instead of deriving the order twice.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    block = _block(target)
    catalogue = block[block.index("findings the aggregator raised") :]
    printed = [int(number) for number in re.findall(r"\*\*\[(\d+)\]\*\*", catalogue)]
    assert printed == sorted(printed)
    assert printed == list(range(1, len(printed) + 1))


def test_the_module_names_no_finding_kind_anywhere(published: dict[str, Any]) -> None:
    """The mechanism is one mechanism, or it is six special cases waiting to be seven.

    Read as string constants rather than as substrings: `resolution` is also an ordinary English
    word, and a substring scan would be asserting that the module's prose avoids it rather than
    that its code never branches on a kind.
    """
    kinds = {finding["kind"] for finding in published["run"]["summary"]["findings"]}
    assert len(kinds) > 1, "the fixture this test rests on is gone"
    literals = {
        node.value
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert kinds & literals == set()


def test_a_finding_naming_two_cells_anchors_at_both(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    pair = [c for c in payload["cells"] if c["kind"] == "auc"][:2]
    payload["run"]["summary"]["findings"] = [
        {
            "kind": "a_kind_naming_two_cells",
            "keys": [dict(pair[0]["key"]), dict(pair[1]["key"])],
            "statement": "",
            "computed": {"figure": 1},
        }
    ]
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)
    assert sum(1 for line in block.splitlines() if line.startswith("| ") and " [1]" in line) >= 1
    assert block.count(" [1]") >= 2


# --- findings that say the same thing, said once -------------------------------------------------


def _two_findings(payload: dict[str, Any], computed: dict[str, Any], other: dict[str, Any]) -> str:
    pair = [c for c in payload["cells"] if c["kind"] == "auc"][:2]
    payload["run"]["summary"]["findings"] = [
        {
            "kind": "a_kind_a_test_invented",
            "keys": [dict(pair[0]["key"])],
            "statement": "",
            "computed": computed,
        },
        {
            "kind": "a_kind_a_test_invented",
            "keys": [dict(pair[1]["key"])],
            "statement": "",
            "computed": other,
        },
    ]
    findings = payload["run"]["summary"]["findings"]
    failures: list[str] = []
    lines = renderer._finding_lines(
        findings, renderer._finding_notes(findings, ()), failures
    )
    assert failures == []
    return "\n".join(lines)


def test_two_findings_of_one_kind_with_one_computed_block_are_stated_once(
    payload: dict[str, Any],
) -> None:
    """A bullet per finding, each repeating one sentence, is a run of lines carrying one fact, in
    the artifact whose claim is that it is readable. The collapse states the shared values once and
    still accounts for every finding it covers, each by its own number -- and here, where nothing
    anchors, by its coordinates too."""
    same = {"n_negative": 41, "one_item_moves_the_rate_by": 0.002}
    rendered = _two_findings(payload, dict(same), dict(same))
    entries = [line for line in rendered.splitlines() if line.startswith("- ")]
    assert len(entries) == 1
    assert entries[0].count("`n_negative` 41") == 1
    assert "**[1]**" in entries[0] and "**[2]**" in entries[0]
    assert "2 findings carrying one `computed`" in entries[0]


def test_two_computed_blocks_that_differ_only_in_key_order_are_one_block(
    payload: dict[str, Any],
) -> None:
    """Key order is not something a reader of the rendered block can see, so it cannot be what
    decides whether one fact is published once or twice.

    `sort_keys=True` in `_grouped_findings` is what makes that true, and it shipped unproven: with
    it deleted the suite stayed green, because every pair of blocks any test compared was built in
    one order. A producer that emitted `n_positive` before `n_negative` on the second of two
    otherwise identical findings would then have published the same sentence twice, each claiming
    to be the one statement of a shared `computed`.
    """
    first = {"n_negative": 41, "n_positive": 97, "one_item_moves_the_rate_by": 0.002}
    reordered = {"one_item_moves_the_rate_by": 0.002, "n_positive": 97, "n_negative": 41}
    assert list(first) != list(reordered) and first == reordered

    rendered = _two_findings(payload, first, reordered)
    entries = [line for line in rendered.splitlines() if line.startswith("- ")]
    assert len(entries) == 1, "the same three values in another order are the same three values"
    assert entries[0].count("`n_negative` 41") == 1
    assert "2 findings carrying one `computed`" in entries[0]
    assert "**[1]**" in entries[0] and "**[2]**" in entries[0]


def test_one_differing_byte_keeps_the_two_findings_apart(payload: dict[str, Any]) -> None:
    """The other half of the same rule, and the reason it is keyed on the block rather than on a
    `kind`: two findings that differ anywhere are two facts and stay two entries."""
    rendered = _two_findings(
        payload,
        {"n_negative": 41, "one_item_moves_the_rate_by": 0.002},
        {"n_negative": 41, "one_item_moves_the_rate_by": 0.0021},
    )
    entries = [line for line in rendered.splitlines() if line.startswith("- ")]
    assert len(entries) == 2
    assert entries[0].startswith("- **[1]**") and entries[1].startswith("- **[2]**")


def test_one_collapsed_finding_is_counted_in_the_singular(payload: dict[str, Any]) -> None:
    """"1 of them repeat another's `computed` exactly and are stated with it" was published.

    Three findings, two of which share a block: exactly one is collapsed into another's entry, so
    the sentence has to read "repeats ... and is stated". The plural-only version of this line
    shipped in the same commit that wrote a singular branch for the fold's `<summary>`.
    """
    pair = [c for c in payload["cells"] if c["kind"] == "auc"][:3]
    same = {"n_negative": 41, "one_item_moves_the_rate_by": 0.002}
    blocks = [dict(same), dict(same), {"n_negative": 42, "one_item_moves_the_rate_by": 0.002}]
    findings = [
        {
            "kind": "a_kind_a_test_invented",
            "keys": [dict(cell["key"])],
            "statement": "",
            "computed": block,
        }
        for cell, block in zip(pair, blocks)
    ]
    failures: list[str] = []
    rendered = "\n".join(
        renderer._finding_lines(findings, renderer._finding_notes(findings, ()), failures)
    )
    assert failures == []
    assert "1 of them repeats another's `computed` exactly and is stated with it." in rendered
    assert "repeat another's" not in rendered


def test_the_committed_findings_collapse_and_no_finding_loses_its_anchor(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Against the file that produced the symptom: one kind whose every finding shares a `computed`.

    The collapse is asserted on the file's own arithmetic rather than on a number typed here, so
    the day the aggregator stops repeating itself this test measures the new file instead of
    failing over the old one.
    """
    findings = published["run"]["summary"]["findings"]
    blocks_of: dict[str, set[str]] = {}
    for finding in findings:
        blocks_of.setdefault(finding["kind"], set()).add(
            json.dumps(finding["computed"], sort_keys=True)
        )
    entries_expected = sum(len(blocks) for blocks in blocks_of.values())
    assert entries_expected < len(findings), "the fixture this test rests on is gone"

    results, target = write(tmp_path, published)
    render_into(results, target)
    catalogue = _block(target)
    catalogue = catalogue[catalogue.index("findings the aggregator raised") :]
    assert len([line for line in catalogue.splitlines() if line.startswith("- ")]) == (
        entries_expected
    )
    numbered = {int(n) for n in re.findall(r"\*\*\[(\d+)\]\*\*", catalogue)}
    assert numbered == set(range(1, len(findings) + 1))


def test_a_collapsed_finding_repeats_its_coordinates_only_when_nothing_anchors_it(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """The first collapse dropped a hundred lines and kept every word: one line, 17,308 characters.

    A finding that anchored is already beside its own cell in a table above, which is the whole
    point of the marker, so inside a collapsed entry its number is the walk back and the tuple is
    redundant. A finding in `Anchors.shared` anchored nowhere and keeps its tuple, because there
    is no row to walk back to. Asserted off the anchors rather than off a `kind`, and both limbs
    are exercised by the committed file.
    """
    loaded = load_results(Path(write(tmp_path, published)[0]))
    anchors = renderer._finding_notes(loaded.findings, loaded.cells)
    anchored = {n for numbers in anchors.at.values() for n in numbers}
    groups = [
        group
        for groups in renderer._grouped_findings(loaded.findings).values()
        for group in groups
        if len(group) > 1
    ]
    all_anchored = [g for g in groups if all(anchors.numbers[i] in anchored for i in g)]
    none_anchored = [g for g in groups if not any(anchors.numbers[i] in anchored for i in g)]
    assert all_anchored and none_anchored, "the fixture this test rests on is gone"

    results, target = write(tmp_path, published)
    render_into(results, target)
    entries = [
        line
        for line in _block(target).splitlines()
        if line.startswith("- **") and "findings carrying one `computed`" in line
    ]

    def entry_for(group: list[int]) -> str:
        first = f"**[{anchors.numbers[group[0]]}]**"
        return next(line for line in entries if f"They are {first}" in line)

    for group in all_anchored:
        covers = entry_for(group).split("They are ", 1)[1]
        assert re.fullmatch(r"(\*\*\[\d+\]\*\* ; )*\*\*\[\d+\]\*\*\.", covers), covers
        assert "=" not in covers, "an anchored finding's tuple is its marker, not a repeat of it"

    for group in none_anchored:
        covers = entry_for(group).split("They are ", 1)[1]
        assert "not anchored:" in covers
        assert any(f"{axis}=" in covers for axis in AXES)


def test_no_line_of_the_committed_block_makes_a_reader_walk_past_a_thousand_characters(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """Line count is not the measure and was the wrong acceptance criterion.

    "no longer occupy 107 lines" is a proxy an implementation can satisfy by joining the lines: it
    did, and published one line of 17,308 characters that carried every word the 107 lines had.
    Reading cost moves with characters, so that is what is bounded, and it is bounded on every
    line rather than on the block's total -- a block whose mean line is short and whose longest is
    a screenful still stops the reader at the screenful.
    """
    results, target = write(tmp_path, published)
    render_into(results, target)
    longest = max(_block(target).splitlines(), key=len)
    assert len(longest) <= 1500, f"{len(longest)} characters on one line: {longest[:200]}..."


# --- rows and columns nothing filled ------------------------------------------------------------------


def _table(lines: list[str], row_axes: int) -> tuple[list[str], list[list[str]]]:
    """The data columns of a built section: its labels, and its rows, both without the row axes.

    The table is found by looking for the first row rather than at a fixed offset: a folded
    section wraps its table in `<details>` and a `<summary>`, so the header sits two lines lower
    there than in an unfolded one, and an index counted from the top would have silently read the
    `<summary>` as a header row.
    """
    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split(" | ")]

    rows = [index for index, line in enumerate(lines) if line.startswith("| ")]
    header, body = rows[0], rows[2:]
    return cells(lines[header])[row_axes:], [cells(lines[index])[row_axes:] for index in body]


def test_no_row_and_no_column_of_the_block_is_entirely_missing(
    published: dict[str, Any], tmp_path: Path
) -> None:
    """A row of em dashes reads as measured-and-lost rather than never-measured.

    Asserted on the data columns, by offset from `len(section.row_axes)`. The test this replaced
    sliced a fixed `[3:]`, which on a four-axis table started on the last *label* column and could
    never see a row of dashes at all.
    """
    results, _ = write(tmp_path, published)
    loaded = load_results(results)
    anchors = renderer._finding_notes(loaded.findings, loaded.cells)

    checked = 0
    for section in renderer._sections(loaded.cells):
        failures: list[str] = []
        lines, _ = renderer._build_section(section, loaded.cells, anchors, failures)
        assert failures == []
        if not lines:
            continue
        checked += 1
        labels, rows = _table(lines, len(section.row_axes))
        assert labels, section.name
        for row in rows:
            assert set(row) != {"--"}, f"{section.name}: a row with nothing measured in it"
        for column in range(len(labels)):
            values = {row[column] for row in rows}
            assert values != {"--"}, f"{section.name}: column {labels[column]} is empty"
    assert checked >= 5, "the fixture this test rests on is gone"


def test_a_row_exists_only_because_a_cell_fills_it(payload: dict[str, Any]) -> None:
    """The reason the property above holds, rather than a filter that claims to make it hold.

    Rows and columns are the coordinates the claimed cells carry, so a row coordinate exists only
    because some cell has it -- and that cell's column coordinate is in the columns, so it fills a
    slot in that row. Two cells sharing neither coordinate is the sharpest case: the grid has holes,
    and still no row and no column is empty. There is no input that produces an empty one, which is
    why the filters that used to sit here could not fire and their tests could not fail.
    """
    failures: list[str] = []
    first = renderer._read_cell(
        next(c for c in payload["cells"] if c["kind"] == "rate"), 0, failures
    )
    second = renderer._read_cell(
        next(
            c
            for c in payload["cells"]
            if c["kind"] == "rate"
            and c["key"]["baseline"] != first.key[AXES.index("baseline")]
            and c["key"]["family"] != first.key[AXES.index("family")]
        ),
        1,
        failures,
    )
    assert failures == [] and first is not None and second is not None

    section = renderer.Section(
        name="a section made by a test",
        lead_in="**two cells sharing neither coordinate.**",
        claim=lambda cell: True,
        row_axes=("baseline",),
        column_axes=("family",),
    )
    lines, placed = renderer._build_section(section, [first, second], _NO_ANCHORS, failures)
    assert failures == []
    assert placed == {first.identity, second.identity}

    labels, rows = _table(lines, len(section.row_axes))
    assert len(labels) == 2 and len(rows) == 2
    assert sum(cell == "--" for row in rows for cell in row) == 2, "the holes are the point"
    for row in rows:
        assert set(row) != {"--"}
    for column in range(len(labels)):
        assert {row[column] for row in rows} != {"--"}


_NO_ANCHORS = renderer.Anchors(at={}, numbers={}, shared={})


# --- the numbers themselves ----------------------------------------------------------------------


def test_a_negative_zero_renders_as_a_measured_zero(payload: dict[str, Any], tmp_path: Path) -> None:
    delta = next(
        cell
        for cell in payload["cells"]
        if cell["kind"] == "delta" and cell["key"]["contrast"] == "canon_on_vs_off"
    )
    delta["value"] = -0.0
    delta["interval"] = {"lo": -0.0, "hi": 0.0, "method": "newcombe-paired-score"}
    payload["run"]["summary"]["findings"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert "-0.00 pp" not in _block(target)


def test_a_nonzero_magnitude_below_the_resolution_says_so(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """`0.00%` over a rate that is not zero is a rendered zero that was never measured."""
    rate = next(cell for cell in payload["cells"] if cell["kind"] == "rate")
    rate["value"] = 1e-9
    rate["interval"] = {"lo": 0.0, "hi": 1e-8, "method": "wilson-score"}
    payload["run"]["summary"]["findings"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert "<0.01%" in _block(target)


def test_a_negative_magnitude_below_the_resolution_is_written_as_a_bound_it_satisfies() -> None:
    """`-<0.0001` reads as "less than minus one ten-thousandth", the opposite of what it means.

    The value is a small negative number, so it is *greater* than that bound. Each form has to be
    literally true of the value it stands for.
    """
    assert renderer._signed(1e-9, 4) == "<0.0001"
    assert renderer._signed(-1e-9, 4) == ">-0.0001"
    assert renderer._signed(-0.0, 4) == "+0.0000"
    assert renderer._signed(-0.5, 4) == "-0.5000"
    assert "-<" not in renderer._signed(-1e-9, 4)


def test_a_pipe_or_a_newline_in_a_value_cannot_split_a_row() -> None:
    """A `|` ends the cell and shifts every column after it; a newline ends the row.

    Chain names, census names and device strings all come from files this module does not control,
    and a table that silently grows a column is a table whose figures are under wrong headings.
    """
    row = renderer._row(["a|b", "c\nd", "plain"])
    assert row.count("|") == 4 + 1  # the four delimiters plus the one escaped pipe
    assert "\n" not in row
    assert row == "| a\\|b | c d | plain |"


def test_an_axis_value_carrying_a_pipe_keeps_every_table_rectangular(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    for cell in payload["cells"]:
        if cell["key"]["dressing_chain"] == "base32":
            cell["key"]["dressing_chain"] = "base32|injected"
        if cell["key"]["contrast"] == "clean_vs_base32":
            cell["key"]["contrast"] = "clean_vs_base32|injected"
    payload["run"]["summary"]["findings"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)
    block = _block(target)

    assert "base32\\|injected" in block
    tables = 0
    width: int | None = None
    for line in block.splitlines():
        if not line.startswith("| "):
            width = None
            continue
        cells = len(re.findall(r"(?<!\\)\|", line))
        if width is None:
            width, tables = cells, tables + 1
        assert cells == width, line
    assert tables >= 5, "the fixture this test rests on is gone"


def test_a_second_kind_in_the_matched_window_population_does_not_collide(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The claim is kind-agnostic, so `kind` has to be one of its column axes.

    Without it a second kind maps to the slot the first holds and the render aborts on data that is
    not wrong -- the sensitivity section already had `kind` in its columns for the same reason.
    """
    companion = copy.deepcopy(
        next(c for c in payload["cells"] if c["key"]["population"] == "single_window")
    )
    companion["kind"] = "rate"
    companion["k"] = 1
    companion["n"] = 2
    companion["value"] = 0.5
    companion["interval"] = {"lo": 0.1, "hi": 0.9, "method": "wilson-score"}
    companion.pop("contrast", None)
    payload["cells"].append(companion)
    payload["run"]["summary"]["findings"] = []
    results, target = write(tmp_path, payload)
    render_into(results, target)  # no slot collision, no unplaced cell
    assert "one window" in _block(target)


def test_a_results_file_with_no_timing_block_still_renders(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """The loader and the renderer agree that `timing` is optional.

    The loader used to demand it and abort with "run.timing is NoneType, not an object" while
    `_what_ran` skipped it when absent -- an abort a reader could not act on, over a field the
    renderer never needed. Which runs are publishable is `results.py`'s decision, not this one's.
    """
    payload["run"].pop("timing")
    results, target = write(tmp_path, payload)
    render_into(results, target)
    assert "What it cost, measured" not in _block(target)


def test_a_timing_block_that_is_present_and_malformed_still_aborts(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """Optional is not unchecked."""
    payload["run"]["timing"] = {"layer_ns": {"overall": {"p95_ns": 1}}}
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        load_results(results)
    assert "carries no p50_ns" in failures_of(caught)


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="descriptor count needs Linux's /proc"
)
def test_a_write_that_cannot_open_its_temp_file_leaks_no_descriptor(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.fdopen` can raise before the file object owns the descriptor, and `with` leaks it then."""
    results, target = write(tmp_path, payload)
    before = len(os.listdir("/proc/self/fd"))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(renderer.os, "fdopen", refuse)
    with pytest.raises(ReportNotRenderable):
        render_into(results, target)

    assert len(os.listdir("/proc/self/fd")) <= before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["README.md", "results.json"]


def test_the_area_scale_is_chosen_by_the_cells_own_interval_method(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    """A delta computed with a structural-components interval is an area, whatever the limb is
    called; a hand-list of limb names would render a new one as a percentage until somebody looked."""
    delta = next(
        cell
        for cell in payload["cells"]
        if cell["kind"] == "delta"
        and cell["interval"]["method"] == "delta-auc-structural-components"
    )
    rendered = renderer._render_delta(load_cell(delta))
    assert "pp" not in rendered and rendered.count(".") == 3

    delta["interval"]["method"] = "newcombe-paired-score"
    assert "pp" in renderer._render_delta(load_cell(delta))


def load_cell(raw: dict[str, Any]) -> renderer.Cell:
    failures: list[str] = []
    cell = renderer._read_cell(raw, 0, failures)
    assert cell is not None, failures
    return cell


def test_every_column_renderer_takes_a_whole_cell_and_never_a_float() -> None:
    """The rule expressed as a signature: there is no way to hand one of these a bare number."""
    import inspect

    for kind, function in renderer._COLUMN_RENDERERS.items():
        parameters = list(inspect.signature(function).parameters.values())
        assert len(parameters) == 1, kind
        assert parameters[0].annotation == "Cell", kind


def test_durations_are_rendered_from_nanoseconds_here_and_the_units_climb() -> None:
    assert renderer._duration(937) == "937 ns"
    assert renderer._duration(1_500) == "1.50 us"
    assert renderer._duration(1_500_000) == "1.50 ms"
    assert renderer._duration(1_500_000_000) == "1.50 s"
    assert renderer._duration(90_000_000_000) == "1.50 min"
    assert renderer._duration(5_400_000_000_000) == "1.50 h"


# --- the command line -------------------------------------------------------------------------------


def test_the_command_line_renders_and_reports_what_it_rendered(
    published: dict[str, Any], tmp_path: Path, repo_root: Path
) -> None:
    results, target = write(tmp_path, published)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbc.report.readme",
            "--results",
            str(results),
            "--readme",
            str(target),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert json.loads(completed.stdout)["cells_rendered"] == len(published["cells"])
    assert RESULTS_START in target.read_text(encoding="utf-8")


def test_the_command_line_exits_36_with_an_empty_stdout_and_no_traceback(
    tmp_path: Path, repo_root: Path
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.report.readme", "--results", os.devnull],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == ReportNotRenderable.exit_code == 36
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


def test_two_processes_render_the_same_bytes(
    published: dict[str, Any], tmp_path: Path, repo_root: Path
) -> None:
    """Byte-identical across two processes, so hash randomization cannot reorder a table."""
    outputs = []
    for name in ("one", "two"):
        directory = tmp_path / name
        results, target = write(directory, published)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbc.report.readme",
                "--results",
                str(results),
                "--readme",
                str(target),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env={**os.environ, "PYTHONHASHSEED": "0" if name == "one" else "12345"},
        )
        assert completed.returncode == EXIT_OK, completed.stderr
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1]


def test_main_returns_zero_and_writes_nothing_outside_the_markers(
    published: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = a_readme()
    results, target = write(tmp_path, published, readme=before)
    assert main(["--results", str(results), "--readme", str(target)]) == EXIT_OK

    after = target.read_text(encoding="utf-8")
    assert after[: after.index(RESULTS_START)] == before[: before.index(RESULTS_START)]
    tail = len(RESULTS_END)
    assert after[after.index(RESULTS_END) + tail :] == before[before.index(RESULTS_END) + tail :]
    assert _block(target).strip() != RESULTS_START

    report = json.loads(capsys.readouterr().out)
    assert report["cells_rendered"] == len(published["cells"])
    assert report["readme"] == str(target)


# --- helpers ------------------------------------------------------------------------------------------


def _block(target: Path) -> str:
    text = target.read_text(encoding="utf-8")
    return text[text.index(RESULTS_START) : text.index(RESULTS_END)]


def test_inject_moves_no_byte_outside_the_markers() -> None:
    before = a_readme()
    after = inject(before, "a body\n")
    assert after == before.replace(f"{RESULTS_START}\n", f"{RESULTS_START}\na body\n")


def test_render_refuses_a_body_that_would_carry_a_marker(
    payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A marker inside the block makes the next injection unable to tell where the block ends."""
    monkeypatch.setattr(renderer, "_PREAMBLE", f"<!-- {RESULTS_END} -->")
    results, _ = write(tmp_path, payload)
    with pytest.raises(ReportNotRenderable) as caught:
        render(load_results(results))
    assert RESULTS_END in str(caught.value)


# --- the abstract --------------------------------------------------------------------------------
#
# The abstract answers the page's own question in a sentence a reader can quote, and the discipline
# is the block's: every figure in it is derived from the results file by the renderer, never typed,
# and the renderer either locates its span exactly or refuses to guess.


def an_abstract_readme() -> str:
    return (
        "# a repository\n\nprose above.\n\n"
        f"{ABSTRACT_START}\nstale abstract\n{ABSTRACT_END}\n\n"
        f"{RESULTS_START}\n{RESULTS_END}\n\nprose below.\n"
    )


def test_the_shipped_readme_carries_the_abstract_the_committed_results_render(
    repo_root: Path,
) -> None:
    """The abstract on the page is the one the committed results render, byte for byte.

    The freshness discipline of `test_the_shipped_readme_carries_the_block_the_committed_results_render`,
    applied to the second generated span: an abstract that has drifted from the file is a typed
    number wearing a marker pair.
    """
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    results = load_results(repo_root / DEFAULT_RESULTS)
    start = readme.index(ABSTRACT_START) + len(ABSTRACT_START)
    end = readme.index(ABSTRACT_END)
    assert readme[start:end] == "\n" + render_abstract(results)


def test_a_readme_without_abstract_markers_gets_the_block_and_no_abstract(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    """No markers is no abstract, reported as such — not an abort and not a silent insertion."""
    results, target = write(tmp_path / "plain", payload)
    report = render_into(results, target)
    assert report["abstract_rendered"] is False
    text = target.read_text(encoding="utf-8")
    assert ABSTRACT_START not in text and ABSTRACT_END not in text


def test_a_readme_with_abstract_markers_gets_a_fresh_abstract(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    results, target = write(tmp_path / "with", payload, readme=an_abstract_readme())
    report = render_into(results, target)
    assert report["abstract_rendered"] is True
    text = target.read_text(encoding="utf-8")
    assert "stale abstract" not in text
    start = text.index(ABSTRACT_START) + len(ABSTRACT_START)
    end = text.index(ABSTRACT_END)
    assert text[start:end] == "\n" + render_abstract(load_results(results))


@pytest.mark.parametrize(
    "markers",
    [
        f"{ABSTRACT_START}\n",
        f"{ABSTRACT_END}\n",
        f"{ABSTRACT_END}\n{ABSTRACT_START}\n",
        f"{ABSTRACT_START}\n{ABSTRACT_START}\n{ABSTRACT_END}\n",
    ],
    ids=["lone-start", "lone-end", "inverted", "duplicated-start"],
)
def test_a_malformed_abstract_marker_pair_aborts_rather_than_guessing(
    tmp_path: Path, payload: dict[str, Any], markers: str
) -> None:
    """One marker without the other, inverted, or duplicated: refused, and the README untouched."""
    readme = f"# a repository\n\n{markers}\n{RESULTS_START}\n{RESULTS_END}\n"
    results, target = write(tmp_path / "broken", payload, readme=readme)
    with pytest.raises(ReportNotRenderable) as caught:
        render_into(results, target)
    assert "abstract" in failures_of(caught)
    assert target.read_text(encoding="utf-8") == readme


def test_the_abstract_quotes_rows_the_tables_carry(repo_root: Path) -> None:
    """Every rendered rate in the abstract also appears in the block, so the two cannot disagree.

    The abstract's whole claim to trust is that it quotes cells rather than summarizing them; a
    rate string of its own would be a figure the tables cannot be checked against.
    """
    results = load_results(repo_root / DEFAULT_RESULTS)
    abstract = render_abstract(results)
    block = render(results)
    rates = re.findall(r"\d+\.\d+% \[[^\]]+\] [\d,]*\d/[\d,]*\d", abstract)
    assert len(rates) >= 4, "the abstract quotes fewer rate cells than the shape it promises"
    for rendered in rates:
        assert rendered in block, f"{rendered!r} is in the abstract and in no table"


def test_the_abstract_names_a_bound_chain_that_does_not_return_to_the_clean_row(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    """The equality claim is derived, proven by breaking it: nudge one cell, the chain is named.

    On the committed file the exception list holds the chain past the decode ceiling; this test
    moves one document in `base64`'s canon-on attack row and expects `base64` to join it, so the
    sentence cannot be a constant that happens to match the data.
    """
    for cell in payload["cells"]:
        key = cell.get("key", {})
        if (
            cell.get("kind") == "rate"
            and key.get("baseline") == "protectai-deberta-v3"
            and key.get("dressing_chain") == "base64"
            and key.get("canon_on") is True
            and key.get("family") == "attack"
            and key.get("population") == "all"
        ):
            cell["k"] = cell["k"] - 1
            cell["value"] = cell["k"] / cell["n"]
            break
    else:
        pytest.fail("the committed file no longer holds the cell this test nudges")
    results, _ = write(tmp_path / "nudged", payload)
    abstract = render_abstract(load_results(results))
    assert "7 of 9" in abstract
    assert "exceptions are `base64`," in abstract
