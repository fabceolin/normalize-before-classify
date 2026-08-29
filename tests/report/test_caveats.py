"""The honesty gate is a gate, proved by breaking the section every way it can break.

The repository's own README is the happy path and is checked as it ships. Every failure is
exercised against a synthetic README built from a fixture, because the interesting cases are the
ones the real file must never be in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nbc.errors import EXIT_OK, NbcError, exit_code_for
from nbc.report.caveats import (
    CAVEATS_HEADING,
    MIN_CAVEAT_BODY_CHARS,
    REQUIRED_LABELS,
    RESERVED_LABEL,
    RESULTS_END,
    RESULTS_START,
    Caveat,
    CaveatsSectionMissing,
    verify_caveats,
    verify_caveats_file,
)

BODY = "x" * (MIN_CAVEAT_BODY_CHARS + 20)


def caveat(label: str) -> str:
    return f"**{label}.** {BODY}"


def section(labels: tuple[str, ...] = REQUIRED_LABELS, reserved: str | None = RESERVED_LABEL) -> str:
    blocks = [caveat(label) for label in labels]
    if reserved is not None:
        blocks.append(f"**{reserved}.** *(reserved for what the run actually revealed)*")
    return CAVEATS_HEADING + "\n\n" + "\n\n".join(blocks) + "\n"


def readme(
    caveats: str | None = None,
    markers: str | None = None,
    trailing: str = "\n## License\n\nMIT.\n",
) -> str:
    if caveats is None:
        caveats = section()
    if markers is None:
        markers = f"{RESULTS_START}\n\n*No run has produced a table yet.*\n\n{RESULTS_END}\n"
    return f"# a repository\n\n## What gets measured\n\n{markers}\n{caveats}{trailing}"


# --- the happy path, including the file this repository actually ships ------------------------


def test_the_repositorys_own_readme_passes(repo_root: Path) -> None:
    report = verify_caveats_file(repo_root / "README.md")
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)
    assert report.section_chars > 0


def test_a_well_formed_section_reports_every_label_in_order() -> None:
    report = verify_caveats(readme())
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)
    assert all(isinstance(item, Caveat) for item in report.caveats)
    assert report.as_run_fields()["caveats_check"] == "ok"
    assert report.as_run_fields()["caveats_required"] == list(REQUIRED_LABELS)


def test_the_check_reads_the_real_readmes_caveats_and_not_the_generated_block(
    repo_root: Path,
) -> None:
    """The section must sit after the generated block, or a run would overwrite it."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert text.index(RESULTS_END) < text.index(CAVEATS_HEADING)
    assert text.count(RESULTS_START) == 1 and text.count(RESULTS_END) == 1


def test_caveat_three_states_the_baseline_count_the_pins_actually_declare(
    repo_root: Path,
) -> None:
    """Caveat 3 states the count as OQ2 decided it, and `pins.toml` is what decided it.

    A baseline swapped in later without the caveat being rewritten would publish a sentence about
    a baseline set the repository no longer has. There is no way to check the caveat's *prose*
    against the architecture workspace — it is not shipped — but the one fact in it that a machine
    can hold to account is the count, and this is that binding.
    """
    from nbc.pins import load_pins

    text = (repo_root / "README.md").read_text(encoding="utf-8")
    section = text[text.index(CAVEATS_HEADING) :]
    caveat_three = section[section.index("**3. ") : section.index("**3b.")]

    assert len(load_pins(repo_root).baselines) == 2
    assert "two baselines" in caveat_three
    assert "third baseline was pinned and then dropped" in caveat_three


def test_the_readme_says_normalization_is_not_a_new_idea_before_the_measurement(
    repo_root: Path,
) -> None:
    """FR19's related-work note: a reader must not conclude the author invented normalization."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    opening = text[: text.index("## What gets measured")]
    assert "not a new idea" in opening
    assert "measurement" in opening


# --- every way the section can fail ------------------------------------------------------------


def test_an_absent_section_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=""))
    assert CAVEATS_HEADING in str(abort.value)


def test_an_empty_section_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=CAVEATS_HEADING + "\n\n   \n\n"))
    assert "empty" in str(abort.value)


@pytest.mark.parametrize("dropped", REQUIRED_LABELS)
def test_dropping_any_required_caveat_aborts_and_names_it(dropped: str) -> None:
    kept = tuple(label for label in REQUIRED_LABELS if label != dropped)
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(labels=kept)))
    assert f"caveat(s) {dropped} are missing" in str(abort.value)


def test_a_thin_caveat_aborts() -> None:
    thin = section().replace(caveat("6"), "**6.** the layer is a surface.")
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=thin))
    message = str(abort.value)
    assert "6 (" in message and "worse than none" in message


def test_caveats_published_out_of_the_prds_order_abort() -> None:
    swapped = list(REQUIRED_LABELS)
    here, there = swapped.index("3c"), swapped.index("3d")
    swapped[here], swapped[there] = swapped[there], swapped[here]
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(labels=tuple(swapped))))
    assert "not the PRD's" in str(abort.value)


def test_a_duplicated_label_aborts() -> None:
    doubled = section() + "\n" + caveat("3c") + "\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=doubled))
    assert "more than once" in str(abort.value)


def test_a_missing_reserved_slot_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(reserved=None)))
    assert f"reserved slot {RESERVED_LABEL} is missing" in str(abort.value)


def test_a_reserved_slot_published_before_the_last_caveat_aborts() -> None:
    early = CAVEATS_HEADING + "\n\n" + f"**{RESERVED_LABEL}.** *(reserved)*\n\n"
    early += "\n\n".join(caveat(label) for label in REQUIRED_LABELS) + "\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=early))
    assert "is the last slot" in str(abort.value)


def test_a_section_inside_the_generated_block_aborts() -> None:
    inside = f"{RESULTS_START}\n\n{section()}\n{RESULTS_END}\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats="", markers=inside))
    assert "generated and replaced on every run" in str(abort.value)


@pytest.mark.parametrize(
    "markers",
    [
        pytest.param("", id="no-markers"),
        pytest.param(f"{RESULTS_START}\n", id="start-only"),
        pytest.param(f"{RESULTS_END}\n", id="end-only"),
        pytest.param(f"{RESULTS_START}\n{RESULTS_START}\n{RESULTS_END}\n", id="doubled-start"),
    ],
)
def test_a_malformed_marker_pair_aborts(markers: str) -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(markers=markers))
    assert "not delimited exactly once" in str(abort.value)


def test_an_inverted_marker_pair_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(markers=f"{RESULTS_END}\n\n{RESULTS_START}\n"))
    assert "inverted" in str(abort.value)


def test_a_duplicated_heading_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section() + "\n" + section()))
    assert "appears 2 times" in str(abort.value)


def test_an_unreadable_readme_aborts_rather_than_crashing(tmp_path: Path) -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats_file(tmp_path / "nope.md")
    assert "could not be read" in str(abort.value)


def test_a_section_that_runs_to_the_end_of_the_file_is_still_read() -> None:
    report = verify_caveats(readme(trailing=""))
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)


# --- the abort itself ---------------------------------------------------------------------------


def test_the_abort_is_one_of_the_projects_declared_aborts() -> None:
    abort = CaveatsSectionMissing("because")
    assert isinstance(abort, NbcError)
    assert exit_code_for(abort) == CaveatsSectionMissing.exit_code == 11
    assert abort.failures == ("because",)


# --- the command line, and what it must not import ----------------------------------------------


def test_the_command_line_reports_the_sections_labels(repo_root: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.report.caveats", "--readme", str(repo_root / "README.md")],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert '"caveats_check": "ok"' in completed.stdout


def test_the_command_line_exits_with_the_aborts_own_code(tmp_path: Path, repo_root: Path) -> None:
    broken = tmp_path / "README.md"
    broken.write_text(readme(caveats=""), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.report.caveats", "--readme", str(broken)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert completed.returncode == CaveatsSectionMissing.exit_code == 11
    assert CAVEATS_HEADING in completed.stderr
    assert completed.stdout == ""


def test_the_check_does_not_import_the_inference_runtime() -> None:
    """AD-16 runs this before any inference; a check that imported the runtime would not be before it."""
    code = "import sys, nbc.report.caveats; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout
