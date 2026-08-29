"""NFR5's budget: the number, the check, and the inputs that make the check fail.

Every gate here ships the input that breaks it. A budget that has never been seen to refuse
anything is a number in a file, and the whole story is that NFR5 stops being that.

The one check that cannot be written as a synthetic fixture is the exclusion: `EXCLUDED_MODULES`
names `vendor_confusables.py`, and the reason it is excluded — the runtime never reaches it — is
read from a real interpreter in a subprocess rather than from the same list that makes the claim.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nbc import report as _report  # noqa: F401  (keeps the package imported for monkeypatching)
from nbc.errors import exit_code_for
from nbc.report import size_budget
from nbc.report.size_budget import (
    BUDGET,
    DEFAULT_LAYER_ROOT,
    EXCLUDED_MODULES,
    MARKER_END,
    MARKER_START,
    LayerSize,
    ModuleSize,
    SizeBudget,
    SizeBudgetViolated,
    budget_fields,
    count_lines,
    measure_layer,
    verify_size_budget,
    verify_size_budget_files,
)

# --- the layer as it actually stands -------------------------------------------------------------


@pytest.fixture(scope="module")
def layer_root(repo_root: Path) -> Path:
    return repo_root / DEFAULT_LAYER_ROOT


def test_the_committed_layer_and_the_committed_readme_agree(
    repo_root: Path, layer_root: Path
) -> None:
    report = verify_size_budget_files(repo_root / "README.md", layer_root)
    for name in budget_fields():
        assert report.layer.measured()[name] <= getattr(BUDGET, name)


def test_the_scan_found_the_modules_it_is_supposed_to_measure(layer_root: Path) -> None:
    # A measurement over an empty file list fits inside every budget. This is what makes the
    # assertion above mean something.
    names = {module.name for module in measure_layer(layer_root).modules}
    assert {
        "__init__.py",
        "pipeline.py",
        "edits.py",
        "confusables_table.py",
        "stages/__init__.py",
        "stages/invisible.py",
        "stages/confusables.py",
        "stages/nfkc.py",
        "stages/decode.py",
    } == names


def module_name(relative: str) -> str:
    """`stages/decode.py` -> `nbc.canon.stages.decode`; `__init__.py` -> `nbc.canon`."""
    parts = relative.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["nbc", "canon", *parts])


def test_the_excluded_module_is_exactly_what_a_real_import_never_reaches(layer_root: Path) -> None:
    """The exclusion, checked against a live interpreter instead of the list that claims it.

    Two failing inputs, both real: wire `vendor_confusables` into the runtime and the difference
    goes empty; add a `canon/` module and never wire it in, and it lands in the difference. Either
    way the two sides stop matching, and they come from different places -- one from a directory
    listing, the other from `sys.modules` after an import in a fresh process.
    """
    code = (
        "import nbc.canon.pipeline, sys;"
        "print(' '.join(sorted(m for m in sys.modules if m.split('.')[0] == 'nbc')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(completed.stdout.split())
    assert "nbc.canon.pipeline" in loaded, completed.stdout

    on_disk = {path.relative_to(layer_root).as_posix() for path in layer_root.rglob("*.py")}
    never_reached = {name for name in on_disk if module_name(name) not in loaded}
    assert never_reached == set(EXCLUDED_MODULES), completed.stdout


def test_the_declared_headroom_is_real_and_finite(layer_root: Path) -> None:
    """The budget sits above the layer and not absurdly above it.

    A budget below the layer is a red gate nobody can land; a budget at ten times the layer is a
    number that will never bind. Both are ways of not having a budget, and this is the check that
    the declared one is neither.
    """
    measured = measure_layer(layer_root).measured()
    for name in budget_fields():
        declared = getattr(BUDGET, name)
        assert measured[name] <= declared, name
        assert declared <= 2 * measured[name], (
            f"{name}: the budget is {declared} against a layer of {measured[name]}; a ceiling at "
            f"more than twice the artifact is a ceiling that never binds"
        )


# --- the line counter ----------------------------------------------------------------------------


def test_the_counter_counts_code_and_ignores_prose_and_comments() -> None:
    source = (
        '"""A module docstring.\n'
        '\n'
        'It runs to three lines.\n'
        '"""\n'
        '\n'
        '# a comment\n'
        'X = 1\n'
        '"""An attribute docstring under the constant."""\n'
        '\n'
        'def f():\n'
        '    """One line of docstring."""\n'
        '    return X\n'
    )
    # 12 physical. Code: `X = 1`, `def f():`, `return X` -- three.
    assert count_lines(source) == (12, 3)


def test_a_triple_quoted_string_inside_an_assignment_is_code_not_prose() -> None:
    # The reason the counter reads the syntax tree: quote counting cannot tell these apart.
    assert count_lines('X = """text\nover two lines"""\n') == (2, 2)


def test_the_counter_reports_an_empty_module_as_empty() -> None:
    assert count_lines("") == (0, 0)


# --- the budget's own invariants ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_budget_field_that_is_not_a_positive_int_is_refused(bad: object) -> None:
    with pytest.raises(ValueError):
        SizeBudget(total_physical_lines=bad, total_code_lines=1, module_physical_lines=1)  # type: ignore[arg-type]


def test_every_declared_budget_field_is_compared_to_something() -> None:
    assert set(budget_fields()) == {
        "total_physical_lines",
        "total_code_lines",
        "module_physical_lines",
    }


def test_a_budget_field_the_verifier_never_reads_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The input that makes the coverage guard fail: a fourth budget field nothing compares.

    Without this the guard is a branch nobody has seen taken. `budget_fields()` reads the module's
    `SizeBudget`, so replacing it is exactly what a later story adding a field would do -- and the
    check must refuse to report on a budget it only partly enforces rather than quietly enforcing
    three quarters of it.
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class Extended:
        total_physical_lines: int = 1
        total_code_lines: int = 1
        module_physical_lines: int = 1
        total_prose_lines: int = 1

    readme, layer = declared_readme(), tiny_layer()
    monkeypatch.setattr(size_budget, "SizeBudget", Extended)
    with pytest.raises(ValueError, match="total_prose_lines"):
        verify_size_budget(readme, layer)


# --- the inputs that make the check fail ----------------------------------------------------------


def readme_with(**stated: int) -> str:
    lines = "\n".join(f"- `{name}`: {value}" for name, value in stated.items())
    return f"# probe\n\n{MARKER_START}\n{lines}\n{MARKER_END}\n\ntail\n"


def declared_readme() -> str:
    return readme_with(**{name: getattr(BUDGET, name) for name in budget_fields()})


def layer_of(*modules: tuple[str, int, int]) -> LayerSize:
    return LayerSize(
        modules=tuple(ModuleSize(name, physical, code) for name, physical, code in modules),
        excluded=tuple(sorted(EXCLUDED_MODULES)),
    )


def tiny_layer() -> LayerSize:
    return layer_of(("pipeline.py", 10, 5))


def failures_of(readme: str, layer: LayerSize) -> tuple[str, ...]:
    with pytest.raises(SizeBudgetViolated) as raised:
        verify_size_budget(readme, layer)
    return raised.value.failures


def test_a_layer_inside_the_budget_and_a_readme_that_states_it_pass() -> None:
    report = verify_size_budget(declared_readme(), tiny_layer())
    assert report.budget == BUDGET


def test_a_layer_over_the_total_physical_budget_fails() -> None:
    # The per-file split is arbitrary; only the total matters. It avoids 500 and 400 deliberately:
    # `tests/test_pins.py` refuses any value declared under a `sample_size*` key in `pins.toml`
    # from appearing as a literal anywhere under `src/`, `spikes/` or `tests/`, and it compares
    # values rather than meanings. The benign frame declares 500 items per class, which would
    # otherwise collide here and make the scan report a second home for a pin that is not one.
    layer = layer_of(("a.py", 501, 1), ("b.py", 501, 1), ("c.py", 501, 1), ("d.py", 498, 1))
    (failure,) = failures_of(declared_readme(), layer)
    assert "total_physical_lines" in failure and "2001" in failure and "over by 1" in failure


def test_a_layer_over_the_total_code_budget_fails() -> None:
    # Same reason as above for the split.
    layer = layer_of(("a.py", 10, 401), ("b.py", 10, 600))
    (failure,) = failures_of(declared_readme(), layer)
    assert "total_code_lines" in failure and "1001" in failure


def test_one_module_over_the_per_module_ceiling_fails_while_the_totals_fit() -> None:
    layer = layer_of(("small.py", 10, 5), ("huge.py", 551, 5))
    (failure,) = failures_of(declared_readme(), layer)
    assert failure.startswith("huge.py is 551 physical lines")


def test_the_per_module_failure_names_every_offender_largest_first() -> None:
    layer = layer_of(("a.py", 600, 5), ("b.py", 700, 5), ("ok.py", 10, 5))
    failures = failures_of(declared_readme(), layer)
    assert [failure.split()[0] for failure in failures] == ["b.py", "a.py"]


def test_a_readme_that_states_a_different_number_fails() -> None:
    stated = {name: getattr(BUDGET, name) for name in budget_fields()}
    stated["total_code_lines"] += 1
    (failure,) = failures_of(readme_with(**stated), tiny_layer())
    assert "the README states `total_code_lines`" in failure
    assert str(BUDGET.total_code_lines) in failure


def test_a_readme_missing_a_budget_field_fails() -> None:
    stated = {name: getattr(BUDGET, name) for name in budget_fields()}
    del stated["module_physical_lines"]
    (failure,) = failures_of(readme_with(**stated), tiny_layer())
    assert "states no `module_physical_lines`" in failure


def test_a_readme_stating_a_field_nobody_declares_fails() -> None:
    stated = {name: getattr(BUDGET, name) for name in budget_fields()}
    stated["total_lines_of_prose"] = 40
    (failure,) = failures_of(readme_with(**stated), tiny_layer())
    assert "`total_lines_of_prose`" in failure


def test_a_readme_with_no_budget_block_fails() -> None:
    (failure,) = failures_of("# probe\n\nno block here\n", tiny_layer())
    assert MARKER_START in failure


def test_a_readme_whose_markers_are_inverted_fails() -> None:
    (failure,) = failures_of(f"{MARKER_END}\n{MARKER_START}\n", tiny_layer())
    assert "comes before" in failure


def test_every_failure_is_collected_before_the_abort_is_raised() -> None:
    stated = {name: getattr(BUDGET, name) for name in budget_fields()}
    stated["total_code_lines"] += 1
    stated["total_physical_lines"] += 1
    layer = layer_of(("huge.py", 600, 5))
    failures = failures_of(readme_with(**stated), layer)
    assert len(failures) == 3, failures


def test_the_abort_carries_the_declared_exit_code() -> None:
    with pytest.raises(SizeBudgetViolated) as raised:
        verify_size_budget("# probe\n", tiny_layer())
    assert exit_code_for(raised.value) == 14


# --- measuring a tree that is not the real one ----------------------------------------------------


def write_layer(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def test_a_layer_root_that_is_not_a_directory_aborts(tmp_path: Path) -> None:
    with pytest.raises(SizeBudgetViolated, match="not a directory"):
        measure_layer(tmp_path / "nowhere")


def test_a_layer_root_with_no_modules_aborts(tmp_path: Path) -> None:
    with pytest.raises(SizeBudgetViolated, match="no Python module"):
        measure_layer(tmp_path)


def test_a_layer_root_missing_a_declared_exclusion_aborts(tmp_path: Path) -> None:
    """The input: the derivation script renamed, or the scan pointed at the wrong directory."""
    write_layer(tmp_path, {"pipeline.py": "X = 1\n"})
    with pytest.raises(SizeBudgetViolated, match="EXCLUDED_MODULES"):
        measure_layer(tmp_path)


def test_the_excluded_module_is_not_measured(tmp_path: Path) -> None:
    write_layer(
        tmp_path,
        {"pipeline.py": "X = 1\n", "vendor_confusables.py": "Y = 1\n" * 5000},
    )
    layer = measure_layer(tmp_path)
    assert layer.total_physical_lines == 1
    assert layer.excluded == ("vendor_confusables.py",)


def test_a_tree_holding_nothing_but_the_exclusion_aborts(tmp_path: Path) -> None:
    write_layer(tmp_path, {"vendor_confusables.py": "Y = 1\n"})
    with pytest.raises(SizeBudgetViolated, match="no runtime layer left"):
        measure_layer(tmp_path)


def test_an_unreadable_readme_is_the_same_abort(tmp_path: Path) -> None:
    write_layer(tmp_path, {"pipeline.py": "X = 1\n", "vendor_confusables.py": "Y = 1\n"})
    with pytest.raises(SizeBudgetViolated, match="could not be read"):
        verify_size_budget_files(tmp_path / "absent.md", tmp_path)


# --- the tighten-only override --------------------------------------------------------------------


def test_tightening_binds_the_measurement_and_leaves_the_readme_half_alone() -> None:
    tighter = SizeBudget(
        total_physical_lines=1, total_code_lines=1, module_physical_lines=1
    )
    failures = []
    with pytest.raises(SizeBudgetViolated) as raised:
        verify_size_budget(declared_readme(), tiny_layer(), effective=tighter)
    failures = raised.value.failures
    # The README still states the declared budget and is still compared to it, so nothing here is
    # a README failure -- only the three measurement ones.
    assert not any("the README states" in failure for failure in failures)
    assert len(failures) == 3, failures


def run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nbc.report.size_budget", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_the_cli_reports_the_measurement_and_exits_zero(repo_root: Path) -> None:
    completed = run_cli(repo_root)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["budget"]["total_code_lines"] == BUDGET.total_code_lines
    assert report["headroom"]["total_code_lines"] == (
        BUDGET.total_code_lines - report["measured"]["total_code_lines"]
    )
    assert report["excluded"] == sorted(EXCLUDED_MODULES)


def test_the_cli_aborts_with_fourteen_under_a_tightened_budget(repo_root: Path) -> None:
    completed = run_cli(repo_root, "--max-total-code-lines", "1")
    assert completed.returncode == 14, completed.stdout


def test_the_cli_refuses_to_be_loosened(repo_root: Path) -> None:
    """The gate hook that could disable itself, refused. Exit 2 is argparse's usage error."""
    completed = run_cli(repo_root, "--max-total-code-lines", str(BUDGET.total_code_lines + 1))
    assert completed.returncode == 2
    assert "tighten" in completed.stderr


def test_the_cli_refuses_a_flag_equal_to_the_declared_budget(repo_root: Path) -> None:
    completed = run_cli(repo_root, "--max-total-code-lines", str(BUDGET.total_code_lines))
    assert completed.returncode == 2


@pytest.mark.parametrize("field", list(budget_fields()))
def test_every_budget_field_has_a_tightening_flag(repo_root: Path, field: str) -> None:
    flag = "--max-" + field.replace("_", "-")
    completed = run_cli(repo_root, flag, "1")
    assert completed.returncode == 14, (flag, completed.stdout, completed.stderr)


# --- the sibling exceptions -----------------------------------------------------------------------


def test_a_readme_that_is_not_valid_utf8_is_the_same_abort(tmp_path: Path) -> None:
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`. The clause that catches only

    `OSError` never sees this file, and the check exits 1 instead of 14 -- which is the difference
    between a caller that can act on the result and one that cannot.
    """
    write_layer(tmp_path, {"pipeline.py": "X = 1\n", "vendor_confusables.py": "Y = 1\n"})
    readme = tmp_path / "README.md"
    readme.write_bytes(b"# probe\n\xff\xfe not utf-8\n")
    with pytest.raises(SizeBudgetViolated, match="could not be read"):
        verify_size_budget_files(readme, tmp_path)


def test_a_layer_module_that_is_not_valid_utf8_is_the_same_abort(tmp_path: Path) -> None:
    write_layer(tmp_path, {"vendor_confusables.py": "Y = 1\n"})
    (tmp_path / "pipeline.py").write_bytes(b"X = '\xff\xfe'\n")
    with pytest.raises(SizeBudgetViolated, match="could not be read"):
        measure_layer(tmp_path)


def test_a_layer_module_that_does_not_parse_is_the_same_abort(tmp_path: Path) -> None:
    """`SyntaxError` is neither an `OSError` nor a `ValueError`, so it needs its own clause."""
    write_layer(tmp_path, {"pipeline.py": "def (\n", "vendor_confusables.py": "Y = 1\n"})
    with pytest.raises(SizeBudgetViolated, match="does not parse"):
        measure_layer(tmp_path)


# --- the README block, read the way a reader would misread it -------------------------------------


def test_a_readme_stating_a_field_twice_fails(tmp_path: Path) -> None:
    """A dict keeps the last value, so without this the check and the reader disagree silently."""
    stated = "\n".join(f"- `{name}`: {getattr(BUDGET, name)}" for name in budget_fields())
    readme = (
        f"{MARKER_START}\n{stated}\n"
        f"- `total_code_lines`: {BUDGET.total_code_lines}\n{MARKER_END}\n"
    )
    (failure,) = failures_of(readme, tiny_layer())
    assert "more than once" in failure


def test_the_excluded_names_are_the_ones_the_scan_actually_skipped(tmp_path: Path) -> None:
    """`LayerSize.excluded` is evidence from the scan, not a copy of the constant it scanned by."""
    write_layer(tmp_path, {"pipeline.py": "X = 1\n", "vendor_confusables.py": "Y = 1\n"})
    layer = measure_layer(tmp_path)
    assert layer.excluded == ("vendor_confusables.py",)
    assert [module.name for module in layer.modules] == ["pipeline.py"]
