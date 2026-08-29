"""The floor is a gate, and every way of being below it is checked here rather than promised.

The interesting half of this file is the four `os.confstr` failure shapes. Only one of them
can happen on the machine running these tests, so `detect_glibc` takes the `confstr` callable
as a parameter and each shape is exercised directly. Likewise `preflight` takes its
`Observation`, so a machine with an old glibc, the wrong interpreter or an unsupported
architecture is tested without owning one.
"""

from __future__ import annotations

import ast
import json
import platform as stdlib_platform
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from nbc.errors import EXIT_OK, NbcError, exit_code_for
import nbc.platform as platform_module
from nbc.platform import (
    GLIBC_CONFSTR_NAME,
    REQUIREMENTS,
    GlibcDetection,
    Observation,
    UnsupportedPlatform,
    detect_glibc,
    observe,
    preflight,
    with_glibc_floor,
)

SUPPORTED = Observation(
    system="linux",
    machine="x86_64",
    implementation="CPython",
    python_version=(3, 13, 5),
    glibc=GlibcDetection((2, 39), "glibc 2.39", "CS_GNU_LIBC_VERSION returned 'glibc 2.39'"),
)


def _with_glibc(detection: GlibcDetection, *, system: str = "linux") -> Observation:
    return replace(SUPPORTED, system=system, glibc=detection)


def _raises(exception: BaseException):
    def confstr(name: str) -> str | None:
        raise exception

    return confstr


def _returns(value: str | None):
    def confstr(name: str) -> str | None:
        assert name == GLIBC_CONFSTR_NAME
        return value

    return confstr


# --- the declared floor ---------------------------------------------------------------------


def test_requirements_declare_the_floor_the_readme_and_results_file_will_state() -> None:
    assert REQUIREMENTS.glibc.minimum == (2, 28)
    assert REQUIREMENTS.interpreter.implementation == "CPython"
    assert REQUIREMENTS.interpreter.version == (3, 13)
    assert REQUIREMENTS.architecture.allowed == ("x86_64", "aarch64")


def test_every_requirement_entry_carries_the_constraint_it_comes_from() -> None:
    for entry in REQUIREMENTS.entries():
        assert entry.reason.strip(), f"{entry.key} declares no reason"


def test_the_interpreter_entry_names_the_vendored_confusables_revision_not_the_wheel_tags() -> None:
    """The whole point of the entry: 3.13 comes from AD-14, not from what the wheels admit.

    A future maintainer widening the range has to re-vendor the table and re-run. If this
    reason ever became "the wheels admit it", the preflight would approve machines whose unit
    suite then fails — NFR6's own failure mode, committed by the check written to prevent it.
    """
    reason = REQUIREMENTS.interpreter.reason
    assert "confusables" in reason
    assert "15.1.0" in reason
    assert "unidata_version" in reason


def test_the_glibc_and_architecture_reasons_name_the_missing_sdist() -> None:
    assert "manylinux_2_28" in REQUIREMENTS.glibc.reason
    assert "sdist" in REQUIREMENTS.glibc.reason
    assert "sdist" in REQUIREMENTS.architecture.reason


def test_each_requirement_string_is_derived_from_the_value_that_is_compared() -> None:
    """The stated floor cannot drift from the enforced one, because it is not stored twice."""
    assert REQUIREMENTS.glibc.requirement == "glibc >= 2.28"
    assert REQUIREMENTS.interpreter.requirement == "CPython 3.13 exactly"
    assert REQUIREMENTS.architecture.requirement == "machine is x86_64 or aarch64"

    raised = with_glibc_floor((99, 0))
    assert raised.glibc.requirement == "glibc >= 99.0"


# --- the four os.confstr failure shapes -------------------------------------------------------


@pytest.mark.parametrize(
    ("confstr", "shape"),
    [
        (_raises(AttributeError("module 'os' has no attribute 'confstr'")), "AttributeError"),
        (_raises(ValueError("unrecognized configuration name")), "ValueError"),
        (_raises(OSError(22, "Invalid argument")), "OSError"),
        (_returns(None), "None"),
    ],
    ids=["attribute-error", "value-error", "oserror-musl", "returns-none"],
)
def test_each_confstr_failure_shape_is_handled_by_name(confstr, shape: str) -> None:
    """All four, because the shape the docs lead you to write is the wrong one.

    On musl the name IS in `os.confstr_names` and the call raises `OSError`. An implementer
    who writes `if value is None` ships an uncaught crash on Alpine.
    """
    detection = detect_glibc(confstr)
    assert detection.version is None
    assert detection.rendered is None
    assert shape in detection.detail, "the recorded detail must name the shape it came from"
    if shape == "OSError":
        assert "musl" in detection.detail


def test_a_detectable_glibc_is_parsed_from_what_confstr_actually_returns() -> None:
    detection = detect_glibc(_returns("glibc 2.39"))
    assert detection.version == (2, 39)
    assert detection.rendered == "2.39"
    assert detection.raw == "glibc 2.39"


def test_output_that_is_not_a_glibc_version_is_undetectable_rather_than_parsed() -> None:
    """`musl 1.2.5` must not be read as a version and compared against the glibc floor."""
    for raw in ("glibc", "musl 1.2.5", ""):
        detection = detect_glibc(_returns(raw))
        assert detection.version is None, raw
        assert repr(raw) in detection.detail

        # And it reaches the same Linux abort rather than a comparison against a number that
        # was never a glibc version.
        with pytest.raises(UnsupportedPlatform, match="musl"):
            preflight(REQUIREMENTS, _with_glibc(detection))


def test_detection_on_this_machine_agrees_with_os_confstr() -> None:
    """The default path is exercised too, not only the injected ones."""
    detection = observe().glibc
    if sys.platform == "linux":
        assert detection.version is not None, detection.detail
    assert detection.detail


def test_the_observation_reports_this_machine_rather_than_anything_derived() -> None:
    observation = observe()
    assert observation.system == sys.platform
    assert observation.machine == stdlib_platform.machine()
    assert observation.implementation == stdlib_platform.python_implementation()
    assert observation.python_version == sys.version_info[:3]


# --- the three outcomes -----------------------------------------------------------------------


def test_a_supported_machine_passes_and_records_what_it_observed() -> None:
    report = preflight(REQUIREMENTS, SUPPORTED)
    assert report.platform_check == "ok"
    assert report.glibc == "2.39"
    assert report.machine == "x86_64"
    assert report.interpreter == "CPython 3.13.5"


def test_the_floor_is_inclusive_at_its_own_value() -> None:
    at_the_floor = _with_glibc(GlibcDetection((2, 28), "glibc 2.28", "detail"))
    assert preflight(REQUIREMENTS, at_the_floor).platform_check == "ok"

    below = _with_glibc(GlibcDetection((2, 27), "glibc 2.27", "detail"))
    with pytest.raises(UnsupportedPlatform):
        preflight(REQUIREMENTS, below)


def test_below_the_floor_aborts_naming_both_the_observed_and_the_required_value() -> None:
    old = _with_glibc(GlibcDetection((2, 17), "glibc 2.17", "detail"))
    with pytest.raises(UnsupportedPlatform) as abort:
        preflight(REQUIREMENTS, old)
    message = str(abort.value)
    assert "2.17" in message
    assert "glibc >= 2.28" in message


@pytest.mark.parametrize(
    "confstr",
    [
        _raises(AttributeError("no confstr")),
        _raises(ValueError("unrecognized configuration name")),
        _raises(OSError(22, "Invalid argument")),
        _returns(None),
    ],
    ids=["attribute-error", "value-error", "oserror-musl", "returns-none"],
)
def test_linux_with_no_detectable_glibc_aborts_naming_musl(confstr) -> None:
    undetectable = _with_glibc(detect_glibc(confstr), system="linux")
    with pytest.raises(UnsupportedPlatform) as abort:
        preflight(REQUIREMENTS, undetectable)
    message = str(abort.value)
    assert "musl" in message
    assert "sdist" in message, "the abort must say there is no source fallback either"


@pytest.mark.parametrize(
    "confstr",
    [
        _raises(AttributeError("no confstr")),
        _raises(ValueError("unrecognized configuration name")),
        _raises(OSError(22, "Invalid argument")),
        _returns(None),
    ],
    ids=["attribute-error", "value-error", "oserror-musl", "returns-none"],
)
@pytest.mark.parametrize("system", ["darwin", "win32"])
def test_a_platform_without_glibc_is_recorded_as_not_applicable_never_skipped(
    confstr, system: str
) -> None:
    """`sys.platform` is the only thing that can tell musl from not-applicable."""
    elsewhere = _with_glibc(detect_glibc(confstr), system=system)
    report = preflight(REQUIREMENTS, elsewhere)
    assert report.platform_check == "not_applicable"
    assert report.system == system, "the outcome carries the platform that was detected"
    assert report.glibc is None
    assert report.glibc_detail, "not a silent skip: the shape that produced it is recorded"


def test_the_wrong_interpreter_aborts_naming_observed_required_and_the_reason() -> None:
    old_python = replace(SUPPORTED, python_version=(3, 11, 9))
    with pytest.raises(UnsupportedPlatform) as abort:
        preflight(REQUIREMENTS, old_python)
    message = str(abort.value)
    assert "CPython 3.11.9" in message
    assert "CPython 3.13 exactly" in message
    assert "confusables" in message


def test_another_implementation_at_the_right_version_still_aborts() -> None:
    with pytest.raises(UnsupportedPlatform):
        preflight(REQUIREMENTS, replace(SUPPORTED, implementation="PyPy"))


def test_an_unsupported_architecture_aborts_naming_observed_and_the_allowed_set() -> None:
    with pytest.raises(UnsupportedPlatform) as abort:
        preflight(REQUIREMENTS, replace(SUPPORTED, machine="armv7l"))
    message = str(abort.value)
    assert "armv7l" in message
    assert "x86_64" in message and "aarch64" in message


def test_a_machine_wrong_in_two_ways_is_told_both_times() -> None:
    broken = replace(SUPPORTED, machine="armv7l", python_version=(3, 11, 9))
    with pytest.raises(UnsupportedPlatform) as abort:
        preflight(REQUIREMENTS, broken)
    assert len(abort.value.failures) == 2


# --- the abort itself -------------------------------------------------------------------------


def test_the_abort_is_one_of_the_declared_ones_with_its_own_exit_code() -> None:
    abort = UnsupportedPlatform("something")
    assert isinstance(abort, NbcError)
    assert exit_code_for(abort) == UnsupportedPlatform.exit_code
    assert UnsupportedPlatform.exit_code not in (EXIT_OK, 1)


# --- the observed values, on their way to results.json ------------------------------------------


def test_the_run_fields_carry_the_floor_and_the_observed_values() -> None:
    fields = preflight(REQUIREMENTS, SUPPORTED).as_run_fields()

    entries = fields["platform_requirements"]
    assert [entry["key"] for entry in entries] == ["glibc", "interpreter", "architecture"]
    assert all(entry["requirement"] and entry["reason"] for entry in entries)

    observed = fields["platform_observed"]
    assert observed == {
        "platform_check": "ok",
        "system": "linux",
        "machine": "x86_64",
        "interpreter": "CPython 3.13.5",
        "glibc": "2.39",
        "glibc_detail": "CS_GNU_LIBC_VERSION returned 'glibc 2.39'",
    }
    json.dumps(fields)  # must survive the trip into results.json unchanged


# --- step 0 of the sequence ---------------------------------------------------------------------


def test_running_the_preflight_does_not_import_the_inference_runtime() -> None:
    """A floor checked after `import onnxruntime` is a floor the import already crashed through."""
    code = (
        "import sys; from nbc.platform import preflight; preflight(); "
        "print('onnxruntime' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout


# --- the CI gate ----------------------------------------------------------------------------------


def test_a_floor_injected_above_this_machines_own_aborts() -> None:
    """What CI asserts, so the check is a gate rather than a promise.

    On a machine where glibc does not apply the floor is not the thing under test, so the
    assertion is the other outcome: `not_applicable`, explicitly, never a silent pass.
    """
    raised = with_glibc_floor((99, 0))
    if sys.platform == "linux":
        with pytest.raises(UnsupportedPlatform) as abort:
            preflight(raised)
        assert "glibc >= 99.0" in str(abort.value)
    else:
        assert preflight(raised).platform_check == "not_applicable"


def test_the_injected_floor_says_it_was_injected() -> None:
    """A results file written under an injected floor must not read like a real run."""
    assert "injected" in with_glibc_floor((99, 0)).glibc.reason


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nbc.platform", *args], capture_output=True, text=True
    )


def test_the_module_entrypoint_prints_the_run_fields_on_a_supported_machine() -> None:
    completed = _run_module()
    assert completed.returncode == EXIT_OK, completed.stderr
    fields = json.loads(completed.stdout)
    assert set(fields) == {"platform_requirements", "platform_observed"}


def test_the_module_entrypoint_is_the_hook_ci_uses_to_inject_a_floor() -> None:
    completed = _run_module("--require-glibc", "99.0")
    if sys.platform == "linux":
        assert completed.returncode == UnsupportedPlatform.exit_code
        assert "glibc >= 99.0" in completed.stderr
    else:
        assert completed.returncode == EXIT_OK


def test_the_module_entrypoint_rejects_a_floor_it_cannot_parse() -> None:
    """And its usage error is distinguishable from the abort, which is why 2 is unclaimed.

    `argparse` exits 2 on a usage error. If the platform abort also exited 2, a typo in CI's
    `--require-glibc` flag would look exactly like a machine below the floor.
    """
    completed = _run_module("--require-glibc", "nonsense")
    assert completed.returncode == 2
    assert UnsupportedPlatform.exit_code != 2


# --- the module stays ahead of everything it protects ---------------------------------------


def test_platform_imports_nothing_from_nbc_but_the_exception_base() -> None:
    """A preflight that drags the project in behind it is not step 0 of anything.

    Read from the parsed source, so an import inside a function body — `main()` has two — is
    caught the same as a top-level one.
    """
    path = Path(platform_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nbc" or alias.name.startswith("nbc."):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                offenders.append(f"{path.name}:{node.lineno} relative import")
            elif node.module == "nbc" or (node.module or "").startswith("nbc."):
                if node.module != "nbc.errors":
                    offenders.append(f"{path.name}:{node.lineno} from {node.module} import ...")

    assert not offenders, "platform.py may import only nbc.errors: " + "; ".join(offenders)


def test_importing_platform_pulls_in_no_other_nbc_module() -> None:
    """The same rule observed at runtime, in a fresh interpreter."""
    code = (
        "import sys, nbc.platform; "
        "print(sorted(m for m in sys.modules if m == 'nbc' or m.startswith('nbc.')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "['nbc', 'nbc.errors', 'nbc.platform']", completed.stdout


def test_the_declared_interpreter_is_the_one_the_project_is_pinned_to(repo_root: Path) -> None:
    """The floor is stated in three files and must be one fact, not three.

    `pyproject.toml` and `.python-version` decide what gets installed; `REQUIREMENTS` decides
    what the preflight approves and what `results.json` publishes. A machine the preflight
    approves and `uv` refuses — or the reverse — is the drift this ties shut.
    """
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    major, minor = REQUIREMENTS.interpreter.version

    assert pyproject["project"]["requires-python"] == f"=={major}.{minor}.*"
    assert (repo_root / ".python-version").read_text(encoding="utf-8").strip() == f"{major}.{minor}"
