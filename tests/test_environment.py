"""The resolved environment is CPU-only and pinned, and that is checked rather than promised.

The CPU-only premise is the reason a stranger can reproduce this table on a laptop. Asserting
it in prose is worthless: an inference session built without an explicit provider list picks
up an accelerator when one exists, and the resulting numbers are neither CPU numbers nor
reproducible. The cheapest place to bind the premise is the dependency resolution — with no
accelerator runtime resolved, there is none to accidentally acquire.
"""

from __future__ import annotations

import re
import sys
import tomllib
import unicodedata
from importlib.metadata import distributions
from pathlib import Path

import pytest

from nbc.canon.confusables_table import discover_revision

# Matched against PyPI-normalized names (lowercased, runs of `-_.` collapsed to `-`).
FORBIDDEN_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^torch", "torch and its companions: the project has no accelerator runtime"),
    (r"^transformers", "transformers: every baseline ships an in-repository ONNX graph"),
    (r"^nvidia-", "an NVIDIA runtime component"),
    (r"(^|-)cuda(-|$)", "a CUDA component"),
    (r"(^|-)cudnn(-|$)", "a cuDNN component"),
    (r"(^|-)nccl(-|$)", "an NCCL component"),
    (r"^tensorrt(-|$)", "a TensorRT component"),
    (r"^onnxruntime-gpu(-|$)", "the GPU build of onnxruntime"),
)

EXPECTED_PINS: dict[str, str] = {
    "onnxruntime": "1.29.0",
    "tokenizers": "0.23.1",
}
EXPECTED_BUILD_PINS: dict[str, str] = {"datasets": "5.0.1"}
EXPECTED_DEV_PINS: dict[str, str] = {"pytest": "9.1.1"}
EXPECTED_UV_VERSION = "0.12.5"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _forbidden(name: str) -> str | None:
    normalized = _normalize(name)
    for pattern, why in FORBIDDEN_NAME_PATTERNS:
        if re.search(pattern, normalized):
            return why
    return None


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pyproject(repo_root: Path) -> dict:
    return _read_toml(repo_root / "pyproject.toml")


@pytest.fixture(scope="module")
def lockfile(repo_root: Path) -> dict:
    path = repo_root / "uv.lock"
    assert path.is_file(), "uv.lock must be committed: it is part of the pin"
    return _read_toml(path)


def test_no_accelerator_distribution_is_installed() -> None:
    offenders = [
        f"{dist.metadata['Name']} ({why})"
        for dist in distributions()
        if dist.metadata["Name"] and (why := _forbidden(dist.metadata["Name"]))
    ]
    assert not offenders, "the resolved environment must be CPU-only: " + "; ".join(offenders)


def test_no_accelerator_package_appears_anywhere_in_the_lockfile(lockfile: dict) -> None:
    """Covers the build-time group too, which the installed set does not.

    `datasets` is only installed when the corpus is rebuilt, so a torch dependency arriving
    through it would never show up in the installed distributions of a measurement run — and
    would still be a GPU path in a repository that advertises none.
    """
    offenders = [
        f"{package.get('name')} ({why})"
        for package in lockfile.get("package", [])
        if (why := _forbidden(str(package.get("name", ""))))
    ]
    assert not offenders, "uv.lock must resolve no accelerator runtime: " + "; ".join(offenders)


def test_every_declared_dependency_is_pinned_to_an_exact_version(pyproject: dict) -> None:
    """No floating constraint anywhere, including the build backend.

    A range would let two clones a month apart install different code behind the same
    committed numbers.
    """
    project = pyproject["project"]
    declared: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    for group in pyproject.get("dependency-groups", {}).values():
        declared.extend(group)
    declared.extend(pyproject["build-system"]["requires"])

    unpinned = [spec for spec in declared if not re.search(r"==\s*[^,\s]+$", spec.strip())]
    assert not unpinned, "every dependency must be pinned with `==`: " + "; ".join(unpinned)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("runtime", EXPECTED_PINS),
        ("build", EXPECTED_BUILD_PINS),
        ("dev", EXPECTED_DEV_PINS),
    ],
)
def test_the_declared_stack_is_the_pinned_stack(
    pyproject: dict, section: str, expected: dict[str, str]
) -> None:
    if section == "runtime":
        specs = pyproject["project"]["dependencies"]
    elif section == "build":
        specs = pyproject["project"]["optional-dependencies"]["build"]
    else:
        specs = pyproject["dependency-groups"]["dev"]

    resolved = dict(spec.split("==", 1) for spec in specs)
    assert resolved == expected


def test_datasets_is_build_time_only(pyproject: dict) -> None:
    """The measurement path must have no external data dependency at all."""
    runtime = [spec.split("==", 1)[0] for spec in pyproject["project"]["dependencies"]]
    assert "datasets" not in runtime
    build = pyproject["project"]["optional-dependencies"]["build"]
    assert any(spec.startswith("datasets==") for spec in build)


def test_the_interpreter_is_pinned_to_cpython_3_13(pyproject: dict, repo_root: Path) -> None:
    """CPython 3.13 exactly — and not because of the wheel tags.

    The onnxruntime wheels would admit 3.11 through 3.14. The vendored Unicode confusables
    table would not: its revision must equal the interpreter's own UCD revision, and that
    moves with the minor version. Publishing the wheels' range would approve a machine whose
    unit suite then fails.
    """
    assert pyproject["project"]["requires-python"] == "==3.13.*"
    assert (repo_root / ".python-version").read_text(encoding="utf-8").strip() == "3.13"
    assert sys.version_info[:2] == (3, 13), f"running on {sys.version_info[:2]}, expected 3.13"


def test_the_lockfile_agrees_with_the_declared_interpreter(lockfile: dict) -> None:
    assert lockfile["requires-python"] == "==3.13.*"


def test_the_resolver_itself_is_pinned(pyproject: dict) -> None:
    """uv is an input to the pin, not a detail of whoever happened to run it.

    A different uv writes a different lockfile format and can resolve differently, so the
    project refuses to be locked or synced by one. Verified against uv 0.6.9, which exits
    with `Required uv version ==0.12.5 does not match the running version 0.6.9`.
    """
    assert pyproject["tool"]["uv"]["required-version"] == f"=={EXPECTED_UV_VERSION}"


def test_the_default_pytest_run_is_the_offline_unit_suite(pyproject: dict) -> None:
    """`-m 'not smoke'` is what makes "the unit suite runs offline" true by default."""
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert "not smoke" in options["addopts"]
    assert any(marker.startswith("smoke:") for marker in options["markers"])


# --- lockfile agreement ---------------------------------------------------------------------
#
# `uv sync --frozen` installs the lockfile without reading `pyproject.toml`, so a dependency
# added to `pyproject.toml` and never re-locked installs *silently* under that flag — verified
# against uv 0.12.5, which exits 0. The flag that refuses is `uv sync --locked`.
#
# Since the documented reproduction command is the frozen one, the agreement is asserted here
# instead, offline, from the two committed files: `uv.lock` records the root package's declared
# requirements, so drift is detectable without re-resolving anything.


def _extra_from_marker(marker: str | None) -> str | None:
    if not marker:
        return None
    match = re.fullmatch(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker.strip())
    return match.group(1) if match else marker


def _declared_in_pyproject(pyproject: dict) -> set[tuple[str, str, str | None]]:
    declared: set[tuple[str, str, str | None]] = set()
    project = pyproject["project"]
    for spec in project.get("dependencies", []):
        name, version = spec.split("==", 1)
        declared.add((_normalize(name), f"=={version}", None))
    for extra, specs in project.get("optional-dependencies", {}).items():
        for spec in specs:
            name, version = spec.split("==", 1)
            declared.add((_normalize(name), f"=={version}", extra))
    return declared


def _recorded_in_lockfile(lockfile: dict, project_name: str) -> set[tuple[str, str, str | None]]:
    roots = [
        package
        for package in lockfile["package"]
        if _normalize(str(package.get("name", ""))) == _normalize(project_name)
    ]
    assert len(roots) == 1, f"expected exactly one root package entry, found {len(roots)}"
    metadata = roots[0].get("metadata", {})
    return {
        (
            _normalize(str(requirement["name"])),
            str(requirement.get("specifier", "")),
            _extra_from_marker(requirement.get("marker")),
        )
        for requirement in metadata.get("requires-dist", [])
    }


def test_the_lockfile_records_the_same_dependencies_as_pyproject(
    pyproject: dict, lockfile: dict
) -> None:
    project_name = pyproject["project"]["name"]
    declared = _declared_in_pyproject(pyproject)
    recorded = _recorded_in_lockfile(lockfile, project_name)

    assert declared == recorded, (
        "uv.lock is out of sync with pyproject.toml — run `uv lock` and commit the result. "
        f"Declared but not locked: {sorted(declared - recorded)}. "
        f"Locked but not declared: {sorted(recorded - declared)}."
    )


def test_the_lockfile_records_the_same_dependency_groups_as_pyproject(
    pyproject: dict, lockfile: dict
) -> None:
    project_name = pyproject["project"]["name"]
    declared = {
        group: {tuple(spec.split("==", 1)) for spec in specs}
        for group, specs in pyproject.get("dependency-groups", {}).items()
    }

    roots = [
        package
        for package in lockfile["package"]
        if _normalize(str(package.get("name", ""))) == _normalize(project_name)
    ]
    recorded_raw = roots[0].get("metadata", {}).get("requires-dev", {})
    recorded = {
        group: {
            (str(req["name"]), str(req.get("specifier", "")).removeprefix("=="))
            for req in reqs
        }
        for group, reqs in recorded_raw.items()
    }

    assert declared == recorded, (
        "uv.lock's dependency groups disagree with pyproject.toml — run `uv lock`. "
        f"Declared: {declared}. Locked: {recorded}."
    )


def test_every_locked_package_comes_from_the_pinned_index(lockfile: dict) -> None:
    """No package may enter the resolution from a path, a git ref, or a second index.

    A git source would make the reproduction depend on a branch that can move under the
    committed numbers; a second index would make the same name resolve to different code.
    """
    offenders = []
    for package in lockfile["package"]:
        source = package.get("source", {})
        if "registry" in source:
            if source["registry"] != "https://pypi.org/simple":
                offenders.append(f"{package.get('name')} from {source['registry']}")
        elif "editable" in source or "virtual" in source:
            continue  # the project itself
        else:
            offenders.append(f"{package.get('name')} from {source}")
    assert not offenders, "; ".join(offenders)


# --- Pass 5: the versions the lockfile actually installs ----------------------------------------
#
# The section above compares `pyproject.toml` against `uv.lock`'s `requires-dist`, which uv
# GENERATES FROM `pyproject.toml` -- so it compares a file to a copy of itself and is blind to
# the resolution. `uv sync --frozen` never reads `pyproject.toml`; the only thing deciding what
# gets installed is the `version` field of each `[[package]]` block, and nothing read it.
#
# Proved before this test existed: changing `onnxruntime`'s `version` from 1.29.0 to 1.28.0 in
# uv.lock, touching nothing else, left the whole suite green.


def _resolved_versions(lockfile: dict) -> dict[str, str]:
    """What `uv sync --frozen` would install, by name."""
    return {
        _normalize(str(package["name"])): str(package["version"])
        for package in lockfile.get("package", [])
        if "version" in package
    }


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("runtime", EXPECTED_PINS),
        ("build", EXPECTED_BUILD_PINS),
        ("dev", EXPECTED_DEV_PINS),
    ],
)
def test_the_lockfile_resolves_the_declared_versions(
    lockfile: dict, group: str, expected: dict[str, str]
) -> None:
    """The resolved version, not the requirement string uv wrote down from our own file."""
    resolved = _resolved_versions(lockfile)

    for name, version in expected.items():
        key = _normalize(name)
        assert key in resolved, f"{group} pin {name} is not resolved by uv.lock at all"
        assert resolved[key] == version, (
            f"uv.lock resolves {name} to {resolved[key]} and the declared {group} pin is "
            f"{version}. `uv sync --frozen` installs the lockfile without reading "
            f"pyproject.toml, so this field is what the published numbers were produced under."
        )


def test_the_lockfile_resolves_no_accelerator_build_of_onnxruntime(lockfile: dict) -> None:
    """The CPU-only premise, checked against the resolution rather than the requirement.

    A GPU build satisfies `onnxruntime==1.29.0` under a different distribution name, and the
    forbidden-name scan above reads names while this reads what would be installed.
    """
    resolved = _resolved_versions(lockfile)
    accelerators = [name for name in resolved if name.startswith("onnxruntime-")]

    assert not accelerators, f"uv.lock resolves accelerator builds: {accelerators}"


def test_the_running_interpreter_is_the_one_the_confusables_data_is_pinned_to() -> None:
    """AD-14 requires the vendored Unicode revision to equal the interpreter's own.

    Asserted here against the live interpreter, not against prose in pyproject.toml: the wheels
    admit 3.11 through 3.14 and AD-14 does not, because UCD moves with the minor version
    (3.11=14.0.0, 3.12=15.0.0, 3.13=15.1.0, 3.14=16.0.0). Publishing the wheel range would hand
    a stranger a preflight that approves their machine and a suite that then fails.

    The vendored revision is read off `canon/data/`'s filename rather than written here as a
    literal. Until Story 2.1 the artifact did not exist and this line said `"15.1.0"`, which is
    evidence recorded beside a value and never compared to it: the literal would have gone on
    passing after a re-vendoring moved the real table.
    """
    assert sys.implementation.name == "cpython"
    assert sys.version_info[:2] == (3, 13)
    assert unicodedata.unidata_version == discover_revision()


# --- Pass 7: the README states the command, and CI runs it --------------------------------------
#
# FR20 puts the exact reproduction command in the README, above the results. It was not there:
# the file contained no `uv sync`, no `uv run` and no `pytest`. A reproduction claim whose command
# lives only in a maintainer's shell history is a claim nobody can act on.


def _readme(repo_root: Path) -> str:
    return (repo_root / "README.md").read_text(encoding="utf-8")


def test_the_readme_names_the_reproduction_command(repo_root: Path) -> None:
    """And it is the frozen sync, which is the one that needs no network beyond the first fetch."""
    assert "uv sync --frozen" in _readme(repo_root)


def test_the_readme_states_the_platform_floor_it_will_be_held_to(repo_root: Path) -> None:
    """A stranger's machine failing for a reason the README never gave them is NFR6's own failure.

    Per decision D-A the floor is Linux-only, which SC3's "clean CPU-only machine" does not say
    on its own, so the README says it.
    """
    readme = _readme(repo_root)
    assert "glibc 2.28" in readme
    assert "CPython 3.13" in readme
    assert "Linux" in readme


def test_the_readme_reproduction_block_precedes_the_results(repo_root: Path) -> None:
    """FR20: above the results, not in an appendix under them."""
    readme = _readme(repo_root)
    command = readme.index("uv sync --frozen")
    for marker in ("<!-- RESULTS:START -->", "## Status"):
        if marker in readme:
            assert command < readme.index(marker)
            return
    pytest.fail("neither the results markers nor the status block was found in README.md")


def test_a_ci_workflow_exists_and_runs_the_targets_nothing_else_runs(repo_root: Path) -> None:
    """The five smoke tests were deselected on every run since the marker was created.

    So were the only two tests naming the production pin resolvers. Until this workflow existed,
    nothing in the repository would ever go red, which is why so many defects survived here with
    a demonstrated regression.
    """
    workflows = sorted((repo_root / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no CI workflow: the verification tiers addressed to CI run nowhere"

    ci = "\n".join(path.read_text(encoding="utf-8") for path in workflows)
    assert "pytest -m smoke" in ci, "the model-touching tests still run nowhere"
    assert "uv sync --locked" in ci, "nothing refuses a lockfile out of sync with pyproject"
    assert "--require-glibc" in ci, "the platform abort is still a code path nothing takes"
    assert "git diff --exit-code" in ci, "nothing stops CI writing the published artifacts"


def test_every_command_the_readme_documents_is_runnable(repo_root: Path) -> None:
    """Prose naming a command nothing verifies is how a reproduction claim rots.

    This caught its own author: an earlier draft of the reproduction block documented
    `uv run nbc all`, which does not exist -- there is no `[project.scripts]` entry and the
    entrypoint arrives with the measurement harness. A README is the one surface a stranger
    actually touches, so what it says can be run has to be runnable.
    """
    readme = _readme(repo_root)
    fenced = re.findall(r"```\n(.*?)```", readme, re.S)
    commands = [
        line.split("#")[0].strip()
        for block in fenced
        for line in block.splitlines()
        if line.strip().startswith("uv ")
    ]
    assert commands, "the README documents no command at all"

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    for command in commands:
        if "-m " in command:
            module = command.split("-m ", 1)[1].split()[0]
            path = repo_root / "src" / Path(module.replace(".", "/") + ".py")
            assert path.exists(), f"README runs `{command}` and {module} does not exist"
            assert "__main__" in path.read_text(encoding="utf-8"), (
                f"README runs `{command}` and {module} has no `__main__` entry point"
            )
        elif command.startswith("uv run ") and "python" not in command:
            entry = command.removeprefix("uv run ").split()[0]
            assert entry == "pytest" or f'{entry} =' in pyproject_text, (
                f"README documents `{command}` and nothing declares the `{entry}` command"
            )


def test_the_readme_does_not_claim_a_command_verifies_more_than_it_does(repo_root: Path) -> None:
    """`python -m nbc.pins` without `--verify` loads the FILE; it asks the world nothing.

    Both the README line and the CI step named it as the verification of every pinned artifact.
    That is the claims-versus-code defect this repository is about, written by the pass that was
    fixing that defect elsewhere.
    """
    readme = _readme(repo_root)
    for line in readme.splitlines():
        if "nbc.pins" in line and ("verif" in line.lower() or "check" in line.lower()):
            assert "--verify" in line, (
                f"this line claims verification from a command that does not verify: {line!r}"
            )
