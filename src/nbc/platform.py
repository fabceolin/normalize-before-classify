"""The reproduction floor: declared once here, checked before anything can crash through it.

SC3 promises a stranger can reproduce the published table with one command on a clean CPU
machine. That promise needs a machine it is true of, and the repository has to say which.
`onnxruntime==1.29.0` publishes only `manylinux_2_28` wheels and **no sdist at any version**,
so on an older-glibc or musl machine the reproduction ends in a resolver error about wheel
tags. That is the same failure as an unpinned model with a worse error message, and this
module exists to replace it with a stated requirement.

Two properties make the floor real rather than documentary:

- **It is declared once.** `REQUIREMENTS` is the single source for the floor. Each entry
  carries the constraint it comes from, and each entry's human-readable string is *derived*
  from the value that is actually compared, so the floor a reader is shown and the floor the
  run enforces cannot drift apart. Both go into the `run` block of `results.json` and into the
  README's generated block.
- **It is the intersection of every declared constraint, not the loosest one.** The
  interpreter entry is where that matters: the wheels would admit CPython 3.11 through 3.14
  and the vendored confusables table would not. Publishing the wheels' range would hand a
  stranger a preflight that *approves* their machine and a unit suite that then fails — the
  exact failure this module was written to prevent, committed by the check written to prevent
  it.

`preflight()` is **step 0** of the entrypoint's sequence: before pins are verified and, above
all, before `onnxruntime` is imported. A floor checked after that import is a floor the import
already crashed through. This module therefore imports the standard library and
`nbc.errors` and nothing else, and a test asserts that running it leaves `onnxruntime` out of
`sys.modules`.

glibc is detected with `os.confstr("CS_GNU_LIBC_VERSION")` rather than `platform.libc_ver()`,
which inspects the executable and can report a version the running system does not have.
`os.confstr` has **four** failure shapes and all four are handled by name, because the one the
CPython docs lead you to write is the wrong one: on musl the name *is* present in
`os.confstr_names` and the call raises `OSError` instead of returning `None`. An implementer
who writes `if value is None` and stops there ships an uncaught crash on Alpine — the obscure
failure this module exists to prevent, moved one step earlier.

Three outcomes, all explicit, never a silent skip:

- below the floor → `UnsupportedPlatform`, naming the observed value and the required one;
- Linux with no detectable glibc → `UnsupportedPlatform`, naming musl as unsupported and
  stating that the pinned `onnxruntime` publishes no sdist at any version, so there is not
  even a slow source fallback to offer;
- a platform where glibc does not apply → `platform_check: not_applicable`, carrying the
  platform that was detected.

CI (Story 4.8) proves the abort is a gate rather than a promise by injecting a floor above the
runner's own::

    python -m nbc.platform --require-glibc 99.0    # must exit with UnsupportedPlatform's code

The injection is a **parameter**, never an environment variable: the floor is configuration
and nothing in this project reads configuration from the environment at point of use.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Callable, ClassVar, Final

# `import platform` inside a module named `nbc/platform.py` resolves to the standard library,
# since Python 3 imports are absolute. Aliased anyway, so a reader never has to know that.
import platform as stdlib_platform

from nbc.errors import NbcError

__all__ = [
    "ArchitectureSet",
    "GLIBC_CONFSTR_NAME",
    "GlibcDetection",
    "GlibcFloor",
    "InterpreterPin",
    "Observation",
    "PlatformReport",
    "REQUIREMENTS",
    "Requirements",
    "UnsupportedPlatform",
    "detect_glibc",
    "main",
    "observe",
    "preflight",
    "with_glibc_floor",
]


class UnsupportedPlatform(NbcError, exit_code=3):
    """This machine is outside the declared reproduction floor.

    One abort covers every way of being outside it — old glibc, musl, wrong interpreter, wrong
    architecture — because no automated caller needs to tell those apart, while the message
    always does.

    The code is 3 rather than the first usable code, 2, because `argparse` exits **2** on a
    usage error and this module has a command line. Sharing that code would make a typo in
    CI's `--require-glibc` flag indistinguishable from a machine below the floor, which is the
    exact ambiguity `errors.py` exists to prevent. 2 stays unclaimed for that reason.
    """

    def __init__(self, *failures: str) -> None:
        super().__init__(
            "this machine is outside the declared reproduction floor:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
        self.failures: tuple[str, ...] = failures


# --- the floor ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlibcFloor:
    """A minimum glibc version, on the platforms where glibc is a thing at all."""

    minimum: tuple[int, int]
    reason: str

    key: ClassVar[str] = "glibc"

    @property
    def requirement(self) -> str:
        return f"glibc >= {self.minimum[0]}.{self.minimum[1]}"


@dataclass(frozen=True, slots=True)
class InterpreterPin:
    """An exact interpreter implementation and minor version — not a range."""

    implementation: str
    version: tuple[int, int]
    reason: str

    key: ClassVar[str] = "interpreter"

    @property
    def requirement(self) -> str:
        return f"{self.implementation} {self.version[0]}.{self.version[1]} exactly"


@dataclass(frozen=True, slots=True)
class ArchitectureSet:
    """The machine architectures a wheel exists for."""

    allowed: tuple[str, ...]
    reason: str

    key: ClassVar[str] = "architecture"

    @property
    def requirement(self) -> str:
        return "machine is " + " or ".join(self.allowed)


Entry = GlibcFloor | InterpreterPin | ArchitectureSet


@dataclass(frozen=True, slots=True)
class Requirements:
    """The floor as one value, so a caller can raise it without editing the module.

    Passing a modified copy is how CI injects a floor above its own runner's; it is also how
    the tests exercise a machine they are not running on.
    """

    glibc: GlibcFloor
    interpreter: InterpreterPin
    architecture: ArchitectureSet

    def entries(self) -> tuple[Entry, ...]:
        return (self.glibc, self.interpreter, self.architecture)

    def as_run_fields(self) -> list[dict[str, str]]:
        """The floor as plain data for the `run` block of `results.json`."""
        return [
            {"key": entry.key, "requirement": entry.requirement, "reason": entry.reason}
            for entry in self.entries()
        ]


REQUIREMENTS: Final = Requirements(
    glibc=GlibcFloor(
        minimum=(2, 28),
        reason=(
            "onnxruntime 1.29.0 publishes only manylinux_2_28 wheels on Linux, and no sdist "
            "at any version, so below this floor there is not even a slow source fallback."
        ),
    ),
    interpreter=InterpreterPin(
        implementation="CPython",
        version=(3, 13),
        reason=(
            "AD-14: the vendored UTS-39 confusables table is pinned at a Unicode revision that "
            "must equal the interpreter's own unicodedata.unidata_version, which is 15.1.0 on "
            "CPython 3.13. The onnxruntime wheels would admit CPython 3.11 through 3.14; this "
            "constraint does not, because 3.11 is UCD 14.0.0, 3.12 is 15.0.0 and 3.14 is "
            "16.0.0. Widening the range means re-vendoring the table and re-running, not "
            "editing this line."
        ),
    ),
    architecture=ArchitectureSet(
        allowed=("x86_64", "aarch64"),
        reason=(
            "onnxruntime 1.29.0 publishes manylinux wheels for x86_64 and aarch64 only; every "
            "other machine has no wheel and, since there is no sdist, nothing to build from."
        ),
    ),
)


# --- glibc detection, all four failure shapes -----------------------------------------------

GLIBC_CONFSTR_NAME: Final = "CS_GNU_LIBC_VERSION"

# Accepts what glibc actually returns ("glibc 2.39") and a bare version, and rejects anything
# else — including "musl 1.2", which must reach the undetectable branch rather than be read as
# a glibc version and compared against the floor.
_GLIBC_VERSION: Final = re.compile(r"^\s*(?:glibc\s+)?(\d+)\.(\d+)")

ConfstrFn = Callable[[str], "str | None"]


def _os_confstr(name: str) -> str | None:
    """`os.confstr`, looked up at call time so its absence is one of the four shapes.

    On a platform with no `os.confstr` at all this raises `AttributeError` from inside the
    call, which is exactly how the caller's `except AttributeError` is meant to see it.
    """
    return os.confstr(name)  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True, slots=True)
class GlibcDetection:
    """What `os.confstr` said, and what it meant.

    `version is None` means glibc could not be detected — which is *not* the same as "no
    glibc": on a non-Linux platform it means the question does not apply. Only
    `sys.platform` can tell those apart, so this type deliberately does not try.
    """

    version: tuple[int, int] | None
    raw: str | None
    detail: str

    @property
    def rendered(self) -> str | None:
        return None if self.version is None else f"{self.version[0]}.{self.version[1]}"


def detect_glibc(confstr: ConfstrFn = _os_confstr) -> GlibcDetection:
    """Detect the running system's glibc, naming whichever of the four shapes came back.

    `confstr` is injectable so every shape is exercised by a test on a machine that produces
    only one of them.
    """
    try:
        raw = confstr(GLIBC_CONFSTR_NAME)
    except AttributeError:
        return GlibcDetection(
            None, None, "os.confstr does not exist on this interpreter (AttributeError)"
        )
    except ValueError:
        return GlibcDetection(
            None,
            None,
            f"{GLIBC_CONFSTR_NAME} is not a name this system defines (ValueError)",
        )
    except OSError as exc:
        return GlibcDetection(
            None,
            None,
            f"{GLIBC_CONFSTR_NAME} is a known name here but the query failed "
            f"(OSError: {exc}) — the shape musl produces",
        )

    if raw is None:
        return GlibcDetection(
            None, None, f"{GLIBC_CONFSTR_NAME} is undefined on this system (returned None)"
        )

    match = _GLIBC_VERSION.match(raw)
    if match is None:
        return GlibcDetection(
            None,
            raw,
            f"{GLIBC_CONFSTR_NAME} returned {raw!r}, which is not a glibc version string",
        )

    version = (int(match.group(1)), int(match.group(2)))
    return GlibcDetection(version, raw, f"{GLIBC_CONFSTR_NAME} returned {raw!r}")


# --- the preflight --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the preflight looks at, gathered in one place and never re-read.

    Gathering is separated from judging so the judgement can be tested against machines this
    one is not, without monkeypatching the standard library.
    """

    system: str
    machine: str
    implementation: str
    python_version: tuple[int, int, int]
    glibc: GlibcDetection

    @property
    def interpreter(self) -> str:
        return f"{self.implementation} " + ".".join(str(part) for part in self.python_version)


def observe(confstr: ConfstrFn = _os_confstr) -> Observation:
    """Read this machine. Cheap, side-effect free, and imports nothing new."""
    return Observation(
        system=sys.platform,
        machine=stdlib_platform.machine(),
        implementation=stdlib_platform.python_implementation(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        glibc=detect_glibc(confstr),
    )


@dataclass(frozen=True, slots=True)
class PlatformReport:
    """The preflight's answer, in the shape `results.json` will carry it.

    `platform_check` is about the glibc check specifically: `ok` when a glibc was detected and
    cleared the floor, `not_applicable` when the platform has no glibc to check. There is no
    third value, because the third case aborts.
    """

    platform_check: str
    system: str
    machine: str
    interpreter: str
    glibc: str | None
    glibc_detail: str
    requirements: Requirements

    def as_run_fields(self) -> dict[str, object]:
        """The two blocks AD-33 puts into `run`: what was required, and what was observed."""
        return {
            "platform_requirements": self.requirements.as_run_fields(),
            "platform_observed": {
                "platform_check": self.platform_check,
                "system": self.system,
                "machine": self.machine,
                "interpreter": self.interpreter,
                "glibc": self.glibc,
                "glibc_detail": self.glibc_detail,
            },
        }


def preflight(
    requirements: Requirements = REQUIREMENTS,
    observation: Observation | None = None,
) -> PlatformReport:
    """Check this machine against the floor. Step 0 of the entrypoint's sequence.

    Every failing entry is collected before aborting, so a machine that is wrong in two ways
    is told both times rather than sent back for a second run.
    """
    if observation is None:
        observation = observe()

    failures: list[str] = []

    if observation.system == "linux":
        # `ok` is provisional: either the two branches below record a failure, in which case
        # this value never leaves the function, or glibc cleared the floor and it is the truth.
        platform_check = "ok"
        detected = observation.glibc.version
        if detected is None:
            failures.append(
                f"{observation.system}: no glibc could be detected "
                f"({observation.glibc.detail}), which is what musl (Alpine) looks like. "
                f"musl is unsupported: {requirements.glibc.reason} "
                f"Required: {requirements.glibc.requirement}."
            )
        elif detected < requirements.glibc.minimum:
            failures.append(
                f"glibc {observation.glibc.rendered} is below the required "
                f"{requirements.glibc.requirement}. {requirements.glibc.reason}"
            )
    else:
        # Decision D-A, and it is the branch this module used to get wrong in two directions at
        # once. The floor's whole evidence base is a glibc version and manylinux wheel tags, both
        # Linux facts -- so applying the Linux architecture NAMES on other platforms approved an
        # Intel Mac (darwin/x86_64), for which no macOS wheel exists, while refusing an Apple
        # Silicon Mac (darwin/arm64) and a Windows box (win32/AMD64), for which wheels do exist,
        # with a message wrong twice over. `android/aarch64` passed outright.
        #
        # A published table always comes from the Linux path, so the honest answer is to say so
        # and abort, naming the platform, rather than to approve a machine that cannot install
        # the pinned runtime and let it discover that from a wheel resolver.
        platform_check = "not_applicable"
        failures.append(
            f"this platform is {observation.system!r}, and the reproduction floor is Linux only. "
            f"{requirements.glibc.reason} The published table is produced on the Linux path, so "
            f"a run here would be neither reproducing it nor comparable to it."
        )

    observed_interpreter = (observation.implementation, *observation.python_version[:2])
    required_interpreter = (
        requirements.interpreter.implementation,
        *requirements.interpreter.version,
    )
    if observed_interpreter != required_interpreter:
        failures.append(
            f"{observation.interpreter} is not the required "
            f"{requirements.interpreter.requirement}. {requirements.interpreter.reason}"
        )

    if observation.machine not in requirements.architecture.allowed:
        failures.append(
            f"machine {observation.machine!r} is not one of "
            f"{', '.join(requirements.architecture.allowed)}. "
            f"{requirements.architecture.reason}"
        )

    if failures:
        raise UnsupportedPlatform(*failures)

    return PlatformReport(
        platform_check=platform_check,
        system=observation.system,
        machine=observation.machine,
        interpreter=observation.interpreter,
        glibc=observation.glibc.rendered,
        glibc_detail=observation.glibc.detail,
        requirements=requirements,
    )


def with_glibc_floor(
    minimum: tuple[int, int], requirements: Requirements = REQUIREMENTS
) -> Requirements:
    """`requirements` with a different glibc floor, for CI's gate assertion.

    The reason string says the floor was injected, so a `results.json` written under an
    injected floor cannot be mistaken for one written under the declared floor.
    """
    if minimum <= requirements.glibc.minimum:
        # Raising the floor proves the abort fires. Lowering it proves nothing and DISABLES the
        # check, while the reason string below still says the floor went up -- so a CI typo of
        # `2.0` for `99.0` turned the gate that proves the abort into a permanently green no-op,
        # and published a `run` block that contradicted itself in one JSON object.
        raise ValueError(
            f"the injected glibc floor {minimum[0]}.{minimum[1]} is not above the declared "
            f"{requirements.glibc.requirement}. This hook exists to prove the abort fires, and "
            f"a floor at or below the declared one cannot: it would pass on every machine the "
            f"real floor passes on, silently."
        )
    return replace(
        requirements,
        glibc=replace(
            requirements.glibc,
            minimum=minimum,
            reason=(
                "floor injected by the caller, above the declared "
                f"{REQUIREMENTS.glibc.requirement}, to prove the abort fires."
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.platform` — run the preflight and report, for CI and for a stranger."""
    import argparse  # imported here: this module is on the import path of everything.
    import json

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.platform",
        description="Check this machine against the declared reproduction floor.",
    )
    parser.add_argument(
        "--require-glibc",
        metavar="X.Y",
        help=(
            "override the declared glibc floor. CI injects a floor above the runner's own to "
            "prove the abort fires, which makes this check a gate rather than a promise."
        ),
    )
    args = parser.parse_args(argv)

    requirements = REQUIREMENTS
    if args.require_glibc is not None:
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", args.require_glibc)
        if match is None:
            parser.error(f"--require-glibc expects X.Y, got {args.require_glibc!r}")
        try:
            requirements = with_glibc_floor((int(match.group(1)), int(match.group(2))))
        except ValueError as refusal:
            # A usage error, not a platform abort: the caller asked for something the hook
            # cannot do. `parser.error` exits 2 and prints to stderr, which is what a CI step
            # with a typo in it should see rather than a green run.
            parser.error(str(refusal))

    try:
        report = preflight(requirements)
    except UnsupportedPlatform as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report.as_run_fields(), sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
