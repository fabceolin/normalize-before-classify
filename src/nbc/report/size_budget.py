"""NFR5's claim, given a number, and the number given a check.

NFR5 says the canonicalization layer "stays small enough to be read end to end by a reviewer". In a
repository that pins a Unicode revision and a glibc minor, that was the one surviving adjective, and
it gated nothing. The architecture deferred the budget on the honest grounds that the number is only
knowable once the layer exists, with the instruction to settle it before the README's claim is
written. The layer exists. This module is the number and the check.

**Three sides, compared to each other.** The README states the budget, this module declares it, and
the layer is measured. A claim recorded beside a value it is never compared to is the failure this
project keeps finding in itself, so all three are read in one pass: the README's stated numbers must
equal `BUDGET` field for field, and the measurement must fit inside it. Editing the layer past the
budget fails. Editing the README's numbers fails. Editing `BUDGET` alone fails.

**What is measured, stated so a reader can disagree with it.** Two counts per module:

- `physical` — every line in the file, which is what a reviewer scrolls.
- `code` — lines that are not blank, not comment-only, and not part of a string-expression
  statement (a docstring, or one of the attribute docstrings this project uses under its constants).
  That is what a reviewer must reason about.

Both, because either alone is gameable in an obvious direction: a total that counts prose punishes
documentation, and a total that ignores it rewards moving logic into comments.

Only `.py` files are counted, so `canon/data/` is out of the number. That is a decision and not an
oversight: the vendored confusables table is a few hundred kilobytes of generated JSON that nobody
reads line by line, and what a reviewer actually audits about it is the derivation script and the
loader's validation — both of which are code, one of them counted here and the other excluded with
the derivation. Counting the artifact would put a five-figure number in a budget about readability
and make every other line invisible next to it.

**What the layer is, structurally rather than by a list somebody chose.** The runtime layer is every
module under `canon/` that the interpreter loads when it imports `nbc.canon.pipeline`.
`vendor_confusables.py` is the build-time derivation that produces `canon/data/`; it runs in no
measurement pass, sits in front of no classifier, and is excluded. "Defensible" is exactly where a
convenient exclusion hides, so the exclusion is not trusted here: `tests/report/test_size_budget.py`
requires `EXCLUDED_MODULES` to equal `canon/` minus the set of modules a real import actually loads,
computed in a subprocess. Wiring the derivation script into the runtime empties that difference and
fails; adding a module and never wiring it lands it in the difference and fails.

**What the budget does not prove.** It is a ceiling on growth and a number to argue with, not
evidence that the layer reads in one sitting. The README says so where it states the budget.

This module imports the standard library and `nbc.errors`, nothing else, and it imports no part of
`canon/` — it reads the layer as text, so measuring it costs no import of the thing being measured.

    python -m nbc.report.size_budget [--readme README.md] [--layer-root src/nbc/canon]

exits 0 with a JSON report of the measurement, the budget and the headroom, or with
`SizeBudgetViolated`'s exit code.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nbc.errors import NbcError

__all__ = [
    "BUDGET",
    "DEFAULT_LAYER_ROOT",
    "DEFAULT_README",
    "EXCLUDED_MODULES",
    "LayerSize",
    "MARKER_END",
    "MARKER_START",
    "ModuleSize",
    "SizeBudget",
    "SizeBudgetReport",
    "SizeBudgetViolated",
    "budget_fields",
    "count_lines",
    "main",
    "measure_layer",
    "stated_budget",
    "verify_size_budget",
    "verify_size_budget_files",
]


class SizeBudgetViolated(NbcError, exit_code=14):
    """The layer outgrew its declared budget, or the README no longer states that budget.

    One abort for both, because the consequence is one: NFR5's claim and the artifact it is about
    have stopped agreeing, and nothing downstream can tell which of the two moved. The message
    always names which failure it is and by how much, and every failure found is collected before
    raising, so a README that is wrong in three fields is told all three.

    The code is 14: 3 is the platform floor, 4-7 the pins, 8 the label mapping, 9 the inference
    session, 10 the window policy, 11 the caveats section, 12 the vendored confusables table, 13 a
    stage contract. 14 is the first free one.
    """

    def __init__(self, *failures: str) -> None:
        super().__init__(
            "the canonicalization layer and its declared size budget disagree:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
        self.failures: Final[tuple[str, ...]] = tuple(failures)


@dataclass(frozen=True, slots=True)
class SizeBudget:
    """The declared ceiling, in lines, over the runtime modules of `canon/`.

    Every field is a budget number and nothing else is, which is what lets the README parser be
    driven by `dataclasses.fields(SizeBudget)` instead of by a second hand-written list of names
    that could drift from this one.
    """

    total_physical_lines: int
    total_code_lines: int
    module_physical_lines: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field.name} must be an int, got {value!r}")
            if value < 1:
                raise ValueError(f"{field.name} must be at least 1, got {value!r}")


BUDGET: Final[SizeBudget] = SizeBudget(
    total_physical_lines=2000,
    total_code_lines=1000,
    module_physical_lines=550,
)
"""The budget, settled here rather than deferred further.

**How much headroom, stated as a rule rather than as a number that goes stale.** Each ceiling sits
above what the layer measured when it was declared and below twice it, and that relation is what
`tests/report/test_size_budget.py::test_the_declared_headroom_is_real_and_finite` compares against a
live measurement — so the rule is checked, where a measurement transcribed into this docstring would
just be a number nothing reads back. A budget with no headroom fails on the next docstring and gets
raised rather than obeyed; a budget at several times the artifact never binds at all. Both are ways
of not having a budget.

What the headroom is for, concretely: Epic 3's homoglyph dressing draws on `canon/data/`, and Epic 4
reads the trace. Neither should become a negotiation about this number, and neither is a second
decode stage arriving unnoticed.

**Why a per-module ceiling as well as a total.** "Read end to end" degrades much faster with one
1500-line module than with five 300-line ones, and a total alone cannot tell those apart.

**What this does not prove.** A ceiling in the low thousands of lines is a couple of hours of
careful reading, not twenty minutes. This is a ceiling on growth and a number to disagree with, not
evidence of readability — and a reviewer who thinks it is the wrong number can now say so about a
number instead of about an adjective. `python -m nbc.report.size_budget` prints where the layer
actually stands.
"""

DEFAULT_LAYER_ROOT: Final[Path] = Path("src") / "nbc" / "canon"
"""The layer, relative to the working directory, as `nbc.report.caveats` treats the README."""

DEFAULT_README: Final[Path] = Path("README.md")

EXCLUDED_MODULES: Final[frozenset[str]] = frozenset({"vendor_confusables.py"})
"""The `canon/` modules that are not the runtime layer, as posix paths under the layer root.

`vendor_confusables.py` derives the vendored table from upstream and writes `canon/data/`. It is
invoked deliberately, offline of every measurement pass, and no classifier ever sits behind it — the
runtime import graph never reaches it, which is the property `tests/report/test_size_budget.py`
checks this set against rather than taking the name on trust.
"""

MARKER_START: Final[str] = "<!-- SIZE-BUDGET:START -->"
MARKER_END: Final[str] = "<!-- SIZE-BUDGET:END -->"
"""The README block this module parses.

Markers rather than prose matching, for the same reason `caveats.py` uses them: a check that
pattern-matched an English sentence would be reading a sentence, and the sentence is not the claim —
the numbers are.
"""

_STATED: Final[re.Pattern[str]] = re.compile(r"^- `([a-z_]+)`: ([0-9]+)$", re.MULTILINE)
"""One stated budget field per line, inside the markers: ``- `total_code_lines`: 1000``."""


def budget_fields() -> tuple[str, ...]:
    """The declared budget field names, in declaration order. The parser's vocabulary."""
    return tuple(field.name for field in dataclasses.fields(SizeBudget))


@dataclass(frozen=True, slots=True)
class ModuleSize:
    """One measured module: its path under the layer root, and its two counts."""

    name: str
    physical: int
    code: int


@dataclass(frozen=True, slots=True)
class LayerSize:
    """The measured runtime layer, and the names of what was excluded from it.

    The excluded modules are carried by **name only**. Recording their line counts here would put a
    number in the report that nothing compares to anything, and an uncompared number beside a
    compared one is how a reader ends up trusting the wrong half.
    """

    modules: tuple[ModuleSize, ...]
    excluded: tuple[str, ...]

    @property
    def total_physical_lines(self) -> int:
        return sum(module.physical for module in self.modules)

    @property
    def total_code_lines(self) -> int:
        return sum(module.code for module in self.modules)

    @property
    def module_physical_lines(self) -> int:
        """The largest single module, which is the value the per-module ceiling bounds."""
        return max((module.physical for module in self.modules), default=0)

    def measured(self) -> dict[str, int]:
        """The measurement keyed by the budget's own field names, so the two line up by name."""
        return {field: getattr(self, field) for field in budget_fields()}


@dataclass(frozen=True, slots=True)
class SizeBudgetReport:
    """What the check found when it found nothing wrong."""

    layer: LayerSize
    budget: SizeBudget

    def as_json(self) -> dict[str, object]:
        measured = self.layer.measured()
        declared = {field: getattr(self.budget, field) for field in budget_fields()}
        return {
            "budget": declared,
            "measured": measured,
            "headroom": {field: declared[field] - measured[field] for field in budget_fields()},
            "modules": [
                {"name": module.name, "physical": module.physical, "code": module.code}
                for module in self.layer.modules
            ],
            "excluded": list(self.layer.excluded),
        }


def count_lines(source: str, *, filename: str = "<source>") -> tuple[int, int]:
    """`(physical, code)` for one module's text.

    `physical` is every line. `code` is every line that is not blank, not comment-only, and not
    covered by a string-expression statement — the module docstring, a class or function docstring,
    and the attribute docstrings this project writes under its constants.

    The prose span is taken from the syntax tree, not from quote counting: a triple-quoted string
    inside an assignment is code that happens to hold text, and a scan that could not tell those
    apart would be pattern-matching where structure is available.

    Stated limitation: a line holding both a docstring and a statement counts as prose, so `code`
    can undercount by that line. No module in this repository writes one, and the direction of the
    error is the conservative one for a ceiling that would otherwise be gamed.
    """
    lines = source.splitlines()
    prose: set[int] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            prose.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    code = 0
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or number in prose:
            continue
        code += 1
    return len(lines), code


def _count_file(path: Path) -> tuple[int, int]:
    """`count_lines` over one file, with every way reading it can fail turned into the abort.

    `UnicodeDecodeError` is a `ValueError`, not an `OSError` — a module saved in some other
    encoding fails the read without the `OSError` clause ever seeing it — and `SyntaxError` is
    neither. Each of the three would otherwise surface as an unclassified exit 1 from a check whose
    entire purpose is to exit with a code a caller can act on.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as unreadable:
        raise SizeBudgetViolated(
            f"{path} could not be read ({unreadable}); the layer cannot be measured and an "
            f"unmeasured layer must not be reported as fitting"
        ) from unreadable
    try:
        return count_lines(source, filename=str(path))
    except SyntaxError as broken:
        raise SizeBudgetViolated(
            f"{path} does not parse ({broken}); its lines cannot be attributed to code or prose"
        ) from broken


def measure_layer(root: Path = DEFAULT_LAYER_ROOT) -> LayerSize:
    """Measure every Python module under `root`, splitting off `EXCLUDED_MODULES`.

    Aborts rather than returning a vacuous zero when the root holds no modules, or when a declared
    exclusion is not there to exclude. Both are the same mistake wearing two faces — the scan is
    pointed somewhere it should not be, or the file it names was renamed — and a measurement of
    nothing passes every ceiling.
    """
    if not root.is_dir():
        raise SizeBudgetViolated(
            f"the layer root {root} is not a directory; the measurement has nothing to read and "
            f"a measurement of nothing fits inside every budget"
        )

    found: dict[str, Path] = {
        path.relative_to(root).as_posix(): path for path in sorted(root.rglob("*.py"))
    }
    if not found:
        raise SizeBudgetViolated(
            f"the layer root {root} holds no Python module; a scan over an empty file list "
            f"passes vacuously"
        )

    missing = sorted(EXCLUDED_MODULES - found.keys())
    if missing:
        raise SizeBudgetViolated(
            f"{', '.join(missing)} is declared in EXCLUDED_MODULES but is not under {root}; "
            f"an exclusion that excludes nothing has stopped describing the layer"
        )

    modules: list[ModuleSize] = []
    skipped: list[str] = []
    for name, path in found.items():
        if name in EXCLUDED_MODULES:
            skipped.append(name)
            continue
        modules.append(ModuleSize(name, *_count_file(path)))

    if not modules:
        raise SizeBudgetViolated(
            f"every module under {root} is excluded; there is no runtime layer left to measure"
        )
    # `excluded` is what this scan actually skipped, not a copy of the constant it skipped them
    # by. The two agree here only because the missing-exclusion check above already fired.
    return LayerSize(modules=tuple(modules), excluded=tuple(sorted(skipped)))


def stated_budget(readme: str, failures: list[str]) -> dict[str, int] | None:
    """The budget the README states, or `None` with the reasons appended to `failures`."""
    start = readme.find(MARKER_START)
    end = readme.find(MARKER_END)
    if start < 0 or end < 0:
        failures.append(
            f"the README carries no {MARKER_START} / {MARKER_END} block, so NFR5's claim states "
            f"no number and the budget is unstated where a reader would look for it"
        )
        return None
    if end < start:
        failures.append(f"the README's {MARKER_END} comes before its {MARKER_START}")
        return None

    block = readme[start + len(MARKER_START) : end]
    found = _STATED.findall(block)
    stated = {name: int(value) for name, value in found}

    seen = [name for name, _ in found]
    for name in sorted({name for name in seen if seen.count(name) > 1}):
        # Two lines for one field, and a dict keeps the last. A README stating a field twice with
        # two values would be checked against whichever came second and would read as whichever
        # came first.
        failures.append(
            f"the README's size-budget block states `{name}` more than once; a field stated twice "
            f"is a field a reader and this check can read differently"
        )

    declared = set(budget_fields())
    for name in sorted(declared - stated.keys()):
        failures.append(
            f"the README's size-budget block states no `{name}`; every declared budget field is "
            f"stated there or the claim is only partly checkable"
        )
    for name in sorted(stated.keys() - declared):
        failures.append(
            f"the README's size-budget block states `{name}`, which is not a field of SizeBudget; "
            f"a stated number nothing declares is a number nothing checks"
        )
    return stated


def verify_size_budget(
    readme: str,
    layer: LayerSize,
    *,
    budget: SizeBudget = BUDGET,
    effective: SizeBudget | None = None,
) -> SizeBudgetReport:
    """Compare the README, the declared budget and the measured layer, and collect every failure.

    `effective` is the budget the **measurement** is held to, and it defaults to `budget`. The two
    are separable for exactly one caller: `main`'s tighten-only flags, which let CI prove the abort
    can fire without pretending the README states a number it does not. The README is always
    compared to `budget`, the declared one, so no flag can make that half pass by lowering the bar.
    """
    failures: list[str] = []

    stated = stated_budget(readme, failures)
    if stated is not None:
        for name in budget_fields():
            if name not in stated:
                continue
            declared_value = getattr(budget, name)
            if stated[name] != declared_value:
                failures.append(
                    f"the README states `{name}`: {stated[name]}, but SizeBudget declares "
                    f"{declared_value}; the claim and the constant have drifted apart and a "
                    f"reader is being shown the wrong one"
                )

    limits = effective if effective is not None else budget

    covered = {"total_physical_lines", "total_code_lines", "module_physical_lines"}
    unchecked = set(budget_fields()) - covered
    if unchecked:
        # A budget field with no comparison is a number that looks like a gate and is not one.
        # Raised rather than collected, and raised *before* anything is measured: this is a defect
        # in this module, not in the layer, and reporting it as a layer failure would be a lie
        # about which of the two moved.
        raise ValueError(
            f"SizeBudget declares {sorted(unchecked)}, which verify_size_budget never compares "
            f"to anything; every declared budget field is checked or it is not a budget"
        )

    measured = layer.measured()

    for name in ("total_physical_lines", "total_code_lines"):
        limit = getattr(limits, name)
        if measured[name] > limit:
            failures.append(
                f"{name}: the layer measures {measured[name]} against a budget of {limit}, "
                f"over by {measured[name] - limit}"
            )

    # The per-module ceiling is checked module by module rather than against the maximum, so the
    # failure names which file to split instead of only how far over the largest one is.
    over = [module for module in layer.modules if module.physical > limits.module_physical_lines]
    for module in sorted(over, key=lambda module: (-module.physical, module.name)):
        failures.append(
            f"{module.name} is {module.physical} physical lines against a per-module ceiling of "
            f"{limits.module_physical_lines}; one module this size is harder to read end to end "
            f"than the same lines split across several"
        )

    if failures:
        raise SizeBudgetViolated(*failures)

    return SizeBudgetReport(layer=layer, budget=budget)


def verify_size_budget_files(
    readme_path: Path = DEFAULT_README,
    layer_root: Path = DEFAULT_LAYER_ROOT,
    *,
    budget: SizeBudget = BUDGET,
    effective: SizeBudget | None = None,
) -> SizeBudgetReport:
    """`verify_size_budget` over the working tree. An unreadable README is the same abort."""
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as unreadable:
        # `UnicodeDecodeError` is a `ValueError`. A README that is not valid UTF-8 is unreadable
        # in exactly the sense that matters here, and the `OSError` clause alone never sees it.
        raise SizeBudgetViolated(
            f"{readme_path} could not be read ({unreadable}); the budget the README states "
            f"cannot be compared to the one this module declares"
        ) from unreadable
    layer = measure_layer(layer_root)
    return verify_size_budget(readme, layer, budget=budget, effective=effective)


def _flag(name: str) -> str:
    """`total_code_lines` -> `--max-total-code-lines`. Derived, so a field cannot lack a flag."""
    return "--max-" + name.replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.report.size_budget` — NFR5's gate, runnable by CI and by a reader.

    The `--max-*` flags **tighten only**. A flag that could raise a ceiling would pass on every
    layer the real ceiling passes on, which is a gate with an off switch; refusing that is a usage
    error, exit 2, and CI proves both directions the way it proves them for `--require-glibc`.
    """
    import argparse
    import json

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.report.size_budget",
        description=(
            "Measure the canonicalization layer and check it against the budget declared here "
            "and restated in the README. NFR5's claim, made falsifiable."
        ),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        metavar="PATH",
        help="the README that states the budget (default: %(default)s)",
    )
    parser.add_argument(
        "--layer-root",
        type=Path,
        default=DEFAULT_LAYER_ROOT,
        metavar="PATH",
        help="the layer to measure (default: %(default)s)",
    )
    for name in budget_fields():
        parser.add_argument(
            _flag(name),
            type=int,
            default=None,
            metavar="N",
            help=(
                f"hold the measurement to a stricter {name} than the declared "
                f"{getattr(BUDGET, name)}; the value must be strictly lower"
            ),
        )
    args = parser.parse_args(argv)

    tightened: dict[str, int] = {}
    for name in budget_fields():
        override = getattr(args, f"max_{name}")
        declared = getattr(BUDGET, name)
        if override is None:
            tightened[name] = declared
            continue
        if override >= declared:
            parser.error(
                f"{_flag(name)} {override} is not below the declared {name} of {declared}; "
                f"these flags tighten the budget and can never loosen it"
            )
        tightened[name] = override
    effective = SizeBudget(**tightened)

    try:
        report = verify_size_budget_files(
            args.readme, args.layer_root, budget=BUDGET, effective=effective
        )
    except SizeBudgetViolated as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report.as_json(), sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
