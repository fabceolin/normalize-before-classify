"""The table is a pure function of `results/results.json`, and nothing in it is typed by a person.

`results.json` holds every cell, every verdict and every finding a run produced. Until this module
existed none of it reached a reader: the README's `RESULTS` markers were empty and every figure
quoted anywhere in the repository was hand-transcribed prose, which had already gone stale three
times -- a commit message quoting a p95 its own file contradicts, a handoff quoting three latencies
the committed file contradicts, another saying four findings where the file has twenty-five. A
number transcribed by hand is a number that drifts silently, so this module is the only writer of
the generated block and it reads exactly one file.

**It opens `results.json` and the README, and nothing else.** No harness import, no `pins.toml`, no
model: a renderer that could reach the harness could recompute a figure it failed to find, and then
the block would no longer be a function of the file. The import bound is the standard library,
`nbc.errors` and `nbc.report.caveats` -- the latter for `RESULTS_START`, `RESULTS_END`,
`DEFAULT_README` and `_locate_markers`, which are declared there once so the checker that verifies
the honesty section sits outside the block and the injector that replaces the block can never
disagree about where the block is. An AST guard in `tests/report/test_readme.py` asserts it.

**Fractions become percentages and nanoseconds become durations here and nowhere else.** Every
figure this module renders carries its `n` and its interval, or its census denominator: the four
column kinds are rendered by four functions each taking a whole cell, never a float, so "a bare
float is not renderable" is a signature rather than a rule someone remembers. Prose carried in the
file is rendered as prose and is guarded against carrying a `%` of its own, which turns
"percentages only here" into a check.

**Completeness is enforced here rather than asserted in a test against today's file.** Every cell
the file holds is placed in exactly one table, or the render aborts naming the ones no table
claimed: a cell with legal coordinates that nothing renders is as invisible to a reader as one that
was never measured. Two cells at one identity abort rather than the second being dropped, and a
section whose body renders no rows is absent -- its lead-in prose with it, because a lead-in
promising a comparison over an empty table is worse than silence.

    python -m nbc.report.readme [--results results/results.json] [--readme README.md]

exits 0 with a JSON report of what it rendered, or with `ReportNotRenderable`'s exit code and an
empty stdout. Every failure to render is that one abort: an unreadable or non-UTF-8 results file,
non-finite JSON, a malformed sub-structure, an unknown coordinate, an unplaced cell, a README whose
markers cannot be located, and a README that cannot be read or written.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from nbc.errors import NbcError
from nbc.report.caveats import (
    DEFAULT_README,
    RESULTS_END,
    RESULTS_START,
    _locate_markers,
)

__all__ = [
    "AXES",
    "Cell",
    "DEFAULT_RESULTS",
    "HEADLINE_WINDOW_POLICY",
    "QUIET_VERDICT_OUTCOME",
    "Results",
    "ReportNotRenderable",
    "SCHEMA_VERSION",
    "SENSITIVITY_WINDOW_POLICIES",
    "inject",
    "load_results",
    "main",
    "render",
    "render_into",
]


class ReportNotRenderable(NbcError, exit_code=36):
    """The block could not be rendered, and nothing was written.

    One abort for every way this can fail, because every one of them has the same consequence --
    the README either keeps the block it has or gets none, and no number reaches a reader that the
    results file did not produce. The message always names which failure it is and where, and every
    failure found is collected before raising, so a results file that is wrong in three ways is
    told all three.

    This is deliberately wider than the failures a renderer would naturally raise. A results file
    that is not readable UTF-8 raises `UnicodeDecodeError`, which is a `ValueError` and not an
    `OSError`; `json.loads` accepts `NaN` and `Infinity` and would publish `nan%`; a sub-structure
    of the wrong shape raises `KeyError` from an index; and the write raises `OSError`. Each of
    those reaches a terminal as a traceback unless it is converted here, and a traceback is not a
    report.

    The code is 36: 3 is the platform floor, 4-7 the pins, 8 the label mapping, 9 the inference
    session, 10 the window policy, 11 the caveats section, 12 the vendored confusables table, 13 a
    stage contract, 14 the size budget, 15-28 the corpus, 29-35 the harness. 36 is the first free
    one.
    """

    def __init__(self, *failures: str) -> None:
        super().__init__(
            "the results file cannot be rendered into the README:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
        self.failures: Final[tuple[str, ...]] = tuple(failures)


# --- the vocabulary the file is read against ----------------------------------------------------

SCHEMA_VERSION: Final[int] = 1
"""The one `schema_version` this renderer understands. A different one aborts naming both."""

DEFAULT_RESULTS: Final[Path] = Path("results") / "results.json"
"""Where the published file lives, relative to the repository root."""

AXES: Final[tuple[str, ...]] = (
    "baseline",
    "dressing_chain",
    "chain_class",
    "window_policy",
    "canon_on",
    "family",
    "benign_class",
    "contrast",
    "population",
)
"""The nine axes of a cell key, in the order the file writes them. A cell's identity is these
nine plus its `kind` and its `census`: a rate and a window-overflow count share all nine axes and
are different measurements, so nine alone would report one of them as a duplicate of the other."""

HEADLINE_WINDOW_POLICY: Final[str] = "shared"
"""The policy the bound results are computed under, and the only one the headline may render.

Restated here rather than imported: `pins.WINDOW_POLICIES` is where the vocabulary is declared, and
reaching it would drag `pins.toml` and the whole pin loader into a renderer. The cost of restating
it is that a policy added there and not here aborts the render -- which is the safe direction, and
is why an unrecognized policy aborts instead of being dropped."""

QUIET_VERDICT_OUTCOME: Final[str] = "not_triggered"
"""The one verdict outcome that does not have to be met before the first table.

The block used to publish the conditions in one list under every table, with no summary over them,
so the one that *fired* was indistinguishable, from above, from the ones that did not. The
headline is therefore defined by exclusion rather than by a list of interesting outcomes: every
outcome except this one is named above the tables. `not_evaluable` is the reason it is written this
way -- it is the artifact not working, it is not `triggered`, and a headline keyed on `triggered`
alone would have hidden it exactly as thoroughly as the old silence hid the condition that fired.

Restated here rather than imported, for the reason `HEADLINE_WINDOW_POLICY` is: `nbc.schema` is
where the outcome vocabulary is declared and the import bound keeps it out of a renderer. A test
binds the two."""

SENSITIVITY_WINDOW_POLICIES: Final[frozenset[str]] = frozenset()
"""The window policies that are a sensitivity pass rather than the headline, which today is none.

`pins.toml` admits exactly one policy, so no run can yet produce a second. The set is empty rather
than absent because the distinction is what keeps a sensitivity cell out of the bound results
instead of being averaged into them, and an empty set makes the sensitivity section absent rather
than an empty stub."""

_AUC_INTERVAL_METHODS: Final[frozenset[str]] = frozenset(
    {"auc-structural-components", "delta-auc-structural-components"}
)
"""Interval methods whose estimand is an area, not a rate.

The scale a figure is rendered on is decided by the cell's own `interval.method` rather than by a
hand-kept list of which limbs happen to be areas: a new limb computed with a structural-components
interval is an area on the day it lands, and a list of names would have rendered it as a
percentage until somebody noticed."""

_MISSING: Final[str] = "--"
_PERCENT_PLACES: Final[int] = 2
_AREA_PLACES: Final[int] = 4
_PLAIN_PLACES: Final[int] = 6


# --- reading the file ---------------------------------------------------------------------------


class _NonFinite(ValueError):
    """Raised out of `json.loads`'s `parse_constant` hook, which is the only place it can be seen."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def _reject_non_finite(name: str) -> float:
    raise _NonFinite(name)


@dataclass(frozen=True, slots=True)
class Cell:
    """One measured cell, validated. Every optional field is `None` only where its kind allows it.

    The four kinds carry different evidence -- a count carries `k` over `n`, a rate carries both
    plus an interval, a delta carries a value and an interval, an AUC carries a value, an interval
    and the two arm sizes -- and each is rendered by a function taking one of these, so no caller
    can hand a renderer a number that has lost the `n` it was measured over.
    """

    kind: str
    census: str | None
    key: tuple[Any, ...]
    value: float | None
    interval: tuple[float, float, str] | None
    k: int | None
    n: int | None
    n_positive: int | None
    n_negative: int | None

    @property
    def identity(self) -> tuple[Any, ...]:
        return (self.kind, self.census, self.key)

    def coord(self, name: str) -> Any:
        """One coordinate of this cell, by name. `kind` and `census` are coordinates too.

        Sections group and pivot by coordinate name, so the two attributes that are not key axes
        have to be reachable the same way or a section could not put the six censuses in columns.
        """
        if name == "kind":
            return self.kind
        if name == "census":
            return self.census
        return self.key[AXES.index(name)]

    @property
    def method(self) -> str | None:
        return self.interval[2] if self.interval is not None else None


@dataclass(frozen=True, slots=True)
class Results:
    """The whole file, validated: the run block, the cells, the verdicts and the findings."""

    run: Mapping[str, Any]
    cells: tuple[Cell, ...]
    verdicts: tuple[Mapping[str, Any], ...]
    findings: tuple[Mapping[str, Any], ...]


def _axis_values(
    block: Mapping[str, Any], where: str, failures: list[str]
) -> tuple[Any, ...] | None:
    """The nine axis values of a key, each checked to be a scalar and a renderable string.

    **Scalar, because a coordinate is what a cell is stored under.** A list or an object in an axis
    reached `dict.get(cell.identity)` and `dict.get(coordinate)` as an unhashable key and came out
    of the process as `TypeError: unhashable type: 'list'` -- exit 1 with a traceback, which
    falsified the promise that every failure to render is one abort. It is caught here, at the one
    place both a cell key and a finding key pass through, rather than at the two dictionaries that
    happened to raise.
    """
    absent = [axis for axis in AXES if axis not in block]
    if absent:
        failures.append(f"{where} carries no {', '.join(absent)}")
        return None

    values = tuple(block[axis] for axis in AXES)
    wrong = [
        f"{axis}={value!r} ({type(value).__name__})"
        for axis, value in zip(AXES, values)
        if value is not None and not isinstance(value, (str, int, float, bool))
    ]
    if wrong:
        failures.append(
            f"{where} carries non-scalar axis value(s) {', '.join(wrong)}; an axis is a coordinate "
            f"a measurement is stored and looked up under, and one that cannot be a dictionary key "
            f"leaves the process as a traceback rather than as a report"
        )
        return None

    # Axis values become row and column labels, so they are strings reaching a reader and go
    # through the same guard the stored prose does.
    for axis, value in zip(AXES, values):
        if isinstance(value, str):
            _stored_text(value, f"{where}.{axis}", failures)
    return values


def _mapping(value: Any, where: str, failures: list[str]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    failures.append(f"{where} is {type(value).__name__}, not an object")
    return None


def _sequence(value: Any, where: str, failures: list[str]) -> Sequence[Any] | None:
    if isinstance(value, list):
        return value
    failures.append(f"{where} is {type(value).__name__}, not an array")
    return None


def _number(value: Any, where: str, failures: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.append(f"{where} is {type(value).__name__}, not a number")
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        failures.append(f"{where} is {value!r}, which is not finite")
        return None
    return number


def _integer(value: Any, where: str, failures: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        failures.append(f"{where} is {type(value).__name__}, not an integer")
        return None
    return value


def _string(value: Any, where: str, failures: list[str]) -> str | None:
    if isinstance(value, str):
        return value
    failures.append(f"{where} is {type(value).__name__}, not a string")
    return None


def _interval_of(raw: Any, where: str, failures: list[str]) -> tuple[float, float, str] | None:
    """An interval, or `None` with the reasons collected. Three fields, each required.

    An interval missing its `method` is the sharp case: `lo` and `hi` alone render as a pair of
    numbers a reader would take for a confidence interval without being told what computed it, and
    five different methods are in use in one file.
    """
    block = _mapping(raw, where, failures)
    if block is None:
        return None
    absent = [name for name in ("lo", "hi", "method") if name not in block]
    if absent:
        failures.append(f"{where} carries no {', '.join(absent)}")
        return None
    lo = _number(block["lo"], f"{where}.lo", failures)
    hi = _number(block["hi"], f"{where}.hi", failures)
    method = _string(block["method"], f"{where}.method", failures)
    if lo is None or hi is None or method is None:
        return None
    if hi < lo:
        failures.append(f"{where} is inverted: hi {hi!r} is below lo {lo!r}")
        return None
    return (lo, hi, method)


def _statistic(raw: Any, where: str, failures: list[str]) -> tuple[int, int, int] | None:
    """A timing statistic: `p50_ns`, `p95_ns` and `n`, all three or none of it renders."""
    block = _mapping(raw, where, failures)
    if block is None:
        return None
    absent = [name for name in ("p50_ns", "p95_ns", "n") if name not in block]
    if absent:
        failures.append(f"{where} carries no {', '.join(absent)}")
        return None
    p50 = _integer(block["p50_ns"], f"{where}.p50_ns", failures)
    p95 = _integer(block["p95_ns"], f"{where}.p95_ns", failures)
    n = _integer(block["n"], f"{where}.n", failures)
    if p50 is None or p95 is None or n is None:
        return None
    return (p50, p95, n)


_VALUE_RANGES: Final[Mapping[str, tuple[float, float]]] = {
    "rate": (0.0, 1.0),
    "auc": (0.0, 1.0),
    "delta": (-1.0, 1.0),
}
"""What each kind's `value` can be, given what it measures.

A rate is a share of `n` and an area under a curve is a share of the pairs; neither can leave
`[0, 1]`, and a difference of two of them cannot leave `[-1, 1]`. Refused rather than rendered,
because every renderable form of an out-of-range value is a lie a reader cannot see: `-0.5` as a
rate is `-50.00%`, which is not a rate, and there is no honest way to draw one. Interval bounds are
deliberately not checked: a structural-components interval can legitimately reach past the
estimand's range, and refusing that would refuse correct arithmetic."""

_CELL_REQUIREMENTS: Final[Mapping[str, tuple[str, ...]]] = {
    "count": ("census", "k", "n"),
    "rate": ("k", "n", "value", "interval"),
    "delta": ("value", "interval"),
    "auc": ("value", "interval", "n_positive", "n_negative"),
}
"""What each kind must carry to be renderable at all. A kind outside this table aborts."""


def _read_cell(raw: Any, index: int, failures: list[str]) -> Cell | None:
    where = f"cells[{index}]"
    block = _mapping(raw, where, failures)
    if block is None:
        return None

    kind = _string(block.get("kind"), f"{where}.kind", failures)
    if kind is None:
        return None
    if kind not in _CELL_REQUIREMENTS:
        failures.append(
            f"{where} is of kind {kind!r}, which this renderer has no column for; the kinds it "
            f"renders are {', '.join(sorted(_CELL_REQUIREMENTS))}"
        )
        return None

    key_block = _mapping(block.get("key"), f"{where}.key", failures)
    if key_block is None:
        return None
    key = _axis_values(key_block, f"{where}.key", failures)
    if key is None:
        return None

    missing = [name for name in _CELL_REQUIREMENTS[kind] if block.get(name) is None]
    if missing:
        failures.append(f"{where} is a {kind} cell and carries no {', '.join(missing)}")
        return None

    census = _string(block["census"], f"{where}.census", failures) if kind == "count" else None
    if census is not None:
        _stored_text(census, f"{where}.census", failures)
    k = _integer(block["k"], f"{where}.k", failures) if "k" in _CELL_REQUIREMENTS[kind] else None
    n = _integer(block["n"], f"{where}.n", failures) if "n" in _CELL_REQUIREMENTS[kind] else None
    value = (
        _number(block["value"], f"{where}.value", failures)
        if "value" in _CELL_REQUIREMENTS[kind]
        else None
    )
    interval = (
        _interval_of(block["interval"], f"{where}.interval", failures)
        if "interval" in _CELL_REQUIREMENTS[kind]
        else None
    )
    n_positive = (
        _integer(block["n_positive"], f"{where}.n_positive", failures)
        if "n_positive" in _CELL_REQUIREMENTS[kind]
        else None
    )
    n_negative = (
        _integer(block["n_negative"], f"{where}.n_negative", failures)
        if "n_negative" in _CELL_REQUIREMENTS[kind]
        else None
    )

    if value is not None and kind in _VALUE_RANGES:
        low, high = _VALUE_RANGES[kind]
        if not low <= value <= high:
            failures.append(
                f"{where} is a {kind} cell whose value is {value!r}, outside the {low} to {high} "
                f"a {kind} can take; there is no honest way to render it"
            )
            return None

    for name, found in (
        ("census", census),
        ("k", k),
        ("n", n),
        ("value", value),
        ("interval", interval),
        ("n_positive", n_positive),
        ("n_negative", n_negative),
    ):
        if name in _CELL_REQUIREMENTS[kind] and found is None:
            return None

    return Cell(
        kind=kind,
        census=census,
        key=key,
        value=value,
        interval=interval,
        k=k,
        n=n,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def _check_coordinates(cells: Sequence[Cell], failures: list[str]) -> None:
    """The two closed vocabularies, checked against the file's own values rather than a hand list.

    `contrast` and `window_policy` are the axes whose values decide which table a cell belongs to.
    A value outside the vocabulary would otherwise fall through every section's predicate and be
    reported as an unplaced cell, which names the symptom rather than the cause.
    """
    chains = {cell.coord("dressing_chain") for cell in cells} - {None}
    benign_classes = {cell.coord("benign_class") for cell in cells} - {None}
    known = (
        {None, "canon_on_vs_off"}
        | {f"attacks_vs_{name}" for name in benign_classes}
        | {f"clean_vs_{name}" for name in chains}
    )
    policies = {HEADLINE_WINDOW_POLICY} | set(SENSITIVITY_WINDOW_POLICIES)

    for cell in cells:
        contrast = cell.coord("contrast")
        if contrast not in known:
            failures.append(
                f"cell {_key_text(cell.key)} declares contrast {contrast!r}, which is none of "
                f"{', '.join(sorted(str(name) for name in known))}"
            )
        policy = cell.coord("window_policy")
        if policy not in policies:
            failures.append(
                f"cell {_key_text(cell.key)} declares window_policy {policy!r}, which is neither "
                f"the headline policy {HEADLINE_WINDOW_POLICY!r} nor a declared sensitivity pass "
                f"({', '.join(sorted(SENSITIVITY_WINDOW_POLICIES)) or 'none is declared'})"
            )


def _read_run(raw: Any, failures: list[str]) -> Mapping[str, Any] | None:
    run = _mapping(raw, "run", failures)
    if run is None:
        return None

    if _string(run.get("build_id"), "run.build_id", failures) is None:
        return None

    files = _sequence(run.get("corpus_files"), "run.corpus_files", failures)
    if files is None:
        return None
    for index, entry in enumerate(files):
        where = f"run.corpus_files[{index}]"
        block = _mapping(entry, where, failures)
        if block is None:
            continue
        absent = [name for name in ("name", "rows", "sha256") if name not in block]
        if absent:
            failures.append(f"{where} carries no {', '.join(absent)}")
            continue
        _string(block["name"], f"{where}.name", failures)
        _integer(block["rows"], f"{where}.rows", failures)
        _string(block["sha256"], f"{where}.sha256", failures)

    # Optional, and validated whenever it is there. A run with no timing pass has no `timing`
    # block, and refusing to render its cells over that would be this module deciding which runs
    # are publishable -- which is the harness's decision, taken in `results.py`. `_what_ran` treats
    # the block the same way, and the two agreeing is the point: the loader demanding a field the
    # renderer then skips is an abort a reader can never act on.
    timing = _mapping(run["timing"], "run.timing", failures) if "timing" in run else None
    if timing is not None:
        layer = _mapping(timing.get("layer_ns"), "run.timing.layer_ns", failures)
        if layer is not None:
            _statistic(layer.get("overall"), "run.timing.layer_ns.overall", failures)
            by_class = _mapping(layer.get("by_class"), "run.timing.layer_ns.by_class", failures)
            for name in sorted(by_class or {}):
                _statistic(by_class[name], f"run.timing.layer_ns.by_class.{name}", failures)
        inference = _mapping(timing.get("inference_ns"), "run.timing.inference_ns", failures)
        if inference is not None:
            by_baseline = _mapping(
                inference.get("by_baseline"), "run.timing.inference_ns.by_baseline", failures
            )
            for name in sorted(by_baseline or {}):
                _statistic(by_baseline[name], f"run.timing.inference_ns.by_baseline.{name}", failures)

    return run


def load_results(path: Path) -> Results:
    """Read and validate `path`. Aborts with every failure collected, and never crashes.

    Three failure modes that are not `OSError` and would otherwise reach a terminal as a traceback:
    a file that is not readable UTF-8 (`UnicodeDecodeError`, a `ValueError`), JSON carrying `NaN`
    or `Infinity` (which `json.loads` accepts by default and which would publish as `nan%`), and a
    sub-structure of the wrong shape reached by indexing a required key.
    """
    failures: list[str] = []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as unreadable:
        raise ReportNotRenderable(
            f"{path} could not be read ({unreadable.strerror or unreadable}); there is no results "
            f"file to render and the README keeps the block it has"
        ) from unreadable
    except ValueError as undecodable:
        raise ReportNotRenderable(
            f"{path} is not readable UTF-8 ({undecodable}); a results file this renderer cannot "
            f"decode is one it cannot publish a number out of"
        ) from undecodable

    try:
        payload = json.loads(text, parse_constant=_reject_non_finite)
    except _NonFinite as non_finite:
        raise ReportNotRenderable(
            f"{path} carries the JSON constant {non_finite.name}, which `json.loads` accepts and "
            f"which would reach a reader as `nan%` or `inf%`; a figure that is not a number is not "
            f"renderable"
        ) from non_finite
    except json.JSONDecodeError as malformed:
        raise ReportNotRenderable(f"{path} is not valid JSON ({malformed})") from malformed

    top = _mapping(payload, str(path), failures)
    if top is None:
        raise ReportNotRenderable(*failures)

    version = top.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ReportNotRenderable(
            f"{path} declares schema_version {version!r}; this renderer understands "
            f"{SCHEMA_VERSION!r} and refuses to guess what a different envelope means"
        )

    run = _read_run(top.get("run"), failures)

    cells: list[Cell] = []
    raw_cells = _sequence(top.get("cells"), "cells", failures)
    for index, raw in enumerate(raw_cells or ()):
        cell = _read_cell(raw, index, failures)
        if cell is not None:
            cells.append(cell)

    seen: dict[tuple[Any, ...], int] = {}
    for index, cell in enumerate(cells):
        first = seen.get(cell.identity)
        if first is None:
            seen[cell.identity] = index
            continue
        failures.append(
            f"cells[{first}] and cells[{index}] are both {cell.kind} cells at "
            f"{_key_text(cell.key)}; one identity is one measurement, and rendering the pair "
            f"would silently drop whichever came second"
        )
    _check_coordinates(cells, failures)

    verdicts: list[Mapping[str, Any]] = []
    raw_verdicts = _sequence(top.get("verdict"), "verdict", failures)
    for index, raw in enumerate(raw_verdicts or ()):
        where = f"verdict[{index}]"
        block = _mapping(raw, where, failures)
        if block is None:
            continue
        absent = [name for name in ("condition", "outcome", "reason", "computed") if name not in block]
        if absent:
            failures.append(f"{where} carries no {', '.join(absent)}")
            continue
        _mapping(block["computed"], f"{where}.computed", failures)
        verdicts.append(block)

    findings: list[Mapping[str, Any]] = []
    summary = _mapping((run or {}).get("summary"), "run.summary", failures) if run else None
    raw_findings = _sequence((summary or {}).get("findings"), "run.summary.findings", failures)
    for index, raw in enumerate(raw_findings or ()):
        where = f"run.summary.findings[{index}]"
        block = _mapping(raw, where, failures)
        if block is None:
            continue
        absent = [name for name in ("kind", "keys", "computed") if name not in block]
        if absent:
            failures.append(f"{where} carries no {', '.join(absent)}")
            continue
        keys = _sequence(block["keys"], f"{where}.keys", failures)
        usable = True
        for position, key in enumerate(keys or ()):
            entry = _mapping(key, f"{where}.keys[{position}]", failures)
            if entry is None or _axis_values(entry, f"{where}.keys[{position}]", failures) is None:
                usable = False
        _stored_text(str(block["kind"]), f"{where}.kind", failures)
        _mapping(block["computed"], f"{where}.computed", failures)
        if usable:
            findings.append(block)

    if failures:
        raise ReportNotRenderable(*failures)

    assert run is not None  # every path that leaves it None appended a failure
    return Results(
        run=run,
        cells=tuple(cells),
        verdicts=tuple(verdicts),
        findings=tuple(findings),
    )


# --- turning numbers into text, which happens here and nowhere else -----------------------------


def _key_text(key: Sequence[Any]) -> str:
    return "(" + ", ".join(f"{axis}={value!r}" for axis, value in zip(AXES, key)) + ")"


def _zeroed(value: float) -> float:
    """`-0.0` is a measured zero written with a sign, and reads as a direction that was measured."""
    return 0.0 if value == 0 else value


def _below_resolution(value: float, places: int) -> bool:
    return value != 0 and abs(round(value, places)) < 10**-places / 2


def _fixed(value: float, places: int) -> str:
    """A magnitude at `places`, saying so when a nonzero one is below the resolution.

    It used to return `abs(value)`, which is a renderer deciding a number is not what it is: a
    malformed rate of `-0.5` published as `50.00%`. The sign is kept here and the range is refused
    at load, so a value outside its own scale aborts instead of being made presentable.
    """
    value = _zeroed(value)
    if _below_resolution(value, places):
        return f"<{10**-places:.{places}f}"
    return f"{value:.{places}f}"


def _signed(value: float, places: int) -> str:
    """A signed magnitude, with the below-resolution case written as a bound and not as a sign.

    `-<0.0001` reads as "less than minus one ten-thousandth", which is the opposite of what it
    means: the value is a small negative one, so it is *greater* than that bound. A negative
    magnitude under the resolution is therefore written `>-0.0001` and a positive one `<0.0001`;
    each is literally true of the value it stands for, which `-<0.0001` was not.
    """
    value = _zeroed(value)
    if _below_resolution(value, places):
        bound = f"{10**-places:.{places}f}"
        return f"<{bound}" if value > 0 else f">-{bound}"
    return f"{'-' if value < 0 else '+'}{abs(value):.{places}f}"


def _percent(value: float) -> str:
    return f"{_fixed(value * 100, _PERCENT_PLACES)}%"


def _signed_percent(value: float) -> str:
    return f"{_signed(value * 100, _PERCENT_PLACES)}"


def _duration(nanoseconds: float) -> str:
    """Nanoseconds as a duration. The one place in this repository that converts them."""
    magnitude = abs(nanoseconds)
    sign = "-" if nanoseconds < 0 else ""
    if magnitude < 1_000:
        return f"{sign}{magnitude:.0f} ns"
    if magnitude < 1_000_000:
        return f"{sign}{magnitude / 1_000:.2f} us"
    if magnitude < 1_000_000_000:
        return f"{sign}{magnitude / 1_000_000:.2f} ms"
    if magnitude < 60_000_000_000:
        return f"{sign}{magnitude / 1_000_000_000:.2f} s"
    if magnitude < 3_600_000_000_000:
        return f"{sign}{magnitude / 60_000_000_000:.2f} min"
    return f"{sign}{magnitude / 3_600_000_000_000:.2f} h"


def _axis_text(value: Any) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"`{value}`"


def _sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, "true" if value else "false")
    if isinstance(value, (int, float)):
        return (2, f"{float(value):030.9f}")
    return (3, str(value))


def _tuple_sort_key(values: Sequence[Any]) -> tuple[tuple[int, str], ...]:
    return tuple(_sort_key(value) for value in values)


# --- the four column kinds, each taking a whole cell --------------------------------------------
#
# Four functions, four signatures, each `(Cell) -> str`. A renderer taking a float would be a
# renderer that could be handed a number which had already lost the `n` it was measured over, and
# that number is exactly what this module exists to stop appearing in the README.


def _render_count(cell: Cell) -> str:
    """`k / n (share%)` -- a census, whose denominator is the whole of its evidence."""
    assert cell.k is not None and cell.n is not None
    share = _percent(cell.k / cell.n) if cell.n else _MISSING
    return f"{cell.k:,}/{cell.n:,} ({share})"


def _render_rate(cell: Cell) -> str:
    """`value% [lo, hi] k/n` -- the rate, its interval and the count it was measured over."""
    assert cell.value is not None and cell.interval is not None
    lo, hi, _ = cell.interval
    return (
        f"{_percent(cell.value)} [{_percent(lo)}, {_percent(hi)}] "
        f"{cell.k:,}/{cell.n:,}"
    )


def _render_delta(cell: Cell) -> str:
    """A difference, on the scale its own interval method says the estimand lives on.

    A difference of rates is percentage points; a difference of areas is an area. Deciding by the
    method rather than by a list of limb names means a limb added later is rendered on the right
    scale the day it lands.
    """
    assert cell.value is not None and cell.interval is not None
    lo, hi, method = cell.interval
    if method in _AUC_INTERVAL_METHODS:
        return (
            f"{_signed(cell.value, _AREA_PLACES)} "
            f"[{_signed(lo, _AREA_PLACES)}, {_signed(hi, _AREA_PLACES)}]"
        )
    return (
        f"{_signed_percent(cell.value)} pp "
        f"[{_signed_percent(lo)}, {_signed_percent(hi)}]"
    )


def _render_auc(cell: Cell) -> str:
    """`auc [lo, hi] n+ vs n-` -- an area, with both arms it was computed over."""
    assert cell.value is not None and cell.interval is not None
    lo, hi, _ = cell.interval
    return (
        f"{_fixed(cell.value, _AREA_PLACES)} "
        f"[{_fixed(lo, _AREA_PLACES)}, {_fixed(hi, _AREA_PLACES)}] "
        f"{cell.n_positive:,} vs {cell.n_negative:,}"
    )


_COLUMN_RENDERERS: Final[Mapping[str, Callable[[Cell], str]]] = {
    "count": _render_count,
    "rate": _render_rate,
    "delta": _render_delta,
    "auc": _render_auc,
}


# --- prose carried in the file ------------------------------------------------------------------


RENDERED_UNITS: Final[tuple[str, ...]] = (
    "%",
    "pp",
    "ns",
    "us",
    "µs",
    "ms",
    "min",
    "bytes",
    "s",
    "h",
)
"""The units this module itself writes. A stored string carrying one is a pre-formatted figure.

The guard is against *these* units rather than against every digit for a stated reason: a reader
cannot tell a figure this module rendered from one the file already carried, and the units are what
make them look alike. N3's stored reason published `18394582 ns` and `1000000.0 ns` two lines above
the same two figures rendered here as `18.39 ms` and `1.00 ms` -- three different spellings of two
numbers, one of them not this module's. A bare number with no unit (a CUDA compute capability
`8.6`, a date, a count of items) is not a figure in any of these units and is left alone."""

_FORMATTED_FIGURE: Final[re.Pattern[str]] = re.compile(
    # The word-shaped units take a `\b` and `%` must not: `%` is not a word character, so `%\b`
    # never matches before a space and the guard silently passed every percentage it was written
    # for. Longest-first, so `18.39 ms` is a duration in milliseconds and not one in seconds.
    r"\d(?:[\d,]*\d)?(?:\.\d+)?\s*(?:%|(?:bytes|min|ms|ns|us|µs|pp|s|h)\b)"
)


def _stored_text(text: str, where: str, failures: list[str]) -> str:
    """A string the file stores, collapsed to one line and refused if it is a pre-formatted figure.

    Every string that reaches the reader goes through this -- narrative prose, the string values
    inside a `computed` block, every axis value that becomes a row or column label, every finding's
    `kind`, and the field names the file supplies -- because the rule is about what a reader sees,
    not about which field it arrived in. A field name is exempt from nothing: `p95_ns` passes
    because the underscore is not a space, and a key that did carry a figure would publish it
    exactly as a value would.

    `summary.py` bakes a `%` into some `statement` strings, which is why the guard existed at all;
    the units above are why it is not about percentages alone.
    """
    collapsed = " ".join(text.split())
    found = _FORMATTED_FIGURE.search(collapsed)
    if found is not None:
        failures.append(
            f"{where} carries the pre-formatted figure {found.group(0)!r}. Figures are computed "
            f"from values here and nowhere else, so a stored string already carrying one would "
            f"publish a number this module never rendered, in units a reader would take for its "
            f"output, beside the same number rendered differently"
        )
    return collapsed


# --- the generic renderer for `computed` blocks -------------------------------------------------


def _attributed(values: Sequence[str], baselines: int, where: str, failures: list[str]) -> str:
    """A chain list from a verdict's `computed`, attributed per baseline rather than collapsed.

    These lists hold **one entry per baseline-chain pair** and record only the chain, so the raw
    list prints an encoding once per baseline and reads as several encodings. Deduplicating it was
    the obvious repair and is wrong: N4 holds `base64+base64+base64+base64` in both
    `chains_recovering_off_distribution` and `chains_degrading_off_distribution` -- it recovers for
    one baseline and degrades for the other -- and two collapsed lists publish the same chain twice
    with opposite meanings and no way for a reader to resolve it.

    So each encoding is named once with the number of baselines it accounts for, out of the number
    the verdict examined. `on 1 of 2` in each of the opposed lists is resolvable; `base32` twice is
    not, and neither is a bare `base32`. The baseline count is read from the verdict's own `keys`
    rather than declared here, so it is the file's number.
    """
    counts: dict[str, int] = {}
    for value in values:
        _stored_text(value, where, failures)
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "none"
    over = f" of {baselines}" if baselines else ""
    named = ", ".join(f"`{name}` on {count}{over}" for name, count in sorted(counts.items()))
    pairs = len(values)
    return f"{pairs} baseline-chain pair{'' if pairs == 1 else 's'}: {named}"


def _computed_scalar(name: str, value: Any, where: str, failures: list[str]) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and name.endswith("_ns"):
        return _duration(float(value))
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _fixed(value, _PLAIN_PLACES) if value >= 0 else _signed(value, _PLAIN_PLACES)
    if isinstance(value, str):
        return f"`{_stored_text(value, where, failures)}`"
    return _MISSING


def _looks_like_interval(value: Any) -> bool:
    return isinstance(value, Mapping) and {"lo", "hi", "method"} <= set(value)


def _interval_text(value: Mapping[str, Any], where: str, failures: list[str]) -> str:
    lo, hi = float(value["lo"]), float(value["hi"])
    method = _stored_text(str(value["method"]), f"{where}.method", failures)
    return f"[{_signed(lo, _PLAIN_PLACES)}, {_signed(hi, _PLAIN_PLACES)}] `{method}`"


def _computed_entry(
    name: str,
    value: Any,
    block: Mapping[str, Any],
    where: str,
    failures: list[str],
    baselines: int | None = None,
) -> str:
    """One `computed` field as text, pairing a value with its own interval when the block has one.

    `baselines` is the number of baselines a **verdict** was measured over, and it is `None`
    everywhere else. Only a verdict's own chain lists carry one entry per baseline; a list of
    execution providers or of interval methods does not, and attributing one of those "per
    baseline" would be a sentence about a structure the value does not have.
    """
    if _looks_like_interval(value):
        return _interval_text(value, f"{where}.{name}", failures)
    if isinstance(value, list):
        if not value:
            return "none"
        if all(isinstance(item, str) for item in value):
            if baselines is None:
                return ", ".join(
                    f"`{_stored_text(item, f'{where}.{name}', failures)}`" for item in value
                )
            return _attributed(value, baselines, f"{where}.{name}", failures)
        return "; ".join(
            _computed_inline(item, f"{where}.{name}", failures) for item in value
        )
    if isinstance(value, Mapping):
        return _computed_inline(value, f"{where}.{name}", failures)
    text = _computed_scalar(name, value, f"{where}.{name}", failures)
    interval = block.get(f"{name}_interval")
    if _looks_like_interval(interval):
        text = f"{text} {_interval_text(interval, f'{where}.{name}_interval', failures)}"
    return text


def _computed_inline(value: Any, where: str, failures: list[str]) -> str:
    if isinstance(value, Mapping):
        if _looks_like_interval(value):
            return _interval_text(value, where, failures)
        return (
            "("
            + ", ".join(
                f"{_stored_text(name, f'{where} field name', failures)} "
                + _computed_entry(name, value[name], value, where, failures)
                for name in value
                if not name.endswith("_interval") or f"{name[: -len('_interval')]}" not in value
            )
            + ")"
        )
    return _computed_scalar("", value, where, failures)


def _rendered_names(block: Mapping[str, Any]) -> list[str]:
    """The field names of a `computed` block that get a line of their own.

    An `X_interval` beside an `X` is not one: it renders inside `X`'s line, where it bounds the
    figure it belongs to instead of floating below it as a second entry.
    """
    return [
        name
        for name in block
        if not (name.endswith("_interval") and name[: -len("_interval")] in block)
    ]


def _computed_lines(
    block: Mapping[str, Any],
    where: str,
    failures: list[str],
    baselines: int | None = None,
    indent: str = "  ",
) -> list[str]:
    """A `computed` block as a bullet list, with each interval attached to the value it bounds."""
    return [
        f"{indent}- `{_stored_text(name, f'{where} field name', failures)}`: "
        + _computed_entry(name, block[name], block, where, failures, baselines)
        for name in _rendered_names(block)
    ]


# --- tables -------------------------------------------------------------------------------------


def _escape(text: str) -> str:
    """One table cell's text, made safe to sit between two pipes.

    A `|` in a value ends the cell and shifts every column after it by one, and a newline ends the
    row and turns the rest into a paragraph. Neither is hypothetical: a chain name, a census name
    and a device string all come from files this module does not control, and a table that silently
    grows a column is a table whose figures are under the wrong headings. Escaped here rather than
    at each call site, so no future cell can be built without it.
    """
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(_escape(cell) for cell in cells) + " |"


@dataclass(frozen=True, slots=True)
class Section:
    """One table, declared as a predicate and two lists of coordinate names.

    Rows and columns are derived from the values the file actually holds on those coordinates, so
    no label in the output is typed here. `claim` decides which cells the section is responsible
    for; every claimed cell lands in exactly one slot, or the render aborts.

    `folded` is the curation lever, and it is **declared per section rather than derived from
    anything a cell carries**. Which of these tables is a headline and which is the evidence under
    one is an editorial judgement about this argument, not a property of the data. Declaring it
    here keeps that judgement in one readable list and out of the render loop, and a folded section
    still emits every byte it emitted before -- a fold removes rendered height, never a figure.
    """

    name: str
    lead_in: str
    claim: Callable[[Cell], bool]
    row_axes: tuple[str, ...]
    column_axes: tuple[str, ...]
    folded: bool = False


def _column_labels(columns: Sequence[tuple[Any, ...]], column_axes: Sequence[str]) -> list[str]:
    """Short labels where they stay distinct, `axis=value` where they would not.

    Dropping a `None` coordinate keeps the header readable -- an attack column has no benign class
    -- but two different coordinate tuples must never print the same heading, so the short form is
    used only when it happens to be injective on this section's columns.
    """
    short = [
        " / ".join(_axis_text(value) for value in column if value is not None) or _MISSING
        for column in columns
    ]
    if len(set(short)) == len(short):
        return short
    return [
        ", ".join(f"{axis}={_axis_text(value)}" for axis, value in zip(column_axes, column))
        for column in columns
    ]


def _build_section(
    section: Section,
    cells: Sequence[Cell],
    anchors: Anchors,
    failures: list[str],
) -> tuple[list[str], set[tuple[Any, ...]]]:
    """The section's lines and the identities it placed. Empty lines mean the section is absent.

    The lead-in travels with the table rather than being emitted by the caller: a section whose
    body renders no rows must take its prose with it, or the README promises a comparison over a
    table that is not there.

    A `folded` section puts its table inside a `<details>` and **leaves the lead-in outside it**:
    the sentence that says what was folded away is the one thing a reader skimming past a closed
    fold must still meet, or the fold hides a claim rather than its evidence. An empty folded
    section emits nothing at all, `<details>` included -- the rule above is about the whole
    section, and a summary offering to expand a table that does not exist is the same broken
    promise the lead-in rule refuses.
    """
    claimed = [cell for cell in cells if section.claim(cell)]
    if not claimed:
        return [], set()

    rows = sorted(
        {tuple(cell.coord(axis) for axis in section.row_axes) for cell in claimed},
        key=_tuple_sort_key,
    )
    columns = sorted(
        {tuple(cell.coord(axis) for axis in section.column_axes) for cell in claimed},
        key=_tuple_sort_key,
    )
    row_at = {row: index for index, row in enumerate(rows)}
    column_at = {column: index for index, column in enumerate(columns)}

    grid: list[list[Cell | None]] = [[None] * len(columns) for _ in rows]
    placed: set[tuple[Any, ...]] = set()
    for cell in claimed:
        row = row_at[tuple(cell.coord(axis) for axis in section.row_axes)]
        column = column_at[tuple(cell.coord(axis) for axis in section.column_axes)]
        occupant = grid[row][column]
        if occupant is not None:
            failures.append(
                f"the {section.name} table would put {_key_text(cell.key)} and "
                f"{_key_text(occupant.key)} in one slot, so one of them would never be rendered"
            )
            continue
        grid[row][column] = cell
        placed.add(cell.identity)

    # **No row is all-missing and no column is empty, by construction rather than by a filter.**
    # `rows` and `columns` are the coordinates the claimed cells actually carry, so a row exists
    # only because some cell has that row coordinate -- and that same cell's column coordinate is
    # in `columns`, so it fills a slot in that row. The one way it does not is the slot collision
    # above, which appends a failure and aborts the whole render.
    #
    # There were filters here. They could not fire in any render that returned, and the two tests
    # holding them were both vacuous: deleting the filters left the suite green. A guard that
    # cannot be false is worse than none, because it reads as the thing that makes the invariant
    # true. `test_no_row_and_no_column_of_the_block_is_entirely_missing` asserts the property, and
    # `test_a_row_exists_only_because_a_cell_fills_it` asserts the reason.
    labels = _column_labels(columns, section.column_axes)
    header = [f"`{axis}`" for axis in section.row_axes] + labels
    rule = ["---"] * len(section.row_axes) + ["---:"] * len(labels)

    table = [_row(header), _row(rule)]
    for index in range(len(rows)):
        texts = [_axis_text(value) for value in rows[index]]
        for column in range(len(columns)):
            cell = grid[index][column]
            if cell is None:
                texts.append(_MISSING)
                continue
            texts.append(_COLUMN_RENDERERS[cell.kind](cell) + anchors.markers(cell))
        table.append(_row(texts))

    if not section.folded:
        return ["", section.lead_in, "", *table], placed

    # The blank line after `<summary>` is not cosmetic: without it the table inside the fold is
    # rendered as one paragraph of literal pipes, and the blank line before `</details>` is what
    # keeps the closing tag out of the table's last row.
    return (
        [
            "",
            section.lead_in,
            "",
            f"<details><summary>{section.name} -- {len(claimed):,} "
            f"cell{'' if len(claimed) == 1 else 's'}</summary>",
            "",
            *table,
            "",
            "</details>",
        ],
        placed,
    )


def _sections(cells: Sequence[Cell]) -> list[Section]:
    """The tables, declared once. Together they must claim every cell the file holds.

    Nothing here names a chain, a baseline or a benign class: a section is a predicate over the
    coordinates a cell declares, and the rows and columns are whatever values the file carries on
    them. The one place a literal appears is where the file's own vocabulary is the predicate --
    the four cell kinds, the two populations, and the contrast prefixes.

    **`folded` is declared here and nowhere else, and it is the only editorial judgement in this
    module.** The rates are the claim and are never folded; the verdicts under the tables are the
    claim being decided and are never folded either. Every other section declared below is the
    evidence *for* a claim one of those two already makes -- the same difference at the threshold
    and threshold-free, the same difference over the matched population, the separation each
    difference is a difference in, what each dressing cost before the layer saw it, the sensitivity
    pass, and the censuses underneath all of it -- so each of those carries its lead-in in the open
    and its table behind a fold. Nothing is dropped: every folded row is still in the file, still
    diffable, still one click away. The count of folded sections is not typed here either; it is
    whatever this list declares, and `test_readme.py` asserts the split against the declaration.
    """
    counts = sum(1 for cell in cells if cell.kind == "count")
    rates = sum(1 for cell in cells if cell.kind == "rate")

    def is_headline(cell: Cell) -> bool:
        return cell.coord("window_policy") == HEADLINE_WINDOW_POLICY

    def is_all(cell: Cell) -> bool:
        return is_headline(cell) and cell.coord("population") == "all"

    return [
        Section(
            name="rates",
            lead_in=(
                f"**The rates, per benign class, never pooled.** {rates} `rate` cells. Each is the "
                f"rate, its interval, and the `k` of `n` it was measured over; the columns are the "
                f"`family` and `benign_class` the rate is about, and the rows carry the layer's "
                f"state in `canon_on`."
            ),
            claim=lambda cell: cell.kind == "rate" and is_all(cell),
            row_axes=("baseline", "dressing_chain", "chain_class", "canon_on"),
            column_axes=("family", "benign_class"),
        ),
        Section(
            name="canon deltas at the threshold",
            lead_in=(
                "**What the layer changes at the threshold: canonicalization on minus off.** "
                "Percentage points, with the paired interval. A positive false-positive column and "
                "a positive recall column are a cost and a recovery respectively, and the "
                "pre-registered conditions below subtract one from the other."
            ),
            claim=lambda cell: (
                cell.kind == "delta"
                and is_all(cell)
                and cell.coord("contrast") == "canon_on_vs_off"
                and cell.method not in _AUC_INTERVAL_METHODS
            ),
            row_axes=("baseline", "dressing_chain", "chain_class"),
            column_axes=("family", "benign_class"),
            folded=True,
        ),
        Section(
            name="canon deltas, threshold-free",
            lead_in=(
                "**The same change, threshold-free.** The difference in area under the ROC curve "
                "between the two canon states, which moves for re-ranking anywhere in the score "
                "range rather than only at the operating point."
            ),
            claim=lambda cell: (
                cell.kind == "delta"
                and is_all(cell)
                and cell.coord("contrast") == "canon_on_vs_off"
                and cell.method in _AUC_INTERVAL_METHODS
            ),
            row_axes=("baseline", "dressing_chain", "chain_class"),
            column_axes=("benign_class",),
            folded=True,
        ),
        Section(
            name="dressing deltas",
            lead_in=(
                "**What each dressing costs before the layer sees it: the clean text minus the "
                "dressed text.** The `contrast` names the chain. These are the differences the "
                "layer is asked to recover, measured with canonicalization both off and on."
            ),
            claim=lambda cell: (
                cell.kind == "delta"
                and is_all(cell)
                and isinstance(cell.coord("contrast"), str)
                and str(cell.coord("contrast")).startswith("clean_vs_")
            ),
            row_axes=("baseline", "contrast", "canon_on"),
            column_axes=("family", "benign_class"),
            folded=True,
        ),
        Section(
            name="separation",
            lead_in=(
                "**Separation, threshold-free.** Area under the ROC curve for attacks against each "
                "benign class, with both arm sizes. A value below 0.5 is an ordering the wrong way "
                "round, not a rounding artefact."
            ),
            claim=lambda cell: cell.kind == "auc" and is_all(cell),
            row_axes=("baseline", "dressing_chain", "chain_class", "canon_on"),
            column_axes=("benign_class",),
            folded=True,
        ),
        Section(
            name="matched windows",
            lead_in=(
                "**The same canon-on-versus-off difference, over the items that occupy one window "
                "under both canon states.** A document over several windows is scored as the "
                "maximum over them, so part of a difference measured over everything is the layer "
                "changing how many windows a document needs. This companion population removes "
                "that."
            ),
            claim=lambda cell: (
                is_headline(cell) and cell.coord("population") == "single_window"
            ),
            row_axes=("baseline", "dressing_chain", "chain_class"),
            # `kind` is a column axis for the reason it is one in the sensitivity section: this
            # claim is kind-agnostic, so a second kind landing in this population would map to the
            # slot the first already holds and the render would abort on data that is not wrong.
            column_axes=("kind", "family", "benign_class"),
            folded=True,
        ),
        Section(
            name="sensitivity",
            lead_in=(
                "**The sensitivity pass, under a window policy other than the headline's.** These "
                "cells are never averaged into the bound results: a sensitivity check folded into "
                "the headline makes the headline neither policy."
            ),
            claim=lambda cell: cell.coord("window_policy") in SENSITIVITY_WINDOW_POLICIES,
            row_axes=("window_policy", "baseline", "dressing_chain", "canon_on"),
            column_axes=("kind", "family", "benign_class"),
            folded=True,
        ),
        Section(
            name="censuses",
            lead_in=(
                f"**What the layer did to the text, counted.** {counts} `count` cells: how many "
                f"items each stage edited, how many hit the recursion ceiling, and how many "
                f"overflowed the window under each canon state. Each is `k` of `n` with its share "
                f"of that denominator."
            ),
            claim=lambda cell: cell.kind == "count" and is_all(cell),
            row_axes=("baseline", "dressing_chain", "chain_class", "family", "benign_class"),
            column_axes=("census", "canon_on"),
            folded=True,
        ),
    ]


# --- the non-tabular blocks ---------------------------------------------------------------------


def _what_ran(run: Mapping[str, Any], failures: list[str]) -> list[str]:
    """The provenance block: what was measured, over what, on what, and by which invocation."""
    lines = ["**What produced these numbers.**", ""]

    def stored(value: object, where: str) -> str:
        return _stored_text(str(value), where, failures)

    lines.append(f"- corpus `build_id`: `{stored(run['build_id'], 'run.build_id')}`")
    for index, entry in enumerate(run["corpus_files"]):
        at = f"run.corpus_files[{index}]"
        lines.append(
            f"- `{stored(entry['name'], at + '.name')}`: {int(entry['rows']):,} rows, "
            f"`sha256` `{stored(entry['sha256'], at + '.sha256')}`"
        )

    profile = run.get("profile")
    if isinstance(profile, str):
        items = run.get("profile_items")
        per_cell = run.get("profile_items_per_cell")
        # **"items scored" named an actor this line cannot name.** The figure is the size of the
        # scored matrix the cells were aggregated from; whether *this* invocation did the scoring
        # is what `run.steps` and the wall-time label below say, and on the committed file they
        # say it did not. So the count is stated as the matrix it is, with no verb attributing it.
        suffix = f", {int(items):,} items in the scored matrix" if isinstance(items, int) else ""
        if isinstance(per_cell, int):
            suffix += f", {per_cell:,} per cell"
        lines.append(f"- profile: `{stored(profile, 'run.profile')}`{suffix}")

    path = run.get("declared_path")
    if isinstance(path, Mapping):
        lines += [
            "- declared execution path: "
            + _computed_inline(path, "run.declared_path", failures)
        ]

    steps = run.get("steps")
    # **An empty list is not a list of steps.** `steps: []` is a list that is not `None`, so the
    # old guard emitted `- steps: ` with nothing after it -- a label whose whole claim is that it
    # names something -- while the wall-time parenthetical below silently vanished, because that
    # one tested the list for truth rather than for presence. One rule now: a run that named no
    # step gets no step line and the same unparenthesised wall-time label as a run with no field.
    named_steps = (
        [stored(step, "run.steps") for step in steps] or None
        if isinstance(steps, list) and all(isinstance(step, str) for step in steps)
        else None
    )
    if named_steps is not None:
        # Printed as the file records them, in order and without collapsing: two `build`
        # entries would be two builds, and a list that quietly reported them as one would be
        # hiding the fact a reader is here for.
        lines.append("- steps: " + ", ".join(f"`{step}`" for step in named_steps))

    wall = run.get("total_wall_ns")
    if isinstance(wall, int):
        # **This is not the scoring run's wall clock and must never be labelled as one.**
        # `total_wall_ns` is started by whichever invocation wrote the file and stopped when it
        # wrote it, so on a `reaggregate` it times a pass that opened no model and scored no item
        # -- and the published block called it "wall time" three lines under "28,600 items
        # scored", where the only reading available to a reader is that scoring 28,600 items took
        # it. The original scoring run's clock is not recoverable: each re-derivation overwrites
        # the field with its own. So the figure is kept and the label is made true, by naming the
        # steps the file itself says this invocation ran.
        covered = (
            " (" + ", ".join(f"`{step}`" for step in named_steps) + ")" if named_steps else ""
        )
        lines.append(f"- wall time of the steps this invocation ran{covered}: {_duration(wall)}")

    methods = run.get("interval_methods")
    if isinstance(methods, list) and all(isinstance(name, str) for name in methods):
        lines.append(
            "- interval methods in this file: "
            + ", ".join(f"`{stored(n, 'run.interval_methods')}`" for n in methods)
        )

    timing = run.get("timing")
    if isinstance(timing, Mapping):
        lines += ["", "**What it cost, measured.**", ""]
        layer = timing.get("layer_ns")
        if isinstance(layer, Mapping):
            overall = layer.get("overall")
            if isinstance(overall, Mapping):
                lines.append(
                    f"- canonicalization layer, overall: p50 {_duration(overall['p50_ns'])}, "
                    f"p95 {_duration(overall['p95_ns'])}, over {int(overall['n']):,} documents"
                )
            by_class = layer.get("by_class")
            if isinstance(by_class, Mapping):
                for name in sorted(by_class):
                    statistic = by_class[name]
                    lines.append(
                        f"- canonicalization layer, "
                        f"`{stored(name, 'run.timing.layer_ns.by_class')}`: p50 "
                        f"{_duration(statistic['p50_ns'])}, p95 {_duration(statistic['p95_ns'])}, "
                        f"over {int(statistic['n']):,} documents"
                    )
        inference = timing.get("inference_ns")
        if isinstance(inference, Mapping):
            by_baseline = inference.get("by_baseline")
            batch = inference.get("batch_size")
            for name in sorted(by_baseline or {}):
                statistic = by_baseline[name]
                at_batch = f", at batch size {batch}" if isinstance(batch, int) else ""
                lines.append(
                    f"- inference, `{stored(name, 'run.timing.inference_ns.by_baseline')}`: "
                    f"p50 {_duration(statistic['p50_ns'])}, p95 "
                    f"{_duration(statistic['p95_ns'])}, over {int(statistic['n']):,} documents"
                    f"{at_batch}"
                )

    reaggregated = run.get("reaggregated")
    if isinstance(reaggregated, Mapping):
        from_steps = reaggregated.get("from_steps")
        named = (
            ", ".join(f"`{stored(s, 'run.reaggregated.from_steps')}`" for s in from_steps)
            if isinstance(from_steps, list) and all(isinstance(s, str) for s in from_steps)
            else _MISSING
        )
        inherited = reaggregated.get("inherited")
        carried = (
            ", ".join(f"`{stored(n, 'run.reaggregated.inherited')}`" for n in inherited)
            if isinstance(inherited, list) and all(isinstance(n, str) for n in inherited)
            else _MISSING
        )
        lines += [
            "",
            (
                "**The latencies above were not measured by the invocation that produced the "
                f"cells.** They were carried forward from a run whose steps were {named}, and the "
                f"fields inherited whole are {carried}. Everything else in this block was "
                "re-derived from the committed scores."
            ),
        ]
        note = reaggregated.get("note")
        if isinstance(note, str):
            lines += ["", _stored_text(note, "run.reaggregated.note", failures)]

    summary = run.get("summary")
    if isinstance(summary, Mapping):
        choice = summary.get("choice")
        if isinstance(choice, str):
            named = stored(choice, "run.summary.choice")
            lines += ["", f"**The threshold-free summary is `{named}`.**", ""]
        rejected = summary.get("rejected")
        if isinstance(rejected, Mapping):
            for name in sorted(rejected):
                text = rejected[name]
                if isinstance(text, str):
                    lines.append(
                        f"- `{stored(name, 'run.summary.rejected')}` was rejected: "
                        + _stored_text(text, f"run.summary.rejected.{name}", failures)
                    )
        justifications = summary.get("rejected_justifications")
        if isinstance(justifications, Mapping):
            for name in sorted(justifications):
                text = justifications[name]
                if isinstance(text, str):
                    lines.append(
                        f"- the `{stored(name, 'run.summary.rejected_justifications')}` "
                        f"justification was withdrawn: "
                        + _stored_text(
                            text, f"run.summary.rejected_justifications.{name}", failures
                        )
                    )
    return lines


def _verdict_headline(verdicts: Sequence[Mapping[str, Any]], failures: list[str]) -> list[str]:
    """What the pre-registered conditions came out as, above the first table.

    The conditions are the only thing in this block that is a *finding* rather than a measurement,
    and until this line existed they were rendered once, in a list under every table. One of them
    triggered. Nothing above the tables said so, and a reader who stopped at the first table --
    which is what a reader with five minutes does -- left having read the evidence for a
    conclusion the block never stated.

    A file that carries conditions always gets a line, because the "nothing fired" case is a
    result too and rendering it as absence makes the presence of the line the message. Which
    outcomes are worth naming is decided by `QUIET_VERDICT_OUTCOME` and by exclusion, so an
    outcome the vocabulary grows later is named by default rather than silently omitted.

    **A file with no conditions at all gets no line, and that is silence chosen deliberately.**
    The alternative was an abort, and an abort is wrong here: `verdict: []` is what a legitimately
    partial file carries -- a results file rendered before any condition was evaluated -- and this
    module renders what a file holds rather than deciding what a file must hold. Nor is it
    silence about a result: there is no result to be silent about, and a sentence saying "0
    conditions were evaluated" above a table would be this module inventing a finding. The rule
    the "never silence" above states is about *outcomes*, and an empty list has none. The
    completeness that would be worth an abort -- a cell nothing renders -- is enforced in `render`,
    against the file's own cells.
    """
    if not verdicts:
        return []
    loud = [
        (
            _stored_text(str(verdict["condition"]), f"verdict[{index}].condition", failures),
            _stored_text(str(verdict["outcome"]), f"verdict[{index}].outcome", failures),
        )
        for index, verdict in enumerate(verdicts)
        if str(verdict["outcome"]) != QUIET_VERDICT_OUTCOME
    ]
    total = len(verdicts)
    quiet = total - len(loud)
    if not loud:
        sentence = (
            f"**What the pre-registered conditions came out as.** All {total} came out "
            f"`{QUIET_VERDICT_OUTCOME}`: none of them fired."
        )
    else:
        # Positive voice, and the outcome spelled out rather than an identifier under a double
        # negative: "1 did not come out `not_triggered`: `N3`" made a reader compose two negations
        # to learn that a condition fired, and then told them only its name.
        named = ", ".join(f"`{condition}` came out `{outcome}`" for condition, outcome in loud)
        # No tail when nothing was quiet, and the count is written as "k of the n" in both limbs so
        # the sentence needs no singular branch to stay grammatical.
        rest = (
            f" {quiet} of the {total} came out `{QUIET_VERDICT_OUTCOME}`." if quiet else ""
        )
        sentence = (
            f"**What the pre-registered conditions came out as.** Of the {total} pre-registered "
            f"falsification conditions, {named}.{rest}"
        )
    return [
        "",
        sentence
        + " Each condition is decided under the tables, from the figures the tables carry.",
    ]


def _verdict_lines(verdicts: Sequence[Mapping[str, Any]], failures: list[str]) -> list[str]:
    """The pre-registered conditions, each rendered from its `computed` block and not its sentence.

    **The stored `reason` is not published.** It is the evaluator's prose and it carries figures
    the evaluator formatted: N3's reads `18394582 ns` against `1000000.0 ns`, two lines above the
    same two numbers rendered here as `18.39 ms` and `1.00 ms`. Three spellings of two figures, one
    of them not this module's, is exactly the drift the block exists to end -- so the rule that
    prefers a finding's `computed` over its `statement` applies to a verdict's `computed` over its
    `reason`, and the guard that would refuse the sentence never has to fire on a file that is
    otherwise fine.

    The limb that says a pre-registered cell could not have decided is not special-cased:
    `cell_could_decide` and `pinned_rates` are fields of `computed` and render like every other
    field, which is what keeps this from parsing a reason string for a fact the file already states.
    """
    if not verdicts:
        return []
    lines = [
        "",
        (
            f"**The {len(verdicts)} pre-registered conditions.** Each states its outcome and the "
            f"figures it was decided on. The evaluator's own sentence is in `results.json` under "
            f"`reason` and is deliberately not reproduced here: it carries figures it formatted "
            f"itself, and two spellings of one number in one document is how a table stops being "
            f"traceable to the file it came from."
        ),
    ]
    for index, verdict in enumerate(verdicts):
        where = f"verdict[{index}]"
        condition = _stored_text(str(verdict["condition"]), f"{where}.condition", failures)
        outcome = _stored_text(str(verdict["outcome"]), f"{where}.outcome", failures)
        keys = verdict.get("keys")
        named = keys if isinstance(keys, list) else []
        # The baseline count comes from the verdict's own keys, so the denominator a chain list is
        # attributed against is the file's number and not one typed here.
        baselines = len({key.get("baseline") for key in named if isinstance(key, Mapping)} - {None})
        lines += ["", f"**`{condition}` -- `{outcome}`.**", ""]
        lines += _computed_lines(verdict["computed"], f"{where}.computed", failures, baselines)
        if named:
            lines.append(
                f"  - measured over: {len(named)} cells, {baselines} baseline"
                f"{'' if baselines == 1 else 's'}"
            )
    return lines


def _finding_lines(
    findings: Sequence[Mapping[str, Any]], anchors: Anchors, failures: list[str]
) -> list[str]:
    """Every finding, numbered in printed order, grouped by the `kind` the file gave it.

    No `kind` is named anywhere in this module. A finding is a set of coordinates and a `computed`
    block, and the anchor in the tables above is its number, so a kind invented tomorrow renders
    beside the cells it names without a line being changed here.

    A finding whose coordinates several measurements share carries no marker in the tables, and its
    entry says so rather than leaving a reader looking for a bracket that is not there.

    **Findings of one kind whose `computed` blocks are the same block are stated once.** The
    committed file raises one kind whose every member carries the identical fields, and the block
    published each of them as its own bullet of the same sentence -- a run of lines carrying one
    fact, in the artifact whose whole claim is that it is readable. The collapse is on the
    `computed` block and on nothing else: no `kind` decides it, so it fires wherever a run repeats
    itself and never where two findings differ by a byte.

    **Inside a collapsed entry a finding carries its coordinates only when nothing else does.**
    The first collapse stated the shared `computed` once and then printed every member's full
    coordinate tuple, which turned a hundred lines into one line of seventeen thousand characters:
    the same words, unwrapped. But a finding that anchored is already beside its own cell in a
    table above -- that is what the anchor mechanism is *for* -- so its entry needs only its
    number, and the walk back to the coordinates is the marker. A finding in `Anchors.shared`
    anchored nowhere: there is no table row to walk back to, so it keeps its tuple and its entry
    says how many measurements share it. Anchored or not is read off the anchors, never off a
    `kind` or a count.
    """
    if not findings:
        return []

    grouped = _grouped_findings(findings)
    # The set decides whether a collapsed entry has to repeat its coordinates; the sum is how many
    # markers a reader will actually meet, and they are not the same number when one finding names
    # two cells.
    anchored_numbers = {number for numbers in anchors.at.values() for number in numbers}
    anchored = sum(len(numbers) for numbers in anchors.at.values())
    lines = [
        "",
        (
            f"**{len(findings)} findings the aggregator raised, in {len(grouped)} kinds.** A "
            f"bracketed number beside a figure above is a finding that names that measurement, and "
            f"{anchored} such markers appear. **A marker sits in the table cell it is "
            f"about, and where that table is inside a fold the fold has to be open before the "
            f"browser's find-in-page will reach it** -- collapsed content is not searched. A "
            f"finding whose nine coordinates are shared by more than one measurement -- a rate and "
            f"a census count can sit at the same coordinates -- is anchored to none of them and "
            f"says so, because the file records no `kind` on a finding's keys and guessing which "
            f"measurement was meant is how a marker lands on a figure it is not about."
        ),
    ]
    for kind, groups in grouped.items():
        held = sum(len(group) for group in groups)
        varying = _varying_axes([findings[index] for group in groups for index in group])
        collapsed = held - len(groups)
        # One entry is "repeats ... and is stated"; more than one is "repeat ... and are stated".
        # The plural-only sentence shipped, and read "1 of them repeat another's `computed`
        # exactly and are stated with it" -- in the same block that grew a singular branch for the
        # fold's cell count.
        stated = ""
        if collapsed == 1:
            stated = " 1 of them repeats another's `computed` exactly and is stated with it."
        elif collapsed:
            stated = (
                f" {collapsed} of them repeat another's `computed` exactly and are stated with it."
            )
        lines += ["", f"**`{kind}`** -- {held}.{stated}", ""]

        def at(index: int, varying: tuple[str, ...] = varying) -> str:
            """One finding's own coordinates, with the note for a marker it could not carry."""
            coordinates = " ; ".join(
                ", ".join(f"{axis}={_axis_text(key[axis])}" for axis in varying)
                for key in findings[index]["keys"]
            )
            shared = anchors.shared.get(index)
            return (coordinates or "every cell") + (
                f" (not anchored: {shared} measurements share these coordinates)" if shared else ""
            )

        for group in groups:
            computed = findings[group[0]]["computed"]
            where = f"finding[{group[0]}].computed"
            figures = ", ".join(
                f"`{_stored_text(name, where + ' field name', failures)}` "
                + _computed_entry(name, computed[name], computed, where, failures)
                for name in _rendered_names(computed)
            )
            if len(group) == 1:
                lines.append(f"- **[{anchors.numbers[group[0]]}]** {at(group[0])}: {figures}")
                continue
            covers = " ; ".join(
                f"**[{anchors.numbers[index]}]**"
                + ("" if anchors.numbers[index] in anchored_numbers else f" {at(index)}")
                for index in group
            )
            lines.append(
                f"- **{len(group)} findings carrying one `computed`**, stated once: {figures}. "
                f"They are {covers}."
            )
    return lines


def _grouped_findings(
    findings: Sequence[Mapping[str, Any]],
) -> dict[str, list[list[int]]]:
    """The findings' positions in the file, grouped as the block prints them: kind, then `computed`.

    One function, called by both the numbering and the printing, because "numbered in the order
    they are printed" stops being true the moment the two derive that order separately -- and a
    collapse that reorders the printing without reordering the numbering is exactly how a reader
    ends up at [19] beside a figure with nowhere to look, which is the defect `_finding_notes`
    already exists to have fixed.

    The `computed` block is compared as one canonically serialized value rather than field by
    field: two blocks are the same block when every name and every value in them is the same, and
    a single differing byte anywhere puts them in different groups. Key *order* is deliberately not
    part of it -- it is not something a reader of the rendered block can see, and a producer that
    emitted its fields in a different order would otherwise publish the same fact twice.
    """
    by_kind: dict[str, dict[str, list[int]]] = {}
    for index, finding in enumerate(findings):
        by_computed = by_kind.setdefault(str(finding["kind"]), {})
        by_computed.setdefault(
            json.dumps(finding["computed"], sort_keys=True), []
        ).append(index)
    return {kind: list(by_kind[kind].values()) for kind in sorted(by_kind)}


def _varying_axes(findings: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The axes that are not the same value on every key of every finding in the group.

    An axis constant across a kind carries no information about which cell a finding is about, and
    printing nine axes per finding when two of them vary is how a table stops being readable.
    """
    varying: list[str] = []
    for axis in AXES:
        seen = {
            json.dumps(key[axis], sort_keys=True)
            for finding in findings
            for key in finding["keys"]
        }
        if len(seen) > 1:
            varying.append(axis)
    return tuple(varying) or AXES


@dataclass(frozen=True, slots=True)
class Anchors:
    """Where each finding's marker goes, what number it carries, and which ones went nowhere.

    `at` is keyed on `Cell.identity`, not on the nine axes. **The anchor and the cell identity are
    one function**, and they were not: cells are stored under `(kind, census, nine axes)` while the
    anchor was keyed on the nine axes alone, so a finding about a false-positive *rate* was stamped
    on the seven census *counts* that share the rate's coordinates -- twenty-one figures carrying a
    marker for a finding that is not about them, which made the block's own sentence about what a
    bracketed number means false.

    A finding names a coordinate and not a measurement: `keys` carries the nine axes and no `kind`.
    So the rule is **exactly one, or none**. A coordinate held by one measurement anchors there. A
    coordinate several measurements share anchors nowhere, and the finding's own entry says how
    many share it, because guessing which of them the finding meant is the defect this replaced.
    """

    at: Mapping[tuple[Any, ...], Sequence[int]]
    numbers: Mapping[int, int]
    shared: Mapping[int, int]

    def markers(self, cell: Cell) -> str:
        return "".join(f" [{number}]" for number in self.at.get(cell.identity, ()))


def _finding_notes(findings: Sequence[Mapping[str, Any]], cells: Sequence[Cell]) -> Anchors:
    """Anchor every finding to the measurement it names, and number them in rendered order.

    Numbered by where they are printed rather than by their position in the file. The findings are
    grouped by `kind` below, so file order made the printed list run [151]-[190], [19]-[21],
    [22]-[125] -- a reader who saw [19] beside a figure had no way to find it.

    The printed order is `_grouped_findings`', not one derived here, because the block also groups
    findings that share a `computed` block into one entry: two orders computed separately would
    drift apart the first time the second grouping moved something, and the numbers would stop
    ascending down the page for a second time.

    Nothing here reads a `kind` for anything but the grouping order, so a kind invented tomorrow
    anchors beside the cells it names without a line changing here.
    """
    at_coordinate: dict[tuple[Any, ...], list[Cell]] = {}
    for cell in cells:
        at_coordinate.setdefault(cell.key, []).append(cell)

    order = [
        index
        for groups in _grouped_findings(findings).values()
        for group in groups
        for index in group
    ]
    numbers = {index: position + 1 for position, index in enumerate(order)}

    at: dict[tuple[Any, ...], list[int]] = {}
    shared: dict[int, int] = {}
    for index, finding in enumerate(findings):
        for key in finding["keys"]:
            here = at_coordinate.get(tuple(key[axis] for axis in AXES), ())
            if len(here) == 1:
                at.setdefault(here[0].identity, []).append(numbers[index])
            else:
                shared[index] = max(shared.get(index, 0), len(here))

    for numbered in at.values():
        numbered.sort()
    return Anchors(at=at, numbers=numbers, shared=shared)


# --- the block, and putting it into the README --------------------------------------------------


_PREAMBLE: Final[str] = (
    "<!-- Everything between these two markers is generated from `results/results.json` by "
    "`python -m nbc.report.readme`. Do not edit it: the next run replaces it wholesale, and a "
    "number here that no run produced cannot survive that. -->"
)


def render(results: Results) -> str:
    """The block body, between the markers. Aborts rather than rendering a partial table.

    Completeness is enforced against the file rather than against a test's expectation: the
    sections place cells, the placed identities are compared to the identities the file holds, and
    a difference aborts naming what nothing rendered.
    """
    failures: list[str] = []
    anchors = _finding_notes(results.findings, results.cells)

    lines: list[str] = ["", _PREAMBLE, ""]
    lines += _what_ran(results.run, failures)
    # Above the first table, and above it deliberately: the conditions are the only thing here a
    # reader is owed before the evidence, and they used to be rendered once, under every table.
    lines += _verdict_headline(results.verdicts, failures)

    placed: set[tuple[Any, ...]] = set()
    for section in _sections(results.cells):
        section_lines, section_placed = _build_section(
            section, results.cells, anchors, failures
        )
        overlap = placed & section_placed
        if overlap:
            failures.append(
                f"the {section.name} table claims "
                + ", ".join(sorted(str(identity) for identity in sorted(overlap, key=str))[:3])
                + " which another table already placed"
            )
        placed |= section_placed
        lines += section_lines

    unplaced = [cell for cell in results.cells if cell.identity not in placed]
    if unplaced:
        shown = ", ".join(f"{cell.kind} {_key_text(cell.key)}" for cell in unplaced[:5])
        failures.append(
            f"{len(unplaced)} of {len(results.cells)} cells are in no table: {shown}"
            + (" ..." if len(unplaced) > 5 else "")
            + ". A cell with legal coordinates that nothing renders is as invisible to a reader "
            "as one that was never measured"
        )

    lines += _verdict_lines(results.verdicts, failures)
    lines += _finding_lines(results.findings, anchors, failures)
    lines.append("")

    if failures:
        raise ReportNotRenderable(*failures)

    body = "\n".join(lines).strip("\n") + "\n"
    for marker in (RESULTS_START, RESULTS_END):
        if marker in body:
            raise ReportNotRenderable(
                f"the rendered block contains {marker!r}; a second marker inside the block makes "
                f"the next injection unable to tell where the block ends"
            )
    return body


def inject(readme: str, body: str) -> str:
    """`readme` with the bytes between the markers replaced by `body`, and no other byte moved."""
    failures: list[str] = []
    span = _locate_markers(readme, failures)
    if span is None:
        raise ReportNotRenderable(*failures)
    start, end = span
    return readme[:start] + RESULTS_START + "\n" + body + RESULTS_END + readme[end:]


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, keeping the file's mode.

    `mkstemp` creates 0600 and `os.replace` carries that mode onto the target, so a README that
    was 0644 would come back 0600 and the only visible change would be one a reader meets as a
    permission error much later. The mode is read before the write and restored onto the temporary
    file before the rename, so the destination survives the write unchanged except in its bytes.

    Atomic because the alternative is a README that is half old and half new: `os.replace` is a
    rename within one directory, so the file a reader opens is either wholly the old one or wholly
    the new one.
    """
    directory = path.parent
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = None

    handle, temporary = tempfile.mkstemp(dir=directory, prefix=path.name + ".", suffix=".tmp")
    try:
        try:
            stream = os.fdopen(handle, "w", encoding="utf-8", newline="\n")
        except BaseException:
            # `os.fdopen` can raise before the file object takes ownership of the descriptor, and
            # the `with` form leaks it when it does. Closing it here is the only place that can.
            os.close(handle)
            raise
        with stream:
            stream.write(text)
            # Flushed and synced *before* the rename, so the atomicity is against a crash and not
            # only against a concurrent reader: `os.replace` orders the directory entry, not the
            # data behind it, and a README pointing at unwritten blocks is neither wholly old nor
            # wholly new.
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def render_into(results_path: Path, readme_path: Path) -> dict[str, object]:
    """Render `results_path` into `readme_path`'s markers. The one entry point every caller uses.

    Returns what it rendered, for the terminal and for a caller that wants to assert on it. Every
    failure -- an unreadable results file, a malformed one, a README whose markers cannot be
    located, a README that cannot be read or written -- is `ReportNotRenderable`.
    """
    results = load_results(results_path)
    body = render(results)

    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError as unreadable:
        raise ReportNotRenderable(
            f"{readme_path} could not be read ({unreadable.strerror or unreadable}); there is "
            f"nothing to inject the block into"
        ) from unreadable
    except ValueError as undecodable:
        raise ReportNotRenderable(
            f"{readme_path} is not readable UTF-8 ({undecodable})"
        ) from undecodable

    updated = inject(readme, body)
    if updated != readme:
        try:
            _write(readme_path, updated)
        except OSError as unwritable:
            raise ReportNotRenderable(
                f"{readme_path} could not be written ({unwritable.strerror or unwritable}); the "
                f"README on disk is unchanged"
            ) from unwritable

    return {
        "readme": str(readme_path),
        "results": str(results_path),
        "cells_rendered": len(results.cells),
        "verdicts_rendered": len(results.verdicts),
        "findings_rendered": len(results.findings),
        "block_chars": len(body),
        "readme_changed": updated != readme,
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.report.readme` -- the renderer, runnable on its own by a reader and by CI."""
    import argparse

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.report.readme",
        description=(
            "Render results/results.json into the README's generated block. Reads those two "
            "files and nothing else, and writes only the bytes between the two markers."
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        metavar="PATH",
        help="the results file to render (default: %(default)s, relative to the working directory)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        metavar="PATH",
        help="the README to inject into (default: %(default)s, relative to the working directory)",
    )
    args = parser.parse_args(argv)

    try:
        report = render_into(args.results, args.readme)
    except ReportNotRenderable as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
