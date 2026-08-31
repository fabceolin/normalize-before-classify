"""One results file, four keys, and the assertion that a half-built table cannot ship.

Six modules now produce the pieces of a table and nothing assembled them. A stranger reproducing
this had no command to run, and -- worse -- nothing would have stopped a **partial** run from
looking like a complete one. A table with recall for one baseline and not the other, or a
false-positive rate for one benign class and not the other, reads exactly like a finished table.
That is the failure this module exists to make impossible.

**"A run that produces only one half fails" is an assertion here, not a promise.** Before anything
is rendered, every `baseline x dressing_chain x canon_on` key must carry an attack recall `Rate`, a
false-positive `Rate` for **each** benign class, and an `Auc` per benign class -- over both chain
classes, and over both window policies where a baseline declares a publisher protocol. The run
aborts naming what is missing, aborts on an empty held-out block, and aborts on any verdict that
came out `not_evaluable`.

**Four keys, and the discipline is the point.** `schema_version`, `run`, `cells`, `verdict`. Every
parameter any decision in this project mandates recording goes into `run` and never into a key of
its own, so a reader looking for what a run was configured with has one place to look. `verdict`
earns its own key because a falsification outcome buried inside a parameter block reads as a
parameter.

**One tension, recorded rather than resolved by adding a key.** Story 4.4's summary *findings* are
not parameters, and the rule puts them inside `run` anyway. The rule is kept because its purpose is
that a reader knows where to look, and a fifth key earned by one story is how a file grows a sixth.

**This module is pure.** It holds the file's shape and the gate; `harness/run.py` does the ordering
and the IO. That is what makes the sharp cases -- a missing benign class, a missing AUC, an empty
held-out block -- tests that run in milliseconds against hand-built cells, rather than tests that
need a model.

**It does not render.** Story 5.1 is "the table is a pure function of the results file", and a
renderer here would be that function written in two epics. The `report` subcommand does everything
up to the file and then says what is missing, rather than emitting an empty table that looks like a
rendered one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from nbc.corpus.matrix import parse_item_id, render_chain
from nbc.errors import NbcError
from nbc.schema import (
    BENIGN_CLASSES,
    CHAIN_CLASSES_FOR_KEYS,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    Auc,
    CellKey,
    POPULATION_ALL,
    Rate,
    Verdict,
)

__all__ = [
    "PROFILES",
    "PROFILE_FULL",
    "PROFILE_SMOKE",
    "RESULT_KEYS",
    "SCHEMA_VERSION",
    "STEPS",
    "ResultsFile",
    "ResultsIncomplete",
    "RunBlock",
    "completeness_problems",
    "refuse_an_incomplete_table",
    "render_results",
    "smoke_sample",
]

SCHEMA_VERSION: Final[int] = 1
"""The results file's shape. A reader that parsed version 1 knows what these four keys mean."""

RESULT_KEYS: Final[tuple[str, ...]] = ("schema_version", "run", "cells", "verdict")
"""The whole top level, closed.

Every parameter any decision mandates recording goes into `run`. `verdict` is the one exception and
earns it: a falsification outcome buried inside a parameter block reads as a parameter.
"""

STEP_PREFLIGHT: Final[str] = "preflight"
STEP_VERIFY: Final[str] = "verify"
STEP_BUILD: Final[str] = "build"
STEP_MEASURE: Final[str] = "measure"
STEP_TIME: Final[str] = "time"
STEP_AGGREGATE: Final[str] = "aggregate"

STEPS: Final[tuple[str, ...]] = (
    STEP_PREFLIGHT,
    STEP_VERIFY,
    STEP_BUILD,
    STEP_MEASURE,
    STEP_TIME,
    STEP_AGGREGATE,
)
"""The order, as a declared tuple rather than as the order a function happens to be written in.

Each boundary was paid for and each is invisible in a function body:

*Preflight before verify* -- and therefore before `onnxruntime` is imported. A preflight that fires
after the runtime is imported is checking a floor the import already crashed through.

*Verify before measure* -- the pins **and** the caveats section, before any inference. Eighty-five
hours of scoring that ends at a missing caveats section is eighty-five hours spent to learn
something a file read would have said.

*Build only on a wholly absent corpus* -- a partial corpus is the state where a rebuild writes
half-new rows while the manifest still describes the old ones.

*Time after measure and before aggregate* -- the timing pass is N3's right-hand side, and a run that
aggregated first would evaluate a condition against a number it had not taken yet.

**Rendering is not one of these six**, and `render` used to be listed here without any code path
emitting it -- a seventh step that never happened, which read as though `all` published a table it
never touched. It is the same shape as `reaggregate`: an act on the file this run produced rather
than a step of producing it, so it is declared beside the command that performs it
(`run.STEP_RENDER`) and not in this tuple. The measuring run stops at the file; `report` publishes
it, and `tests/report/test_readme.py` is what refuses a published block the committed file no
longer renders.
"""


PROFILE_FULL: Final[str] = "full"
PROFILE_SMOKE: Final[str] = "smoke"
PROFILES: Final[tuple[str, ...]] = (PROFILE_FULL, PROFILE_SMOKE)
"""Which run produced a table, closed.

A smoke run at a small sample produces a table that is **structurally identical** to the published
one -- same columns, same intervals, same verdicts -- with a small `n` as the only tell. `profile`
goes into the run block so the README's claim about which run produced the table is generated from
the file rather than asserted beside it.
"""


def smoke_sample(items: Sequence[object], items_per_cell: int) -> tuple[object, ...]:
    """The first `items_per_cell` items of every cell group, by item id.

    **Per cell, not a total**, because a smoke run executes the same completeness assertion as a
    full one: a total drawn from the whole corpus can miss a benign class or a chain and abort at
    that assertion, which is a gate going red because of the sampling rather than because of the
    code.

    **Sorted by item id, which is content-derived**, so the sample is a function of the corpus and
    not of how the file was read. The same argument story 4.2 made about shard membership: a sample
    taken by row position would score a different set on a re-read, and two smoke runs would not be
    comparable.

    A group smaller than the size contributes all of it. That is not a special case to tolerate --
    it is the honest answer, and raising instead would make a smoke run's success depend on the
    corpus being large enough in every cell.
    """
    if items_per_cell < 1:
        raise ResultsIncomplete(
            f"a smoke sample of {items_per_cell} items per cell is not a sample; a run with no "
            f"rows passes every gate by having nothing left to fail one"
        )

    groups: dict[tuple[str, str, str], list[object]] = {}
    for item in items:
        groups.setdefault(_sample_group(item), []).append(item)

    sampled: list[object] = []
    for _, members in sorted(groups.items()):
        members.sort(key=lambda row: str(getattr(row, "id", "")))
        sampled.extend(members[:items_per_cell])
    return tuple(sampled)


def _sample_group(item: object) -> tuple[str, str, str]:
    """What the completeness assertion demands be represented: a family, a class, and a chain.

    The chain comes off the item id through `matrix.parse_item_id`, the declared inverse, rather
    than off a field -- a `CorpusItem` carries its dressing, but reading it here would make the
    sampler and the aggregator disagree the day one of them changed which they trusted.
    """
    _, chain = parse_item_id(str(getattr(item, "id", "")))
    return (
        str(getattr(item, "family", "")),
        str(getattr(item, "benign_class", "")),
        render_chain(chain),
    )


class ResultsIncomplete(NbcError, exit_code=34):
    """The run cannot publish a table, and the reason is that something is missing rather than wrong.

    Code 34 because 3 through 33 are taken. The inputs that produce it, each with the test that
    fires it:

    - a `baseline x chain x canon_on` key with no attack recall, or with a false-positive rate for
      one benign class and not the other, or with no AUC against a class;
    - a cell set with no held-out chain class at all, because a table that tested no held-out chain
      has not tested generalization and N4's answer would be about nothing;
    - a baseline declaring a publisher window protocol with only one policy measured;
    - a corpus that is partially present, where a rebuild would write half-new rows against a
      manifest describing the old ones;
    - the `report` step, until story 5.1 lands the renderer.
    """


@dataclass(frozen=True, slots=True)
class RunBlock:
    """Everything about how a run was configured, in one place because the rule says one place.

    A free mapping rather than a field per parameter, and that is deliberate: the parameters this
    project mandates recording are declared by a dozen decisions across four epics, and a dataclass
    enumerating them here would be a second declaration that drifts from the first. What is
    enforced is that the block is present, non-empty, and JSON-serialisable -- and that it is where
    every parameter goes, which `ResultsFile` checks by having nowhere else to put one.
    """

    fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping) or not self.fields:
            raise ValueError("the run block carries every recorded parameter and is never empty")
        for key in self.fields:
            if not isinstance(key, str) or not key:
                raise ValueError(f"run block keys are non-empty strings, got {key!r}")
        object.__setattr__(self, "fields", dict(self.fields))

    def as_json_object(self) -> dict[str, object]:
        return dict(self.fields)


@dataclass(frozen=True, slots=True)
class ResultsFile:
    """The published file: four keys, and no way to add a fifth.

    The key set is checked at construction rather than at serialisation, because a fifth key added
    to a dataclass is a diff somebody reviews and a fifth key added to a dict at write time is not.
    """

    run: RunBlock
    cells: tuple[object, ...]
    verdict: tuple[Verdict, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"this build writes schema_version {SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if not isinstance(self.run, RunBlock):
            raise ValueError(f"run must be a RunBlock, got {self.run!r}")
        if not self.cells:
            raise ValueError("a results file with no cells is not a table")
        if not all(isinstance(v, Verdict) for v in self.verdict):
            raise ValueError("verdict is a tuple of Verdicts")

    def as_json_object(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "run": self.run.as_json_object(),
            "cells": [cell.as_json_object() for cell in self.cells],  # type: ignore[attr-defined]
            "verdict": [v.as_json_object() for v in self.verdict],
        }
        if tuple(payload) != RESULT_KEYS:
            raise ValueError(
                f"the results file's top level is exactly {RESULT_KEYS}, got {tuple(payload)}"
            )
        return payload


def render_results(results: ResultsFile) -> str:
    """The exact bytes of `results/results.json`: sorted keys are not used and the order is
    `RESULT_KEYS`, so a reader diffing two runs sees the same shape in the same sequence."""
    return json.dumps(results.as_json_object(), ensure_ascii=False, indent=2) + "\n"


# --- the assertion that a half-built table cannot ship -------------------------------------------------


def _published(key: CellKey) -> bool:
    """Whether a cell is one of the published estimates rather than a comparison or a companion."""
    return key.contrast is None and key.population == POPULATION_ALL


def completeness_problems(
    cells: Sequence[object],
    window_policies: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Every cell the table demands and does not have, named.

    `window_policies` maps each baseline key to the policies it must be measured under: one for a
    baseline declaring no publisher protocol, two for one that does. It is passed in rather than
    read from the pins here, because this module is pure and because the caller already holds them.

    Three demands, each with its own reason. **An attack recall** per key, because a column with no
    recall is a column about nothing. **A false-positive rate for each benign class**, because a
    class silently absent is as unreadable as one that was pooled -- and FR3.1 exists for that
    distinction. **An AUC per benign class**, because a table defensible only at the threshold has
    not answered "you only shifted the scores".
    """
    problems: list[str] = []

    recalls: set[tuple[str, ...]] = set()
    false_positives: dict[tuple[str, ...], set[str]] = {}
    aucs: dict[tuple[str, ...], set[str]] = {}
    chain_classes: set[str] = set()
    policies_seen: dict[str, set[str]] = {}

    for cell in cells:
        key = getattr(cell, "key", None)
        if not isinstance(key, CellKey):
            continue
        if key.chain_class is not None:
            chain_classes.add(key.chain_class)
        if key.baseline is not None and key.window_policy is not None:
            policies_seen.setdefault(key.baseline, set()).add(key.window_policy)

        if isinstance(cell, Rate) and _published(key):
            if key.family == FAMILY_ATTACK:
                recalls.add(_column(key))
            elif key.family == FAMILY_BENIGN and key.benign_class is not None:
                false_positives.setdefault(_column(key), set()).add(key.benign_class)
        elif isinstance(cell, Auc) and key.benign_class is not None:
            aucs.setdefault(_column(key), set()).add(key.benign_class)

    demanded = recalls | set(false_positives) | set(aucs)
    for column in sorted(demanded):
        rendered = "/".join(column)
        if column not in recalls:
            problems.append(f"{rendered}: no attack recall rate")
        missing_fpr = sorted(set(BENIGN_CLASSES) - false_positives.get(column, set()))
        if missing_fpr:
            problems.append(f"{rendered}: no false-positive rate for {missing_fpr}")
        missing_auc = sorted(set(BENIGN_CLASSES) - aucs.get(column, set()))
        if missing_auc:
            problems.append(f"{rendered}: no AUC against {missing_auc}")

    missing_classes = sorted(set(CHAIN_CLASSES_FOR_KEYS) - chain_classes)
    if missing_classes:
        problems.append(
            f"no cell has chain_class in {missing_classes}; a table that tested no held-out chain "
            f"has not tested generalization, and N4's answer would be about nothing"
        )

    for baseline, required in sorted(window_policies.items()):
        missing_policies = sorted(set(required) - policies_seen.get(baseline, set()))
        if missing_policies:
            problems.append(
                f"{baseline}: no cell under window policy {missing_policies}; the baseline "
                f"declares a publisher protocol, so both policies are part of its table"
            )

    return tuple(problems)


def _column(key: CellKey) -> tuple[str, ...]:
    """The `baseline x dressing_chain x canon_on` coordinate the assertion is stated over.

    `chain_class` and `window_policy` ride along because a cell cannot exist without them and a
    column that dropped them would demand one recall where the table has four.
    """
    return (
        str(key.baseline),
        str(key.dressing_chain),
        str(key.chain_class),
        str(key.window_policy),
        str(key.canon_on),
    )


def refuse_an_incomplete_table(problems: Iterable[str]) -> None:
    """Abort naming every missing cell at once.

    All of them, not the first: a reader who fixes one and re-runs an eighty-five-hour pass to find
    the second has been told the truth twice and helped once.
    """
    listed = tuple(problems)
    if not listed:
        return
    raise ResultsIncomplete(
        f"the table is missing {len(listed)} required cells, so this run publishes nothing. "
        + "; ".join(listed)
    )
