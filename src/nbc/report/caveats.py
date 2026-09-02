"""The honesty section is checked before any inference, or it is not a gate at all.

SC4 asks for a "what this does not show" section a skeptical reviewer would accept as honest,
and AD-16 makes step 1 of the entrypoint — before the corpus is built and long before a model
is loaded — verify that the section is present and non-empty. This module is that check.

Two failures it is written against, both of which have shipped in other repositories:

- **the section is missing and the table is not.** A credible-looking table with no statement of
  its limits is the failure AD-12 names, and a run that discovers it *after* injecting a fresh
  table has already published the thing it meant to refuse.
- **the section is present and thin.** FR19 says it plainly: a thin caveats section is worse than
  none, because it looks like diligence. "Present and non-empty" alone is satisfied by a heading
  and one sentence, so the check enumerates the caveats the PRD's FR19 owns —
  `1, 1b, 2, 3, 3b, 3c, 3d, 4, 5, 6, 7` — in that order, and requires each to carry a body. AD-12
  states the count as the list rather than as a number for exactly this reason: a rule asserting
  "eleven" passes a section missing four caveats and repeating four others.

Slot **8** is reserved and must be present as a reserved slot: it is the one caveat written after
the numbers exist (Story 5.3), and a section that has quietly dropped the placeholder has lost the
promise that something is still owed.

The section must also sit **outside** the `RESULTS` markers. Text between them is generated from
`results/results.json` by `report/readme.py` and replaced wholesale on every run; an honesty
section that drifted inside them would be deleted by the next injection, and the check would then
be verifying prose that a machine writes.

This module imports the standard library and `nbc.errors`, and nothing else — the check runs
before `onnxruntime` is imported, and a test asserts it leaves the runtime out of `sys.modules`.
Nothing here reads `results.json`, and nothing here writes to the README.

    python -m nbc.report.caveats [--readme README.md]

exits 0 with a JSON report of what it found, or with `CaveatsSectionMissing`'s exit code.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nbc.errors import NbcError

__all__ = [
    "ABSTRACT_END",
    "ABSTRACT_START",
    "CAVEATS_HEADING",
    "Caveat",
    "CaveatsReport",
    "CaveatsSectionMissing",
    "DEFAULT_README",
    "MIN_CAVEAT_BODY_CHARS",
    "REQUIRED_LABELS",
    "RESERVED_LABEL",
    "RESULTS_END",
    "RESULTS_START",
    "main",
    "verify_caveats",
    "verify_caveats_file",
]


class CaveatsSectionMissing(NbcError, exit_code=11):
    """The README's honesty section is absent, empty, incomplete, thin, or in the wrong place.

    One abort covers every one of those, because no automated caller needs to tell them apart —
    each is the same fact, that this run would publish numbers without the statement of what they
    do not show — while the message always names which one and where. Every failure found is
    collected before raising, so a section that is wrong in three ways is told all three times.

    The code is 11: 3 is the platform floor, 4-7 the pins, 8 the label mapping, 9 the inference
    session, 10 the window policy. `errors.py`'s own docstring reserves 11 for this abort.
    """

    def __init__(self, *failures: str) -> None:
        super().__init__(
            "the README's \"what this does not show\" section is not publishable:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
        self.failures: tuple[str, ...] = failures


# --- what the section has to contain -----------------------------------------------------------
#
# The algorithm's constants live in the module that owns the algorithm, and this is the only
# declaration of them. `REQUIRED_LABELS` is the PRD's FR19 enumeration, in the PRD's published
# order, and the order is checked as well as the membership: the numbering is what lets a reader
# move between the PRD and the README and land on the same caveat.

CAVEATS_HEADING: Final[str] = "## What this does not show"

REQUIRED_LABELS: Final[tuple[str, ...]] = (
    "1",
    "1b",
    "2",
    "3",
    "3b",
    "3c",
    "3d",
    "4",
    "5",
    "6",
    "7",
)

RESERVED_LABEL: Final[str] = "8"

MIN_CAVEAT_BODY_CHARS: Final[int] = 200
"""A floor under each caveat's body, in characters, with the reserved slot exempt.

Not a quality measure — nothing here can measure that — but a floor under the specific failure
FR19 warns about: eleven headings with a sentence each, which satisfies "present and non-empty"
and tells a reader nothing. The published caveats run from ~320 to ~1500 characters, so the floor
is well under the shortest of them and leaves room to tighten prose without tripping it.
"""

RESULTS_START: Final[str] = "<!-- RESULTS:START -->"
RESULTS_END: Final[str] = "<!-- RESULTS:END -->"

ABSTRACT_START: Final[str] = "<!-- ABSTRACT:START -->"
ABSTRACT_END: Final[str] = "<!-- ABSTRACT:END -->"
"""The abstract's markers, owned here beside the block's for the same reason the block's are:
one module declares every span a run writes into the README, so the two checkers that must tell
generated text from hand-written text (`readme.py`, `timed_read.py`) read the same constants
instead of each spelling its own."""

DEFAULT_README: Final[Path] = Path("README.md")

_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\*\*(\d+[a-z]?)\.", re.MULTILINE)
_NEXT_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^## ", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Caveat:
    """One caveat found in the section: its label, and how much text it carries.

    `body_chars` counts the whole block from its own `**label.**` up to the next caveat or the
    end of the section, stripped — the label included, since a caveat is not shorter for having
    a short label.
    """

    label: str
    body_chars: int


@dataclass(frozen=True, slots=True)
class CaveatsReport:
    """What the check found, for the `run` block of `results.json` and for a human at a terminal."""

    caveats: tuple[Caveat, ...]
    section_chars: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(caveat.label for caveat in self.caveats)

    def as_run_fields(self) -> dict[str, object]:
        return {
            "caveats_check": "ok",
            "caveats_labels": list(self.labels),
            "caveats_required": list(REQUIRED_LABELS),
            "caveats_reserved_slot": RESERVED_LABEL,
            "caveats_section_chars": self.section_chars,
        }


def _locate_markers(readme: str, failures: list[str]) -> tuple[int, int] | None:
    """The span of the generated block, or `None` when it cannot be located.

    A malformed marker pair is a failure rather than a shrug: without a located block the check
    cannot tell a hand-written section from one a run would overwrite, which is half of what it
    is for.
    """
    starts = [match.start() for match in re.finditer(re.escape(RESULTS_START), readme)]
    ends = [match.start() for match in re.finditer(re.escape(RESULTS_END), readme)]

    if len(starts) != 1 or len(ends) != 1:
        failures.append(
            f"the generated block is not delimited exactly once: found {len(starts)} "
            f"{RESULTS_START!r} and {len(ends)} {RESULTS_END!r}; the check cannot tell which "
            f"text a run would overwrite"
        )
        return None

    if ends[0] < starts[0]:
        failures.append(
            f"{RESULTS_END!r} appears before {RESULTS_START!r}; the generated block is inverted"
        )
        return None

    return starts[0], ends[0] + len(RESULTS_END)


def verify_caveats(readme: str) -> CaveatsReport:
    """Verify the honesty section in `readme`. Raises `CaveatsSectionMissing`, or returns what it found.

    Step 1 of the entrypoint's sequence (AD-16), alongside pin verification and before anything is
    built, measured or downloaded.
    """
    failures: list[str] = []

    span = _locate_markers(readme, failures)

    heading_matches = [
        match.start()
        for match in re.finditer(rf"^{re.escape(CAVEATS_HEADING)}[ \t]*$", readme, re.MULTILINE)
    ]
    if not heading_matches:
        failures.append(
            f"no {CAVEATS_HEADING!r} heading anywhere in the README; SC4's section is the one "
            f"part of this repository that a run must not publish numbers without"
        )
        raise CaveatsSectionMissing(*failures)
    if len(heading_matches) > 1:
        failures.append(
            f"{CAVEATS_HEADING!r} appears {len(heading_matches)} times; the section a reader "
            f"finds and the section this check reads must be the same one"
        )

    heading_at = heading_matches[0]
    if span is not None and span[0] <= heading_at < span[1]:
        failures.append(
            f"{CAVEATS_HEADING!r} sits inside the {RESULTS_START} / {RESULTS_END} block, whose "
            f"text is generated and replaced on every run; the section is hand-written and must "
            f"sit outside it"
        )

    body_start = heading_at + len(CAVEATS_HEADING)
    next_section = _NEXT_SECTION_PATTERN.search(readme, body_start)
    body_end = next_section.start() if next_section is not None else len(readme)
    section = readme[body_start:body_end]

    if not section.strip():
        failures.append(
            "the section is present but empty; an empty honesty section is the failure AD-16's "
            "pre-inference check exists to catch"
        )
        raise CaveatsSectionMissing(*failures)

    found: list[tuple[str, int, int]] = [
        (match.group(1), match.start(), match.end())
        for match in _LABEL_PATTERN.finditer(section)
    ]

    caveats: list[Caveat] = []
    for index, (label, start, _) in enumerate(found):
        next_start = found[index + 1][1] if index + 1 < len(found) else len(section)
        body = section[start:next_start].strip()
        caveats.append(Caveat(label=label, body_chars=len(body)))

    order = [caveat.label for caveat in caveats]

    duplicates = sorted({label for label in order if order.count(label) > 1})
    if duplicates:
        failures.append(
            f"caveat label(s) {', '.join(duplicates)} appear more than once; a repeated label "
            f"makes the PRD's numbering ambiguous"
        )

    missing = [label for label in REQUIRED_LABELS if label not in order]
    if missing:
        failures.append(
            f"caveat(s) {', '.join(missing)} are missing; FR19's enumeration is "
            f"{', '.join(REQUIRED_LABELS)} and the check requires the labels rather than a count, "
            f"because a rule asserting a count passes a section missing four of them"
        )

    present_required = [label for label in order if label in REQUIRED_LABELS]
    if not missing and not duplicates and tuple(present_required) != REQUIRED_LABELS:
        failures.append(
            f"the caveats are published in the order {', '.join(present_required)}, not the "
            f"PRD's {', '.join(REQUIRED_LABELS)}; the numbering is what lets a reader move "
            f"between the two documents"
        )

    if RESERVED_LABEL not in order:
        failures.append(
            f"the reserved slot {RESERVED_LABEL} is missing; it is what the run has yet to reveal, "
            f"and dropping the placeholder drops the promise that something is still owed"
        )
    elif present_required and order.index(RESERVED_LABEL) < order.index(present_required[-1]):
        failures.append(
            f"the reserved slot {RESERVED_LABEL} is published before caveat "
            f"{present_required[-1]}; it is the last slot, written after the numbers exist"
        )

    thin = [
        caveat
        for caveat in caveats
        if caveat.label in REQUIRED_LABELS and caveat.body_chars < MIN_CAVEAT_BODY_CHARS
    ]
    if thin:
        failures.append(
            "caveat(s) "
            + ", ".join(f"{caveat.label} ({caveat.body_chars} chars)" for caveat in thin)
            + f" carry less than {MIN_CAVEAT_BODY_CHARS} characters; a thin caveats section is "
            f"worse than none, because it looks like diligence"
        )

    if failures:
        raise CaveatsSectionMissing(*failures)

    return CaveatsReport(caveats=tuple(caveats), section_chars=len(section.strip()))


def verify_caveats_file(path: Path = DEFAULT_README) -> CaveatsReport:
    """`verify_caveats` over a file. An unreadable README is the same abort, not a crash."""
    try:
        readme = path.read_text(encoding="utf-8")
    except OSError as unreadable:
        raise CaveatsSectionMissing(
            f"{path} could not be read ({unreadable.strerror or unreadable}); the honesty "
            f"section cannot be verified and the run must not proceed"
        ) from unreadable
    return verify_caveats(readme)


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.report.caveats` — the honesty check, runnable on its own by CI and by a reader."""
    import argparse
    import json

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.report.caveats",
        description=(
            "Verify the README's \"what this does not show\" section is present, complete and "
            "outside the generated block. Step 1 of the run, before any inference."
        ),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        metavar="PATH",
        help="the README to check (default: %(default)s, relative to the working directory)",
    )
    args = parser.parse_args(argv)

    try:
        report = verify_caveats_file(args.readme)
    except CaveatsSectionMissing as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report.as_run_fields(), sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
