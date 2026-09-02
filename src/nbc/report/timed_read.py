"""SC1 is a claim about a human, so the gate is a human — and everything around the human is checked.

SC1 says a reader who has never seen this repository understands the question, the table and the
caveats from the README alone, in under five minutes. UX-DR1 makes that falsifiable: a **timed read
by someone who has never seen the repository**, who then answers three questions, one per noun of
SC1's own sentence — *what is the question*, *what does the table show*, *what do the caveats say* —
with a wrong answer on any one of them failing the criterion, and the README, never the criterion,
changing on a failure.

**Why this is not a CI gate, stated here rather than assumed.** `caveats.py` already aborts when the
honesty section is absent, thin or in the wrong place. That guarantees the section exists and
guarantees nothing about whether it is any good, which is FR19's own warning: a thin caveats section
is worse than none because it looks like diligence. No check in this repository can read the page and
know whether a stranger understood it. So the reading is done by a person, and what a machine can do
is everything else — hold the protocol, hold the record, refuse a record that has been quietly edited
to fit a result, and measure the page the reader is being handed.

**What is measured, stated so a reader can disagree with it.** The page load is what a reader meets
on one screen-scroll of the file, in words:

- fenced blocks are skipped. A reader passing a `mermaid` diagram or a shell command is not reading
  prose at that rate, and counting the fence contents as prose would inflate the number in the
  direction that flatters nobody.
- table rows are skipped, for the same reason and the opposite direction: a 197-row findings table
  is scanned, not read, and pricing it per word would drown every real sentence on the page.
- text inside a `<details>` fold is skipped and its `<summary>` is counted. The fold is what the
  reader actually meets; the fold's contents are what they meet only if they choose to.
- HTML comments are stripped over the **whole text** before any of that, so a comment spanning six
  lines costs nothing and a marker is not a word. Per line it would not work: `re.DOTALL` on a
  pattern applied one line at a time is an inert flag, which is exactly what this module shipped.
- the generated block between the `RESULTS` markers is counted **separately** from the hand-written
  prose, because only one of the two is anybody's to shorten. The headline figure is the
  **hand-written** half against the budget: pooling the two produces a ratio nobody can act on,
  since the generated half is not anybody's to cut.

**A measurement that cannot be trusted aborts instead of under-counting.** An unclosed fence, a
`<details>` that never closes, an inverted or duplicated `RESULTS` marker pair: each of those makes
the count silently *smaller* than the page, which is the one direction a page-load figure must never
be wrong in, because it is the direction that reports a page as being inside a budget it is not.

Fences close the way CommonMark says they close: a run of at least as many characters of the same
kind, alone on its line. A four-backtick block is not closed by the three-backtick line inside it,
which is the whole reason anybody opens a four-backtick block.

The words-per-minute figure is a **declared parameter, not a fact about a person**: readers differ by
more than the factor this page is over budget. It is declared once, below, and the report says what
it used, so a reader who thinks 250 is the wrong number can recompute rather than argue. Minutes are
a rounded projection of words at that rate, so the per-section minutes column need not add up to the
page's minutes to the tenth; **words are the column that reconciles exactly**, and the report prints
both for that reason.

**No word count is typed anywhere.** Not in this module, not in the record. The record cites the
figure this module printed, on a stated date, and that is the whole discipline: a repository whose
subject is claims that outrun their evidence does not get to put a hand-typed measurement beside a
checked one.

This module imports the standard library, `nbc.errors`, and `nbc.report.caveats` for the `RESULTS`
markers -- declared once, in the module that owns the honesty check, as `readme.py` already imports
them. It loads no inference runtime, reads no `results.json`, and writes to nothing.

    python -m nbc.report.timed_read [--readme README.md] [--record sc1-timed-read.md]

exits 0 with a JSON report of the page load and the record's state -- including `not yet run`, which
is the honest state of a human gate nobody has yet walked through and is not a failure of this check
-- or with `Sc1RecordUnusable`'s exit code, which the class below states and explains.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Final

from nbc.errors import NbcError
from nbc.report.caveats import RESULTS_END, RESULTS_START

__all__ = [
    "ANSWER_VERDICTS",
    "BUDGET_MINUTES",
    "CHANGES_END",
    "CHANGES_START",
    "CHANGE_COLUMNS",
    "DEFAULT_README",
    "DEFAULT_RECORD",
    "PREAMBLE_HEADING",
    "PageLoad",
    "QUESTIONS",
    "READS_END",
    "READS_START",
    "READ_COLUMNS",
    "ReadmeChange",
    "Sc1Record",
    "Sc1RecordUnusable",
    "SectionLoad",
    "TimedRead",
    "TimedReadAnswer",
    "TimedReadReport",
    "WORDS_PER_MINUTE",
    "main",
    "measure_page",
    "parse_record",
    "verify_timed_read",
    "verify_timed_read_files",
]


class Sc1RecordUnusable(NbcError, exit_code=37):
    """The SC1 record cannot be read as evidence, or the page behind it cannot be measured.

    One abort for both, because the consequence is one: this repository would be claiming SC1 was
    tested while holding nothing a later reader could check that against. The message always names
    which failure it is and which read it is about, and every failure found is collected before
    raising, so a record that is wrong in three places is told all three.

    A **recorded failure is not one of these.** A read that failed is the criterion working, and it
    exits 0 with the failure reported. What aborts is a failure with no consequence -- a wrong answer
    and no README change recorded after it -- because UX-DR1 says the page changes on a failure and
    the record is the only place that is visible.

    An unmeasurable README is here too. A record that cites no measurement is not evidence about a
    page, so a page that cannot be measured makes the record unusable in exactly the sense this
    abort is named for.

    The code is 37, the next free rung: 3 is the platform floor, 4-7 the pins, 8 the label mapping,
    9 the inference session, 10 the window policy, 11 the caveats section, 12 the vendored
    confusables table, 13 a stage contract, 14 the size budget, 15-26 and 28 the corpus, 27 and
    29-34 the harness, 35 the confirmatory cell the corpus manifest guards, 36 the rendered block.
    The ladder is spelt out by span rather than as one range because two of the codes sit outside
    the block their neighbours belong to, and a docstring that rounded that off would be telling a
    reader something `declared_exit_codes()` contradicts.
    """

    def __init__(self, *failures: str) -> None:
        super().__init__(
            "the SC1 timed-read record is not usable as evidence:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
        self.failures: Final[tuple[str, ...]] = tuple(failures)


# --- the criterion's own constants ---------------------------------------------------------------
#
# The three questions are enumerated and never counted, for the reason `caveats.py:84-89` gives
# about FR19's eleven: a rule asserting "three" passes a record that asks the same easy question
# three times, and a criterion that can be reworded after the answers are in is not a criterion.

QUESTIONS: Final[tuple[str, ...]] = (
    "What is the question",
    "What does the table show",
    "What do the caveats say",
)

BUDGET_MINUTES: Final[int] = 5
"""SC1's own number. A read at or over this is a failure however good the answers are."""

WORDS_PER_MINUTE: Final[int] = 250
"""The reading-rate convention this report uses. A parameter, not a measurement of anybody.

Silent prose reading in adults is usually put somewhere between 200 and 300 words per minute for
material like this, and 250 sits in the middle of that. It is stated here, and restated in every
report this module prints, so that the ratio it produces can be recomputed by a reader who prefers a
different number instead of being taken on trust. Nothing in this repository depends on 250 being
right; what depends on it is a figure that is currently wrong by a factor no plausible rate rescues.
"""

ANSWER_VERDICTS: Final[tuple[str, ...]] = ("correct", "wrong")
"""How a graded answer opens its cell. Two words, so the grading cannot be a shade of grey."""

DEFAULT_README: Final[Path] = Path("README.md")
DEFAULT_RECORD: Final[Path] = Path("sc1-timed-read.md")
"""Root-level and tracked. `docs/` is gitignored, and a record a later reader cannot see is not one."""

READS_START: Final[str] = "<!-- SC1-READS:START -->"
READS_END: Final[str] = "<!-- SC1-READS:END -->"
CHANGES_START: Final[str] = "<!-- SC1-CHANGES:START -->"
CHANGES_END: Final[str] = "<!-- SC1-CHANGES:END -->"
"""Markers rather than heading matching, for the reason `size_budget.py:174-181` gives.

A check that found its table by looking for an English heading would be reading a heading, and the
heading is not the record.
"""

READ_COLUMNS: Final[tuple[str, ...]] = ("Reader", "Date", "Elapsed", *QUESTIONS)
"""The reads table's header, question columns included, so renaming a question is a parse failure."""

CHANGE_COLUMNS: Final[tuple[str, ...]] = ("Date", "After read", "What changed")

_ISO_DAY: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""`YYYY-MM-DD` and nothing else.

`date.fromisoformat` alone is looser than the message beside it: it accepts `20260905` and ISO week
dates such as `2026-W36-6`. Both round-trip to a different string than the one written, and the
record's own cross-table check compares `After read` as **text** -- so a record spelling one date
compactly in both tables would abort saying the read is not in the reads table, which is a true
statement about the wrong thing. The shape is checked before the calendar so the message and the
rule are the same rule.
"""

_ELAPSED: Final[re.Pattern[str]] = re.compile(r"^(\d{1,3}):([0-5]\d)$")
_ANSWER: Final[re.Pattern[str]] = re.compile(
    r"^(" + "|".join(re.escape(verdict) for verdict in ANSWER_VERDICTS) + r"):\s*(\S.*)$"
)
"""Built from `ANSWER_VERDICTS` rather than repeating it.

Two declarations of the same closed vocabulary is one of them going stale: adding a verdict to the
constant and not to the pattern would leave a grade the record declares and the parser refuses.
"""

_CELL_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<!\\)\|")
_SEPARATOR_ROW: Final[re.Pattern[str]] = re.compile(r"^[\s:|-]+$")
_FENCE: Final[re.Pattern[str]] = re.compile(r"^(`{3,}|~{3,})(.*)$")
_HEADING: Final[re.Pattern[str]] = re.compile(r"^(#{1,2})\s+(.*\S)\s*$")
_HTML_COMMENT: Final[re.Pattern[str]] = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]*>")

PREAMBLE_HEADING: Final[str] = "(before the first heading)"
"""What a section row is called when the prose it holds arrives before any heading.

Without it the words are counted into the page total and into no section, and the invariant the
report is read by -- the sections add up to the hand-written half -- quietly stops holding. It is
created only when there is prose above the first heading, so a page that starts with its title does
not grow an empty row.
"""


# --- measuring the page --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionLoad:
    """One hand-written section of the page and what it costs a reader, in words."""

    heading: str
    words: int

    @property
    def minutes(self) -> float:
        return round(self.words / WORDS_PER_MINUTE, 1)


@dataclass(frozen=True, slots=True)
class PageLoad:
    """What a first-time reader meets on the page, split into the half anybody can shorten and the half nobody can."""

    hand_written_words: int
    generated_words: int
    sections: tuple[SectionLoad, ...]
    lines: int

    @property
    def total_words(self) -> int:
        return self.hand_written_words + self.generated_words

    @property
    def minutes(self) -> float:
        return round(self.total_words / WORDS_PER_MINUTE, 1)

    @property
    def hand_written_minutes(self) -> float:
        return round(self.hand_written_words / WORDS_PER_MINUTE, 1)

    @property
    def generated_minutes(self) -> float:
        """Reported because it is the half a reader still has to scroll past, and nobody's to cut."""
        return round(self.generated_words / WORDS_PER_MINUTE, 1)

    @property
    def budget_words(self) -> int:
        """What five minutes buys at the declared rate. The number the page is measured against."""
        return BUDGET_MINUTES * WORDS_PER_MINUTE

    @property
    def over_budget(self) -> float:
        """How many times over the budget the whole page is, generated block included."""
        return round(self.total_words / self.budget_words, 2)

    @property
    def over_budget_hand_written(self) -> float:
        """The actionable factor: the half somebody could shorten, against the budget.

        The headline. Pooling the two halves reports a page as further over budget than anything a
        person could do about, because the generated block is a pure function of `results.json` and
        is not anybody's to cut -- so a total-only figure names a gap and hides where it is.
        """
        return round(self.hand_written_words / self.budget_words, 2)

    def as_json(self) -> dict[str, object]:
        return {
            "lines": self.lines,
            "words_hand_written": self.hand_written_words,
            "words_generated_block": self.generated_words,
            "words_total": self.total_words,
            "words_per_minute": WORDS_PER_MINUTE,
            "minutes_hand_written": self.hand_written_minutes,
            "minutes_generated_block": self.generated_minutes,
            "minutes": self.minutes,
            "budget_minutes": BUDGET_MINUTES,
            "budget_words": self.budget_words,
            "over_budget_factor_hand_written": self.over_budget_hand_written,
            "over_budget_factor_total": self.over_budget,
            "sections": [
                {"heading": section.heading, "words": section.words, "minutes": section.minutes}
                for section in self.sections
            ],
        }


def _prose_words(line: str) -> int:
    """The words a reader meets on one line, with markup removed and nothing else counted.

    HTML comments are already gone by the time a line reaches here -- they are stripped over the
    whole text, because a comment is not a per-line thing and a `re.DOTALL` pattern applied one line
    at a time is a flag that does nothing.
    """
    return len(_HTML_TAG.sub(" ", line).split())


def _without_comments(readme: str) -> str:
    """`readme` with every HTML comment blanked, line count preserved.

    Blanked rather than deleted so that line indices -- which is how the generated block's span is
    expressed -- still point at the same lines. A six-line comment becomes six empty lines and
    contributes nothing, where the per-line strip this module shipped counted five of them as prose.
    """
    return _HTML_COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), readme)


def _locate_block(readme: str, failures: list[str]) -> tuple[int, int] | None:
    """The generated block's line span, or `None` with the reason appended.

    Line indices rather than character offsets, because the measurement walks lines. A malformed
    marker pair is a failure and not a shrug, for the reason `caveats.py:161-167` gives: without a
    located block the two halves of the page cannot be told apart, and the split is half the point
    of measuring at all.
    """
    lines = readme.splitlines()
    starts = [index for index, line in enumerate(lines) if RESULTS_START in line]
    ends = [index for index, line in enumerate(lines) if RESULTS_END in line]

    if len(starts) != 1 or len(ends) != 1:
        failures.append(
            f"the generated block is not delimited exactly once: found {len(starts)} "
            f"{RESULTS_START!r} and {len(ends)} {RESULTS_END!r}; the page load cannot be split "
            f"into the half a person wrote and the half a run writes"
        )
        return None
    if ends[0] < starts[0]:
        failures.append(
            f"{RESULTS_END!r} appears before {RESULTS_START!r} in the README; the generated block "
            f"is inverted and the split would attribute each half to the other"
        )
        return None
    return starts[0], ends[0]


def measure_page(readme: str) -> PageLoad:
    """Measure the reading load of `readme`, or raise `Sc1RecordUnusable` saying why it cannot.

    It raises rather than returning a best effort, and the caller collecting other failures catches
    it -- an earlier draft took a `failures` list and returned a degenerate measurement nobody could
    observe, because every caller that passed the list went on to raise before returning the report.
    A page-load figure that is wrong in the small direction is the one failure this module must not
    have: it reports a page as inside a budget it is outside.
    """
    failures: list[str] = []
    span = _locate_block(readme, failures)
    if span is None:
        raise Sc1RecordUnusable(*failures)

    block_start, block_end = span
    # `lines` is the file's own line count, reported as it stands; the walk is over the same lines
    # with comments blanked, which preserves the numbering so the two cannot drift apart.
    line_count = len(readme.splitlines())
    lines = _without_comments(readme).splitlines()

    hand_written = 0
    generated = 0
    fence: tuple[str, int] | None = None
    fence_opened_at: int = 0
    folds = 0
    fold_opened_at: int = 0
    current: int | None = None
    section_names: list[str] = []
    section_words: list[int] = []

    def open_section(name: str) -> int:
        section_names.append(name)
        section_words.append(0)
        return len(section_names) - 1

    for index, line in enumerate(lines):
        text = line.strip()
        in_block = block_start <= index <= block_end

        marker = _FENCE.match(text)
        if marker is not None:
            run, rest = marker.group(1), marker.group(2)
            if fence is None:
                # An info string (```mermaid) may only sit on an opening fence, so a line carrying
                # one is never a close, whatever is open.
                fence = (run[0], len(run))
                fence_opened_at = index + 1
                continue
            character, length = fence
            # CommonMark: the closer is the same character, at least as long, and alone on its
            # line. A three-backtick line inside a four-backtick block is content, which is the
            # entire reason a block that contains fences is opened with four.
            if run[0] == character and len(run) >= length and not rest.strip():
                fence = None
            continue
        if fence is not None:
            continue

        # A fold is met; what is folded is not. `<summary>` is the line the reader actually sees,
        # so the summary counts and the body does not.
        if "<details" in text:
            if folds == 0:
                fold_opened_at = index + 1
            folds += 1
        if "</details" in text:
            folds = max(0, folds - 1)
            continue
        if folds > 0 and "<summary" not in text:
            continue

        # Headings are read after the fold, not before it. A heading a reader only sees by opening
        # a fold is not a section they met, and registering it there did two wrong things at once:
        # it published a section with no words in it, and it captured every hand-written word after
        # the fold closed, filing prose under a heading nobody had scrolled past.
        if not in_block:
            titled = _HEADING.match(line)
            if titled is not None:
                current = open_section(titled.group(0).strip())

        if text.startswith("|"):
            continue

        words = _prose_words(text)
        if not words:
            continue
        if in_block:
            generated += words
        else:
            hand_written += words
            if current is None:
                current = open_section(PREAMBLE_HEADING)
            section_words[current] += words

    if fence is not None:
        failures.append(
            f"the fenced block opened on line {fence_opened_at} of the README is never closed, so "
            f"every word after it was dropped from the count; a page load that is silently short is "
            f"worse than one that refuses to be measured, because it reports the page as inside a "
            f"budget it is outside"
        )
    if folds:
        failures.append(
            f"the `<details>` fold opened on line {fold_opened_at} of the README is never closed, "
            f"so the rest of the page was counted as folded away and dropped from the count"
        )
    if failures:
        raise Sc1RecordUnusable(*failures)

    # Every section the walk opened is reported, including a short one. There is no empty-section
    # filter: a section carries at least the words of its own heading line, so a filter on the count
    # could never drop an empty row and could only ever drop a real one -- which is what it did to
    # the preamble bucket before the bucket existed.
    sections = tuple(
        SectionLoad(heading=name, words=count)
        for name, count in zip(section_names, section_words)
    )
    return PageLoad(
        hand_written_words=hand_written,
        generated_words=generated,
        sections=sections,
        lines=line_count,
    )


# --- reading the record --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimedReadAnswer:
    """One graded answer: which question, whether it was right, and what the reader actually said."""

    question: str
    verdict: str
    text: str

    @property
    def correct(self) -> bool:
        return self.verdict == ANSWER_VERDICTS[0]


@dataclass(frozen=True, slots=True)
class TimedRead:
    """One person, one clock, three answers."""

    reader: str
    day: str
    elapsed: str
    elapsed_seconds: int
    answers: tuple[TimedReadAnswer, ...]

    @property
    def identity(self) -> str:
        """How a README change names the read it followed. Reader and date, which is what a person has."""
        return f"{self.reader} {self.day}"

    @property
    def within_budget(self) -> bool:
        return self.elapsed_seconds < BUDGET_MINUTES * 60

    @property
    def passed(self) -> bool:
        """Under five minutes and right on all three. Anything else is a failure of the criterion."""
        return self.within_budget and all(answer.correct for answer in self.answers)

    def as_json(self) -> dict[str, object]:
        return {
            "reader": self.reader,
            "date": self.day,
            "elapsed": self.elapsed,
            "elapsed_seconds": self.elapsed_seconds,
            "within_budget": self.within_budget,
            "outcome": "pass" if self.passed else "fail",
            "answers": [
                {"question": answer.question, "verdict": answer.verdict, "answer": answer.text}
                for answer in self.answers
            ],
        }


@dataclass(frozen=True, slots=True)
class ReadmeChange:
    """A change made to the README because a read failed. The consequence, written down."""

    day: str
    after_read: str
    what: str

    def as_json(self) -> dict[str, str]:
        return {"date": self.day, "after_read": self.after_read, "what_changed": self.what}


@dataclass(frozen=True, slots=True)
class Sc1Record:
    """The record as parsed: every read taken, and every README change a failed read produced."""

    reads: tuple[TimedRead, ...]
    changes: tuple[ReadmeChange, ...]

    @property
    def latest(self) -> TimedRead | None:
        """The most recent read **by date**, not the bottom row of the table.

        Rows are appended by hand, and a hand-appended row is not an ordering. Reading the outcome
        off the last row means a record whose rows were pasted out of order reports the wrong
        verdict for SC1, and nothing about the file would look wrong. Dates are normalised to
        `YYYY-MM-DD` at parse time, so lexicographic order is chronological order; the row position
        breaks a tie, which is the only thing it is trusted for.
        """
        if not self.reads:
            return None
        return max(enumerate(self.reads), key=lambda pair: (pair[1].day, pair[0]))[1]

    @property
    def status(self) -> str:
        """`not yet run` until somebody reads the page; after that, the most recent read's outcome."""
        latest = self.latest
        if latest is None:
            return "not yet run"
        return "pass" if latest.passed else "fail"

    def as_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "questions": list(QUESTIONS),
            "reads": [read.as_json() for read in self.reads],
            "readme_changes": [change.as_json() for change in self.changes],
        }


@dataclass(frozen=True, slots=True)
class TimedReadReport:
    """What the check found: the page a reader is handed, and the state of the human gate over it."""

    page: PageLoad
    record: Sc1Record

    def as_json(self) -> dict[str, object]:
        return {"page": self.page.as_json(), "record": self.record.as_json()}


def _block(text: str, start: str, end: str, label: str, failures: list[str]) -> str | None:
    """The text between one marker pair, or `None` with the reason appended."""
    starts = text.count(start)
    ends = text.count(end)
    if starts != 1 or ends != 1:
        failures.append(
            f"the {label} block is not delimited exactly once: found {starts} {start!r} and "
            f"{ends} {end!r}; the checker cannot tell which rows are the record"
        )
        return None
    opened = text.index(start)
    closed = text.index(end)
    if closed < opened:
        failures.append(f"the {label} block's {end!r} comes before its {start!r}")
        return None
    return text[opened + len(start) : closed]


def _rows(block: str) -> list[list[str]]:
    """The pipe-table rows inside `block`, header included, separator dropped.

    A `|` inside a verbatim answer is written `\\|` and comes back unescaped, so a reader quoting a
    command with a pipe in it does not silently gain a column.
    """
    rows: list[list[str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _SEPARATOR_ROW.match(stripped):
            continue
        cells = _CELL_SPLIT.split(stripped)
        # A pipe table opens and closes with a delimiter, so the split yields an empty cell at each
        # end. Dropping them by position rather than by emptiness keeps a genuinely empty first
        # column visible as the missing field it is.
        rows.append([cell.replace("\\|", "|").strip() for cell in cells[1:-1]])
    return rows


def _check_header(
    rows: list[list[str]], columns: tuple[str, ...], label: str, why: str, failures: list[str]
) -> bool:
    """The header is the contract. A renamed column is the criterion being edited to fit a result.

    An empty block is the same failure and not a lesser one: markers with no table under them is
    what a record edited down to fit a result looks like from here, and a check that shrugged at it
    would report `not yet run` over a file somebody had emptied.
    """
    if not rows:
        failures.append(
            f"the {label} block holds no table at all; the record needs its header even when no "
            f"row has been written under it, and markers with nothing between them is a record "
            f"that has been emptied rather than one nobody has written in"
        )
        return False
    header = tuple(rows[0])
    if header != columns:
        failures.append(
            f"the {label} table's columns are {', '.join(header) or '(none)'}, not "
            f"{', '.join(columns)}; {why}"
        )
        return False
    return True


_READS_HEADER_WHY: Final[str] = (
    "the three questions are enumerated and are not the record's to rename -- a criterion that "
    "can be reworded after the answers are in is not one"
)
_CHANGES_HEADER_WHY: Final[str] = (
    "the consequence of a failed read is recorded under these three columns and nothing checks a "
    "column it cannot find, so a renamed header is a failure that quietly stops being answerable"
)


def _parse_day(value: str, where: str, failures: list[str]) -> str | None:
    """`YYYY-MM-DD`, spelt exactly that way, and a real calendar date.

    Both halves are needed. The shape alone would accept `2026-13-45`; `date.fromisoformat` alone
    accepts `20260905` and `2026-W36-6`, neither of which is what the message says and both of which
    normalise to a string the record's own `After read` column would then fail to match.
    """
    if _ISO_DAY.match(value) is None or _from_iso(value) is None:
        failures.append(
            f"{where}: the date {value!r} is not an ISO date (YYYY-MM-DD); a read whose date "
            f"cannot be ordered cannot be told from the README change that answered it"
        )
        return None
    return value


def _from_iso(value: str) -> Date | None:
    try:
        return Date.fromisoformat(value)
    except ValueError:
        return None


def _parse_read(cells: list[str], number: int, failures: list[str]) -> TimedRead | None:
    """One row of the reads table, with every missing field named against the read it is missing from."""
    where = f"read {number}"
    if len(cells) != len(READ_COLUMNS):
        failures.append(
            f"{where} has {len(cells)} cells against the {len(READ_COLUMNS)} the header declares "
            f"({', '.join(READ_COLUMNS)}); a read with two answers instead of three is a read "
            f"missing a field, not a shorter criterion"
        )
        return None

    reader, day, elapsed, *answers = cells
    where = f"read {number} ({reader or 'unnamed reader'})"

    ok = True
    if not reader:
        failures.append(f"{where}: no reader; an unattributed read is not evidence a person read it")
        ok = False

    parsed_day: str | None = None
    if not day:
        failures.append(f"{where}: no date")
        ok = False
    else:
        parsed_day = _parse_day(day, where, failures)
        ok = ok and parsed_day is not None

    seconds: int | None = None
    if not elapsed:
        failures.append(f"{where}: no elapsed time; the read is timed or it tests nothing")
        ok = False
    else:
        match = _ELAPSED.match(elapsed)
        if match is None:
            failures.append(
                f"{where}: the elapsed time {elapsed!r} does not parse as M:SS; a duration nothing "
                f"can compare to five minutes decides nothing"
            )
            ok = False
        elif int(match.group(1)) * 60 + int(match.group(2)) == 0:
            failures.append(
                f"{where}: the elapsed time is {elapsed!r}; a read that took no time is not a "
                f"timed read, and recording one as a pass would be the criterion certifying a "
                f"reading nobody did"
            )
            ok = False
        else:
            seconds = int(match.group(1)) * 60 + int(match.group(2))

    graded: list[TimedReadAnswer] = []
    for question, cell in zip(QUESTIONS, answers):
        if not cell:
            failures.append(
                f"{where}: nothing recorded for {question!r}; all three are answered or the "
                f"criterion was not applied"
            )
            ok = False
            continue
        match = _ANSWER.match(cell)
        if match is None:
            failures.append(
                f"{where}: the answer to {question!r} reads {cell!r}, which does not open with "
                f"{' or '.join(f'{verdict}:' for verdict in ANSWER_VERDICTS)} followed by what the "
                f"reader said; a grade with no answer under it cannot be disagreed with"
            )
            ok = False
            continue
        graded.append(
            TimedReadAnswer(question=question, verdict=match.group(1), text=match.group(2).strip())
        )

    if not ok or parsed_day is None or seconds is None or len(graded) != len(QUESTIONS):
        return None
    return TimedRead(
        reader=reader,
        day=parsed_day,
        elapsed=elapsed,
        elapsed_seconds=seconds,
        answers=tuple(graded),
    )


def _parse_change(cells: list[str], number: int, failures: list[str]) -> ReadmeChange | None:
    where = f"README change {number}"
    if len(cells) != len(CHANGE_COLUMNS):
        failures.append(
            f"{where} has {len(cells)} cells against the {len(CHANGE_COLUMNS)} the header declares "
            f"({', '.join(CHANGE_COLUMNS)})"
        )
        return None
    day, after, what = cells
    ok = True
    parsed_day: str | None = None
    if not day:
        failures.append(f"{where}: no date")
        ok = False
    else:
        parsed_day = _parse_day(day, where, failures)
        ok = ok and parsed_day is not None
    if not after:
        failures.append(
            f"{where}: no read named; a change that names no failed read cannot answer one"
        )
        ok = False
    if not what:
        failures.append(f"{where}: no description of what changed")
        ok = False
    if not ok or parsed_day is None:
        return None
    return ReadmeChange(day=parsed_day, after_read=after, what=what)


def parse_record(record: str, failures: list[str]) -> Sc1Record:
    """Parse the record's two tables, appending every reason it is unusable to `failures`."""
    reads: list[TimedRead] = []
    changes: list[ReadmeChange] = []

    reads_block = _block(record, READS_START, READS_END, "reads", failures)
    if reads_block is not None:
        rows = _rows(reads_block)
        if _check_header(rows, READ_COLUMNS, "reads", _READS_HEADER_WHY, failures):
            for number, cells in enumerate(rows[1:], start=1):
                read = _parse_read(cells, number, failures)
                if read is not None:
                    reads.append(read)

    changes_block = _block(record, CHANGES_START, CHANGES_END, "README changes", failures)
    if changes_block is not None:
        rows = _rows(changes_block)
        if _check_header(rows, CHANGE_COLUMNS, "README changes", _CHANGES_HEADER_WHY, failures):
            for number, cells in enumerate(rows[1:], start=1):
                change = _parse_change(cells, number, failures)
                if change is not None:
                    changes.append(change)

    seen: dict[str, int] = {}
    for read in reads:
        seen[read.identity] = seen.get(read.identity, 0) + 1
    for identity, count in seen.items():
        if count > 1:
            failures.append(
                f"{identity} appears as {count} separate reads; a README change names the read it "
                f"followed by reader and date, and two reads sharing both cannot be told apart"
            )

    # And the protocol's own rule, which reader-and-date cannot express: a read is never repeated
    # with the same person, on any date. They have seen the page now, so the second read is not a
    # first-time reader's and measures something SC1 does not claim.
    readers: dict[str, list[str]] = {}
    for read in reads:
        readers.setdefault(read.reader, []).append(read.day)
    for reader, days in readers.items():
        if len(days) > 1:
            failures.append(
                f"{reader} appears in {len(days)} reads ({', '.join(sorted(days))}); SC1 is about a "
                f"reader who has never seen the page, and the protocol says a read is never "
                f"repeated with the same person -- the next read is a new person against the "
                f"changed page"
            )

    answered = {change.after_read for change in changes}
    for read in reads:
        if read.passed:
            continue
        if read.identity not in answered:
            wrong = [answer.question for answer in read.answers if not answer.correct]
            reason = (
                f"answered {', '.join(repr(question) for question in wrong)} wrong"
                if wrong
                else f"took {read.elapsed}, which is not under {BUDGET_MINUTES}:00"
            )
            failures.append(
                f"the read by {read.identity} failed -- it {reason} -- and no README change is "
                f"recorded after it; UX-DR1 says the page changes on a failure and never the "
                f"criterion, and this record is the only place that is visible"
            )

    by_identity = {read.identity: read for read in reads}
    for change in changes:
        answered_read = by_identity.get(change.after_read)
        if answered_read is None:
            failures.append(
                f"the README change dated {change.day} names the read {change.after_read!r}, which "
                f"is not in the reads table; a consequence recorded against a read nobody took "
                f"cannot be checked against anything"
            )
        elif change.day < answered_read.day:
            failures.append(
                f"the README change dated {change.day} claims to answer the read by "
                f"{change.after_read}, which happened later, on {answered_read.day}; a change made "
                f"before the read it answers is not a consequence of it, and a record where the "
                f"ordering does not hold cannot show that the page changed because a reader failed"
            )

    return Sc1Record(reads=tuple(reads), changes=tuple(changes))


def verify_timed_read(readme: str, record: str) -> TimedReadReport:
    """Measure the page and verify the record, collecting every failure before raising once.

    The measurement raises on its own, and its failures are folded in here rather than being
    collected through an out-parameter: a caller that has already found the page unmeasurable has
    no use for a page measurement, and the alternative was a fallback value no caller could observe.
    """
    failures: list[str] = []
    page: PageLoad | None = None
    try:
        page = measure_page(readme)
    except Sc1RecordUnusable as unmeasurable:
        failures.extend(unmeasurable.failures)
    parsed = parse_record(record, failures)
    if failures or page is None:
        raise Sc1RecordUnusable(*failures)
    return TimedReadReport(page=page, record=parsed)


def verify_timed_read_files(
    readme_path: Path = DEFAULT_README, record_path: Path = DEFAULT_RECORD
) -> TimedReadReport:
    """`verify_timed_read` over the working tree. A file that cannot be read is the same abort.

    `UnicodeDecodeError` is a `ValueError` and not an `OSError`, so both are caught: a record saved
    in some other encoding is unusable in exactly the sense this abort is named for, and letting it
    surface as an unclassified exit 1 would hide that from every caller.
    """
    failures: list[str] = []
    readme = ""
    record = ""
    for path, label in ((readme_path, "README"), (record_path, "record")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as unreadable:
            failures.append(
                f"the SC1 {label} {path} could not be read ({unreadable}); SC1 cannot be reported "
                f"as tested against a file nothing can open"
            )
            continue
        if label == "README":
            readme = text
        else:
            record = text
    if failures:
        raise Sc1RecordUnusable(*failures)
    return verify_timed_read(readme, record)


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.report.timed_read` — the page load, and where the human gate stands.

    Exit 0 with `not yet run` is the correct report for a gate nobody has walked through yet. This
    check cannot manufacture a reader, and a checker that failed for the absence of one would be
    telling a maintainer to do the one thing the criterion forbids.
    """
    import argparse
    import json

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.report.timed_read",
        description=(
            "Measure the README's reading load and report the state of SC1's timed read. "
            "The reading is a person's; everything around it is checked here."
        ),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        metavar="PATH",
        help="the page the reader is handed (default: %(default)s)",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD,
        metavar="PATH",
        help="the timed-read record (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        report = verify_timed_read_files(args.readme, args.record)
    except Sc1RecordUnusable as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report.as_json(), sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
