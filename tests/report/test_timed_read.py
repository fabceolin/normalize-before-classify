"""The gate around the human gate, proved by breaking the record every way it can break.

The repository's own `README.md` and `sc1-timed-read.md` are the happy path and are checked as they
ship. Every failure is exercised against a synthetic page and a synthetic record, because the
interesting cases are the ones the real files must never be in -- and because one of them, a
recorded failure with no README change after it, is a state this repository is *supposed* to reach
one day and must abort on when it does.

**Both halves of the module get red inputs.** The record's failure modes were enumerated first and
the measurement's were not, which is how a module that silently dropped every word after an unclosed
fence shipped green. A measurement is a claim about a file, so each way of mis-measuring the file --
an unclosed fence, a fence closed by a shorter run, a multi-line comment, prose above the first
heading, two sections sharing a heading, a heading inside the generated block -- is its own case
here, against a fixture of known length rather than against the page whose number is in dispute.

**A test's name is an assertion.** Anything named for a property asserts that property: tracking is
asserted with `git ls-files`, which observes tracking, and not with `git check-ignore`, which exits
non-zero for a file no clone will ever see.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from nbc.errors import EXIT_OK, NbcError, exit_code_for
from nbc.report.caveats import RESULTS_END, RESULTS_START
from nbc.report.timed_read import (
    ANSWER_VERDICTS,
    BUDGET_MINUTES,
    CHANGE_COLUMNS,
    CHANGES_END,
    CHANGES_START,
    DEFAULT_RECORD,
    PREAMBLE_HEADING,
    QUESTIONS,
    READ_COLUMNS,
    READS_END,
    READS_START,
    WORDS_PER_MINUTE,
    Sc1RecordUnusable,
    measure_page,
    verify_timed_read,
    verify_timed_read_files,
)

# --- fixtures: a page of known length, and a record of known shape ------------------------------

WORD = "alpha"


def words(count: int) -> str:
    """`count` words of prose, so a measurement can be compared to a number and not to itself."""
    return " ".join([WORD] * count)


def page(*, hand: int = 100, block: int = 40, extra: str = "", above: str = "") -> str:
    """A page whose two halves are exactly `hand` and `block` words of prose, plus `extra`.

    `above` goes before everything, which is where the generated block's markers are not: it is how
    a case about text preceding the first heading is written without moving the markers.
    """
    return (
        f"{above}"
        f"{words(hand)}\n"
        f"{extra}\n"
        f"{RESULTS_START}\n"
        f"{words(block)}\n"
        f"{RESULTS_END}\n"
    )


def table(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, rule, body]) if body else "\n".join([header, rule])


def answers(*verdicts: str) -> tuple[str, ...]:
    return tuple(
        f"{verdict}: what the reader said about {index}"
        for index, verdict in enumerate(verdicts)
    )


def read_row(
    reader: str = "FC",
    day: str = "2026-09-05",
    elapsed: str = "4:12",
    verdicts: tuple[str, ...] = ("correct", "correct", "correct"),
) -> tuple[str, ...]:
    return (reader, day, elapsed, *answers(*verdicts))


def record(
    reads: tuple[tuple[str, ...], ...] = (),
    changes: tuple[tuple[str, ...], ...] = (),
    read_columns: tuple[str, ...] = READ_COLUMNS,
    change_columns: tuple[str, ...] = CHANGE_COLUMNS,
    reads_body: str | None = None,
    changes_body: str | None = None,
) -> str:
    """The record as a string. `*_body` replaces a whole table, which is how "no table" is written."""
    between_reads = table(read_columns, reads) if reads_body is None else reads_body
    between_changes = table(change_columns, changes) if changes_body is None else changes_body
    return (
        "# SC1\n\n"
        f"{READS_START}\n\n{between_reads}\n\n{READS_END}\n\n"
        f"{CHANGES_START}\n\n{between_changes}\n\n{CHANGES_END}\n"
    )


def change_row(
    day: str = "2026-09-06", after: str = "FC 2026-09-05", what: str = "cut the third section"
) -> tuple[str, ...]:
    return (day, after, what)


# --- the files this repository actually ships ----------------------------------------------------


def test_the_shipped_record_parses_and_reports_not_yet_run(repo_root: Path) -> None:
    """The state the story lands in. `not yet run` is a report, not a failure of the check."""
    report = verify_timed_read_files(repo_root / "README.md", repo_root / DEFAULT_RECORD)
    assert report.record.status == "not yet run"
    assert report.record.reads == ()
    assert report.page.total_words > 0


TRACKED = (
    DEFAULT_RECORD,
    Path("src/nbc/report/timed_read.py"),
    Path("tests/report/test_timed_read.py"),
)


@pytest.mark.parametrize("path", TRACKED, ids=lambda path: path.name)
def test_the_record_and_its_module_are_tracked_by_git(repo_root: Path, path: Path) -> None:
    """Tracking is **observed**, with the command that fails on an untracked path.

    The check this replaces asked `git check-ignore` for a non-zero exit, which an untracked file
    produces too -- so it was green over three files that would not have reached a clone, including
    the record the criterion exists to leave for a later reader.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, f"{path} is not tracked by git; no clone would carry it"


def test_the_record_sits_at_the_root_and_not_under_the_gitignored_docs(repo_root: Path) -> None:
    """`docs/` is gitignored, and a record the later reader cannot see does not inform anyone."""
    assert (repo_root / DEFAULT_RECORD).is_file()
    assert DEFAULT_RECORD.parent == Path(".")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(DEFAULT_RECORD)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode != 0, "the SC1 record is gitignored; nobody downstream can read it"


def reads_header(text: str) -> tuple[str, ...]:
    """The reads table's own header row, read out of the record between its markers."""
    block = text.split(READS_START, 1)[1].split(READS_END, 1)[0]
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            return tuple(cell.strip() for cell in stripped.split("|")[1:-1])
    raise AssertionError("the reads block holds no table")


def test_the_shipped_record_enumerates_the_three_questions_as_its_columns(repo_root: Path) -> None:
    """The header row itself, not the strings appearing somewhere in the file.

    Asking whether each question appears anywhere passes a record whose table has been renamed
    wholesale as long as the prose above it still quotes the old names -- which is the edit the
    enumeration exists to refuse.
    """
    text = (repo_root / DEFAULT_RECORD).read_text(encoding="utf-8")
    assert reads_header(text) == READ_COLUMNS
    assert tuple(reads_header(text)[3:]) == QUESTIONS


def linking_passage(readme: str) -> str:
    """The one bullet or paragraph that links the record, reassembled from its wrapped lines."""
    lines = readme.splitlines()
    hits = [index for index, line in enumerate(lines) if str(DEFAULT_RECORD) in line]
    assert hits, "the README does not mention the SC1 record"
    start = hits[0]
    while start > 0 and lines[start].startswith(" ") and lines[start].strip():
        start -= 1
    end = hits[-1]
    while end + 1 < len(lines) and lines[end + 1].startswith(" ") and lines[end + 1].strip():
        end += 1
    passage = " ".join(line.strip() for line in lines[start : end + 1])
    assert all(start <= hit <= end for hit in hits), (
        "the record is linked from more than one place in the README"
    )
    return passage


SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

MAX_LINKING_WORDS = 50
"""One sentence, made countable. A budget stated in sentences is enforced by nothing."""


def test_the_readme_links_the_record_in_one_sentence(repo_root: Path) -> None:
    """The whole budget this gate may spend on the page it is gating.

    Counting the *lines* that mention the file passes a five-line paragraph as long as it wraps
    tightly. What is counted here is the reassembled passage: at most one sentence terminator, and
    if there is one it ends the passage rather than sitting in the middle of it with a second
    sentence after.
    """
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    passage = linking_passage(readme).rstrip()
    assert f"]({DEFAULT_RECORD})" in passage, "the record is named but not linked"
    terminators = list(SENTENCE_END.finditer(passage))
    assert len(terminators) <= 1, f"the link has grown to more than one sentence: {passage!r}"
    assert not terminators or terminators[0].end() >= len(passage), (
        f"a second sentence follows the linking one: {passage!r}"
    )
    assert len(passage.split()) <= MAX_LINKING_WORDS, (
        f"the linking sentence is {len(passage.split())} words; a gate that lengthens the page it "
        f"gates by more than a sentence has failed at its own job"
    )


def test_the_readme_names_the_check_the_way_it_names_its_siblings(repo_root: Path) -> None:
    """`caveats` and `size_budget` are each named on the page with a sentence saying what they do.

    A command a reader cannot find is a check that exists for CI and not for them, and this one is
    the check about the reader.
    """
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    for module in ("caveats", "size_budget", "timed_read"):
        assert f"python -m nbc.report.{module}" in readme, module


def claims_unwritten(text: str) -> set[str]:
    """Caveat labels the text describes as still owed rather than written."""
    pattern = re.compile(
        r"\b(?:slot|caveat)s?\s+(\d+[a-z]?)\b[^.]*?"
        r"\b(?:still reserved|is reserved|are reserved|not yet written|has yet to be|"
        r"remains? reserved|owes|awaits)\b",
        re.IGNORECASE,
    )
    return {match.group(1) for match in pattern.finditer(text)}


def test_the_status_section_does_not_claim_a_caveat_the_page_has_already_written(
    repo_root: Path,
) -> None:
    """The Status section is what a first-time reader reads as the project's own state.

    It said slot 8 was still reserved and "the next thing this page owes a reader" while slot 8 was
    written and on the page, which is the Status section being wrong about the one thing it is for.
    The scan is checked against that exact sentence first, so a test that could no longer see the
    claim it exists to catch fails here rather than passing quietly.
    """
    from nbc.report.caveats import MIN_CAVEAT_BODY_CHARS, verify_caveats

    stale = (
        "Slot 8 is still reserved: it is the one caveat that had to wait for numbers, the numbers "
        "now exist, and writing it is the next thing this page owes a reader"
    )
    assert claims_unwritten(stale) == {"8"}, "the scan cannot see the claim it exists to catch"

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    written = {
        caveat.label
        for caveat in verify_caveats(readme).caveats
        if caveat.body_chars >= MIN_CAVEAT_BODY_CHARS
    }
    status = readme.split("\n## Status\n", 1)[1].split("\n## ", 1)[0]
    claimed = claims_unwritten(status)
    assert not (claimed & written), (
        f"the Status section calls caveat(s) {sorted(claimed & written)} unwritten, and "
        f"caveats.py reports them present on the page with a full body"
    )


# --- the measurement ------------------------------------------------------------------------------


def test_the_page_load_is_measured_against_a_fixture_of_known_length() -> None:
    load = measure_page(page(hand=100, block=40))
    assert load.hand_written_words == 100
    assert load.generated_words == 40
    assert load.total_words == 140


def test_minutes_and_the_budget_come_from_the_declared_rate() -> None:
    """The rate is a parameter, so everything derived from it is recomputable rather than asserted."""
    load = measure_page(page(hand=WORDS_PER_MINUTE * 2, block=0))
    assert load.minutes == 2.0
    assert load.hand_written_minutes == 2.0
    assert load.budget_words == BUDGET_MINUTES * WORDS_PER_MINUTE


def test_the_headline_factor_is_the_half_a_person_could_shorten() -> None:
    """Pooling the two halves reports a gap larger than anything anybody can act on.

    The generated block is a pure function of `results.json`; nobody shortens it by editing prose.
    So the hand-written factor is reported separately, and it is the one the record cites.
    """
    budget = BUDGET_MINUTES * WORDS_PER_MINUTE
    load = measure_page(page(hand=budget * 2, block=budget))
    assert load.over_budget_hand_written == 2.0
    assert load.over_budget == 3.0
    assert load.generated_minutes == round(budget / WORDS_PER_MINUTE, 1)
    printed = load.as_json()
    assert printed["over_budget_factor_hand_written"] == 2.0
    assert printed["over_budget_factor_total"] == 3.0


def test_the_section_minutes_and_the_page_minutes_reconcile(repo_root: Path) -> None:
    """Words reconcile exactly; minutes reconcile to within the rounding the column itself does.

    Reading the per-section column and the page's own figure as if they must be equal to the tenth
    is what made 35.1 look like a contradiction of 35.0. They differ by accumulated rounding and by
    nothing else, and the bound on that is half a tenth per section.
    """
    load = measure_page((repo_root / "README.md").read_text(encoding="utf-8"))
    assert sum(section.words for section in load.sections) == load.hand_written_words
    drift = abs(sum(section.minutes for section in load.sections) - load.hand_written_minutes)
    assert drift <= 0.05 * (len(load.sections) + 1)


def test_a_fenced_block_is_not_prose() -> None:
    fenced = "```bash\n" + words(433) + "\n```"
    assert measure_page(page(hand=10, block=0, extra=fenced)).hand_written_words == 10


def test_a_tagged_fence_and_a_plain_one_each_open_and_close() -> None:
    """An info string opens a block and never closes one, so two blocks in a row do not interleave."""
    fenced = "```mermaid\n" + words(211) + "\n```\n\n```\n" + words(211) + "\n```"
    assert measure_page(page(hand=10, block=0, extra=fenced)).hand_written_words == 10


def test_a_backtick_fence_is_not_closed_by_a_tilde_one() -> None:
    """Different characters are different fences, so a `~~~` line inside a ``` block is content."""
    fenced = "```\n" + words(50) + "\n~~~\n" + words(50) + "\n```"
    assert measure_page(page(hand=10, block=0, extra=fenced)).hand_written_words == 10


def test_a_longer_fence_is_not_closed_by_a_shorter_run() -> None:
    """CommonMark, and the only reason anybody opens a four-backtick block.

    A four-backtick block whose contents are three-backtick fences was being closed by the first of
    them, so everything after it -- including the rest of the real fenced block -- was counted as
    prose or dropped depending on where the runs fell.
    """
    nested = "````\n```python\n" + words(300) + "\n```\n" + words(300) + "\n````"
    assert measure_page(page(hand=10, block=0, extra=nested)).hand_written_words == 10


def test_an_unclosed_fence_aborts_rather_than_dropping_the_rest_of_the_page() -> None:
    """The defect this replaces measured the prose above the fence and called the page short.

    Under budget for the wrong reason is the one direction a page-load figure must not be wrong in.
    """
    with pytest.raises(Sc1RecordUnusable, match="never closed"):
        measure_page(f"# t\n\n```bash\n{words(9000)}\n{RESULTS_START}\n{RESULTS_END}\n")


def test_an_unclosed_details_fold_aborts_for_the_same_reason() -> None:
    with pytest.raises(Sc1RecordUnusable, match="never closed"):
        measure_page(
            f"# t\n\n<details><summary>s</summary>\n\n{words(9000)}\n\n"
            f"{RESULTS_START}\n{RESULTS_END}\n"
        )


def test_a_multi_line_html_comment_is_not_prose() -> None:
    """`re.DOTALL` on a pattern applied one line at a time is a flag that does nothing.

    The comment was stripped per line, so only its first and last lines lost their text and every
    line between them was counted as prose a reader meets.
    """
    comment = "<!--\n" + words(411) + "\n" + words(413) + "\n-->"
    assert measure_page(page(hand=10, block=0, extra=comment)).hand_written_words == 10


def test_table_rows_are_scanned_and_not_read() -> None:
    rows = "\n".join("| " + words(20) + " | " + words(20) + " |" for _ in range(30))
    assert measure_page(page(hand=10, block=0, extra=rows)).hand_written_words == 10


def test_a_fold_costs_its_summary_and_not_its_body() -> None:
    """What a reader meets on page load is the fold. Its contents are met only if they choose."""
    fold = f"<details><summary>{words(5)}</summary>\n\n{words(397)}\n\n</details>"
    load = measure_page(page(hand=10, block=0, extra=fold))
    assert load.hand_written_words == 15


def test_the_two_halves_of_the_page_are_counted_separately() -> None:
    """Only one of them is anybody's to shorten, so a total alone would hide which."""
    load = measure_page(page(hand=7, block=11))
    assert (load.hand_written_words, load.generated_words) == (7, 11)


def test_sections_are_reported_and_add_up_to_the_hand_written_half() -> None:
    body = f"# title\n\n{words(10)}\n\n## first\n\n{words(20)}\n\n## second\n\n{words(30)}\n"
    load = measure_page(f"{body}\n{RESULTS_START}\n{words(5)}\n{RESULTS_END}\n")
    # Each section carries its own heading line, which is two tokens: the `#` markers and the word.
    # Markup counted as prose is one token per heading against a five-figure page, and leaving it in
    # keeps this measurement equal to the one the story's Code Map recorded off the same file.
    assert [section.words for section in load.sections] == [12, 22, 32]
    assert sum(section.words for section in load.sections) == load.hand_written_words


def test_prose_before_the_first_heading_gets_a_section_of_its_own() -> None:
    """Otherwise it is counted into the page and into no section, and the invariant above breaks.

    Measured on a real page it did: `words_hand_written` 12 against a section sum of 5.
    """
    load = measure_page(page(hand=12, block=0, above=f"{words(12)}\n\n# title\n\n"))
    assert load.sections[0].heading == PREAMBLE_HEADING
    assert load.sections[0].words == 12
    assert sum(section.words for section in load.sections) == load.hand_written_words


def test_a_page_that_opens_with_its_title_grows_no_preamble_row() -> None:
    """The bucket is seeded by prose, not by position, so the common page is unchanged."""
    load = measure_page(page(hand=5, block=0, above="# title\n\n"))
    assert [section.heading for section in load.sections] == ["# title"]


def test_two_sections_sharing_a_heading_are_two_rows() -> None:
    """Merging them reports one section of the sum, which is a section nobody can go and find."""
    body = f"# t\n\n## same\n\n{words(20)}\n\n## other\n\n{words(5)}\n\n## same\n\n{words(30)}\n"
    load = measure_page(f"{body}\n{RESULTS_START}\n{words(5)}\n{RESULTS_END}\n")
    assert [section.heading for section in load.sections] == [
        "# t",
        "## same",
        "## other",
        "## same",
    ]
    assert [section.words for section in load.sections] == [2, 22, 7, 32]
    assert sum(section.words for section in load.sections) == load.hand_written_words


def test_a_section_whose_only_content_is_its_own_heading_is_still_reported() -> None:
    """A heading a reader meets and nothing under it is a fact about the page, not a row to drop.

    There is no empty-section filter any more, and there is nothing left for one to catch: a
    section carries at least its own heading line, so the filter could only ever have dropped a
    real row. The one construction that used to reach zero -- a heading inside a `<details>` body --
    is the next test, and it is fixed by not opening a section there at all.
    """
    body = f"# t\n\n{words(3)}\n\n## empty\n\n```\n{words(40)}\n```\n\n## after\n\n{words(4)}\n"
    load = measure_page(f"{body}\n{RESULTS_START}\n{words(5)}\n{RESULTS_END}\n")
    assert dict((section.heading, section.words) for section in load.sections) == {
        "# t": 5,
        "## empty": 2,
        "## after": 6,
    }


def test_a_heading_inside_a_fold_is_not_a_section_and_captures_nothing_after_it() -> None:
    """The reader never saw it, so it is not a section they met -- and it was stealing the rest.

    Registering it opened a row with no words under it and then filed every hand-written word after
    the fold closed under a heading nobody had scrolled past.
    """
    body = (
        f"# t\n\n## outer\n\n{words(3)}\n\n<details><summary>{words(2)}</summary>\n\n"
        f"## folded\n\n{words(50)}\n\n</details>\n\n{words(7)}\n"
    )
    load = measure_page(f"{body}\n{RESULTS_START}\n{words(5)}\n{RESULTS_END}\n")
    assert [section.heading for section in load.sections] == ["# t", "## outer"]
    assert [section.words for section in load.sections] == [2, 3 + 2 + 2 + 7]
    assert sum(section.words for section in load.sections) == load.hand_written_words


def test_a_heading_inside_the_generated_block_is_not_a_hand_written_section() -> None:
    """`readme.py` may emit a heading tomorrow, and no fixture had one until this.

    Without the guard the block's own heading becomes a section, and every hand-written word after
    the block is attributed to text a run wrote -- prose nobody can find under a heading nobody
    typed.
    """
    body = (
        f"# t\n\n## outer\n\n{words(4)}\n\n{RESULTS_START}\n## generated\n{words(9)}\n"
        f"{RESULTS_END}\n\n{words(6)}\n"
    )
    load = measure_page(body)
    assert [section.heading for section in load.sections] == ["# t", "## outer"]
    assert [section.words for section in load.sections] == [2, 12]
    assert load.generated_words == 9 + 2  # the heading a run wrote is the run's words, not a section


def test_a_page_whose_generated_block_is_not_delimited_once_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="not delimited exactly once"):
        measure_page(f"{words(10)}\n{RESULTS_START}\n{RESULTS_START}\n{RESULTS_END}\n")


def test_an_inverted_generated_block_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="inverted"):
        measure_page(f"{RESULTS_END}\n{words(10)}\n{RESULTS_START}\n")


def test_an_unmeasurable_page_is_reported_alongside_the_records_own_failures() -> None:
    """The collect-then-raise-once idiom across both halves, which is the only caller shape there is.

    The measurement raises on its own and `verify_timed_read` folds its failures in; the earlier
    out-parameter returned a fallback measurement that no caller could ever observe.
    """
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read(f"{words(10)}\n", record(reads=(read_row(elapsed="soon"),)))
    joined = "\n".join(raised.value.failures)
    assert "not delimited exactly once" in joined
    assert "does not parse as M:SS" in joined


# --- the I/O matrix, row by row ------------------------------------------------------------------


def test_no_read_yet_reports_not_yet_run_and_the_page_load() -> None:
    report = verify_timed_read(page(), record())
    assert report.record.status == "not yet run"
    assert report.as_json()["page"]["words_total"] == report.page.total_words


def test_a_passing_read_reports_the_pass_and_the_elapsed_time() -> None:
    report = verify_timed_read(page(), record(reads=(read_row(elapsed="4:12"),)))
    assert report.record.status == "pass"
    (read,) = report.record.reads
    assert read.passed and read.elapsed == "4:12" and read.elapsed_seconds == 252


def test_a_failing_read_is_reported_and_is_not_an_abort() -> None:
    """A recorded failure is the criterion working. Aborting on it would punish honesty."""
    reads = (read_row(verdicts=("correct", "correct", "wrong")),)
    report = verify_timed_read(page(), record(reads=reads, changes=(change_row(),)))
    assert report.record.status == "fail"
    assert report.record.reads[0].passed is False


def test_the_status_is_the_latest_read_by_date_and_not_the_last_row() -> None:
    """Rows are appended by hand, and a hand-appended row is not an ordering.

    Both orders are asserted, so neither `reads[0]` nor `reads[-1]` survives: the same two reads
    are written newest-first and newest-last, and the verdict is the later date's either way.
    """
    early = read_row(reader="GH", day="2026-09-05", verdicts=("correct", "correct", "correct"))
    late = read_row(reader="FC", day="2026-09-10", verdicts=("correct", "correct", "wrong"))
    answered = (change_row(day="2026-09-11", after="FC 2026-09-10"),)

    newest_first = verify_timed_read(page(), record(reads=(late, early), changes=answered))
    newest_last = verify_timed_read(page(), record(reads=(early, late), changes=answered))
    assert newest_first.record.status == "fail"
    assert newest_last.record.status == "fail"
    assert newest_first.record.latest is not None
    assert newest_first.record.latest.day == "2026-09-10"
    assert newest_last.record.latest is not None
    assert newest_last.record.latest.reader == "FC"


def test_a_failure_with_no_readme_change_after_it_aborts() -> None:
    reads = (read_row(verdicts=("correct", "correct", "wrong")),)
    with pytest.raises(Sc1RecordUnusable, match="no README change is recorded after it"):
        verify_timed_read(page(), record(reads=reads))


@pytest.mark.parametrize("position", range(len(QUESTIONS)))
def test_a_wrong_answer_on_any_one_of_the_three_fails(position: int) -> None:
    """Not two out of three. The three questions are the criterion."""
    verdicts = ["correct"] * len(QUESTIONS)
    verdicts[position] = "wrong"
    reads = (read_row(verdicts=tuple(verdicts)),)
    with pytest.raises(Sc1RecordUnusable, match=re.escape(QUESTIONS[position])):
        verify_timed_read(page(), record(reads=reads))


@pytest.mark.parametrize("verdict", ANSWER_VERDICTS)
def test_every_declared_verdict_is_one_the_parser_accepts(verdict: str) -> None:
    """The pattern is built from `ANSWER_VERDICTS`, so a verdict added to the constant works here.

    Two hand-written copies of one closed vocabulary is one of them going stale, and the stale one
    would be the parser refusing a grade the record tells a recorder to write.
    """
    reads = (read_row(verdicts=(verdict, verdict, verdict)),)
    changes = () if verdict == ANSWER_VERDICTS[0] else (change_row(),)
    report = verify_timed_read(page(), record(reads=reads, changes=changes))
    assert [answer.verdict for answer in report.record.reads[0].answers] == [verdict] * 3


def test_a_read_at_or_over_the_budget_fails_however_good_the_answers() -> None:
    reads = (read_row(elapsed=f"{BUDGET_MINUTES}:00"),)
    with pytest.raises(Sc1RecordUnusable, match="which is not under"):
        verify_timed_read(page(), record(reads=reads))


def test_a_read_of_no_time_at_all_is_refused_rather_than_recorded_as_the_fastest_pass() -> None:
    """`0:00` parses as M:SS and beats every budget, and three correct answers made it a pass."""
    with pytest.raises(Sc1RecordUnusable, match="a read that took no time is not a timed read"):
        verify_timed_read(page(), record(reads=(read_row(elapsed="0:00"),)))


def test_a_read_with_no_elapsed_time_aborts_naming_the_field_and_the_read() -> None:
    reads = (read_row(elapsed=""),)
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read(page(), record(reads=reads))
    (failure,) = raised.value.failures
    assert "no elapsed time" in failure and "FC" in failure


def test_a_read_with_two_answers_instead_of_three_aborts() -> None:
    short = read_row()[:-1]
    with pytest.raises(Sc1RecordUnusable, match="cells against the"):
        verify_timed_read(page(), record(reads=(short,)))


def test_a_read_with_an_empty_answer_cell_aborts_naming_the_question() -> None:
    row = list(read_row())
    row[-1] = ""
    with pytest.raises(Sc1RecordUnusable, match=re.escape(QUESTIONS[-1])):
        verify_timed_read(page(), record(reads=(tuple(row),)))


def test_a_graded_answer_with_nothing_under_it_aborts() -> None:
    """`correct` alone cannot be disagreed with, which is the whole value of a verbatim answer."""
    row = list(read_row())
    row[-1] = "correct"
    with pytest.raises(Sc1RecordUnusable, match="does not open with"):
        verify_timed_read(page(), record(reads=(tuple(row),)))


def test_a_pipe_inside_an_answer_is_escaped_and_comes_back_whole() -> None:
    """The record tells a recorder to write `\\|`, and following that instruction must not fail.

    Both directions matter. Without the lookbehind the escaped pipe opens a column and the row is
    refused as short a field -- the check rejecting valid evidence, which is the failure this module
    can least afford -- and the verbatim answer, which is the only part a later reader can disagree
    with, arrives cut in half.
    """
    quoted = r"correct: it printed `a \| b`, so the pipe survived"
    row = ("FC", "2026-09-05", "4:12", quoted, *answers("correct", "correct"))
    report = verify_timed_read(page(), record(reads=(row,)))
    (read,) = report.record.reads
    assert len(read.answers) == len(QUESTIONS)
    assert read.answers[0].text == "it printed `a | b`, so the pipe survived"
    assert "\\" not in read.answers[0].text


def test_an_unparseable_elapsed_time_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="does not parse as M:SS"):
        verify_timed_read(page(), record(reads=(read_row(elapsed="soon"),)))


def test_a_seconds_field_above_fifty_nine_aborts() -> None:
    """`4:75` is not four minutes and seventy-five seconds; it is somebody typing.

    Widening the bound to `\\d\\d` reads it as 315 seconds and calls it inside the budget, which is
    a pass awarded to a number nobody can defend.
    """
    with pytest.raises(Sc1RecordUnusable, match="does not parse as M:SS"):
        verify_timed_read(page(), record(reads=(read_row(elapsed="4:75"),)))


def test_an_unparseable_date_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="not an ISO date"):
        verify_timed_read(page(), record(reads=(read_row(day="last tuesday"),)))


@pytest.mark.parametrize("spelling", ["20260905", "2026-W36-6", "2026-9-5"])
def test_a_date_that_is_not_spelt_yyyy_mm_dd_aborts_saying_so(spelling: str) -> None:
    """`date.fromisoformat` is looser than the message beside it, and the looseness is not harmless.

    A record spelling its dates compactly in both tables parsed, then failed the cross-table check
    saying the read was not in the reads table -- a true sentence about the wrong problem, aimed at
    a recorder who had written the same date in both places.
    """
    with pytest.raises(Sc1RecordUnusable, match="not an ISO date"):
        verify_timed_read(page(), record(reads=(read_row(day=spelling),)))


def test_an_impossible_calendar_date_aborts_too() -> None:
    """The shape alone would accept it; the calendar is checked as well as the spelling."""
    with pytest.raises(Sc1RecordUnusable, match="not an ISO date"):
        verify_timed_read(page(), record(reads=(read_row(day="2026-13-45"),)))


def test_a_read_with_no_reader_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="no reader"):
        verify_timed_read(page(), record(reads=(read_row(reader=""),)))


@pytest.mark.parametrize("position", range(len(QUESTIONS)))
def test_a_renamed_question_aborts_and_names_the_enumerated_three(position: int) -> None:
    """The criterion is not the record's to edit once the answers are in."""
    columns = list(READ_COLUMNS)
    columns[3 + position] = "Something easier"
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read(page(), record(read_columns=tuple(columns)))
    (failure,) = raised.value.failures
    assert all(question in failure for question in QUESTIONS)
    assert "Something easier" in failure


def test_a_reordered_question_set_aborts() -> None:
    columns = (*READ_COLUMNS[:3], *reversed(QUESTIONS))
    with pytest.raises(Sc1RecordUnusable, match="not the record's to rename"):
        verify_timed_read(page(), record(read_columns=columns))


def test_a_renamed_changes_column_aborts() -> None:
    """The other table has a header too, and nothing was passing the fixture's own argument for it.

    A renamed `After read` is a consequence column no check can find, so a failed read stops being
    answerable while the record still looks like a record.
    """
    columns = (CHANGE_COLUMNS[0], "Follows", CHANGE_COLUMNS[2])
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read(page(), record(change_columns=columns))
    (failure,) = raised.value.failures
    assert "README changes table's columns are" in failure
    assert "Follows" in failure


@pytest.mark.parametrize("empty", ["reads", "changes"])
def test_markers_with_no_table_between_them_abort(empty: str) -> None:
    """The record-edited-to-fit-a-result case, in the shape it actually takes.

    Deleting the rows and the header leaves markers a checker still finds, and a check that shrugged
    would report `not yet run` over a file somebody had emptied of its evidence.
    """
    bodies = {f"{empty}_body": "nothing here at all"}
    with pytest.raises(Sc1RecordUnusable, match="holds no table at all"):
        verify_timed_read(page(), record(**bodies))  # type: ignore[arg-type]


def test_a_header_with_no_rows_under_it_is_the_honest_empty_state() -> None:
    """The other direction of the same guard, so it cannot be satisfied by refusing everything."""
    report = verify_timed_read(page(), record())
    assert report.record.status == "not yet run"


def test_an_absent_record_aborts_naming_the_path(tmp_path: Path, repo_root: Path) -> None:
    missing = tmp_path / "nowhere.md"
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read_files(repo_root / "README.md", missing)
    assert str(missing) in str(raised.value)


def test_a_record_that_is_not_valid_utf8_is_the_same_abort(
    tmp_path: Path, repo_root: Path
) -> None:
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, and an `OSError` clause misses it."""
    broken = tmp_path / "sc1.md"
    broken.write_bytes(b"# probe\n\xff\xfe not utf-8\n")
    with pytest.raises(Sc1RecordUnusable, match="could not be read"):
        verify_timed_read_files(repo_root / "README.md", broken)


def test_an_unreadable_readme_is_the_same_abort(tmp_path: Path, repo_root: Path) -> None:
    with pytest.raises(Sc1RecordUnusable, match="could not be read"):
        verify_timed_read_files(tmp_path / "no-such-readme.md", repo_root / DEFAULT_RECORD)


def test_a_record_with_no_reads_block_aborts() -> None:
    with pytest.raises(Sc1RecordUnusable, match="reads block is not delimited exactly once"):
        verify_timed_read(page(), "# SC1\n\nnothing here.\n")


def test_two_reads_sharing_a_reader_and_a_date_abort() -> None:
    """A README change names the read it followed by reader and date; two of those are ambiguous."""
    reads = (read_row(), read_row())
    with pytest.raises(Sc1RecordUnusable, match="separate reads"):
        verify_timed_read(page(), record(reads=reads))


def test_the_same_reader_on_a_different_date_is_refused() -> None:
    """The protocol says a first-time reader, and reader-and-date cannot express that.

    Deduplicating on the pair accepts the same person reading the changed page a week later, which
    is a measurement of somebody who has seen the page -- not the claim SC1 makes.
    """
    reads = (read_row(day="2026-09-05"), read_row(day="2026-09-12"))
    with pytest.raises(Sc1RecordUnusable, match="never repeated with the same person"):
        verify_timed_read(page(), record(reads=reads))


def test_a_change_naming_a_read_nobody_took_aborts() -> None:
    changes = (change_row(after="Someone 2026-01-01"),)
    with pytest.raises(Sc1RecordUnusable, match="not in the reads table"):
        verify_timed_read(page(), record(reads=(read_row(),), changes=changes))


def test_a_change_dated_before_the_read_it_answers_aborts() -> None:
    """A 2020 edit does not answer a 2026 failure, and the ordering is the whole causal claim."""
    reads = (read_row(verdicts=("wrong", "correct", "correct")),)
    changes = (change_row(day="2020-01-01"),)
    with pytest.raises(Sc1RecordUnusable, match="which happened later"):
        verify_timed_read(page(), record(reads=reads, changes=changes))


def test_a_change_with_no_description_aborts() -> None:
    reads = (read_row(verdicts=("wrong", "correct", "correct")),)
    with pytest.raises(Sc1RecordUnusable, match="no description of what changed"):
        verify_timed_read(page(), record(reads=reads, changes=(change_row(what=""),)))


def test_every_failure_is_collected_before_the_abort_is_raised() -> None:
    """A record wrong in three places is told all three, as the siblings do it."""
    reads = (read_row(elapsed="", day="nope"), read_row(reader="", day="2026-01-02"))
    with pytest.raises(Sc1RecordUnusable) as raised:
        verify_timed_read(page(), record(reads=reads))
    assert len(raised.value.failures) >= 3


# --- nothing is typed that could be measured -------------------------------------------------------


def _numeric_literals(source: str) -> set[float]:
    return {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }


def headline_counts(repo_root: Path) -> set[int]:
    """Every count the report prints that a person might be tempted to retype.

    `lines` is one of them: the record hand-typed "line 960 of 1,262" while `lines` is a measured
    field, and the guard could not see it.
    """
    load = measure_page((repo_root / "README.md").read_text(encoding="utf-8"))
    return {
        load.hand_written_words,
        load.generated_words,
        load.total_words,
        load.budget_words,
        load.lines,
    }


def spellings(count: int) -> tuple[str, ...]:
    """A count as a person writes it: bare, and with the thousands separator.

    `\\b1262\\b` does not match "1,262", so the comma form walked past the guard.
    """
    return (str(count), f"{count:,}")


def test_no_word_count_is_a_literal_in_the_module(repo_root: Path) -> None:
    """The figure is derived from the file or it is a number that goes stale the next time anyone edits."""
    module = repo_root / "src" / "nbc" / "report" / "timed_read.py"
    literals = _numeric_literals(module.read_text(encoding="utf-8"))
    typed = headline_counts(repo_root) & {int(value) for value in literals}
    assert not typed, f"{sorted(typed)} is typed into the module that is supposed to measure it"


def typed_counts(prose: str, counts: set[int]) -> set[int]:
    """Which of `counts` the prose spells out, in either the bare or the comma form."""
    return {
        count
        for count in counts
        for spelling in spellings(count)
        if re.search(rf"(?<![\d,]){re.escape(spelling)}(?![\d,])", prose)
    }


def test_the_guard_can_see_a_count_written_with_a_thousands_separator(repo_root: Path) -> None:
    """Its own red input, in both spellings and in neither.

    `\\b1262\\b` does not match "1,262", so the comma form walked past the guard and a hand-typed
    line number sat in the record beside a measured one. The last case is the control: a number the
    page does not measure is not flagged, so the guard is not simply matching every digit it sees.
    """
    counts = headline_counts(repo_root)
    biggest = max(counts)
    assert typed_counts(f"the page is {biggest:,} words long", counts) == {biggest}
    assert typed_counts(f"the page is {biggest} words long", counts) == {biggest}
    assert typed_counts("the page is 7 words long", counts) == set()


def test_the_record_cites_the_measurement_rather_than_typing_it(repo_root: Path) -> None:
    """Every figure in the record is inside the pasted transcript, and the transcript is the output.

    The newest transcript is compared to what the module prints for the page as it stands. An older
    dated observation is history and is not compared to anything -- the point of dating it is that
    the page has moved since. So when the README changes, what this test asks for is a **new**
    dated observation appended below the old ones, produced by rerunning the command, which is the
    same discipline `size_budget.py` applies to the number it states.
    """
    text = (repo_root / DEFAULT_RECORD).read_text(encoding="utf-8")
    transcripts = re.findall(r"^```json\n(.*?)^```", text, re.S | re.M)
    assert transcripts, "the record cites no measurement at all"

    live = measure_page((repo_root / "README.md").read_text(encoding="utf-8")).as_json()
    assert json.loads(transcripts[-1]) == {"page": live}, (
        "the record's newest observation is not what the module prints for the page as it stands; "
        "rerun `python -m nbc.report.timed_read` and append a new dated observation"
    )

    outside = re.sub(r"^```json\n.*?^```", "", text, flags=re.S | re.M)
    typed = typed_counts(outside, headline_counts(repo_root))
    assert not typed, f"{sorted(typed)} is typed into the record's prose instead of being cited"


# --- the abort, and the command line ----------------------------------------------------------------


def test_the_abort_is_one_of_the_projects_declared_aborts() -> None:
    abort = Sc1RecordUnusable("because")
    assert isinstance(abort, NbcError)
    assert exit_code_for(abort) == Sc1RecordUnusable.exit_code == 37
    assert abort.failures == ("because",)


def run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nbc.report.timed_read", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def test_the_command_line_reports_the_page_load_and_the_records_state(repo_root: Path) -> None:
    completed = run_cli(repo_root)
    assert completed.returncode == EXIT_OK, completed.stderr
    report = json.loads(completed.stdout)
    assert report["record"]["status"] == "not yet run"
    assert report["page"]["words_per_minute"] == WORDS_PER_MINUTE
    assert report["page"]["words_total"] > report["page"]["budget_words"]
    assert report["page"]["over_budget_factor_hand_written"] > 1


def test_the_command_line_prints_stable_key_order(repo_root: Path) -> None:
    """`sort_keys=True`, pinned. The record's protocol tells a human to paste this output whole.

    Two transcripts pasted a week apart diff cleanly only if the key order is a property of the
    command rather than of the dictionary literal it happened to be built from.
    """
    first = run_cli(repo_root)
    second = run_cli(repo_root)
    assert first.stdout == second.stdout
    keys = list(json.loads(first.stdout)["page"])
    assert keys == sorted(keys)


def test_the_command_line_exits_with_the_aborts_own_code_and_writes_no_report(
    repo_root: Path,
) -> None:
    completed = run_cli(repo_root, "--record", "/dev/null")
    assert completed.returncode == Sc1RecordUnusable.exit_code == 37
    assert completed.stdout == ""
    assert "not usable as evidence" in completed.stderr


def test_the_check_does_not_import_the_inference_runtime() -> None:
    """A README-claim check that loaded `onnxruntime` would be a check with a 300 MB prerequisite."""
    code = "import sys, nbc.report.timed_read; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout
