# SC1: the five-minute read

This repository's success criterion SC1 says that **a reader who has never seen it understands the
question, the table and the caveats from `README.md` alone, in under five minutes.** That is the only
claim here about a human, and it is the only one no test in this repository can settle.

This file is the protocol for settling it, and the record of every attempt.

## Why a person and not CI

`python -m nbc.report.caveats` already aborts the run when the "what this does not show" section is
missing, incomplete, thin, or has drifted inside the generated block. That guarantees the section
**exists**. It guarantees nothing whatever about whether it is any **good** — which is FR19's own
warning, that a thin caveats section is worse than none because it looks like diligence.

No check in this repository can read the page and know whether a stranger understood it. Every
machine-checkable property of the README is already gated: the block is a pure function of
`results/results.json`, the caveats are enumerated rather than counted, the size budget is compared
three ways, and every command the page documents is executed by a test. Understanding is not on that
list and cannot be added to it. So SC1 is a human gate, deliberately, and what a machine does here is
everything around the human: hold the protocol, hold the record, measure the page, and refuse a
record that has been quietly edited to fit a result.

    python -m nbc.report.timed_read

prints the measured page load and the state of this record, and aborts with its own exit code if the
record has stopped being usable as evidence. It reports `not yet run` when nobody has read the page,
and that is not a failure of the check.

## The protocol

1. **Find someone who has never seen this repository.** Not someone who has skimmed it, not someone
   who has heard it described. A first-time reader is a resource that does not regenerate: each
   person can be spent on this exactly once, so do not spend one to learn something a word count
   already says.
2. **Give them `README.md` and nothing else.** No repository tour, no framing sentence, no answer to
   "what should I look for". The criterion says "from the README alone", and a link is not the
   README: the page is handed over without the repository around it, and a linked file the reader
   would need to open in order to answer is a gap in the page, not a resource of the read.
3. **Start a clock.** They read at their own pace and stop when they are done, or at five minutes,
   whichever comes first. Record the elapsed time either way.
4. **Take the page away.** The three answers are given without looking back. A reader who can find
   the answer again with the file open has demonstrated that the file contains it, which nobody
   doubted; the criterion is about what they carry away.
5. **Ask the three questions below, in order, and write the answers down verbatim.** Not a summary of
   the answer, not a paraphrase that already agrees with the page. The verbatim answer is the only
   part of this record that a later reader can disagree with.
6. **Grade each answer `correct` or `wrong`** and add a row to the reads table.
7. **On any failure, change the README** and record what changed in the second table.

## The three questions

These three are **enumerated, never counted**, for the same reason the eleven caveats are: a rule
that asserts "three questions" is satisfied by asking the easiest one three times, and a criterion
that can be reworded once the answers are in is not a criterion. They are also the reads table's
column headers, so renaming one is a parse failure rather than an edit.

Reworded once, on **2026-09-02**, while the reads table below was still empty: the third question
used to be *what does it not show*, which is `epics.md`'s operationalisation of SC1 and accepted any
one caveat. `prd.md`'s verbatim third noun is **the caveats**, and the question now grades that. A
rewording after a read is the move this file forbids; a rewording before any read, toward the
criterion's own sentence, is the opposite move — and it is written down so nobody has to reconstruct
which wording the first read was graded against.

1. **What is the question** — what is this repository trying to find out? An answer naming
   canonicalization and prompt-injection classification, in any words, is correct. "It is about
   Unicode" is not.
2. **What does the table show** — what did the measurement come out as? An answer that gets the shape
   of the result right — that there is a recovery on the attack side, a cost on the benign side, and
   that one pre-registered condition triggered — is correct. Naming the exact figures is not
   required and getting a direction backwards is wrong.
3. **What do the caveats say** — what does the page itself say its numbers do not establish? SC1's
   third noun is "the caveats", plural and section-sized, so one remembered bullet does not
   demonstrate it: an answer stating **at least two** of the published caveats, each as the page
   states it, is correct. A single caveat is wrong, "nothing comes to mind" is wrong, and so is a
   limitation the reader invented that the page does not claim.

## The rule

- **A wrong answer on any one of the three fails the criterion.** Not two out of three. The three
  questions are the criterion; passing two of them is failing.
- **A read at or over 5:00 fails**, however good the answers are. The claim is about five minutes.
- **On a failure, the README changes. The criterion never does.** This is the whole point of writing
  the questions down before the read. Rewording a question, or relaxing the clock, to make a
  recorded failure into a pass is the one move this file exists to make visible, and
  `nbc.report.timed_read` aborts on a record that has no README change following a failed read.
- **A failure is read for its reason, not only its verdict.** Where the reader stopped, what they
  went looking for and could not find, and which of the three they got wrong say more about what to
  cut than the verdict does.
- **A read is never repeated with the same person.** They have seen the page now. The next read is a
  new person, against the changed page. The checker refuses a record where one reader appears twice,
  on any dates.
- **A read of `0:00` is not a timed read**, and the checker refuses one rather than recording it as
  the fastest pass on the table. So is a `Date` written any way other than `YYYY-MM-DD`, and so is a
  README change dated before the read it claims to answer.

## What the page measured before the first read

Recorded on **2026-09-01**, before any read, so that whatever a reader eventually does is comparable
to the page as it stood when this protocol was written. **Nothing below was typed**: it is the `page`
half of what `python -m nbc.report.timed_read` printed on that date, pasted whole.

```json
{
  "page": {
    "budget_minutes": 5,
    "budget_words": 1250,
    "lines": 1263,
    "minutes": 47.1,
    "minutes_generated_block": 12.0,
    "minutes_hand_written": 35.1,
    "over_budget_factor_hand_written": 7.01,
    "over_budget_factor_total": 9.42,
    "sections": [
      {
        "heading": "# normalize-before-classify",
        "minutes": 0.7,
        "words": 182
      },
      {
        "heading": "## The question",
        "minutes": 0.7,
        "words": 173
      },
      {
        "heading": "## What gets measured",
        "minutes": 6.4,
        "words": 1591
      },
      {
        "heading": "## Reproducing this",
        "minutes": 7.4,
        "words": 1841
      },
      {
        "heading": "## Status",
        "minutes": 1.3,
        "words": 315
      },
      {
        "heading": "## How big the layer is",
        "minutes": 1.3,
        "words": 326
      },
      {
        "heading": "## What this does not show",
        "minutes": 14.5,
        "words": 3613
      },
      {
        "heading": "## License",
        "minutes": 0.2,
        "words": 39
      },
      {
        "heading": "## Redistribution of undeclared material",
        "minutes": 2.7,
        "words": 685
      }
    ],
    "words_generated_block": 3004,
    "words_hand_written": 8765,
    "words_per_minute": 250,
    "words_total": 11769
  }
}
```

**Read `over_budget_factor_hand_written` and not the total.** The transcript reports both because
they answer different questions. The total prices the whole scroll, generated tables included, and
nobody can shorten those: the block between the `RESULTS` markers is a pure function of
`results/results.json`. The hand-written factor is the one a person could act on, and it is the
headline for that reason.

**The prediction, written before the read rather than after it.** The first question is answered in
the page's opening section and the second inside the generated block, but the third — *what does it
not show* — lives in the longest section on the page, the one the transcript above prices at more
than twice the whole five-minute budget on its own, and there is no table of contents to reach it
with. **A wrong answer on the third question is the expected outcome.** It is recorded here so that
the read can falsify it rather than confirm it — and so that a pass, if it comes, is a genuine
surprise on the record instead of a retrospective claim that the page was fine all along.

No line numbers are quoted for those three answers. Three were, and one of them was wrong: the
verdict shape a reader needs for the second question is not on the line that was cited, and nothing
in this repository was checking any of the three. A line number in a hand-written file beside a
generated measurement is exactly the stale figure this record is written to avoid.

Measured again on **2026-09-02**, still before any read, after the page grew an abstract and a
conclusion. The abstract is generated between its own markers and is counted with the generated
half, because `python -m nbc.report.readme` replaces it wholesale and shortening it by hand is
futile; the conclusion and the abstract's framing sentences are hand-written and count against the
half a person can act on. As above, nothing below was typed: it is the `page` half of what
`python -m nbc.report.timed_read` printed on that date, pasted whole.

```json
{
  "page": {
    "budget_minutes": 5,
    "budget_words": 1250,
    "lines": 1312,
    "minutes": 49.3,
    "minutes_generated_block": 13.3,
    "minutes_hand_written": 35.9,
    "over_budget_factor_hand_written": 7.18,
    "over_budget_factor_total": 9.85,
    "sections": [
      {
        "heading": "# normalize-before-classify",
        "minutes": 0.7,
        "words": 182
      },
      {
        "heading": "## Abstract",
        "minutes": 0.2,
        "words": 52
      },
      {
        "heading": "## The question",
        "minutes": 0.7,
        "words": 173
      },
      {
        "heading": "## What gets measured",
        "minutes": 6.4,
        "words": 1591
      },
      {
        "heading": "## Reproducing this",
        "minutes": 7.4,
        "words": 1841
      },
      {
        "heading": "## Status",
        "minutes": 1.3,
        "words": 315
      },
      {
        "heading": "## How big the layer is",
        "minutes": 1.3,
        "words": 326
      },
      {
        "heading": "## What this does not show",
        "minutes": 14.5,
        "words": 3613
      },
      {
        "heading": "## Conclusion",
        "minutes": 0.7,
        "words": 163
      },
      {
        "heading": "## License",
        "minutes": 0.2,
        "words": 39
      },
      {
        "heading": "## Redistribution of undeclared material",
        "minutes": 2.7,
        "words": 685
      }
    ],
    "words_generated_block": 3335,
    "words_hand_written": 8980,
    "words_per_minute": 250,
    "words_total": 12315
  }
}
```

Two things this measurement is not. It is not a reading time: readers differ by far more than the
250 words per minute the report declares, and that constant is a stated convention rather than a
fact about anybody. And it is not an argument for cutting any particular paragraph — curation is real
work with a real risk of dropping a disclosure, and it is deliberately not what this file does.

## The reads

Empty is the honest state. `python -m nbc.report.timed_read` reports `not yet run`, and a criterion
nobody has tested yet is a criterion nobody has tested yet.

One row per read. `Reader` is a name or initials, `Date` is ISO, `Elapsed` is `M:SS`, and each of the
three answer cells opens with `correct:` or `wrong:` followed by what the reader actually said. A
literal `|` inside an answer is written `\|`.

<!-- SC1-READS:START -->

| Reader | Date | Elapsed | What is the question | What does the table show | What do the caveats say |
|--------|------|---------|----------------------|--------------------------|-----------------------|

<!-- SC1-READS:END -->

## What changed in the README after a failed read

One row per change, naming the read it answers as `<Reader> <Date>` — the same reader and date as the
row above. A failed read with no row here is what makes this record unusable, and the checker says so.

<!-- SC1-CHANGES:START -->

| Date | After read | What changed |
|------|------------|--------------|

<!-- SC1-CHANGES:END -->
