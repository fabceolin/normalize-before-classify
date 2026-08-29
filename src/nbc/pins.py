"""The only module that reads `pins.toml`, and the only place a remote artifact is named.

A published table claims its numbers stay comparable. That claim rests on every remote input
being fixed: a model that moved under the experiment, or a dataset whose rows changed, turns a
reproducible table into a snapshot of one afternoon. So every remote artifact is named once, in
`pins.toml`, and this module is the only reader.

**A revision alone is not a pin.** One pinned baseline ships two different `tokenizer.json` at
the same commit -- one at the repository root, one beside the ONNX graph -- differing in the
truncation they declare. Pinned by revision alone, the choice between them falls to whichever
loader convention the code happens to follow, and two strangers reproducing the table window the
same document differently. The pin therefore names the *path*: graph, tokenizer and config, per
baseline.

Four aborts, four codes, because the remedies differ:

- `PinsFileInvalid` (4) -- the file is missing, unparseable, or says something it may not say.
  The author has to fix the repository.
- `BaselineSetInvalid` (5) -- the file is well formed and the set it declares violates SC5:
  fewer than two baselines, or two that share an (architecture, tokenizer) family pair. The pins
  have to change, and a baseline is *replaced*, never removed.
- `PinMismatch` (6) -- the file is right and the world moved. Nothing in the repository is
  wrong; the run must not proceed.
- `BaselineIneligible` (7) -- the file is well formed and honest, and what it honestly declares
  disqualifies one baseline from being scored over the pinned corpus at all.

**No baseline is scored over its own training text, and reading model cards was twice not
enough.** A classifier scored on its own training data reports memory rather than detection, and
it cuts both ways: recall looks better on attacks it has seen, and the false-positive rate looks
better on benign text it was taught to call safe. The second half is the dangerous one, because
the false-positive rate is the counter-metric the whole artifact rests on. Two of the three teeth
that guard it live here, and both are *declaration* teeth -- what the cards say, recorded as data
so a rule can read it instead of a person:

1. **Declared lineage.** Every baseline declares its relationship to every pinned attack dataset,
   from a closed vocabulary, with the date the card was read and the revision it was read at. A
   baseline declaring `trained-on` a pinned dataset is ineligible and the run aborts naming both.
2. **One hop of declared provenance.** Every pinned dataset declares the sources its *own card*
   names as seeds it was built from. A baseline declaring `trained-on` one of those seeds reaches
   the dataset at one remove, which no model card can show -- nothing on a baseline's card
   mentions a dataset built downstream of it. That hop is ineligible too, unless the pins declare
   the reach and its remedy: `seeded-from-declared-training-source` says the coincident rows are
   removed from the corpus before anything is measured, and it turns every seed it covers into a
   **required exclusion source** (`Pins.required_exclusion_sources()`). An *undeclared* hop
   aborts, which is the failure that got through twice.

The third tooth -- measured text overlap, computed and removed at build time -- is the corpus
builder's, not this module's. This module states the obligation; `corpus/` discharges it. What
this module adds is that the obligation is now *pinned*: `[[exclusion_source]]` names every
source the two declaration blocks imply -- every source a baseline declares `trained-on`, and
every seed a pinned dataset's own card names -- and the file is refused when that array and the
derived set disagree in either direction. One of those sources answers HTTP 401 and hands back
no commit, so it is declared `unavailable` with the status it returns and pinned without a
revision, which is the only shape that is honest about it.

**A check that was never re-run after a pin moved is visible rather than assumed.** Every lineage
and provenance declaration records the card revision it was performed against, and the file is
refused when that revision is not the one pinned. The date is metadata; the revision is the gate.

Structure, the baseline set and the lineage gate are all checked by `load_pins()`, so every
consumer gets them for free and the entrypoint cannot forget. `verify_revisions()` is the
separate step that asks the world, and it is the one that touches a network.

**Offline after first fetch (and the resolver is a parameter).** AD-9 wants the resolved commit
compared against the pin; NFR3 wants no network once the models are cached. Both hold because
resolution goes to the local Hugging Face cache first: a snapshot directory *named by the sha*
is proof the artifact at that sha is on this machine, which is the question a run has to answer.
Whether the remote still resolves that sha is a different question and belongs to the smoke job,
over the network, once. Passing the resolver in is also what lets the offline unit suite cover
every outcome without reaching for a socket.

This module imports the standard library and `nbc.errors`, and nothing else. It is step 1 of the
entrypoint's sequence, immediately after the platform preflight and well before `onnxruntime`
is imported, and a test asserts that importing or running it leaves the runtime out of
`sys.modules`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import datetime
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, ClassVar, Final, Mapping, Sequence

from nbc.errors import NbcError

__all__ = [
    "AttackDataset",
    "AttackDraw",
    "BENIGN_CHAT_CLASS",
    "BENIGN_CODE_CLASS",
    "BENIGN_CODE_ELIGIBILITIES",
    "BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE",
    "BenignChatFrame",
    "BenignCodeFrame",
    "BenignCodeRepository",
    "BenignFrame",
    "ConfirmatoryCell",
    "FRAME_ID_FIELD",
    "MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY",
    "MINIMUM_BENIGN_CODE_REPOSITORIES",
    "DRAW_HEAD",
    "DRAW_METHODS",
    "DRAW_SEEDED_RANDOM",
    "DRAW_SORT_KEYS",
    "ExclusionSource",
    "Baseline",
    "BaselineIneligible",
    "EXCLUSION_AVAILABILITIES",
    "EXCLUSION_AVAILABLE",
    "EXCLUSION_UNREACHABLE",
    "EXCLUSION_UNREADABLE",
    "HTTP_OK",
    "LINEAGE_RELATIONSHIPS",
    "Lineage",
    "Licence",
    "MINIMUM_BASELINES",
    "NOT_DECLARED",
    "OQ2_KEPT",
    "OQ2_OUTCOMES",
    "OQ2_REPLACEMENT",
    "Oq2Check",
    "PINS_FILENAME",
    "PINNED_PRECISION",
    "PinMismatch",
    "Pins",
    "PinsFileInvalid",
    "BaselineSetInvalid",
    "Provenance",
    "RemoteArtifact",
    "Resolution",
    "Resolver",
    "CHECKED_AGAINST_CACHE",
    "CHECKED_AGAINST_HUB",
    "SCHEMA_VERSION",
    "SEEDED_FROM_TRAINING_SOURCE",
    "SHARED_WINDOW_POLICY",
    "TRAINED_ON",
    "TRAINING_SOURCE_RELATIONSHIPS",
    "WINDOW_POLICIES",
    "canonical_source_id",
    "hf_cache_root",
    "load_pins",
    "main",
    "resolve_from_cache_then_hub",
    "resolve_over_http",
    "verify_revisions",
]

PINS_FILENAME: Final[str] = "pins.toml"
"""The file's name, at the repository root. Named once, here."""

SCHEMA_VERSION: Final[int] = 3
"""The shape this module knows how to read. A file declaring another version is refused.

Version 3 adds the required `[attack_dataset.draw]` sub-table. A version-2 file has no draw at
all, and defaulting one would be a draw nobody declared, so the version moves rather than the
field becoming optional.
"""

MINIMUM_BASELINES: Final[int] = 2
"""Two baselines is the floor of SC5 and the run sits exactly on it.

One baseline makes every result read as a property of that model. Two only fix that if they can
disagree, which is why the family-pair check below sits next to this one.
"""

PINNED_PRECISION: Final[str] = "fp32"
"""The only precision a graph may be pinned at.

fp16 and mixed-precision exports move scores in the last decimals, and a hard decision threshold
turns that into a class flip on exactly the borderline encoded items this experiment measures.
"""

SHARED_WINDOW_POLICY: Final[str] = "shared"
"""AD-19's length policy: fixed non-overlapping windows, scored independently, reduced by maximum.

The name is spelled here rather than in `baselines/tokenization.py` because `pins.toml`'s
vocabulary is this module's business and `pins.py` may import nothing else in the project. The
strategy behind the name lives in `tokenization.py`, which reads it from here, so the string has
one home and the two halves are checked against each other as that module imports.
"""

WINDOW_POLICIES: Final[frozenset[str]] = frozenset({SHARED_WINDOW_POLICY})
"""Every `window_policy` a baseline may declare, which today is exactly one.

AD-29's `publisher` value is deliberately **absent** rather than admitted-and-unused. A pin may
only declare a policy something can actually run, and a `publisher` policy is meaningless without
the parameters its baseline's model card would have to transcribe -- fields this schema does not
yet carry. Admitting the name now would let a pin select a policy with no strategy behind it,
which is a run that windows every document under whatever the fallback happened to be.

The *axis* is what ships from the first run, not the second value: `window_policy` is part of the
cell key from the beginning, because a key retro-fitted into a published envelope is a schema
break, while a key that was always there costs a constant.
"""

OQ2_KEPT: Final[str] = "kept"
"""This baseline was already pinned when OQ2 measured it, and the measurement kept it."""

OQ2_REPLACEMENT: Final[str] = "replacement"
"""This baseline entered the set because OQ2 failed the one it replaced.

The distinction is worth a word because SC5's floor is two and the run sits on it: a baseline
too weak on clean text is **replaced, never removed**, so the file has to be able to say which
of the two things happened without a reader reconstructing it from commit history.
"""

OQ2_OUTCOMES: Final[frozenset[str]] = frozenset({OQ2_KEPT, OQ2_REPLACEMENT})
"""How a baseline came to be in the surviving set. There is no third way in, and no way out.

`dropped` is deliberately absent: a dropped baseline is not in this file, and admitting the word
would let the set shrink below SC5's floor while every declaration still read as valid.
"""

NOT_DECLARED: Final[str] = "not-declared"
"""What a licence or a lineage field says when the publisher declares nothing.

Not `None` and not an empty string: an absent declaration is a *finding*, and it has to survive
into `results.json` as one rather than as a missing key a reader can mistake for an oversight.
"""

TRAINED_ON: Final[str] = "trained-on"
"""The card or the config declares this source among the data the baseline was trained on."""

SEEDED_FROM_TRAINING_SOURCE: Final[str] = "seeded-from-declared-training-source"
"""The baseline reaches this pinned attack dataset at one hop, and the overlap is removed.

Declared per (baseline, dataset) pair, never inferred. It says: this baseline declares training
on at least one source the dataset's own card names as a seed it was built from, so scoring the
dataset unfiltered would score the baseline over its own training text at one remove -- and the
coincident rows are therefore removed at build time before anything is measured.

It is the only way past the one-hop gate, and it is not free: every seed it covers becomes a
**required exclusion source**, derivable from the pins by `Pins.required_exclusion_sources()`
with no prose in between. An undeclared hop aborts.
"""

LINEAGE_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {NOT_DECLARED, TRAINED_ON, SEEDED_FROM_TRAINING_SOURCE}
)
"""What a baseline may declare about a pinned attack dataset.

The set is closed because the eligibility rule *reads this value*. A free-text relationship is a
sentence a human interprets, and the two rounds of card-reading this rule exists to replace were
exactly that.
"""

TRAINING_SOURCE_RELATIONSHIPS: Final[frozenset[str]] = frozenset({NOT_DECLARED, TRAINED_ON})
"""What a baseline may declare about a source a pinned dataset names as a seed.

`SEEDED_FROM_TRAINING_SOURCE` is absent on purpose: it describes a baseline's relationship to a
*dataset*, reached through a seed. A relationship to the seed itself is only ever read from the
baseline's own card, so it is declared or it is not.
"""

EXCLUSION_AVAILABLE: Final[str] = "available"
"""The hub answers for this source and the pinned reader returns its rows."""

EXCLUSION_UNREACHABLE: Final[str] = "unreachable"
"""The hub does not answer for this source at all, so there is not even a revision to pin.

An access-restricted training source contributes an unknown number of overlapping rows, not
zero. Declaring it here is what lets the corpus build *name* it in the results instead of
leaving a reader to assume the filter reached everything it lists.
"""

EXCLUSION_UNREADABLE: Final[str] = "unreadable"
"""The hub answers, the revision pins, and the pinned reader refuses to load the rows.

A distinct value rather than a shade of `unreachable`, because the two are different facts with
different evidence and different remedies. This one is a repository published as a loading
script, which `datasets` 5 refuses outright: the sha resolves and `verify_revisions` checks it
like any other pin, and the rows are still unreadable. Collapsing it into `unreachable` would
mean discarding a revision that does resolve, and calling it `available` would mean the build
crashes on a source the pins called fine.
"""

EXCLUSION_AVAILABILITIES: Final[frozenset[str]] = frozenset(
    {EXCLUSION_AVAILABLE, EXCLUSION_UNREACHABLE, EXCLUSION_UNREADABLE}
)
"""What an `[[exclusion_source]]` may say about itself. Closed, because the build reads it.

The build probes and loads each source and compares what it observes against this value, in
**both** directions: a source declared available that no longer resolves or no longer loads, and
a source declared unreachable or unreadable that suddenly works, are all moves in the exclusion
set -- and a moving exclusion set silently changes which rows survive into the corpus.
"""

DRAW_HEAD: Final[str] = "head"
"""Take the first N of the pool under a declared sort key. Needs `sort_key`, refuses `seed`."""

DRAW_SEEDED_RANDOM: Final[str] = "seeded_random"
"""Shuffle the sorted pool under an explicit integer seed. Needs `seed`, refuses `sort_key`."""

DRAW_METHODS: Final[tuple[str, ...]] = (DRAW_HEAD, DRAW_SEEDED_RANDOM)
"""The only two ways a draw may be declared.

A closed vocabulary rather than free text, for the reason every other vocabulary in this file is
one: a method nobody implemented reads exactly like a method somebody did, and the difference
only shows up in which rows the corpus ends up holding.
"""

DRAW_SORT_KEYS: Final[tuple[str, ...]] = ("text", "payload_id")
"""What a `head` draw may sort by. Both are pure functions of the payload and of nothing else.

Neither admits row position, split or file order, which is what would make the same pins produce
two different corpora on two machines.
"""

HTTP_OK: Final[int] = 200
"""The status an `available` exclusion source must declare, and the one the build must observe."""

# --- the benign frame's vocabulary and its floors ----------------------------------------------
#
# FR5.1 fixes three things about the benign frame that a declaration may not weaken, so each is a
# requirement constant here and a declared value in `pins.toml`, and the reader **compares** them.
# The alternative -- reading the floor off the file it is supposed to bound -- is a check whose two
# sides come from the same place, which is the defect this project has found in its own history
# often enough to name. `MINIMUM_BASELINES` above is the same shape and predates these.
#
# The per-class item count is deliberately NOT one of them. It is declared in `pins.toml` and
# nowhere else, because `tests/test_pins.py` refuses any value under a `sample_size*` key from
# appearing as a literal under `src/`, `spikes/` or `tests/` -- and a requirement constant here
# would be exactly that second copy. What FR5.1's "exactly, never at least" actually forbids is a
# realized count that silently differs from the declared one, and that is enforced where the
# realized count exists: `corpus/benign.py` aborts when the two disagree in either direction.

FRAME_ID_FIELD: Final[str] = "frame_id"
"""The field inside `[benign_frame]` that carries the digest of the rest of the block."""

MINIMUM_BENIGN_CODE_REPOSITORIES: Final[int] = 50
"""FR5.1's floor on how many repositories B-code is drawn from.

Not a taste: files from one repository share a language, a style and a base64 idiom, so 500 files
drawn 50 at a time from 10 repositories carry a design effect that puts the effective n near 150
and widens every B-code interval by roughly a factor of two. The frame declares its own floor and
this constant refuses a frame that declares a smaller one; the **realized** count is a separate
question, answered by the builder against the frame's number.
"""

MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY: Final[int] = 10
"""FR5.1's cap on how many files any one repository may contribute. Same reason, other side."""

BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE: Final[str] = "layer_decode_candidate"
"""The one declared file-eligibility rule: the layer's decode stage examines a run in this file.

A closed vocabulary of one, for the reason `window_policy` is one: the rule decides which files
become corpus rows, so it belongs in the frame that is hashed rather than in whichever module
happened to implement it first. The predicate itself lives in `corpus/benign.py`, which is where
the canonicalization layer may be imported.
"""

BENIGN_CODE_ELIGIBILITIES: Final[tuple[str, ...]] = (
    BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
)

BENIGN_CODE_CLASS: Final[str] = "b_code"
BENIGN_CHAT_CLASS: Final[str] = "b_chat"
"""The two benign classes, as the names of the frame's two sub-tables.

`nbc.schema.BENIGN_CLASSES` is the vocabulary's home and this module may not import it -- it is a
leaf over `nbc.errors` and a test holds that. So the two names are spelled here and
`tests/test_pins.py` compares them against `schema.BENIGN_CLASSES`, which makes the duplication a
comparison rather than a second opinion.
"""

_REDUCED_PRECISION: Final[re.Pattern[str]] = re.compile(
    r"fp16|float16|bf16|bfloat16|mixed|int8|uint8|quant|_q4|_q8", re.IGNORECASE
)
_SHA: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{40}\Z")
_REPOSITORY: Final[re.Pattern[str]] = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z", re.ASCII
)
r"""A `namespace/name` id, ASCII only, each segment starting with an alphanumeric.

`re.ASCII` and the explicit class are the whole point: `\w` is Unicode-aware, so the previous
pattern admitted Cyrillic homoglyphs, and a repository id reaches an API URL. Requiring an
alphanumeric first character also refuses the `..` and `.` segments that traverse that URL.
"""

_FIELD_REFERENCE: Final[str] = "::"
"""How a pin names a field inside a file it also pins: `<pinned path>::<field>`.

The separator is the whole convention, and it is what lets a rule check that the file a window
was read from is the file this baseline actually fetches, without a second module spelling out
a filename that belongs in `pins.toml`.
"""

_HTTP_TIMEOUT_SECONDS: Final[float] = 30.0

_KIND_ENDPOINT: Final[Mapping[str, str]] = {"model": "models", "dataset": "datasets"}
_KIND_CACHE_PREFIX: Final[Mapping[str, str]] = {"model": "models", "dataset": "datasets"}


class PinsFileInvalid(NbcError, exit_code=4):
    """`pins.toml` is missing, unparseable, or declares something it may not declare.

    Code 4 rather than 2: `argparse` exits 2 on a usage error and this module has a command
    line, so 2 stays unclaimed. 3 belongs to the platform floor.

    Every structural problem in the file collects under this one abort, and the message carries
    all of them at once: a file that is wrong in three places should say so in one run.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            f"{PINS_FILENAME} is not a usable pin file:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
        self.problems: tuple[str, ...] = problems


class BaselineSetInvalid(NbcError, exit_code=5):
    """The file is well formed and the baseline set it declares cannot support SC5.

    Distinct from `PinsFileInvalid` because the remedy is distinct: nothing here is malformed,
    and the fix is a pin decision -- an ineligible or duplicated baseline is *replaced*, never
    removed, since the set already sits on its floor of two.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "the pinned baseline set cannot support the independence claim:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
        self.problems: tuple[str, ...] = problems


class BaselineIneligible(NbcError, exit_code=7):
    """A pinned baseline may not be scored over the pinned corpus at all.

    Distinct from `BaselineSetInvalid` (5) because the remedy is different in kind. A thin or
    non-independent set is fixed by editing the set; an ineligible baseline is fixed by finding
    another model, and that replacement has its own bar to clear.

    **Replaced, never removed.** SC5's floor is two baselines and the run sits exactly on it, so
    dropping one does not restore eligibility -- it breaks the independence claim outright.
    """

    REPLACEMENT_BAR: ClassVar[str] = (
        "a replacement is required, not a removal: two baselines is SC5's floor and the run "
        "sits exactly on it. Any replacement clears the same bar -- an ONNX graph inside the "
        "repository, a fast tokenizer artifact, a resolvable id2label, an architecture and "
        "tokenizer family not already pinned, and this lineage check"
    )

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "a pinned baseline is ineligible to be scored over the pinned corpus:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + f"\n{self.REPLACEMENT_BAR}"
        )
        self.problems: tuple[str, ...] = problems


class PinMismatch(NbcError, exit_code=6):
    """A pinned revision no longer resolves to the recorded commit, or resolves to nothing.

    Nothing in the repository is wrong when this fires; the world moved. It aborts before any
    inference, because a table computed over an artifact that is not the pinned one is a table
    whose comparability claim is false and whose numbers look exactly the same.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "a pinned revision does not resolve to the recorded commit:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
        self.problems: tuple[str, ...] = problems


# --- the records ------------------------------------------------------------------------------
#
# These live here rather than in `schema.py`, which enumerates the record types the corpus, the
# harness and the report exchange and contains no pin record. A pin is configuration this module
# owns, in the same way `platform.py` owns the floor's dataclasses.


@dataclass(frozen=True, slots=True)
class Licence:
    """What a pinned source declares, recorded as found -- including declaring nothing."""

    identifier: str
    source: str
    attribution: str
    redistributed: bool
    unresolved: str = ""

    @property
    def declared(self) -> bool:
        return self.identifier != NOT_DECLARED

    @property
    def blocks_redistribution(self) -> bool:
        """Material this repository ships whose licence nobody has established.

        FR5.2 makes this a build abort in the corpus story. It is surfaced here, at load, because
        the pin file is where the fact is recorded and a fact recorded beside a value that nothing
        compares it to is the defect this epic is full of.
        """
        return self.redistributed and not self.declared

    def as_run_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "identifier": self.identifier,
            "source": self.source,
            "attribution": self.attribution,
            "redistributed": self.redistributed,
        }
        if self.unresolved:
            # Published rather than kept local: a reader of results.json is entitled to see that
            # this repository redistributes material under a licence nobody established.
            fields["unresolved"] = self.unresolved
        return fields


@dataclass(frozen=True, slots=True)
class WindowPin:
    """A window length together with the file that declared it, and the revision that was read.

    The source travels with the value because "the model's maximum sequence length" resolves
    from three files that routinely disagree, and a length with no stated origin is a number the
    next reader re-derives from whichever file they open first.

    `confirmed_revision` is the gate and `confirmed_on` is the metadata, for the reason the
    lineage block already learned: a publisher's card can declare an operative window smaller
    than the graph's positional capacity, that reading is a human's, and a date alone would let
    it keep looking fresh while describing an artifact this file no longer pins.
    """

    length: int
    source: str
    confirmed_on: str
    confirmed_revision: str

    def as_run_fields(self) -> dict[str, object]:
        return {
            "length": self.length,
            "source": self.source,
            "confirmed_on": self.confirmed_on,
            "confirmed_revision": self.confirmed_revision,
        }


@dataclass(frozen=True, slots=True)
class Oq2Check:
    """What OQ2 measured about this baseline, and the artifact it measured.

    OQ2 asks whether a baseline is strong enough on *clean* text for its degradation under
    encoding to mean anything. The answer is a measurement, not a reading, so the number is
    recorded next to the pin it belongs to rather than left in a commit message or a spike's
    scrollback.

    `decided_revision` is the gate and `decided_on` is the metadata, for the third time in this
    file and for the same reason: a recall measured against a revision this file no longer pins
    is a check nobody re-ran, and a pin can move on the same day.

    `sample_size` is the OQ2 draw and nothing else reads it. The corpus draw is a separate
    declaration made by the story that builds the corpus.

    **The order after a pin moves is: declare, then measure.** The block is required, so a pin
    that moves makes this file unloadable until the declaration names the new revision -- which
    is the same sequencing `window.confirmed_revision` already imposes, and it is deliberate: it
    is what stops a recall measured against the old artifact from being inherited in silence.
    Measure against a *copy* of this file carrying the new revision -- the spike `source` names
    takes the pins root as an argument for exactly this -- and update the committed file with the
    revision and the measurement together, so it is never left declaring a recall it does not
    have. A `pending` outcome is not admitted, because an outcome nothing can act on is a hole in
    a gate rather than a state of it.
    """

    outcome: str
    decided_on: str
    decided_revision: str
    dataset_revision: str
    measured_at_threshold: float
    hits: int
    clean_recall: float
    sample_size: int
    overlap_rows: int
    judged_sufficient_by: str
    source: str

    @property
    def ceiling(self) -> float:
        """The recall as measured, which is an upper bound when `overlap_rows` is not zero."""
        return self.clean_recall

    @property
    def floor(self) -> float:
        """The recall if *every* overlapping row were a hit by memory rather than detection.

        FR3.3's removal belongs to the corpus build, so a recall measured before it runs is
        inflated by an unknown amount bounded above by the rows that reach the baseline's
        training data. Publishing the ceiling alone invites the reader to discover the bound
        themselves, which is the worst available outcome for a project about honest measurement.
        """
        if self.overlap_rows == 0:
            # No overlap means no inflation, so the floor IS the measured value. Re-deriving it
            # from the integers here would publish a floor differing from the ceiling in the
            # fourth decimal purely because one is rounded and the other is not.
            return self.clean_recall
        remaining = self.sample_size - self.overlap_rows
        if remaining <= 0:
            return 0.0
        return max(0, self.hits - self.overlap_rows) / remaining

    def as_run_fields(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "decided_on": self.decided_on,
            "decided_revision": self.decided_revision,
            "dataset_revision": self.dataset_revision,
            "measured_at_threshold": self.measured_at_threshold,
            "hits": self.hits,
            "clean_recall": self.clean_recall,
            "sample_size": self.sample_size,
            "overlap_rows": self.overlap_rows,
            # Published together, always. The ceiling on its own is the number a reader would
            # otherwise have to bound themselves from caveat 3d's disclosed overlap.
            "ceiling": self.ceiling,
            "floor": self.floor,
            "judged_sufficient_by": self.judged_sufficient_by,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Lineage:
    """What a baseline's card declares, read once by a human and recorded as data.

    Two maps, because they answer two different questions. `attack_datasets` is the baseline's
    relationship to each pinned attack dataset -- the thing that will actually be scored.
    `training_sources` is its relationship to each source a pinned dataset's own card names as a
    seed, which is how the one-hop reach is computed: nothing on a model card mentions a dataset
    built downstream of it, so the hop is only visible from the two declarations together.

    Every seed must be answered for; sources beyond them are admitted rather than refused. A
    card's full `datasets:` block is what the measured-overlap filter will draw its exclusion
    set from, and refusing the ones that are nobody's seed today would mean deleting a read
    fact to satisfy a check.

    The date and the card revision are part of the record: a lineage check that was never re-run
    after a pin changed has to be *visible*, not silently assumed to still hold.
    """

    checked_on: str
    card_revision: str
    attack_datasets: Mapping[str, str]
    training_sources: Mapping[str, str]

    def relationship_to(self, repository: str) -> str | None:
        return self.attack_datasets.get(repository)

    def trains_on(self, repository: str) -> bool:
        """Whether the card declares this source among the baseline's training data."""
        wanted = _canonical(repository)
        return any(
            _canonical(declared) == wanted and relationship == TRAINED_ON
            for declared, relationship in self.training_sources.items()
        )

    def as_run_fields(self) -> dict[str, object]:
        return {
            "checked_on": self.checked_on,
            "card_revision": self.card_revision,
            "attack_datasets": dict(self.attack_datasets),
            "training_sources": dict(self.training_sources),
        }



def _canonical(value: str) -> str:
    """The form two declarations are compared in, never the form either is published in.

    Identity in this file is decided by string equality in two places that matter: SC5's
    architecture/tokenizer family pair, and the repository ids the one-hop lineage reach joins
    on. Both were comparing spellings. `DeBERTa-v2` and `deberta_v2` are one family declared
    two ways, and the pair check reads them as two families -- passing SC5 while the run has one
    architecture measured twice. Hugging Face resolves repository ids case-insensitively, so
    `VMware/open-instruct` and `vmware/open-instruct` are one source, and the reach that joins
    them missed it.

    The raw spelling is what `as_run_fields` publishes; this is only ever the comparison key.
    """
    # Separators are REMOVED rather than normalized to one of them, which is what makes this
    # closed under both spellings that occur in practice. Folding CamelCase to a separator fixes
    # `DebertaV2` and breaks `SentencePiece`, which becomes `sentence-piece`; dropping separators
    # entirely collapses `DebertaV2`, `deberta-v2`, `deberta_v2` and `SentencePiece Unigram` onto
    # the forms the pinned file already uses, with no case left where one spelling of a family
    # reads as a second family.
    return re.sub(r"[\s_-]+", "", value.strip().casefold())


def canonical_source_id(value: str) -> str:
    """The comparison key a repository id is joined on, for a caller outside this module.

    The corpus build joins `required_exclusion_sources()` against the `[[exclusion_source]]`
    array, which is the same join `_canonical` already does inside this file for the one-hop
    reach. A second spelling-folding rule in `corpus/` would be a second answer to "are these two
    ids the same source", and the one place that question has ever been answered wrongly here is
    exactly a second answer.
    """
    return _canonical(value)


@dataclass(frozen=True, slots=True)
class Provenance:
    """What a pinned dataset's *own card* says it was built from, and when that was read.

    A dataset seeded from a model's training data is that training data at one remove. The model
    cards cannot show it and the dataset card can, so the seeds are pinned here as data and the
    one-hop check reads them rather than a person re-reading a card every time a pin moves.

    An empty `seeds` tuple is a fact -- a card that names no seed -- not a missing declaration.
    The block itself is mandatory; what it contains is whatever the card says.
    """

    checked_on: str
    card_revision: str
    seeds: tuple[str, ...]

    def as_run_fields(self) -> dict[str, object]:
        return {
            "checked_on": self.checked_on,
            "card_revision": self.card_revision,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class RemoteArtifact:
    """Something with a repository and a revision that a resolver can be asked about."""

    kind: str
    repository: str
    revision: str

    def __str__(self) -> str:
        return f"{self.kind} {self.repository}@{self.revision}"

    @property
    def cache_directory(self) -> str:
        """The Hugging Face cache directory name for this artifact.

        `org/name` becomes `models--org--name`; the layout is the hub's, not ours.
        """
        prefix = _KIND_CACHE_PREFIX[self.kind]
        return f"{prefix}--" + self.repository.replace("/", "--")

    def snapshot_dir(self, cache_root: Path | None = None) -> Path:
        """Where this machine holds the pinned revision's files, if it holds them at all.

        The hub names a snapshot directory by the commit it was fetched at, so this path is
        both the resolution of the pin and the root every pinned artifact path hangs off. It
        lives here, once, because a second module spelling out `snapshots/<revision>/` is a
        second place the hub's layout can drift away from ours.
        """
        root = cache_root if cache_root is not None else hf_cache_root()
        return root / self.cache_directory / "snapshots" / self.revision

    @property
    def repository_url(self) -> str:
        """The hub's API endpoint for the repository, with no revision in it.

        Split out from `api_url` because one artifact this project pins has no revision to name:
        an access-restricted exclusion source answers 401 and hands back no commit, so the only
        URL that can be asked about it is this one. Both forms are built here so the hub's API
        layout has a single home.
        """
        endpoint = _KIND_ENDPOINT[self.kind]
        return f"https://huggingface.co/api/{endpoint}/{self.repository}"

    @property
    def api_url(self) -> str:
        return f"{self.repository_url}/revision/{self.revision}"


@dataclass(frozen=True, slots=True)
class Baseline:
    """One pinned model, down to the files inside it."""

    key: str
    repository: str
    revision: str
    threshold: float
    graph_path: str
    precision: str
    graph_bytes: int
    tokenizer_path: str
    config_path: str
    architecture_family: str
    tokenizer_family: str
    window_policy: str
    window: WindowPin
    licence: Licence
    lineage: Lineage
    oq2: Oq2Check

    @property
    def artifact(self) -> RemoteArtifact:
        return RemoteArtifact("model", self.repository, self.revision)

    @property
    def family_pair(self) -> tuple[str, str]:
        """The pair SC5 rests on. Two baselines sharing it cannot corroborate each other."""
        return (_canonical(self.architecture_family), _canonical(self.tokenizer_family))

    def as_run_fields(self) -> dict[str, object]:
        return {
            "key": self.key,
            "repository": self.repository,
            "revision": self.revision,
            "threshold": self.threshold,
            "graph_path": self.graph_path,
            "precision": self.precision,
            "graph_bytes": self.graph_bytes,
            "tokenizer_path": self.tokenizer_path,
            "config_path": self.config_path,
            "architecture_family": self.architecture_family,
            "tokenizer_family": self.tokenizer_family,
            "window_policy": self.window_policy,
            "window": self.window.as_run_fields(),
            "licence": self.licence.as_run_fields(),
            "lineage": self.lineage.as_run_fields(),
            "oq2": self.oq2.as_run_fields(),
        }


@dataclass(frozen=True, slots=True)
class AttackDraw:
    """How many attack positives are drawn from a pinned dataset, and by what rule.

    Every field is declared; none is defaulted. A default draw is a draw nobody declared, and the
    whole point of writing it here is that a reviewer can read the rule without reading the code.

    **The size is in positives.** The pinned pool is mixed-label and holds more than three times
    as many rows as positives, so a size read as rows would draw roughly a third of what it says.
    The key name carries the unit rather than a comment doing it.

    `declared_on` is the date a human wrote this block. Nothing *compares* it to anything, and it is
    not decoration either: story 3.6's `corpus/manifest.py::build_id` folds the whole build
    declaration -- this block included -- into the corpus' identity, so re-declaring the draw on a
    later date produces a different `build_id` and the committed corpus stops being readable until
    it is rebuilt. The date is part of what a changed corpus is distinguishable by.

    The two methods each admit exactly one companion field, and the reader refuses the other:
    `head` consumes `sort_key` and `seeded_random` consumes `seed`. A declaration carrying a
    parameter its own method ignores is refused rather than tolerated, because an ignored seed
    reads as a declared draw and is not one.
    """

    declared_on: str
    sample_size_positives: int
    method: str
    seed: int | None
    sort_key: str | None

    def as_run_fields(self) -> dict[str, object]:
        return {
            "declared_on": self.declared_on,
            "sample_size_positives": self.sample_size_positives,
            "method": self.method,
            "seed": self.seed,
            "sort_key": self.sort_key,
        }


@dataclass(frozen=True, slots=True)
class AttackDataset:
    """One pinned attack dataset, by identity and by draw.

    `splits` is the whole set the counts are taken over, never one of them: the build compares
    this list against the splits the reader actually yields, in both directions, because reading
    a split as the dataset is the same error as reading a row count as a positive count.
    """

    key: str
    repository: str
    revision: str
    splits: tuple[str, ...]
    attack_label: int
    draw: AttackDraw
    licence: Licence
    provenance: Provenance

    @property
    def artifact(self) -> RemoteArtifact:
        return RemoteArtifact("dataset", self.repository, self.revision)

    def as_run_fields(self) -> dict[str, object]:
        return {
            "key": self.key,
            "repository": self.repository,
            "revision": self.revision,
            "splits": list(self.splits),
            "attack_label": self.attack_label,
            "draw": self.draw.as_run_fields(),
            "licence": self.licence.as_run_fields(),
            "provenance": self.provenance.as_run_fields(),
        }


@dataclass(frozen=True, slots=True)
class BenignCodeRepository:
    """One public source repository B-code is drawn from, pinned by repository and commit sha.

    **Not a `RemoteArtifact`.** `remote_artifacts()` is the set `verify_revisions` asks the
    Hugging Face hub about, and these are git repositories on a different host with a different
    API. Putting them in there would ask the wrong server about the wrong id and report the
    resulting `None` as a pin that could not be resolved. What verifies these instead is the build
    itself: it fetches the archive **at this sha**, and a sha that no longer exists is a 404 rather
    than a silent fallback to a branch head.

    The licence is recorded here because story 3.8's redistribution gate reads it; nothing in this
    story aborts on it. `redistributed` is `true` for every entry, because B-code rows are the file
    text and they land in `data/*.jsonl` in an MIT repository.
    """

    key: str
    repository: str
    revision: str
    licence: Licence

    @property
    def archive_url(self) -> str:
        """The gzipped tar of the whole repository at the pinned sha.

        `codeload` rather than the REST API on purpose: listing a tree through the API costs one
        request against an unauthenticated hourly budget of sixty, and the frame declares more
        repositories than that budget would allow. This endpoint carries no such budget, and one
        request per repository yields every path and every byte the draw needs.
        """
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"

    def file_source(self, path: str) -> str:
        """How one drawn file names itself in the corpus: repository, commit sha and path.

        This string is the `source` field of the `CorpusItem`, so FR5.1's "pinned by repository,
        commit sha and path" is a property of the committed corpus row rather than of a build log.
        """
        return f"github.com/{self.repository}@{self.revision}:{path}"

    def as_run_fields(self) -> dict[str, object]:
        return {
            "key": self.key,
            "repository": self.repository,
            "revision": self.revision,
            "licence": self.licence.as_run_fields(),
        }


@dataclass(frozen=True, slots=True)
class BenignCodeFrame:
    """How B-code is drawn: from which repositories, which files, and how many of each.

    Every field decides which rows the corpus ends up holding, which is why all of them sit inside
    the block `frame_id` hashes. `min_repositories` and `max_files_per_repository` are the frame's
    own copies of FR5.1's floor and cap, compared at load against the requirement constants above;
    the **realized** repository count is the builder's output and is compared against
    `min_repositories` there.
    """

    min_repositories: int
    max_files_per_repository: int
    eligibility: str
    min_file_bytes: int
    max_file_bytes: int
    file_extensions: tuple[str, ...]
    repositories: tuple[BenignCodeRepository, ...]

    def as_run_fields(self) -> dict[str, object]:
        return {
            "min_repositories": self.min_repositories,
            "max_files_per_repository": self.max_files_per_repository,
            "eligibility": self.eligibility,
            "min_file_bytes": self.min_file_bytes,
            "max_file_bytes": self.max_file_bytes,
            "file_extensions": list(self.file_extensions),
            "repositories": [entry.as_run_fields() for entry in self.repositories],
        }


@dataclass(frozen=True, slots=True)
class BenignChatFrame:
    """How B-chat is drawn: from the pinned dataset's benign rows, plus a closed hand-authored set.

    The dataset is **not named here**. It is the pinned attack dataset, whose benign-labelled rows
    are what FR5.1 draws from, and naming it a second time in this block would be a second home for
    a repository id that `[[attack_dataset]]` already pins -- the exact drift `pins.toml`'s own
    header refuses. `build_id` covers the attack draw, so the dataset's identity is inside the build
    identity either way.

    `hand_authored_items` is the size of the allowance FR5.1 grants for material no public dataset
    carries: messages legitimately containing a JWT, a content hash, a data URI or an SSH public
    key. It is a **declared count compared against what `corpus/sources/` actually holds**, so an
    item added there without widening the frame fails the build rather than quietly enlarging the
    hand-authored share of the counter-metric.
    """

    hand_authored_items: int

    def as_run_fields(self) -> dict[str, object]:
        return {"hand_authored_items": self.hand_authored_items}


@dataclass(frozen=True, slots=True)
class ConfirmatoryCell:
    """The one `(baseline, dressing_chain, benign_class)` cell the N1 verdict rests on.

    Declared here, in the epic that builds the corpus, and hashed into `build_id` with everything
    else, so the cell cannot be chosen after the numbers exist. This story declares it, checks its
    shape and hashes it. **Story 3.9 owns the rest**: that the chain must be held out or nested past
    the recursion ceiling and never a bound one, and the reason. That split is deliberate and is
    FR5.4's own: declaration and hashing live in the corpus epic, emission and evaluation in the
    measurement epic, because a requirement whose pre-registration half lands in the epic that
    evaluates it is a requirement that gets pre-registered after measuring.

    `baseline` is a baseline **key**, and `load_pins` refuses one that names no declared baseline.
    `benign_class` and `dressing_chain` are checked for shape here and against the corpus
    vocabulary in `corpus/manifest.py`, which is the module allowed to import `nbc.schema` and
    `nbc.corpus.matrix`; this one is a leaf over `nbc.errors`.
    """

    declared_on: str
    baseline: str
    dressing_chain: str
    benign_class: str

    def as_run_fields(self) -> dict[str, object]:
        return {
            "declared_on": self.declared_on,
            "baseline": self.baseline,
            "dressing_chain": self.dressing_chain,
            "benign_class": self.benign_class,
        }


@dataclass(frozen=True, slots=True)
class BenignFrame:
    """FR5.1's sampling frame: fixed, hashed and declared before anything is measured.

    The suspicion this exists to answer is that the benign corpus was grown until the number looked
    reasonable. What answers it is not a promise but `frame_id`: the digest of this whole block,
    written in the block, recomputed by `load_pins`, and refused on mismatch -- so an edit to any
    field of the frame that is not accompanied by a new digest stops the run, and an edit that *is*
    accompanied by one is a visible diff against a corpus whose manifest still records the old id.

    The draw vocabulary is the attack draw's, read by the same function, so "the same
    selection-method vocabulary as the attack draw" is a shared implementation rather than two
    declarations that happen to agree today.
    """

    declared_on: str
    sample_size_items: int
    method: str
    seed: int | None
    sort_key: str | None
    frame_id: str
    b_code: BenignCodeFrame
    b_chat: BenignChatFrame
    confirmatory_cell: ConfirmatoryCell

    def as_run_fields(self) -> dict[str, object]:
        return {
            "declared_on": self.declared_on,
            "sample_size_items": self.sample_size_items,
            "method": self.method,
            "seed": self.seed,
            "sort_key": self.sort_key,
            FRAME_ID_FIELD: self.frame_id,
            BENIGN_CODE_CLASS: self.b_code.as_run_fields(),
            BENIGN_CHAT_CLASS: self.b_chat.as_run_fields(),
            "confirmatory_cell": self.confirmatory_cell.as_run_fields(),
        }


def frame_digest(frame: Mapping[str, Any]) -> str:
    """The digest `frame_id` must equal: SHA-256 over the frame block with `frame_id` removed.

    Over the **raw table as the file declares it**, not over the parsed dataclasses, and that is
    the point: a field added to the block that no reader consumes yet still changes the digest, so
    the frame cannot grow a parameter behind the id that is supposed to fix it.

    Order-sensitive on lists, unlike `exclusion.declaration_digest`, which sorts. Reordering the
    repository array draws the same corpus, so a sorted digest would be defensible -- but the frame
    is a declaration a reviewer reads, and this errs toward refusing an edit that changes nothing
    rather than admitting one that changes something. The conservative direction is the one that
    fails loudly.

    `default=str` renders the TOML date and datetime values `tomllib` produces; every other value
    in the block is a string, an integer, a boolean or a container of those.
    """
    payload = {key: value for key, value in frame.items() if key != FRAME_ID_FIELD}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExclusionSource:
    """One training source the corpus build must remove its overlap with, pinned.

    The set these entries form is **derived** from the two lineage blocks and refused when it
    disagrees with them (`Pins.derived_exclusion_sources()`), so this array carries identity and
    reachability and never the reason -- a restated reason is a second place the obligation can
    drift from the declarations that create it.

    `revision` is absent exactly when `availability` is `unavailable`. A source that answers 401
    hands back no commit, so there is no sha to pin; admitting a placeholder would put a fake
    revision under a `revision` key. The absence is never silent: an `available` entry without a
    revision is refused, and an `unavailable` entry with one is refused, because a sha for a
    source nobody can read came from somewhere this file cannot name.

    `http_status` is the declared status, and it is not decoration: the build probes `probe_url`
    and compares what it observes against this number, so `Harelix/...`'s 401 is a checked fact
    rather than a sentence in a comment.

    The licence is declared here for the same reason it is declared on a baseline whose weights
    this repository never ships: AD-34's rule is that **every** pinned source carries one, so a
    source arrives with its licence read rather than with the question deferred. `redistributed`
    is `false` for every entry -- an exclusion source's rows are intersected against the corpus
    and the matches are *removed*, so no byte of one can reach `data/`. What that makes the field
    is a checkable claim rather than a formality: `corpus/attribution.py` refuses a row whose
    source is not one of the pinned redistributing identities, so an exclusion source that ever
    did contribute a row would abort the build rather than appear uncredited.
    """

    key: str
    repository: str
    revision: str
    availability: str
    http_status: int
    checked_on: str
    evidence: str
    licence: Licence

    @property
    def resolvable(self) -> bool:
        """Whether this source declares a revision the ordinary pin verification can resolve.

        True for `unreadable` too: the sha resolves and must keep resolving even though nothing
        can read the rows at it. That is the whole reason `unreadable` is not `unreachable`.
        """
        return bool(self.revision)

    @property
    def loadable(self) -> bool:
        """Whether the pinned reader is expected to return this source's rows."""
        return self.availability == EXCLUSION_AVAILABLE

    @property
    def artifact(self) -> RemoteArtifact:
        return RemoteArtifact("dataset", self.repository, self.revision)

    @property
    def probe_url(self) -> str:
        """What the build asks the hub about: the pinned revision, or the bare repository.

        A source with no revision still has to be probed -- that is the whole point of declaring
        it -- and the only question the hub can answer about it is whether the repository is
        reachable at all.
        """
        return self.artifact.api_url if self.revision else self.artifact.repository_url

    def as_run_fields(self) -> dict[str, object]:
        return {
            "key": self.key,
            "repository": self.repository,
            "revision": self.revision,
            "availability": self.availability,
            "http_status": self.http_status,
            "checked_on": self.checked_on,
            "evidence": self.evidence,
            "licence": self.licence.as_run_fields(),
        }


def _derived_exclusion_sources(
    baselines: Sequence["Baseline"], attack_datasets: Sequence["AttackDataset"]
) -> dict[str, str]:
    """Canonical id -> raw spelling, for every source the build owes the corpus a removal against.

    Two contributors, both read from declarations already in this file:

    - every source a baseline declares `trained-on`, because a corpus row that appears in one is
      text that baseline was taught, and the counter-metric is the number that suffers most;
    - every source a pinned dataset's own card names as a seed, because the dataset is that
      source at one remove whether or not any baseline declares the hop today.

    Derived rather than restated, so the `[[exclusion_source]]` array cannot drift away from the
    declarations that create the obligation. `Pins.required_exclusion_sources()` is the strictly
    smaller set the *declared hops* oblige, and it is what the build must actually download.
    """
    derived: dict[str, str] = {}
    for baseline in baselines:
        for source, relationship in baseline.lineage.training_sources.items():
            if relationship == TRAINED_ON:
                derived.setdefault(_canonical(source), source)
    for dataset in attack_datasets:
        for seed in dataset.provenance.seeds:
            derived.setdefault(_canonical(seed), seed)
    return {key: value for key, value in derived.items() if key}


@dataclass(frozen=True, slots=True)
class Pins:
    """Every remote artifact this project touches, as data."""

    schema_version: int
    verified_on: str
    verified_against: str
    baselines: tuple[Baseline, ...]
    attack_datasets: tuple[AttackDataset, ...]
    exclusion_sources: tuple[ExclusionSource, ...]
    benign_frame: BenignFrame
    path: Path

    def remote_artifacts(self) -> tuple[RemoteArtifact, ...]:
        """Every pinned artifact whose revision `verify_revisions` can ask the world about.

        Exclusion sources are in here, and that is the point: their revisions decide which rows
        survive into the corpus, so a pin that quietly stopped resolving has to fail the same
        gate every other pin fails. The one that declares no revision is not in here, because
        there is no sha to compare -- the corpus build probes it instead, and compares the
        status it observes against the status this file declares.
        """
        return tuple(
            [baseline.artifact for baseline in self.baselines]
            + [dataset.artifact for dataset in self.attack_datasets]
            + [source.artifact for source in self.exclusion_sources if source.resolvable]
        )

    def derived_exclusion_sources(self) -> tuple[str, ...]:
        """Every source the lineage and provenance blocks imply the build must remove against.

        `load_pins` refuses a file whose `[[exclusion_source]]` array is not exactly this set, so
        adding a training source or a seed and forgetting to pin it is a load failure rather than
        a source the filter silently never sees.
        """
        return tuple(
            sorted(_derived_exclusion_sources(self.baselines, self.attack_datasets).values())
        )

    def one_hop_reaches(self) -> dict[tuple[str, str], tuple[str, ...]]:
        """Every (baseline, dataset) pair the seeds connect, and the seeds that connect it.

        A pair appears only when the baseline declares training on at least one source the
        dataset's own card names as a seed. This is the computation the gate refuses to leave
        to a reader, and it is also what the corpus build consumes.
        """
        reaches: dict[tuple[str, str], tuple[str, ...]] = {}
        for baseline in self.baselines:
            for dataset in self.attack_datasets:
                seeds = tuple(
                    seed
                    for seed in dataset.provenance.seeds
                    if baseline.lineage.trains_on(seed)
                )
                if seeds:
                    reaches[(baseline.repository, dataset.repository)] = seeds
        return reaches

    def required_exclusion_sources(self) -> tuple[str, ...]:
        """The seeds a declared hop obliges the build to remove the corpus' overlap with.

        The obligation is derived from the pins rather than restated beside them, so the corpus
        builder and this gate cannot disagree about which sources the declaration bought. This
        module names them and stops there: downloading them, intersecting them against the
        corpus and dropping the matching rows is the corpus builder's work, not a pin file's.
        """
        return tuple(
            sorted({seed for seeds in self.one_hop_reaches().values() for seed in seeds})
        )

    def as_run_fields(self) -> dict[str, object]:
        """The `pins` block of `results.json`: what was pinned, and when it was verified."""
        return {
            "pins": {
                "schema_version": self.schema_version,
                "verified_on": self.verified_on,
                "verified_against": self.verified_against,
                "baselines": [baseline.as_run_fields() for baseline in self.baselines],
                "attack_datasets": [
                    dataset.as_run_fields() for dataset in self.attack_datasets
                ],
                "exclusion_sources": [
                    source.as_run_fields() for source in self.exclusion_sources
                ],
                # FR5.1's frame, published with everything else it constrains. The `frame_id` in
                # here is what `data/manifest.json` is compared against before a corpus row is
                # handed to anything that measures.
                "benign_frame": self.benign_frame.as_run_fields(),
                # Derived, not declared: a reader can recompute it from the two blocks above,
                # and the corpus build reads it rather than a second copy of the same list.
                "required_exclusion_sources": list(self.required_exclusion_sources()),
                # The wider set: what a declared hop obliges is a subset of what the two lineage
                # blocks imply, and the array above must name exactly this.
                "derived_exclusion_sources": list(self.derived_exclusion_sources()),
            }
        }


# --- reading the file -------------------------------------------------------------------------
#
# Every problem is collected before aborting. A file that is wrong in three places should tell a
# reader all three in one run, the same way the platform preflight does.


class _Reader:
    """Typed access to a TOML table that records what was wrong instead of raising."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems

    def note(self, problem: str) -> None:
        self.problems.append(problem)

    def table(self, parent: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
        value = parent.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return {}
        if not isinstance(value, dict):
            self.note(f"{where}.{key} must be a table, got {type(value).__name__}")
            return {}
        return value

    def string(self, table: Mapping[str, Any], key: str, where: str) -> str:
        value = table.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return ""
        if not isinstance(value, str):
            self.note(f"{where}.{key} must be a string, got {type(value).__name__}")
            return ""
        if not value:
            self.note(f"{where}.{key} is empty")
        return value

    def integer(self, table: Mapping[str, Any], key: str, where: str) -> int:
        value = table.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            self.note(f"{where}.{key} must be an integer, got {type(value).__name__}")
            return 0
        return value

    def boolean(self, table: Mapping[str, Any], key: str, where: str) -> bool:
        value = table.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return False
        if not isinstance(value, bool):
            self.note(f"{where}.{key} must be a boolean, got {type(value).__name__}")
            return False
        return value

    def number(self, table: Mapping[str, Any], key: str, where: str) -> float:
        value = table.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.note(f"{where}.{key} must be a number, got {type(value).__name__}")
            return 0.0
        return float(value)

    def strings(self, table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
        value = table.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return ()
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            self.note(f"{where}.{key} must be a list of strings")
            return ()
        if not value:
            self.note(f"{where}.{key} is empty")
        return tuple(value)

    def string_table(
        self, parent: Mapping[str, Any], key: str, where: str, admitted: frozenset[str]
    ) -> dict[str, str]:
        """A `repository -> relationship` table, with the vocabulary checked as it is read.

        The vocabulary is closed because a rule reads these values. A relationship spelled in
        free text is a sentence a human interprets, and interpreting sentences is what this
        whole declaration exists to stop doing.
        """
        value = parent.get(key)
        if value is None:
            self.note(f"{where}.{key} is missing")
            return {}
        if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(item, str) for name, item in value.items()
        ):
            self.note(f"{where}.{key} must be a table of strings")
            return {}
        for name, relationship in sorted(value.items()):
            if relationship not in admitted:
                self.note(
                    f"{where}.{key}[{name!r}] is {relationship!r}, which is not a relationship "
                    f"this file may declare; it must be one of "
                    f"{', '.join(sorted(admitted))}, because the eligibility rule reads this "
                    f"value rather than the prose around it"
                )
        return dict(value)

    def matching(
        self, value: str, pattern: re.Pattern[str], where: str, expected: str
    ) -> str:
        if value and pattern.match(value) is None:
            self.note(f"{where} must be {expected}, got {value!r}")
        return value

    def distinct_strings(
        self, table: Mapping[str, Any], key: str, where: str
    ) -> tuple[str, ...]:
        """A string list with no repeats.

        `splits = ["train", "train"]` reads the pool twice, and the doubled count reaches a
        published recall as its denominator. Order is kept because the read order is declared.
        """
        values = self.strings(table, key, where)
        seen: list[str] = []
        for value in values:
            if value in seen:
                self.note(f"{where}.{key} repeats {value!r}; a split read twice doubles the pool")
            else:
                seen.append(value)
        return tuple(seen)

    def label_value(self, table: Mapping[str, Any], key: str, where: str) -> int:
        """A binary label value.

        The pinned datasets are binary, and every rate in this project is computed by comparing a
        row's label against this number. A value outside {0, 1} matches no row, which silently
        yields a recall over an empty pool rather than an abort.
        """
        value = self.integer(table, key, where)
        if value not in (0, 1):
            self.note(f"{where}.{key} must be 0 or 1, got {value!r}")
        return value

    def calendar_date(self, value: str, where: str) -> str:
        """An ISO date that exists on a calendar.

        `_DATE` checked the shape and nothing else, so `2026-13-45` satisfied every field that
        records *when a human checked something* -- the one class of field whose whole purpose is
        to be compared against a real date later.
        """
        if not value:
            return value
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            self.note(
                f"{where} must be an ISO calendar date (YYYY-MM-DD) that exists, got {value!r}"
            )
        return value

    def contained_path(self, value: str, where: str) -> str:
        """A path that stays inside the snapshot the pin verified.

        `Path("/snapshot") / "/etc/passwd"` is `/etc/passwd`: an absolute right operand silently
        discards the left one. So an absolute `graph_path` escapes the directory whose revision
        was checked, and every downstream join inherits it. Refused here, once, because two
        modules join these paths and a check in either leaves the other open.
        """
        if not value:
            return value
        pure = PurePosixPath(value)
        if pure.is_absolute() or value.startswith("\\") or ":" in value.split("/")[0]:
            self.note(f"{where} must be relative to the pinned snapshot, got {value!r}")
        elif ".." in pure.parts:
            self.note(f"{where} must not traverse upwards, got {value!r}")
        elif "." in pure.parts or value != str(pure):
            self.note(
                f"{where} must be a normalized relative path, got {value!r} "
                f"(normalizes to {str(pure)!r})"
            )
        return value


def _read_licence(reader: _Reader, parent: Mapping[str, Any], where: str) -> Licence:
    table = reader.table(parent, "licence", where)
    at = f"{where}.licence"
    identifier = reader.string(table, "identifier", at)
    redistributed = reader.boolean(table, "redistributed", at)
    unresolved = str(table.get("unresolved", "")).strip()
    if redistributed and identifier == NOT_DECLARED and not unresolved:
        reader.note(
            f"{at} redistributes material under an undeclared licence. Either declare the "
            f"identifier, or record the open question in an `unresolved` field stating the date "
            f"and what has to happen: contact the publisher, find a licensed source, or state a "
            f"redistribution position. FR5.2 makes this a build abort in the corpus story; it is "
            f"refused here so the decision is taken before the corpus is built, not after."
        )
    return Licence(
        identifier=identifier,
        source=reader.string(table, "source", at),
        attribution=reader.string(table, "attribution", at),
        redistributed=redistributed,
        unresolved=unresolved,
    )


def _read_baseline(reader: _Reader, table: Mapping[str, Any], index: int) -> Baseline:
    where = f"baseline[{index}]"
    key = reader.string(table, "key", where)
    if key:
        where = f"baseline[{index}] ({key})"

    repository = reader.matching(
        reader.string(table, "repository", where),
        _REPOSITORY,
        f"{where}.repository",
        "a `namespace/name` repository id",
    )
    revision = reader.matching(
        reader.string(table, "revision", where),
        _SHA,
        f"{where}.revision",
        "a 40-character lowercase hex commit sha",
    )

    threshold = reader.number(table, "threshold", where)
    if "threshold" in table and not 0.0 <= threshold <= 1.0:
        reader.note(f"{where}.threshold must lie in [0, 1], got {threshold!r}")

    graph_path = reader.contained_path(reader.string(table, "graph_path", where), f"{where}.graph_path")
    precision = reader.string(table, "precision", where)
    if precision and precision != PINNED_PRECISION:
        reader.note(
            f"{where}.precision is {precision!r}; only {PINNED_PRECISION!r} may be pinned, "
            f"because reduced precision moves scores in the last decimals and the decision "
            f"threshold turns that into a class flip on the borderline encoded items"
        )
    if graph_path and _REDUCED_PRECISION.search(graph_path) is not None:
        reader.note(
            f"{where}.graph_path is {graph_path!r}, which names a reduced-precision or "
            f"quantized graph; the {PINNED_PRECISION} graph is the pinned one, because reduced "
            f"precision moves scores in the last decimals and the decision threshold turns "
            f"that into a class flip on the borderline encoded items"
        )

    graph_bytes = reader.integer(table, "graph_bytes", where)
    if "graph_bytes" in table and graph_bytes <= 0:
        reader.note(f"{where}.graph_bytes must be positive, got {graph_bytes!r}")

    config_path = reader.contained_path(reader.string(table, "config_path", where), f"{where}.config_path")

    window_policy = reader.string(table, "window_policy", where)
    if window_policy and window_policy not in WINDOW_POLICIES:
        admitted = ", ".join(sorted(WINDOW_POLICIES))
        reader.note(
            f"{where}.window_policy is {window_policy!r}, and the admitted values are "
            f"{admitted}. A policy is a declared strategy -- a window length, a stride and an "
            f"aggregation together -- and a name with no strategy behind it selects whatever "
            f"the fallback happened to be, for every document this baseline ever scores"
        )

    window_table = reader.table(table, "window", where)
    window = WindowPin(
        length=reader.integer(window_table, "length", f"{where}.window"),
        source=reader.string(window_table, "source", f"{where}.window"),
        confirmed_on=reader.calendar_date(
            reader.string(window_table, "confirmed_on", f"{where}.window"),
            f"{where}.window.confirmed_on",
        ),
        confirmed_revision=reader.matching(
            reader.string(window_table, "confirmed_revision", f"{where}.window"),
            _SHA,
            f"{where}.window.confirmed_revision",
            "a 40-character lowercase hex commit sha",
        ),
    )
    if window.length <= 0 and "length" in window_table:
        reader.note(f"{where}.window.length must be positive, got {window.length!r}")
    if window.source and "tokenizer_config.json" in window.source:
        reader.note(
            f"{where}.window.source reads tokenizer_config.json, whose model_max_length is a "
            f"~1e30 sentinel in the pinned repositories; the window comes from the model config"
        )
    if _FIELD_REFERENCE in window.source and config_path:
        named_file = window.source.split(_FIELD_REFERENCE, 1)[0]
        if named_file != config_path:
            reader.note(
                f"{where}.window.source reads a field of {named_file!r} and this baseline pins "
                f"{config_path!r}: a window read from a file this baseline does not pin is a "
                f"window read from a file nobody fetches. One pinned repository ships two "
                f"different files of the same name at one revision, which is why the pin names "
                f"the path and not just the repository"
            )
    _note_stale_check(
        reader,
        f"{where}.window",
        recorded=window.confirmed_revision,
        pinned=revision,
        checked_on=window.confirmed_on,
        what="the model's declared window",
        field="confirmed_revision",
    )

    lineage_table = reader.table(table, "lineage", where)
    lineage = Lineage(
        checked_on=reader.calendar_date(
            reader.string(lineage_table, "checked_on", f"{where}.lineage"),
            f"{where}.lineage.checked_on",
        ),
        card_revision=reader.matching(
            reader.string(lineage_table, "card_revision", f"{where}.lineage"),
            _SHA,
            f"{where}.lineage.card_revision",
            "a 40-character lowercase hex commit sha",
        ),
        attack_datasets=reader.string_table(
            lineage_table, "attack_datasets", f"{where}.lineage", LINEAGE_RELATIONSHIPS
        ),
        training_sources=reader.string_table(
            lineage_table, "training_sources", f"{where}.lineage", TRAINING_SOURCE_RELATIONSHIPS
        ),
    )
    _note_stale_check(
        reader,
        f"{where}.lineage",
        recorded=lineage.card_revision,
        pinned=revision,
        checked_on=lineage.checked_on,
        what="the baseline's card",
    )

    oq2 = _read_oq2(reader, table, where, revision, threshold)

    return Baseline(
        key=key,
        repository=repository,
        revision=revision,
        threshold=threshold,
        graph_path=graph_path,
        precision=precision,
        graph_bytes=graph_bytes,
        tokenizer_path=reader.contained_path(
            reader.string(table, "tokenizer_path", where), f"{where}.tokenizer_path"
        ),
        config_path=config_path,
        architecture_family=reader.string(table, "architecture_family", where),
        tokenizer_family=reader.string(table, "tokenizer_family", where),
        window_policy=window_policy,
        window=window,
        licence=_read_licence(reader, table, where),
        lineage=lineage,
        oq2=oq2,
    )


def _read_oq2(
    reader: _Reader,
    parent: Mapping[str, Any],
    where: str,
    revision: str,
    threshold: float,
) -> Oq2Check:
    """The OQ2 record: what this baseline scored on clean attack text, and against which pin.

    The block is required rather than optional. OQ2 gates the epic and the publication date, so
    a baseline with no recorded answer is a baseline nobody asked -- and an optional block is
    exactly how the lineage check went unre-run through a change of pin.
    """
    table = reader.table(parent, "oq2", where)
    at = f"{where}.oq2"

    outcome = reader.string(table, "outcome", at)
    if outcome and outcome not in OQ2_OUTCOMES:
        reader.note(
            f"{at}.outcome is {outcome!r}, and the admitted values are "
            f"{', '.join(sorted(OQ2_OUTCOMES))}. A baseline too weak on clean text is replaced, "
            f"never removed: SC5's floor is {MINIMUM_BASELINES} baselines and the run sits "
            f"exactly on it, so removal does not weaken that criterion, it fails it"
        )

    decided_on = reader.calendar_date(
        reader.string(table, "decided_on", at),
        f"{at}.decided_on",
    )
    decided_revision = reader.matching(
        reader.string(table, "decided_revision", at),
        _SHA,
        f"{at}.decided_revision",
        "a 40-character lowercase hex commit sha",
    )

    clean_recall = reader.number(table, "clean_recall", at)
    if "clean_recall" in table and not 0.0 <= clean_recall <= 1.0:
        reader.note(f"{at}.clean_recall is a rate and must lie in [0, 1], got {clean_recall!r}")

    sample_size = reader.integer(table, "sample_size", at)
    if "sample_size" in table and sample_size <= 0:
        reader.note(
            f"{at}.sample_size must be positive, got {sample_size!r}; a rate over no items is "
            f"not a rate"
        )

    _note_stale_check(
        reader,
        at,
        recorded=decided_revision,
        pinned=revision,
        checked_on=decided_on,
        what="the baseline's clean-recall measurement",
        field="decided_revision",
    )

    # A recall is a function of two pinned artifacts and one parameter: the model, the rows, and
    # the threshold. The record gated only the model, so a dataset re-pin left `clean_recall`
    # reading as current while describing rows this file no longer pins.
    dataset_revision = reader.matching(
        reader.string(table, "dataset_revision", at),
        _SHA,
        f"{at}.dataset_revision",
        "a 40-character lowercase hex commit sha",
    )
    measured_at_threshold = reader.number(table, "measured_at_threshold", at)
    if "measured_at_threshold" in table and measured_at_threshold != threshold:
        reader.note(
            f"{at}.measured_at_threshold is {measured_at_threshold!r} and the baseline pins "
            f"threshold {threshold!r}. A recall counts the items scoring at or above a "
            f"threshold, so the two are one measurement and a threshold re-pin invalidates it"
        )

    hits = reader.integer(table, "hits", at)
    if "hits" in table and hits < 0:
        reader.note(f"{at}.hits must not be negative, got {hits!r}")
    if hits and sample_size and hits > sample_size:
        reader.note(
            f"{at}.hits is {hits} over a sample of {sample_size}: a recall cannot count more "
            f"items than it scored"
        )
    elif sample_size and "hits" in table and "clean_recall" in table:
        # Guarded on PRESENCE, not truthiness. `hits and sample_size and clean_recall` skipped the
        # check whenever any of the three was zero -- so `hits = 0` beside `clean_recall = 0.836`
        # loaded clean, and `floor` then computed above `ceiling`. A cross-check disabled by one
        # of its own operands is the shape this file was fixed to remove, reintroduced by the fix.
        # The rate is derived from two integers this block also records, so it is checkable.
        # Recording a rate nothing recomputes is how a transposed digit survives publication.
        places = len(str(clean_recall).partition(".")[2]) or 4
        if round(hits / sample_size, places) != round(clean_recall, places):
            reader.note(
                f"{at}.clean_recall is {clean_recall!r} and {hits}/{sample_size} is "
                f"{hits / sample_size:.{places}f}: the rate and its two integers disagree"
            )

    overlap_rows = reader.integer(table, "overlap_rows", at)
    if "overlap_rows" in table and not 0 <= overlap_rows <= max(sample_size, 0):
        reader.note(
            f"{at}.overlap_rows is {overlap_rows!r} and must lie in [0, {sample_size}]: it "
            f"counts sampled rows that reach this baseline's declared training data"
        )

    return Oq2Check(
        outcome=outcome,
        decided_on=decided_on,
        decided_revision=decided_revision,
        dataset_revision=dataset_revision,
        measured_at_threshold=measured_at_threshold,
        hits=hits,
        clean_recall=clean_recall,
        sample_size=sample_size,
        overlap_rows=overlap_rows,
        # OQ2's floor is a judgement, not a constant: "strong enough for its degradation to mean
        # anything" has no number the run can supply. So the file records WHO judged it, rather
        # than pretending a threshold exists or letting `outcome = "kept"` admit any rate at all.
        judged_sufficient_by=reader.string(table, "judged_sufficient_by", at),
        source=reader.string(table, "source", at),
    )


def _read_selection_method(
    reader: _Reader, table: Mapping[str, Any], where: str
) -> tuple[str, int | None, str | None]:
    """The `(method, seed, sort_key)` triple, with the coupling between method and parameter.

    **One implementation, two callers.** The attack draw declared this vocabulary first and FR5.1
    requires the benign frame to use "the same selection-method vocabulary". Two readers that
    happened to accept the same words would satisfy that sentence today and drift the first time
    one of them grew a method; one reader makes it the same vocabulary by construction, and the
    tests that supply a failing input to one supply it to both.

    Four refusals, each with an input that produces it:

    - a `method` outside `DRAW_METHODS` -- a method nobody implemented reads exactly like one
      somebody did, and the difference is which rows end up in the corpus;
    - `head` with no `sort_key`, or a `sort_key` outside `DRAW_SORT_KEYS` -- a head take with no
      declared order is "whatever the reader yielded first", which is not a rule;
    - `seeded_random` with no `seed`, or a non-integer one -- the same, one level down;
    - either method carrying the *other* one's parameter. That one is refused rather than
      ignored: an ignored seed sitting in the file reads as a declared draw and is not one, and a
      later edit to `method` would silently start consuming it.
    """
    method = reader.string(table, "method", where)
    if method and method not in DRAW_METHODS:
        reader.note(
            f"{where}.method must be one of {', '.join(DRAW_METHODS)}, got {method!r}"
        )

    has_seed = "seed" in table
    has_sort_key = "sort_key" in table
    seed: int | None = None
    sort_key: str | None = None

    if method == DRAW_HEAD:
        if has_seed:
            reader.note(
                f"{where} declares method {DRAW_HEAD!r} and a seed; a head take consumes no "
                f"randomness, so the seed would sit in the file looking like a declared draw"
            )
        if not has_sort_key:
            reader.note(
                f"{where} declares method {DRAW_HEAD!r} and no sort_key; the first N of an "
                f"undeclared order is whatever the reader yielded first, which is not a rule"
            )
        else:
            sort_key = reader.string(table, "sort_key", where)
            if sort_key and sort_key not in DRAW_SORT_KEYS:
                reader.note(
                    f"{where}.sort_key must be one of {', '.join(DRAW_SORT_KEYS)}, "
                    f"got {sort_key!r}"
                )
    elif method == DRAW_SEEDED_RANDOM:
        if has_sort_key:
            reader.note(
                f"{where} declares method {DRAW_SEEDED_RANDOM!r} and a sort_key; the shuffle "
                f"decides the order, so the sort_key would sit in the file consumed by nothing"
            )
        if not has_seed:
            reader.note(
                f"{where} declares method {DRAW_SEEDED_RANDOM!r} and no seed; a shuffle with no "
                f"declared seed produces a different corpus on every run"
            )
        else:
            seed = reader.integer(table, "seed", where)

    return method, seed, sort_key


def _read_attack_draw(
    reader: _Reader, parent: Mapping[str, Any], where: str
) -> AttackDraw:
    """The `[attack_dataset.draw]` sub-table.

    The method and its companion parameter are `_read_selection_method`'s, shared with the benign
    frame so the two declarations cannot drift into two vocabularies.

    `sample_size_positives` must be positive. Zero would draw an empty corpus and publish a rate
    over nothing.
    """
    table = reader.table(parent, "draw", where)
    where = f"{where}.draw"

    declared_on = reader.calendar_date(
        reader.string(table, "declared_on", where), f"{where}.declared_on"
    )

    size = reader.integer(table, "sample_size_positives", where)
    if size <= 0:
        reader.note(
            f"{where}.sample_size_positives must be a positive count of attack positives, "
            f"got {size!r}; a draw of none publishes a rate over nothing"
        )

    method, seed, sort_key = _read_selection_method(reader, table, where)

    return AttackDraw(
        declared_on=declared_on,
        sample_size_positives=size,
        method=method,
        seed=seed,
        sort_key=sort_key,
    )


def _read_attack_dataset(
    reader: _Reader, table: Mapping[str, Any], index: int
) -> AttackDataset:
    where = f"attack_dataset[{index}]"
    key = reader.string(table, "key", where)
    if key:
        where = f"attack_dataset[{index}] ({key})"

    repository = reader.matching(
        reader.string(table, "repository", where),
        _REPOSITORY,
        f"{where}.repository",
        "a `namespace/name` repository id",
    )
    revision = reader.matching(
        reader.string(table, "revision", where),
        _SHA,
        f"{where}.revision",
        "a 40-character lowercase hex commit sha",
    )

    provenance_table = reader.table(table, "provenance", where)
    seeds = provenance_table.get("seeds")
    if seeds is None:
        reader.note(f"{where}.provenance.seeds is missing")
        seeds = []
    elif not isinstance(seeds, list) or not all(isinstance(seed, str) for seed in seeds):
        reader.note(f"{where}.provenance.seeds must be a list of strings")
        seeds = []
    for seed in seeds:
        reader.matching(
            seed, _REPOSITORY, f"{where}.provenance.seeds", "a `namespace/name` repository id"
        )
    provenance = Provenance(
        checked_on=reader.calendar_date(
            reader.string(provenance_table, "checked_on", f"{where}.provenance"),
            f"{where}.provenance.checked_on",
        ),
        card_revision=reader.matching(
            reader.string(provenance_table, "card_revision", f"{where}.provenance"),
            _SHA,
            f"{where}.provenance.card_revision",
            "a 40-character lowercase hex commit sha",
        ),
        seeds=tuple(seeds),
    )
    for seed in sorted({seed for seed in provenance.seeds if provenance.seeds.count(seed) > 1}):
        reader.note(f"{where}.provenance.seeds names {seed} twice")
    if repository and repository in provenance.seeds:
        reader.note(
            f"{where}.provenance.seeds names {repository}, which is the dataset itself; a seed "
            f"is a source the dataset was built from, not the dataset"
        )
    _note_stale_check(
        reader,
        f"{where}.provenance",
        recorded=provenance.card_revision,
        pinned=revision,
        checked_on=provenance.checked_on,
        what="the dataset's own card",
    )

    return AttackDataset(
        key=key,
        repository=repository,
        revision=revision,
        splits=reader.distinct_strings(table, "splits", where),
        attack_label=reader.label_value(table, "attack_label", where),
        draw=_read_attack_draw(reader, table, where),
        licence=_read_licence(reader, table, where),
        provenance=provenance,
    )


def _read_benign_code_repository(
    reader: _Reader, table: Mapping[str, Any], index: int
) -> BenignCodeRepository:
    where = f"benign_frame.code_repository[{index}]"
    key = reader.string(table, "key", where)
    if key:
        where = f"{where} ({key})"
    repository = reader.matching(
        reader.string(table, "repository", where),
        _REPOSITORY,
        f"{where}.repository",
        "a `namespace/name` repository id",
    )
    revision = reader.matching(
        reader.string(table, "revision", where),
        _SHA,
        f"{where}.revision",
        "a 40-character lowercase commit sha",
    )
    return BenignCodeRepository(
        key=key,
        repository=repository,
        revision=revision,
        licence=_read_licence(reader, table, where),
    )


def _read_benign_frame(
    reader: _Reader, document: Mapping[str, Any], baselines: Sequence[Baseline]
) -> BenignFrame:
    """The `[benign_frame]` block: FR5.1's sampling frame, and the digest that fixes it.

    The digest is computed over the **raw** block and compared against the declared `frame_id`
    before anything else about the frame is judged, because every other problem in here is a
    problem with a frame that at least still is the frame it says it is.
    """
    frame = reader.table(document, "benign_frame", "")
    where = "benign_frame"

    declared_id = reader.string(frame, FRAME_ID_FIELD, where)
    computed = frame_digest(frame)
    if declared_id and declared_id != computed:
        reader.note(
            f"{where}.{FRAME_ID_FIELD} is {declared_id!r} and the block hashes to {computed!r}. "
            f"The frame is the answer to 'was the benign corpus grown until the number looked "
            f"reasonable', and an edited frame carrying its previous digest is that answer "
            f"withdrawn. Re-declare the frame deliberately, with a new id, so the change is a "
            f"diff against a corpus whose manifest still records the old one"
        )

    declared_on = reader.calendar_date(
        reader.string(frame, "declared_on", where), f"{where}.declared_on"
    )

    size = reader.integer(frame, "sample_size_items", where)
    if size <= 0:
        reader.note(
            f"{where}.sample_size_items must be a positive count of items per benign class, "
            f"got {size!r}; a benign class of none publishes a false-positive rate over nothing"
        )

    method, seed, sort_key = _read_selection_method(reader, frame, where)

    # --- B-code -------------------------------------------------------------------------------
    code_where = f"{where}.{BENIGN_CODE_CLASS}"
    code = reader.table(frame, BENIGN_CODE_CLASS, where)

    min_repositories = reader.integer(code, "min_repositories", code_where)
    if "min_repositories" in code and min_repositories < MINIMUM_BENIGN_CODE_REPOSITORIES:
        reader.note(
            f"{code_where}.min_repositories is {min_repositories} and FR5.1's floor is "
            f"{MINIMUM_BENIGN_CODE_REPOSITORIES}; files from one repository share a language, a "
            f"style and a base64 idiom, so a frame drawing from fewer carries a design effect "
            f"that widens every B-code interval while the reported n stays the same"
        )

    max_files = reader.integer(code, "max_files_per_repository", code_where)
    if (
        "max_files_per_repository" in code
        and max_files > MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY
    ):
        reader.note(
            f"{code_where}.max_files_per_repository is {max_files} and FR5.1's cap is "
            f"{MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY}; the cap is the other side of the floor "
            f"and lifting it reaches the same design effect from the other direction"
        )
    if "max_files_per_repository" in code and max_files < 1:
        reader.note(
            f"{code_where}.max_files_per_repository is {max_files}; a cap below one draws no "
            f"file from any repository"
        )

    eligibility = reader.string(code, "eligibility", code_where)
    if eligibility and eligibility not in BENIGN_CODE_ELIGIBILITIES:
        reader.note(
            f"{code_where}.eligibility must be one of "
            f"{', '.join(BENIGN_CODE_ELIGIBILITIES)}, got {eligibility!r}; the rule decides which "
            f"files become corpus rows, so a rule nobody implemented reads exactly like one "
            f"somebody did"
        )

    min_bytes = reader.integer(code, "min_file_bytes", code_where)
    max_bytes = reader.integer(code, "max_file_bytes", code_where)
    if "min_file_bytes" in code and min_bytes < 1:
        reader.note(f"{code_where}.min_file_bytes is {min_bytes}; an empty file is not a corpus row")
    if "min_file_bytes" in code and "max_file_bytes" in code and max_bytes <= min_bytes:
        reader.note(
            f"{code_where} declares min_file_bytes {min_bytes} and max_file_bytes {max_bytes}; "
            f"the band is empty and no file could be drawn"
        )

    extensions = reader.strings(code, "file_extensions", code_where)
    if "file_extensions" in code and not extensions:
        reader.note(
            f"{code_where}.file_extensions is empty; a draw over every path in a repository "
            f"reads lockfiles, minified bundles and vendored blobs as source code"
        )
    for extension in extensions:
        if not extension.startswith(".") or extension != extension.lower():
            reader.note(
                f"{code_where}.file_extensions holds {extension!r}; each entry is a lowercase "
                f"suffix beginning with a dot, and the match is on the path's own suffix"
            )
    if len(set(extensions)) != len(extensions):
        reader.note(f"{code_where}.file_extensions repeats an entry: {list(extensions)}")

    repositories = tuple(
        _read_benign_code_repository(reader, table, index)
        for index, table in enumerate(_entries(reader, code, "repository"))
    )
    if len(repositories) < min_repositories:
        reader.note(
            f"{code_where} declares min_repositories {min_repositories} and pins "
            f"{len(repositories)}; a frame that cannot reach its own floor could only reach it "
            f"by topping up from somewhere the frame does not name"
        )
    _note_duplicates(reader, f"{code_where}.repository", [entry.key for entry in repositories], "key")
    _note_duplicates(
        reader,
        f"{code_where}.repository",
        [entry.repository for entry in repositories],
        "repository",
    )

    # --- B-chat -------------------------------------------------------------------------------
    chat_where = f"{where}.{BENIGN_CHAT_CLASS}"
    chat = reader.table(frame, BENIGN_CHAT_CLASS, where)
    hand_authored = reader.integer(chat, "hand_authored_items", chat_where)
    if "hand_authored_items" in chat and hand_authored < 0:
        reader.note(
            f"{chat_where}.hand_authored_items is {hand_authored!r}; it counts items and a "
            f"negative allowance is not a smaller one"
        )
    if "hand_authored_items" in chat and size > 0 and hand_authored >= size:
        reader.note(
            f"{chat_where}.hand_authored_items is {hand_authored} against a class of {size}; the "
            f"allowance is for what no public dataset carries, and a class made entirely of it "
            f"would be a benign corpus this repository wrote for itself"
        )

    # --- the confirmatory cell ------------------------------------------------------------------
    cell_where = f"{where}.confirmatory_cell"
    cell_table = reader.table(frame, "confirmatory_cell", where)
    cell = ConfirmatoryCell(
        declared_on=reader.calendar_date(
            reader.string(cell_table, "declared_on", cell_where), f"{cell_where}.declared_on"
        ),
        baseline=reader.string(cell_table, "baseline", cell_where),
        dressing_chain=reader.string(cell_table, "dressing_chain", cell_where),
        benign_class=reader.string(cell_table, "benign_class", cell_where),
    )
    keys = {baseline.key for baseline in baselines}
    # Only when there is a baseline set to name. With none declared, `_check_baseline_set` already
    # aborts on the count, and adding "names no declared baseline" would report the same absence
    # twice under a different diagnosis.
    if cell.baseline and keys and cell.baseline not in keys:
        reader.note(
            f"{cell_where}.baseline is {cell.baseline!r}, which is not a declared baseline key "
            f"({sorted(keys)}); the verdict rests on this cell and a cell naming no column of the "
            f"table cannot be evaluated"
        )

    return BenignFrame(
        declared_on=declared_on,
        sample_size_items=size,
        method=method,
        seed=seed,
        sort_key=sort_key,
        frame_id=declared_id,
        b_code=BenignCodeFrame(
            min_repositories=min_repositories,
            max_files_per_repository=max_files,
            eligibility=eligibility,
            min_file_bytes=min_bytes,
            max_file_bytes=max_bytes,
            file_extensions=extensions,
            repositories=repositories,
        ),
        b_chat=BenignChatFrame(hand_authored_items=hand_authored),
        confirmatory_cell=cell,
    )


def _read_exclusion_source(
    reader: _Reader, table: Mapping[str, Any], index: int
) -> ExclusionSource:
    where = f"exclusion_source[{index}]"
    key = reader.string(table, "key", where)
    if key:
        where = f"exclusion_source[{index}] ({key})"

    repository = reader.matching(
        reader.string(table, "repository", where),
        _REPOSITORY,
        f"{where}.repository",
        "a `namespace/name` repository id",
    )

    availability = reader.string(table, "availability", where)
    if availability and availability not in EXCLUSION_AVAILABILITIES:
        reader.note(
            f"{where}.availability is {availability!r}, and the admitted values are "
            f"{', '.join(sorted(EXCLUSION_AVAILABILITIES))}. The corpus build reads this value "
            f"and compares it against what the hub actually answers, so a spelling nothing "
            f"matches would leave the comparison to a reader"
        )

    http_status = reader.integer(table, "http_status", where)
    if "http_status" in table and not 100 <= http_status <= 599:
        reader.note(
            f"{where}.http_status must be an HTTP status code, got {http_status!r}"
        )

    # The revision, the status and the availability decide each other, in both directions. A
    # source the hub does not answer for hands back no commit, so a sha beside `unreachable`
    # came from somewhere this file cannot name; and any source the hub DOES answer for is
    # pinned by revision like every other artifact here, whether or not its rows can be read,
    # because a moving exclusion set silently changes which rows survive into the corpus.
    declared_revision = table.get("revision")
    revision = ""
    if availability in {EXCLUSION_AVAILABLE, EXCLUSION_UNREADABLE}:
        if declared_revision is None:
            reader.note(
                f"{where}.revision is missing and this source declares itself "
                f"{availability!r}; the hub answers for it, so it resolves to a commit and is "
                f"pinned by it like every other artifact in this file"
            )
        else:
            revision = reader.matching(
                reader.string(table, "revision", where),
                _SHA,
                f"{where}.revision",
                "a 40-character lowercase hex commit sha",
            )
        if "http_status" in table and http_status != HTTP_OK:
            reader.note(
                f"{where} declares {availability!r} and http_status {http_status}; both "
                f"{EXCLUSION_AVAILABLE!r} and {EXCLUSION_UNREADABLE!r} describe a source the "
                f"hub answers {HTTP_OK} for -- they differ in whether its rows load, not in "
                f"whether it resolves"
            )
    elif availability == EXCLUSION_UNREACHABLE:
        if declared_revision is not None:
            reader.note(
                f"{where}.revision is declared and this source declares itself "
                f"{EXCLUSION_UNREACHABLE!r}; a source the hub does not answer for hands back no "
                f"commit, so a revision here was copied from somewhere this file cannot name"
            )
        if "http_status" in table and http_status == HTTP_OK:
            reader.note(
                f"{where} declares {EXCLUSION_UNREACHABLE!r} and http_status {HTTP_OK}, which "
                f"is the status of a source the hub answers for"
            )

    return ExclusionSource(
        key=key,
        repository=repository,
        revision=revision,
        availability=availability,
        http_status=http_status,
        checked_on=reader.calendar_date(
            reader.string(table, "checked_on", where), f"{where}.checked_on"
        ),
        evidence=reader.string(table, "evidence", where),
        licence=_read_licence(reader, table, where),
    )


def _note_stale_check(
    reader: _Reader,
    where: str,
    *,
    recorded: str,
    pinned: str,
    checked_on: str,
    what: str,
    field: str = "card_revision",
) -> None:
    """A check performed against a revision this file no longer pins is a check nobody re-ran.

    The date is metadata and the revision is the gate: a pin can move on the same day, and a
    date alone would let the declaration keep looking fresh while describing a different card.

    `field` names the key that recorded the revision, because two different declarations are
    checked this way -- what a card said, and what a human confirmed a window to be -- and an
    error message that names the wrong key sends its reader to the wrong line.
    """
    if not recorded or not pinned or recorded == pinned:
        return
    reader.note(
        f"{where}.{field} is {recorded} and the pinned revision is {pinned}: the check "
        f"recorded here was performed on {checked_on or 'an unrecorded date'} against a "
        f"revision of {what} that this file no longer pins. Re-read {what} at the pinned "
        f"revision and record what it says, rather than carrying an answer to a question about "
        f"a different artifact"
    )


def _entries(reader: _Reader, document: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = document.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        reader.note(f"[[{key}]] must be an array of tables")
        return []
    return list(value)


def _check_baseline_set(baselines: Sequence[Baseline]) -> None:
    """SC5's independence claim, checked rather than asserted in prose."""
    problems: list[str] = []

    if len(baselines) < MINIMUM_BASELINES:
        problems.append(
            f"{len(baselines)} baseline(s) are pinned and the floor is {MINIMUM_BASELINES}: "
            f"one baseline makes every result read as a property of that model, so an "
            f"ineligible baseline is replaced, never removed"
        )

    by_pair: dict[tuple[str, str], list[str]] = {}
    for baseline in baselines:
        by_pair.setdefault(baseline.family_pair, []).append(baseline.repository)
    for pair, repositories in by_pair.items():
        if len(repositories) > 1:
            problems.append(
                f"{' and '.join(repositories)} both declare architecture/tokenizer families "
                f"{pair[0]}/{pair[1]}; the mechanism under study is how encoded text "
                f"tokenizes, so two baselines that tokenize alike cannot corroborate each other"
            )

    if problems:
        raise BaselineSetInvalid(*problems)


def _check_oq2_records(
    baselines: Sequence[Baseline], datasets: Sequence[AttackDataset]
) -> None:
    """What OQ2's numbers are a function of, checked across the whole file.

    A single `[baseline.oq2]` block cannot see the dataset it names or the other baseline it will
    be read beside, so these three checks have no home inside `_read_oq2`. They are the ones that
    matter: OQ2's entire reading is a *comparison* between the two baselines, and a comparison
    across two different pools or two different pins is not one.
    """
    problems: list[str] = []
    pinned = {dataset.revision: dataset.repository for dataset in datasets}

    for baseline in baselines:
        oq2 = baseline.oq2
        if oq2.dataset_revision and oq2.dataset_revision not in pinned:
            problems.append(
                f"{baseline.key} recorded its clean recall against dataset revision "
                f"{oq2.dataset_revision[:8]}, which this file no longer pins. The recall "
                f"describes rows that are not the rows the run would score: re-measure against "
                f"the pinned revision and update the block with the number and the sha together"
            )

    sizes = {baseline.oq2.sample_size for baseline in baselines if baseline.oq2.sample_size}
    if len(sizes) > 1:
        listed = ", ".join(
            f"{baseline.key}={baseline.oq2.sample_size}" for baseline in baselines
        )
        problems.append(
            f"the OQ2 records describe different sample sizes ({listed}). OQ2 is read as a "
            f"comparison between baselines -- which of them has headroom, and whether the "
            f"obscure one holds up -- and two recalls over two different pools do not compare"
        )

    for baseline in baselines:
        if baseline.oq2.outcome == OQ2_KEPT and not baseline.oq2.judged_sufficient_by:
            problems.append(
                f"{baseline.key} is kept with no `judged_sufficient_by`. OQ2's floor is a "
                f"judgement -- \"strong enough for its degradation to mean anything\" names no "
                f"number -- so the file records who made it. Without that, `outcome = \"kept\"` "
                f"admits any recall in [0, 1] and the gate cannot fail"
            )

    if problems:
        raise BaselineSetInvalid(*problems)


def _check_lineage(
    baselines: Sequence[Baseline], attack_datasets: Sequence[AttackDataset]
) -> None:
    """No baseline is scored over its own training text, declared and at one hop.

    The two teeth are one loop because they are one rule reaching two distances: a baseline
    trained on the pinned pool, and a baseline trained on what the pinned pool was built from.
    The second is the one the cards cannot show, and it is the one that got through.
    """
    problems: list[str] = []

    for baseline in baselines:
        lineage = baseline.lineage
        read_at = (
            f"card revision {lineage.card_revision}, read {lineage.checked_on}"
            if lineage.card_revision and lineage.checked_on
            else "an unrecorded reading"
        )
        for dataset in attack_datasets:
            declared = lineage.relationship_to(dataset.repository)

            if declared == TRAINED_ON:
                problems.append(
                    f"{baseline.repository} declares training on the pinned attack dataset "
                    f"{dataset.repository} ({read_at}); scored over it, the baseline reports "
                    f"memory rather than detection, and its false-positive rate is measured "
                    f"over benign text it was taught to call safe"
                )
                continue

            reached = tuple(
                seed for seed in dataset.provenance.seeds if lineage.trains_on(seed)
            )
            if reached and declared != SEEDED_FROM_TRAINING_SOURCE:
                problems.append(
                    f"{baseline.repository} declares training on "
                    f"{', '.join(reached)}, which {dataset.repository}'s own card names among "
                    f"the seeds it was built from (provenance read "
                    f"{dataset.provenance.checked_on} at card revision "
                    f"{dataset.provenance.card_revision}); the baseline reaches the pinned "
                    f"attack dataset at one hop that no model card mentions, while declaring "
                    f"{declared!r} against it. Either the reach is removed from the corpus "
                    f"before anything is measured -- declared here as "
                    f"{SEEDED_FROM_TRAINING_SOURCE!r}, which makes every seed above a required "
                    f"exclusion source -- or this baseline is ineligible"
                )

    if problems:
        raise BaselineIneligible(*problems)


def load_pins(root: Path | str | None = None) -> Pins:
    """Read, validate and return the pins. Step 1 of the entrypoint's sequence.

    Structure, the baseline set and the lineage gate are all checked here, so no consumer can
    load the pins and forget to ask whether the set they describe is one SC5 admits, or whether
    a baseline in it would be scored over its own training text.
    """
    root_path = Path(root) if root is not None else _repository_root()
    path = root_path / PINS_FILENAME

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise PinsFileInvalid(f"no {PINS_FILENAME} at {path}") from None
    except OSError as error:
        raise PinsFileInvalid(f"{path} could not be read: {error}") from None

    try:
        document = tomllib.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise PinsFileInvalid(f"{path} is not valid UTF-8: {error}") from None
    except tomllib.TOMLDecodeError as error:
        raise PinsFileInvalid(f"{path} is not valid TOML: {error}") from None

    problems: list[str] = []
    reader = _Reader(problems)

    meta = reader.table(document, "meta", "")
    schema_version = reader.integer(meta, "schema_version", "meta")
    if "schema_version" in meta and schema_version != SCHEMA_VERSION:
        reader.note(
            f"meta.schema_version is {schema_version}, and this module reads "
            f"{SCHEMA_VERSION}; a pin file from another shape is not one it may guess at"
        )
    verified_on = reader.calendar_date(
        reader.string(meta, "verified_on", "meta"),
        "meta.verified_on",
    )
    verified_against = reader.string(meta, "verified_against", "meta")

    baselines = tuple(
        _read_baseline(reader, table, index)
        for index, table in enumerate(_entries(reader, document, "baseline"))
    )
    attack_datasets = tuple(
        _read_attack_dataset(reader, table, index)
        for index, table in enumerate(_entries(reader, document, "attack_dataset"))
    )
    exclusion_sources = tuple(
        _read_exclusion_source(reader, table, index)
        for index, table in enumerate(_entries(reader, document, "exclusion_source"))
    )
    benign_frame = _read_benign_frame(reader, document, baselines)

    _note_duplicates(reader, "baseline", [b.key for b in baselines], "key")
    _note_duplicates(reader, "attack_dataset", [d.key for d in attack_datasets], "key")
    _note_duplicates(reader, "exclusion_source", [s.key for s in exclusion_sources], "key")
    # Across sections, not within them. Hugging Face keeps models and datasets in separate
    # namespaces, so one id can legally name both -- and `verify_revisions` reports its
    # resolutions by repository id, where two artifacts sharing one would collapse into one
    # line and one of the two pins would go unreported while looking verified.
    _note_duplicates(
        reader,
        "baseline/attack_dataset/exclusion_source",
        [b.repository for b in baselines]
        + [d.repository for d in attack_datasets]
        + [s.repository for s in exclusion_sources],
        "repository",
    )

    # The benign code repositories are on a different host and in a different namespace, so they
    # are not in the duplicate check above -- one id can legitimately name a GitHub repository and
    # a Hugging Face one. This is the check that matters instead: B-code is drawn from material
    # this project is measuring a **false-positive** rate over, and a repository that is also a
    # pinned baseline's training source, a pinned attack pool or an exclusion source is material
    # the corpus is supposed to be removing rather than drawing. It is also the enforceable half of
    # the README's "not security or guardrail repositories" criterion; the rest of that criterion is
    # a human reading, declared as one.
    hosted_elsewhere = (
        {_canonical(entry.repository) for entry in baselines}
        | {_canonical(entry.repository) for entry in attack_datasets}
        | {_canonical(entry.repository) for entry in exclusion_sources}
    )
    for entry in benign_frame.b_code.repositories:
        if _canonical(entry.repository) in hosted_elsewhere:
            reader.note(
                f"benign_frame.{BENIGN_CODE_CLASS}.repository {entry.repository!r} is also pinned "
                f"as a baseline, an attack dataset or an exclusion source; B-code is the material "
                f"the false-positive rate is measured over, and drawing it from a source the "
                f"corpus is meant to be removing against measures the filter rather than the layer"
            )

    # A relationship to a dataset that is not pinned is a relationship to nothing, and a pinned
    # dataset a baseline says nothing about is the gap the lineage check exists to close.
    pinned_datasets = {dataset.repository for dataset in attack_datasets}
    if not attack_datasets:
        # With no dataset pinned, "every baseline declares its relationship to every pinned
        # attack dataset" is vacuously true and the lineage gate has nothing to check against.
        reader.note(
            "no [[attack_dataset]] is pinned; the attack payloads come from a pinned public "
            "dataset, and a lineage declaration against an empty set checks nothing"
        )
    for baseline in baselines:
        declared = set(baseline.lineage.attack_datasets)
        for missing in sorted(pinned_datasets - declared):
            reader.note(
                f"baseline {baseline.repository or '?'} declares no relationship to the pinned "
                f"attack dataset {missing}"
            )
        for extra in sorted(declared - pinned_datasets):
            reader.note(
                f"baseline {baseline.repository or '?'} declares a relationship to {extra}, "
                f"which is not a pinned attack dataset"
            )

    # One hop is only mechanical if every baseline answers for every seed. A baseline silent
    # about a source a pinned dataset says it was built from is the exact gap this tooth exists
    # to close, and silence there would read as "no reach" while meaning "nobody looked".
    seeds = {seed for dataset in attack_datasets for seed in dataset.provenance.seeds}
    for baseline in baselines:
        for missing in sorted(seeds - set(baseline.lineage.training_sources)):
            reader.note(
                f"baseline {baseline.repository or '?'} declares nothing about {missing}, named "
                f"on a pinned attack dataset's own card as a seed it was built from; one hop of "
                f"provenance is a check only if every baseline answers for every seed"
            )
        # The declaration and the seeds have to agree in both directions. A hop the seeds do not
        # carry is a claim about nothing, and a claim about nothing is how an exclusion source
        # gets pinned for a reason that stopped being true.
        for dataset in attack_datasets:
            if baseline.lineage.relationship_to(dataset.repository) != (
                SEEDED_FROM_TRAINING_SOURCE
            ):
                continue
            if not any(baseline.lineage.trains_on(seed) for seed in dataset.provenance.seeds):
                reader.note(
                    f"baseline {baseline.repository or '?'} declares "
                    f"{SEEDED_FROM_TRAINING_SOURCE!r} against {dataset.repository}, but declares "
                    f"training on none of the seeds that dataset's card names; there is no hop "
                    f"to remove and no exclusion source the declaration would buy"
                )

    # The exclusion array against the declarations that create the obligation, in both
    # directions. A missing entry is a training source the filter never downloads and therefore
    # never removes; an extra entry is a source pinned for a reason that has stopped being true,
    # which is how an exclusion set outlives the lineage it came from. The comparison is on the
    # canonical form because the hub resolves ids case-insensitively and the two blocks are
    # written by hand from two different cards.
    declared_exclusions: dict[str, str] = {}
    for source in exclusion_sources:
        if source.repository:
            declared_exclusions.setdefault(_canonical(source.repository), source.repository)
    derived_exclusions = _derived_exclusion_sources(baselines, attack_datasets)
    for missing in sorted(
        derived_exclusions[name] for name in derived_exclusions.keys() - declared_exclusions.keys()
    ):
        reader.note(
            f"no [[exclusion_source]] pins {missing}, which this file declares either as a "
            f"training source of a pinned baseline or as a seed on a pinned dataset's own card; "
            f"a corpus row that appears in it is text a baseline was taught to call safe, and a "
            f"source the build never downloads is one it silently treats as contributing zero"
        )
    for extra in sorted(
        declared_exclusions[name] for name in declared_exclusions.keys() - derived_exclusions.keys()
    ):
        reader.note(
            f"[[exclusion_source]] pins {extra}, which no pinned baseline declares training on "
            f"and no pinned dataset's card names as a seed; an exclusion source nothing derives "
            f"removes rows from the corpus for a reason this file no longer states"
        )

    if problems:
        raise PinsFileInvalid(*problems)

    _check_baseline_set(baselines)
    _check_oq2_records(baselines, attack_datasets)
    _check_lineage(baselines, attack_datasets)

    return Pins(
        schema_version=schema_version,
        verified_on=verified_on,
        verified_against=verified_against,
        baselines=baselines,
        attack_datasets=attack_datasets,
        exclusion_sources=exclusion_sources,
        benign_frame=benign_frame,
        path=path,
    )


def _note_duplicates(
    reader: _Reader, section: str, values: Sequence[str], field: str
) -> None:
    seen: set[str] = set()
    for value in values:
        if value and value in seen:
            reader.note(f"two [[{section}]] entries share {field} {value!r}")
        seen.add(value)


def _repository_root() -> Path:
    """`src/nbc/pins.py` -> the repository root, two parents above the package."""
    return Path(__file__).resolve().parent.parent.parent


# --- asking the world -------------------------------------------------------------------------

CHECKED_AGAINST_HUB: Final[str] = "hub"
"""The resolution asked the publisher what the pin points at, and got an answer."""

CHECKED_AGAINST_CACHE: Final[str] = "cache"
"""The resolution found a snapshot directory named after the pin on this machine.

**This is not a check against the world and the file must not read as though it were.** The hub
names a snapshot directory by the commit it was fetched at, so the directory's existence is the
pin's own sha spelled back -- comparing it to the pin compares a value to itself. It is still
worth recording: it is what keeps a reproduction offline after the first fetch, and it is the
honest name for what happened.
"""


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a pinned revision resolved to, and what it was resolved against.

    The second field is the whole point. `verify_revisions` used to compare a returned sha to the
    pin, and on every machine that had fetched once the returned sha *was* the pin, read off a
    directory name. The comparison could not fail, while the module's docstring promised it was
    the guarantee that the numbers came from the pinned artifacts.
    """

    sha: str
    checked_against: str


Resolver = Callable[[RemoteArtifact], Resolution | None]
"""Given an artifact, the commit its revision resolves to, or `None` if it resolves to
nothing."""


def hf_cache_root() -> Path:
    """Where the Hugging Face hub cache lives on this machine.

    This reads `HF_HUB_CACHE` and `HF_HOME` and is not an exception to "nothing reads
    configuration from the environment at point of use": the location of somebody else's cache
    is a property of the machine, not a parameter of this run. Every parameter this run has is
    in `pins.toml` or is passed in.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_from_cache(
    artifact: RemoteArtifact, cache_root: Path | None = None
) -> Resolution | None:
    """A cache resolution if this machine holds a non-empty snapshot, else `None`.

    The directory's existence is not evidence about the world: the hub names it after the commit
    it was fetched at, so it is the pin spelled back. The `Resolution` says so, and
    `verify_revisions` records it per artifact rather than letting a cache hit read as a check.

    An **empty** directory is refused. An interrupted fetch leaves one behind, and treating that
    as a resolution means the run declares an artifact verified that it does not hold.
    """
    snapshot = artifact.snapshot_dir(cache_root)
    if not snapshot.is_dir():
        return None
    if not any(snapshot.iterdir()):
        return None
    return Resolution(artifact.revision, CHECKED_AGAINST_CACHE)


def resolve_over_http(artifact: RemoteArtifact) -> Resolution | None:
    """Ask the hub what the pinned revision resolves to. The first fetch, and the smoke job.

    Any failure to get an answer is reported as `None` rather than raised: "the pin could not be
    resolved" is one of the two outcomes `verify_revisions` aborts on, and turning a network
    error into an unclassified crash would lose the exit code that tells CI which one happened.
    """
    import http.client
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        artifact.api_url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        OSError,
        http.client.HTTPException,
    ):
        # HTTPException is not an OSError and not a URLError. A truncated response left it to
        # escape as exit 1, which is precisely the unclassified crash this handler prevents.
        return None

    resolved = payload.get("sha") if isinstance(payload, dict) else None
    return Resolution(resolved, CHECKED_AGAINST_HUB) if isinstance(resolved, str) else None


def resolve_from_cache_then_hub(artifact: RemoteArtifact) -> Resolution | None:
    """Cache first, hub only for an artifact this machine has never fetched."""
    cached = resolve_from_cache(artifact)
    if cached is not None:
        return cached
    return resolve_over_http(artifact)


def verify_revisions(
    pins: Pins, resolve: Resolver = resolve_from_cache_then_hub
) -> Mapping[str, str]:
    """Assert every pinned revision still resolves to itself, before any inference.

    Returns the resolved sha per artifact so the run can record what it actually verified.
    Aborts loudly, naming the artifact and both shas, because a table computed over an artifact
    that is not the pinned one looks exactly like a table computed over the pinned one.
    """
    problems: list[str] = []
    resolved_by_artifact: dict[str, str] = {}

    for artifact in pins.remote_artifacts():
        resolution = resolve(artifact)
        if resolution is None:
            problems.append(
                f"{artifact} could not be resolved: it is not in this machine's Hugging "
                f"Face cache, and the hub returned no commit for it. Either the revision is "
                f"gone or the hub could not be reached; the two are different diagnoses and "
                f"this abort cannot tell them apart"
            )
            continue
        if resolution.sha != artifact.revision:
            problems.append(
                f"{artifact.kind} {artifact.repository} is pinned at {artifact.revision} and "
                f"now resolves to {resolution.sha}"
            )
            continue
        # The value AND what it was checked against. A cache resolution is the pin read off a
        # directory name, so a run verified entirely from cache has compared nothing to the
        # world, and results.json says which artifacts those were rather than implying all of
        # them were asked about.
        resolved_by_artifact[artifact.repository] = (
            f"{resolution.sha}@{resolution.checked_against}"
        )

    problems.extend(_size_problems(pins))

    if problems:
        raise PinMismatch(*problems)

    return resolved_by_artifact


def _size_problems(pins: Pins) -> list[str]:
    """`graph_bytes` compared to the graph, wherever this machine holds it.

    The field was declared as the evidence for `precision` -- an fp16 export is a fraction of its
    fp32 original, so the size is what would catch a swapped graph that kept its filename -- and
    nothing read it. It was recorded, copied into the run fields, checked for being positive, and
    never once compared to a file. That is this epic's most common defect in its purest form.

    Silent when the artifact is not on this machine: an absent file is `verify_revisions`'
    business, and reporting it twice under two diagnoses is what makes an abort unreadable.
    """
    problems: list[str] = []
    for baseline in pins.baselines:
        graph = baseline.artifact.snapshot_dir() / baseline.graph_path
        try:
            actual = graph.stat().st_size
        except OSError:
            continue
        if actual != baseline.graph_bytes:
            problems.append(
                f"{baseline.key} pins {baseline.graph_path} at {baseline.graph_bytes} bytes and "
                f"the file on this machine is {actual}. The size is the evidence for the pinned "
                f"{baseline.precision} precision, and a graph that changed while keeping its "
                f"name moves every score in the last decimals"
            )
    return problems


# --- the command line -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.pins [--verify]` -- load and validate, and optionally ask the world."""
    import argparse
    import json

    from nbc.errors import EXIT_OK, exit_code_for

    parser = argparse.ArgumentParser(
        prog="python -m nbc.pins",
        description=f"Load and validate {PINS_FILENAME}.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "also resolve every pinned revision -- from this machine's Hugging Face cache "
            "where the artifact is present, and over the network only where it is not"
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help=f"directory holding {PINS_FILENAME} (default: the repository root)",
    )
    args = parser.parse_args(argv)

    try:
        pins = load_pins(args.root)
        fields = pins.as_run_fields()
        if args.verify:
            fields["pins_resolved"] = dict(verify_revisions(pins))
    except NbcError as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(fields, sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
