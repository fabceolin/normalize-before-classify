"""AD-34 / FR5.2: nothing is redistributed under a licence nobody read, and credits are generated.

**What this repository actually does with other people's material.** The two corpus halves under
`data/` are committed files in a public MIT repository, and AD-20 fills them with drawn
rows from a pinned Hugging Face dataset and whole source files from 63 pinned GitHub repositories.
That is redistribution, not reference. It is also the one defect class in this project that surfaces
as a takedown or an issue thread after publication rather than as a red test before it, which is why
it is a build gate and not a packaging note.

**The evidence was already recorded and nothing read it.** `pins.toml` has carried a `[licence]`
block beside every baseline, the attack dataset and every B-code repository since Epic 1, and
`pins.Licence.blocks_redistribution` has been a published property with no caller outside `pins.py`
and its own tests. A field whose comment says it is the evidence for something, never compared to
the thing it is evidence for, is the pattern the Epic 1 review found 24 times. This module is the
comparison.

**Declaration is universal; the abort is about redistribution.** Every pinned source declares an
identifier, a source for that reading, an attribution and a `redistributed` flag -- models and
exclusion sources included, neither of which ships a byte into `data/`. The abort fires on a source
that *is* redistributed and whose identifier is absent, unrecognized or refused. That split is not
invented here: `[baseline.licence]` in `pins.toml` has said since Epic 1 that "the weights are
fetched into the Hugging Face cache and never committed here... the build-time licence abort is
about rows that ARE redistributed", and applying the abort to every declaration instead would stop
the run over a model nobody redistributes while saying nothing new about the rows that are.

**What happens when the answer is "publish it anyway".** The abort has one way through and it is
not a flag: `pins.Licence.accepted`, a table naming a human, a date, the README heading that
argues the position, and the reasoning. It answers the **absence** of a licence and nothing else
-- a refused identifier stays refused, an empty one stays a malformed pin -- and it lives inside
one source's `[licence]` block, so the next undeclared source aborts exactly as that one did.
`identifier` stays `not-declared` through this module, into `ATTRIBUTION.md` and into
`results.json`: what the acceptance changes is that a decision exists, never that a licence does.
Writing a compatible SPDX identifier instead would have passed this gate in one line and published
a false statement of fact to everyone who reads the credits file.

**A closed vocabulary, never a pattern.** `COMPATIBLE` and `REFUSED` are two disjoint frozensets of
SPDX identifiers, compared case-normalized. An identifier in neither is *unrecognized* and aborts:
the failure mode of a substring rule ("contains `MIT`", "starts with `BSD`") is that
`CC-BY-NC-4.0` and `MIT-0` sail through a check nobody can enumerate, and the failure mode of a
default-allow is that the first licence nobody thought about is the one that ships.

**Why the row counts come from the rows.** `ATTRIBUTION.md` claims how many rows this repository
redistributes from each source. The draw report knows how many *positives* were selected, which
differs from the row count by the chain multiplier, and a credits file derived from the declaration
rather than from the artifact drifts from it exactly the way a hand-maintained one does -- one level
in, and less visibly. So the counts are a tally over the `source` field of the rows actually
written, and a row whose source matches no pinned identity aborts the build rather than going
uncredited.

**Attribution by parsed identity, not by substring.** A B-code row names itself
`github.com/{repository}@{revision}:{path}`; the host prefix is stripped, the path split off at the
first colon, the revision split off at the last `@`, and the resulting `(repository, revision)` pair
compared against the pinned pair. Dataset rows are an exact match against `repository` (attack rows)
or `repository@revision` (B-chat rows); hand-authored rows are an exact match against
`benign.HAND_AUTHORED_SOURCE`. A row naming a pinned repository at a sha the frame does not pin is
refused, and a substring rule would have accepted it.

**This module names no corpus file, locates none and writes none.** It is handed the rows and
returns text. Two AST scans hold both halves: `tests/corpus/test_manifest.py` refuses a module
outside the declared two that can locate a corpus file, and `tests/corpus/test_build.py` refuses a
module outside the declared two that can put bytes on disk. So the renderer lives here and the write
lives in `corpus/build.py`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from nbc.corpus.benign import HAND_AUTHORED_SOURCE
from nbc.errors import NbcError
from nbc.pins import Licence, Pins
from nbc.schema import CorpusItem

__all__ = [
    "ATTRIBUTION_FILENAME",
    "COMPATIBLE",
    "KIND_ATTACK_DATASET",
    "KIND_BASELINE",
    "KIND_BENIGN_CODE",
    "KIND_EXCLUSION_SOURCE",
    "KIND_HAND_AUTHORED",
    "LOCAL_LICENCE",
    "LOCAL_REPOSITORY",
    "REFUSED",
    "RedistributionRefused",
    "SourceRecord",
    "attribution_problems",
    "counts_by_key",
    "licence_problems",
    "pinned_sources",
    "render",
]


ATTRIBUTION_FILENAME: Final[str] = "ATTRIBUTION.md"
"""The generated credits file, beside the corpus it credits.

Deliberately **not** an entry in the corpus manifest: `manifest.read_corpus` refuses a recorded
file that is not one of the two corpus halves, and the manifest's hashes are what a table is
computed over. What guards this file instead is regeneration inside `verify-corpus`, which is the
stronger check of the two -- a hash catches an edit, regeneration also catches a file that was never
right in the first place.
"""


class RedistributionRefused(NbcError, exit_code=25):
    """A source without a licence this repository may redistribute under, or credits that drifted.

    Code 25. Distinct from `CorpusWriteRefused` (18) and `CorpusManifestMismatch` (22) because the
    diagnosis is different in kind: those say the corpus on disk is not the one this declaration
    describes, and this one says the material must not be published at all, or that the credits do
    not describe what was published.
    """

    def __init__(self, *problems: str) -> None:
        super().__init__(
            "redistribution is refused:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
        self.problems: tuple[str, ...] = problems


# --- the vocabulary -----------------------------------------------------------------------------
#
# SPDX identifiers, compared case-normalized because publishers spell them both ways and this file
# already carries `Apache-2.0` from a GitHub LICENSE file beside `apache-2.0` from a model card's
# YAML. The two sets are disjoint and a test asserts it, because an identifier in both would make
# the answer depend on which loop ran first.

COMPATIBLE: Final[frozenset[str]] = frozenset(
    {
        "mit",
        "mit-0",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "cc0-1.0",
        "unlicense",
        "0bsd",
        "psf-2.0",
        "python-2.0",
    }
)
"""Licences under which material may be redistributed inside an MIT repository.

Permissive, no share-alike, no field-of-use restriction: what each asks for is attribution and the
preservation of a notice, which is exactly what `ATTRIBUTION.md` provides. Attribution-only Creative
Commons licences are **not** here and that is deliberate: `CC-BY-4.0` and `CC-BY-3.0` are compatible
in the ordinary sense, but they are drafted for creative works rather than for code and their
attribution requirements are stricter than a credits line, so admitting one is a decision a human
takes by adding it here and not a default this list grants in advance. Nothing pinned as
redistributed carries one today, so the choice costs this build nothing and would cost it an abort
the day it did.
"""

REFUSED: Final[Mapping[str, str]] = {
    "gpl-2.0": "copyleft: redistribution obliges this repository to relicense",
    "gpl-3.0": "copyleft: redistribution obliges this repository to relicense",
    "gpl-3.0-only": "copyleft: redistribution obliges this repository to relicense",
    "gpl-3.0-or-later": "copyleft: redistribution obliges this repository to relicense",
    "agpl-3.0": "network copyleft: redistribution obliges this repository to relicense",
    "lgpl-3.0": "weak copyleft: incompatible with an unqualified MIT offer over the same file",
    "cc-by-sa-4.0": "share-alike: derivative corpora must carry the same licence",
    "cc-by-nc-4.0": "non-commercial: a field-of-use restriction MIT does not carry",
    "cc-by-nd-4.0": "no-derivatives: a dressed row is a derivative",
    "openrail": "use-based restrictions MIT does not carry",
    "other": "a placeholder, not an identifier: nobody can act on it",
    "unknown": "a placeholder, not an identifier: nobody can act on it",
}
"""Identifiers this project recognizes and refuses, each with the reason it is refused.

Separate from "unrecognized" on purpose. Both abort, and the diagnosis differs: a refused licence
means the source cannot be used as it stands, and an unrecognized one means somebody has to read it
and decide, which is a different next action for the human reading the abort.
"""

KIND_BASELINE: Final[str] = "baseline"
KIND_ATTACK_DATASET: Final[str] = "attack_dataset"
KIND_EXCLUSION_SOURCE: Final[str] = "exclusion_source"
KIND_BENIGN_CODE: Final[str] = "benign_code_repository"
KIND_HAND_AUTHORED: Final[str] = "hand_authored"

LOCAL_REPOSITORY: Final[str] = "normalize-before-classify"
LOCAL_LICENCE: Final[Licence] = Licence(
    identifier="MIT",
    source="LICENSE at the root of this repository",
    attribution=(
        "normalize-before-classify, MIT, nbc/corpus/sources/encoded_messages.py "
        "(written for this project)"
    ),
    redistributed=True,
)
"""The hand-authored B-chat items: this project's own material, under this project's own licence.

FR5.1 restricts hand-authored material to what no public dataset carries -- a message legitimately
containing a JWT, a content hash, a data URI or an SSH public key -- and those 20 items are written
in `nbc/corpus/sources/encoded_messages.py`. They are redistributed like every other row and are
credited like every other source; what makes them different is only that the licence is this
repository's own, which `tests/corpus/test_attribution.py` checks against the `LICENSE` file rather
than taking from this constant.
"""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One pinned source, in the shape the licence gate and the credits file both need.

    A single record type for five kinds of pin, because the gate's question is the same for all of
    them and a per-kind branch is a per-kind place to forget one. `revision` is empty exactly for a
    source whose pin carries none -- today the 401-gated exclusion source, and the attack dataset
    seen from the attack rows, which name the repository without a sha.
    """

    kind: str
    key: str
    repository: str
    revision: str
    licence: Licence

    @property
    def identifier(self) -> str:
        return self.licence.identifier.strip().casefold()

    @property
    def label(self) -> str:
        """How this source is named in a problem message and in the credits file."""
        return f"{self.kind} {self.key} ({self.repository}" + (
            f"@{self.revision})" if self.revision else ")"
        )


def pinned_sources(pins: Pins) -> tuple[SourceRecord, ...]:
    """Every source this build touches, pinned or local, in a fixed order.

    The order is declaration order within each kind and the kinds in the order the build meets
    them, so two runs of the same pins render byte-identical credits -- which is what makes the
    regeneration check in `verify-corpus` a check rather than a diff of two orderings.
    """
    records: list[SourceRecord] = []
    for baseline in pins.baselines:
        records.append(
            SourceRecord(
                kind=KIND_BASELINE,
                key=baseline.key,
                repository=baseline.repository,
                revision=baseline.revision,
                licence=baseline.licence,
            )
        )
    for dataset in pins.attack_datasets:
        records.append(
            SourceRecord(
                kind=KIND_ATTACK_DATASET,
                key=dataset.key,
                repository=dataset.repository,
                revision=dataset.revision,
                licence=dataset.licence,
            )
        )
    for source in pins.exclusion_sources:
        records.append(
            SourceRecord(
                kind=KIND_EXCLUSION_SOURCE,
                key=source.key,
                repository=source.repository,
                revision=source.revision,
                licence=source.licence,
            )
        )
    for repository in pins.benign_frame.b_code.repositories:
        records.append(
            SourceRecord(
                kind=KIND_BENIGN_CODE,
                key=repository.key,
                repository=repository.repository,
                revision=repository.revision,
                licence=repository.licence,
            )
        )
    records.append(
        SourceRecord(
            kind=KIND_HAND_AUTHORED,
            key=HAND_AUTHORED_SOURCE,
            repository=LOCAL_REPOSITORY,
            revision="",
            licence=LOCAL_LICENCE,
        )
    )
    return tuple(records)


def licence_problems(pins: Pins) -> tuple[str, ...]:
    """Every reason this build may not publish what it is about to publish.

    Pure, and cheap: it reads `pins.toml` and nothing else, which is why `build.py` calls it before
    the first byte is fetched. A build that would redistribute unlicensed material must not first
    download twelve exclusion sources and sixty-three archives to find that out.

    Every problem is collected before any is raised, so one run tells a reader everything that has
    to change rather than one thing per re-run.
    """
    problems: list[str] = []
    for record in pinned_sources(pins):
        licence = record.licence

        # The attribution is the evidence for the credit this repository will publish, so it is
        # compared against the identity it claims rather than merely being non-empty.
        if record.repository not in licence.attribution:
            problems.append(
                f"{record.label} declares an attribution that does not name the repository: "
                f"{licence.attribution!r}. The attribution is what `ATTRIBUTION.md` publishes, "
                f"and a credit that does not name what it credits is not a credit"
            )
        if record.revision and record.revision not in licence.attribution:
            problems.append(
                f"{record.label} declares an attribution that does not name the pinned revision "
                f"{record.revision}: {licence.attribution!r}. A credit naming a repository "
                f"without a revision credits whatever that repository holds today"
            )

        if not licence.redistributed:
            continue

        identifier = record.identifier
        # `Licence.blocks_redistribution` is the pin layer's own name for material this repository
        # ships whose licence nobody established, and `Licence.refuses_publication` is that fact
        # together with the absence of a human decision about it. Both are called rather than
        # restated: D-C of the Epic 1 decisions is about exactly these properties being published
        # and consumed by nothing, and a second spelling of either rule here would be a second
        # place it can drift from. The empty-identifier limb is the one case neither property
        # covers, since `"" != "not-declared"` reads as declared -- and it is tested first so an
        # acceptance can never stand in for a reading that was never made.
        if not identifier:
            problems.append(
                f"{record.label} is redistributed into data/ and declares an empty licence "
                f"identifier. `not-declared` is a reading of the publisher's card; an empty "
                f"string is a field nobody filled in, and there is nothing to accept about a "
                f"reading that does not exist"
            )
        elif licence.refuses_publication:
            problems.append(
                f"{record.label} is redistributed into data/ and declares no licence "
                f"({licence.source}). Nothing in this repository can choose one on the "
                f"publisher's behalf"
                + (
                    f". The pin records an open question rather than a waiver: {licence.unresolved}"
                    if licence.unresolved
                    else ""
                )
            )
        elif licence.blocks_redistribution:
            # Undeclared, and a named human accepted publishing it anyway on a stated date with
            # the reasoning in the README. The identifier stays `not-declared` from here into
            # `ATTRIBUTION.md` and into `results.json`, so nothing downstream can read this as a
            # grant: what exists is a decision, not a licence, and the two are rendered apart.
            continue
        elif identifier in REFUSED:
            problems.append(
                f"{record.label} is redistributed into data/ under {licence.identifier!r}, which "
                f"this project refuses: {REFUSED[identifier]}"
            )
        elif identifier not in COMPATIBLE:
            problems.append(
                f"{record.label} is redistributed into data/ under {licence.identifier!r}, which "
                f"is not in this project's compatible set {sorted(COMPATIBLE)} nor in its refused "
                f"set. An unrecognized identifier is a licence somebody has to read and decide on, "
                f"in `corpus/attribution.py`"
            )
    return tuple(problems)


def _b_code_identity(source: str) -> tuple[str, str] | None:
    """`github.com/{repository}@{revision}:{path}` parsed into its pinned pair, or `None`.

    Parsed, never matched: the caller compares the returned pair against a pinned pair, so a row
    naming the right repository at the wrong sha is refused. A prefix or substring rule would
    accept it.
    """
    prefix = "github.com/"
    if not source.startswith(prefix):
        return None
    remainder = source[len(prefix) :]
    located, separator, path = remainder.partition(":")
    if not separator or not path:
        return None
    repository, at, revision = located.rpartition("@")
    if not at or not repository or not revision:
        return None
    return repository, revision


def counts_by_key(
    items: Iterable[CorpusItem], pins: Pins
) -> tuple[Mapping[str, int], tuple[str, ...]]:
    """How many rows each pinned source contributed, and every row nothing pinned accounts for.

    The tally is over `CorpusItem.source` -- the identity AD-2 already puts in every row -- so it
    describes the corpus that was written rather than the draw that was declared.
    """
    exact: dict[str, str] = {}
    by_pair: dict[tuple[str, str], str] = {}
    for record in pinned_sources(pins):
        if record.kind == KIND_ATTACK_DATASET:
            # Attack rows name the repository alone (`corpus/attack.py` renders them that way) and
            # B-chat rows name `repository@revision`; both are this one pin.
            exact[record.repository] = record.key
            exact[f"{record.repository}@{record.revision}"] = record.key
        elif record.kind == KIND_BENIGN_CODE:
            by_pair[(record.repository, record.revision)] = record.key
        elif record.kind == KIND_HAND_AUTHORED:
            exact[record.key] = record.key

    counts: Counter[str] = Counter()
    unattributed: Counter[str] = Counter()
    for item in items:
        key = exact.get(item.source)
        if key is None:
            identity = _b_code_identity(item.source)
            key = by_pair.get(identity) if identity is not None else None
        if key is None:
            unattributed[item.source] += 1
        else:
            counts[key] += 1

    problems = tuple(
        f"{rows} corpus row(s) name source {source!r}, which is not a pinned redistributing "
        f"source. Every row this repository publishes is credited to a source with a licence, "
        f"so a row nothing accounts for is a row nobody may publish"
        for source, rows in sorted(unattributed.items())
    )
    return dict(counts), problems


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render(pins: Pins, counts: Mapping[str, int], *, build_id: str) -> str:
    """`data/ATTRIBUTION.md`: every source, its licence, its revision and its row count.

    Generated on every build and regenerated by `verify-corpus`, which is what keeps it from
    drifting the way a hand-maintained credits file does. `build_id` travels in the header because
    a credits file that does not say which corpus it credits is a credits file a reader cannot
    check.
    """
    records = pinned_sources(pins)
    redistributed = [record for record in records if record.licence.redistributed]
    consulted = [record for record in records if not record.licence.redistributed]

    lines: list[str] = [
        "# Attribution",
        "",
        "**This file is generated. Do not edit it.** It is written by",
        "`python -m nbc.corpus.build build-corpus` and regenerated by",
        "`python -m nbc.corpus.build verify-corpus`, which refuses a corpus whose committed",
        "attribution differs from the one this declaration produces. A hand-maintained credits",
        "file drifts from what was actually redistributed and the drift is invisible, which is the",
        "same reason gold labels are emitted by the builder rather than annotated (FR4, FR5.2).",
        "",
        f"Corpus `build_id`: `{build_id}`",
        "",
        "## Redistributed into `data/`",
        "",
        "Rows in the two corpus halves under `data/` come from these sources. The row count",
        "is counted from the `source` field of the rows actually written, not from the declared",
        "draw: one drawn item becomes one row per dressing chain, and a credits file derived from",
        "the declaration would state a different number from the one on disk.",
        "",
        _row(["Source", "Kind", "Licence", "Revision", "Rows"]),
        _row(["---", "---", "---", "---", "---:"]),
    ]
    for record in redistributed:
        lines.append(
            _row(
                [
                    f"`{record.repository}`",
                    record.kind,
                    record.licence.identifier,
                    f"`{record.revision}`" if record.revision else "--",
                    str(counts.get(record.key, 0)),
                ]
            )
        )
    # Immediately under the table rather than at the end: a reader who stops at the first
    # `not-declared` cell has to meet this paragraph without scrolling for it.
    accepted = [
        (record, acceptance)
        for record in redistributed
        if (acceptance := record.licence.accepted) is not None
    ]
    if accepted:
        lines += [
            "",
            "### Published without a licence, by decision",
            "",
            "The sources below declare **no licence** at their pinned revision, and nothing in",
            "this repository chose one on their behalf -- their identifier in the table above",
            "still reads `not-declared`. What follows is a named person's dated decision to",
            "publish the rows anyway, with the reasoning in this repository's `README.md` rather",
            "than in a generated file.",
            "",
            "**If you redistribute this corpus further, read this before the licence column.**",
            "Nobody granted a licence for these rows, and the MIT licence at the root of this",
            "repository does not reach them.",
            "",
        ]
        for record, acceptance in accepted:
            lines += [
                f"- `{record.repository}` -- accepted by {acceptance.by} on "
                f"{acceptance.on}; position stated in `{acceptance.position}`.",
                # Collapsed to one line because the declaration is a TOML multi-line string and a
                # raw newline here would end the list item and silently drop the rest of it.
                f"  {' '.join(acceptance.reasoning.split())}",
            ]

    lines += [
        "",
        "### Required attribution, as each source states it",
        "",
    ]
    for record in redistributed:
        lines.append(f"- {record.licence.attribution}")
    lines += [
        "",
        "## Consulted, not redistributed",
        "",
        "No byte of these reaches `data/`. The baselines are fetched into the Hugging Face cache",
        "and never committed; an exclusion source's rows are intersected against the corpus and",
        "every match is **removed**. Their licences are read and recorded anyway, because the rule",
        "is that every pinned source carries one.",
        "",
        _row(["Source", "Kind", "Licence", "Revision"]),
        _row(["---", "---", "---", "---"]),
    ]
    for record in consulted:
        lines.append(
            _row(
                [
                    f"`{record.repository}`",
                    record.kind,
                    record.licence.identifier,
                    f"`{record.revision}`" if record.revision else "--",
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def attribution_problems(committed: str | None, expected: str) -> tuple[str, ...]:
    """Every way the committed credits file differs from the one this build would generate.

    `None` means the file is not there. A digest of each side rather than a diff: the message has to
    name the failure, and the fix is always the same one command, so a reader needs to know *that*
    it drifted and over how much, not which line moved.
    """
    if committed is None:
        return (
            f"{ATTRIBUTION_FILENAME} is not beside the corpus. It is generated by the build that "
            f"writes the corpus, so a corpus without one was either written by something else or "
            f"had its credits removed",
        )
    if committed == expected:
        return ()
    return (
        f"{ATTRIBUTION_FILENAME} is not the file this declaration generates: it holds "
        f"{len(committed)} bytes hashing to "
        f"{hashlib.sha256(committed.encode('utf-8')).hexdigest()[:16]} and the generated text is "
        f"{len(expected)} bytes hashing to "
        f"{hashlib.sha256(expected.encode('utf-8')).hexdigest()[:16]}. It is generated, never "
        f"hand-maintained: rebuild the corpus rather than editing it",
    )
