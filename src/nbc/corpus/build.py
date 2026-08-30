"""The corpus builder, and the only module in this project that imports `datasets`.

Story 3.1 fills in one half of it: the network side of the training-overlap filter. The decision
procedure -- what the exclusion set is, what counts as the same text, which source may be missing
and which may not -- is `corpus/exclusion.py` and is pure. This module is what reaches the hub:
it probes each declared source, loads its rows, hands the texts to the index, and hands the
observations back to the gate.

**The import rule, and why it is a rule.** `datasets` is declared in the `build` optional group,
never in the runtime dependencies, and it is imported **inside a function here and nowhere else**.
Two tests hold that: an AST scan over `src/` and `spikes/` for the name, and a subprocess that
imports this module and asserts `datasets` did not land in `sys.modules`. The measurement path's
offline guarantee is the reason -- a build-time dependency that a runtime import drags in is a
runtime dependency that nobody declared.

**Why a row is walked for every string it holds.** The alternative is a declared text column per
source, and its failure mode is silent: a column name that stopped being right yields zero matches
and looks exactly like a source with no overlap. It is also wrong on its face for at least one
pinned source, whose text lives inside nested `messages`/`chosen`/`rejected` records rather than in
any top-level string column. So every string a row holds, at any depth, enters the index. The cost
is stated rather than hidden: short label values (`"user"`, `"safe"`) enter it too, so a corpus row
that *is* one of those words would be removed. That errs toward removal, which costs sample size
and never validity, and the per-source counts published beside the table make an absurd removal
visible. What replaces the column declaration as a check is `texts_loaded > 0` per source, in
`exclusion.verify_observations`.

**The CLI is subcommands, and that is AD-1's requirement rather than a style choice.**

    python -m nbc.corpus.build exclusion-report   # probe and load every exclusion source
    python -m nbc.corpus.build build-attack       # draw and write data/attack.jsonl
    python -m nbc.corpus.build rebuild-attack     # the same, over a corpus that already exists
    python -m nbc.corpus.build build-corpus       # both halves and data/manifest.json
    python -m nbc.corpus.build rebuild-corpus     # the same, over a corpus that already exists
    python -m nbc.corpus.build verify-corpus      # the guarded read, over what is on disk

Rebuilding an existing corpus is an **explicit subcommand and never a side effect**: `build-attack`
aborts with `CorpusWriteRefused` rather than overwriting, so no run can replace the corpus a table
was computed over while looking like it did something else.

`build-corpus` is the one that produces a **measurable** corpus, and `build-attack` deliberately
does not: only `build-corpus` writes `data/manifest.json`, and `manifest.read_corpus` -- the only
door into `data/*.jsonl` -- refuses without one. An attack half on disk with no manifest is
therefore inert rather than half-usable, which is the outcome FR5.1 asks for one level up.
`build-attack` survives because it is the cheap path that proves the gold-label abort fires against
the live pool, and CI runs it for exactly that.

Every subcommand except `verify-corpus` touches the network, so none of those is part of the
offline unit suite.

**Where B-code comes from, and why not the REST API.** Each pinned repository is read once, as the
gzipped tar of its tree at the pinned sha, streamed rather than downloaded whole. Listing a tree
through GitHub's REST API costs one request against an unauthenticated budget of sixty an hour and
the frame pins more repositories than that; `codeload` carries no such budget. Nothing here reads a
token from the environment, so the build has the same access a stranger has.

**This module is the only writer under `data/`** (AD-1) -- the two corpus halves, the manifest and
the generated `ATTRIBUTION.md`. An AST scan over `src/` and `spikes/`
in `tests/corpus/test_build.py` holds that: it collects every write primitive in the tree and
refuses any file outside a declared allow-list, and the scan's own predicate is tested against a
synthetic offender so it cannot pass by failing to look.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Iterable, Iterator, Sequence

from nbc.corpus.attribution import (
    ATTRIBUTION_FILENAME,
    RedistributionRefused,
    attribution_problems,
    counts_by_key,
    licence_problems,
    render as render_attribution,
)
from nbc.corpus.attack import (
    AttackDrawReport,
    AttackDrawUnsatisfiable,
    LabelContradiction,
    PoolRow,
    WithdrawalDoesNotMatchPool,
    contradictions,
    draw_attack_items,
    serialize,
    verify_splits,
    withdraw,
)
from nbc.corpus.benign import (
    CodeFile,
    SourceFile,
    default_eligibility_context,
    draw_benign_items,
    select_repository_files,
)
from nbc.corpus.manifest import (
    ATTACK_CORPUS_FILENAME,
    BENIGN_CORPUS_FILENAME,
    DATA_DIRNAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    CorpusManifestMismatch,
    Manifest,
    build_id,
    confirmatory_cell_problems,
    corpus_directory,
    files_for,
    read_corpus,
    render as render_manifest,
)
from nbc.corpus.exclusion import (
    NO_ANSWER,
    NORMALIZATION,
    ExclusionIndex,
    ExclusionReport,
    Observation,
    PlannedSource,
    build_index,
    declaration_digest,
    filter_rows,
    normalized_texts,
    outcomes_of,
    plan,
    verify_observations,
)
from nbc.errors import NbcError, exit_code_for
from nbc.schema import CorpusItem
from nbc.pins import (
    HTTP_OK,
    AttackDataset as AttackDatasetPin,
    BenignCodeRepository,
    ExclusionSource,
    Pins,
    load_pins,
)

__all__ = [
    "ARCHIVE_MEMBER_LIMIT",
    "ATTRIBUTION_FILENAME",
    "ARCHIVE_TIMEOUT_SECONDS",
    "ATTACK_CORPUS_FILENAME",
    "BENIGN_CORPUS_FILENAME",
    "CorpusWriteRefused",
    "DATA_DIRNAME",
    "HTTP_TIMEOUT_SECONDS",
    "RepositoryUnreadable",
    "attribution_text",
    "build_attack_corpus",
    "build_corpus",
    "corpus_directory",
    "iter_exclusion_texts",
    "main",
    "observe_exclusion_sources",
    "probe",
    "read_attack_pool",
    "read_benign_code",
    "read_benign_rows",
    "read_exclusion_index",
    "read_repository_files",
    "selection_overlap",
    "write_corpus",
]

class CorpusWriteRefused(NbcError, exit_code=18):
    """A corpus file already exists and this call was not the explicit rebuild.

    Code 18. AD-1 requires that rebuilding an existing corpus be an explicit subcommand and never
    a side effect of anything else, and a refusal that has its own exit code is what lets a caller
    tell "you meant `rebuild-attack`" apart from "the data is contradictory".
    """


HTTP_TIMEOUT_SECONDS: Final[float] = 30.0
"""How long one hub probe may take. A timeout is `NO_ANSWER`, which fails the declared status.

The same 30 seconds `pins.py` gives its own resolver, arrived at separately rather than imported:
that one is a private constant of a module this project keeps as a leaf, and reaching into it to
save a line would make the pin reader's internals part of the corpus builder's contract.
"""


def probe(source: ExclusionSource, timeout: float = HTTP_TIMEOUT_SECONDS) -> int:
    """The HTTP status the hub answers for this source, or `NO_ANSWER` if it answered nothing.

    The status is the observation `verify_observations` compares against the pinned one, which is
    why a failure is reported as a status rather than raised: "the hub could not be reached" and
    "the hub said 404" are different diagnoses, and both must fail the comparison rather than
    escape as an unclassified crash.
    """
    import http.client
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        source.probe_url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as answered:
        # An HTTP error IS an answer, and it is the one that matters here: 401 is what the
        # access-restricted source declares, and losing it in the generic handler below would
        # turn a checked fact into "the network was unreachable".
        return int(answered.code)
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        OSError,
        http.client.HTTPException,
    ):
        # The same five `resolve_over_http` catches, for the same reason: `HTTPException` is
        # neither an `OSError` nor a `URLError`, and a malformed URL raises `ValueError` before
        # any socket opens. Either one escaping would turn "the hub did not answer" into an
        # unclassified crash, losing the exit code that says which failure this was.
        return NO_ANSWER


def _strings_in(value: object) -> Iterator[str]:
    """Every string a loaded row holds, at any depth.

    A row is a dict of columns, and a column can be a string, a list of strings, or a list of
    records -- one pinned source keeps its text inside nested role/content records. Walking the
    value is what reaches all three without a per-source declaration of where to look.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings_in(item)


def iter_exclusion_texts(source: ExclusionSource) -> Iterator[str]:
    """Every string in every row of every config and every split, at the pinned revision.

    **Every config, never one.** A dataset with five configs read as one is four fifths of a
    training source silently treated as contributing zero, which is the exact failure this whole
    filter exists to prevent -- one level down from reading one split instead of both.

    A generator rather than a list: the largest pinned source is a third of a million rows with
    several strings each, and the caller only ever wants the distinct normalized keys.
    """
    import datasets

    configs = datasets.get_dataset_config_names(
        source.repository, revision=source.revision
    )
    for config in configs:
        loaded = datasets.load_dataset(
            source.repository, config, revision=source.revision
        )
        for split in loaded:
            for row in loaded[split]:
                yield from _strings_in(row)


def observe_exclusion_sources(
    planned: tuple[PlannedSource, ...],
) -> tuple[dict[str, Observation], dict[str, set[str]]]:
    """Probe and load every planned source. Returns what was seen, and the keys to index.

    **A load is attempted wherever the hub answers, including for a source the pins call
    unreadable.** Skipping the ones the pins say will fail would leave that declaration compared
    to nothing -- and a source that quietly became readable would go on being reported as a gap
    forever, with rows this run could have removed left in the corpus.

    Whether an outcome is a limit to publish or a reason to stop is `verify_observations`', not
    this function's: deciding it here would put the same rule in two places.
    """
    observations: dict[str, Observation] = {}
    texts_by_source: dict[str, set[str]] = {}

    for entry in planned:
        status = probe(entry.source)
        if status != HTTP_OK:
            observations[entry.key] = Observation(http_status=status)
            continue

        try:
            # Normalized as it streams, so no training source is ever held whole in memory.
            keys = normalized_texts(iter_exclusion_texts(entry.source))
        except Exception as refusal:  # noqa: BLE001 - see below
            # Deliberately broad, and it is not a swallow: `datasets` reports a repository it
            # will not load as `RuntimeError`, a missing config as `ValueError`, a network fault
            # as any of a dozen library-specific types, and the pinned reader's exception
            # taxonomy is not something this project may pin. The refusal is not discarded --
            # it becomes the observation `verify_observations` compares against the pins, and it
            # is published verbatim in the report.
            observations[entry.key] = Observation(
                http_status=status,
                loadable=False,
                load_error=f"{type(refusal).__name__}: {refusal}",
            )
            continue

        texts_by_source[entry.key] = keys
        observations[entry.key] = Observation(
            http_status=status, loadable=True, texts_loaded=len(keys)
        )

    return observations, texts_by_source


def read_exclusion_index(
    pins: Pins,
) -> tuple[ExclusionIndex, tuple[PlannedSource, ...], dict[str, Observation]]:
    """The whole network half: plan, probe, load, verify. Aborts before returning an index.

    The verification runs before the index is handed back, so no caller can filter a corpus
    against a set the pins do not describe.
    """
    planned = plan(pins)
    observations, texts_by_source = observe_exclusion_sources(planned)
    verify_observations(planned, observations)
    return build_index(texts_by_source), planned, observations


TEXT_COLUMN: Final[str] = "text"
LABEL_COLUMN: Final[str] = "label"
"""The two columns the pinned attack dataset publishes, read by name and checked before use.

Named here rather than declared per source in `pins.toml`: one dataset is pinned, both columns are
in its declared feature list at the pinned revision, and `read_attack_pool` aborts naming what it
found if either is absent. A declaration in the pin file would be a second home for a fact the
dataset itself carries, and the failure mode of a *wrong* declaration is the silent one -- zero
rows read as a dataset with nothing in it.

This is not the exclusion sources' problem, and the two are read differently on purpose: those are
twelve heterogeneous sources whose text hides at arbitrary depth, so they are walked. This is the
one dataset whose rows the corpus is made of, and a payload read out of the wrong column would be
the corpus rather than a miscount in a filter.
"""


def read_attack_pool(
    dataset: AttackDatasetPin,
) -> tuple[tuple[PoolRow, ...], tuple[str, ...], tuple[PoolRow, ...]]:
    """The pinned attack pool this build may use, the splits observed, and the rows withdrawn.

    Returns the rows and the splits **as observed**, never as declared: the caller compares the
    two, in both directions. Handing back the declared list would make that comparison a check of
    a value against itself, which is the pattern this project keeps finding in its own history.

    **The withdrawal is applied here, at the one door, and that placement is a bug fix.** It used
    to live inside `draw_attack_items`, which meant the attack half saw the filtered pool and
    `read_benign_rows` -- reading the same tuple, one call later -- saw the unfiltered one. The
    withdrawn texts went straight back into the corpus through the benign half, and `2026-08-30`'s
    first full build caught it only because `selection_overlap` fired on rows that should not have
    been in either half. The rule now has one home and the pool has one reader: nothing downstream
    can hold rows this build declared it does not use, because nothing downstream is handed them.

    Aborting rather than filtering silently: a declaration that does not describe the pool exactly,
    in either direction, is `WithdrawalDoesNotMatchPool`.

    **The withdrawn rows come back out of the same door** rather than being dropped on the floor.
    A caller that wants the artifact as published -- the smoke test that compares the pool's
    contradictions against a count a human reviewed -- reconstructs it from the two halves. The
    alternative was a second reader returning unfiltered rows, and a second reader is how the
    benign half came to be reading a different pool from the attack half in the first place.

    `datasets.load_dataset` with no `split` argument returns every split, which is what makes the
    observation an observation. Configs are enumerated for the same reason `iter_exclusion_texts`
    enumerates them: a dataset read at one config is a fraction of a source silently treated as
    the whole of it.
    """
    import datasets

    configs = datasets.get_dataset_config_names(
        dataset.repository, revision=dataset.revision
    )
    rows: list[PoolRow] = []
    observed: list[str] = []
    for config in sorted(configs):
        loaded = datasets.load_dataset(
            dataset.repository, config, revision=dataset.revision
        )
        for split in sorted(loaded):
            table = loaded[split]
            missing = [
                column
                for column in (TEXT_COLUMN, LABEL_COLUMN)
                if column not in table.column_names
            ]
            if missing:
                raise AttackDrawUnsatisfiable(
                    f"{dataset.repository}@{dataset.revision} split {split!r} does not publish "
                    f"column(s) {missing}; it publishes {sorted(table.column_names)}. A payload "
                    f"read out of the wrong column would be the corpus, not a miscount"
                )
            # A split name is unique per config here because one config is pinned; a second
            # config would collide, so the observed name carries the config when there is more
            # than one rather than silently merging two splits under one label.
            name = split if len(configs) == 1 else f"{config}/{split}"
            observed.append(name)
            texts = table[TEXT_COLUMN]
            labels = table[LABEL_COLUMN]
            rows.extend(
                PoolRow(split=name, index=index, text=text, label=label)
                for index, (text, label) in enumerate(zip(texts, labels))
            )

    as_published = tuple(rows)
    surviving, problems = withdraw(as_published, dataset.withdrawn)
    if problems:
        raise WithdrawalDoesNotMatchPool(*problems)
    kept = set(surviving)
    return surviving, tuple(observed), tuple(row for row in as_published if row not in kept)


def refuse_unlicensed_redistribution(pins: Pins) -> None:
    """AD-34's gate: abort before anything is fetched if a redistributed source has no licence.

    A pure read of `pins.toml`, so it costs nothing and can therefore sit ahead of every other
    step in both writing paths. The abort names the source, its revision and what is wrong with
    its declaration; `corpus/attribution.py` owns the vocabulary and the reasons.
    """
    problems = licence_problems(pins)
    if problems:
        raise RedistributionRefused(*problems)


def attribution_text(pins: Pins, items: Iterable[CorpusItem], build_id_value: str) -> str:
    """The generated `ATTRIBUTION.md` for these rows, or an abort if a row credits nothing.

    Counted from the rows themselves rather than from the draw report, so the file describes the
    corpus on disk. One function for both the build and `verify-corpus`, which is what makes the
    regeneration check a check: two renderers could drift from each other and neither would fail.
    """
    counts, problems = counts_by_key(items, pins)
    if problems:
        raise RedistributionRefused(*problems)
    return render_attribution(pins, counts, build_id=build_id_value)


def build_attack_corpus(
    pins: Pins, *, root: str | None = None, rebuild: bool = False
) -> tuple[AttackDrawReport, ExclusionReport, Path, int]:
    """The whole attack build: pool, exclusion index, gates, draw, write.

    The exclusion index is read **before** the pool is drawn against it and after its own
    verification, so a corpus is never filtered against a set the pins do not describe. Both
    reports come back so the caller publishes the accounting for what it removed as well as for
    what it kept.

    **The licence gate runs first**, before the pool is read. This path writes `data/attack.jsonl`,
    which is redistribution exactly as `build-corpus` is, and a build that may not publish what it
    is about to publish must not first download the pool to find that out.
    """
    refuse_unlicensed_redistribution(pins)

    if len(pins.attack_datasets) != 1:
        # `load_pins` requires at least one. More than one is a real possibility FR1 leaves open,
        # and it needs a declared rule for merging two label vocabularies and two draws. Nothing
        # here invents one.
        raise AttackDrawUnsatisfiable(
            f"{len(pins.attack_datasets)} attack datasets are pinned and this build implements "
            f"the draw for exactly one; merging two pools needs a declared rule for their label "
            f"vocabularies and their sample sizes, and none is declared"
        )
    dataset = pins.attack_datasets[0]

    rows, observed_splits, _withdrawn = read_attack_pool(dataset)

    # The exclusion index is the largest download this build makes, and it is built lazily so a
    # pool that fails a cheap gate -- a text carried at both labels, a split nobody declared --
    # aborts before any of it is fetched. `draw_attack_items` decides when; this closure only
    # keeps what the report needs afterwards.
    read: dict[str, object] = {}

    def index_of() -> ExclusionIndex:
        index, planned, observations = read_exclusion_index(pins)
        read["planned"] = planned
        read["observations"] = observations
        return index

    items, _payloads, draw_report, matches = draw_attack_items(
        rows, observed_splits, dataset, index_of
    )
    planned = read["planned"]
    observations = read["observations"]

    exclusion_report = ExclusionReport(
        normalization=NORMALIZATION,
        declaration_digest=declaration_digest(pins),
        rows_in=draw_report.unique_positives,
        rows_removed=draw_report.removed_by_exclusion,
        outcomes=outcomes_of(planned, observations, matches),  # type: ignore[arg-type]
    )

    directory = corpus_directory(root)
    path = directory / ATTACK_CORPUS_FILENAME
    # The credits are rendered before the rows are written and land beside them, so there is no
    # state in which a file of redistributed rows sits on disk without generated credits for it.
    # This half carries no manifest by design (see the module docstring), which is what makes it
    # unreadable through the guarded door; that is a separate property from being uncredited.
    credits = attribution_text(pins, items, build_id(pins))
    credits_path = directory / ATTRIBUTION_FILENAME
    if credits_path.exists() and not rebuild:
        raise CorpusWriteRefused(
            f"{credits_path} already exists and this is not a rebuild"
        )
    written = write_corpus(path, items, rebuild=rebuild)
    directory.mkdir(parents=True, exist_ok=True)
    credits_path.write_text(credits, encoding="utf-8", newline="\n")
    return draw_report, exclusion_report, path, written



ARCHIVE_TIMEOUT_SECONDS: Final[float] = 300.0
"""How long one repository archive may take. Longer than a hub probe, because it is a whole tree."""

ARCHIVE_MEMBER_LIMIT: Final[int] = 200_000
"""How many entries one pinned archive may hold before the build refuses to keep reading.

A pin at a sha is this build's trust boundary for a git repository -- there is no second party to
verify the bytes against -- so the guard that matters is the one that keeps a pathological archive
from being read forever rather than one that tries to judge its contents. It is a **loud abort
naming the repository**, never a silent truncation: a repository read halfway would contribute a
different candidate set from the one its sha describes, and nothing downstream could tell.

The other half of the guard is the frame's own size band: no member above `max_file_bytes` is ever
read, so the bytes this build holds are bounded by that number times the members it accepts.
"""


class RepositoryUnreadable(NbcError, exit_code=23):
    """A pinned benign-code repository could not be read at its pinned sha.

    Code 23 because 3 through 22 are taken. An abort rather than a skip, and the reason is FR5.1's:
    a repository silently contributing zero is a frame quietly drawing from fewer sources than it
    declares, and the floor on realized repositories would then be met by whatever happened to be
    reachable that afternoon. A 404 here usually means the sha moved or the repository was renamed,
    which is a pin that has stopped describing the world -- exactly what `--verify` catches for the
    Hugging Face artifacts and what nothing else could catch for these.
    """


def read_repository_files(
    repository: BenignCodeRepository,
    minimum_bytes: int,
    maximum_bytes: int,
    timeout: float = ARCHIVE_TIMEOUT_SECONDS,
) -> Iterator[SourceFile]:
    """Every file in one pinned repository, at its pinned sha, inside the frame's size band.

    **Streamed, never buffered.** `tarfile` in `r|gz` mode reads forward through the response, so a
    repository is never held whole in memory or on disk. Nothing is extracted: members are read out
    of the stream and decoded, so no path from the archive ever reaches the filesystem and the
    directory-traversal question does not arise.

    The size band applied here is the **frame's own**, passed in rather than read from a constant of
    this module, and it is a read guard rather than a second rule: `benign.eligible` remains the
    authority and re-checks it. Skipping a member the frame could never accept is what keeps the
    build from decoding a repository's vendored bundles to throw them away.

    A member whose bytes are not UTF-8 is skipped. `UnicodeDecodeError` is a `ValueError`, and the
    narrow name alone would let the sibling escape.
    """
    import tarfile
    import urllib.error
    import urllib.request

    try:
        response = urllib.request.urlopen(repository.archive_url, timeout=timeout)
    except urllib.error.HTTPError as answered:
        raise RepositoryUnreadable(
            f"{repository.repository} at {repository.revision} answered HTTP {answered.code}; the "
            f"pin names a commit this host no longer serves, so the frame is drawing from a "
            f"repository nobody can reproduce"
        ) from None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as refusal:
        raise RepositoryUnreadable(
            f"{repository.repository} at {repository.revision} could not be fetched: "
            f"{type(refusal).__name__}: {refusal}"
        ) from None

    seen = 0
    with response:
        try:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    seen += 1
                    if seen > ARCHIVE_MEMBER_LIMIT:
                        raise RepositoryUnreadable(
                            f"{repository.repository} at {repository.revision} holds more than "
                            f"{ARCHIVE_MEMBER_LIMIT} entries; the build stops rather than reading "
                            f"part of a repository and reporting it as the whole"
                        )
                    if not member.isfile():
                        continue
                    if not minimum_bytes <= member.size <= maximum_bytes:
                        continue
                    # `codeload` wraps the tree in one top-level directory named for the sha.
                    # Dropping exactly one component is what turns the archive's path into the
                    # repository path a reader can open on the web.
                    _root, _slash, path = member.name.partition("/")
                    if not path:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    try:
                        text = handle.read().decode("utf-8")
                    except ValueError:
                        continue
                    yield SourceFile(path=path, text=text)
        except tarfile.TarError as broken:
            raise RepositoryUnreadable(
                f"{repository.repository} at {repository.revision} is not a readable archive: "
                f"{type(broken).__name__}: {broken}"
            ) from None


def read_benign_code(pins: Pins) -> dict[str, tuple[CodeFile, ...]]:
    """Every pinned repository's contribution to B-code, capped by the frame, keyed by pin key.

    The layer context is built **once**, here, and handed down: `default_context()` reads and
    validates the vendored confusables table on every call by design, and the eligibility rule runs
    over every candidate file in sixty-three repositories.

    A repository that yields no eligible file is present in the result with an empty tuple rather
    than absent, so the difference between "read and contributed nothing" and "never read" survives
    into the report.
    """
    frame = pins.benign_frame
    ctx = default_eligibility_context()
    contributions: dict[str, tuple[CodeFile, ...]] = {}
    for repository in frame.b_code.repositories:
        files = read_repository_files(
            repository, frame.b_code.min_file_bytes, frame.b_code.max_file_bytes
        )
        contributions[repository.key] = select_repository_files(repository, files, frame, ctx)
        print(
            f"{repository.repository}: {len(contributions[repository.key])} eligible files",
            file=sys.stderr,
        )
    return contributions


def read_benign_rows(
    dataset: AttackDatasetPin, rows: Sequence[PoolRow]
) -> tuple[str, ...]:
    """The pinned dataset's benign texts: every row not carrying the declared attack label.

    Derived from `attack_label` rather than from a second declared value, because a benign label
    declared separately could disagree with the attack one and the disagreement would be invisible:
    a third label value would then be drawn into neither half and silently leave the corpus.
    """
    return tuple(
        sorted({row.text for row in rows if row.label != dataset.attack_label and row.text})
    )


def selection_overlap(
    dataset: AttackDatasetPin, rows: Sequence[PoolRow], benign_texts: Sequence[str]
) -> tuple[str, ...]:
    """One message if the two halves of the corpus were selected from overlapping text. Empty if not.

    A **different computation** from the contradiction gate `draw_attack_items` already ran: that
    one asks whether the *pool* carries one text under two labels, this one asks whether the two
    *selections* overlap. The failing input is a `read_benign_rows` that selected on the attack
    label instead of against it, which would put every attack payload into the benign corpus under a
    benign gold label and would not trip a single other check in this repository.

    A separate function so that input can be handed to it offline; the caller reaches it only after
    a download.
    """
    positives = {row.text for row in rows if row.label == dataset.attack_label and row.text}
    both = sorted(positives & set(benign_texts))
    if not both:
        return ()
    return (
        f"{len(both)} text(s) were selected into both halves of the corpus, starting with "
        f"{both[0]!r}; the benign selection is the complement of the attack label, and an overlap "
        f"means it is not",
    )


def build_corpus(
    pins: Pins, *, root: str | None = None, rebuild: bool = False
) -> tuple[Manifest, dict[str, object]]:
    """Both halves of the corpus and the manifest that identifies them. Aborts before it writes.

    The order is what makes an abort cheap where it can be:

    1. the pool, the contradiction gate and the split gate -- seconds, no archive fetched;
    2. the exclusion index -- the largest download, and the filter both halves depend on;
    3. the attack draw, which fails if the declared size does not survive the filter;
    4. the sixty-three archives, which is the longest step and the one worth not reaching if
       anything above it was going to fail anyway;
    5. the benign draw, then one write of three files.

    **Three files, one manifest, one `build_id`.** A corpus is both halves or it is not a corpus:
    writing one half against a declaration and the other against a later one is exactly what
    `build_id` exists to make impossible, and building them in one call is what makes the id
    describe both.

    Step 0 is the licence gate (AD-34), ahead of even the confirmatory cell: both are pure reads of
    `pins.toml`, and this one decides whether the rows may be published at all.
    """
    refuse_unlicensed_redistribution(pins)

    if len(pins.attack_datasets) != 1:
        raise AttackDrawUnsatisfiable(
            f"{len(pins.attack_datasets)} attack datasets are pinned and this build implements "
            f"the draw for exactly one"
        )
    dataset = pins.attack_datasets[0]

    # Before the first byte is fetched: the confirmatory cell has to name a cell this corpus will
    # actually carry rows in, or the whole build produces a verdict computed over nothing.
    cell_problems = confirmatory_cell_problems(pins)
    if cell_problems:
        raise CorpusManifestMismatch(*cell_problems)

    rows, observed_splits, _withdrawn = read_attack_pool(dataset)

    read: dict[str, object] = {}

    def index_of() -> ExclusionIndex:
        index, planned, observations = read_exclusion_index(pins)
        read["planned"] = planned
        read["observations"] = observations
        read["index"] = index
        return index

    attack_items, attack_payloads, attack_report, matches = draw_attack_items(
        rows, observed_splits, dataset, index_of
    )
    index: ExclusionIndex = read["index"]  # type: ignore[assignment]

    benign_texts = read_benign_rows(dataset, rows)

    overlap = selection_overlap(dataset, rows, benign_texts)
    if overlap:
        raise LabelContradiction(*overlap)
    filtered = filter_rows(benign_texts, index, lambda text: text)

    code_by_repository = read_benign_code(pins)

    benign_items, benign_report = draw_benign_items(
        frame=pins.benign_frame,
        code_by_repository=code_by_repository,
        chat_surviving=filtered.kept,
        dataset=dataset,
        chat_rows_in=len(benign_texts),
        chat_rows_removed=len(filtered.removed),
        # AD-27: the undressed attack payloads this same build drew, handed to the benign draw so
        # it can cross-check its own undressed sources against them before it renders a row.
        attack_payloads=attack_payloads,
    )

    exclusion_report = ExclusionReport(
        normalization=NORMALIZATION,
        declaration_digest=declaration_digest(pins),
        rows_in=attack_report.unique_positives + len(benign_texts),
        rows_removed=attack_report.removed_by_exclusion + len(filtered.removed),
        outcomes=outcomes_of(read["planned"], read["observations"], matches),  # type: ignore[arg-type]
    )

    directory = corpus_directory(root)
    payloads = (
        (ATTACK_CORPUS_FILENAME, serialize(attack_items), len(attack_items)),
        (BENIGN_CORPUS_FILENAME, serialize(benign_items), len(benign_items)),
    )
    for name in (
        *(entry[0] for entry in payloads),
        MANIFEST_FILENAME,
        ATTRIBUTION_FILENAME,
    ):
        target = directory / name
        if target.exists() and not rebuild:
            raise CorpusWriteRefused(
                f"{target} already exists and this is not a rebuild. Rebuilding an existing corpus "
                f"is an explicit subcommand and never a side effect of anything else, because a "
                f"corpus silently replaced mid-run publishes a table computed over two different "
                f"corpora"
            )

    identity = build_id(pins)
    # Rendered before anything is written, because a row nothing credits is an abort and an abort
    # after two files are on disk leaves a half-corpus behind.
    attribution = attribution_text(pins, (*attack_items, *benign_items), identity)

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        frame_id=pins.benign_frame.frame_id,
        build_id=identity,
        files=files_for(
            [(name, text.encode("utf-8"), rows) for name, text, rows in payloads]
        ),
        reports={
            **attack_report.as_run_fields(),
            **benign_report.as_run_fields(),
            **exclusion_report.as_run_fields(),
        },
    )

    directory.mkdir(parents=True, exist_ok=True)
    for name, text, _rows in payloads:
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    (directory / MANIFEST_FILENAME).write_text(
        render_manifest(manifest), encoding="utf-8", newline="\n"
    )
    # Beside the corpus and outside the manifest: `manifest.read_corpus` refuses a recorded file
    # that is not a corpus half, and what guards this one is regeneration in `verify-corpus`.
    (directory / ATTRIBUTION_FILENAME).write_text(
        attribution, encoding="utf-8", newline="\n"
    )

    return manifest, {
        **attack_report.as_run_fields(),
        **benign_report.as_run_fields(),
        **exclusion_report.as_run_fields(),
        "written": {
            "directory": str(directory),
            "files": [entry.as_json_object() for entry in manifest.files],
            "manifest": MANIFEST_FILENAME,
            "attribution": ATTRIBUTION_FILENAME,
        },
    }


def write_corpus(path: Path, items: Iterable[CorpusItem], *, rebuild: bool = False) -> int:
    """Write the corpus, refusing to overwrite unless this was the explicit rebuild. Returns bytes.

    **This is the only call in `src/` or `spikes/` that puts a corpus on disk**, which is AD-1's
    rule rather than a preference, and `tests/corpus/test_build.py` holds it with an AST scan over
    the whole tree. `corpus/attack.py` renders the bytes and does not write them: keeping the
    serialization pure is what lets every ordering and encoding claim be tested offline, and
    keeping the write here is what makes the writer countable.

    UTF-8 with no BOM (`encoding="utf-8"`, never `utf-8-sig`) and `newline="\n"`, so the bytes are
    the same on every platform the project runs on -- and `.gitattributes` marks `*.jsonl` as LF
    text so a checkout cannot reintroduce the difference.
    """
    if path.exists() and not rebuild:
        raise CorpusWriteRefused(
            f"{path} already exists and this is not a rebuild. Rebuilding an existing corpus is "
            f"an explicit subcommand and never a side effect of anything else, because a corpus "
            f"silently replaced mid-run publishes a table computed over two different corpora"
        )
    payload = serialize(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return len(payload.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    """The corpus build's three subcommands. Every one of them touches the network.

    Subcommands rather than flags because AD-1 requires that rebuilding an existing corpus be an
    explicit act. `build-attack` refuses to overwrite; `rebuild-attack` is the same code path with
    the refusal lifted, and nothing else in the project calls the writer.
    """
    import argparse
    import json

    from nbc.errors import EXIT_OK

    parser = argparse.ArgumentParser(
        prog="python -m nbc.corpus.build",
        description="Build-time steps for the corpus. Every subcommand touches the network.",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="directory holding pins.toml and data/ (default: the repository root)",
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    subcommands.add_parser(
        "exclusion-report",
        help=(
            "probe and load every declared exclusion source and print the accounting, without "
            "drawing or writing a corpus"
        ),
    )
    subcommands.add_parser(
        "attack-pool-report",
        help=(
            "read the pinned attack pool and run every gate that needs no exclusion index -- "
            "the declared splits, the declared withdrawals and the contradiction gate -- then "
            "print the accounting, without drawing or writing a corpus"
        ),
    )
    subcommands.add_parser(
        "build-attack",
        help=(
            f"draw the declared attack positives and write data/{ATTACK_CORPUS_FILENAME}; "
            f"refuses to overwrite an existing corpus"
        ),
    )
    subcommands.add_parser(
        "rebuild-attack",
        help=(
            "the same build, over a corpus that already exists. Explicit because a corpus "
            "replaced as a side effect publishes a table computed over two different corpora"
        ),
    )
    subcommands.add_parser(
        "build-corpus",
        help=(
            f"draw both halves and write data/{ATTACK_CORPUS_FILENAME}, "
            f"data/{BENIGN_CORPUS_FILENAME}, data/{MANIFEST_FILENAME} and "
            f"data/{ATTRIBUTION_FILENAME}; refuses to overwrite"
        ),
    )
    subcommands.add_parser(
        "rebuild-corpus",
        help="the same build, over a corpus that already exists",
    )
    subcommands.add_parser(
        "verify-corpus",
        help=(
            f"read the committed corpus through the one guarded door: refuses when the recorded "
            f"frame_id or build_id is not the one this declaration computes, when a corpus "
            f"file's bytes no longer hash to what the manifest records, or when "
            f"{ATTRIBUTION_FILENAME} is missing or is not the file this declaration generates. "
            f"Touches no network"
        ),
    )
    args = parser.parse_args(argv)

    try:
        pins = load_pins(args.root)
        if args.subcommand == "exclusion-report":
            _index, planned, observations = read_exclusion_index(pins)
            report: dict[str, object] = ExclusionReport(
                normalization=NORMALIZATION,
                declaration_digest=declaration_digest(pins),
                rows_in=0,
                rows_removed=0,
                outcomes=outcomes_of(planned, observations, {}),
            ).as_run_fields()
        elif args.subcommand == "attack-pool-report":
            # Everything the full build does to the pool before the exclusion index is fetched,
            # and nothing after it. It exists so CI can assert against the **real artifact** that
            # the declarations still describe it: the licence gate is a pure read of `pins.toml`
            # and needs no network, but "the withdrawn rows are the rows the pool carries" is a
            # claim about somebody else's dataset that only the dataset can settle. Reaching it
            # through `build-attack` would cost the largest download this project makes and would
            # leave a corpus on disk; this costs one parquet pair and writes nothing.
            refuse_unlicensed_redistribution(pins)
            dataset = pins.attack_datasets[0]
            # `read_attack_pool` already applied the declared withdrawals and aborted if they did
            # not describe the pool exactly, so reaching this line is most of what this subcommand
            # exists to check.
            pool, observed_splits, withdrawn = read_attack_pool(dataset)
            split_problems = verify_splits(dataset.splits, observed_splits)
            if split_problems:
                raise AttackDrawUnsatisfiable(*split_problems)
            contradiction_problems = contradictions(pool)
            if contradiction_problems:
                raise LabelContradiction(*contradiction_problems)
            withdrawn_rows = len(withdrawn)
            report = {
                "attack_pool": {
                    "repository": dataset.repository,
                    "revision": dataset.revision,
                    "observed_splits": list(observed_splits),
                    # The artifact's own count, recovered from the two halves the one door
                    # returns rather than from a second read of the dataset.
                    "rows_read": len(pool) + withdrawn_rows,
                    "withdrawn_rows": withdrawn_rows,
                    "rows_used": len(pool),
                    "unique_positives": len(
                        {
                            row.text
                            for row in pool
                            if row.text and row.label == dataset.attack_label
                        }
                    ),
                    # Published so the report says which decisions it applied, not merely that it
                    # applied some: a count of five with no names beside it is a number a reader
                    # has to go and reconstruct from another file.
                    "withdrawn": [entry.as_run_fields() for entry in dataset.withdrawn],
                    "licence": dataset.licence.as_run_fields(),
                }
            }
        elif args.subcommand == "verify-corpus":
            manifest, items = read_corpus(pins, args.root)
            # The credits are generated, so verifying them is regenerating them. A hash would
            # catch an edit; regeneration also catches a file that was never right.
            expected = attribution_text(pins, items, manifest.build_id)
            path = corpus_directory(args.root) / ATTRIBUTION_FILENAME
            committed: str | None = None
            if path.is_file():
                try:
                    committed = path.read_text(encoding="utf-8")
                except (OSError, ValueError) as error:
                    # `ValueError` covers `UnicodeDecodeError`, which is what a corrupted or
                    # re-encoded credits file raises and which is not an `OSError`.
                    raise RedistributionRefused(
                        f"{ATTRIBUTION_FILENAME} is at {path} and cannot be read as UTF-8 text: "
                        f"{error}. It is generated; rebuild the corpus rather than repairing it"
                    ) from None
            drift = attribution_problems(committed, expected)
            if drift:
                raise RedistributionRefused(*drift)
            report = {
                "verified_corpus": {
                    "frame_id": manifest.frame_id,
                    "build_id": manifest.build_id,
                    "files": [entry.as_json_object() for entry in manifest.files],
                    "items_read": len(items),
                    "attribution": ATTRIBUTION_FILENAME,
                }
            }
        elif args.subcommand in ("build-corpus", "rebuild-corpus"):
            _manifest, report = build_corpus(
                pins, root=args.root, rebuild=args.subcommand == "rebuild-corpus"
            )
        else:
            draw_report, exclusion_report, path, written = build_attack_corpus(
                pins, root=args.root, rebuild=args.subcommand == "rebuild-attack"
            )
            report = {
                **draw_report.as_run_fields(),
                **exclusion_report.as_run_fields(),
                "written": {"path": str(path), "bytes": written},
            }
    except NbcError as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
