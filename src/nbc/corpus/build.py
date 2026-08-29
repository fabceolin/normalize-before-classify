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

Rebuilding an existing corpus is an **explicit subcommand and never a side effect**: `build-attack`
aborts with `CorpusWriteRefused` rather than overwriting, so no run can replace the corpus a table
was computed over while looking like it did something else. Every subcommand touches the network,
so none of them is part of the offline unit suite.

**This module is the only writer of `data/*.jsonl`** (AD-1). An AST scan over `src/` and `spikes/`
in `tests/corpus/test_build.py` holds that: it collects every write primitive in the tree and
refuses any file outside a declared allow-list, and the scan's own predicate is tested against a
synthetic offender so it cannot pass by failing to look.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Iterable, Iterator

from nbc.corpus.attack import (
    AttackDrawReport,
    AttackDrawUnsatisfiable,
    PoolRow,
    draw_attack_items,
    serialize,
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
    normalized_texts,
    outcomes_of,
    plan,
    verify_observations,
)
from nbc.errors import NbcError, exit_code_for
from nbc.schema import CorpusItem
from nbc.pins import HTTP_OK, AttackDataset as AttackDatasetPin, ExclusionSource, Pins, load_pins

__all__ = [
    "ATTACK_CORPUS_FILENAME",
    "CorpusWriteRefused",
    "DATA_DIRNAME",
    "HTTP_TIMEOUT_SECONDS",
    "build_attack_corpus",
    "corpus_directory",
    "iter_exclusion_texts",
    "main",
    "observe_exclusion_sources",
    "probe",
    "read_attack_pool",
    "read_exclusion_index",
    "write_corpus",
]

class CorpusWriteRefused(NbcError, exit_code=18):
    """A corpus file already exists and this call was not the explicit rebuild.

    Code 18. AD-1 requires that rebuilding an existing corpus be an explicit subcommand and never
    a side effect of anything else, and a refusal that has its own exit code is what lets a caller
    tell "you meant `rebuild-attack`" apart from "the data is contradictory".
    """


ATTACK_CORPUS_FILENAME: Final[str] = "attack.jsonl"
"""The attack corpus' filename under `data/`. Named once, here."""

DATA_DIRNAME: Final[str] = "data"
"""Where the committed corpus lives, relative to the repository root. Named once, here."""


def corpus_directory(root: str | None = None) -> Path:
    """The directory `data/*.jsonl` is written to.

    Takes a root so a test can build into `tmp_path`: CI refuses a dirty tree, and a builder that
    could only ever write into the checkout would make every end-to-end test of it a violation.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    return base / DATA_DIRNAME

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
) -> tuple[tuple[PoolRow, ...], tuple[str, ...]]:
    """Every row of the pinned attack dataset, over every split it ships, at the pinned revision.

    Returns the rows and the splits **as observed**, never as declared: the caller compares the
    two, in both directions. Handing back the declared list would make that comparison a check of
    a value against itself, which is the pattern this project keeps finding in its own history.

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
    return tuple(rows), tuple(observed)


def build_attack_corpus(
    pins: Pins, *, root: str | None = None, rebuild: bool = False
) -> tuple[AttackDrawReport, ExclusionReport, Path, int]:
    """The whole attack build: pool, exclusion index, gates, draw, write.

    The exclusion index is read **before** the pool is drawn against it and after its own
    verification, so a corpus is never filtered against a set the pins do not describe. Both
    reports come back so the caller publishes the accounting for what it removed as well as for
    what it kept.
    """
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

    rows, observed_splits = read_attack_pool(dataset)

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

    items, draw_report, matches = draw_attack_items(
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

    path = corpus_directory(root) / ATTACK_CORPUS_FILENAME
    written = write_corpus(path, items, rebuild=rebuild)
    return draw_report, exclusion_report, path, written



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
