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

    python -m nbc.corpus.build --exclusion-report

probes and loads every declared exclusion source and prints the accounting as JSON, with no corpus
to filter -- the corpus arrives in story 3.2. It touches the network, so it is not part of the
offline unit suite.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Final, Iterator

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
from nbc.pins import HTTP_OK, ExclusionSource, Pins, load_pins

__all__ = [
    "HTTP_TIMEOUT_SECONDS",
    "iter_exclusion_texts",
    "main",
    "observe_exclusion_sources",
    "probe",
    "read_exclusion_index",
]

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


def main(argv: list[str] | None = None) -> int:
    """`python -m nbc.corpus.build --exclusion-report` -- the accounting, over the network."""
    import argparse
    import json

    from nbc.errors import EXIT_OK

    parser = argparse.ArgumentParser(
        prog="python -m nbc.corpus.build",
        description="Build-time steps for the corpus. Touches the network.",
    )
    parser.add_argument(
        "--exclusion-report",
        action="store_true",
        help=(
            "probe and load every declared exclusion source and print the accounting; there is "
            "no corpus to filter yet, so every removal count is zero by construction"
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="directory holding pins.toml (default: the repository root)",
    )
    args = parser.parse_args(argv)

    if not args.exclusion_report:
        parser.error("nothing to do; --exclusion-report is the only step this story ships")

    try:
        pins = load_pins(args.root)
        _index, planned, observations = read_exclusion_index(pins)
        report = ExclusionReport(
            normalization=NORMALIZATION,
            declaration_digest=declaration_digest(pins),
            rows_in=0,
            rows_removed=0,
            outcomes=outcomes_of(planned, observations, {}),
        )
    except NbcError as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report.as_run_fields(), sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
