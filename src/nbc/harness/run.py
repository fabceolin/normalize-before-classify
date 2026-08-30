"""The scoring pass: open the pinned baselines, walk one shard, and merge the shards into one file.

Everything that touches a model, a file or the outside world is here, and everything that decides
anything is in `harness/score.py`. That is the same split `corpus/build.py` and `corpus/attack.py`
already use, and it is what lets the shard algebra -- membership, coverage, agreement, the merged
bytes -- be covered by a suite with no model in the process.

**The corpus is read through the one guarded door.** `manifest.read_corpus` verifies the recorded
frame id, recomputes the build id and hashes every corpus file before it hands back a row. This
module names no corpus file and locates none; it asks for rows and is given them, which is what
`tests/corpus/test_manifest.py`'s scan over `src/` enforces. Scoring a corpus nobody verified would
publish a table computed over rows that are not the committed ones, and by the time the numbers
existed there would be nothing left to compare them against.

**Two subcommands, not a flag.** `score-shard` runs inference for hours; `merge` verifies and
writes. `corpus/build.py` splits `build-attack` from `rebuild-attack` for the same reason: an
expensive or destructive act reached as a side effect of another one is an act nobody chose.

**`onnxruntime` is imported inside a function, after the platform preflight.** Importing it at
module scope would make `python -m nbc.harness.run --help` build against a runtime the preflight
has not cleared, and would check a floor the import already crashed through -- story 1.2's whole
step-0 argument. `merge` pays that import too, and deliberately: the execution path it compares
the shard headers against is declared in `baselines/onnx_adapter.py`, and a second spelling of
those constants here is how the published path and the checked path would come to differ.

**One document per `score` call, appended and flushed as it is produced.** The pass is measured at
~85 h for the full matrix, so a kill at hour five must cost one item and not five hours. Batch
composition is not a free parameter of a score -- `tests/baselines/test_onnx_adapter.py` measures
that over both pinned graphs at zero difference -- so calling one document at a time buys
durability at no cost to the numbers.

Wall-clock is deliberately not recorded anywhere in this module. Story 4-5 owns cost and owns it in
a dedicated pass, because a sharded run has many processes contending for one machine's cache and
memory bandwidth, and a latency measured here would describe the contention.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

from nbc import platform
from nbc.canon.pipeline import canonicalize, default_context
from nbc.corpus.manifest import read_corpus
from nbc.errors import EXIT_OK, NbcError, exit_code_for
from nbc.harness.score import (
    DeclaredPath,
    ExecutionPath,
    ScoreSetIncomplete,
    ShardFile,
    ShardHeader,
    SHARD_SCHEMA_VERSION,
    expected_keys,
    key_of,
    merge,
    parse_shard,
    render_record,
    render_shard,
    score_key,
    serialize,
    shard_of,
)
from nbc.corpus.manifest import corpus_presence
from nbc.harness.aggregate import cells as aggregate_cells
from nbc.harness.aggregate import read_scores, read_traces
from nbc.harness.results import (
    PROFILE_FULL,
    PROFILE_SMOKE,
    PROFILES,
    RESULT_KEYS,
    STEP_AGGREGATE,
    STEP_BUILD,
    STEP_MEASURE,
    STEP_PREFLIGHT,
    STEP_TIME,
    STEP_VERIFY,
    ResultsFile,
    ResultsIncomplete,
    RunBlock,
    completeness_problems,
    refuse_an_incomplete_table,
    render_results,
    smoke_sample,
)
from nbc.harness.summary import (
    REJECTED_JUSTIFICATIONS,
    REJECTED_SUMMARIES,
    SUMMARY_CHOICE,
    findings,
)
from nbc.harness.timing import run_timing_pass
from nbc.harness.verdict import refuse_an_unevaluable_run, verdicts
from nbc.pins import Pins, load_pins
from nbc.report.caveats import verify_caveats_file
from nbc.schema import INTERVAL_METHODS
from nbc.schema import CANONICAL, CONDITIONS, CorpusItem, ItemScore, Score

__all__ = [
    "RESULTS_DIRNAME",
    "SCORES_FILENAME",
    "ScoringBaseline",
    "declared_path",
    "main",
    "merge_shards",
    "open_baselines",
    "timing_pass",
    "RESULTS_FILENAME",
    "TRACES_FILENAME",
    "full_run",
    "results_directory",
    "traces_path",
    "scores_path",
    "score_shard",
    "shard_path",
]


RESULTS_DIRNAME: Final[str] = "results"
"""Where the run's output lives, beside the declared `results/results.json`. Named once, here."""

RESULTS_FILENAME: Final[str] = "results.json"
"""The published file. Named once, here, beside the scores it is computed from."""

SCORES_FILENAME: Final[str] = "scores.jsonl"
"""The merged file every later story in this epic reads. One name, in one place."""

_SHARD_STEM: Final[str] = "scores"
"""The shard files are `scores-<n>-<i>.jsonl`, which no glob for the merged name can match.

Deliberate: a merge that swept up its own previous output would report every key twice and call
the pass a non-partition, which is a confusing way to say "there is a stale file here".
"""


class ScoringBaseline(Protocol):
    """What the shard walk needs from a baseline, and the whole of it.

    A `Protocol` rather than `OnnxBaseline` because the offline suite has to be able to walk a
    shard with no model in the process, and because naming the two methods states exactly how much
    of the adapter this module depends on. `as_run_fields` is the existing shape for recording an
    execution path (`OnnxBaseline.as_run_fields`), and the shard header extends it rather than
    restating it -- so a field the adapter starts reporting differently is reported differently
    here, instead of being restated correctly in one place and wrongly in the other.
    """

    def score(self, texts: Sequence[str]) -> list[Score]: ...

    def as_run_fields(self) -> dict[str, object]: ...


BaselineOpener = Callable[[Pins], Mapping[str, ScoringBaseline]]
"""How `score_shard` gets its columns. The seam the offline tests replace."""


def results_directory(root: str | Path | None = None) -> Path:
    """The directory the shard files and the merged scores file live in.

    Takes a root for the reason `corpus_directory` does: CI refuses a dirty tree, and a writer that
    could only ever write into the checkout would make every end-to-end test of it a violation.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    return base / RESULTS_DIRNAME


def scores_path(root: str | Path | None = None) -> Path:
    return results_directory(root) / SCORES_FILENAME


def shard_path(shards: int, shard: int, root: str | Path | None = None) -> Path:
    """Where shard `shard` of `shards` writes. The shard count is in the name on purpose.

    A run at `--shards 4` and a run at `--shards 7` produce different partitions of the same
    corpus, and a file named only by its index would let the second overwrite half of the first
    and leave the rest -- a set of files that is neither partition and passes no check until the
    coverage report, which would then blame the corpus.
    """
    return results_directory(root) / f"{_SHARD_STEM}-{shards}-{shard}.jsonl"


def open_baselines(pins: Pins, *, cache_root: Path | None = None) -> dict[str, ScoringBaseline]:
    """Every pinned baseline, opened against the files its pin names and no others.

    `onnxruntime` arrives with this call and not with the module, after `platform.preflight` has
    run: a preflight that fires after the runtime is imported is checking a floor the import
    already crashed through.
    """
    from nbc.baselines.onnx_adapter import open_baseline
    from nbc.baselines.tokenization import open_windower

    opened: dict[str, ScoringBaseline] = {}
    for baseline in pins.baselines:
        windower = open_windower(baseline, cache_root=cache_root)
        opened[baseline.key] = open_baseline(baseline, windower, cache_root=cache_root)
    return opened


def declared_path(pins: Pins) -> DeclaredPath:
    """The execution path this repository publishes under, gathered from where it is declared.

    The constants come from `baselines/onnx_adapter.py` and the revisions from `pins.toml`; the
    `ExecutionPath` records they are compared against come out of files other processes wrote.
    Two sides, two sources. Building this from the same values a shard header was built from in
    this process would produce a check that agrees with itself whatever the shard files say.
    """
    from nbc.baselines.onnx_adapter import BATCH_SIZE, INTRA_OP_NUM_THREADS, PROVIDERS

    return DeclaredPath(
        providers=tuple(PROVIDERS),
        intra_op_num_threads=INTRA_OP_NUM_THREADS,
        batch_size=BATCH_SIZE,
        revisions={baseline.key: baseline.revision for baseline in pins.baselines},
    )


def _execution_paths(
    pins: Pins, baselines: Mapping[str, ScoringBaseline]
) -> tuple[ExecutionPath, ...]:
    """What each opened baseline reports about how it will run, plus the revision it was pinned at.

    `providers` is the adapter's **observed** value -- what the runtime made active -- rather than
    the constant that asked for it. Those are two different facts, and the gap between them is the
    one `path_problems` exists to catch across processes.
    """
    revisions = {baseline.key: baseline.revision for baseline in pins.baselines}
    paths: list[ExecutionPath] = []
    for key in sorted(baselines):
        fields = baselines[key].as_run_fields()
        recorded_key = str(fields["key"])
        if recorded_key != key:
            raise ScoreSetIncomplete(
                f"the baseline opened for {key!r} reports itself as {recorded_key!r}; a column "
                f"scored by another baseline's graph is a column nobody declared"
            )
        revision = revisions.get(key)
        if revision is None:
            raise ScoreSetIncomplete(
                f"a baseline was opened for {key!r}, which this declaration does not pin"
            )
        paths.append(
            ExecutionPath(
                baseline_key=key,
                revision=revision,
                providers=tuple(str(name) for name in fields["providers"]),  # type: ignore[union-attr]
                intra_op_num_threads=int(fields["intra_op_num_threads"]),  # type: ignore[arg-type]
                batch_size=int(fields["batch_size"]),  # type: ignore[arg-type]
            )
        )
    return tuple(paths)


def _demanded(
    items: Sequence[CorpusItem],
    baseline_keys: Sequence[str],
    *,
    shards: int,
    shard: int,
) -> dict[str, tuple[CorpusItem, str, str]]:
    """The keys **this shard** owes, mapped to the item, baseline and condition each stands for.

    `expected_keys` is the authority on *which* keys the pass demands and on refusing a corpus that
    cannot be scored once; this is the lookup that turns one of them back into work to do, and it
    is filtered here rather than after the fact so a shard's working set is its own share of the
    pass and not the whole of it.
    """
    return {
        key: (item, baseline_key, condition)
        for item in items
        for baseline_key in baseline_keys
        for condition in CONDITIONS
        if shard_of(key := score_key(item.id, baseline_key, condition), shards) == shard
    }


def _read_existing(path: Path, *, repair: bool) -> ShardFile | None:
    """The shard file already on disk, or `None` if there is not one yet.

    With `repair`, a trailing partial line is removed from the file before it is parsed. That is
    not a softer rule than the merge's -- it is the only safe one here, because the next thing this
    function's caller does is **append**, and appending onto a half-written line concatenates two
    records into one that parses as neither. The record it drops was never written, so the item it
    belongs to is simply re-scored; at merge time nobody is going to re-score anything, so there
    the same input is an abort that names the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except ValueError as error:
        # `UnicodeDecodeError` is a `ValueError` and not an `OSError`: a shard file written by a
        # process with a different encoding escapes as an unclassified crash if only OSError is
        # caught, which is the sibling this project has missed before.
        raise ScoreSetIncomplete(f"{path} is not valid UTF-8: {error}") from None
    except OSError as error:
        raise ScoreSetIncomplete(f"{path} could not be read: {error}") from None

    if repair and text and not text.endswith("\n"):
        kept = text[: text.rfind("\n") + 1]
        path.write_text(kept, encoding="utf-8", newline="\n")
        text = kept
    if not text:
        return None
    return parse_shard(path.name, text)


def score_shard(
    pins: Pins,
    *,
    shards: int,
    shard: int,
    root: str | Path | None = None,
    opener: BaselineOpener | None = None,
    profile: str = PROFILE_FULL,
) -> dict[str, object]:
    """Score every key this shard owns that is not already scored, and report what it did.

    Resuming is not a second opinion: a key already in the shard file is not re-scored, and the
    header the file carries is compared against the one this process would write before a single
    record is appended. A shard resumed on a different machine, against a rebuilt corpus, or under
    a different split therefore aborts instead of producing one file with two provenances in it.
    """
    _check_split(shards, shard)
    manifest, all_items = read_corpus(pins, root)
    items = _items_for_profile(all_items, pins, profile)
    baseline_keys = [baseline.key for baseline in pins.baselines]
    demanded = _demanded(items, baseline_keys, shards=shards, shard=shard)
    mine = [key for key in expected_keys(items, baseline_keys) if shard_of(key, shards) == shard]

    path = shard_path(shards, shard, root)
    existing = _read_existing(path, repair=True)
    already: dict[str, ItemScore] = {}
    if existing is not None:
        _refuse_a_foreign_shard_file(
            existing, shards=shards, shard=shard, build_id=manifest.build_id
        )
        already = {key_of(score): score for score in existing.scores}
        stray = sorted(set(already) - set(mine))
        if stray:
            raise ScoreSetIncomplete(
                f"{path.name} carries {len(stray)} record(s) this shard does not own, starting "
                f"with {stray[:3]}; appending to it would publish a key twice"
            )

    todo = [key for key in mine if key not in already]
    if not todo and existing is not None:
        # Nothing to do and the file that says so is already on disk. The models are not opened:
        # the answer does not depend on them, and a resumed pass whose shards are mostly complete
        # would otherwise pay a minute per shard to learn it has nothing to do.
        return _resumed(shards, shard, manifest.build_id, mine, already, path)

    # A shard that owns no key at all still writes its file. Otherwise the merge cannot tell a
    # shard that ran and had nothing to do from a shard nobody started, and would report the one
    # as the other -- which at a shard count above the key count is not hypothetical.
    baselines = (opener or open_baselines)(pins)
    unopened = sorted(set(baseline_keys) - set(baselines))
    if unopened:
        raise ScoreSetIncomplete(
            f"no baseline was opened for pinned column(s) {unopened}; the shard would write a "
            f"file that is short by a whole column and the merge would blame the corpus"
        )
    header = ShardHeader(
        schema_version=SHARD_SCHEMA_VERSION,
        shards=shards,
        shard=shard,
        build_id=manifest.build_id,
        paths=_execution_paths(pins, baselines),
    )
    if existing is not None and existing.header != header:
        raise ScoreSetIncomplete(
            f"{path.name} was written under [{_describe_paths(existing.header)}] and this process "
            f"would append under [{_describe_paths(header)}]; one file with two execution paths in "
            f"it is a file no reader can attribute a number in"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if existing is None:
        path.write_text(render_shard(header, ()), encoding="utf-8", newline="\n")
    if not todo:
        return _resumed(shards, shard, manifest.build_id, mine, already, path)

    # The trace is off: `canonicalize` returns the same text, the same `ceiling_hit` and the same
    # `max_depth_reached` either way -- asserted over a battery in `tests/canon/test_pipeline.py`
    # -- and the trace is per-edit memory this pass has no consumer for.
    context = default_context(trace_enabled=False)

    # Grouped by item, so the canonical form is computed once per item and held for exactly as
    # long as that item's columns are being scored. Reusing it across baselines is the point --
    # the canonical text is a property of the text and not of the column -- but a cache keyed on
    # the whole shard would hold a second copy of this shard's share of a 130 MB corpus, which is
    # the unbounded structure the grouping avoids rather than the reuse.
    by_item: dict[str, list[str]] = {}
    for key in todo:
        by_item.setdefault(demanded[key][0].id, []).append(key)

    scored = 0
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for keys in by_item.values():
            canonical: tuple[str, int, bool] | None = None
            for key in keys:
                item, baseline_key, condition = demanded[key]
                if condition == CANONICAL:
                    if canonical is None:
                        result = canonicalize(item.text, context)
                        canonical = (
                            result.text,
                            result.max_depth_reached,
                            result.ceiling_hit,
                        )
                    text, depth, ceiling_hit = canonical
                else:
                    text, depth, ceiling_hit = item.text, None, None

                measured = baselines[baseline_key].score([text])
                if len(measured) != 1:
                    raise ScoreSetIncomplete(
                        f"baseline {baseline_key!r} returned {len(measured)} scores for one "
                        f"document"
                    )
                handle.write(
                    render_record(
                        ItemScore(
                            item_id=item.id,
                            family=item.family,
                            benign_class=item.benign_class,
                            label=item.label,
                            baseline_key=baseline_key,
                            condition=condition,
                            p_injection=measured[0].p_injection,
                            n_windows=measured[0].n_windows,
                            max_depth_reached=depth,
                            ceiling_hit=ceiling_hit,
                        )
                    )
                )
                # Flushed per record rather than per file. The pass is measured in tens of hours
                # and a flush costs nothing beside one 512-token window of inference.
                handle.flush()
                scored += 1

    return {
        "scored_shard": {
            "shards": shards,
            "shard": shard,
            "build_id": manifest.build_id,
            "keys_owned": len(mine),
            "keys_scored": scored,
            "keys_resumed": len(already),
            "path": str(path),
            "resumed": bool(already),
        }
    }


def merge_shards(
    pins: Pins, *, shards: int, root: str | Path | None = None, profile: str = PROFILE_FULL
) -> dict[str, object]:
    """Verify the shard files add up to one pass over the corpus and write the merged scores file.

    Nothing is written unless every check passes. A partial scores file is worse than none: it is
    the shape a downstream story reads without complaint.
    """
    _check_split(shards, 0)
    manifest, all_items = read_corpus(pins, root)
    items = _items_for_profile(all_items, pins, profile)
    baseline_keys = [baseline.key for baseline in pins.baselines]
    expected = expected_keys(items, baseline_keys)

    files: list[ShardFile] = []
    missing: list[str] = []
    for index in range(shards):
        path = shard_path(shards, index, root)
        found = _read_existing(path, repair=False)
        if found is None:
            missing.append(path.name)
            continue
        files.append(found)
    if missing:
        raise ScoreSetIncomplete(
            f"the split declares {shards} shard(s) and {missing} are not in "
            f"{results_directory(root)}; those shards have not run"
        )

    scores = merge(
        files,
        expected=expected,
        build_id=manifest.build_id,
        declared=declared_path(pins),
        shards=shards,
    )

    target = scores_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize(scores)
    target.write_text(payload, encoding="utf-8", newline="\n")
    return {
        "merged_scores": {
            "shards": shards,
            "build_id": manifest.build_id,
            "baselines": sorted(baseline_keys),
            "items": len(items),
            "records": len(scores),
            "path": str(target),
            "bytes": len(payload.encode("utf-8")),
        }
    }


def _resumed(
    shards: int,
    shard: int,
    build_id: str,
    mine: Sequence[str],
    already: Mapping[str, ItemScore],
    path: Path,
) -> dict[str, object]:
    """The report for a shard this run scored nothing for, whether it was complete or empty."""
    return {
        "scored_shard": {
            "shards": shards,
            "shard": shard,
            "build_id": build_id,
            "keys_owned": len(mine),
            "keys_scored": 0,
            "keys_resumed": len(already),
            "path": str(path),
            # A shard that owned nothing was not resumed from anything; the distinction matters to
            # an operator reading a hundred of these to find out which machines still owe work.
            "resumed": bool(already),
        }
    }


def _check_split(shards: int, shard: int) -> None:
    if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
        raise ScoreSetIncomplete(f"--shards must be a positive integer, got {shards!r}")
    if isinstance(shard, bool) or not isinstance(shard, int) or not 0 <= shard < shards:
        raise ScoreSetIncomplete(
            f"--shard must name one of the {shards} shard(s), 0..{shards - 1}, got {shard!r}"
        )


def _refuse_a_foreign_shard_file(
    existing: ShardFile, *, shards: int, shard: int, build_id: str
) -> None:
    """A shard file this process may append to declares this split and this corpus, or none.

    Checked before the models are opened, because opening them costs a minute and the answer does
    not depend on them.
    """
    problems: list[str] = []
    if existing.header.shards != shards or existing.header.shard != shard:
        problems.append(
            f"{existing.name} declares shard {existing.header.shard} of "
            f"{existing.header.shards} and this run is shard {shard} of {shards}"
        )
    if existing.header.build_id != build_id:
        problems.append(
            f"{existing.name} was scored over build_id {existing.header.build_id!r} and the "
            f"corpus on disk is {build_id!r}; the corpus was rebuilt after that shard ran, so "
            f"resuming would put rows from two corpora in one file"
        )
    if problems:
        raise ScoreSetIncomplete(*problems)


def _describe_paths(header: ShardHeader) -> str:
    return "; ".join(path.describe() for path in sorted(header.paths, key=lambda p: p.baseline_key))


TRACES_FILENAME: Final[str] = "traces.jsonl"
"""One object per item, keyed by item id, and **not committed**.

Its consumer is a person debugging one document; what the table needs is the per-stage edit counts,
which are `Count` cells in the results file. `.gitignore` names it, and the README says where it
lives and how to regenerate it, so the absence is documented rather than discovered.
"""


def traces_path(root: str | Path | None = None) -> Path:
    return results_directory(root) / TRACES_FILENAME


def full_run(
    pins: Pins,
    *,
    root: str | Path | None = None,
    shards: int = 1,
    profile: str = PROFILE_FULL,
) -> dict[str, object]:
    """The six steps, in the declared order, ending at `results/results.json`.

    The order is `results.STEPS` and each boundary was paid for. The preflight runs before the pins
    are read and therefore before `onnxruntime` is imported -- a preflight that fires after the
    runtime is imported is checking a floor the import already crashed through. The pins and the
    caveats section are verified before any inference, because eighty-five hours of scoring that
    ends at a missing caveats section is eighty-five hours spent to learn something a file read
    would have said. The corpus is built only when it is **wholly** absent, because a partial one
    is the state where a rebuild writes half-new rows against a manifest describing the old ones.

    The trace file is written from the timing pass rather than the scoring pass, and that is not
    where it looks like it belongs. Story 4.2 turned tracing off in the scoring pass because the
    edits were per-item memory with no consumer, and the scoring pass is **sharded**, so an item
    scored under two baselines lives in two shards and would write its trace twice. The timing pass
    canonicalizes every item exactly once with tracing on -- it refuses a context that has it off --
    so the trace exists there, once per item, and is streamed out through a callback rather than
    accumulated. Verified while writing this: tracing changes only whether edits are recorded, not
    the canonical text, the depth, or the ceiling flag, so no published score moves either way.
    """
    if profile not in PROFILES:
        raise ResultsIncomplete(
            f"profile must be one of {PROFILES}, got {profile!r}; a run whose profile is not one of "
            f"the two cannot say which run produced its table"
        )
    if profile == PROFILE_SMOKE and _is_the_published_root(root):
        raise ResultsIncomplete(
            "a smoke run may not write into the repository's own results directory. Its table is "
            "structurally identical to the published one with a small n as the only tell, so "
            "overwriting the published file would replace the headline table with a sample of it. "
            "Pass --root pointing somewhere else"
        )

    started = time.perf_counter_ns()
    steps: list[str] = []

    # (0) and (1). `platform.preflight` before `load_pins`, and both before anything opens a model.
    platform.preflight()
    steps.append(STEP_PREFLIGHT)
    # The pins arrive already loaded, because `main` loads them before dispatching and reloading
    # them here would have made the parameter decorative -- which it was, in the first version of
    # this function, so a caller handing in one set of pins was silently measured against another.
    caveats = verify_caveats_file()
    steps.append(STEP_VERIFY)

    # (2). Wholly absent, or wholly present. Anything between aborts.
    presence = corpus_presence(root)
    if any(presence.values()) and not all(presence.values()):
        missing = sorted(name for name, there in presence.items() if not there)
        raise ResultsIncomplete(
            f"the corpus is partially present: {missing} are absent while the others are not. A "
            f"rebuild here would write half-new rows against a manifest that describes the old "
            f"ones, so the run stops rather than choosing which half to trust"
        )
    if not any(presence.values()):
        raise ResultsIncomplete(
            "no corpus is present. Build it with `python -m nbc.corpus.build build-corpus` before "
            "measuring; this command does not build one for you, because a build is a decision "
            "about which rows exist and a measurement is not"
        )
    steps.append(STEP_BUILD)

    # (3). The scoring pass, through the same function the sharded CLI calls.
    merge_shards(pins, shards=shards, root=root, profile=profile)
    steps.append(STEP_MEASURE)

    # (4). The dedicated cost pass, which also streams the trace file out.
    manifest, all_items = read_corpus(pins, root)
    items = _items_for_profile(all_items, pins, profile)
    context = default_context()
    baselines = open_baselines(pins)
    traces_at = traces_path(root)
    traces_at.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_at, "w", encoding="utf-8", newline="\n") as handle:
        report = run_timing_pass(
            items,
            context,
            baselines,
            expected_keys=[baseline.key for baseline in pins.baselines],
            on_trace=lambda item, result: handle.write(
                json.dumps(
                    {
                        "item_id": item.id,
                        "stages": [edit.stage for edit in result.edits],
                        "edits": [
                            {
                                "stage": edit.stage,
                                "span": list(edit.span),
                                "before": edit.before,
                                "after": edit.after,
                                "depth": edit.depth,
                            }
                            for edit in result.edits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            ),
        )
    steps.append(STEP_TIME)

    # (5). Aggregate, evaluate, and write. Nothing here computes a cell.
    scores = read_scores(scores_path(root))
    traces = read_traces(traces_at)
    produced = aggregate_cells(scores, pins, traces)
    limits = findings(produced)
    decided = verdicts(produced, report, pins.benign_frame.confirmatory_cell)

    refuse_an_incomplete_table(
        completeness_problems(produced, _required_window_policies(pins))
    )
    refuse_an_unevaluable_run(decided)

    results = ResultsFile(
        run=RunBlock(
            {
                "build_id": manifest.build_id,
                "corpus_files": [
                    {"name": entry.name, "sha256": entry.sha256, "rows": entry.rows}
                    for entry in manifest.files
                ],
                # `dataclasses.asdict` rather than a method, because `DeclaredPath` has none and
                # the alternative found while writing this -- a `hasattr` fallback to `str()` --
                # would have put a Python repr in a published file the first time the type changed.
                "declared_path": dataclasses.asdict(declared_path(pins)),
                "caveats": caveats.as_run_fields(),
                "confirmatory_cell": pins.benign_frame.confirmatory_cell.as_run_fields(),
                "timing": report.as_json_object(),
                "summary": {
                    "choice": SUMMARY_CHOICE,
                    "rejected": dict(REJECTED_SUMMARIES),
                    "rejected_justifications": dict(REJECTED_JUSTIFICATIONS),
                    "findings": [limit.as_json_object() for limit in limits],
                },
                "interval_methods": list(INTERVAL_METHODS),
                "profile": profile,
                "profile_items_per_cell": (
                    None if profile == PROFILE_FULL else pins.smoke.items_per_cell
                ),
                "profile_items": len(items),
                "steps": steps + [STEP_AGGREGATE],
                "total_wall_ns": time.perf_counter_ns() - started,
            }
        ),
        cells=produced,
        verdict=decided,
    )
    path = results_directory(root) / RESULTS_FILENAME
    path.write_text(render_results(results), encoding="utf-8")
    steps.append(STEP_AGGREGATE)

    return {
        "results": str(path),
        "cells": len(produced),
        "verdicts": [v.outcome for v in decided],
        "findings": len(limits),
        "steps": steps,
    }


def _items_for_profile(items: Sequence[CorpusItem], pins: Pins, profile: str) -> tuple[CorpusItem, ...]:
    """The corpus a run at `profile` measures over.

    In one place because THREE passes read the corpus -- the shard walk, the merge and the full run
    -- and a profile applied to some of them is worse than a profile applied to none: the merge
    would compute a demand set over the whole corpus and refuse a shard file that scored the sample,
    naming a coverage failure that is really a disagreement about what the corpus is.

    Found by writing the tag workflow: `--profile` reached `full_run` and not `score_shard`, so the
    smoke job would have scored the entire corpus -- about eighty-five hours on a runner -- and then
    failed the merge.
    """
    if profile not in PROFILES:
        raise ResultsIncomplete(
            f"profile must be one of {PROFILES}, got {profile!r}; a run whose profile is not one of "
            f"the two cannot say which run produced its table"
        )
    if profile == PROFILE_FULL:
        return tuple(items)
    return tuple(smoke_sample(items, pins.smoke.items_per_cell))  # type: ignore[arg-type]


def _is_the_published_root(root: str | Path | None) -> bool:
    """Whether this run would write into the repository's own results directory.

    `None` means the repository root, which is what every published run uses. A path that resolves
    to the same directory is the same thing spelled differently, so the comparison is on the
    resolved directory rather than on the argument.
    """
    if root is None:
        return True
    return results_directory(root).resolve() == results_directory(None).resolve()


def _required_window_policies(pins: Pins) -> dict[str, list[str]]:
    """Which window policies each baseline's table must cover.

    Read off the pins rather than assumed, so a baseline that starts declaring a second policy
    starts demanding a second column on the same day.

    **Vacuous today, and said rather than left to be discovered.** `tokenization.WINDOW_POLICIES`
    has exactly one member and every pinned baseline declares it, so this returns one policy per
    baseline and the completeness assertion's window-policy branch cannot fire on a real run. The
    branch stays because the axis is protected and the requirement is real the day a publisher
    protocol is declared, and `tests/harness/test_results.py` fires it on a synthetic two-policy
    input so it is a gate somebody has seen work rather than a branch nobody has run.
    """
    required: dict[str, list[str]] = {}
    for baseline in pins.baselines:
        required[baseline.key] = sorted({baseline.window_policy})
    return required


def timing_pass(pins: Pins, *, root: str | Path | None = None) -> dict[str, object]:
    """Read the corpus, build the one context, open the baselines, and hand all three to the pass.

    This function exists because `harness/timing.py` may do none of those things. It may not build
    a `CanonContext` -- AD-6 gives one run one context and the tripwire in
    `tests/canon/test_recursion.py` names that module as refused -- and it may not name a corpus
    file, which AD-1's locator scan enforces. Both restrictions land here, on the entrypoint that
    is already allowed to do both, and the pass receives what it measures.

    The context is `default_context()` with no `ceiling=` of its own, exactly as `score_shard`
    takes it: a timing pass that set its own ceiling would be timing a layer nobody scored under,
    which is the failure AD-6 exists to prevent.
    """
    from nbc.harness.timing import run_timing_pass

    _, items = read_corpus(pins, root)
    context = default_context()
    baselines = open_baselines(pins)
    report = run_timing_pass(
        items,
        context,
        baselines,
        expected_keys=[baseline.key for baseline in pins.baselines],
    )
    return {"timing": report.as_json_object(), "items": len(items)}


def main(argv: list[str] | None = None) -> int:
    """`score-shard` runs inference for hours; `merge` verifies and writes. Two subcommands.

    Both read the corpus through the guarded door, so both refuse a corpus that is not the one this
    declaration describes -- including `merge`, which would otherwise happily verify a set of
    shards against a demand set computed from rows nobody checked.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m nbc.harness.run",
        description=(
            "Score the verified corpus under both conditions, in shards, and merge the shards "
            "into one scores file."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help=(
            f"directory holding pins.toml, the corpus and {RESULTS_DIRNAME}/ "
            f"(default: the repository root)"
        ),
    )
    parser.add_argument(
        "--shards",
        metavar="N",
        type=int,
        default=None,
        help=(
            "how many shards the pass is split into. Membership is derived from each key's "
            "digest, so the same N always produces the same partition. Required by score-shard "
            "and merge; the timing pass is not sharded and does not take it"
        ),
    )
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    score = subcommands.add_parser(
        "score-shard",
        help=(
            f"score the keys shard I owns and append them to "
            f"{RESULTS_DIRNAME}/{_SHARD_STEM}-N-I.jsonl; a key already in that file is not "
            f"scored again"
        ),
    )
    score.add_argument(
        "--shard",
        metavar="I",
        type=int,
        required=True,
        help="which of the N shards to score, 0-based",
    )
    subcommands.add_parser(
        "merge",
        help=(
            f"verify that the N shard files are one pass over the corpus and write "
            f"{RESULTS_DIRNAME}/{SCORES_FILENAME}. Writes nothing if any check fails"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default=PROFILE_FULL,
        help=(
            "which run this is. `smoke` samples the declared number of items per cell from "
            "pins.toml and refuses to write into the repository's own results directory, because "
            "its table is structurally identical to the published one with a small n as the only "
            "tell"
        ),
    )
    subcommands.add_parser(
        "all",
        help=(
            "the whole run: preflight, verify the pins and the caveats, refuse a partial corpus, "
            "merge the scores, run the timing pass, aggregate, evaluate the four conditions and "
            "write results/results.json. Not sharded -- score the shards first"
        ),
    )
    subcommands.add_parser(
        "report",
        help=(
            "render the table from results/results.json and inject it into the README. Story 5.1 "
            "owns the renderer; this aborts rather than emitting an empty table"
        ),
    )
    subcommands.add_parser(
        "timing",
        help=(
            "measure the canonicalization layer's cost and each baseline's inference latency in "
            "one dedicated pass, one document at a time at batch size 1. Not sharded: a sharded "
            "pass has many processes contending for one machine and would time the contention"
        ),
    )
    args = parser.parse_args(argv)

    # `--shards` stays on the top-level parser so the documented `--shards N score-shard --shard I`
    # keeps working, and is checked here rather than by argparse because `timing` genuinely takes
    # no shard count and a required flag it must pass a meaningless value for is a flag that will
    # eventually be given a meaningful-looking one.
    if args.subcommand in {"score-shard", "merge"} and args.shards is None:
        parser.error(f"{args.subcommand} requires --shards N")

    report: dict[str, object]
    try:
        platform.preflight()
        pins = load_pins(args.root)
        if args.subcommand == "score-shard":
            report = score_shard(
                pins,
                shards=args.shards,
                shard=args.shard,
                root=args.root,
                profile=args.profile,
            )
        elif args.subcommand == "timing":
            report = timing_pass(pins, root=args.root)
        elif args.subcommand == "all":
            report = full_run(
                pins, shards=args.shards or 1, root=args.root, profile=args.profile
            )
        elif args.subcommand == "report":
            raise ResultsIncomplete(
                "the renderer is story 5.1 -- 'the table is a pure function of the results file' -- "
                "and it has not landed. `all` produces results/results.json, which is that "
                "function's input. This aborts rather than writing an empty table into the README, "
                "because an empty rendered block and a rendered block are indistinguishable to a "
                "reader who did not run the command"
            )
        else:
            report = merge_shards(
                pins, shards=args.shards, root=args.root, profile=args.profile
            )
    except NbcError as abort:
        print(abort, file=sys.stderr)
        return exit_code_for(abort)

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the tests
    raise SystemExit(main())
