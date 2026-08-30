""""Scored exactly once" as an invariant that survives being split across processes.

Measured on 2026-08-30 against the committed pins: one pass over the corpus this build writes is
160,575 windows of 512 tokens for the primary baseline, at 0.6 to 1.3 s per window on a 12th-gen
i7 under the adapter's declared path. That is ~40 h for the primary baseline and ~85 h for the
full matrix. SC3 promises a stranger reproduces the table with one documented command, and a
command nobody can wait out is not a reproduction path -- so the pass is a **set of shards**, each
independently executable, whose union is exactly the corpus and whose merge is byte-identical to
what one process would have produced.

That split is allowed because of a measurement rather than an assumption. On 2026-08-30 the same
six items were scored in three separate Python processes, one of them with `OMP_NUM_THREADS=8` in
the environment, and `p_injection` came out identical to all 17 significant digits.
`tests/harness/test_run.py::test_cross_process_scoring_of_the_same_items_agrees_to_the_last_bit`
is that measurement as a test, because the whole design rests on it.

Two things that measurement does **not** say, checked the same day rather than left implied. It
does not say the process boundary is free in general -- it says these two pinned graphs on this
runtime are insensitive to it. And it does not say `INTRA_OP_NUM_THREADS = 1` is what makes it so:
with the session option removed the numbers did not move either, at 1, 4 or 8 threads. The option
stays set and stays recorded per shard regardless, because an observation about two graphs on one
machine is not a property of float32 reduction, and `path_problems` refuses a shard that disagrees
with the declared value whether or not this hardware would have noticed.

This module is **pure**: the standard library, `nbc.errors`, `nbc.schema` and `nbc.pins`. It
imports no model, opens no socket and reads no file. `harness/run.py` is where everything remote
lives -- the same split `corpus/attack.py` and `corpus/build.py` already use, and for the same
reason: the whole decision procedure (which shard an item belongs to, what counts as a complete
shard set, what a disagreement is, what the merged file looks like byte for byte) is covered by a
suite that runs with no network and no model in the process.

**Three properties, and each is a check rather than a convention.**

*Membership is content-derived.* `shard_of` hashes the key, never a row index. A row index is a
property of how the corpus file was read, and nothing promises a resumed run, a partially rebuilt
corpus or a future reader preserves the file's order. Indexed membership would move items between
shards between runs and the failure would be silent in the worst way: the coverage check would
still pass, every key appearing exactly once, while the resumed shard scored a set it did not
claim.

*Coverage is checked in both directions.* The merge computes the keys the corpus x baselines x
conditions demands and compares them against the keys the shard files carry, each way. A missing
key and a duplicated key are different messages because they have different causes -- a shard
that did not run, and two shards that claimed the same item.

*Disagreement aborts and is never resolved.* Any resolution rule -- first wins, last wins, the
mean -- publishes a number no single execution produced, and does it precisely on the items where
the execution paths differ, which are the borderline ones the threshold decides. The cost is that
one crossed shard stops a merge that is otherwise complete; what it buys is that a table's numbers
were each produced by one execution under one declared path. `pins.toml` already makes the same
argument about reduced precision.

**No threshold, no rate, no interval, no cell, and no wall-clock.** `p_injection` is what this
module carries. Turning it into a class is 4-3's, at the declared per-baseline threshold, in one
place. Cost is 4-5's, in a dedicated pass, because a sharded run has many processes contending for
one machine's cache and memory bandwidth and a latency taken here would describe the contention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from nbc.errors import NbcError
from nbc.schema import CONDITIONS, CorpusItem, ItemScore

__all__ = [
    "SHARD_SCHEMA_VERSION",
    "DeclaredPath",
    "ExecutionPath",
    "ScoreSetIncomplete",
    "ShardFile",
    "ShardHeader",
    "agreement_problems",
    "coverage_problems",
    "expected_keys",
    "key_of",
    "merge",
    "parse_shard",
    "path_problems",
    "render_record",
    "render_shard",
    "score_key",
    "serialize",
    "shard_of",
]


class ScoreSetIncomplete(NbcError, exit_code=27):
    """The shard files are not one pass over the corpus, so no scores file may be written.

    Code 27 because 3 through 26 are taken. One class for every way a set of shards fails to be a
    partition of the work, because the remedy is the same in every case -- find out which process
    produced what, and re-run the shards that are wrong -- and because a caller that wanted to
    distinguish them can read `problems`.

    The inputs that produce it, each with the test that fires it:

    - a key the corpus demands that no shard file carries, or a key a shard file carries that the
      corpus does not demand;
    - one key carried by two shard files, whether or not they agree about it;
    - one key carried by two shard files with different `p_injection` or `n_windows`;
    - shard files recording different execution paths -- provider, `intra_op_num_threads`, batch
      size -- or a path that is not the declared one;
    - shard files recording different pinned revisions for one baseline, or a revision the pins no
      longer name;
    - shard files recording different `build_id`s, or one that is not the corpus on disk;
    - a shard file whose last line has no terminator, which is what a process killed mid-write
      leaves behind;
    - a corpus with no rows, or a baseline set with no baselines: a pass over nothing publishes a
      rate over nothing.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("ScoreSetIncomplete must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "the shard files are not one pass over the corpus:\n  - " + "\n  - ".join(problems)
        )


SHARD_SCHEMA_VERSION: Final[int] = 1
"""The shape of a shard file's header line. Part of the header, so a reader can refuse a shape.

A shard file written by one checkout is merged by another -- that is the point of sharding -- so
the two are not guaranteed to be the same code. A version the reader does not know is a refusal
rather than a guess at which fields moved.
"""

_HASH_HEX_DIGITS: Final[int] = 8
"""How much of the key digest `shard_of` reduces. 32 bits, so the modulo bias over any shard count
a person would type is far below the level at which an uneven split would matter, and the number
stays small enough to print in a message a human is reading.
"""


def score_key(item_id: str, baseline_key: str, condition: str) -> str:
    """The identity of one cell of the pass: one item, one baseline, one condition.

    A JSON array of the three rather than a joined string, and that is not fussiness. Corpus ids
    already carry `::` (`<payload id>::<chain>`), so a `::`-joined key maps
    `("a::b", "c", "raw")` and `("a", "b::c", "raw")` onto the same string -- and a collision
    between two keys is a coverage check that reports one item scored twice and another never
    scored, for two items that were both scored exactly once. JSON's string escaping is
    unambiguous, so this encoding is injective for every triple.

    The key is never written to a file. It exists to be hashed by `shard_of`, compared as a set by
    `coverage_problems`, and printed in a message, so its length costs nothing.
    """
    return json.dumps([item_id, baseline_key, condition], separators=(",", ":"), ensure_ascii=False)


def key_of(score: ItemScore) -> str:
    """`score_key` for a record that already exists, so the two spellings cannot drift."""
    return score_key(score.item_id, score.baseline_key, score.condition)


def shard_of(key: str, shards: int) -> int:
    """Which shard `key` belongs to. A function of the key alone, and of nothing else.

    Never `index % shards`. See the module docstring: an index is a property of how the corpus
    file was read, and membership keyed on one moves items between shards between runs while every
    check stays green.
    """
    if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
        raise ValueError(f"shards must be a positive int, got {shards!r}")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_HEX_DIGITS]
    return int(digest, 16) % shards


def expected_keys(items: Sequence[CorpusItem], baseline_keys: Sequence[str]) -> tuple[str, ...]:
    """Every key the pass must produce: the corpus crossed with the baselines and the conditions.

    Sorted, so the demand set is a function of its inputs and not of the order they arrived in.

    Three refusals rather than an empty or a short answer, because each of them is a pass that
    would complete and publish a rate over the wrong population:

    - **no items.** A merge over an empty demand set succeeds against an empty shard set, and
      writes a scores file with nothing in it that every downstream check reads as consistent.
    - **no baselines.** The same, one axis over.
    - **a repeated item id.** "Exactly once" is counted over keys, so two rows sharing an id are
      one key: the second row is never scored and nothing anywhere says so. Corpus ids are
      content-derived and unique by construction today, which is exactly why this would be
      invisible if it stopped being true.
    """
    problems: list[str] = []
    if not items:
        problems.append(
            "the corpus carries no rows; a scoring pass over nothing merges cleanly and "
            "publishes a rate over nothing"
        )
    if not baseline_keys:
        problems.append(
            "no baseline was named; a pass with no columns merges cleanly and publishes a table "
            "with no measurements in it"
        )

    seen: dict[str, int] = {}
    for item in items:
        seen[item.id] = seen.get(item.id, 0) + 1
    repeated = sorted(item_id for item_id, count in seen.items() if count > 1)
    if repeated:
        problems.append(
            f"the corpus carries {len(repeated)} repeated item id(s) {repeated[:5]}; a key is one "
            f"item, one baseline and one condition, so the second row under a repeated id is "
            f"never scored and the coverage check cannot see that it was not"
        )

    duplicate_baselines = sorted(
        {key for key in baseline_keys if list(baseline_keys).count(key) > 1}
    )
    if duplicate_baselines:
        problems.append(
            f"baseline key(s) {duplicate_baselines} were named more than once; one column scored "
            f"twice is one column short"
        )

    if problems:
        raise ScoreSetIncomplete(*problems)

    return tuple(
        sorted(
            score_key(item.id, baseline_key, condition)
            for item in items
            for baseline_key in baseline_keys
            for condition in CONDITIONS
        )
    )


@dataclass(frozen=True, slots=True)
class ExecutionPath:
    """How one baseline was executed by one shard, as that shard observed it.

    Recorded **per shard** rather than once per run, because the whole point of sharding is that
    shards run in different processes and, at the scale this pass needs, on different machines.
    "The run used CPU" is a claim about one process; what a merged file needs is a claim about
    every process that contributed to it. Per-shard, "somebody ran one shard on the GPU box to
    save an hour" is an abort that names the shard instead of an invisible corruption in the
    borderline rows.

    `providers` is what the runtime made **active** for the session (`OnnxBaseline.providers`),
    not what the module constant asked for. Those are two different facts and the gap between
    them is the one this field exists to record.
    """

    baseline_key: str
    revision: str
    providers: tuple[str, ...]
    intra_op_num_threads: int
    batch_size: int

    def __post_init__(self) -> None:
        for name in ("baseline_key", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        providers = self.providers
        if isinstance(providers, list):
            # A header read back from JSON arrives with lists; the record is a tuple.
            providers = tuple(providers)
            object.__setattr__(self, "providers", providers)
        if not isinstance(providers, tuple) or not providers:
            raise ValueError(f"providers must be a non-empty tuple, got {self.providers!r}")
        if not all(isinstance(name, str) and name for name in providers):
            raise ValueError(f"providers must hold non-empty names, got {self.providers!r}")
        for name in ("intra_op_num_threads", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

    def as_json_object(self) -> dict[str, object]:
        return {
            "baseline_key": self.baseline_key,
            "revision": self.revision,
            "providers": list(self.providers),
            "intra_op_num_threads": self.intra_op_num_threads,
            "batch_size": self.batch_size,
        }

    def describe(self) -> str:
        """The path as one line, for a message that has to name two of them side by side."""
        return (
            f"{self.baseline_key}@{self.revision} on {list(self.providers)} "
            f"intra_op_num_threads={self.intra_op_num_threads} batch_size={self.batch_size}"
        )


@dataclass(frozen=True, slots=True)
class DeclaredPath:
    """The execution path this repository publishes under, as the declaration states it.

    The other side of `path_problems`' comparison, and it comes from somewhere else on purpose:
    the constants live in `baselines/onnx_adapter.py` and the revisions in `pins.toml`, while the
    `ExecutionPath` records come out of a file another process wrote. A check whose two sides were
    both built from the adapter's constants in this process would agree with itself no matter what
    the shard files said -- and shard-versus-shard agreement alone is not enough either, since a
    set of shards that all ran on CUDA agrees perfectly.
    """

    providers: tuple[str, ...]
    intra_op_num_threads: int
    batch_size: int
    revisions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ShardHeader:
    """The first line of a shard file: which shard this is, and under what it was produced.

    `build_id` is the corpus the shard scored. The corpus is verified on read, so a shard cannot
    have been scored over rows that failed the manifest -- but it can have been scored over a
    *previous* corpus, and by the time the merge runs the only trace of that is this field.
    """

    schema_version: int
    shards: int
    shard: int
    build_id: str
    paths: tuple[ExecutionPath, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SHARD_SCHEMA_VERSION:
            raise ValueError(
                f"a shard file declares schema_version {self.schema_version!r} and this reader "
                f"reads {SHARD_SCHEMA_VERSION}"
            )
        for name in ("shards", "shard"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {value!r}")
        if self.shards < 1:
            raise ValueError(f"shards must be at least 1, got {self.shards!r}")
        if not 0 <= self.shard < self.shards:
            raise ValueError(
                f"shard {self.shard!r} is outside 0..{self.shards - 1}"
            )
        if not isinstance(self.build_id, str) or not self.build_id:
            raise ValueError(f"build_id must be a non-empty string, got {self.build_id!r}")
        paths = tuple(self.paths)
        object.__setattr__(self, "paths", paths)
        if not paths:
            raise ValueError("a shard header records the execution path of every baseline it ran")
        keys = [path.baseline_key for path in paths]
        if len(set(keys)) != len(keys):
            raise ValueError(f"a shard header records one baseline twice: {sorted(keys)}")

    def path_for(self, baseline_key: str) -> ExecutionPath | None:
        for path in self.paths:
            if path.baseline_key == baseline_key:
                return path
        return None

    def as_json_object(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "shards": self.shards,
            "shard": self.shard,
            "build_id": self.build_id,
            "paths": [path.as_json_object() for path in sorted(self.paths, key=_path_order)],
        }


def _path_order(path: ExecutionPath) -> str:
    return path.baseline_key


@dataclass(frozen=True, slots=True)
class ShardFile:
    """One shard file as read: where it came from, what it declares, and what it carries.

    `name` is carried so every problem below can say which file it is about. A merge that reports
    "two shards disagree" without naming them leaves the operator to find the pair by hand across
    however many machines ran the pass.
    """

    name: str
    header: ShardHeader
    scores: tuple[ItemScore, ...]


def render_shard(header: ShardHeader, scores: Iterable[ItemScore]) -> str:
    """A whole shard file: the header line, then one record per line, LF-terminated.

    A shard file's record order is deliberately **not** a promise. The walk appends one record at
    a time and flushes, so a kill at hour five costs one item rather than five hours, and an
    append-only file cannot be kept in a canonical order without rewriting it -- which is the one
    operation that could lose the five hours. Order is fixed where it is load-bearing, in
    `serialize`, over the merged set.
    """
    lines = [_line(header.as_json_object())]
    lines.extend(_line(score.as_json_object()) for score in scores)
    return "".join(f"{line}\n" for line in lines)


def render_record(score: ItemScore) -> str:
    """One score as the terminated line a shard walk appends.

    Here rather than in `harness/run.py` so that every byte this project writes into a scores file
    is rendered by the pure module, and the appended line and the whole-file render cannot come to
    disagree about how a record is spelled.
    """
    return f"{_line(score.as_json_object())}\n"


def parse_shard(name: str, text: str) -> ShardFile:
    """Read a shard file back, refusing anything that is not one.

    **A record is complete if and only if its line is terminated.** A process killed mid-write
    leaves a partial line, and the danger is not that it fails to parse -- it usually does -- but
    that nobody looks. Two files truncated at different points would otherwise merge into a set
    that passes agreement and fails coverage, reported as "these keys were never scored" when what
    happened was that two processes were killed. So the terminator is checked structurally, before
    anything is parsed, and the tail is named.

    Truncation that lands exactly on a line boundary leaves a file this function accepts. That is
    not a hole: the records it is missing are missing from the coverage check too, which is the
    other half of the pair and reports them as unscored keys.
    """
    if not text:
        raise ScoreSetIncomplete(
            f"{name} is empty; a shard file carries its header on the first line, so an empty one "
            f"is a file that was created and never written to"
        )
    if not text.endswith("\n"):
        tail = text.rsplit("\n", 1)[-1]
        raise ScoreSetIncomplete(
            f"{name} ends without a line terminator, so its last {len(tail)} byte(s) are a record "
            f"a process was still writing when it stopped: {tail[:120]!r}. Re-run that shard "
            f"rather than merging what it managed to write"
        )

    lines = text.split("\n")[:-1]
    try:
        header = _parse_header(name, lines[0])
    except (KeyError, TypeError, ValueError) as error:
        # `json.JSONDecodeError` is a `ValueError`, and so is `UnicodeDecodeError`; catching only
        # the narrow name is the sibling this project has missed before. `KeyError` is in here for
        # the same reason one step over: a header missing `build_id` is a malformed shard file, and
        # letting it out as an unclassified crash would report a refusal as a bug in the reader.
        raise ScoreSetIncomplete(f"{name}:1 is not a shard header: {error}") from None

    scores: list[ItemScore] = []
    problems: list[str] = []
    for number, line in enumerate(lines[1:], start=2):
        try:
            scores.append(_parse_score(line))
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"{name}:{number} is not a score record: {error}")
    if problems:
        raise ScoreSetIncomplete(*problems)
    return ShardFile(name=name, header=header, scores=tuple(scores))


def coverage_problems(
    expected: Sequence[str], files: Sequence[ShardFile], *, shards: int
) -> tuple[str, ...]:
    """Every way the keys the shard files carry are not exactly the keys the corpus demands.

    Four questions, and each has a different cause, so each gets its own message:

    - a demanded key nobody carries -- **which shard was it?** The operator's next action is to
      re-run one shard, and the key alone does not say which;
    - a carried key nobody demanded -- an item that is no longer in the corpus, or a condition
      outside the vocabulary that got as far as a file;
    - a key two files carry -- the split was not a partition, whether or not the two agree;
    - a key in the wrong file -- shard 2 carrying a key `shard_of` puts in shard 0. That one
      matters because it is the exact signature of a run whose membership was computed some other
      way, and by itself it breaks nothing a reader would see.
    """
    problems: list[str] = []
    demanded = set(expected)

    owners: dict[str, list[str]] = {}
    for file in files:
        for score in file.scores:
            owners.setdefault(key_of(score), []).append(file.name)

    missing = sorted(demanded - set(owners))
    if missing:
        by_shard: dict[int, list[str]] = {}
        for key in missing:
            by_shard.setdefault(shard_of(key, shards), []).append(key)
        for index in sorted(by_shard):
            keys = by_shard[index]
            problems.append(
                f"shard {index} of {shards} owes {len(keys)} key(s) that no shard file carries, "
                f"starting with {keys[:3]}; that shard did not run, or did not finish"
            )

    unexpected = sorted(set(owners) - demanded)
    if unexpected:
        problems.append(
            f"{len(unexpected)} key(s) are carried by a shard file and demanded by nothing, "
            f"starting with {unexpected[:3]}; they were scored over a corpus or a vocabulary "
            f"this declaration does not have"
        )

    duplicated = sorted(key for key, names in owners.items() if len(names) > 1)
    for key in duplicated[:5]:
        problems.append(
            f"key {key} is carried by {sorted(owners[key])}; a key claimed twice means the split "
            f"was not a partition, so 'scored once' is not what happened"
        )
    if len(duplicated) > 5:
        problems.append(f"and {len(duplicated) - 5} further key(s) carried by more than one file")

    for file in files:
        misplaced = sorted(
            {key_of(score) for score in file.scores if shard_of(key_of(score), shards) != file.header.shard}
        )
        if misplaced:
            problems.append(
                f"{file.name} declares shard {file.header.shard} and carries {len(misplaced)} "
                f"key(s) that belong to another shard, starting with {misplaced[:3]}; membership "
                f"is a function of the key, so a file that disagrees was walked some other way"
            )

    return tuple(problems)


def agreement_problems(files: Sequence[ShardFile]) -> tuple[str, ...]:
    """Every key two shard files carry with different numbers, naming the key and both values.

    Not resolved, in any direction. See the module docstring: a first-wins, last-wins or mean rule
    publishes a number no execution produced, on exactly the borderline items the threshold
    decides. The whole record is compared and not only `p_injection`, because two shards that
    agree on the probability and disagree on `n_windows` were tokenizing differently, which is the
    same fault one step earlier.
    """
    problems: list[str] = []
    seen: dict[str, tuple[str, ItemScore]] = {}
    for file in files:
        for score in file.scores:
            key = key_of(score)
            first = seen.get(key)
            if first is None:
                seen[key] = (file.name, score)
                continue
            name, other = first
            if other == score:
                continue
            problems.append(
                f"key {key} is {_describe(other)} in {name} and {_describe(score)} in "
                f"{file.name}; two executions of one item produced different numbers, and no "
                f"rule for choosing between them publishes a number either of them computed"
            )
    return tuple(problems)


def path_problems(
    files: Sequence[ShardFile], *, build_id: str, declared: DeclaredPath
) -> tuple[str, ...]:
    """Every way the shard files were not all produced under the one declared execution path.

    Two comparisons, and both are needed. Shard against shard catches the run where one machine
    differed. Shard against `declared` catches the run where every machine differed the same way
    -- a whole pass on CUDA agrees with itself perfectly, and it is the one this gate is named for.
    """
    problems: list[str] = []
    if not files:
        return (
            "no shard file was found; a merge over nothing writes a scores file with nothing in "
            "it, which every downstream check reads as consistent",
        )

    declared_shards = {file.header.shards for file in files}
    if len(declared_shards) > 1:
        problems.append(
            f"the shard files declare different shard counts {sorted(declared_shards)}; two "
            f"different splits of one corpus do not compose into one pass"
        )
    indices = [file.header.shard for file in files]
    repeated = sorted({index for index in indices if indices.count(index) > 1})
    if repeated:
        problems.append(f"shard index/indices {repeated} are declared by more than one file")

    for file in files:
        if file.header.build_id != build_id:
            problems.append(
                f"{file.name} was scored over build_id {file.header.build_id!r} and the corpus on "
                f"disk is {build_id!r}; the corpus was rebuilt after that shard ran, so its "
                f"records describe rows this table would not be computed over"
            )

    # Every path, keyed by baseline, with the file that recorded it -- so a disagreement names
    # both sides rather than reporting that a set has two members.
    by_baseline: dict[str, list[tuple[str, ExecutionPath]]] = {}
    for file in files:
        for path in file.header.paths:
            by_baseline.setdefault(path.baseline_key, []).append((file.name, path))

    for baseline_key in sorted(by_baseline):
        recorded = by_baseline[baseline_key]
        first_name, first = recorded[0]
        for name, path in recorded[1:]:
            if path == first:
                continue
            problems.append(
                f"baseline {baseline_key!r} was executed as [{first.describe()}] by {first_name} "
                f"and as [{path.describe()}] by {name}; a score from another device, another "
                f"reduction order or another revision diverges in the last decimals, and the "
                f"decision threshold turns that into a class flip"
            )

        expected_revision = declared.revisions.get(baseline_key)
        if expected_revision is None:
            problems.append(
                f"the shard files record baseline {baseline_key!r}, which this declaration does "
                f"not pin; a column nobody declared cannot be published"
            )
        for name, path in recorded:
            if expected_revision is not None and path.revision != expected_revision:
                problems.append(
                    f"{name} scored baseline {baseline_key!r} at revision {path.revision!r} and "
                    f"the pins declare {expected_revision!r}"
                )
            if path.providers != declared.providers:
                problems.append(
                    f"{name} scored baseline {baseline_key!r} on {list(path.providers)} and the "
                    f"published execution path is {list(declared.providers)}"
                )
            if path.intra_op_num_threads != declared.intra_op_num_threads:
                problems.append(
                    f"{name} scored baseline {baseline_key!r} with intra_op_num_threads="
                    f"{path.intra_op_num_threads} and the declared path is "
                    f"{declared.intra_op_num_threads}; a threaded float32 reduction does not add "
                    f"up in the same order twice"
                )
            if path.batch_size != declared.batch_size:
                problems.append(
                    f"{name} scored baseline {baseline_key!r} at batch_size={path.batch_size} and "
                    f"the declared path is {declared.batch_size}"
                )

    missing_columns = sorted(set(declared.revisions) - set(by_baseline))
    if missing_columns:
        problems.append(
            f"no shard file records an execution path for pinned baseline(s) {missing_columns}; "
            f"the pass did not run that column"
        )

    return tuple(problems)


def merge(
    files: Sequence[ShardFile],
    *,
    expected: Sequence[str],
    build_id: str,
    declared: DeclaredPath,
    shards: int,
) -> tuple[ItemScore, ...]:
    """The one pass the shard files add up to, or an abort naming everything that is wrong.

    Every problem is collected before aborting -- a run that is wrong in two ways tells the
    operator both, rather than sending them back to a machine for a second answer. The order is
    path, then agreement, then coverage: a crossed execution path explains a disagreement, and
    reading "two shards disagree about this key" first sends the reader looking at the item.

    Returns records, not bytes. `serialize` renders them, for the reason `attack.serialize` does:
    keeping the serialization pure is what lets every claim about the file be tested offline, and
    keeping the write in one module is what makes the writer countable.
    """
    problems = (
        path_problems(files, build_id=build_id, declared=declared)
        + agreement_problems(files)
        + coverage_problems(expected, files, shards=shards)
    )
    if problems:
        raise ScoreSetIncomplete(*problems)

    # Deduplication is not needed here and is deliberately not done: `coverage_problems` has
    # already refused any key carried twice, so a dict keyed on the key would be hiding the
    # very thing that just passed.
    return _ordered(score for file in files for score in file.scores)


def serialize(scores: Iterable[ItemScore]) -> str:
    """The exact bytes of the merged scores file: one JSON object per line, LF-terminated.

    Sorted by `(item_id, baseline_key, condition)`, all three content-derived, so the file depends
    on what was scored and never on how the work was divided or on the order the shard files were
    read. That is the load-bearing property of this whole module and
    `tests/harness/test_score.py` checks it by scoring one corpus at 1, 3 and 7 shards and
    comparing bytes.

    `ensure_ascii=False` and a compact separator, matching `corpus/attack.serialize`, so the two
    committed JSONL files in this repository are written by one convention.
    """
    return "".join(f"{_line(score.as_json_object())}\n" for score in _ordered(scores))


def _ordered(scores: Iterable[ItemScore]) -> tuple[ItemScore, ...]:
    return tuple(
        sorted(scores, key=lambda score: (score.item_id, score.baseline_key, score.condition))
    )


def _line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _describe(score: ItemScore) -> str:
    # 17 significant digits: the number of decimal digits that round-trips a float64, so two
    # values that differ in the last bit are visibly different in the message rather than both
    # printing as the same rounded number.
    return f"p_injection={score.p_injection:.17g} over {score.n_windows} window(s)"


def _parse_header(name: str, line: str) -> ShardHeader:
    document: Any = json.loads(line)
    if not isinstance(document, Mapping):
        raise ValueError(f"the first line holds {type(document).__name__}, not an object")
    paths = document.get("paths")
    if not isinstance(paths, list):
        raise ValueError("the header records no paths list")
    return ShardHeader(
        schema_version=document["schema_version"],
        shards=document["shards"],
        shard=document["shard"],
        build_id=document["build_id"],
        paths=tuple(
            ExecutionPath(
                baseline_key=entry["baseline_key"],
                revision=entry["revision"],
                providers=tuple(entry["providers"]),
                intra_op_num_threads=entry["intra_op_num_threads"],
                batch_size=entry["batch_size"],
            )
            for entry in paths
        ),
    )


def _parse_score(line: str) -> ItemScore:
    document: Any = json.loads(line)
    if not isinstance(document, Mapping):
        raise ValueError(f"the line holds {type(document).__name__}, not an object")
    # Every key is read by name and none defaulted: a record missing `condition` would otherwise
    # arrive as one of the two conditions rather than as the malformed record it is.
    return ItemScore(
        item_id=document["item_id"],
        family=document["family"],
        benign_class=document["benign_class"],
        label=document["label"],
        baseline_key=document["baseline_key"],
        condition=document["condition"],
        p_injection=document["p_injection"],
        n_windows=document["n_windows"],
        max_depth_reached=document["max_depth_reached"],
        ceiling_hit=document["ceiling_hit"],
    )
