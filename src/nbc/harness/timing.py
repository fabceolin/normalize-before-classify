"""What the layer costs, measured away from everything that would make the number wrong.

The reader this artifact is for asks one question the table cannot otherwise answer: *what does
this cost me?* N3 -- the third falsification condition -- is decided by dividing the layer's p95 by
a baseline's, so without this pass that condition has no right-hand side at all.

**It cannot be measured where the scores are.** Story 4.2 sharded the scoring pass across processes
precisely so that an eighty-five-hour matrix finishes, and every one of those processes contends
for the same cache and memory bandwidth. A latency taken there would describe the contention rather
than the layer, and would describe it differently on every machine that chose a different number of
shards. So cost gets its own pass: one document at a time, single-threaded, with nothing else
running.

**Inference is measured in the same pass and at batch size 1.** The same pass, because two numbers
taken under different conditions cannot be divided and N3 divides them. Batch size 1, because the
ratio is a *per-document* statement: a batched configuration amortises fixed per-call work across a
batch, so its per-item average describes a throughput scenario rather than a document's latency. It
is the smaller number, and publishing it as "inference latency" would make the layer's share of the
budget look larger than it is at the operating point a reader cares about -- the conservative
direction, and still wrong, because the two figures would not be measurable against each other.

**This module receives everything it measures and opens nothing**, and three separate rules point
that way rather than one:

*AD-6.* The entrypoint constructs one `CanonContext` and hands it to every pass, so that a second
pass cannot invent a second recursion ceiling and time a layer nobody scored under.
`tests/canon/test_recursion.py::test_only_the_pipeline_constructs_a_context` names this module as
refused, and the right answer is to receive the context rather than widen the allow-list.

*AD-1.* Only `corpus/build.py` and `corpus/manifest.py` may name a corpus file. Items arrive here as
`CorpusItem`s that `harness/run.py` read through the guarded door.

*The offline suite.* The layer is pure Python, so it can be timed for real with no model in the
process -- but only by a module that does not have to open a corpus first. Taking its inputs
satisfies all three at once instead of making each a special case.

**Tracing stays on, and the pass refuses a context that has it off.** The measurement pass runs the
layer with tracing enabled and the published recall was produced under that. Timing it with tracing
off would measure a layer nobody scored under and report a cost lower than the one a reader would
pay; refusing the cheaper configuration is better than accepting it and noting it.

**Never a mean, and there is no field for one.** A mean over a latency distribution with a tail is
precisely the number that hides the tail. Saying so while leaving an optional field would last until
the first person who found the gap untidy, so `Percentiles` has p50 and p95 and nothing else, and
every value it carries is an observed sample -- `stats.nearest_rank_percentile` returns one by
construction.

**Layer cost and inference latency are separate fields.** A single "cost" number that sometimes
means one and sometimes the other is exactly the shape N3 cannot be computed from.

**What this module does not do.** It evaluates no condition -- N3 is 4-6's, and this produces its
right-hand side. It writes no file and generates no README block; `results.json` is 4-7's, and this
returns the record it carries. It measures its own wall clock, not the full run's: assembling that
around all six stages is the entrypoint's, and reporting a partial figure under the full run's name
would be worse than reporting none.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from nbc.canon.pipeline import canonicalize
from nbc.errors import NbcError
from nbc.harness.stats import P50, P95, nearest_rank_percentile
from nbc.schema import (
    BENIGN_CLASSES,
    FAMILY_ATTACK,
    CanonContext,
    CorpusItem,
    Score,
)

__all__ = [
    "CORPUS_CLASSES",
    "CLASS_ATTACKS",
    "InferenceTiming",
    "LayerTiming",
    "Percentiles",
    "TimedBaseline",
    "TimingIncomplete",
    "TimingReport",
    "corpus_class_of",
    "run_timing_pass",
    "time_inference",
    "time_layer",
]

CLASS_ATTACKS: Final[str] = "attacks"
CORPUS_CLASSES: Final[tuple[str, ...]] = (CLASS_ATTACKS, *BENIGN_CLASSES)
"""The three groups the cost is broken out over.

The epic calls them "corpus files" and they are not: the corpus ships as two files, one per
family, and the last two groups here are classes inside the benign one. They are derived from each
record's own `family` and `benign_class` -- the same pair `CorpusItem` already checks -- rather than
from which file a row came out of, because a row's class is a property of the row and a grouping
keyed on a filename is one that a change to the corpus layout silently re-partitions.

Neither filename is spelled here, and that is enforced rather than remembered: AD-1's locator scan
in `tests/corpus/test_manifest.py` refuses a module outside `corpus/build.py` and
`corpus/manifest.py` that so much as contains one of those names in a string, prose included. It
caught this docstring on its first run, and the scan is right to: it cannot tell a path from a
sentence, which is exactly why the names stay out of modules that have no business holding one.

Built from `BENIGN_CLASSES` rather than spelled again, so a third benign class is covered here the
day it is declared instead of being silently left out of the cost table.
"""

BATCH_SIZE_ONE: Final[int] = 1
"""The batch every inference sample is taken at, named because it is the measurement.

`baselines/onnx_adapter.BATCH_SIZE` is the scoring pass's setting and is deliberately not used
here: a batched per-item average is a different quantity wearing the same name.
"""


class TimingIncomplete(NbcError, exit_code=32):
    """The timing pass cannot produce a number that describes the run that produced the table.

    Code 32 because 3 through 31 are taken. The inputs that produce it, each with the test that
    fires it:

    - a context whose `trace_enabled` is false, which would time a cheaper layer than the one the
      published recall was produced under;
    - a corpus with no rows of one of the three classes, because a cost reported over two of them
      and silently not the third reads as a cost over the corpus;
    - no items at all;
    - no baselines at all, or a baseline the caller named and did not supply, because N3's
      right-hand side would be missing and the condition would be `not_evaluable`.
    """


class TimedBaseline(Protocol):
    """What this pass needs from a baseline, and the whole of it.

    A `Protocol` for the reason `run.ScoringBaseline` is one: the offline suite has to be able to
    run this pass with no model in the process, and naming the one method states exactly how much
    of the adapter the timing depends on.
    """

    def score(self, texts: Sequence[str]) -> list[Score]: ...


@dataclass(frozen=True, slots=True)
class Percentiles:
    """p50 and p95 over a sample of integer nanoseconds, and **no mean**.

    There is no field for one, which is the only way to say "a mean is not reported here" that
    survives somebody helpfully adding it. A mean over a latency distribution with a tail is the
    number that hides the tail, and the tail is what a reader budgeting for this layer needs.

    Both values are observed samples: `stats.nearest_rank_percentile` returns one by construction,
    so no reported latency is a number that no clock produced.
    """

    p50: int
    p95: int
    n: int

    def __post_init__(self) -> None:
        for name in ("p50", "p95", "n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} is an integer count of nanoseconds, got {value!r}")
        if self.n < 1:
            raise ValueError("a percentile over no samples is undefined")
        if self.p50 > self.p95:
            raise ValueError(f"p50 {self.p50} exceeds p95 {self.p95}")

    def as_json_object(self) -> dict[str, object]:
        return {"p50_ns": self.p50, "p95_ns": self.p95, "n": self.n}


def _percentiles(samples: Sequence[int]) -> Percentiles:
    """The two declared percentiles, both through `stats`. Nothing here sorts or ranks."""
    return Percentiles(
        p50=nearest_rank_percentile(samples, P50),
        p95=nearest_rank_percentile(samples, P95),
        n=len(samples),
    )


@dataclass(frozen=True, slots=True)
class LayerTiming:
    """The canonicalization layer's cost, overall and per corpus class.

    `trace_enabled` is on the record rather than assumed, because a cost measured with tracing off
    is a cost for a layer nobody scored under, and a reader comparing two runs needs to know which
    one they are reading.

    Today it can only be `True`, because `time_layer` refuses the other value. The field is not
    therefore decorative: the epic requires the results file to record the fact, and a reader of a
    published file has no access to the refusal that made it true. Said plainly rather than left for
    somebody to notice it never varies.
    """

    overall: Percentiles
    by_class: Mapping[str, Percentiles]
    trace_enabled: bool

    def __post_init__(self) -> None:
        missing = [name for name in CORPUS_CLASSES if name not in self.by_class]
        if missing:
            raise ValueError(f"the layer timing covers {CORPUS_CLASSES}; it is missing {missing}")
        object.__setattr__(self, "by_class", dict(self.by_class))

    def as_json_object(self) -> dict[str, object]:
        return {
            "overall": self.overall.as_json_object(),
            "by_class": {
                name: self.by_class[name].as_json_object() for name in CORPUS_CLASSES
            },
            "trace_enabled": self.trace_enabled,
        }


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    """Per-baseline inference latency at batch size 1, in a field distinct from the layer's."""

    by_baseline: Mapping[str, Percentiles]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.by_baseline:
            raise ValueError("inference timing over no baseline gives N3 no right-hand side")
        if self.batch_size != BATCH_SIZE_ONE:
            raise ValueError(
                f"inference latency is measured at batch size {BATCH_SIZE_ONE}, got "
                f"{self.batch_size!r}; a batched per-item average is a different quantity"
            )
        object.__setattr__(self, "by_baseline", dict(self.by_baseline))

    def as_json_object(self) -> dict[str, object]:
        return {
            "by_baseline": {
                key: value.as_json_object() for key, value in sorted(self.by_baseline.items())
            },
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True, slots=True)
class TimingReport:
    """One dedicated pass: what the layer cost, what inference cost, and how long the pass took.

    The two costs are separate fields and neither is derivable from the other. A single "cost"
    number that sometimes means one and sometimes the other is the shape N3 cannot be computed from,
    and N3 is the whole reason this pass is mandatory.
    """

    layer: LayerTiming
    inference: InferenceTiming
    elapsed_ns: int

    def __post_init__(self) -> None:
        if isinstance(self.elapsed_ns, bool) or not isinstance(self.elapsed_ns, int):
            raise ValueError(f"elapsed_ns is an integer, got {self.elapsed_ns!r}")
        if self.elapsed_ns < 0:
            raise ValueError(f"elapsed_ns must not be negative, got {self.elapsed_ns}")

    def as_json_object(self) -> dict[str, object]:
        return {
            "layer_ns": self.layer.as_json_object(),
            "inference_ns": self.inference.as_json_object(),
            "elapsed_ns": self.elapsed_ns,
        }


def corpus_class_of(item: CorpusItem) -> str:
    """Which of the three groups a row belongs to, from the row rather than from a filename.

    The `family`/`benign_class` pair is the one `CorpusItem` already checks at construction, so a
    row that reached here is one whose two fields agree; this reads them rather than re-deriving
    the rule.

    The abort below is therefore **unreachable through a `CorpusItem`** -- and it stays, because
    this function takes anything with those two attributes and a caller in a later story may hand
    it something else. `tests/harness/test_timing.py` fires it with a stand-in rather than leaving
    a branch nobody has seen run.
    """
    if item.family == FAMILY_ATTACK:
        return CLASS_ATTACKS
    if item.benign_class in BENIGN_CLASSES:
        return item.benign_class
    raise TimingIncomplete(
        f"corpus item {item.id!r} is family {item.family!r} with benign class "
        f"{item.benign_class!r}, which is none of {CORPUS_CLASSES}"
    )


def time_layer(items: Sequence[CorpusItem], context: CanonContext) -> LayerTiming:
    """The layer's cost, one document at a time, with the context the entrypoint built.

    Every sample is one document canonicalized once, in the order the corpus was read. No warm-up
    run is discarded and no document is timed twice: the published recall was produced by a pass
    that saw each document once, and a warmed measurement would be a cost for a run nobody made.
    """
    if not context.trace_enabled:
        raise TimingIncomplete(
            "the timing pass was handed a context with tracing disabled. The measurement pass runs "
            "the layer with tracing on and the published recall was produced under that, so a cost "
            "measured here would be for a layer nobody scored under -- and it would be the lower "
            "of the two"
        )
    if not items:
        raise TimingIncomplete("the timing pass was handed no corpus items")

    samples: dict[str, list[int]] = {name: [] for name in CORPUS_CLASSES}
    overall: list[int] = []
    for item in items:
        name = corpus_class_of(item)
        started = time.perf_counter_ns()
        canonicalize(item.text, context)
        elapsed = time.perf_counter_ns() - started
        samples[name].append(elapsed)
        overall.append(elapsed)

    empty = [name for name in CORPUS_CLASSES if not samples[name]]
    if empty:
        raise TimingIncomplete(
            f"the corpus handed to the timing pass has no rows of {empty}; a cost reported over "
            f"the other classes and silently not these reads as a cost over the corpus"
        )

    return LayerTiming(
        overall=_percentiles(overall),
        by_class={name: _percentiles(samples[name]) for name in CORPUS_CLASSES},
        trace_enabled=context.trace_enabled,
    )


def time_inference(
    items: Sequence[CorpusItem],
    baselines: Mapping[str, TimedBaseline],
    *,
    expected_keys: Sequence[str] | None = None,
) -> InferenceTiming:
    """Per-baseline latency at batch size 1, in the same pass and by the same clock as the layer.

    `expected_keys` is the set the pins declare. Supplied, it is compared: a baseline the run says
    it measured and did not is how N3 acquires a right-hand side for one column and not another,
    and the missing one is the one whose absence nobody notices.

    **The measurement order is the mapping's, and that is a declared bias rather than a neutral
    choice.** Each baseline's samples are taken in one block, so its session stays warm across them
    -- which is what a deployment running one model looks like -- but the first baseline measured
    pays any lazy initialisation and runs on a machine in a different state from the last.
    Interleaving the baselines would trade that for alternating cache eviction, which is a different
    distortion and not obviously a smaller one. It matters because N3 takes a `min` over the
    baselines' p95: a large enough ordering artifact could pick the wrong "fastest" one. A
    discarded warm-up would remove it and this pass refuses warm-ups on other grounds, so the
    honest resolution is to say which order was used rather than to imply there was a free one.
    """
    if not baselines:
        raise TimingIncomplete(
            "the timing pass was handed no baseline. N3 compares the layer's p95 against the "
            "fastest baseline's, so with none the condition is not_evaluable and the run aborts"
        )
    if not items:
        raise TimingIncomplete("the timing pass was handed no corpus items")

    if expected_keys is not None:
        missing = [key for key in expected_keys if key not in baselines]
        if missing:
            raise TimingIncomplete(
                f"the pins declare baselines {sorted(expected_keys)} and the timing pass was "
                f"handed {sorted(baselines)}; {missing} would have no latency and N3 would be "
                f"evaluable for one column and not another"
            )

    by_baseline: dict[str, Percentiles] = {}
    for key, baseline in baselines.items():
        samples: list[int] = []
        for item in items:
            started = time.perf_counter_ns()
            baseline.score([item.text])
            samples.append(time.perf_counter_ns() - started)
        by_baseline[key] = _percentiles(samples)

    return InferenceTiming(by_baseline=by_baseline, batch_size=BATCH_SIZE_ONE)


def run_timing_pass(
    items: Sequence[CorpusItem],
    context: CanonContext,
    baselines: Mapping[str, TimedBaseline],
    *,
    expected_keys: Sequence[str] | None = None,
) -> TimingReport:
    """Both halves, in one pass, on one clock, and the pass's own wall clock around them.

    `elapsed_ns` is this pass's, not the full run's. Assembling the figure the README states around
    all six stages is the entrypoint's job; reporting a partial one under the full run's name would
    be worse than reporting none.
    """
    started = time.perf_counter_ns()
    layer = time_layer(items, context)
    inference = time_inference(items, baselines, expected_keys=expected_keys)
    return TimingReport(
        layer=layer,
        inference=inference,
        elapsed_ns=time.perf_counter_ns() - started,
    )
