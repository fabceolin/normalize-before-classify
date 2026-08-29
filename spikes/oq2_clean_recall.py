"""OQ2: clean recall per pinned baseline, measured before a corpus exists.

**This is a spike.** It is exploratory, it is outside `src/nbc/`, it writes to neither `data/`
nor `results/`, and nothing in the published measurement path imports it. See `spikes/README.md`.

The question (PRD open question OQ2) is whether each pinned baseline is strong enough on *clean*
attack text for its degradation under encoding to mean anything. A baseline whose clean recall is
already poor makes its own degradation meaningless, and the answer has to arrive before the corpus
is built: a baseline swapped afterwards costs the corpus and the whole re-run.

Two rules shape the code:

- **The spike never does the model's work.** Tokenization, the window policy, the softmax and the
  window-to-document reduction all come from `nbc.baselines`, exactly as the published run reaches
  them. A spike that rolled its own would risk failing a baseline on its own bug, and this is the
  one measurement whose false negative is expensive.
- **The spike owns no identity.** Repository ids, revisions, artifact paths, splits, the attack
  label and the decision threshold all come from `pins.toml` through `nbc.pins.load_pins()`, which
  also applies the baseline-set and lineage gates before anything is scored.

Reading rows needs `pyarrow`, which arrives with the already-declared build extra
(`uv sync --frozen --extra build`); the measurement runtime deliberately does not carry it.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# A spike is run from the repository root, not installed. This is what lets `python
# spikes/oq2_clean_recall.py` work in a checkout where `nbc` was never `pip install -e`'d,
# and it is the only liberty this file takes with the import system.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbc import pins  # noqa: E402
from nbc.baselines.onnx_adapter import BATCH_SIZE, open_baseline  # noqa: E402
from nbc.baselines.tokenization import open_windower  # noqa: E402

SELECTION_SEED = 20260829
"""The seed the sample is drawn under, printed beside every number it produced.

A recall over an undeclared subset is not a measurement. The seed is a spike constant rather than
a pin because `pins.toml` declares the *corpus* draw, which is a different story's decision.
"""

CONFIDENCE_Z = 1.959963984540054
"""The two-sided 95% normal quantile, spelled out rather than imported from a stats package."""

REPORT_CHUNK = 32
"""Documents handed to `score()` at once, so peak memory does not scale with the whole pool.

A memory knob and nothing else, which is a claim rather than an assumption: it decides batch
composition, `_feed` pads each batch to its own longest window, and a graph that ignored its
attention mask would turn this into a measurement parameter. Both pinned graphs honour it,
measured, and `tests/baselines/test_onnx_adapter.py` holds that as a property. The value is
printed in the report so a number can be tied to the chunk that produced it either way.
"""


@dataclass(frozen=True, slots=True)
class Interval:
    """A Wilson score interval for a binomial proportion."""

    low: float
    high: float


def wilson_interval(hits: int, n: int, z: float = CONFIDENCE_Z) -> Interval:
    """The 95% Wilson interval for `hits` out of `n`.

    Wilson rather than the normal approximation because recall on a strong classifier sits near
    1.0, where the normal interval runs past it and stops being an interval at all. `n == 0` has
    no rate, so it has no interval either.
    """
    if n < 0 or hits < 0 or hits > n:
        raise ValueError(f"hits={hits!r} out of n={n!r} is not a proportion")
    if n == 0:
        return Interval(float("nan"), float("nan"))

    p = hits / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(max(0.0, centre - spread), min(1.0, centre + spread))


def select_positives(
    rows: Iterable[tuple[str, int]], *, attack_label: int, limit: int, seed: int
) -> list[str]:
    """The attack positives to score, deduplicated and drawn under a declared rule.

    Exact duplicates are collapsed because a payload that appears four times would otherwise
    weigh four times in a recall this experiment reads as "per attack". The pool is sorted before
    it is shuffled so the draw depends on the seed and on nothing else -- not on parquet row
    order, not on which split was read first.
    """
    unique = sorted({text for text, label in rows if label == attack_label and text})
    if limit and limit < len(unique):
        random.Random(seed).shuffle(unique)
        return sorted(unique[:limit])
    return unique


def read_rows(dataset: pins.AttackDataset) -> list[tuple[str, int]]:
    """Every `(text, label)` row of the pinned dataset, over every pinned split.

    Counts taken over a single split are the same error as counts taken over dataset rows
    instead of attack positives, so the splits are read together and never one at a time.
    """
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as failure:  # pragma: no cover - environment, not logic
        raise SystemExit(
            "this spike reads the pinned dataset's parquet rows and needs `pyarrow`, which "
            "comes with the build extra: `uv sync --frozen --extra build`"
        ) from failure

    rows: list[tuple[str, int]] = []
    for split in dataset.splits:
        for path in _dataset_files(dataset, split):
            table = parquet.read_table(path, columns=["text", "label"])
            rows.extend(
                zip(table.column("text").to_pylist(), table.column("label").to_pylist())
            )
    return rows


def _is_shard_of(name: str, split: str) -> bool:
    """Whether a repository file is one of `split`'s parquet shards.

    The hub's convention is `data/<split>-<shard>-of-<count>.parquet`, and a split can be
    sharded. This reads the convention off the file listing rather than writing a shard count
    into the spike, so a dataset that ships four shards is read whole rather than quarter.
    """
    stem = Path(name).name
    if not name.endswith(".parquet"):
        return False
    return stem == f"{split}.parquet" or stem.startswith(f"{split}-")


def _dataset_files(dataset: pins.AttackDataset, split: str) -> list[Path]:
    """The pinned split's parquet shards, fetched at the pinned revision if not already cached."""
    snapshot = dataset.artifact.snapshot_dir()
    if snapshot.is_dir():
        cached = sorted(
            path
            for path in snapshot.rglob("*.parquet")
            if _is_shard_of(str(path.relative_to(snapshot)), split)
        )
        if cached:
            return cached

    try:
        from huggingface_hub import list_repo_files
    except ModuleNotFoundError as failure:  # pragma: no cover - environment, not logic
        raise SystemExit(
            f"{dataset.repository}@{dataset.revision} is not in the local cache and "
            "`huggingface_hub` is not installed, so it cannot be fetched"
        ) from failure

    listed = [
        name
        for name in list_repo_files(
            dataset.repository, revision=dataset.revision, repo_type="dataset"
        )
        if _is_shard_of(name, split)
    ]
    if not listed:
        raise SystemExit(
            f"{dataset.repository}@{dataset.revision} ships no parquet shard for split "
            f"{split!r}, which `pins.toml` declares"
        )
    return [
        Path(_download(dataset.repository, name, dataset.revision, kind="dataset"))
        for name in sorted(listed)
    ]


def _download(repository: str, filename: str, revision: str, *, kind: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as failure:  # pragma: no cover - environment, not logic
        raise SystemExit(
            f"{repository}@{revision}:{filename} is not in the local cache and "
            "`huggingface_hub` is not installed, so it cannot be fetched"
        ) from failure

    repo_type = "dataset" if kind == "dataset" else "model"
    print(f"  fetching {repository}@{revision[:8]} {filename}", file=sys.stderr, flush=True)
    return hf_hub_download(
        repo_id=repository, filename=filename, revision=revision, repo_type=repo_type
    )


def ensure_cached(baseline: pins.Baseline) -> None:
    """Fetch the three files this baseline's pin names, and no others, if they are missing."""
    snapshot = baseline.artifact.snapshot_dir()
    for name in (baseline.config_path, baseline.tokenizer_path, baseline.graph_path):
        if not (snapshot / name).is_file():
            _download(baseline.repository, name, baseline.revision, kind="model")


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one baseline said about one pool of clean attack positives."""

    key: str
    threshold: float
    n: int
    hits: int
    interval: Interval
    quartiles: tuple[float, float, float]
    seconds: float

    @property
    def recall(self) -> float:
        return self.hits / self.n if self.n else float("nan")


def measure(baseline: pins.Baseline, texts: Sequence[str], *, chunk: int) -> Measurement:
    """Score every text through the pinned model boundary and count what fired.

    The windower and the adapter are opened from the *same* `baseline`, which is what makes a
    mismatch impossible here: the two carry different frames and window lengths per baseline, and
    handing one baseline's windower to another's graph would score real text under the wrong
    tokenizer with nothing aborting. A harness that opens them separately has to check that; this
    caller cannot get it wrong.
    """
    ensure_cached(baseline)
    scorer = open_baseline(baseline, open_windower(baseline))

    probabilities: list[float] = []
    started = time.monotonic()
    for start in range(0, len(texts), chunk):
        batch = texts[start : start + chunk]
        probabilities.extend(score.p_injection for score in scorer.score(batch))
        print(
            f"  {baseline.key}: {len(probabilities)}/{len(texts)}",
            file=sys.stderr,
            end="\r",
            flush=True,
        )
    elapsed = time.monotonic() - started
    print(file=sys.stderr)

    hits = sum(1 for p in probabilities if p >= baseline.threshold)
    return Measurement(
        key=baseline.key,
        threshold=baseline.threshold,
        n=len(probabilities),
        hits=hits,
        interval=wilson_interval(hits, len(probabilities)),
        quartiles=_quartiles(probabilities),
        seconds=elapsed,
    )


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Where the probabilities sit, so a rate at one threshold is not the only thing reported."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    ordered = sorted(values)
    return tuple(ordered[min(len(ordered) - 1, int(q * len(ordered)))] for q in (0.25, 0.5, 0.75))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score this many attack positives instead of every one of them (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--chunk", type=int, default=REPORT_CHUNK)
    parser.add_argument(
        "--pins-root",
        type=Path,
        default=None,
        help=(
            "read pins.toml from this directory instead of the repository root. This is the "
            "bootstrap: `pins.toml` requires an OQ2 record per baseline, so re-measuring after "
            "a pin moves is done against a scratch copy carrying the new revision, and the "
            "committed file is never left declaring a recall it does not have"
        ),
    )
    arguments = parser.parse_args(argv)

    pinned = pins.load_pins(arguments.pins_root)
    # Step 2 of the published run's own sequence, and not skipped because this is a spike: a
    # revision that no longer resolves to the recorded commit aborts before any inference, or
    # the number below is about an artifact nobody can fetch again.
    pins.verify_revisions(pinned, pins.resolve_from_cache_then_hub)

    pool: list[tuple[str, int]] = []
    labels: list[int] = []
    for dataset in pinned.attack_datasets:
        rows = read_rows(dataset)
        pool.extend(rows)
        labels.append(dataset.attack_label)
        print(
            f"{dataset.repository}@{dataset.revision[:8]} splits={list(dataset.splits)} "
            f"rows={len(rows)}",
            file=sys.stderr,
        )
    if not labels:  # pragma: no cover - `load_pins` refuses a file pinning no dataset
        raise SystemExit("no attack dataset is pinned, so there is nothing to measure recall on")
    if len({*labels}) > 1:  # pragma: no cover - one dataset is pinned today
        raise SystemExit("the pinned datasets disagree about which label is the attack one")

    attack_label = labels[0]
    texts = select_positives(
        pool, attack_label=attack_label, limit=arguments.limit, seed=arguments.seed
    )
    positive_rows = [text for text, label in pool if label == attack_label and text]
    print(
        f"attack positives: {len(positive_rows)} rows, {len(set(positive_rows))} unique, "
        f"scoring {len(texts)} (seed {arguments.seed})",
        file=sys.stderr,
    )

    measurements = [
        measure(baseline, texts, chunk=arguments.chunk) for baseline in pinned.baselines
    ]

    drawn = (
        "every one of them"
        if not arguments.limit
        else f"a seeded sample of {arguments.limit}"
    )
    print()
    print("OQ2 -- clean recall on pinned attack positives, no canonicalization, no dressing")
    print(
        f"pool: {len(texts)} unique attack positives, "
        f"selection: sorted, shuffled under seed {arguments.seed}, "
        f"{drawn}"
    )
    # `--chunk` decides how documents group into batches, and `_feed` pads each batch to its own
    # longest window -- so composition reaches the tensor a document is scored on. Measured on
    # both pinned graphs it moves nothing (max|delta| 0.0 over 300 payloads at chunk 8/32/32
    # reversed against one-per-call), because both honour attention_mask, and the adapter's tests
    # now hold that as a property rather than an observation. Printed anyway: a parameter that
    # could reach a published rate on some future pin is not one a report should leave silent.
    print(
        f"batching: --chunk {arguments.chunk}, adapter BATCH_SIZE {BATCH_SIZE} "
        f"(scores are batch-invariant on graphs honouring attention_mask)"
    )
    print()
    header = (
        f"{'baseline':<26} {'thr':>5} {'n':>6} {'hits':>6} {'recall':>8} "
        f"{'95% Wilson':>18} {'p25/p50/p75':>22} {'sec':>7}"
    )
    print(header)
    print("-" * len(header))
    for m in measurements:
        interval = f"[{m.interval.low:.4f}, {m.interval.high:.4f}]"
        quartiles = "/".join(f"{q:.3f}" for q in m.quartiles)
        print(
            f"{m.key:<26} {m.threshold:>5.2f} {m.n:>6} {m.hits:>6} {m.recall:>8.4f} "
            f"{interval:>18} {quartiles:>22} {m.seconds:>7.1f}"
        )
    print()
    print(
        "Reading this: recall here is measured on rows the training-overlap filter (FR3.3) has\n"
        "not yet removed, because that filter belongs to the corpus build. A baseline declaring\n"
        "`seeded-from-declared-training-source` against the pinned dataset therefore reads as an\n"
        "upper bound on its own clean recall. Each baseline's own declaration:"
    )
    for baseline in pinned.baselines:
        for dataset in pinned.attack_datasets:
            relationship = baseline.lineage.relationship_to(dataset.repository)
            print(f"  {baseline.key:<26} {dataset.key}: {relationship}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the spike's entrypoint
    raise SystemExit(main())
