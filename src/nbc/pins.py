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
builder's, not this module's. This module states the obligation; `corpus/` discharges it.

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

import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Final, Mapping, Sequence

from nbc.errors import NbcError

__all__ = [
    "AttackDataset",
    "Baseline",
    "BaselineIneligible",
    "LINEAGE_RELATIONSHIPS",
    "Lineage",
    "Licence",
    "MINIMUM_BASELINES",
    "NOT_DECLARED",
    "PINS_FILENAME",
    "PINNED_PRECISION",
    "PinMismatch",
    "Pins",
    "PinsFileInvalid",
    "BaselineSetInvalid",
    "Provenance",
    "RemoteArtifact",
    "Resolver",
    "SCHEMA_VERSION",
    "SEEDED_FROM_TRAINING_SOURCE",
    "TRAINED_ON",
    "TRAINING_SOURCE_RELATIONSHIPS",
    "hf_cache_root",
    "load_pins",
    "main",
    "resolve_from_cache_then_hub",
    "resolve_over_http",
    "verify_revisions",
]

PINS_FILENAME: Final[str] = "pins.toml"
"""The file's name, at the repository root. Named once, here."""

SCHEMA_VERSION: Final[int] = 1
"""The shape this module knows how to read. A file declaring another version is refused."""

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

_REDUCED_PRECISION: Final[re.Pattern[str]] = re.compile(
    r"fp16|float16|bf16|bfloat16|mixed|int8|uint8|quant|_q4|_q8", re.IGNORECASE
)
_SHA: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{40}\Z")
_DATE: Final[re.Pattern[str]] = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")
_REPOSITORY: Final[re.Pattern[str]] = re.compile(r"\A[\w.-]+/[\w.-]+\Z")

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

    @property
    def declared(self) -> bool:
        return self.identifier != NOT_DECLARED

    def as_run_fields(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "source": self.source,
            "attribution": self.attribution,
            "redistributed": self.redistributed,
        }


@dataclass(frozen=True, slots=True)
class WindowPin:
    """A window length together with the file that declared it.

    The source travels with the value because "the model's maximum sequence length" resolves
    from three files that routinely disagree, and a length with no stated origin is a number the
    next reader re-derives from whichever file they open first.
    """

    length: int
    source: str
    confirmed_on: str

    def as_run_fields(self) -> dict[str, object]:
        return {"length": self.length, "source": self.source, "confirmed_on": self.confirmed_on}


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
        return self.training_sources.get(repository) == TRAINED_ON

    def as_run_fields(self) -> dict[str, object]:
        return {
            "checked_on": self.checked_on,
            "card_revision": self.card_revision,
            "attack_datasets": dict(self.attack_datasets),
            "training_sources": dict(self.training_sources),
        }


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
    def api_url(self) -> str:
        endpoint = _KIND_ENDPOINT[self.kind]
        return (
            f"https://huggingface.co/api/{endpoint}/{self.repository}"
            f"/revision/{self.revision}"
        )


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

    @property
    def artifact(self) -> RemoteArtifact:
        return RemoteArtifact("model", self.repository, self.revision)

    @property
    def family_pair(self) -> tuple[str, str]:
        """The pair SC5 rests on. Two baselines sharing it cannot corroborate each other."""
        return (self.architecture_family, self.tokenizer_family)

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
        }


@dataclass(frozen=True, slots=True)
class AttackDataset:
    """One pinned attack dataset, by identity.

    The draw -- sample size in attack positives, selection method, seed or sort key -- is
    declared by the corpus story in this same file. It is absent rather than defaulted, because
    a default draw is a draw nobody declared.
    """

    key: str
    repository: str
    revision: str
    splits: tuple[str, ...]
    attack_label: int
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
            "licence": self.licence.as_run_fields(),
            "provenance": self.provenance.as_run_fields(),
        }


@dataclass(frozen=True, slots=True)
class Pins:
    """Every remote artifact this project touches, as data."""

    schema_version: int
    verified_on: str
    verified_against: str
    baselines: tuple[Baseline, ...]
    attack_datasets: tuple[AttackDataset, ...]
    path: Path

    def remote_artifacts(self) -> tuple[RemoteArtifact, ...]:
        return tuple(
            [baseline.artifact for baseline in self.baselines]
            + [dataset.artifact for dataset in self.attack_datasets]
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
                # Derived, not declared: a reader can recompute it from the two blocks above,
                # and the corpus build reads it rather than a second copy of the same list.
                "required_exclusion_sources": list(self.required_exclusion_sources()),
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


def _read_licence(reader: _Reader, parent: Mapping[str, Any], where: str) -> Licence:
    table = reader.table(parent, "licence", where)
    at = f"{where}.licence"
    identifier = reader.string(table, "identifier", at)
    return Licence(
        identifier=identifier,
        source=reader.string(table, "source", at),
        attribution=reader.string(table, "attribution", at),
        redistributed=reader.boolean(table, "redistributed", at),
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

    graph_path = reader.string(table, "graph_path", where)
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

    window_table = reader.table(table, "window", where)
    window = WindowPin(
        length=reader.integer(window_table, "length", f"{where}.window"),
        source=reader.string(window_table, "source", f"{where}.window"),
        confirmed_on=reader.matching(
            reader.string(window_table, "confirmed_on", f"{where}.window"),
            _DATE,
            f"{where}.window.confirmed_on",
            "an ISO date (YYYY-MM-DD)",
        ),
    )
    if window.length <= 0 and "length" in window_table:
        reader.note(f"{where}.window.length must be positive, got {window.length!r}")
    if window.source and "tokenizer_config.json" in window.source:
        reader.note(
            f"{where}.window.source reads tokenizer_config.json, whose model_max_length is a "
            f"~1e30 sentinel in the pinned repositories; the window comes from the model config"
        )

    lineage_table = reader.table(table, "lineage", where)
    lineage = Lineage(
        checked_on=reader.matching(
            reader.string(lineage_table, "checked_on", f"{where}.lineage"),
            _DATE,
            f"{where}.lineage.checked_on",
            "an ISO date (YYYY-MM-DD)",
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

    return Baseline(
        key=key,
        repository=repository,
        revision=revision,
        threshold=threshold,
        graph_path=graph_path,
        precision=precision,
        graph_bytes=graph_bytes,
        tokenizer_path=reader.string(table, "tokenizer_path", where),
        config_path=reader.string(table, "config_path", where),
        architecture_family=reader.string(table, "architecture_family", where),
        tokenizer_family=reader.string(table, "tokenizer_family", where),
        window_policy=reader.string(table, "window_policy", where),
        window=window,
        licence=_read_licence(reader, table, where),
        lineage=lineage,
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
        checked_on=reader.matching(
            reader.string(provenance_table, "checked_on", f"{where}.provenance"),
            _DATE,
            f"{where}.provenance.checked_on",
            "an ISO date (YYYY-MM-DD)",
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
        splits=reader.strings(table, "splits", where),
        attack_label=reader.integer(table, "attack_label", where),
        licence=_read_licence(reader, table, where),
        provenance=provenance,
    )


def _note_stale_check(
    reader: _Reader,
    where: str,
    *,
    recorded: str,
    pinned: str,
    checked_on: str,
    what: str,
) -> None:
    """A check performed against a revision this file no longer pins is a check nobody re-ran.

    The date is metadata and the revision is the gate: a pin can move on the same day, and a
    date alone would let the declaration keep looking fresh while describing a different card.
    """
    if not recorded or not pinned or recorded == pinned:
        return
    reader.note(
        f"{where}.card_revision is {recorded} and the pinned revision is {pinned}: the check "
        f"recorded here was performed on {checked_on or 'an unrecorded date'} against a "
        f"revision of {what} that this file no longer pins. Re-read the card at the pinned "
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
        document = tomllib.loads(raw.decode("utf-8"))
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
    verified_on = reader.matching(
        reader.string(meta, "verified_on", "meta"),
        _DATE,
        "meta.verified_on",
        "an ISO date (YYYY-MM-DD)",
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

    _note_duplicates(reader, "baseline", [b.key for b in baselines], "key")
    _note_duplicates(reader, "attack_dataset", [d.key for d in attack_datasets], "key")
    # Across sections, not within them. Hugging Face keeps models and datasets in separate
    # namespaces, so one id can legally name both -- and `verify_revisions` reports its
    # resolutions by repository id, where two artifacts sharing one would collapse into one
    # line and one of the two pins would go unreported while looking verified.
    _note_duplicates(
        reader,
        "baseline/attack_dataset",
        [b.repository for b in baselines] + [d.repository for d in attack_datasets],
        "repository",
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

    if problems:
        raise PinsFileInvalid(*problems)

    _check_baseline_set(baselines)
    _check_lineage(baselines, attack_datasets)

    return Pins(
        schema_version=schema_version,
        verified_on=verified_on,
        verified_against=verified_against,
        baselines=baselines,
        attack_datasets=attack_datasets,
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

Resolver = Callable[[RemoteArtifact], str | None]
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
) -> str | None:
    """The pinned sha if this machine already holds that snapshot, else `None`.

    The hub names a snapshot directory by the commit it was fetched at, so the directory's
    existence *is* the resolution. This is what keeps a reproduction run offline after its first
    fetch.
    """
    return artifact.revision if artifact.snapshot_dir(cache_root).is_dir() else None


def resolve_over_http(artifact: RemoteArtifact) -> str | None:
    """Ask the hub what the pinned revision resolves to. The first fetch, and the smoke job.

    Any failure to get an answer is reported as `None` rather than raised: "the pin could not be
    resolved" is one of the two outcomes `verify_revisions` aborts on, and turning a network
    error into an unclassified crash would lose the exit code that tells CI which one happened.
    """
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        artifact.api_url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

    resolved = payload.get("sha") if isinstance(payload, dict) else None
    return resolved if isinstance(resolved, str) else None


def resolve_from_cache_then_hub(artifact: RemoteArtifact) -> str | None:
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
        resolved = resolve(artifact)
        if resolved is None:
            problems.append(
                f"{artifact} could not be resolved: it is not in this machine's Hugging "
                f"Face cache, and the hub returned no commit for it. Either the revision is "
                f"gone or the hub could not be reached; the two are different diagnoses and "
                f"this abort cannot tell them apart"
            )
            continue
        if resolved != artifact.revision:
            problems.append(
                f"{artifact.kind} {artifact.repository} is pinned at {artifact.revision} and "
                f"now resolves to {resolved}"
            )
            continue
        resolved_by_artifact[artifact.repository] = resolved

    if problems:
        raise PinMismatch(*problems)

    return resolved_by_artifact


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
