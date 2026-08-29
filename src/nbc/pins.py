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

Three aborts, three codes, because the remedies differ:

- `PinsFileInvalid` (4) -- the file is missing, unparseable, or says something it may not say.
  The author has to fix the repository.
- `BaselineSetInvalid` (5) -- the file is well formed and the set it declares violates SC5:
  fewer than two baselines, or two that share an (architecture, tokenizer) family pair. The pins
  have to change, and a baseline is *replaced*, never removed.
- `PinMismatch` (6) -- the file is right and the world moved. Nothing in the repository is
  wrong; the run must not proceed.

Structure and the baseline set are checked by `load_pins()`, so every consumer gets them for
free and the entrypoint cannot forget. `verify_revisions()` is the separate step that asks the
world, and it is the one that touches a network.

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
from typing import Any, Callable, Final, Mapping, Sequence

from nbc.errors import NbcError

__all__ = [
    "AttackDataset",
    "Baseline",
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
    "RemoteArtifact",
    "Resolver",
    "SCHEMA_VERSION",
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
"""What a licence field says when the publisher declares none.

Not `None` and not an empty string: an absent licence is a *finding*, and it has to survive into
`results.json` as one rather than as a missing key a reader can mistake for an oversight.
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
    """A baseline's declared relationship to every pinned attack dataset, and when it was read.

    The date and the card revision are part of the record: a lineage check that was never re-run
    after a pin changed has to be *visible*, not silently assumed to still hold.
    """

    checked_on: str
    card_revision: str
    attack_datasets: Mapping[str, str]

    def relationship_to(self, repository: str) -> str | None:
        return self.attack_datasets.get(repository)

    def as_run_fields(self) -> dict[str, object]:
        return {
            "checked_on": self.checked_on,
            "card_revision": self.card_revision,
            "attack_datasets": dict(self.attack_datasets),
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
    declared = lineage_table.get("attack_datasets")
    if declared is None:
        reader.note(f"{where}.lineage.attack_datasets is missing")
        declared = {}
    elif not isinstance(declared, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in declared.items()
    ):
        reader.note(f"{where}.lineage.attack_datasets must be a table of strings")
        declared = {}
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
        attack_datasets=dict(declared),
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

    return AttackDataset(
        key=key,
        repository=reader.matching(
            reader.string(table, "repository", where),
            _REPOSITORY,
            f"{where}.repository",
            "a `namespace/name` repository id",
        ),
        revision=reader.matching(
            reader.string(table, "revision", where),
            _SHA,
            f"{where}.revision",
            "a 40-character lowercase hex commit sha",
        ),
        splits=reader.strings(table, "splits", where),
        attack_label=reader.integer(table, "attack_label", where),
        licence=_read_licence(reader, table, where),
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


def load_pins(root: Path | str | None = None) -> Pins:
    """Read, validate and return the pins. Step 1 of the entrypoint's sequence.

    Structure and the baseline set are both checked here, so no consumer can load the pins and
    forget to ask whether the set they describe is one SC5 admits.
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

    if problems:
        raise PinsFileInvalid(*problems)

    _check_baseline_set(baselines)

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
    root = cache_root if cache_root is not None else hf_cache_root()
    snapshot = root / artifact.cache_directory / "snapshots" / artifact.revision
    return artifact.revision if snapshot.is_dir() else None


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
