"""The vendored confusable mapping: its contract, its loader, and its abort.

This module owns the contract; `vendor_confusables.py` derives an artifact that satisfies it.
That direction matters. If the derivation owned the rules, the committed file would define what
"valid" means and the loader would only be able to agree with it.

**Direction and domain are the load-bearing part.** Keys are single **non-ASCII** code points
inside a declared, closed set of Cyrillic and Greek blocks; values are strings whose every code
point is **ASCII**. The mapping is therefore the identity on all of `U+0000..U+007F`, and a test
asserts that over the whole range rather than inferring it from the key rule.

The alternative — a full UTS-39 skeleton transform — is wrong here, and specifically wrong:
upstream row `0031 ; 006C ; MA # ( 1 -> l ) DIGIT ONE -> LATIN SMALL LETTER L` folds ASCII `1`
to `l`, and `004F`/`0030` fold `O` and `0` together. Applied to this experiment that would turn
the benign-code counter-metric into a number about ASCII folding, and it would corrupt base64 and
hex runs before the decode stage ever saw them.

**The revision is pinned to the interpreter, not to a date.** The artifact's filename carries the
Unicode revision and `load()` refuses unless it equals `unicodedata.unidata_version`. Step 2 of
the pipeline (this mapping) and step 3 (NFKC, which is the interpreter's own tables) would
otherwise disagree about the same character, silently, for as long as nobody looked. A Python
minor bump moves `unidata_version` — 3.11 carries UCD 14.0.0, 3.12 carries 15.0.0, 3.13 carries
15.1.0, 3.14 carries 16.0.0 — so a bump means re-vendoring and a full re-run, not a test update.

**No caching.** `load()` reads and validates on every call and holds no module-level state; a
caller that needs the table in a hot path holds the returned value. A module-level cache would be
mutable state inside a layer whose determinism is one of its claims.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from nbc.errors import NbcError

__all__ = (
    "ARTIFACT_PREFIX",
    "ARTIFACT_SUFFIX",
    "ASCII_LAST",
    "Block",
    "ConfusablesTable",
    "ConfusablesTableInvalid",
    "DATA_DIR",
    "REVISION_REASON",
    "SCOPED_BLOCKS",
    "ScopedSource",
    "artifact_filename",
    "declared_blocks",
    "discover_revision",
    "in_scope",
    "load",
    "revision_pattern",
)


class ConfusablesTableInvalid(NbcError, exit_code=12):
    """The vendored confusable mapping is absent, ambiguous, or not what it declares itself to be.

    An abort rather than a warning: every one of these means the layer would map characters by
    a table nobody pinned, or by a table that disagrees with the interpreter's own NFKC — and
    both change the meaning of every number the run publishes.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("ConfusablesTableInvalid must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__("; ".join(problems))


# --- the contract ---------------------------------------------------------------------------

ASCII_LAST: Final[int] = 0x7F
"""The last code point the mapping must leave alone. `U+0000..U+007F` inclusive."""


@dataclass(frozen=True, slots=True)
class Block:
    """One Unicode block, by name and inclusive code-point range.

    A block range is a **closed, enumerable vocabulary** that a reviewer audits by reading two
    integers. The alternative on offer in the standard library is
    `unicodedata.name(ch).startswith("CYRILLIC ")`, which is a textual pattern standing in for a
    structural fact — the same substitution that let `"Tokenizer(" in "WindowedTokenizer("` pass
    for identity in Epic 1. There is no script property in the standard library, so the ranges
    are vendored here rather than derived at runtime.
    """

    name: str
    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first > self.last:
            raise ValueError(f"block {self.name} has first {self.first:04X} > last {self.last:04X}")
        if self.first <= ASCII_LAST:
            raise ValueError(
                f"block {self.name} starts at {self.first:04X}, inside ASCII; a scoped block "
                f"whose range reaches ASCII would admit an ASCII key"
            )

    def contains(self, code_point: int) -> bool:
        return self.first <= code_point <= self.last

    def as_declared(self) -> list[str]:
        """The block as it appears in the vendored payload, so the two can be compared."""
        return [self.name, f"{self.first:04X}", f"{self.last:04X}"]


SCOPED_BLOCKS: Final[tuple[Block, ...]] = (
    # Greek. "Greek and Coptic" also holds Coptic letters; they are admitted by the block rule
    # and simply never appear, because none of them has an all-ASCII prototype upstream. Naming
    # the block honestly and letting the data be empty there is better than a second, undeclared
    # filter that removes characters nobody listed.
    Block("Greek and Coptic", 0x0370, 0x03FF),
    Block("Greek Extended", 0x1F00, 0x1FFF),
    # Cyrillic, all seven blocks that exist at UCD 15.1.0.
    Block("Cyrillic", 0x0400, 0x04FF),
    Block("Cyrillic Supplement", 0x0500, 0x052F),
    Block("Cyrillic Extended-C", 0x1C80, 0x1C8F),
    Block("Cyrillic Extended-A", 0x2DE0, 0x2DFF),
    Block("Cyrillic Extended-B", 0xA640, 0xA69F),
    Block("Cyrillic Extended-D", 0x1E030, 0x1E08F),
)
"""The declared domain of the keys: Cyrillic and Greek, and nothing else.

The corpus's homoglyph dressing is built from these two scripts (Epic 3 draws its characters from
this same artifact rather than from a second table), and every block here is disjoint from ASCII
by `Block.__post_init__`, which is what makes the identity on `U+0000..U+007F` structural rather
than a property of the data that happened to be derived.
"""

REVISION_REASON: Final[str] = (
    "The vendored Unicode revision must equal unicodedata.unidata_version, which is the revision "
    "the interpreter's own NFKC tables were built from. Step 2 of the pipeline is this mapping and "
    "step 3 is NFKC; at two different revisions they can disagree about the same character with no "
    "symptom other than a number that moved. The interpreter pin in nbc/platform.py exists for this "
    "reason and for no other: the onnxruntime wheels would admit CPython 3.11 through 3.14, and UCD "
    "moves with the minor version (3.11=14.0.0, 3.12=15.0.0, 3.13=15.1.0, 3.14=16.0.0). Widening "
    "that range means re-vendoring this table at the new revision and re-running everything, not "
    "editing a test."
)

DATA_DIR: Final[Path] = Path(__file__).resolve().parent / "data"
ARTIFACT_PREFIX: Final[str] = "confusables-"
ARTIFACT_SUFFIX: Final[str] = ".json"

_REVISION: Final[re.Pattern[str]] = re.compile(r"\A[0-9]+\.[0-9]+\.[0-9]+\Z")


def revision_pattern() -> re.Pattern[str]:
    """The shape of a Unicode revision, as both the filename and the payload must spell it."""
    return _REVISION


def artifact_filename(revision: str) -> str:
    return f"{ARTIFACT_PREFIX}{revision}{ARTIFACT_SUFFIX}"


def declared_blocks() -> list[list[str]]:
    """`SCOPED_BLOCKS` in the form the payload declares, for comparison against it."""
    return [block.as_declared() for block in SCOPED_BLOCKS]


def in_scope(code_point: int) -> bool:
    """True when `code_point` may be a key: non-ASCII and inside a declared block."""
    if code_point <= ASCII_LAST:
        return False
    return any(block.contains(code_point) for block in SCOPED_BLOCKS)


@dataclass(frozen=True, slots=True)
class ScopedSource:
    """Where the artifact came from, recorded so the derivation can be checked against it.

    `sha256` and `byte_count` are not decoration and are not taken on trust: the `smoke` tier
    fetches `url` and compares both, then re-derives the mapping and compares that too. Offline
    they are inert, which is why the comparison lives where the network is allowed rather than
    being left as a field nobody reads.
    """

    url: str
    sha256: str
    byte_count: int
    notice: str

    def as_run_fields(self) -> dict[str, object]:
        return {
            "url": self.url,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "notice": self.notice,
        }


@dataclass(frozen=True, slots=True)
class ConfusablesTable:
    """The loaded artifact: a validated per-code-point mapping and its provenance."""

    revision: str
    rule_version: int
    source: ScopedSource
    mapping: Mapping[str, str]
    translate_table: Mapping[int, str]

    def as_run_fields(self) -> dict[str, object]:
        """The table as plain data for the `run` block of `results.json`.

        Published here and consumed by the entrypoint that writes `results.json`, which does not
        exist yet. Until it does, this is an accessor with no caller — stated rather than
        implied, because a published obligation nobody discharges is how the `pins.toml`
        exemption went unenforced through eight stories of Epic 1.
        """
        return {
            "unicode_revision": self.revision,
            "derivation_rule_version": self.rule_version,
            "entry_count": len(self.mapping),
            "scoped_blocks": declared_blocks(),
            "source": self.source.as_run_fields(),
            "reason": REVISION_REASON,
        }


# --- loading --------------------------------------------------------------------------------


def discover_revision(data_dir: Path = DATA_DIR) -> str:
    """The revision in the artifact's filename, read from the filename and nothing else.

    Deliberately does **not** compare against `unicodedata.unidata_version`. A test that wants to
    check the filename against the interpreter needs one side that has not already been checked
    against the other; `load()` performs the comparison and would hand back only agreement.
    """
    if not data_dir.is_dir():
        raise ConfusablesTableInvalid(f"{data_dir} is not a directory")

    candidates = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.name.startswith(ARTIFACT_PREFIX)
        and path.name.endswith(ARTIFACT_SUFFIX)
    )
    if not candidates:
        raise ConfusablesTableInvalid(
            f"no vendored confusables artifact in {data_dir}: expected exactly one "
            f"{ARTIFACT_PREFIX}<revision>{ARTIFACT_SUFFIX}. {REVISION_REASON}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ConfusablesTableInvalid(
            f"{len(candidates)} vendored confusables artifacts in {data_dir} ({names}); exactly "
            f"one may exist, or which table the layer used is decided by a sort order"
        )

    revision = candidates[0].name[len(ARTIFACT_PREFIX) : -len(ARTIFACT_SUFFIX)]
    if not _REVISION.match(revision):
        raise ConfusablesTableInvalid(
            f"{candidates[0].name} does not carry a Unicode revision of the form N.N.N"
        )
    return revision


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """`json.load` keeps the last of a repeated key. A repeated key is a defect, not a preference."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ConfusablesTableInvalid(
                f"the vendored artifact repeats the key {key!r}; JSON would silently keep the "
                f"last one and the mapping would depend on file order"
            )
        seen[key] = value
    return seen


def _describe(text: str) -> str:
    """A code point sequence named by its code points, so a message is readable in a terminal."""
    return "+".join(f"U+{ord(ch):04X}" for ch in text) or "<empty>"


def load(data_dir: Path = DATA_DIR) -> ConfusablesTable:
    """Read, validate and return the vendored mapping, or abort naming every problem at once.

    Every rule enforced here has a test that supplies the input which makes it fail. A rule
    whose failing input nobody can construct is not a check.
    """
    revision = discover_revision(data_dir)
    path = data_dir / artifact_filename(revision)

    interpreter_revision = unicodedata.unidata_version
    if revision != interpreter_revision:
        raise ConfusablesTableInvalid(
            f"{path.name} is vendored at Unicode {revision}; this interpreter's unicodedata is "
            f"{interpreter_revision}. Re-vendor with "
            f"`python -m nbc.canon.vendor_confusables --write` and re-run the whole measurement; "
            f"do not edit a test. {REVISION_REASON}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except UnicodeDecodeError as error:
        # A ValueError, not an OSError: the sibling that a narrow `except OSError` misses.
        raise ConfusablesTableInvalid(f"{path.name} is not valid UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfusablesTableInvalid(f"{path.name} is not valid JSON: {error}") from error
    except OSError as error:
        raise ConfusablesTableInvalid(f"{path.name} could not be read: {error}") from error

    if not isinstance(payload, dict):
        raise ConfusablesTableInvalid(
            f"{path.name} holds a {type(payload).__name__}; the artifact is a JSON object"
        )

    problems: list[str] = []

    def require(
        key: str, kind: type, where: Mapping[str, Any] = payload, under: str = ""
    ) -> Any:
        location = f"{path.name}{under}"
        value = where.get(key)
        if value is None:
            problems.append(f"{location} declares no {key!r}")
            return None
        if isinstance(value, bool) or not isinstance(value, kind):
            problems.append(
                f"{location} declares {key!r} as a {type(value).__name__}, expected "
                f"{kind.__name__}"
            )
            return None
        return value

    declared_revision = require("unicode_revision", str)
    if declared_revision is not None and declared_revision != revision:
        problems.append(
            f"{path.name} declares unicode_revision {declared_revision!r} inside a file named "
            f"for {revision!r}; the filename and the payload must be the same revision or the "
            f"one a reader checks is not the one the loader used"
        )

    derivation = require("derivation", dict)
    rule_version = 0
    if derivation is not None:
        maybe_rule = require("rule_version", int, derivation, " derivation")
        if maybe_rule is not None:
            rule_version = maybe_rule
        blocks = derivation.get("scoped_blocks")
        if blocks != declared_blocks():
            problems.append(
                f"{path.name} declares a scoped_blocks set that is not the one "
                f"nbc.canon.confusables_table.SCOPED_BLOCKS declares; the artifact was derived "
                f"under a different domain than the loader enforces"
            )

    source_block = require("source", dict)
    source = None
    if source_block is not None:
        url = require("url", str, source_block, " source")
        sha256 = require("sha256", str, source_block, " source")
        byte_count = require("bytes", int, source_block, " source")
        notice = require("notice", str, source_block, " source")
        if None not in (url, sha256, byte_count, notice):
            malformed = False
            if not re.fullmatch(r"[0-9a-f]{64}", str(sha256)):
                problems.append(f"{path.name} declares a source sha256 that is not 64 hex digits")
                malformed = True
            if int(byte_count) <= 0:
                problems.append(
                    f"{path.name} declares source bytes {byte_count}, which is not a size"
                )
                malformed = True
            if not str(notice).strip():
                problems.append(
                    f"{path.name} declares an empty source notice; the upstream copyright and "
                    f"terms-of-use lines travel with the data derived from it"
                )
                malformed = True
            if not malformed:
                source = ScopedSource(
                    url=str(url), sha256=str(sha256), byte_count=int(byte_count), notice=str(notice)
                )

    raw_mapping = require("mapping", dict)
    mapping: dict[str, str] = {}
    if raw_mapping is not None:
        for key, value in raw_mapping.items():
            if not isinstance(value, str):
                problems.append(
                    f"{path.name} maps {_describe(key)} to a {type(value).__name__}, not a string"
                )
                continue
            if len(key) != 1:
                problems.append(
                    f"{path.name} has the key {_describe(key)}, which is {len(key)} code points; "
                    f"the mapping is applied per code point"
                )
                continue
            code_point = ord(key)
            if code_point <= ASCII_LAST:
                problems.append(
                    f"{path.name} has the ASCII key U+{code_point:04X}; the mapping must be the "
                    f"identity on U+0000..U+{ASCII_LAST:04X}, or the layer folds ordinary source "
                    f"code and corrupts base64 runs before the decode stage sees them"
                )
                continue
            if not in_scope(code_point):
                problems.append(
                    f"{path.name} has the key U+{code_point:04X}, which is in no declared Cyrillic "
                    f"or Greek block"
                )
                continue
            if not value:
                problems.append(f"{path.name} maps U+{code_point:04X} to the empty string")
                continue
            outside = [ch for ch in value if ord(ch) > ASCII_LAST]
            if outside:
                problems.append(
                    f"{path.name} maps U+{code_point:04X} to {_describe(value)}, which leaves "
                    f"ASCII; every value is an ASCII string"
                )
                continue
            mapping[key] = value

    entry_count = require("entry_count", int)
    if entry_count is not None and raw_mapping is not None and entry_count != len(raw_mapping):
        problems.append(
            f"{path.name} declares entry_count {entry_count} over a mapping of "
            f"{len(raw_mapping)}; the count is the evidence for the mapping, so it is compared "
            f"to it"
        )

    if raw_mapping is not None and not raw_mapping:
        problems.append(f"{path.name} carries an empty mapping")

    if problems:
        raise ConfusablesTableInvalid(*problems)

    if source is None:  # unreachable: every path that leaves it None appends a problem
        raise ConfusablesTableInvalid(f"{path.name} carries no usable source block")
    return ConfusablesTable(
        revision=revision,
        rule_version=rule_version,
        source=source,
        mapping=MappingProxyType(dict(mapping)),
        translate_table=MappingProxyType({ord(key): value for key, value in mapping.items()}),
    )
