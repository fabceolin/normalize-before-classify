"""Where the corpus lives, what identifies it, and the one door anything reads it through.

Three jobs, and they are one module because they are one question -- *is this the corpus the
current declaration describes?*

1. **`build_id`.** FR5.1 requires an identity over the **whole** build declaration: the attack
   draw, the benign frame, `CHAINS`, `HELDOUT_CHAINS` and the confirmatory cell. The reason is
   stated in the requirement itself: a `frame_id` guarding only the benign half would let an edit
   to the attack sample size publish a table computed over the previous corpus with every check
   green. The exclusion declaration is in there too, through
   `exclusion.declaration_digest`, because which training sources are pinned decides which rows
   survive, and a corpus filtered against a different set is a different corpus.

2. **`data/manifest.json`.** Written beside the corpus by the one writer, carrying `frame_id`,
   `build_id`, the content hash and row count of every corpus file, and the two draw reports.

3. **The guarded read.** `read_corpus` refuses to hand back a single row when the recorded
   `frame_id` differs from `pins.toml`'s, when the recomputed `build_id` differs from the recorded
   one, or when a corpus file's bytes no longer hash to what the manifest says. FR5.1's "the
   entrypoint refuses to measure when the recorded `frame_id` differs" is enforced here rather than
   in the entrypoint, and deliberately: an entrypoint check is one caller's discipline, while a
   reader that verifies is the only way in. `tests/corpus/test_manifest.py` holds that with an AST
   scan -- no other module under `src/` may name a corpus filename or reach a corpus path.

**There is no timestamp in the manifest, and that is not an omission.** Two builds of the same pins
must produce identical bytes, in `data/*.jsonl` and here; a build time would make every rebuild a
diff and would make "the corpus changed" indistinguishable from "the corpus was rebuilt". When the
build ran is the git history's question.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from nbc.corpus.exclusion import declaration_digest
from nbc.corpus.matrix import CHAINS, HELDOUT_CHAINS, render_chain
from nbc.errors import NbcError
from nbc.pins import Pins
from nbc.schema import ATTACK, BENIGN, BENIGN_CLASSES, FAMILY_ATTACK, CorpusItem

__all__ = [
    "ATTACK_CORPUS_FILENAME",
    "BENIGN_CORPUS_FILENAME",
    "BUILD_ID_VERSION",
    "CORPUS_FILENAMES",
    "DATA_DIRNAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "CorpusFile",
    "CorpusManifestMismatch",
    "Manifest",
    "build_id",
    "confirmatory_cell_problems",
    "content_hash",
    "corpus_directory",
    "files_for",
    "manifest_path",
    "parse",
    "read_corpus",
    "render",
]


class CorpusManifestMismatch(NbcError, exit_code=22):
    """The committed corpus is not the corpus the current declaration describes.

    Code 22 because 3 through 21 are taken. An abort rather than a warning, and it is the whole
    content of FR5.1's last clause: a frame that can be edited while the corpus it drew is reused
    is not a frame, it is a comment. The same applies to the build declaration one level up -- an
    attack sample size edited after the corpus was built would otherwise publish a table computed
    over the previous corpus.

    Four inputs produce it:

    - the manifest records a `frame_id` that is not the one `pins.toml` declares;
    - the recomputed `build_id` is not the recorded one;
    - a corpus file's bytes do not hash to the recorded digest, or its row count differs;
    - a corpus file the manifest names is missing, or the manifest itself is.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("CorpusManifestMismatch must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "the committed corpus is not the one this declaration describes:\n  - "
            + "\n  - ".join(problems)
        )


DATA_DIRNAME: Final[str] = "data"
"""Where the committed corpus lives, relative to the repository root. Named once, here."""

ATTACK_CORPUS_FILENAME: Final[str] = "attack.jsonl"
BENIGN_CORPUS_FILENAME: Final[str] = "benign.jsonl"
MANIFEST_FILENAME: Final[str] = "manifest.json"

CORPUS_FILENAMES: Final[tuple[str, ...]] = (
    ATTACK_CORPUS_FILENAME,
    BENIGN_CORPUS_FILENAME,
)
"""The corpus files, in a declared order. The manifest records one entry per name in this tuple.

Two files rather than one, because the two halves are drawn by different rules from different
sources and a reader opening `data/` should be able to read either without the other. They share
one manifest because they share one `build_id`: a corpus is both halves or it is not a corpus.
"""

MANIFEST_SCHEMA_VERSION: Final[int] = 1
BUILD_ID_VERSION: Final[int] = 1
"""What the `build_id` payload's shape is. Part of the payload, so changing the shape changes the id.

Without it, a future component added to the payload would produce a new `build_id` that a reader
could not distinguish from a changed declaration -- and, worse, a component *removed* could
reproduce an old id over a different declaration.
"""


def corpus_directory(root: str | Path | None = None) -> Path:
    """The directory `data/*.jsonl` and the manifest live in.

    Takes a root so a test can build into `tmp_path`: CI refuses a dirty tree, and a builder that
    could only ever write into the checkout would make every end-to-end test of it a violation.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    return base / DATA_DIRNAME


def manifest_path(root: str | Path | None = None) -> Path:
    return corpus_directory(root) / MANIFEST_FILENAME


def content_hash(payload: bytes) -> str:
    """The SHA-256 of a corpus file's exact bytes, as the manifest records it."""
    return hashlib.sha256(payload).hexdigest()


def build_id(pins: Pins) -> str:
    """The identity of the whole build declaration. FR5.1's `build_id`.

    Five components, four of them named by the requirement, the fifth (the confirmatory cell)
    covered through the frame it is declared inside, and the last by the mechanism:

    - the **attack draw**, through each pinned dataset's full run fields, so the pool's identity
      travels with the draw rather than only its size and seed;
    - the **benign frame**, whole, exactly as `frame_id` covers it;
    - **`CHAINS`** and **`HELDOUT_CHAINS`**, as rendered chain names per corpus class, because the
      dressing axis of the table *is* those constants and a chain added to one of them is a column
      the previous corpus does not have;
    - the **confirmatory cell**, which the requirement names on its own and which this payload
      covers **through the benign frame it is declared inside**. It had its own key here until a
      mutation test showed what that key was worth: deleting it changed no id, because the cell was
      already in the frame, so the test that was supposed to prove the coverage could not fail. One
      appearance and a test that moves the cell and watches the id move is the honest version;
      `tests/corpus/test_manifest.py` is where that test lives, and it also fails the day somebody
      moves the cell out of the frame block;
    - the **exclusion declaration**, through `exclusion.declaration_digest`, which is a hash of what
      was declared and never of what was observed -- so two runs of the same pins agree even though
      one fetched over the network and the other read a cache.

    Sorted keys and a fixed separator, so the id is a function of the values and not of the order a
    dictionary happened to be built in.
    """
    frame = pins.benign_frame
    payload: dict[str, Any] = {
        "version": BUILD_ID_VERSION,
        "attack_draw": [dataset.as_run_fields() for dataset in pins.attack_datasets],
        "benign_frame": frame.as_run_fields(),
        "chains": {
            corpus_class: [render_chain(chain) for chain in declared]
            for corpus_class, declared in sorted(CHAINS.items())
        },
        "heldout_chains": {
            corpus_class: [render_chain(chain) for chain in declared]
            for corpus_class, declared in sorted(HELDOUT_CHAINS.items())
        },
        "exclusion_declaration": declaration_digest(pins),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def confirmatory_cell_problems(pins: Pins) -> tuple[str, ...]:
    """Every reason the declared confirmatory cell does not name a cell of the table.

    `pins.py` is a leaf over `nbc.errors` and cannot import the corpus vocabulary, so it checks the
    cell's shape and that its `baseline` is a declared baseline key, and stops there. This is the
    other half, in the module that may read `nbc.schema` and `corpus/matrix.py`: the class has to be
    a benign class the table has a column for, and the chain has to be one the corpus is actually
    built in. A cell naming a chain no registry declares is a verdict with no data behind it, and it
    would look exactly like one with data until the run reached it.

    **Story 3.9 adds the requirement this does not check**: that the chain must be held out or
    nested past the recursion ceiling and never a bound one, with the reason. That is deliberately
    not here -- FR5.4 splits declaration and hashing into the corpus epic and the assertion into the
    story that owns it, and writing half of 3.9 here would leave a check nobody can find.
    """
    cell = pins.benign_frame.confirmatory_cell
    problems: list[str] = []
    if cell.benign_class not in BENIGN_CLASSES:
        problems.append(
            f"the confirmatory cell names benign class {cell.benign_class!r}, which is not one of "
            f"{list(BENIGN_CLASSES)}; the verdict rests on this cell and the table has no row for it"
        )
        return tuple(problems)

    declared = {
        render_chain(chain)
        for registry in (CHAINS, HELDOUT_CHAINS)
        for chain in registry.get(cell.benign_class, ())
    }
    if cell.dressing_chain not in declared:
        problems.append(
            f"the confirmatory cell names dressing chain {cell.dressing_chain!r}, which is not "
            f"declared for {cell.benign_class} in either registry ({sorted(declared)}); the corpus "
            f"would carry no row in that cell and the verdict would be computed over nothing"
        )
    return tuple(problems)


@dataclass(frozen=True, slots=True)
class CorpusFile:
    """One corpus file, by name, content hash and row count.

    The row count is recorded **beside** the hash and compared **against** the file: a digest tells
    a reader the bytes are unchanged, and a row count tells them how many rows those bytes are, and
    a manifest carrying one without the other leaves half the question open.
    """

    name: str
    sha256: str
    rows: int
    bytes: int

    def as_json_object(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "rows": self.rows,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """`data/manifest.json`: the corpus' identity and the accounting for how it was drawn."""

    schema_version: int
    frame_id: str
    build_id: str
    files: tuple[CorpusFile, ...]
    reports: Mapping[str, object]

    def as_json_object(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "build_id": self.build_id,
            "files": [entry.as_json_object() for entry in self.files],
            "reports": dict(self.reports),
        }


def render(manifest: Manifest) -> str:
    """The exact bytes of `data/manifest.json`: sorted keys, two-space indent, one trailing newline.

    Rendered here and written by `corpus/build.py`, for the same reason `attack.serialize` renders
    and does not write: keeping the serialization pure is what lets every claim about it be tested
    offline, and keeping the write in one module is what makes the writer countable.
    """
    return json.dumps(manifest.as_json_object(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse(text: str) -> Manifest:
    """Read a manifest back. Raises `CorpusManifestMismatch` on anything that is not one."""
    try:
        document = json.loads(text)
    except ValueError as error:
        # `json.JSONDecodeError` and `UnicodeDecodeError` are both `ValueError`; catching the
        # narrow one would let a manifest written in another encoding escape as an unclassified
        # crash rather than as the refusal it is.
        raise CorpusManifestMismatch(f"{MANIFEST_FILENAME} is not valid JSON: {error}") from None
    if not isinstance(document, dict):
        raise CorpusManifestMismatch(
            f"{MANIFEST_FILENAME} holds {type(document).__name__}, not an object"
        )

    problems: list[str] = []
    version = document.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        problems.append(
            f"{MANIFEST_FILENAME} declares schema_version {version!r} and this reader reads "
            f"{MANIFEST_SCHEMA_VERSION}; a manifest from another shape is not one it may guess at"
        )
    files: list[CorpusFile] = []
    entries = document.get("files")
    if not isinstance(entries, list):
        problems.append(f"{MANIFEST_FILENAME}.files is not a list")
    else:
        for entry in entries:
            if not isinstance(entry, dict):
                problems.append(f"{MANIFEST_FILENAME}.files holds a non-object entry")
                continue
            try:
                files.append(
                    CorpusFile(
                        name=str(entry["name"]),
                        sha256=str(entry["sha256"]),
                        rows=int(entry["rows"]),
                        bytes=int(entry["bytes"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                problems.append(f"{MANIFEST_FILENAME}.files holds an unreadable entry: {error}")

    if problems:
        raise CorpusManifestMismatch(*problems)

    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        frame_id=str(document.get("frame_id", "")),
        build_id=str(document.get("build_id", "")),
        files=tuple(files),
        reports=document.get("reports", {}) if isinstance(document.get("reports"), dict) else {},
    )


def read_corpus(
    pins: Pins, root: str | Path | None = None
) -> tuple[Manifest, tuple[CorpusItem, ...]]:
    """The only way into `data/*.jsonl`, and it verifies before it returns a row.

    Every problem is collected before aborting, so a stale corpus tells a reader all of what is
    stale rather than one thing at a time.
    """
    directory = corpus_directory(root)
    path = directory / MANIFEST_FILENAME
    try:
        manifest = parse(path.read_bytes().decode("utf-8"))
    except ValueError as error:
        # `UnicodeDecodeError` is a `ValueError`. Catching only `OSError` below would let a
        # manifest written in another encoding escape as an unclassified crash rather than as the
        # refusal it is -- the sibling exception this project has already missed once.
        raise CorpusManifestMismatch(f"{path} is not valid UTF-8: {error}") from None
    except FileNotFoundError:
        raise CorpusManifestMismatch(
            f"no {MANIFEST_FILENAME} at {path}; the corpus is what a table is computed over and an "
            f"unidentified corpus is one nobody can say was drawn under the declared frame"
        ) from None
    except OSError as error:
        raise CorpusManifestMismatch(f"{path} could not be read: {error}") from None

    problems: list[str] = list(confirmatory_cell_problems(pins))
    declared_frame = pins.benign_frame.frame_id
    if manifest.frame_id != declared_frame:
        problems.append(
            f"{MANIFEST_FILENAME} records frame_id {manifest.frame_id!r} and pins.toml declares "
            f"{declared_frame!r}. The frame was re-declared after this corpus was drawn, so the "
            f"corpus is not a sample from the frame the run would publish. Rebuild it"
        )
    expected_build = build_id(pins)
    if manifest.build_id != expected_build:
        problems.append(
            f"{MANIFEST_FILENAME} records build_id {manifest.build_id!r} and the current "
            f"declaration computes {expected_build!r}. Something in the attack draw, the benign "
            f"frame, the dressing registries, the confirmatory cell or the exclusion set changed "
            f"after this corpus was built"
        )

    recorded = {entry.name: entry for entry in manifest.files}
    missing = [name for name in CORPUS_FILENAMES if name not in recorded]
    if missing:
        problems.append(
            f"{MANIFEST_FILENAME} records no entry for {missing}; a corpus is both halves or it is "
            f"not a corpus"
        )
    extra = sorted(set(recorded) - set(CORPUS_FILENAMES))
    if extra:
        problems.append(f"{MANIFEST_FILENAME} records {extra}, which this build does not write")

    items: list[CorpusItem] = []
    for name in CORPUS_FILENAMES:
        entry = recorded.get(name)
        if entry is None:
            continue
        try:
            payload = (directory / name).read_bytes()
        except FileNotFoundError:
            problems.append(f"{MANIFEST_FILENAME} names {name}, which is not in {directory}")
            continue
        except OSError as error:
            problems.append(f"{directory / name} could not be read: {error}")
            continue
        digest = content_hash(payload)
        if digest != entry.sha256:
            problems.append(
                f"{name} hashes to {digest} and the manifest records {entry.sha256}; the file was "
                f"edited after the build, so the rows a table would be computed over are not the "
                f"rows the accounting describes"
            )
            continue
        if len(payload) != entry.bytes:
            problems.append(
                f"{name} is {len(payload)} bytes and the manifest records {entry.bytes}"
            )
        rows = _rows_of(name, payload, problems)
        if len(rows) != entry.rows:
            problems.append(
                f"{name} holds {len(rows)} rows and the manifest records {entry.rows}"
            )
        items.extend(rows)

    if problems:
        raise CorpusManifestMismatch(*problems)
    return manifest, tuple(items)


def _rows_of(name: str, payload: bytes, problems: list[str]) -> tuple[CorpusItem, ...]:
    try:
        text = payload.decode("utf-8")
    except ValueError as error:
        # `UnicodeDecodeError` is a `ValueError`, and catching only the narrow name is the sibling
        # this project has already missed once.
        problems.append(f"{name} is not valid UTF-8: {error}")
        return ()

    rows: list[CorpusItem] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
            # The label is **derived from the family and compared** against the one on disk, never
            # copied off it. Two things fall out of that, and both are the point. FR4's rule that a
            # gold label names a schema constant survives the round trip, which the AST scan in
            # `tests/corpus/test_build.py` enforces over this very call. And a row whose recorded
            # label disagrees with its family is refused rather than loaded -- the file passed its
            # content hash, so such a row could only come from a build that wrote it, and a corpus
            # that labels an attack row benign is the one defect that makes every rate wrong in the
            # direction nobody would question.
            attack = record["family"] == FAMILY_ATTACK
            expected = ATTACK if attack else BENIGN
            if record["label"] != expected:
                problems.append(
                    f"{name}:{number} is family {record['family']!r} and carries label "
                    f"{record['label']!r}; the gold label of that family is {expected}"
                )
                continue
            # The constructor is written twice, differing only in the label, because the scan
            # requires the argument to NAME a schema constant and `label=expected` names a local.
            # The rule is right and this is what obeying it costs: a reader following the name
            # arrives at `schema.py` rather than at whatever the local was last assigned.
            fields = {
                "id": record["id"],
                "source": record["source"],
                "family": record["family"],
                "benign_class": record["benign_class"],
                "dressing": tuple(record["dressing"]),
                "text": record["text"],
            }
            rows.append(
                CorpusItem(**fields, label=ATTACK)
                if attack
                else CorpusItem(**fields, label=BENIGN)
            )
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"{name}:{number} is not a corpus row: {error}")
    return tuple(rows)


def files_for(payloads: Sequence[tuple[str, bytes, int]]) -> tuple[CorpusFile, ...]:
    """`(name, bytes, rows)` triples to `CorpusFile` records, hashed as they are recorded.

    The digest is taken from the **same bytes the writer wrote**, in one place, so the manifest can
    never record a hash of something other than the file beside it.
    """
    return tuple(
        CorpusFile(name=name, sha256=content_hash(payload), rows=rows, bytes=len(payload))
        for name, payload, rows in payloads
    )
