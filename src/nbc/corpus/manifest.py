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
   one, when a corpus file's bytes no longer hash to what the manifest says, or when a corpus file
   is the Git LFS pointer a clone that never fetched the object holds. FR5.1's "the
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
from nbc.corpus.matrix import (
    CHAIN_CLASS_HELD_OUT,
    CHAINS,
    HELDOUT_CHAINS,
    chain_class,
    encoding_depth,
    parse_chain,
    render_chain,
)
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
    "ConfirmatoryCellNotFalsifiable",
    "CorpusFile",
    "CorpusManifestMismatch",
    "LFS_POINTER_VERSION",
    "Manifest",
    "build_id",
    "confirmatory_cell_falsifiability_problems",
    "confirmatory_cell_problems",
    "content_hash",
    "corpus_presence",
    "corpus_directory",
    "files_for",
    "lfs_pointer_problem",
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

    Five inputs produce it:

    - the manifest records a `frame_id` that is not the one `pins.toml` declares;
    - the recomputed `build_id` is not the recorded one;
    - a corpus file's bytes do not hash to the recorded digest, or its row count differs;
    - a corpus file the manifest names is missing, or the manifest itself is;
    - a corpus file is a **Git LFS pointer** rather than the corpus, which is what a clone made
      without the LFS filters holds. Reported as itself rather than as a hash mismatch, because
      the remedy is `git lfs pull` and the hash mismatch's message says to rebuild.
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

    **Story 3.9's assertion is not here, and it now has an address**:
    `confirmatory_cell_falsifiability_problems`, below, which refuses a cell whose chain is bound
    and inside the decode budget. It is a separate function rather than four more lines in this one
    for the reason this paragraph used to give: a check folded into a differently named gate is a
    check nobody can find. What it needs and this one does not is the recursion ceiling, which
    arrives as a parameter from a caller holding the layer's own context.
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


class ConfirmatoryCellNotFalsifiable(NbcError, exit_code=35):
    """The pre-registered N1 cell names a chain on which N1 cannot come out either way.

    Code 35 because 3 through 34 are taken. An abort rather than a warning, and the reason is that
    the alternative is worse than a wrong number: a run that reached the verdict would report N1
    `not_triggered` with every field present and every interval computed, and nothing in the output
    would say the condition had been unfalsifiable from the moment it was declared.

    **Why a bound chain inside the decode budget is refused.** Story 3.4's round-trip contract makes
    recovery on a bound chain total -- the layer undoes the dressing it was written against -- so
    `Delta recall` sits at or near its maximum by construction. `D = FPR delta - recall delta` could
    then lie wholly above zero only if the layer converted a **majority of the declared benign
    class** into fresh false positives. That is an impossibility dressed as a threshold, and a cell
    pre-registered on it would satisfy every word of FR5.4 while guaranteeing the verdict in
    advance. It would do so legibly, which is what makes it worse than an obvious error.

    **Why "held out or past the ceiling" is the admissible pair.** Those are the two halves of N4's
    generalization set and the only two places recovery is not decided in advance: a held-out chain
    names encodings the layer was never written against, and a chain nested past the recursion
    ceiling is one the layer is told to stop short of. On both, `Delta recall` is a measurement.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("ConfirmatoryCellNotFalsifiable must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "the pre-registered confirmatory cell could not have falsified anything:\n  - "
            + "\n  - ".join(problems)
        )


def confirmatory_cell_falsifiability_problems(pins: Pins, *, ceiling: int) -> tuple[str, ...]:
    """Every reason the declared confirmatory cell's chain guarantees N1's verdict in advance.

    Story 3.9. The chain must be **held out** or **nested past the recursion ceiling**, never a
    bound chain the layer will fully recover; `ConfirmatoryCellNotFalsifiable` carries the reason.

    **`ceiling` is a parameter and has no default.** `tests/canon/test_recursion.py` holds
    `DEFAULT_CEILING` to a single reader under `src/` and that rule is not weakened here: the caller
    passes the ceiling its own `CanonContext` carries, which is the ceiling the layer will actually
    apply rather than a second copy of the constant. Story 4.6 met the same problem from the other
    end and read `over_ceiling` off the run's own ceiling-hit census; at declaration time there is no
    run, and the honest source is the shipped context.

    **The depth comparison is not a claim about the layer.** `tests/corpus/test_matrix.py`
    canonicalizes every declared chain at the shipped ceiling and asserts
    `ceiling_hit is (encoding_depth(chain) > ctx.ceiling)`, so `encoding_depth > ceiling` here and
    `ceiling_hit` there are two computations of one fact that are compared rather than assumed
    equal.

    **Precondition.** The chain must be one some registry declares; `parse_chain` raises
    `CorpusMatrixInvalid` otherwise. `build.py` runs `confirmatory_cell_problems` first, which
    refuses an undeclared chain with the message that names the registries -- this function is not a
    second copy of that check and does not repeat its message.
    """
    cell = pins.benign_frame.confirmatory_cell
    links = parse_chain(cell.dressing_chain)
    if chain_class(links) == CHAIN_CLASS_HELD_OUT:
        return ()

    depth = encoding_depth(links)
    if depth > ceiling:
        return ()

    admissible = sorted(
        render_chain(chain)
        for chain in HELDOUT_CHAINS.get(cell.benign_class, ())
    ) + sorted(
        render_chain(chain)
        for chain in CHAINS.get(cell.benign_class, ())
        if encoding_depth(chain) > ceiling
    )
    # Empty is unreachable behind the shape gate, which refuses a class no registry declares before
    # this function is called, and it is empty in production for that reason rather than by luck.
    # Said in words anyway: a bare `[]` at the end of a refusal reads as "there is no way out",
    # which is a different and much worse claim than "you asked about a class this table has no
    # registry for". `tests/corpus/test_confirmatory_cell.py` reaches it with a synthetic cell.
    way_out = (
        f"{admissible}"
        if admissible
        else (
            f"no chain is admissible for {cell.benign_class!r}, because neither registry declares "
            f"any for it -- which is a problem confirmatory_cell_problems reports first and this "
            f"message is not a second copy of"
        )
    )
    return (
        f"the confirmatory cell names dressing chain {cell.dressing_chain!r}, which is a bound "
        f"chain of encoding_depth {depth} against a recursion ceiling of {ceiling}, so the layer "
        f"recovers it completely and story 3.4's round-trip contract says so. Recall recovery on "
        f"that chain is at its maximum by construction, so D = FPR delta - recall delta could lie "
        f"wholly above zero only if the layer turned a majority of {cell.benign_class} into fresh "
        f"false positives -- an impossibility dressed as a threshold. A cell pre-registered here "
        f"would satisfy every word of the requirement while guaranteeing the verdict in advance, "
        f"and it would do it legibly, which is worse. Declare a held-out chain or one nested past "
        f"the ceiling: {way_out}",
    )


LFS_POINTER_VERSION: Final[str] = "https://git-lfs.github.com/spec/v1"
"""The `version` line of a Git LFS pointer file, which is the spec's own identifier for one."""

_LFS_POINTER_MAX_BYTES: Final[int] = 1024
"""The spec caps a pointer at well under this. A bound so a 130 MB corpus is never line-split."""


def lfs_pointer_problem(
    name: str, payload: bytes, *, expected_sha256: str = "", expected_bytes: int = 0
) -> str | None:
    """The message for a corpus file that is a Git LFS pointer, or `None` if it is not one.

    `data/benign.jsonl` is tracked by Git LFS, so a clone made without the LFS filters installed --
    or one whose objects were never fetched -- gets a 131-byte text file where the corpus should
    be. Without this, the content hash below is what notices, and what it says is *"the file was
    edited after the build"*. That is a true statement about the bytes and the wrong diagnosis for
    the reader: they edited nothing, and the thing they have to do is not rebuild the corpus.

    Recognised **structurally**, as the spec defines a pointer -- a small UTF-8 file of `key value`
    lines whose first key is `version` naming the LFS spec, carrying an `oid` and a `size` -- and
    never by searching for a substring. A corpus row that happened to quote the spec URL inside a
    payload is not a pointer, and this repository's corpus is a corpus *of prompt injections*,
    which is the one place a string that looks like a marker is likely to appear on purpose.

    **And the pointer is compared to the manifest, not merely reported.** Git LFS names its objects
    by the SHA-256 of their content, which is the same digest `content_hash` records here, so
    `oid sha256:...` and the manifest's `sha256` are two spellings of one fact that arrived from
    two places: one written by `git add` and one written by the build. Verified against the real
    corpus on 2026-08-30, where both read `22f8ee6d44d5e2...`. That makes the difference between
    two situations a reader must not confuse:

    - the pointer names the object this manifest expects, and the only thing wrong is that it was
      never fetched. `git lfs pull` and the corpus is correct.
    - the pointer names **another** object, so the committed corpus is not the one the manifest
      describes and fetching it would not help. That is the drift the digest check exists for,
      and it survives the file being a pointer.

    `expected_sha256` and `expected_bytes` default to empty, in which case the comparison is
    skipped and the pointer is only reported. A caller that has the manifest entry should pass it.
    """
    if len(payload) > _LFS_POINTER_MAX_BYTES:
        return None
    try:
        text = payload.decode("utf-8")
    except ValueError:
        # `UnicodeDecodeError` is a `ValueError`. Binary that is not UTF-8 is not a pointer, and
        # whatever it is, the hash below is the right thing to report it.
        return None

    fields: dict[str, str] = {}
    for index, line in enumerate(text.splitlines()):
        key, separator, value = line.partition(" ")
        if not separator:
            return None
        if index == 0 and (key != "version" or value != LFS_POINTER_VERSION):
            # The spec fixes `version` as the first line. Checking the position as well as the
            # value is what stops a file that merely mentions the URL from being read as one.
            return None
        fields[key] = value

    if fields.get("version") != LFS_POINTER_VERSION or not {"oid", "size"} <= set(fields):
        return None

    declared_oid = fields["oid"].removeprefix("sha256:")
    head = (
        f"{name} is a Git LFS pointer, not the corpus: {len(payload)} bytes declaring oid "
        f"sha256:{declared_oid} and size {fields['size']}"
    )
    if expected_sha256 and declared_oid != expected_sha256:
        return (
            f"{head}. The manifest records {expected_sha256}, so this pointer names a DIFFERENT "
            f"object: fetching it would not give you the corpus this manifest describes. The "
            f"committed corpus and the manifest disagree, and that is not something `git lfs "
            f"pull` fixes"
        )
    if expected_bytes and fields["size"] != str(expected_bytes):
        return (
            f"{head}. The manifest records {expected_bytes} bytes and the pointer claims "
            f"{fields['size']}; the pointer and the manifest describe different files"
        )
    return (
        f"{head}, which is the object this manifest expects. This clone never fetched it. Run "
        f"`git lfs install` and then `git lfs pull`. Nothing here was edited and the corpus does "
        f"not need rebuilding"
    )


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


def corpus_presence(root: str | Path | None = None) -> dict[str, bool]:
    """Which corpus files exist on disk, by name. Neither read nor verified -- only present.

    Lives here because this is one of the two modules AD-1 lets name a corpus file, and story 4.7's
    entrypoint needs the answer to decide whether to build: it builds only when the corpus is
    **wholly** absent, and a partial corpus aborts. A partial corpus is the state where a rebuild
    writes half-new rows while the manifest still describes the old ones, so "some files exist" has
    to be distinguishable from "none do" without opening either.
    """
    directory = corpus_directory(root)
    return {name: (directory / name).exists() for name in CORPUS_FILENAMES}


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
        pointer = lfs_pointer_problem(
            name, payload, expected_sha256=entry.sha256, expected_bytes=entry.bytes
        )
        if pointer is not None:
            # Ahead of the digest, because the digest is what would otherwise report this, and
            # what it would say sends the reader to rebuild a corpus that is not the problem.
            problems.append(pointer)
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
