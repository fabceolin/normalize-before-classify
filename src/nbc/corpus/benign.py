"""The benign half of the corpus, drawn under a frame that was fixed before anything is measured.

This module is **pure and offline**, exactly as `corpus/attack.py` is. It imports the standard
library, `nbc.errors`, `nbc.pins`, `nbc.schema`, the corpus modules and -- unlike `attack.py` --
the canonicalization layer, because FR5.1's declared file-eligibility rule *is* a question about
the layer. Everything that reaches a network is `corpus/build.py`, which hands the files and the
rows in as data.

**The suspicion this exists to answer.** A benign corpus can always be grown until the
false-positive rate looks reasonable, and no amount of prose refutes that. What refutes it is a
frame declared and hashed in `pins.toml` before the corpus is built, plus a builder that **fails**
when the frame cannot be filled rather than reaching for another source. Both classes are drawn
here, and both aborts are `BenignDrawUnsatisfiable`.

**Why the two classes are drawn differently, and why that is not an inconsistency.** B-chat draws
from one pinned dataset's benign rows after the training-overlap filter; B-code draws files out of
sixty-odd pinned git repositories. FR3.1 reports the two separately and never pools them precisely
because they are different populations: a layer that is safe on conversational text and destructive
on source code looks acceptable in a pooled number, and that is the failure a reader cares about.

**What B-code is a sample of, stated rather than implied.** The frame's `eligibility` rule admits a
file only when the layer's decode stage examines at least one run in it. That is the population
where a decoding false positive can happen at all -- on a file the layer leaves untouched, both
routes score the identical text and the delta is zero by construction. It is therefore **not** a
uniform sample of public source code, and the B-code false-positive rate must not be read as one.
The bias runs against this project's own thesis, which is the direction to err in.

**Every ordering input is content-derived**, as in the attack half: files are keyed by their own
text, pools are sorted before they are shuffled, per-repository draws are seeded from the frame's
seed and the repository's own name, and nothing reads an archive order, a directory order or a
process hash seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

from nbc.canon.pipeline import canonicalize, default_context
from nbc.canon.stages import decode
from nbc.corpus.draw import take
from nbc.corpus.dressings import DRESSINGS, dress_declared
from nbc.corpus.heldout import validate_heldout
from nbc.corpus.matrix import (
    CHAINS,
    HELDOUT_CHAINS,
    id_collisions,
    item_id,
    payload_id,
    render_chain,
    validate as validate_matrix,
)
from nbc.corpus.sources.encoded_messages import (
    MESSAGES,
    EncodedMessage,
    problems as hand_authored_problems,
)
from nbc.errors import NbcError
from nbc.pins import (
    BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
    DRAW_HEAD,
    DRAW_SEEDED_RANDOM,
    AttackDataset,
    BenignCodeRepository,
    BenignFrame,
)
from nbc.schema import (
    BENIGN,
    BENIGN_CLASSES,
    FAMILY_BENIGN,
    CanonContext,
    CorpusItem,
)

__all__ = [
    "BENIGN_CHAT",
    "BENIGN_CODE",
    "DECODE_STAGE_NAMES",
    "HAND_AUTHORED_SOURCE",
    "BenignDrawReport",
    "default_eligibility_context",
    "offers_decode_candidate",
    "BenignDrawUnsatisfiable",
    "CodeFile",
    "SourceFile",
    "chains_for",
    "draw_benign_items",
    "draw_chat_texts",
    "eligible",
    "render_benign_item",
    "repository_seed",
    "select_repository_files",
]


class BenignDrawUnsatisfiable(NbcError, exit_code=21):
    """The declared benign frame cannot be filled from the material the frame itself names.

    Code 21 because 3 through 20 are taken. **This abort is the frame**, not a failure of it. FR5.1
    fixes the per-class count exactly and forbids topping up from another source, so a class that
    falls short has exactly two honest outcomes: stop, or re-declare the frame deliberately and
    re-run everything. Quietly publishing whatever survived would make the declared size a number
    that describes nothing, and it would do so with every other check still green.

    Six inputs produce it, and they are different diagnoses:

    - fewer eligible source files survive than the class declares;
    - fewer benign rows survive the training-overlap filter than the class declares;
    - the **realized** repository count is below the frame's floor, so the design effect the floor
      exists to bound is back;
    - `corpus/sources/` holds a different number of hand-authored items than the frame allows;
    - a hand-authored item is not the kind it declares itself to be;
    - a selection method nothing implements, or two distinct texts colliding on one item id.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("BenignDrawUnsatisfiable must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "the declared benign frame could not be filled:\n  - " + "\n  - ".join(problems)
        )


BENIGN_CODE: Final[str] = BENIGN_CLASSES[0]
BENIGN_CHAT: Final[str] = BENIGN_CLASSES[1]
"""The two benign classes, read out of `schema.BENIGN_CLASSES` rather than spelled again.

`pins.py` spells them, because it is a leaf that may not import `nbc.schema`, and
`tests/test_pins.py` compares the two spellings. Here there is no such constraint, so the
vocabulary is taken from its home.
"""

HAND_AUTHORED_SOURCE: Final[str] = "nbc/corpus/sources/encoded_messages.py"
"""What a hand-authored B-chat row names as its source: the module it was written in.

A row's `source` is where a reader goes to check it. For a drawn row that is a pinned dataset at a
pinned revision; for these twenty it is a file in this repository, and saying so is the point --
they are the only corpus text this project wrote itself, and a reader must be able to tell them
apart from the rest without running anything.
"""

DECODE_STAGE_NAMES: Final[frozenset[str]] = frozenset({decode.NAME, decode.CEILING_NAME})
"""The trace stage names that mean "the layer examined a decode candidate here".

**Both**, and the second is not decoration: a candidate the recursion ceiling refused is still a
candidate the layer offered, and a file whose only candidate sits past the ceiling is still a file
where a decoding false positive can happen under a different ceiling. Read off `decode`'s own
constants rather than spelled, so a renamed stage moves this set instead of silently emptying it.
"""


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file as `corpus/build.py` read it out of a pinned archive: its path and its text."""

    path: str
    text: str


@dataclass(frozen=True, slots=True)
class CodeFile:
    """One eligible B-code candidate, carrying the pin that identifies it.

    `source` is `github.com/<owner>/<name>@<sha>:<path>`, built by
    `pins.BenignCodeRepository.file_source`, so FR5.1's "pinned by repository, commit sha and path"
    is a property of the committed corpus row rather than of a build log somebody kept.
    """

    repository_key: str
    source: str
    path: str
    text: str


def eligible(
    file: SourceFile, frame: BenignFrame, ctx: CanonContext
) -> bool:
    """Whether this file is a B-code candidate under the frame's declared rule.

    Three conditions, all declared in `[benign_frame.b_code]` and therefore all inside `frame_id`:
    the path's own suffix is one of the declared extensions, the UTF-8 length is inside the declared
    band, and the layer's decode stage examines at least one run in the text.

    The suffix is matched **structurally**, off `str.rpartition`, rather than by searching for the
    extension anywhere in the path: `assets.js.map` and `notes.py.txt` are not source files and a
    substring test admits both.
    """
    _stem, dot, suffix = file.path.rpartition(".")
    if not dot or f".{suffix}".lower() not in frame.b_code.file_extensions:
        return False
    size = len(file.text.encode("utf-8"))
    if not frame.b_code.min_file_bytes <= size <= frame.b_code.max_file_bytes:
        return False
    return offers_decode_candidate(file.text, ctx)


def offers_decode_candidate(text: str, ctx: CanonContext) -> bool:
    """Whether the canonicalization layer examines a decode candidate anywhere in `text`.

    Measured against the layer, never matched with a pattern of this module's own. The decode stage
    records an `Edit` for every candidate it examines, including the ones it refuses -- a refusal is
    a no-op edit rather than an absence -- so the trace answers the question directly and a change
    to the candidate test moves this predicate with it.

    `ctx.trace_enabled` must be true or the trace is empty and this answers `False` for every
    document. `default_context()` enables it; a caller that turns it off gets told.
    """
    if not ctx.trace_enabled:
        raise ValueError(
            "offers_decode_candidate reads the trace, and this context has tracing off; with no "
            "trace every file would be judged ineligible and the B-code draw would silently empty"
        )
    result = canonicalize(text, ctx)
    return any(edit.stage in DECODE_STAGE_NAMES for edit in result.edits)


def repository_seed(frame: BenignFrame, repository: BenignCodeRepository) -> int:
    """The per-repository shuffle seed: content-derived from the frame's seed and the repository id.

    One declared seed in `pins.toml` and one derivation, rather than sixty-three declared seeds or
    one shared stream whose per-repository result would depend on the order the archives were read.
    `payload_id` is the same truncated SHA-256 the item ids use, so the derivation is the project's
    one hashing convention rather than a second one invented here.
    """
    return int(payload_id(f"{frame.seed}:{repository.repository}"), 16)


def select_repository_files(
    repository: BenignCodeRepository,
    files: Iterable[SourceFile],
    frame: BenignFrame,
    ctx: CanonContext,
) -> tuple[CodeFile, ...]:
    """The files one repository contributes: eligible, deduplicated, and capped by the frame.

    The cap is FR5.1's `max_files_per_repository` and it is applied **here**, per repository, before
    anything global happens. Applying it after a global draw would let one large repository fill the
    class and leave the floor to be satisfied by whatever was left.

    Deduplicated by text within the repository: a vendored file that appears twice is one payload,
    and two rows carrying it would share an item id.
    """
    if frame.b_code.eligibility != BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE:
        raise BenignDrawUnsatisfiable(
            f"the frame declares file eligibility {frame.b_code.eligibility!r}, which nothing "
            f"implements; the admissible rule is "
            f"{BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE!r}"
        )

    by_text: dict[str, CodeFile] = {}
    for file in sorted(files, key=lambda entry: entry.path):
        if not eligible(file, frame, ctx):
            continue
        by_text.setdefault(
            file.text,
            CodeFile(
                repository_key=repository.key,
                source=repository.file_source(file.path),
                path=file.path,
                text=file.text,
            ),
        )

    drawn = take(
        by_text.keys(),
        frame.b_code.max_files_per_repository,
        method=frame.method,
        seed=repository_seed(frame, repository),
        sort_key=frame.sort_key,
    )
    if drawn is None:
        raise BenignDrawUnsatisfiable(_unimplemented_method(frame))
    return tuple(by_text[text] for text in drawn)


def _unimplemented_method(frame: BenignFrame) -> str:
    return (
        f"the benign frame declares selection method {frame.method!r}, which nothing implements; "
        f"the admissible methods are {DRAW_HEAD!r} and {DRAW_SEEDED_RANDOM!r}"
    )


def chains_for(benign_class: str) -> tuple[tuple[str, ...], ...]:
    """Every chain a row of `benign_class` is rendered in: the bound registry, then the held-out one.

    Both, in a declared order, for AD-28's reason: benign items are dressed in the held-out chains
    too, because a held-out block carrying recall and no counter-metric invites the answer that
    non-recovery does not matter since its cost is unknown.
    """
    bound = tuple(tuple(chain) for chain in CHAINS[benign_class])
    held_out = tuple(tuple(chain) for chain in HELDOUT_CHAINS[benign_class])
    return bound + held_out


def render_benign_item(
    text: str, *, source: str, benign_class: str, chain: Sequence[str]
) -> CorpusItem:
    """One benign corpus row: the rendered text and the gold label, in one constructor call.

    `label=BENIGN` is a named schema constant and an AST scan enforces it, for the same reason the
    attack half's is: a label read off a source row would be somebody else's annotation wearing this
    builder's name. Here the point is sharper still -- nothing about a public source file or a
    dataset row *says* it is benign, and what makes this label true is the construction: the file
    came from a repository pinned as benign material, and the row from the benign side of a declared
    label column. Story 3.7 is the check that the construction was not wrong.

    The **payload id is the id of the undressed text**, so one file's thirteen rows share a stem and
    the harness can pair the same item across the dressing axis.
    """
    return CorpusItem(
        id=item_id(payload_id(text), chain),
        source=source,
        family=FAMILY_BENIGN,
        benign_class=benign_class,
        dressing=tuple(chain),
        text=dress_declared(text, chain),
        label=BENIGN,
    )


def draw_chat_texts(
    surviving: Iterable[str],
    frame: BenignFrame,
    messages: Sequence[EncodedMessage] = MESSAGES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The B-chat draw: `(hand_authored_texts, drawn_dataset_texts)`.

    Two tuples rather than one, because the two carry different `source` values and a reader of
    `data/*.jsonl` has to be able to tell which rows this repository wrote. The counts are also
    reported separately for the same reason.

    Every gate here names the input that makes it fail:

    - `corpus/sources/` holding a number of items the frame does not allow;
    - a hand-authored item that is not the kind it declares;
    - a hand-authored text that also appears among the surviving dataset rows, which would put one
      payload under two sources and one item id;
    - fewer surviving rows than the class needs after the allowance.
    """
    problems = list(hand_authored_problems(messages))

    allowance = frame.b_chat.hand_authored_items
    if len(messages) != allowance:
        problems.append(
            f"the frame allows {allowance} hand-authored items and corpus/sources/ holds "
            f"{len(messages)}. The allowance is compared rather than trusted: an item added there "
            f"without widening the frame would enlarge the share of the counter-metric this "
            f"repository wrote for itself, and nothing would show it"
        )

    hand_authored = tuple(message.text for message in messages)
    if len(set(hand_authored)) != len(hand_authored):
        problems.append("two hand-authored items carry the same text")

    pool = {text for text in surviving if text}
    shared = sorted(pool & set(hand_authored))
    if shared:
        problems.append(
            f"{len(shared)} hand-authored text(s) also appear among the surviving dataset rows, "
            f"so one payload would carry two sources and one item id: {shared[:3]}"
        )
    pool -= set(hand_authored)

    needed = frame.sample_size_items - allowance
    if needed < 0:
        problems.append(
            f"the frame allows {allowance} hand-authored items in a class of "
            f"{frame.sample_size_items}"
        )
    elif len(pool) < needed:
        problems.append(
            f"{BENIGN_CHAT} needs {needed} rows from the pinned dataset after its "
            f"{allowance}-item hand-authored allowance, and {len(pool)} benign rows survive the "
            f"training-overlap filter. The build fails rather than topping up from another "
            f"dataset: a frame that quietly substitutes sources is not a frame"
        )

    if problems:
        raise BenignDrawUnsatisfiable(*problems)

    drawn = take(
        pool, needed, method=frame.method, seed=frame.seed, sort_key=frame.sort_key
    )
    if drawn is None:
        raise BenignDrawUnsatisfiable(_unimplemented_method(frame))
    return hand_authored, drawn


@dataclass(frozen=True, slots=True)
class BenignDrawReport:
    """What the benign draw did, in counts a reader can check against the declared frame.

    `realized_repositories` and `files_by_repository` are FR5.1's own requirement: the frame states
    a floor and a cap, and what a reader needs is what actually happened. The two are published
    together because the floor is about the *number* of repositories and the cap is about the
    *shape* of their contribution, and a corpus can satisfy one while failing the other.
    """

    frame_id: str
    sample_size_items: int
    chains: tuple[str, ...]
    held_out_chains: tuple[str, ...]
    code_candidates: int
    code_repositories_pinned: int
    code_repositories_realized: int
    files_by_repository: Mapping[str, int]
    chat_rows_in: int
    chat_rows_removed: int
    chat_rows_surviving: int
    chat_drawn_from_dataset: int
    chat_hand_authored: int
    items_written: int

    def as_run_fields(self) -> dict[str, object]:
        return {
            "benign_draw": {
                "frame_id": self.frame_id,
                "sample_size_items": self.sample_size_items,
                "chains": list(self.chains),
                "held_out_chains": list(self.held_out_chains),
                BENIGN_CODE: {
                    "candidates": self.code_candidates,
                    "repositories_pinned": self.code_repositories_pinned,
                    # The realized count, which is what FR5.1 asks for and what the floor is
                    # checked against. A pinned repository that yielded no eligible file
                    # contributes nothing and is absent from `files_by_repository`.
                    "repositories_realized": self.code_repositories_realized,
                    "files_by_repository": dict(sorted(self.files_by_repository.items())),
                },
                BENIGN_CHAT: {
                    "rows_in": self.chat_rows_in,
                    "rows_removed_by_exclusion": self.chat_rows_removed,
                    "rows_surviving": self.chat_rows_surviving,
                    "drawn_from_dataset": self.chat_drawn_from_dataset,
                    "hand_authored": self.chat_hand_authored,
                },
                "items_written": self.items_written,
            }
        }


def draw_benign_items(
    *,
    frame: BenignFrame,
    code_by_repository: Mapping[str, Sequence[CodeFile]],
    chat_surviving: Sequence[str],
    dataset: AttackDataset,
    chat_rows_in: int,
    chat_rows_removed: int,
    messages: Sequence[EncodedMessage] = MESSAGES,
) -> tuple[tuple[CorpusItem, ...], BenignDrawReport]:
    """Both benign classes, drawn, gated and rendered. Aborts before returning anything.

    The order is the same as the attack half's and for the same reasons: validate the declared
    matrix first, so a chain naming a dressing nothing implements fails before a file is looked at;
    then draw; then render once per declared chain over both registries.

    **`sample_size_items` is checked as an equality, in both directions, at the end.** The shortfall
    gates above give a reader the diagnosis and the numbers; this one is the last line of defence
    against a draw that silently returned something other than what the frame declared. FR5.1's
    "exactly, never at least" is that equality and nothing else.
    """
    validate_matrix(CHAINS, DRESSINGS)
    validate_heldout()

    # --- B-code ---------------------------------------------------------------------------------
    by_text: dict[str, CodeFile] = {}
    for key in sorted(code_by_repository):
        for file in sorted(code_by_repository[key], key=lambda entry: entry.source):
            # Across repositories as well as within one: two repositories vendoring the same file
            # would otherwise produce two rows under one item id, and the corpus would hold one
            # where the report counted two. The first `source` in sorted order wins, so which of
            # the two survives is a property of the pins rather than of the fetch order.
            by_text.setdefault(file.text, file)

    problems: list[str] = []
    if len(by_text) < frame.sample_size_items:
        problems.append(
            f"{BENIGN_CODE} needs {frame.sample_size_items} files and "
            f"{len(by_text)} eligible files were found over "
            f"{len(code_by_repository)} pinned repositories. The build fails rather than topping "
            f"up from elsewhere, and rather than lowering the declared count to what it found"
        )
    if problems:
        raise BenignDrawUnsatisfiable(*problems)

    drawn_code = take(
        by_text.keys(),
        frame.sample_size_items,
        method=frame.method,
        seed=frame.seed,
        sort_key=frame.sort_key,
    )
    if drawn_code is None:
        raise BenignDrawUnsatisfiable(_unimplemented_method(frame))

    code_files = tuple(by_text[text] for text in drawn_code)
    files_by_repository: dict[str, int] = {}
    for file in code_files:
        files_by_repository[file.repository_key] = (
            files_by_repository.get(file.repository_key, 0) + 1
        )

    if len(files_by_repository) < frame.b_code.min_repositories:
        raise BenignDrawUnsatisfiable(
            f"{BENIGN_CODE} realized {len(files_by_repository)} repositories against a declared "
            f"floor of {frame.b_code.min_repositories}. The floor is not a preference: files from "
            f"one repository share a language, a style and a base64 idiom, so a corpus drawn from "
            f"fewer carries a design effect that widens every B-code interval while the reported "
            f"n stays {frame.sample_size_items}"
        )
    over_cap = sorted(
        key
        for key, count in files_by_repository.items()
        if count > frame.b_code.max_files_per_repository
    )
    if over_cap:
        # Cannot happen while `select_repository_files` applied the cap, which is exactly why it is
        # checked here: this function is also reachable with candidates a caller assembled, and a
        # cap enforced only where it is applied is a cap nothing verifies.
        raise BenignDrawUnsatisfiable(
            f"{BENIGN_CODE} drew more than {frame.b_code.max_files_per_repository} files from "
            f"{over_cap}; the cap is the other half of the design effect the floor bounds"
        )

    # --- B-chat ---------------------------------------------------------------------------------
    hand_authored, drawn_chat = draw_chat_texts(chat_surviving, frame, messages)

    # --- render ---------------------------------------------------------------------------------
    code_chains = chains_for(BENIGN_CODE)
    chat_chains = chains_for(BENIGN_CHAT)
    dataset_source = f"{dataset.repository}@{dataset.revision}"

    rendered: list[tuple[CorpusItem, str]] = []
    for file in code_files:
        for chain in code_chains:
            rendered.append(
                (
                    render_benign_item(
                        file.text,
                        source=file.source,
                        benign_class=BENIGN_CODE,
                        chain=chain,
                    ),
                    file.text,
                )
            )
    for text in hand_authored:
        for chain in chat_chains:
            rendered.append(
                (
                    render_benign_item(
                        text,
                        source=HAND_AUTHORED_SOURCE,
                        benign_class=BENIGN_CHAT,
                        chain=chain,
                    ),
                    text,
                )
            )
    for text in drawn_chat:
        for chain in chat_chains:
            rendered.append(
                (
                    render_benign_item(
                        text,
                        source=dataset_source,
                        benign_class=BENIGN_CHAT,
                        chain=chain,
                    ),
                    text,
                )
            )

    items = tuple(item for item, _text in rendered)

    # Two checks, because they catch different things and the first one alone reads as if it
    # caught both. `id_collisions` answers "did two *different* payloads land on one id", which is
    # a SHA-256 prefix collision and cannot be constructed. The duplicate-id count answers "did two
    # *rows* land on one id", which absolutely can: the two benign classes are deduplicated
    # separately, so one text drawn into both -- a source file that is also a chat message -- would
    # produce two rows sharing an id and differing in `benign_class`, and every pairing across the
    # dressing axis downstream keys on that id.
    collisions = list(id_collisions((item.id, text) for item, text in rendered))
    seen: dict[str, CorpusItem] = {}
    for item in items:
        first = seen.setdefault(item.id, item)
        if first is not item:
            collisions.append(
                f"two rows share item id {item.id}: {first.benign_class}/{first.source} and "
                f"{item.benign_class}/{item.source}. One payload drawn into both classes is one "
                f"row where the frame counted two, and the harness keys its pairing on this id"
            )
    if collisions:
        raise BenignDrawUnsatisfiable(*collisions)

    realized = {
        BENIGN_CODE: len(code_files),
        BENIGN_CHAT: len(hand_authored) + len(drawn_chat),
    }
    wrong = sorted(
        f"{name} realized {count} items against a declared {frame.sample_size_items}"
        for name, count in realized.items()
        if count != frame.sample_size_items
    )
    if wrong:
        raise BenignDrawUnsatisfiable(*wrong)

    report = BenignDrawReport(
        frame_id=frame.frame_id,
        sample_size_items=frame.sample_size_items,
        chains=tuple(render_chain(chain) for chain in CHAINS[BENIGN_CODE]),
        held_out_chains=tuple(render_chain(chain) for chain in HELDOUT_CHAINS[BENIGN_CODE]),
        code_candidates=len(by_text),
        code_repositories_pinned=len(frame.b_code.repositories),
        code_repositories_realized=len(files_by_repository),
        files_by_repository=files_by_repository,
        chat_rows_in=chat_rows_in,
        chat_rows_removed=chat_rows_removed,
        chat_rows_surviving=len(set(chat_surviving)),
        chat_drawn_from_dataset=len(drawn_chat),
        chat_hand_authored=len(hand_authored),
        items_written=len(items),
    )
    return items, report


def default_eligibility_context() -> CanonContext:
    """The layer context the eligibility rule is measured under, built once per build.

    `default_context()` reads and validates the vendored confusables table on every call by design,
    so calling it per file would put disk I/O inside a loop over a hundred thousand files. The
    ceiling is the layer's declared default, because the eligibility question is what the layer as
    shipped does with the file.
    """
    return default_context()
