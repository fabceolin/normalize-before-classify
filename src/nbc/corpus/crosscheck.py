"""AD-27 / FR3.2: a benign item that carries a pinned attack payload stops the build.

**The error this catches, and why it is the expensive one.** FR4 has the builder emit the gold
label, and `benign.render_benign_item` writes `label=BENIGN` **by construction** -- because the file
came from a repository pinned as benign material, never because anything read the text. So a pinned
public source file that happens to carry an injection payload (a test fixture, a prompt-hardening
example, a security tutorial) becomes a benign item that is actually an attack. The classifier then
fires on it *correctly* and the run records a false positive. The counter-metric, which is the number
the whole argument rests on, gets worse for being right.

**Why it aborts rather than filters, which is the opposite of what `corpus/exclusion.py` does.**
Training overlap is not a labelling error: the row is correctly labelled and merely uninformative for
one baseline, so removing it costs sample size and nothing else, and what made silent exclusion
unacceptable *here* was the silence rather than the exclusion. This is a **gold-label error pointing
the wrong way**: something is mislabelled, nothing in this repository can say what the right label
would be, and a silent exclusion would reshape the benign corpus exactly the way a silent inclusion
reshapes the number. An abort forces a human to look. `attack.LabelContradiction` is this abort's
sibling, for the same reason and in the other half of the corpus.

**The undressed source, and why checking rows instead would be a silent no-op.** The check reads the
benign **source** text before any dressing is applied, against the drawn attack payloads in their
clean form. Once AD-3 and AD-28 dress a benign item, a base64-dressed row never literally contains a
plaintext payload; neither does a hex, base32, url_percent or rot13 row; and neither do the
`homoglyph` and `zero_width` rows, because NFKC maps neither a Cyrillic homoglyph nor U+200B to
anything the payload contains. Every comparison would return false and the gate would pass on the
exact corpus it exists to stop -- with no error, no count and nothing in the report to show it never
fired. `tests/corpus/test_crosscheck.py::test_checking_the_dressed_rows_would_be_a_silent_no_op`
is that claim as an executable test, over every chain in `CHAINS` and `HELDOUT_CHAINS`. Checking the
source is also **sufficient**, because dressing is a pure function of it (AD-3): one check covers
every descendant row.

**The metric, declared with its unit rather than as a bare number.** M-02 of the architecture review
observed that "a declared similarity threshold" without a declared metric means `0.85` can be three
different things. So the metric is named, the width is named, and both travel with the threshold into
the build report:

    shingle-containment  =  |shingles(payload) & shingles(source)| / |shingles(payload)|

a fraction in [0, 1] of the payload's `SHINGLE_WIDTH`-token shingles present anywhere in the source.
It is **asymmetric on purpose**. The question is "how much of this payload is in this file", and a
symmetric measure -- Jaccard, say -- between a 20 KB source file and a 200-character payload is near
zero for every pair that exists, so a symmetric threshold could never be crossed and the limb would
be decorative.

**Two limbs, and the second is not redundant.** Contiguous-token containment implies a ratio of
exactly 1.0, so the similarity limb subsumes it for any payload long enough to have a shingle. That
is precisely why containment stays a separate limb: a threshold somebody later raises leaves the
exact tooth biting, and a payload shorter than `SHINGLE_WIDTH` tokens has **no shingles at all** and
is reachable only through containment. `CrossCheckReport.payloads_containment_only` counts those, so
a pool full of one-word payloads is a number a reader sees rather than a limb that quietly covers
nothing.

**Token sequences, not substrings.** Both limbs compare token sequences under the project's one
declared normalization, `exclusion.normalize` (NFKC, lowercased, whitespace collapsed). A raw
substring test would match the payload `ignore all previous instructions` inside the identifier
`xignore all previous instructions`, and would also miss a payload split across a line break. The
padded-space test below is contiguous-token containment written as one string operation.

**What it costs, measured rather than assumed.** At the declared sizes -- 1200 attack positives
against 1000 benign sources of roughly 3000 tokens each -- building the index takes 0.05 s and the
scan 0.85 s on the reference machine. The naive shape, a substring scan of every payload against
every source, is 1.2 million scans over documents of up to 20 KB and is not a cost this build can
carry; `PayloadIndex` is why the honest comparison is affordable rather than a reason to weaken it.

This module is **pure and offline**: standard library, `nbc.errors`, and `nbc.corpus.exclusion` for
the normalization. It opens no socket and imports nothing under `nbc.canon`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

from nbc.corpus.exclusion import NORMALIZATION, normalize
from nbc.errors import NbcError

__all__ = [
    "CROSS_CHECK_METRIC",
    "KIND_CONTAINS",
    "KIND_SIMILAR",
    "SHINGLE_WIDTH",
    "SIMILARITY_THRESHOLD",
    "BenignItemMislabelled",
    "BenignSource",
    "Collision",
    "CrossCheckReport",
    "PayloadIndex",
    "collisions",
    "contains_token_run",
    "cross_check",
    "shingles",
    "tokens",
]


# --- the declared block -------------------------------------------------------------------------
#
# AD-27 requires the threshold to live "with the other build constants rather than inline", and the
# repository convention is that algorithm constants live in one declared block in the module that
# owns the algorithm -- which is this one. All three are copied into `CrossCheckReport` and compared
# against these constants by a test, so a threshold changed for a build is a threshold visible in
# the corpus manifest rather than only in a diff nobody read.

CROSS_CHECK_METRIC: Final[str] = "shingle-containment"
"""The name of the similarity metric, published beside its value.

Named rather than implied: a bare `0.85` is meaningless until a reader knows whether it is a
normalized edit distance, a token Jaccard or this. See the module docstring for the formula.
"""

SHINGLE_WIDTH: Final[int] = 5
"""Tokens per shingle.

Five, because it is long enough that two unrelated English texts sharing one is already unusual --
so a ratio near the threshold means the payload's *phrasing* is present, not that both texts use the
word `the` -- and short enough that a payload of a couple of sentences still yields tens of shingles
for the ratio to be a fraction of. It is also the reason `payloads_containment_only` exists and is
published: a payload of fewer than this many tokens has no shingles and only the containment limb can
reach it.
"""

SIMILARITY_THRESHOLD: Final[float] = 0.85
"""The fraction of a payload's shingles that must appear in a benign source to abort the build.

0.85 rather than a round 1.0 because the case this limb exists for is a payload the benign source
carries in a slightly edited form -- a test fixture that renamed a variable, a tutorial that
reformatted a sentence -- which containment misses by one token. It is not near the accidental range:
for a fifty-token payload, 0.85 means about thirty-nine of its forty-six five-token shingles occur in
the same file, which is the payload itself rather than a coincidence of vocabulary. Erring low costs
an abort that makes a human look at two texts; erring high costs a mislabelled row in the number the
whole artifact defends.
"""


class BenignItemMislabelled(NbcError, exit_code=24):
    """A benign source item carries a pinned attack payload. The build stops; nothing is written.

    Code 24 because 3 through 23 are taken. **The abort is the requirement**, not a failure of it:
    filtering the item would silently reshape the benign corpus, and the builder cannot say which of
    the two labels is wrong -- the file is pinned as benign material and the payload is pinned as an
    attack, and exactly one of those pins is describing this text incorrectly. A human has to look.

    Four inputs produce it, and they are different diagnoses:

    - a benign source whose token sequence contains a payload's;
    - a benign source whose shingle-containment against a payload reaches `SIMILARITY_THRESHOLD`;
    - a cross-check with **no payloads at all**, which is a gate that cannot fire dressed as a gate
      that found nothing;
    - a payload that normalizes to the empty string, which would match every source ever handed in.

    Every collision is collected before raising, sorted, so a build that is wrong in three places
    says all three in one run.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("BenignItemMislabelled must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "a benign source item carries a pinned attack payload:\n  - " + "\n  - ".join(problems)
        )


KIND_CONTAINS: Final[str] = "contains"
KIND_SIMILAR: Final[str] = "similar"
"""Which limb fired. Carried on the `Collision` so the abort message says which claim it is making."""


def tokens(text: str) -> tuple[str, ...]:
    """The normalized token sequence a comparison is made over.

    `exclusion.normalize` is the project's one declared comparison form -- NFKC, lowercased,
    whitespace collapsed -- and it is reused here rather than re-spelled, so the corpus has one
    answer to "when are two texts the same text" instead of two that agree today. It returns a
    space-joined string with no leading, trailing or repeated spaces, so splitting on a single space
    recovers exactly its tokens.
    """
    joined = normalize(text)
    return tuple(joined.split(" ")) if joined else ()


def shingles(sequence: Sequence[str], width: int = SHINGLE_WIDTH) -> frozenset[tuple[str, ...]]:
    """The set of `width`-token windows in a token sequence. Empty when the sequence is shorter.

    A **set**, not a multiset: the ratio asks which of the payload's phrasings occur in the source,
    and counting a repeated phrase twice would let a source that repeats one shingle of the payload
    a hundred times score higher than one that carries half of it.
    """
    if width <= 0:
        raise ValueError(f"shingle width must be positive, got {width}")
    if len(sequence) < width:
        return frozenset()
    return frozenset(
        tuple(sequence[start : start + width]) for start in range(len(sequence) - width + 1)
    )


def contains_token_run(source: str, payload: str) -> bool:
    """Whether the payload's normalized token sequence is a contiguous run of the source's.

    Written as a padded substring test because the normalized form separates every token by exactly
    one space, so ` payload ` occurring inside ` source ` is contiguous-token containment and nothing
    else. The padding is what makes it structural rather than textual: without it, the payload
    `ignore all previous` matches inside the identifier `xignore all previous`, and a gate that fires
    on a variable name is a gate somebody will delete.
    """
    if not payload:
        raise ValueError("a blank payload is contained in every text; refuse it before asking")
    return f" {payload} " in f" {source} "


@dataclass(frozen=True, slots=True)
class BenignSource:
    """One **undressed** benign item, as the draw holds it before anything is rendered.

    `source` is what the abort names: `github.com/<owner>/<name>@<sha>:<path>` for a B-code file,
    the pinned dataset for a drawn B-chat row, `nbc/corpus/sources/encoded_messages.py` for a
    hand-authored one. `benign_class` travels with it so the message says which half of the
    counter-metric the collision is in.
    """

    source: str
    benign_class: str
    text: str


@dataclass(frozen=True, slots=True)
class Collision:
    """One benign source that carries one attack payload, with the limb that found it.

    `excerpt` is the beginning of the benign item's own text. The `source` alone is enough to find a
    B-code file, whose pin names one path -- but all twenty hand-authored B-chat items share one
    `source`, so without this a reader would be told which module to open and not which item in it.

    `score` is the shingle-containment ratio for both limbs -- 1.0 for a containment hit with at
    least one shingle, and `None` for a payload too short to have any, which is exactly the case the
    similarity limb cannot reach. Recording it as `None` rather than as 0.0 keeps "the metric did not
    apply" distinguishable from "the metric said zero", the same distinction
    `exclusion.SourceOutcome.matched_rows` makes for an unread source.
    """

    source: str
    benign_class: str
    payload: str
    excerpt: str
    kind: str
    score: float | None

    def message(self) -> str:
        where = f"{self.source} ({self.benign_class})"
        if self.kind == KIND_CONTAINS:
            claim = "contains the pinned attack payload"
        else:
            claim = (
                f"matches the pinned attack payload at {CROSS_CHECK_METRIC} "
                f"{self.score:.4f} >= {SIMILARITY_THRESHOLD}"
            )
        return (
            f"{where}, which begins {_excerpt(self.excerpt, 60)!r}, {claim}: "
            f"{_excerpt(self.payload)!r}. The builder labels this item benign by "
            f"construction, so one of the two labels is wrong and nothing here can say which; the "
            f"build aborts rather than dropping the item, because a silent exclusion reshapes the "
            f"benign corpus the same way a silent inclusion reshapes the number"
        )


def _excerpt(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


@dataclass(frozen=True, slots=True)
class PayloadIndex:
    """The pinned attack payloads, normalized once, with an inverted shingle index over them.

    Built once per build and walked once per benign source. The alternative -- a substring scan of
    every payload against every source -- is 1200 x 1000 scans over files of up to 20 KB, which is
    not a cost this build can carry. The index makes it one pass over each source's own shingles,
    and the exact containment test then runs **only** where the ratio is already 1.0. That is sound
    rather than an approximation: contiguous-token containment implies every one of the payload's
    shingles occurs in the source, so a payload whose ratio is below 1.0 cannot be contained.

    `short` holds the payloads with no shingles at all -- fewer than `SHINGLE_WIDTH` tokens. They are
    scanned directly against every source, because the index cannot see them, and their count is
    published so the coverage of the two limbs is a number rather than an assumption.
    """

    payloads: tuple[str, ...]
    normalized: tuple[str, ...]
    shingle_counts: tuple[int, ...]
    by_shingle: Mapping[tuple[str, ...], tuple[int, ...]]
    short: tuple[int, ...]
    width: int

    @property
    def payloads_containment_only(self) -> int:
        return len(self.short)


def build_index(payloads: Sequence[str], width: int = SHINGLE_WIDTH) -> PayloadIndex:
    """Normalize every payload and invert its shingles. Aborts on an input that would match anything.

    Two refusals, both `BenignItemMislabelled` because both mean the gate cannot do its job and the
    build must not continue as though it had:

    - **no payloads at all.** A cross-check over an empty payload set returns no collisions for every
      corpus ever built, which is indistinguishable from a clean corpus. That is P1 in the form this
      project keeps finding: a check whose passing carries no information.
    - **a payload that normalizes to nothing.** ` ` is a token run of every text, so it would abort
      on the first benign source and name a payload that is not there.
    """
    if not payloads:
        raise BenignItemMislabelled(
            "the benign cross-check was handed no attack payloads; a gate with nothing to compare "
            "against returns 'no collisions' for every corpus, which is exactly what a corpus "
            "carrying a mislabelled item also returns"
        )

    normalized: list[str] = []
    blank: list[int] = []
    for position, payload in enumerate(payloads):
        form = normalize(payload)
        normalized.append(form)
        if not form:
            blank.append(position)
    if blank:
        raise BenignItemMislabelled(
            *(
                f"attack payload at position {position} normalizes to the empty string under "
                f"{NORMALIZATION}; it is a token run of every text and would abort on the first "
                f"benign source it met, naming a payload that is not in it"
                for position in blank
            )
        )

    counts: list[int] = []
    short: list[int] = []
    inverted: dict[tuple[str, ...], list[int]] = {}
    for position, form in enumerate(normalized):
        grams = shingles(form.split(" "), width)
        counts.append(len(grams))
        if not grams:
            short.append(position)
        for gram in grams:
            inverted.setdefault(gram, []).append(position)

    return PayloadIndex(
        payloads=tuple(payloads),
        normalized=tuple(normalized),
        shingle_counts=tuple(counts),
        by_shingle={gram: tuple(positions) for gram, positions in inverted.items()},
        short=tuple(short),
        width=width,
    )


def collisions(
    sources: Iterable[BenignSource],
    index: PayloadIndex,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[Collision, ...]:
    """Every (benign source, attack payload) pair either limb fires on, in a deterministic order.

    Returns rather than raises, so the decision procedure can be handed inputs a test constructs and
    inspected without an exception. `cross_check` is the caller that turns a non-empty result into
    the abort.

    Order is `(source, benign_class, payload)`, all content-derived, so two runs over the same inputs
    produce the same message list and the abort a reader sees is a property of the corpus rather than
    of iteration order.
    """
    found: list[Collision] = []

    for entry in sources:
        source_form = normalize(entry.text)
        source_tokens = source_form.split(" ")
        hits: dict[int, int] = {}
        for gram in shingles(source_tokens, index.width):
            for position in index.by_shingle.get(gram, ()):
                hits[position] = hits.get(position, 0) + 1

        for position, shared in hits.items():
            total = index.shingle_counts[position]
            ratio = shared / total
            if ratio >= 1.0 and contains_token_run(source_form, index.normalized[position]):
                found.append(
                    Collision(
                        source=entry.source,
                        benign_class=entry.benign_class,
                        payload=index.payloads[position],
                        excerpt=entry.text,
                        kind=KIND_CONTAINS,
                        score=1.0,
                    )
                )
            elif ratio >= threshold:
                found.append(
                    Collision(
                        source=entry.source,
                        benign_class=entry.benign_class,
                        payload=index.payloads[position],
                        excerpt=entry.text,
                        kind=KIND_SIMILAR,
                        score=ratio,
                    )
                )

        # The payloads the index cannot see. Scanned directly, because a payload of fewer than
        # `width` tokens has no shingles and the loop above would never reach it -- and a payload
        # that short is the easiest of all to find inside a source file.
        for position in index.short:
            if contains_token_run(source_form, index.normalized[position]):
                found.append(
                    Collision(
                        source=entry.source,
                        benign_class=entry.benign_class,
                        payload=index.payloads[position],
                        excerpt=entry.text,
                        kind=KIND_CONTAINS,
                        score=None,
                    )
                )

    return tuple(
        sorted(found, key=lambda hit: (hit.source, hit.benign_class, hit.payload, hit.kind))
    )


@dataclass(frozen=True, slots=True)
class CrossCheckReport:
    """What the cross-check compared, in the shape the corpus manifest publishes.

    The three declared constants travel with the counts, which is M-02's closing rule: a rebuild at a
    different threshold or a different shingle width otherwise produces a different admissible corpus
    with no visible change anywhere. `tests/corpus/test_crosscheck.py` compares each of the three
    against the module constant, so this is evidence that is read rather than evidence that is merely
    recorded.

    `payloads_containment_only` is the count of payloads the similarity limb cannot reach. It is
    published for the same reason `attack.payloads_below_decode_floor` is: an exemption nobody counts
    is an exemption nobody can size.
    """

    metric: str
    shingle_width: int
    similarity_threshold: float
    payloads_checked: int
    sources_checked: int
    payloads_containment_only: int

    def as_run_fields(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "shingle_width": self.shingle_width,
            "similarity_threshold": self.similarity_threshold,
            "normalization": NORMALIZATION,
            "payloads_checked": self.payloads_checked,
            "sources_checked": self.sources_checked,
            "payloads_containment_only": self.payloads_containment_only,
        }


def cross_check(
    sources: Sequence[BenignSource],
    payloads: Sequence[str],
    *,
    width: int = SHINGLE_WIDTH,
    threshold: float = SIMILARITY_THRESHOLD,
) -> CrossCheckReport:
    """AD-27's gate. Aborts naming every offending source and payload, or returns what it compared.

    The **undressed** sources and the **clean** payloads. Nothing here is given a rendered row, and
    `benign.draw_benign_items` calls it before its render loop for that reason: the check must run at
    the last point where the source text still exists, because after dressing every comparison is
    false and the gate passes on the corpus it exists to stop.
    """
    index = build_index(payloads, width)
    hits = collisions(sources, index, threshold)
    if hits:
        raise BenignItemMislabelled(*(hit.message() for hit in hits))
    return CrossCheckReport(
        metric=CROSS_CHECK_METRIC,
        shingle_width=width,
        similarity_threshold=threshold,
        payloads_checked=len(payloads),
        sources_checked=len(sources),
        payloads_containment_only=index.payloads_containment_only,
    )
