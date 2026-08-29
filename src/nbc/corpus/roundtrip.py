"""The round-trip contract: the layer undoes its own corpus's dressing, or CI goes red.

`canonicalize(d(p)).text == canonicalize(p).text`, for every bound chain `d` and every payload
`p`. There is no wrapper to strip on either side: story 3.3's encoded dressings emit the blob and
nothing around it, so the "modulo the wrapper" clause of the requirement is satisfied by a wrapper
that is empty.

**Why this exists.** `corpus/dressings.py` draws its zero-width character from
`canon/stages/invisible.py` and its homoglyph substitutes from the vendored table in
`canon/data/`, so the dressing and the layer already share one character source. Sharing a source
is not the same as agreeing, and nothing compared them until this module: a layer that stopped
neutralizing a character its own corpus emits would not raise anything, it would publish a smaller
recall number. README caveat 7 has claimed since story 1.8 that "a test that fails the build if the
layer does not undo its own corpus's dressing" exists. `tests/corpus/test_roundtrip.py` is that
test, and this module is the contract it runs.

**Two filters scope the contract, and both are structural.** Neither is a list of exceptions; both
answer "what is this pair" rather than "who put it on the list".

1. **The registry filter.** A chain is in scope when every one of its links is in the bound
   dressing registry `corpus/dressings.py::DRESSINGS`. Story 3.5's held-out encodings will live in
   their own module with their own registry, so they fall out of scope *because of what they are*
   -- a held-out chain names links this registry does not hold -- rather than because anyone
   remembered to exempt them. That is testable today, before that module exists: `in_scope` refuses
   a chain naming `rot13`, the name the held-out registry will carry. Nothing can exempt a bound
   chain by adding it to something, and the complementary gate is in the test: the union of the
   links of the in-scope chains must equal `set(DRESSINGS)`, so narrowing the scope until the
   contract passes fails a different test than the one it was narrowed for. AD-28 exists because
   that is the tempting repair.

2. **The layer's own published candidate floor.** `decode.py` declines to decode a run below
   `min_encoded_chars` or below `min_entropy_bits_per_char`, and both floors are declared,
   defended and costed there. A payload small enough or repetitive enough that its own base64 falls
   under one of them is not recovered, and that is the decode policy's published cost rather than
   a character nobody stripped. Measured, not assumed: the round trip fails under 16 payload bytes
   on every base64 chain, under 8 on every hex chain, and for `"a" * 20`, whose base64 measures
   2.3 bits per character against a floor of 3.0.

**What keeps the second filter from being a hiding place.** It applies **only** when the run is
drawn entirely from the encoding's own declared alphabet. A run carrying anything else -- URL-safe
base64, a homoglyph the layer does not map, `U+1D6A8`, a line break through a PEM-wrapped blob --
is never exempt, so a dressing cannot buy an exemption by emitting something the layer does not
recognize. That is the exact failure the contract exists to catch, and three rogue dressings ship
as tests: one emitting `U+1D6A8`, one emitting URL-safe base64, one wrapping its base64 across
lines. The run is also computed from the **prefix** of the chain ending at the encoding link, so an
outer dressing cannot reach back and exempt an inner one.

The exemption is counted rather than merely permitted: `corpus/attack.py` publishes
`payloads_below_decode_floor` on its draw report, so a corpus full of structurally unrecoverable
short rows is a number a reader can see beside the table instead of a quiet depression of every
encoded column.

**AD-23, decided.** The pipeline maps confusables at step 2 and applies NFKC at step 3, so the 144
code points that become confusables only *under* NFKC leave the layer partly canonical --
`U+1D6A8` MATHEMATICAL BOLD CAPITAL ALPHA comes out as Greek, not as `A`, which
`tests/canon/test_pipeline.py` pins. The homoglyph dressing cannot emit one of them:
`homoglyph_substitutes` inverts the vendored table, so every substitute it can emit is a **key** of
that table and step 2 maps it directly, while the 144 are by definition not keys. So the dressing
draws only from the characters the layer fully neutralizes, and the contract is not scoped around
that hole because the hole is unreachable from here. `tests/corpus/test_roundtrip.py` intersects
the two sets rather than leaving the argument as prose.

**The ceiling is raised, and is not a scope filter.** The deepest bound chain spends four levels of
the per-branch decode budget against a `DEFAULT_CEILING` of three, so at the shipped ceiling the
contract would fail on a chain the corpus declares on purpose. The contract therefore runs at
`CONTRACT_CEILING` and reports `ceiling_hit` as a **problem naming the ceiling**, so a ceiling set
too low fails with that sentence instead of as a mysterious inequality. Exempting past-ceiling
chains instead would be wrong for the reason the requirement gives: a benign code payload
legitimately embedding base64 already spends decode levels of its own, and one more dressing would
push the pair past any ceiling that could be declared.

**What is not here.** The scope's consequence for the reader -- on these chains recovery is total by
construction, and the column shows the layer was implemented as specified rather than that
canonicalization is a good idea -- is published in README caveat 7, where the number is. This
module is what makes that paragraph true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Final, Iterable, Mapping, Sequence

from nbc.canon.pipeline import canonicalize
from nbc.canon.stages import decode
from nbc.corpus.dressings import DRESSINGS, Dressing, dress
from nbc.corpus.matrix import CHAINS, render_chain
from nbc.schema import CanonContext

__all__ = [
    "CONTRACT_CEILING",
    "ENCODINGS",
    "Encoding",
    "bound_chains",
    "declined_reason",
    "in_scope",
    "min_payload_bytes",
    "payloads_below_decode_floor",
    "round_trip_problems",
]

CONTRACT_CEILING: Final[int] = 12
"""The declared test-only recursion ceiling the round-trip contract runs at.

**Not** `DEFAULT_CEILING`, and the difference is the point. The corpus declares a chain that nests
four levels deep precisely so N4's first limb has data, and at the shipped ceiling of three that
chain reports `ceiling_hit` and does not round-trip -- correctly, because the layer was told to
stop. Raising the ceiling for the contract asks a different question: does the layer undo the
dressing *when it is allowed to finish*.

**Why twelve.** It has to clear the deepest bound chain (four) plus whatever a payload spends on
encodings of its own -- a benign code payload embedding a JWT already costs one level before any
dressing is applied. Twelve is four with room for a payload nesting several levels on its own, and
`tests/corpus/test_roundtrip.py` asserts it exceeds the deepest bound chain's `encoding_depth`
plus the deepest self-nesting in the committed battery, rather than asserting the number twelve.

**This value never reaches a measurement.** `default_context` applies `DEFAULT_CEILING`, the
entrypoint overrides it once, and nothing in `src/` reads this constant: it is an argument the test
passes to `default_context(ceiling=...)`. That last sentence is a claim, so it is checked --
`tests/corpus/test_roundtrip.py` walks every module under `src/` and requires zero loads of this
name, the same shape `tests/canon/test_recursion.py` uses for `DEFAULT_CEILING`. A run that measured
at this ceiling would be publishing a different layer than the one the results declare.
"""


@dataclass(frozen=True, slots=True)
class Encoding:
    """One bound dressing link that produces a run the layer must decode, and what it costs.

    `expansion` is the **declared** arithmetic of the encoding, not a measurement of the dressing:
    it answers "how many characters does this link produce from `n` bytes" without running
    anything, which is what lets `min_payload_bytes` derive the floor from the layer's constant
    rather than by searching for the length at which the round trip starts working.

    The two sides are compared: `tests/corpus/test_roundtrip.py` runs `expansion` against
    `len(DRESSINGS[link](text))` over every length from 0 to 40, so an expansion that disagrees
    with the dressing it describes fails there rather than silently mis-scoping the contract.
    """

    link: str
    test: decode.CandidateTest
    expansion: Callable[[int], int]


def _base64_expansion(byte_length: int) -> int:
    """RFC 4648 §4 with padding: four characters per three bytes, rounded up to a whole group."""
    return 4 * math.ceil(byte_length / 3)


def _hex_expansion(byte_length: int) -> int:
    """Two characters per byte, exactly."""
    return 2 * byte_length


ENCODINGS: Final[Mapping[str, Encoding]] = {
    "base64": Encoding("base64", decode.BASE64, _base64_expansion),
    "hex": Encoding("hex", decode.HEX, _hex_expansion),
}
"""The bound links whose output the layer decodes, each bound to the layer's own candidate test.

Keyed by the dressing name that appears in a chain, so membership is a lookup in a closed
vocabulary rather than a name test. A test asserts the key set equals `matrix.DECODED_LINKS` and
that each entry's `test` is the very `CandidateTest` the layer applies, because a floor read off a
second declaration would scope the contract by a number the layer does not use.
"""


def min_payload_bytes(link: str) -> int:
    """The smallest payload, in UTF-8 bytes, whose `link` run reaches the layer's length floor.

    Derived from `decode`'s `min_encoded_chars` through the declared expansion, never written down:
    base64's floor of 24 characters is reached by **16** bytes, not 18, because 24 characters of
    base64 carry *up to* 18 bytes and `4 * ceil(16/3)` is already 24 with padding. That off-by-two
    was in two docstrings in this repository until this contract derived the number.

    Returns `0` for a link that produces no decodable run, so a caller can ask about any link.
    """
    encoding = ENCODINGS.get(link)
    if encoding is None:
        return 0
    floor = encoding.test.min_encoded_chars
    byte_length = 0
    while encoding.expansion(byte_length) < floor:
        byte_length += 1
    return byte_length


def in_scope(chain: Sequence[str], registry: Mapping[str, Dressing] = DRESSINGS) -> bool:
    """Whether the contract covers `chain`: every link is a **bound** dressing.

    The filter over the registries. A held-out chain names links that live in story 3.5's own
    module and are not in this registry, so it falls out because of what it is. The failing input
    is a chain naming `rot13`, which is exactly the name the held-out registry will carry.
    """
    return all(link in registry for link in chain)


def bound_chains(
    chains: Mapping[str, Sequence[Sequence[str]]] = CHAINS,
    registry: Mapping[str, Dressing] = DRESSINGS,
) -> tuple[tuple[str, ...], ...]:
    """Every distinct in-scope chain the corpus declares, ordered by its rendered name.

    Distinct across corpus classes: `CHAINS` writes the same ten chains out once per class so that
    their agreement is a comparison rather than an identity, and the contract has no reason to
    compare the same chain three times. Ordered by the rendered name so a failure list is stable.

    Epic 4 will read this to say which cells of the table are bound, which is why the scope is a
    function in `src/` rather than a fixture in the test module.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for declared in chains.values():
        for chain in declared:
            links = tuple(chain)
            if in_scope(links, registry):
                seen.setdefault(render_chain(links), links)
    return tuple(seen[name] for name in sorted(seen))


def declined_reason(
    payload: str,
    chain: Sequence[str],
    *,
    dress_fn: Callable[[str, Sequence[str]], str] = dress,
) -> str | None:
    """Why the layer declines this pair by its own published floors, or `None` if it does not.

    Walks the chain and, at each link that produces a decodable run, rebuilds the run the layer
    will see for it -- `dress_fn(payload, chain[:index + 1])`, the **prefix** ending at that link,
    because the character stages restore whatever an outer dressing did before step 4 looks. So an
    outer dressing cannot exempt an inner link, and a rogue outer dressing changes nothing here.

    A run carrying a character outside the encoding's declared alphabet is **not** a decline: it is
    the failure the contract exists to catch, and it is passed through to the comparison.
    """
    for index, link in enumerate(chain):
        encoding = ENCODINGS.get(link)
        if encoding is None:
            continue
        run = dress_fn(payload, tuple(chain[: index + 1]))
        if not set(run) <= encoding.test.alphabet:
            continue
        if len(run) < encoding.test.min_encoded_chars:
            return (
                f"link {index} ({link}) produces a {len(run)}-character run, below the layer's "
                f"declared min_encoded_chars of {encoding.test.min_encoded_chars}; the smallest "
                f"payload this link can carry is {min_payload_bytes(link)} bytes"
            )
        entropy = decode.shannon_bits_per_char(run)
        if entropy < encoding.test.min_entropy_bits_per_char:
            return (
                f"link {index} ({link}) produces a run measuring {entropy:.2f} bits per "
                f"character, below the layer's declared min_entropy_bits_per_char of "
                f"{encoding.test.min_entropy_bits_per_char}"
            )
    return None


def payloads_below_decode_floor(
    payloads: Iterable[str],
    chains: Sequence[Sequence[str]] | None = None,
    *,
    dress_fn: Callable[[str, Sequence[str]], str] = dress,
) -> tuple[str, ...]:
    """The payloads the layer declines on at least one in-scope chain, in the order given.

    Published as a count by `corpus/attack.py`, because an exemption nobody counts is an exemption
    nobody can size. A reader of the report can divide it by the drawn positives and know what
    fraction of the encoded columns is structurally unrecoverable before any classifier ran.
    """
    if chains is None:
        chains = bound_chains()
    declined: list[str] = []
    for payload in payloads:
        if any(declined_reason(payload, chain, dress_fn=dress_fn) is not None for chain in chains):
            declined.append(payload)
    return tuple(declined)


def _excerpt(text: str, limit: int = 60) -> str:
    """A failure message quotes enough to identify the text and never the whole document."""
    return text if len(text) <= limit else text[:limit] + "..."


def round_trip_problems(
    payloads: Iterable[str],
    chains: Sequence[Sequence[str]],
    ctx: CanonContext,
    *,
    dress_fn: Callable[[str, Sequence[str]], str] = dress,
    registry: Mapping[str, Dressing] = DRESSINGS,
) -> tuple[str, ...]:
    """Every way the layer failed to undo the dressing. Empty when the contract holds.

    Problems rather than an abort, and every argument a parameter rather than a module constant:
    that is what gives this gate the inputs that make it fail. `tests/corpus/test_roundtrip.py`
    hands it a dressing that substitutes `U+1D6A8` for a letter, a dressing that emits URL-safe
    base64, a dressing that wraps its base64 across lines, and a context whose ceiling is too low.
    A gate that could only ever be called with the one input that passes is not a gate.

    `ceiling_hit` on either side is reported as its own problem, naming the ceiling, so the
    diagnosis is "raise the contract ceiling" rather than an inequality between two texts that
    differ for a reason nothing states.
    """
    problems: list[str] = []
    for payload in payloads:
        plain = canonicalize(payload, ctx)
        if plain.ceiling_hit:
            problems.append(
                f"the undressed payload {_excerpt(payload)!r} already reports ceiling_hit at "
                f"ceiling {ctx.ceiling}; the contract's ceiling must clear what the payload "
                f"spends on its own before it can say anything about a dressing"
            )
            continue
        for chain in chains:
            links = tuple(chain)
            if not in_scope(links, registry):
                continue
            if declined_reason(payload, links, dress_fn=dress_fn) is not None:
                continue
            dressed = canonicalize(dress_fn(payload, links), ctx)
            if dressed.ceiling_hit:
                problems.append(
                    f"{render_chain(links)} on payload {_excerpt(payload)!r} reports ceiling_hit "
                    f"at ceiling {ctx.ceiling}; the contract is not scoped by the ceiling, so "
                    f"raise it rather than reading the comparison below"
                )
                continue
            if dressed.text != plain.text:
                problems.append(
                    f"{render_chain(links)} on payload {_excerpt(payload)!r} does not round "
                    f"trip: the layer leaves {_excerpt(dressed.text)!r} where the undressed "
                    f"payload canonicalizes to {_excerpt(plain.text)!r}"
                )
    return tuple(problems)
