"""Step 4: decode what is actually an encoding, and say so when it is not.

**The whole test is in one block, and each constant declares its unit and its side.** AD-18's
worked failure is an entropy threshold that means bits per character in one implementation and a
normalized score in another, where the same declared `3.5` decodes everything or nothing. So
`CONSTANTS` states, for each constant, what it is called, what it counts, and which side of the
comparison it sits on; `CandidateTest` holds the values; and a test compares the two, because a
unit recorded beside a value and never compared to it is the shape this project keeps finding.

**Run, candidate, decision — three words that are not synonyms.**

- A **run** is a maximal sequence of characters drawn from one alphabet. Maximal matters: the
  characters bounding it are not in the alphabet, which is why a document of English prose is a
  sequence of short runs rather than one long one.
- A run of at least `min_encoded_chars` characters is a **candidate**. A shorter run is ordinary
  text. It is left alone *and not reported*: reporting every run would put one trace entry on every
  word in the document, and the trace exists to explain a change, not to enumerate the alphabet.
- A candidate is **decoded** when its entropy clears the floor, its characters decode structurally
  as that encoding, and the resulting bytes decode as **strict UTF-8**. Otherwise it is left
  untouched and reported as an `Edit` whose `before` equals its `after`, so a reader asking why a
  benign file was flagged finds the refusal in the trace rather than finding nothing.

Structural decodability is not a fourth constant. It is what "the bytes" in AD-18 presupposes, and
`base64.b64decode(validate=True)` and `bytes.fromhex` are its one implementation here; a run with
an odd number of hex characters, or a base64 run whose length is not a multiple of four, or one
carrying `=` anywhere but at the end, is a refused candidate, not a special case.

**Hex before base64, and the reason is checked rather than asserted.** The hex alphabet is a strict
subset of base64's, so a hex run is *always* also a base64 run and the more specific test has to win
or the replacement text of every benign item containing a hash depends on which branch ran first.
`ORDER` states the order and a test asserts both the order and the subset relation, so the premise
and the consequence are checked against each other rather than restated in prose.

That subset relation is also what makes the scan simple: every maximal hex run lies inside exactly
one maximal base64 run. So the scan walks maximal base64 runs and, inside each one, offers the hex
candidates first. A run **accepted** by hex is not re-offered to base64. A run hex refused still
belongs to the wider base64 candidate when there is one, and is reported once, as that candidate —
never twice, because two overlapping edits are not a trace, they are two claims about one span.

**"The result is not itself a longer run of the same encoding alphabet."** AD-18's third decode
condition is not a branch in this module, and that is deliberate. Read as a length comparison it can
never fail: base64 yields at most three characters per four consumed, hex exactly one per two, and
UTF-8 decoding never increases a character count, so a decode is strictly contracting. Read as "the
result is not a run of the same alphabet **at all**" it can fail, but it would refuse the inner
level of `base64(base64(x))` — which AD-4 requires to decode one level per recursion depth, Story
3.3 requires to exist deeper than the ceiling, and Story 3.4 requires to round-trip. The condition
therefore holds by construction, proved by enumeration in `tests/canon/test_decode.py`, rather than
shipping as an `if` no input can reach.

**Two entry points, and still no depth arithmetic here.** AD-5 gives every stage the signature
`Stage(text, ctx)` with no depth in it, and a stage genuinely does not know how deep it is being
run. So this module declares *what to do at the ceiling* — `run_at_ceiling`, which decides exactly
as `run` does and then replaces nothing, reporting a would-have-decoded candidate under
`CEILING_NAME` — and `canon/pipeline.py` decides *when* that is the right entry point. The one
comparison this story owns, `depth >= ceiling`, is in the runner; the ceiling itself is
`CanonContext.ceiling`, defaulted once in `canon/pipeline.py::DEFAULT_CEILING`. No literal here,
no constant here, and nothing here reads a depth.

The text an accepted decode inserts has not been through steps 1 to 3, because this is the last
step: it can carry a ligature or a zero-width character. AD-4 closes that outside this module, by
canonicalizing the decoded segment as an independent document at `depth + 1` — the runner's job,
because the runner is the only thing that knows what `depth` is.
"""

from __future__ import annotations

import base64
import math
from collections import Counter
from dataclasses import dataclass
from typing import Final

from nbc.canon.edits import Report, build_reported_edits
from nbc.schema import CanonContext, StageResult

__all__ = (
    "BASE64",
    "CEILING_NAME",
    "CONSTANTS",
    "HEX",
    "NAME",
    "ORDER",
    "CandidateTest",
    "Constant",
    "decide",
    "passes_candidate_test",
    "run",
    "run_at_ceiling",
    "shannon_bits_per_char",
)

NAME: Final[str] = "decode"

CEILING_NAME: Final[str] = "decode-ceiling"
"""The stage name a candidate carries when the recursion ceiling is the only thing that refused it.

AD-6 asks for a name distinct from `NAME` so a reader of the trace can tell "the layer would have
opened this and was not allowed to" from "the layer examined this and decided it was not an
encoding". The two are the same shape — an `Edit` whose `before` equals its `after` — and without
two names they would be the same entry. They can occur in one document at one depth, interleaved,
which is why the name travels per reported span and not per stage call.
"""


# --- the declared block ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Constant:
    """One declared candidate-test constant: its name, what it counts, and which side it bounds.

    `side` is the half of AD-18 that is easiest to lose. `min_encoded_chars` and
    `min_entropy_bits_per_char` are **inclusive lower bounds**: a run exactly at the number passes.
    `alphabet` bounds nothing; it decides membership. Spelling that out is what stops a later
    reader from reading `min_` as exclusive and moving every boundary by one.
    """

    name: str
    unit: str
    side: str


CONSTANTS: Final[tuple[Constant, ...]] = (
    Constant(
        name="alphabet",
        unit="code points, as an explicit set",
        side="membership",
    ),
    Constant(
        name="min_encoded_chars",
        unit="characters of the candidate run, not bytes of its output",
        side="inclusive lower bound",
    ),
    Constant(
        name="min_entropy_bits_per_char",
        unit="bits per character; Shannon, base 2, unnormalized, over the candidate run",
        side="inclusive lower bound",
    ),
)
"""Every constant of the candidate test, with its unit and its side, in one place.

Epic 4 writes these names and units verbatim into `results.json`. A test asserts this tuple covers
exactly the value-carrying fields of `CandidateTest`, so a constant cannot be added without its
unit and a unit cannot outlive the constant it describes.
"""


@dataclass(frozen=True, slots=True)
class CandidateTest:
    """The candidate test for one encoding. One instance per encoding, both declared below."""

    encoding: str
    alphabet: frozenset[str]
    min_encoded_chars: int
    min_entropy_bits_per_char: float

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError(f"encoding must be a non-empty name, got {self.encoding!r}")
        if not isinstance(self.alphabet, frozenset) or not self.alphabet:
            raise ValueError(f"{self.encoding}: alphabet must be a non-empty frozenset")
        for char in self.alphabet:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError(f"{self.encoding}: alphabet holds {char!r}, not one code point")
        if isinstance(self.min_encoded_chars, bool) or not isinstance(self.min_encoded_chars, int):
            raise ValueError(
                f"{self.encoding}: min_encoded_chars counts characters and must be an int, "
                f"got {self.min_encoded_chars!r}"
            )
        if self.min_encoded_chars < 1:
            raise ValueError(
                f"{self.encoding}: min_encoded_chars must be at least 1, got "
                f"{self.min_encoded_chars!r}"
            )
        floor = self.min_entropy_bits_per_char
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise ValueError(f"{self.encoding}: min_entropy_bits_per_char must be a real number")
        limit = math.log2(len(self.alphabet))
        if not 0.0 <= floor <= limit:
            # The evidence for the floor is the alphabet it is measured over: a run drawn from an
            # n-symbol alphabet cannot exceed log2(n) bits per character, so a floor above that
            # refuses every candidate and a negative one refuses none. Compared, not recorded.
            raise ValueError(
                f"{self.encoding}: min_entropy_bits_per_char {floor!r} is outside "
                f"[0, log2({len(self.alphabet)}) = {limit:.4f}], the range a run over this "
                f"alphabet can occupy"
            )
        object.__setattr__(self, "min_entropy_bits_per_char", float(floor))


HEX: Final[CandidateTest] = CandidateTest(
    encoding="hex",
    alphabet=frozenset("0123456789abcdefABCDEF"),
    min_encoded_chars=16,
    min_entropy_bits_per_char=2.5,
)
"""The hex candidate test.

**Alphabet**, stated member by member as AD-18 requires: the sixteen hex digits in **both** cases.
There is no padding character in hex, so `=` is **not** a member. URL-safe `-_` are **not** members.
Whitespace is **not** a member, so a hex run never spans a line break.

**`min_encoded_chars = 16`** — sixteen characters of run, eight bytes of output. Hex is a
sixteen-character alphabet and hardly ever claims ordinary text, so this floor costs nothing: over
this repository's own `src/**/*.py` it leaves **one** candidate run, against five for base64 at a
floor half again as long. `tests/canon/test_decode.py` re-measures both, so the sentence is
compared to what it describes rather than filed beside it.

**`min_entropy_bits_per_char = 2.5`** — a hex run of ASCII text measures about 2.9 to 3.3 bits per
character, and the floor sits below that. What it refuses is the degenerate run: sixteen zeros
decode to eight NUL bytes, which *are* valid UTF-8, so without an entropy floor a run of zeros in a
memory dump would be replaced by NUL characters. Content hashes clear this floor comfortably
(sha-256 hex measures about 3.7) and are refused by strict UTF-8 instead, which is the intended
division of labour: entropy refuses the degenerate, UTF-8 refuses the random.
"""

BASE64: Final[CandidateTest] = CandidateTest(
    encoding="base64",
    alphabet=frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    ),
    min_encoded_chars=24,
    min_entropy_bits_per_char=3.0,
)
"""The base64 candidate test.

**Alphabet**, stated member by member: the sixty-four standard characters of RFC 4648 §4, `+` and
`/` included, **plus `=`**, which is a member so that a run carries its own padding rather than
ending one character before it. URL-safe `-_` (RFC 4648 §5) are **not** members: `-` and `_` are
pervasive in ordinary prose and in source identifiers, and admitting them would make every
kebab-case and snake_case name part of a candidate. Whitespace is **not** a member, so PEM-style
base64 wrapped across lines is seen as one run per line — a stated limitation, not an oversight.

**`min_encoded_chars = 24`** — twenty-four characters of run, eighteen bytes of output. The base64
alphabet is sixty-four characters wide and therefore claims ordinary CamelCase identifiers, so this
floor is where the noise stops rather than where decoding stops being possible: over this
repository's own `src/**/*.py`, a floor of 16 makes 193 candidate runs and a floor of 24 makes 5.
`tests/canon/test_decode.py` re-measures that and requires the reduction, so the number is compared
to its evidence. The cost, stated: a plaintext shorter than eighteen bytes is not recovered from
base64.

**`min_entropy_bits_per_char = 3.0`** — base64 of ASCII text measures about 4.4 to 5.2 bits per
character and the floor sits well below that, because a *short* base64 run of repetitive plaintext
measures lower. What it refuses is again the degenerate run: twenty-four `A` characters decode to
eighteen NUL bytes, valid UTF-8, and nothing but the entropy floor stands between that and a
document full of NULs.
"""

ORDER: Final[tuple[CandidateTest, ...]] = (HEX, BASE64)
"""Hex first, because its alphabet is a strict subset of base64's and the more specific test wins.

A test asserts both halves of that sentence — the order of this tuple and the subset relation —
because the order is only correct *given* the relation, and a premise that is only ever written in
prose is a premise nobody checks.
"""


# --- the test ------------------------------------------------------------------------------------


def shannon_bits_per_char(run: str) -> float:
    """Shannon entropy of `run` over its own characters, in bits per character, unnormalized.

    Base 2, and **not** divided by `log2(len(alphabet))`: AD-18 names the unit precisely because a
    normalized score and a bits-per-character score put the same declared number in two different
    places. The empty string has no distribution and scores `0.0`.

    The sum runs over characters in sorted order so the floating-point result cannot depend on
    dictionary insertion order, which is one of the determinism rules `canon/` is held to.
    """
    length = len(run)
    if length == 0:
        return 0.0
    counts = Counter(run)
    total = 0.0
    for _, count in sorted(counts.items()):
        p = count / length
        total -= p * math.log2(p)
    return total


def _raw_bytes(run: str, test: CandidateTest) -> bytes | None:
    """The bytes `run` encodes under `test`, or `None` if it does not structurally encode any.

    `ValueError` covers both structural failures this can meet: `binascii.Error` (bad base64
    length, misplaced or excess padding) is a `ValueError`, and `bytes.fromhex` raises `ValueError`
    on an odd-length run. Catching the base class is deliberate rather than lax — catching only
    `binascii.Error` would leave the hex branch's own failure uncaught.
    """
    if test.encoding == HEX.encoding:
        try:
            return bytes.fromhex(run)
        except ValueError:
            return None
    if test.encoding == BASE64.encoding:
        try:
            return base64.b64decode(run, validate=True)
        except ValueError:
            return None
    raise ValueError(
        f"no decoder is declared for {test.encoding!r}; the declared encodings are "
        f"{[t.encoding for t in ORDER]}"
    )


def passes_candidate_test(run: str, test: CandidateTest) -> bool:
    """Whether `run` passes the declared candidate test: all three constants, in one place.

    Alphabet, length and entropy — the whole of AD-18's test and nothing else. It is one function
    rather than three conditions spread across the scan and the decision, because a caller holding
    only part of the test would be applying a different test under the same name; Story 2.4 will
    hold exactly one condition of its own (`depth < ceiling`) and needs this one whole.
    """
    return (
        set(run) <= test.alphabet
        and len(run) >= test.min_encoded_chars
        and shannon_bits_per_char(run) >= test.min_entropy_bits_per_char
    )


def decide(run: str, test: CandidateTest) -> str | None:
    """The decoded text for `run` under `test`, or `None` if it is refused.

    Every condition AD-18 names, in the order that lets each one be the reason: the declared
    candidate test, then a structural decode, then **strict** UTF-8. `bytes.decode("utf-8")` is
    strict by default and raises `UnicodeDecodeError`, which is a `ValueError` and not an
    `OSError` — caught by its own name here so a structural failure and a text failure never share
    a branch.

    Public, and it applies the whole test rather than trusting the scan to have applied part of
    it: a test can then put one candidate under the *other* encoding's test and see what the
    pipeline's ordering actually bought.
    """
    if not passes_candidate_test(run, test):
        return None
    raw = _raw_bytes(run, test)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --- the scan ------------------------------------------------------------------------------------


def _runs(
    text: str, alphabet: frozenset[str], *, start: int = 0, end: int | None = None
) -> list[tuple[int, int]]:
    """Every maximal run of `alphabet` characters in `text[start:end]`, as spans into `text`.

    The window is passed as indices rather than as a slice on purpose: the inner scan runs over
    every base64 run in the document, and slicing there would copy each one — doubling the peak
    memory of a document that is mostly one long run, on the path whose p50 and p95 the run
    publishes.
    """
    stop = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    open_at = -1
    for index in range(start, stop):
        if text[index] in alphabet:
            if open_at < 0:
                open_at = index
        elif open_at >= 0:
            spans.append((open_at, index))
            open_at = -1
    if open_at >= 0:
        spans.append((open_at, stop))
    return spans


Decision = tuple[int, int, str | None]
"""`(start, end, decoded)`: one examined candidate. `decoded` is `None` when it was refused.

The decision is separate from the report it becomes because the two entry points below differ only
in what they *do* with it: below the ceiling an accepted decision replaces its span, at the ceiling
it replaces nothing and is reported under `CEILING_NAME` instead. Deciding once and reporting twice
is what makes `ceiling_hit` mean "solely because of the depth" — the candidate went through the
whole of AD-18, structural decode and strict UTF-8 included, before the depth was consulted.
"""


def _decisions(text: str) -> list[Decision]:
    """The reported spans of `text`, in order: one per candidate, decoded or refused.

    The length constant appears here as well as inside the candidate test, and reads the same
    declared value: it decides which runs are **reported**, where the test decides which are
    decoded. A run below it is ordinary text and the trace says nothing about it at all.

    Walks the maximal base64 runs, because every maximal hex run lies inside exactly one of them.
    Inside each, hex is offered first. If hex accepted anything, the run is not re-offered to
    base64 and hex's decisions are what the trace carries. If hex accepted nothing, the wider base64
    candidate is reported instead where the run is long enough to be one — one span, one report —
    and hex's refusals stand alone only where it is not.
    """
    reported: list[Decision] = []

    for start, end in _runs(text, BASE64.alphabet):
        hex_spans = [
            span
            for span in _runs(text, HEX.alphabet, start=start, end=end)
            if span[1] - span[0] >= HEX.min_encoded_chars
        ]
        hex_decisions = [(a, b, decide(text[a:b], HEX)) for a, b in hex_spans]
        accepted_hex = any(decoded is not None for _, _, decoded in hex_decisions)

        if not accepted_hex and end - start >= BASE64.min_encoded_chars:
            reported.append((start, end, decide(text[start:end], BASE64)))
            continue

        reported.extend(hex_decisions)

    return reported


def run(text: str, ctx: CanonContext) -> StageResult:
    """Replace every accepted candidate in place, and report every refused one as a no-op edit.

    This is step 4 below the recursion ceiling. `ctx.trace_enabled` is deliberately not consulted:
    these edits are how `canon/pipeline.py` learns which spans it must canonicalize as independent
    documents at `depth + 1`, so switching them off would change the canonical text rather than
    only the trace. The runner drops them from the document's trace when tracing is off.
    """
    reports: list[Report] = [
        (start, end, text[start:end] if decoded is None else decoded, NAME)
        for start, end, decoded in _decisions(text)
    ]
    new_text, edits = build_reported_edits(text, reports)
    return StageResult(text=new_text, edits=edits)


def run_at_ceiling(text: str, ctx: CanonContext) -> StageResult:
    """Step 4 at the recursion ceiling: decide everything, replace nothing, report why.

    Every candidate goes through exactly the decision `run` applies, because AD-6's `ceiling_hit`
    is true only for a candidate refused **solely** because of the depth. A candidate this stage
    would have refused anyway is reported under `NAME`, as an ordinary AD-18 rejection, and does
    not make the run a ceiling hit; a candidate that would have decoded is reported under
    `CEILING_NAME`. The text is returned unchanged either way.

    The cost, stated rather than discovered: at the ceiling the bytes are decoded and thrown away,
    because there is no way to know whether the depth was the *only* reason without doing so. That
    is one extra level of decoding, not an unbounded one, and it is what the amplification bound
    the ceiling exists for still holds against.
    """
    reports: list[Report] = [
        (start, end, text[start:end], NAME if decoded is None else CEILING_NAME)
        for start, end, decoded in _decisions(text)
    ]
    new_text, edits = build_reported_edits(text, reports)
    return StageResult(text=new_text, edits=edits)

