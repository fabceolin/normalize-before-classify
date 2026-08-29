"""The dressings: four named pure functions, one registry, and the fold that composes them.

**The fold, and its direction.** `dress(payload, chain)` is `reduce(apply, chain, payload)`, so
`["base64", "homoglyph"]` means `homoglyph(base64(payload))` -- the later link **wraps** the
earlier. "Applied left to right" never said left to right *of what*, which is why AD-3 spells the
reduction out and why `tests/corpus/dressing_golden.py` commits one literal per two-link chain:
the direction is executable rather than asserted. The empty chain is `clean` and returns the
payload object itself.

**Every function here is pure and deterministic.** No RNG, no clock, no network, no module-level
mutable state, and nothing that depends on set or dict iteration order -- the one place an
iteration order could have leaked in is the many-to-one inverse of the confusables table, and it
is resolved by a `min` over a sorted iteration rather than by whichever key arrived first. AD-1's
byte-identical claim rests on this: the corpus text is a function of the payload and the chain and
of nothing else.

**The characters come from the layer, not from a second list.** The homoglyph substitutes are
drawn from the vendored table in `canon/data/` and the zero-width character from
`canon/stages/invisible.py`. This module therefore imports `nbc.canon` on purpose -- story 3.4's
requirement, and the exact inverse of the rule story 3.5 imposes on `corpus/heldout.py`, which
must import nothing from the layer at all. The two rules together are what keep the bound and the
held-out halves of the table from collapsing into each other: a bound dressing cannot emit a
character the layer has never heard of, and a held-out encoding cannot be built out of the layer's
own alphabet.

**What is not here.** The round-trip contract -- `canonicalize(d(p)).text == canonicalize(p).text`
over the built corpus, under a raised test-only ceiling -- is story 3.4's, and so is the
statement of its scope. This module supplies the character source that makes it satisfiable; it
does not claim the contract holds.
"""

from __future__ import annotations

import base64 as _base64
import functools
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Final, Mapping, Sequence

from nbc.canon import confusables_table
from nbc.canon.stages import invisible
from nbc.corpus.matrix import CLEAN_CHAIN, CorpusMatrixInvalid

__all__ = [
    "DRESSINGS",
    "Dressing",
    "ZERO_WIDTH_NAME",
    "apply",
    "dress",
    "homoglyph",
    "homoglyph_substitutes",
    "to_base64",
    "to_hex",
    "zero_width",
    "zero_width_character",
]

Dressing = Callable[[str], str]
"""The one shape a dressing has: text in, text out, and nothing else in the signature.

A dressing that took a parameter would be a family of dressings, and the chain would have to carry
its arguments; AD-3's `dressing` field is a list of names, so the arguments would have nowhere to
live and the item id would name several different documents.
"""


# --- the encoded dressings ----------------------------------------------------------------------


def to_base64(text: str) -> str:
    """Standard RFC 4648 §4 base64 of the UTF-8 bytes, with padding, and nothing around it.

    The alphabet and the `=` padding are exactly what `canon.stages.decode.BASE64` declares as its
    candidate alphabet, so a dressed run is a candidate the layer can offer to its decoder. URL-safe
    base64 is deliberately not used here: it is not in the layer's alphabet, and an encoding the
    layer was never written against belongs in story 3.5's held-out registry, where the result of
    not recovering it is reported instead of being mistaken for a bug.

    **No wrapper.** No `decode this:` prefix, no delimiters. A wrapper would add plaintext that is
    not the payload, and the recall column would then be partly a measurement of whether the
    classifier reacts to the wrapper. The cost, stated: a payload shorter than eighteen bytes
    produces a run below the layer's `min_encoded_chars` and is not recovered -- the same floor
    `decode.py` already publishes.
    """
    return _base64.b64encode(text.encode("utf-8")).decode("ascii")


def to_hex(text: str) -> str:
    """Lowercase hex of the UTF-8 bytes, two characters per byte, and nothing around it.

    Lowercase because `bytes.hex()` is, and because the choice must be *a* choice rather than a
    property of the input; the layer's hex alphabet admits both cases, so the dressing does not
    depend on which one is used and a reviewer can read the corpus rows without wondering.
    """
    return text.encode("utf-8").hex()


# --- the character dressings --------------------------------------------------------------------


ZERO_WIDTH_NAME: Final[str] = "ZERO WIDTH SPACE"
"""Which of the layer's zero-width characters this dressing inserts, named as Unicode names it.

Chosen by name out of `invisible.ZERO_WIDTH`, a committed closed vocabulary of code-point/name
pairs, rather than by an index into it -- an index would silently start naming a different
character the moment a code point were added -- and rather than by a literal `\\u200b`, which would
be a second declaration of a character the layer already owns. `zero_width_character()` asserts
the resolved character is in `invisible.REMOVED`, so the claim that the layer strips what this
dressing inserts is compared rather than recorded.
"""


def zero_width_character() -> str:
    """The character `zero_width` inserts, looked up in the layer's own declared set.

    The membership assertion is the point: `invisible.ZERO_WIDTH` is what the stage documents and
    `invisible.REMOVED` is what it actually deletes, and they are built in two steps. A name that
    resolved to a character the stage does not remove would make every `zero_width` row
    unrecoverable, and the symptom would be a recall number rather than an error.
    """
    matches = [code_point for code_point, name in invisible.ZERO_WIDTH if name == ZERO_WIDTH_NAME]
    if len(matches) != 1:
        raise CorpusMatrixInvalid(
            f"{len(matches)} characters in nbc.canon.stages.invisible.ZERO_WIDTH are named "
            f"{ZERO_WIDTH_NAME!r}; the zero-width dressing needs exactly one"
        )
    character = chr(matches[0])
    if character not in invisible.REMOVED:
        raise CorpusMatrixInvalid(
            f"U+{matches[0]:04X} {ZERO_WIDTH_NAME} is declared in ZERO_WIDTH but is not in "
            f"invisible.REMOVED, so the layer would not strip what this dressing inserts"
        )
    return character


def zero_width(text: str) -> str:
    """Insert the declared zero-width character between every pair of adjacent code points.

    Between, not around: a leading or trailing insertion would change what a downstream encoding
    sees at the boundary without changing anything about the interior, and "between every pair" is
    the rule with no boundary case to get wrong. A one-character text is returned unchanged, which
    is what `str.join` does and what the rule says.

    Total rather than sampled, for the same reason `homoglyph` is total: every partial rule needs a
    position selector, and every deterministic position selector is a free parameter that would
    then have to be declared and defended.
    """
    return zero_width_character().join(text)


@functools.lru_cache(maxsize=8)
def homoglyph_substitutes(data_dir: Path = confusables_table.DATA_DIR) -> Mapping[str, str]:
    """ASCII character -> the confusable this dressing substitutes for it.

    The inverse of the vendored table, which maps confusable to ASCII and is **many-to-one**: at
    UCD 15.1.0 it holds 95 entries, 88 of which have a single-character value, and those 88
    collapse onto 45 distinct ASCII characters. Two rules resolve that, and both are consequences
    rather than preferences:

    - the **lowest code point** among the keys mapping to a given ASCII character wins, computed by
      a `min` over a sorted iteration so the result does not depend on dict order;
    - entries whose value is more than one character (`Ы -> bl`, `Ӕ -> AE`, five others) are
      **skipped**, because a one-to-many substitution has no exact inverse and the layer's output
      would then depend on where the dressing chose to split.

    Cached by data directory rather than recomputed per row: `confusables_table.load` validates on
    every call and the corpus builder dresses tens of thousands of rows. The cache is bounded, and
    what it holds is an immutable mapping that is a pure function of its key, so it is a memo
    rather than state.
    """
    table = confusables_table.load(data_dir)
    inverse: dict[str, str] = {}
    for confusable, ascii_form in sorted(table.mapping.items()):
        if len(ascii_form) != 1:
            continue
        current = inverse.get(ascii_form)
        if current is None or confusable < current:
            inverse[ascii_form] = confusable
    return MappingProxyType(inverse)


def homoglyph(text: str) -> str:
    """Substitute every character that has a confusable in the vendored table.

    **Total substitution, and what it costs.** A partial rule would need a position selector --
    first n, every k-th, a seeded sample -- and every deterministic selector is a free parameter
    that would have to be declared, pinned and defended. Total substitution has no parameter. The
    price, stated rather than discovered: a fully substituted document is far more conspicuous than
    a real homoglyph attack, which usually swaps one letter of one word. This dressing therefore
    measures whether the layer's confusable mapping covers the substitutes it can emit, not whether
    a subtle attack evades a classifier -- which is the same thing FR19's caveat already says about
    every bound chain.

    A text holding nothing substitutable comes back unchanged.
    """
    substitutes = homoglyph_substitutes()
    return "".join(substitutes.get(character, character) for character in text)


# --- the registry and the fold --------------------------------------------------------------------


DRESSINGS: Final[Mapping[str, Dressing]] = MappingProxyType(
    {
        "base64": to_base64,
        "hex": to_hex,
        "homoglyph": homoglyph,
        "zero_width": zero_width,
    }
)
"""Every dressing this corpus can build, by the name that appears in a chain and an item id.

A closed registry rather than a name convention or a module scan: membership is a lookup, so a
chain naming something that is not here aborts with the name it asked for instead of finding a
helper function whose name happened to match a pattern.
"""


def apply(text: str, name: str) -> str:
    """One link of the fold: the dressing `name`, applied to `text`.

    The argument order is `reduce`'s, accumulator first, which is why this is a named function
    rather than a lambda: `reduce(apply, chain, payload)` reads as the rule AD-3 states, and a
    reader can check the direction by reading the two lines rather than by remembering which
    argument `functools.reduce` passes first.
    """
    dressing = DRESSINGS.get(name)
    if dressing is None:
        raise CorpusMatrixInvalid(
            f"the chain names the dressing {name!r}, which is not in the registry "
            f"({sorted(DRESSINGS)}); a new dressing is a new named function in dressings.py and "
            f"nothing else"
        )
    return dressing(text)


def dress(payload: str, chain: Sequence[str] = CLEAN_CHAIN) -> str:
    """AD-3's fold: `reduce(apply, chain, payload)`. The later link wraps the earlier.

    `dress(p, ["base64", "homoglyph"])` is `homoglyph(to_base64(p))`. The empty chain returns
    `payload` itself, unchanged and `is`-equal, which is what makes `clean` the identity element
    rather than a dressing that happens to be a no-op.
    """
    return functools.reduce(apply, chain, payload)
