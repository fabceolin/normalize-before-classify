"""Committed before-and-after literals for every chain in `CHAINS`.

Not a test module: the data `test_dressings.py` runs. AD-3 requires one golden fixture per
two-link chain **so the fold direction is executable rather than asserted** -- `["base64",
"homoglyph"]` and `["homoglyph", "base64"]` are different documents, and a project that only ever
wrote the rule down in prose would find out which one it built when someone tried to reproduce a
cell.

**Where these literals came from, and why it matters.** They were not produced by running
`nbc.corpus.dressings`. The base64 values come from coreutils `base64(1)` piped through itself
once per link; the hex value from `xxd -p`; the homoglyph substitutes were read out of
`src/nbc/canon/data/confusables-15.1.0.json` directly; the zero-width value is the declared
insertion written by hand. So the comparison in `test_dressings.py` has two independent sides,
which is the only thing that makes a golden a golden.

**Why the payload is six characters.** `"Do it."` is short enough that every literal below can be
checked by eye against the rule it encodes -- eight base64 characters, twelve hex characters, five
inserted separators. It is far too short for the layer to decode (`decode.BASE64` requires
twenty-four characters of run), and that is deliberate: these fixtures pin the **fold**, not the
layer. What the layer does with a dressed document is `tests/corpus/test_matrix.py`'s subject and,
for the round trip, story 3.4's.
"""

from __future__ import annotations

PAYLOAD = "Do it."
"""The one payload every fixture below dresses."""

ZERO_WIDTH = "​"
"""ZERO WIDTH SPACE, written out here so the literals below are readable as literals.

Deliberately a second spelling of the character `dressings.zero_width_character()` resolves out of
`invisible.ZERO_WIDTH`: if this file imported that function, the fixture and the code under test
would be the same side of the comparison. `test_dressings.py` asserts the two agree.
"""

GOLDEN: dict[str, str] = {
    # --- the five singletons of FR2 -------------------------------------------------------------
    # The empty chain is the identity element of the fold, so `clean` is the payload itself.
    "clean": "Do it.",
    # `base64 -w0` of the six ASCII bytes: 6 bytes -> 8 characters, no padding needed.
    "base64": "RG8gaXQu",
    # `xxd -p`: 44 6f 20 69 74 2e.
    "hex": "446f2069742e",
    # o -> U+03BF GREEK SMALL LETTER OMICRON, i -> U+037A GREEK YPOGEGRAMMENI. `D`, the space,
    # `t` and `.` have no Cyrillic or Greek prototype in the vendored table and are left alone,
    # which is what makes this fixture also a fixture for the "leaves the rest untouched" half.
    "homoglyph": "Dο ͺt.",
    # Between every adjacent pair, never around: six characters, five separators.
    "zero_width": "D" + ZERO_WIDTH + "o" + ZERO_WIDTH + " " + ZERO_WIDTH + "i" + ZERO_WIDTH
    + "t" + ZERO_WIDTH + ".",
    # --- the four two-link chains, which are what AD-3 asks for ---------------------------------
    # `base64 | base64`: "RG8gaXQu" (8 characters, 8 bytes) -> 12 characters with one `=`.
    "base64+base64": "Ukc4Z2FYUXU=",
    # base64(zero_width(payload)). The inner text is 6 ASCII characters and 5 ZERO WIDTH SPACEs,
    # which are three UTF-8 bytes each: 6 + 15 = 21 bytes -> 28 base64 characters, no padding.
    # This is AD-20's `zero_width+base64`, and the fold reading is the outer link last.
    "zero_width+base64": "ROKAi2/igIsg4oCLaeKAi3TigIsu",
    # homoglyph(base64(payload)) -- AD-3's own worked example. Of "RG8gaXQu": G -> U+050C
    # CYRILLIC CAPITAL LETTER KOMI SJE, a -> U+03B1 ALPHA, X -> U+03A7 CHI, u -> U+03C5 UPSILON.
    # `R`, `8`, `g` and `Q` have no prototype. Reversing the chain would base64-encode Greek and
    # Cyrillic code points instead and produce a completely different string, which is the whole
    # reason this fixture is committed.
    "base64+homoglyph": "RԌ8gαΧQυ",
    # zero_width(hex(payload)): twelve hex characters, eleven separators. Written out link by
    # link rather than with a `join`, which is the operation under test and would put both sides
    # of the comparison in the same place.
    "hex+zero_width": "4" + ZERO_WIDTH + "4" + ZERO_WIDTH + "6" + ZERO_WIDTH + "f"
    + ZERO_WIDTH + "2" + ZERO_WIDTH + "0" + ZERO_WIDTH + "6" + ZERO_WIDTH + "9"
    + ZERO_WIDTH + "7" + ZERO_WIDTH + "4" + ZERO_WIDTH + "2" + ZERO_WIDTH + "e",
    # --- the chain that nests past the ceiling ---------------------------------------------------
    # `base64` four times: 8 -> 12 -> 16 -> 24 characters. Its `encoding_depth` is 4, one more
    # than `DEFAULT_CEILING`, which is why AD-20 requires a chain of this shape to exist.
    "base64+base64+base64+base64": "Vld0ak5Gb3lSbGxWV0ZVOQ==",
}
"""One committed literal per chain in `CHAINS`, keyed by the chain as `render_chain` spells it.

Keyed by the rendered name rather than by the tuple so a reader of this file sees the same string
that appears in an item id and in the dressing axis of the table. `test_dressings.py` asserts the
key set is exactly the declared chain set, so a chain added to `CHAINS` without a fixture fails
here rather than shipping unexercised.
"""
