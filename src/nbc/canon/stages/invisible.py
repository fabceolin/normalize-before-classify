"""Step 1: remove zero-width characters and bidirectional controls.

**Why first.** These characters exist to be unseen. Left in place they split every run the later
steps look for: a confusable letter separated from its neighbours still maps, but a base64 run with
a zero-width joiner in the middle is no longer a base64 run, and Story 2.3's candidate test would
never see it. Removing them first is what makes steps 2 to 4 see the text a reader sees.

**The set is closed, declared, and cross-checked.** There is no Unicode property for "zero-width",
so that family is a committed list, each entry carrying the name it has in the standard. The
bidirectional family is not a list at all in spirit: it is every code point whose *bidirectional
class* is one of the nine explicit directional controls, and a test walks the whole code-point
range and asserts the committed tuple is exactly that set. So the one family that can be derived
from the interpreter is checked against the interpreter, and the one that cannot says so.

**What is deliberately not removed**, so a reader knows the boundary rather than guessing it:

- `U+00AD` SOFT HYPHEN — invisible in most rendering and a genuine splitter, but it is neither
  zero-width nor directional, and the two families this step declares are the two the story names.
- `U+2061..U+2064`, the invisible mathematical operators.
- `U+E0000..U+E007F`, the deprecated tag characters, and the variation selectors.

Each of those is a real evasion this layer does not cover. They belong to the held-out block
(Epic 3), which exists precisely so recovery is not measured only on encodings the layer was
written against. Widening this set later is a change to a published number, not a bug fix.
"""

from __future__ import annotations

from typing import Final

from nbc.canon.edits import build_edits, map_code_points
from nbc.schema import CanonContext, StageResult

__all__ = [
    "BIDI_CONTROL",
    "BIDI_CONTROL_CLASSES",
    "NAME",
    "REMOVED",
    "ZERO_WIDTH",
    "run",
]

NAME: Final[str] = "invisible"

ZERO_WIDTH: Final[tuple[tuple[int, str], ...]] = (
    (0x061C, "ARABIC LETTER MARK"),
    (0x180E, "MONGOLIAN VOWEL SEPARATOR"),
    (0x200B, "ZERO WIDTH SPACE"),
    (0x200C, "ZERO WIDTH NON-JOINER"),
    (0x200D, "ZERO WIDTH JOINER"),
    (0x200E, "LEFT-TO-RIGHT MARK"),
    (0x200F, "RIGHT-TO-LEFT MARK"),
    (0x2060, "WORD JOINER"),
    (0xFEFF, "ZERO WIDTH NO-BREAK SPACE"),
)
"""Zero-width format characters, by code point and Unicode name.

The three directional *marks* (`U+061C`, `U+200E`, `U+200F`) sit here rather than with the
controls below because they are zero-width and their bidirectional class is an ordinary strong
class (`AL`, `L`, `R`), not one of the nine explicit controls. Putting them in the other family
would break the completeness check that makes that family derivable.
"""

BIDI_CONTROL_CLASSES: Final[frozenset[str]] = frozenset(
    {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
)
"""The nine explicit directional formatting classes, as `unicodedata.bidirectional` spells them."""

BIDI_CONTROL: Final[tuple[tuple[int, str], ...]] = (
    (0x202A, "LEFT-TO-RIGHT EMBEDDING"),
    (0x202B, "RIGHT-TO-LEFT EMBEDDING"),
    (0x202C, "POP DIRECTIONAL FORMATTING"),
    (0x202D, "LEFT-TO-RIGHT OVERRIDE"),
    (0x202E, "RIGHT-TO-LEFT OVERRIDE"),
    (0x2066, "LEFT-TO-RIGHT ISOLATE"),
    (0x2067, "RIGHT-TO-LEFT ISOLATE"),
    (0x2068, "FIRST STRONG ISOLATE"),
    (0x2069, "POP DIRECTIONAL ISOLATE"),
)
"""Every code point whose bidirectional class is in `BIDI_CONTROL_CLASSES`.

Committed rather than computed at import so the layer's domain is readable without running it, and
asserted against the interpreter's own tables by a test that scans the whole code-point range. A
Unicode revision that adds a directional control fails that test instead of silently widening the
layer, which is the same coupling the vendored confusables revision has.
"""

REMOVED: Final[frozenset[str]] = frozenset(
    chr(code_point) for code_point, _ in ZERO_WIDTH + BIDI_CONTROL
)
"""The characters this step deletes. Epic 3's zero-width dressing draws from here, not a second list."""


def _replace(char: str) -> str | None:
    return "" if char in REMOVED else None


def run(text: str, ctx: CanonContext) -> StageResult:
    """Delete every declared invisible character, coalescing adjacent deletions into one span."""
    new_text, edits = build_edits(
        text, map_code_points(text, _replace), stage=NAME, trace=ctx.trace_enabled
    )
    return StageResult(text=new_text, edits=edits)
