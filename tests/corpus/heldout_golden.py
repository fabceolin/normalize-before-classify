"""Committed before-and-after literals for every chain in `HELDOUT_CHAINS`.

Not a test module: the data `test_heldout.py` runs. The same fixture discipline
`dressing_golden.py` applies to the bound registry, applied to the held-out one for a sharper
reason. A held-out encoding is the only thing in this corpus the layer is *expected* to fail on, so
a defect in one of these functions produces exactly the result the story predicts. Nothing in the
recovery numbers would look wrong. A committed literal is what separates "the layer cannot decode
base32" from "this module does not emit base32".

**Where these literals came from.** Not from `nbc.corpus.heldout`.

- `base32` from coreutils `base32(1)`, cross-checked against a hand-written bit-packer that reads
  five bits at a time out of `44 6f 20 69 74 2e` and pads to a multiple of eight characters.
- `url_percent` written out by hand from RFC 3986: only the space is outside the unreserved set,
  so only the space is escaped, and `.` stays literal because `-._~` are unreserved.
- `rot13` written out by hand: `D`->`Q`, `o`->`b`, `i`->`v`, `t`->`g`, and the space and the full
  stop are not letters.

**The same six-character payload as the bound fixtures**, `"Do it."`, so the two files can be read
against each other. It is far below every candidate floor the layer declares, which is again
deliberate: these pin the **encodings**, not what the layer does with them. What the layer does is
measured over `heldout.PROBE_PAYLOADS`, in `test_heldout.py`.
"""

from __future__ import annotations

PAYLOAD = "Do it."
"""The one payload every fixture below encodes, shared with `dressing_golden.py` on purpose."""

GOLDEN: dict[str, str] = {
    # RFC 4648 section 6. Six bytes is not a multiple of five, so the last group is padded: ten
    # significant characters and six `=`. Every character is in base64's alphabet too, which is
    # the mechanism `probes: decode` names -- the layer is offered this whole string.
    "base32": "IRXSA2LUFY======",
    # RFC 3986. `D`, `o`, `i`, `t` and `.` are unreserved and survive; the space is not and becomes
    # `%20`. Two hex digits behind a character the layer scans nothing for, which is the mechanism
    # `probes: partial` names.
    "url_percent": "Do%20it.",
    # Letters rotated thirteen places, everything else untouched. No marker, no length change, no
    # change in the character classes present -- the mechanism `probes: none` names.
    "rot13": "Qb vg.",
}
"""One committed literal per chain in `HELDOUT_CHAINS`, keyed as `render_chain` spells it.

`test_heldout.py` asserts the key set is exactly the declared held-out chain set, so an encoding
added to the registry without a fixture fails there rather than shipping unexercised.
"""
