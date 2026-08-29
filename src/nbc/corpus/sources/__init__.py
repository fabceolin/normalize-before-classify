"""Hand-authored corpus material, and the only place in this project where any exists.

FR5.1 draws B-chat from the benign rows of a pinned public dataset and grants exactly one
exception: messages legitimately containing a **JWT, a content hash, a data URI or an SSH public
key**. No conversational dataset carries those, and they are precisely the legitimate encoded
content a canonicalization layer turns into false positives, so a counter-metric with none of them
would be a counter-metric measured over the wrong text.

The allowance is closed in three ways at once, because "hand-authored, but only a little" is not a
constraint anybody can check:

- the **kinds** are a closed vocabulary, and a fifth kind is a code change rather than a data entry;
- the **count** is declared in `pins.toml`'s `[benign_frame.b_chat]` and compared against what this
  package actually holds, so an item added here without widening the frame fails the build;
- every item is **verified structurally against its declared kind** -- a JWT header that is not
  base64url-encoded JSON with an `alg`, an SSH blob whose first length-prefixed field is not its own
  algorithm name, a content hash that is not the digest of the bytes the item says it is the digest
  of. A `kind` field recorded beside a text and never compared to it is the defect this project has
  found in its own history more than any other.
"""

from __future__ import annotations

__all__ = ["encoded_messages"]
