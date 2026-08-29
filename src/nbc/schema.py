"""Every record type in this project, defined here and nowhere else.

This module is a **leaf**: it imports the standard library and nothing from `nbc`. That is
what lets `canon/` depend on it without breaking the canonicalization layer's isolation, and
it is enforced by a test that reads this file's source rather than by convention.

Two consequences of the leaf rule that look like omissions and are not:

- Validation here raises `ValueError`, not `nbc.errors.NbcError`. It cannot raise `NbcError`
  without importing it. That is the right outcome anyway: a malformed record is a
  programming error inside one process, not one of the aborts that change the meaning of a
  published number.
- No module adds a field to these types from outside. A field that six modules write into
  and no module declares is exactly the drift this module exists to prevent.

Types are added here **by the story that first needs them**, not all at once. The rule
governs *where* a record type lives, not *when* it appears: enumerating fields before
anything consumes them freezes names against requirements no one has met yet. `Score` is
here because the model boundary returns it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Score"]


@dataclass(frozen=True, slots=True)
class Score:
    """One document's score from one baseline, under one condition.

    `p_injection` is `softmax(logits)[positive_index]` over the model's full label axis,
    computed in one shared helper so an adapter's only freedom is producing the logits.

    `n_windows` rides on the score rather than on hidden module state: a document longer
    than one window is split into fixed non-overlapping windows, each scored
    independently, and the document takes the maximum. How many windows a document needed
    is part of what the run reports, so it travels with the number it produced.
    """

    p_injection: float
    n_windows: int

    def __post_init__(self) -> None:
        p = self.p_injection
        if isinstance(p, bool) or not isinstance(p, (float, int)):
            raise ValueError(f"p_injection must be a real number, got {p!r}")
        if not 0.0 <= p <= 1.0:
            # Also rejects NaN, which fails every comparison, and both infinities.
            raise ValueError(f"p_injection must lie in [0, 1], got {p!r}")
        if not isinstance(p, float):
            # Stored as a float so the serialized form does not depend on how the caller
            # happened to spell 0 or 1.
            object.__setattr__(self, "p_injection", float(p))

        n = self.n_windows
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"n_windows must be an int, got {n!r}")
        if n < 1:
            raise ValueError(
                f"n_windows must be at least 1: a scored document occupies at least one "
                f"window, got {n!r}"
            )
