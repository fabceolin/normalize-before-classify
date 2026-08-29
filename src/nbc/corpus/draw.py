"""The declared selection methods, implemented once and consumed by both halves of the corpus.

`pins.toml` declares a draw the same way for the attack pool and for the benign frame -- a `method`
from a closed vocabulary, and exactly one companion parameter -- and `pins._read_selection_method`
reads both declarations with one function. This module is the other half of that: the two draws are
**taken** by one function too. Two implementations that happened to agree today would satisfy
FR5.1's "the same selection-method vocabulary as the attack draw" and drift the first time one of
them grew a method.

**The pool is sorted before anything else happens.** That is what makes a draw a function of the
declared seed and of nothing else: not of parquet row order, not of which split or which archive
was read first, not of the process hash seed. The result is sorted again on the way out, so the
returned order is content-derived too.

`random.Random(seed).shuffle` is the shuffle. The interpreter is pinned to CPython 3.13 exactly by
`pyproject.toml`, and the Mersenne Twister stream for a given seed is stable within it, so the same
seed reproduces the same sample wherever this project is allowed to run at all.
"""

from __future__ import annotations

import random
from typing import Callable, Final, Iterable, Mapping

from nbc.corpus.matrix import payload_id
from nbc.pins import DRAW_HEAD, DRAW_SEEDED_RANDOM

__all__ = ["SORT_KEYS", "sort_key_for", "take"]

SORT_KEYS: Final[Mapping[str, Callable[[str], str]]] = {
    "text": lambda text: text,
    "payload_id": payload_id,
}
"""The declared sort keys for a `head` draw, as functions of the drawn text.

A closed mapping rather than a lookup by attribute name: both keys are pure functions of the text,
and neither can be made to read a row position, a split or a file order. `pins.DRAW_SORT_KEYS` is
the vocabulary this must cover, and `tests/corpus/test_draw.py` compares the two -- a sort key a
pin file may declare and this module cannot apply would be a declared draw nothing can take.
"""


def sort_key_for(name: str) -> Callable[[str], str]:
    return SORT_KEYS[name]


def take(
    pool: Iterable[str],
    size: int,
    *,
    method: str,
    seed: int | None,
    sort_key: str | None,
) -> tuple[str, ...] | None:
    """The declared draw of `size` items from `pool`, or `None` if nothing implements `method`.

    **`None` rather than an abort**, and the reason is that the two callers abort differently:
    a method nothing implements is `AttackDrawUnsatisfiable` on one side and
    `BenignDrawUnsatisfiable` on the other, and a shared raise here would make the corpus report one
    diagnosis for two different halves of the build. Returning the sentinel keeps the exit codes
    where the requirements put them. It is never confused with an empty draw: an empty draw is `()`.

    A pool at or below `size` is returned whole. Nothing here decides whether that is acceptable --
    FR5.1's "the build fails rather than topping up" belongs to the caller that knows which class
    fell short and by how much, and putting a silent truncation here would half-enforce it in a
    place with nothing to name.
    """
    unique = sorted(set(pool))
    if size >= len(unique):
        return tuple(unique)

    if method == DRAW_HEAD:
        ordered = sorted(unique, key=sort_key_for(str(sort_key)))
        return tuple(sorted(ordered[:size]))
    if method == DRAW_SEEDED_RANDOM:
        shuffled = list(unique)
        random.Random(seed).shuffle(shuffled)
        return tuple(sorted(shuffled[:size]))
    return None
