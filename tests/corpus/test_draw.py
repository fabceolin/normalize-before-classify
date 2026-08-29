"""The shared selection methods: one implementation, and the vocabulary it must cover.

`corpus/draw.py` exists so the attack draw and the benign frame are not two implementations that
happen to agree. What that buys is only real if this module covers exactly the vocabulary
`pins.toml` may declare, so the first test here is that comparison.
"""

from __future__ import annotations

import random

from nbc.corpus.draw import SORT_KEYS, sort_key_for, take
from nbc.corpus.matrix import payload_id
from nbc.pins import DRAW_HEAD, DRAW_METHODS, DRAW_SEEDED_RANDOM, DRAW_SORT_KEYS

POOL = tuple(f"payload-{index:02d}" for index in range(20))


def test_this_module_implements_exactly_the_sort_keys_a_pin_file_may_declare() -> None:
    """A key `pins.toml` admits and this module cannot apply is a declared draw nothing can take."""
    assert set(SORT_KEYS) == set(DRAW_SORT_KEYS), sorted(
        set(SORT_KEYS).symmetric_difference(DRAW_SORT_KEYS)
    )


def test_both_declared_methods_produce_a_draw() -> None:
    for method, seed, sort_key in (
        (DRAW_HEAD, None, "text"),
        (DRAW_SEEDED_RANDOM, 7, None),
    ):
        assert method in DRAW_METHODS
        drawn = take(POOL, 5, method=method, seed=seed, sort_key=sort_key)
        assert drawn is not None and len(drawn) == 5


def test_a_method_nothing_implements_returns_the_sentinel_rather_than_a_draw() -> None:
    """`None`, never `()`: an empty draw and an unimplemented method are different answers."""
    assert take(POOL, 5, method="every_other_row", seed=1, sort_key=None) is None
    assert take((), 5, method=DRAW_SEEDED_RANDOM, seed=1, sort_key=None) == ()


def test_the_draw_does_not_depend_on_the_order_the_pool_arrived_in() -> None:
    shuffled = list(POOL)
    random.Random(99).shuffle(shuffled)
    assert take(POOL, 6, method=DRAW_SEEDED_RANDOM, seed=3, sort_key=None) == take(
        shuffled, 6, method=DRAW_SEEDED_RANDOM, seed=3, sort_key=None
    )


def test_two_seeds_draw_two_samples() -> None:
    """Otherwise the seed would be a field nothing consumes."""
    assert take(POOL, 6, method=DRAW_SEEDED_RANDOM, seed=3, sort_key=None) != take(
        POOL, 6, method=DRAW_SEEDED_RANDOM, seed=4, sort_key=None
    )


def test_a_head_draw_follows_its_declared_sort_key() -> None:
    by_text = take(POOL, 3, method=DRAW_HEAD, seed=None, sort_key="text")
    by_id = take(POOL, 3, method=DRAW_HEAD, seed=None, sort_key="payload_id")
    assert by_text == tuple(sorted(POOL)[:3])
    assert by_id == tuple(sorted(sorted(POOL, key=payload_id)[:3]))
    assert by_text != by_id, "the two sort keys would be indistinguishable on this pool"


def test_a_pool_at_or_below_the_size_is_taken_whole_and_deduplicated() -> None:
    """Nothing here decides whether that is acceptable; the callers raise their own abort."""
    assert take(("b", "a", "a"), 5, method=DRAW_SEEDED_RANDOM, seed=1, sort_key=None) == ("a", "b")


def test_the_result_is_sorted_so_the_returned_order_is_content_derived() -> None:
    drawn = take(POOL, 7, method=DRAW_SEEDED_RANDOM, seed=5, sort_key=None)
    assert drawn is not None and list(drawn) == sorted(drawn)


def test_sort_key_for_is_a_closed_mapping() -> None:
    assert sort_key_for("payload_id")("abc") == payload_id("abc")
