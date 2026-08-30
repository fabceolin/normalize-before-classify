"""Cells, and the pooling that cannot be written down.

The score files here are built by hand, line by line, with no corpus and no model in the process.
That is deliberate beyond speed: the sharp tests are about what happens when two benign classes,
two conditions or two chain classes meet in one file, and a fixture that produced a realistic
corpus would make those cases hard to arrange and harder to read.

`fake_pins` is a stand-in for `pins.Pins` carrying only what this module reads -- the baseline key,
threshold and window policy. Building the real object would drag the whole pin file in and would
make every test here depend on a declaration this story does not touch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nbc.corpus.matrix import (
    CHAIN_CLASSES,
    CorpusMatrixInvalid,
    CHAINS,
    CLEAN_CHAIN_NAME,
    HELDOUT_CHAINS,
    chain_class,
    item_id,
    render_chain,
)
from nbc.errors import declared_exit_codes
from nbc.harness.aggregate import (
    CENSUS_CEILING_HIT,
    CENSUS_WINDOW_OVERFLOW,
    CellsInvalid,
    auc_cells,
    canon_delta_cells,
    cells,
    census_cells,
    chain_delta_cells,
    classify,
    false_positive_cells,
    key_for,
    pooling_problems,
    read_scores,
    recall_cells,
)
from nbc.schema import (
    AUC_STRUCTURAL,
    AXIS_BENIGN_CLASS,
    AXIS_CANON_ON,
    AXIS_CHAIN_CLASS,
    AXIS_DRESSING_CHAIN,
    AXIS_FAMILY,
    BENIGN_CLASSES,
    CANONICAL,
    CHAIN_CLASSES_FOR_KEYS,
    CONTRAST_ATTACKS_VS_BENIGN_CLASS,
    CONTRAST_BOUND_VS_HELD_OUT,
    CONTRAST_CANON_ON_VS_OFF,
    CONTRAST_CLEAN_VS_CHAIN,
    FAMILY_ATTACK,
    FAMILY_BENIGN,
    NEWCOMBE_PAIRED,
    POPULATION_ALL,
    RAW,
    WILSON_SCORE,
    Auc,
    CellKey,
    Contrast,
    Count,
    Delta,
    Interval,
    ItemScore,
    Rate,
)

BASELINE = "primary"
THRESHOLD = 0.5
POLICY = "shared"


@dataclass(frozen=True)
class FakeBaseline:
    key: str
    threshold: float
    window_policy: str


@dataclass(frozen=True)
class FakePins:
    baselines: tuple[FakeBaseline, ...]


PINS = FakePins((FakeBaseline(BASELINE, THRESHOLD, POLICY),))


def payload(n: int) -> str:
    """A payload id of the shape `matrix.payload_id` produces: 16 lowercase hex characters."""
    return f"{n:016x}"


def score(
    n: int,
    p: float,
    *,
    chain: tuple[str, ...] = (),
    family: str = FAMILY_ATTACK,
    benign_class: str | None = None,
    condition: str = RAW,
    n_windows: int = 1,
    ceiling_hit: bool | None = None,
    max_depth_reached: int | None = None,
) -> ItemScore:
    # A canonical record must carry a depth and a ceiling flag and a raw one must carry neither;
    # `ItemScore` enforces that, and the helper fills the defaults rather than making every caller
    # restate them.
    canonical = condition == CANONICAL
    return ItemScore(
        item_id=item_id(payload(n), chain),
        family=family,
        benign_class=benign_class,
        label=1 if family == FAMILY_ATTACK else 0,
        baseline_key=BASELINE,
        condition=condition,
        p_injection=p,
        n_windows=n_windows,
        max_depth_reached=(0 if max_depth_reached is None else max_depth_reached)
        if canonical
        else None,
        ceiling_hit=(False if ceiling_hit is None else ceiling_hit) if canonical else None,
    )


def benign(n: int, p: float, klass: str, **kwargs: object) -> ItemScore:
    return score(n, p, family=FAMILY_BENIGN, benign_class=klass, **kwargs)  # type: ignore[arg-type]


def write(tmp_path: Path, scores: list[ItemScore], name: str = "scores.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(s.as_json_object(), ensure_ascii=False) + "\n" for s in scores),
        encoding="utf-8",
    )
    return path


# --- reading ---------------------------------------------------------------------------------------


def test_read_scores_returns_every_record_in_file_order(tmp_path: Path) -> None:
    written = [score(1, 0.9), score(2, 0.1)]
    assert [s.item_id for s in read_scores(write(tmp_path, written))] == [
        s.item_id for s in written
    ]


def test_an_empty_scores_file_aborts(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(CellsInvalid) as caught:
        read_scores(path)
    assert "no records" in str(caught.value)


def test_a_truncated_trailing_record_aborts_naming_its_line(tmp_path: Path) -> None:
    path = write(tmp_path, [score(1, 0.9)])
    path.write_text(path.read_text(encoding="utf-8") + '{"item_id": "trunc', encoding="utf-8")
    with pytest.raises(CellsInvalid) as caught:
        read_scores(path)
    assert ":2" in str(caught.value)


def test_a_line_missing_a_field_aborts(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text('{"item_id": "x"}\n', encoding="utf-8")
    with pytest.raises(CellsInvalid) as caught:
        read_scores(path)
    assert "not an ItemScore" in str(caught.value)


def test_a_duplicated_key_aborts_rather_than_letting_the_last_line_win(tmp_path: Path) -> None:
    """The merge already proved the shard set was a partition, so a duplicate here means the file
    changed after that proof -- and picking a winner would publish a number chosen by file order."""
    one = score(1, 0.9)
    path = write(tmp_path, [one, one])
    with pytest.raises(CellsInvalid) as caught:
        read_scores(path)
    assert "repeats" in str(caught.value)


def test_a_json_array_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(CellsInvalid):
        read_scores(path)


def test_a_missing_file_aborts_as_an_abort_and_not_an_oserror(tmp_path: Path) -> None:
    with pytest.raises(CellsInvalid):
        read_scores(tmp_path / "absent.jsonl")


# --- the key ----------------------------------------------------------------------------------------


def test_the_key_derives_every_axis_rather_than_assuming_one() -> None:
    key = key_for(benign(1, 0.2, "b_code", condition=CANONICAL), PINS)
    assert key.baseline == BASELINE
    assert key.dressing_chain == CLEAN_CHAIN_NAME
    assert key.chain_class == chain_class(())
    assert key.window_policy == POLICY
    assert key.canon_on is True
    assert (key.family, key.benign_class) == (FAMILY_BENIGN, "b_code")


def test_an_undeclared_baseline_aborts() -> None:
    stray = ItemScore(
        item_id=item_id(payload(1), ()),
        family=FAMILY_ATTACK,
        benign_class=None,
        label=1,
        baseline_key="nobody-pinned-this",
        condition=RAW,
        p_injection=0.9,
        n_windows=1,
    )
    with pytest.raises(CellsInvalid) as caught:
        key_for(stray, PINS)
    assert "pins.toml does not" in str(caught.value)


def test_an_item_id_naming_an_undeclared_chain_aborts_before_a_cell_exists() -> None:
    """A column keyed on a dressing nobody declared is a column that came out of a file rather than
    out of AD-20's one constant. It aborts at 19, `matrix`'s code, because the id is what is
    wrong."""
    stray = ItemScore(
        item_id=f"{payload(1)}::a-dressing-nobody-declared",
        family=FAMILY_ATTACK,
        benign_class=None,
        label=1,
        baseline_key=BASELINE,
        condition=RAW,
        p_injection=0.9,
        n_windows=1,
    )
    with pytest.raises(CorpusMatrixInvalid) as caught:
        key_for(stray, PINS)
    assert caught.value.exit_code == 19


def test_the_two_spellings_of_the_chain_classes_agree() -> None:
    """`schema` is a leaf and cannot import `matrix`, so the pair is written twice. This is the
    comparison that makes the second spelling safe, and its two sides come from two modules."""
    assert tuple(CHAIN_CLASSES) == tuple(CHAIN_CLASSES_FOR_KEYS)


# --- the pooling rule, at the constructor -------------------------------------------------------------


def plain_key(**overrides: object) -> CellKey:
    fields: dict[str, object] = {
        "baseline": BASELINE,
        "dressing_chain": CLEAN_CHAIN_NAME,
        "chain_class": chain_class(()),
        "window_policy": POLICY,
        "canon_on": True,
        "family": FAMILY_ATTACK,
        "benign_class": None,
    }
    fields.update(overrides)
    return CellKey(**fields)  # type: ignore[arg-type]


def test_a_benign_cell_with_no_class_cannot_be_constructed() -> None:
    with pytest.raises(ValueError) as caught:
        plain_key(family=FAMILY_BENIGN, benign_class=None)
    assert "FR3.1" in str(caught.value)


@pytest.mark.parametrize("axis", [AXIS_CHAIN_CLASS, "window_policy"])
def test_a_protected_axis_left_null_cannot_be_constructed(axis: str) -> None:
    with pytest.raises(ValueError) as caught:
        plain_key(**{axis: None})
    assert "protected axes" in str(caught.value)


def test_an_attack_cell_carrying_a_benign_class_is_refused() -> None:
    with pytest.raises(ValueError):
        plain_key(family=FAMILY_ATTACK, benign_class="b_code")


def test_a_spanned_axis_must_be_null() -> None:
    contrast = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))
    with pytest.raises(ValueError) as caught:
        plain_key(canon_on=True, contrast=contrast)
    assert "authoritative" in str(caught.value)
    assert plain_key(canon_on=None, contrast=contrast).canon_on is None


def test_a_contrast_that_needs_an_argument_refuses_to_be_built_without_one() -> None:
    with pytest.raises(ValueError):
        Contrast(CONTRAST_CLEAN_VS_CHAIN, None, frozenset({AXIS_DRESSING_CHAIN}))


def test_a_contrast_that_takes_no_argument_refuses_one() -> None:
    with pytest.raises(ValueError):
        Contrast(CONTRAST_CANON_ON_VS_OFF, "base64", frozenset({AXIS_CANON_ON}))


def test_a_contrast_naming_an_undeclared_benign_class_is_refused() -> None:
    with pytest.raises(ValueError):
        Contrast(
            CONTRAST_ATTACKS_VS_BENIGN_CLASS,
            "b_video",
            frozenset({AXIS_FAMILY}),
        )


def test_a_contrast_cannot_claim_to_span_more_than_its_kind_permits() -> None:
    """Spanning an axis is how a cell is allowed to leave it null. A contrast free to declare any
    span would be a way to write down a pooled number and label it a comparison."""
    with pytest.raises(ValueError) as caught:
        Contrast(
            CONTRAST_CANON_ON_VS_OFF,
            None,
            frozenset({AXIS_CANON_ON, AXIS_BENIGN_CLASS}),
        )
    assert "aggregation wearing a label" in str(caught.value)


def test_a_contrast_must_span_what_its_kind_requires() -> None:
    with pytest.raises(ValueError):
        Contrast(CONTRAST_BOUND_VS_HELD_OUT, None, frozenset({AXIS_CHAIN_CLASS}))


def test_the_contrast_name_is_the_one_a_results_file_carries() -> None:
    assert Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON})).name == (
        "canon_on_vs_off"
    )
    assert (
        Contrast(
            CONTRAST_ATTACKS_VS_BENIGN_CLASS,
            "b_code",
            frozenset({AXIS_FAMILY}),
        ).name
        == "attacks_vs_b_code"
    )


# --- the four kinds -------------------------------------------------------------------------------------


def test_recall_is_a_rate_with_a_wilson_interval() -> None:
    scores = [score(i, 0.9 if i < 3 else 0.1) for i in range(4)]
    (cell,) = recall_cells(scores, PINS)
    assert (cell.k, cell.n) == (3, 4)
    assert cell.interval.method == WILSON_SCORE
    assert cell.lo == cell.interval.lo and cell.hi == cell.interval.hi
    assert cell.value == 0.75


def test_a_score_exactly_at_the_threshold_counts_as_a_detection() -> None:
    """`>=` is the declared convention. Which way the tie falls changes recall on any item that
    lands on the boundary, and a run deciding it the other way publishes a different number."""
    assert classify(THRESHOLD, THRESHOLD) is True
    (cell,) = recall_cells([score(1, THRESHOLD)], PINS)
    assert cell.k == 1


def test_the_two_benign_classes_produce_two_rates_and_never_their_sum() -> None:
    """The load-bearing test of this story. `b_code` flags 2 of 3 and `b_chat` flags 0 of 3; the
    pooled number would be 2 of 6, and it is the number FR3.1 exists to keep off the table."""
    scores = [
        benign(1, 0.9, "b_code"),
        benign(2, 0.9, "b_code"),
        benign(3, 0.1, "b_code"),
        benign(4, 0.1, "b_chat"),
        benign(5, 0.1, "b_chat"),
        benign(6, 0.1, "b_chat"),
    ]
    produced = false_positive_cells(scores, PINS)
    by_class = {cell.key.benign_class: cell for cell in produced}
    assert set(by_class) == set(BENIGN_CLASSES)
    assert (by_class["b_code"].k, by_class["b_code"].n) == (2, 3)
    assert (by_class["b_chat"].k, by_class["b_chat"].n) == (0, 3)
    assert all(cell.n != 6 for cell in produced), "no cell carries the pooled denominator"


def test_a_census_count_has_no_interval_and_no_field_for_one() -> None:
    """A ceiling hit is a property of the corpus as built, not a sample from a population of
    possible ceiling hits. Saying so in a docstring while leaving an optional field would last
    until the first person who found the gap untidy."""
    scores = [score(i, 0.9, condition=CANONICAL, ceiling_hit=i == 0) for i in range(3)]
    (cell,) = census_cells(scores, PINS, CENSUS_CEILING_HIT)
    assert (cell.k, cell.n) == (1, 3)
    assert "interval" not in Count.__slots__
    assert "interval" not in cell.as_json_object()


def test_a_ceiling_census_skips_the_raw_half_rather_than_counting_its_nulls() -> None:
    """`ceiling_hit` is `None` under RAW by construction -- the raw condition does not run the
    layer. Counting a null as false would report FR10's census over twice the population it can
    occur in, which is a wrong number that looks entirely reasonable."""
    scores = [
        score(1, 0.9, condition=RAW),
        score(1, 0.9, condition=CANONICAL, ceiling_hit=True),
    ]
    produced = census_cells(scores, PINS, CENSUS_CEILING_HIT)
    assert len(produced) == 1
    assert produced[0].key.canon_on is True


def test_window_overflow_counts_items_needing_a_second_window() -> None:
    scores = [score(1, 0.9, n_windows=3), score(2, 0.9, n_windows=1)]
    (cell,) = census_cells(scores, PINS, CENSUS_WINDOW_OVERFLOW)
    assert (cell.k, cell.n) == (1, 2)


def test_an_unknown_census_kind_aborts() -> None:
    with pytest.raises(CellsInvalid):
        census_cells([score(1, 0.9)], PINS, "edits_by_stage")


def test_an_auc_scores_attacks_against_one_benign_class_at_a_time() -> None:
    scores = [
        score(1, 0.9),
        score(2, 0.8),
        benign(3, 0.1, "b_code"),
        benign(4, 0.2, "b_code"),
        benign(5, 0.95, "b_chat"),
        benign(6, 0.96, "b_chat"),
    ]
    produced = auc_cells(scores, PINS)
    # Keyed by the field, not by the contrast's name. Story 4.4 corrected that: reading the class
    # out of `attacks_vs_b_code` is a substring where a field belongs.
    by_class = {cell.key.benign_class: cell for cell in produced}
    assert set(by_class) == set(BENIGN_CLASSES)
    assert by_class["b_code"].value == 1.0
    assert by_class["b_chat"].value == 0.0
    assert by_class["b_code"].interval.method == AUC_STRUCTURAL
    assert by_class["b_code"].key.family is None, "the two sides differ on family"
    assert by_class["b_code"].key.contrast is not None
    assert by_class["b_code"].key.contrast.name == "attacks_vs_b_code"
    assert all(cell.n_negative == 2 for cell in produced), "never pooled to four"


def test_a_pooled_auc_is_not_constructible() -> None:
    """Not refused by a check in `auc_cells` -- unreachable, because the key it would need cannot
    be built. The comparison spans `family` and `benign_class`; a pooled one would leave
    `benign_class` null with nothing spanning it."""
    contrast = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))
    with pytest.raises(ValueError):
        CellKey(
            baseline=BASELINE,
            dressing_chain=CLEAN_CHAIN_NAME,
            chain_class=chain_class(()),
            window_policy=POLICY,
            canon_on=None,
            family=FAMILY_BENIGN,
            benign_class=None,
            contrast=contrast,
        )


def test_a_delta_comes_from_the_paired_2x2_and_never_from_two_rates() -> None:
    """Canon-on flags 3 of 4 and canon-off 1 of 4, on the same items. The Newcombe interval over
    the full table is the published one; two Wilson intervals subtracted would be a different and
    wider claim on the comparison N1 reads."""
    scores = []
    for i, (off, on) in enumerate([(0.1, 0.9), (0.1, 0.9), (0.9, 0.9), (0.1, 0.1)]):
        scores.append(score(i, off, condition=RAW))
        scores.append(score(i, on, condition=CANONICAL))
    (cell,) = canon_delta_cells(scores, PINS)
    assert cell.value == pytest.approx(0.5)
    assert cell.interval.method == NEWCOMBE_PAIRED
    assert cell.contrast.name == "canon_on_vs_off"
    assert cell.key.canon_on is None


def test_a_delta_whose_halves_cover_different_items_aborts() -> None:
    scores = [
        score(1, 0.1, condition=RAW),
        score(1, 0.9, condition=CANONICAL),
        score(2, 0.9, condition=CANONICAL),
    ]
    with pytest.raises(CellsInvalid) as caught:
        canon_delta_cells(scores, PINS)
    assert "different items" in str(caught.value)


def test_a_delta_refuses_a_wilson_interval() -> None:
    """The type is where "never a subtraction of two rates" is enforced, because the arithmetic of
    a subtraction looks entirely right."""
    contrast = Contrast(CONTRAST_CANON_ON_VS_OFF, None, frozenset({AXIS_CANON_ON}))
    key = plain_key(canon_on=None, contrast=contrast)
    with pytest.raises(ValueError) as caught:
        Delta(0.1, Interval(0.0, 0.2, WILSON_SCORE), key)
    assert "inherits neither" in str(caught.value)


def test_a_delta_without_a_contrast_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        Delta(0.1, Interval(0.0, 0.2, NEWCOMBE_PAIRED), plain_key())
    assert "two equally defensible homes" in str(caught.value)


# --- clean_vs_<chain>, and the span the registry decides --------------------------------------------------


def a_bound_chain() -> tuple[str, ...]:
    for chains in CHAINS.values():
        for chain in chains:
            if chain and chain_class(tuple(chain)) == CHAIN_CLASSES[0]:
                return tuple(chain)
    raise AssertionError("CHAINS declares no non-empty bound chain")


def a_held_out_chain() -> tuple[str, ...]:
    for chains in HELDOUT_CHAINS.values():
        for chain in chains:
            if chain:
                return tuple(chain)
    raise AssertionError("HELDOUT_CHAINS declares no chain")


def test_a_bound_chain_against_clean_spans_the_dressing_only() -> None:
    chain = a_bound_chain()
    scores = [score(1, 0.9), score(1, 0.1, chain=chain)]
    (cell,) = chain_delta_cells(scores, PINS)
    assert cell.contrast.name == f"clean_vs_{render_chain(chain)}"
    assert cell.contrast.spans == frozenset({AXIS_DRESSING_CHAIN})
    assert cell.key.dressing_chain is None
    assert cell.key.chain_class == chain_class(())


def test_a_held_out_chain_against_clean_also_spans_the_chain_class() -> None:
    """`clean` is bound, so this comparison crosses AD-11's protected axis -- and whether it does
    is decided by `matrix.chain_class` against the registries, never by a literal in this module."""
    chain = a_held_out_chain()
    scores = [score(1, 0.9), score(1, 0.1, chain=chain)]
    (cell,) = chain_delta_cells(scores, PINS)
    assert AXIS_CHAIN_CLASS in cell.contrast.spans
    assert cell.key.chain_class is None


def test_the_chain_delta_pairs_by_payload_and_not_by_item_id() -> None:
    """A dressed item and its clean twin share a payload id and differ in chain, so pairing on the
    whole item id finds no overlap at all and produces an empty comparison rather than an error."""
    chain = a_bound_chain()
    scores = [score(i, 0.9) for i in range(3)] + [score(i, 0.1, chain=chain) for i in range(3)]
    (cell,) = chain_delta_cells(scores, PINS)
    assert cell.value == pytest.approx(1.0), "clean detects all three, the dressed chain none"


def test_a_chain_with_no_clean_twin_produces_no_delta() -> None:
    chain = a_bound_chain()
    assert chain_delta_cells([score(1, 0.9, chain=chain)], PINS) == ()


# --- the whole set --------------------------------------------------------------------------------------


def a_realistic_file() -> list[ItemScore]:
    scores: list[ItemScore] = []
    for i in range(4):
        for condition in (RAW, CANONICAL):
            scores.append(score(i, 0.9 if condition == CANONICAL else 0.2, condition=condition))
    for klass, base in (("b_code", 10), ("b_chat", 20)):
        for i in range(3):
            for condition in (RAW, CANONICAL):
                scores.append(
                    benign(base + i, 0.1, klass, condition=condition)
                )
    return scores


def test_every_cell_names_its_protected_axes_or_declares_a_span() -> None:
    produced = cells(a_realistic_file(), PINS)
    assert produced, "the file supports cells"
    assert pooling_problems(produced) == ()


def test_the_cell_set_covers_all_four_kinds() -> None:
    kinds = {cell.as_json_object()["kind"] for cell in cells(a_realistic_file(), PINS)}
    assert kinds == {"rate", "count", "auc", "delta"}


def test_the_cell_order_does_not_depend_on_the_order_of_the_file() -> None:
    """4-7 writes these into a file a reader recomputes and compares. An order that came from dict
    insertion would make that comparison fail for no reason anybody could see."""
    forward = cells(a_realistic_file(), PINS)
    backward = cells(list(reversed(a_realistic_file())), PINS)
    assert [c.as_json_object() for c in forward] == [c.as_json_object() for c in backward]


def test_pooling_problems_fires_on_a_benign_class_that_is_absent() -> None:
    """The input the constructor cannot see: not a pooled cell, a missing one. A class that
    vanishes from the table is as unreadable as one that was averaged away."""
    scores = [s for s in a_realistic_file() if s.benign_class != "b_chat"]
    problems = pooling_problems(cells(scores, PINS))
    assert any("b_chat" in problem for problem in problems)


def escaped_cell(**axes: object) -> object:
    """A cell whose key evaded the constructor, so the set-level checker has a red input.

    Built by hand because `CellKey` refuses to make one -- which is the point of the constructor,
    and is also why the checker would otherwise be a function nobody has seen report anything.
    """
    fields: dict[str, object] = {
        "baseline": BASELINE,
        "dressing_chain": CLEAN_CHAIN_NAME,
        "chain_class": chain_class(()),
        "window_policy": POLICY,
        "canon_on": True,
        "family": FAMILY_BENIGN,
        "benign_class": "b_code",
        "contrast": None,
        "population": POPULATION_ALL,
    }
    fields.update(axes)
    fields["as_json_object"] = lambda self: {k: v for k, v in fields.items() if k != "as_json_object"}
    key = type("EscapedKey", (), fields)()
    return type(
        "EscapedCell",
        (),
        {"key": key, "as_json_object": lambda self: {"kind": "rate"}},
    )()


@pytest.mark.parametrize("axis", [AXIS_CHAIN_CLASS, "window_policy", AXIS_BENIGN_CLASS])
def test_pooling_problems_fires_on_each_null_protected_axis(axis: str) -> None:
    problems = pooling_problems([escaped_cell(**{axis: None})])  # type: ignore[list-item]
    assert any(axis in problem for problem in problems), problems


def test_pooling_problems_does_not_report_an_attack_cell_for_having_no_benign_class() -> None:
    """The pair `CellKey` checks, checked the same way here. An attack cell has no benign class by
    construction, and a checker reading the axis alone would report every recall cell in the table
    as pooled -- which is the exact ambiguity `family` was put on the key to resolve."""
    assert (
        pooling_problems(
            [escaped_cell(family=FAMILY_ATTACK, benign_class=None)]  # type: ignore[list-item]
        )
        == ()
    )


def test_the_new_abort_declares_exit_code_30_and_declares_it_once() -> None:
    codes = declared_exit_codes()
    assert codes[30] is CellsInvalid
    assert CellsInvalid.exit_code == 30
