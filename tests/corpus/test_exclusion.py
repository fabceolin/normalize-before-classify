"""The training-overlap filter, gate by gate, each with the input that makes it fail.

Every check here is offline. What the hub answers arrives as an `Observation` the test writes, so
the whole decision procedure -- which source may be missing, which may not, what counts as the same
text -- is covered by a suite that never opens a socket.

The pins used below are **built in code**, not read from `pins.toml`. A test that asserted the
filter's behaviour by re-reading the committed file would be comparing the pins against themselves;
the fixtures here name their own sources, so a gate that stopped firing is visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nbc.corpus.exclusion import (
    NO_ANSWER,
    NORMALIZATION,
    ExclusionReport,
    ExclusionSetUnusable,
    Observation,
    build_index,
    declaration_digest,
    filter_rows,
    normalize,
    outcomes_of,
    plan,
    verify_observations,
)
from nbc.errors import NbcError, declared_exit_codes
from nbc.pins import (
    BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
    DRAW_SEEDED_RANDOM,
    EXCLUSION_AVAILABLE,
    EXCLUSION_UNREACHABLE,
    EXCLUSION_UNREADABLE,
    HTTP_OK,
    BenignChatFrame,
    BenignCodeFrame,
    BenignFrame,
    ConfirmatoryCell,
    ExclusionSource,
    Pins,
    load_pins,
)

from tests.test_pins import SEED_ONE, SEED_TWO, SHA_A, SHA_E  # fixture constants, reused

GATED = 401
"""What the one access-restricted source answers. Named here so the tests read as English."""


def _source(
    repository: str,
    *,
    availability: str = EXCLUSION_AVAILABLE,
    http_status: int = HTTP_OK,
    revision: str = SHA_E,
) -> ExclusionSource:
    return ExclusionSource(
        key=repository.replace("/", "-"),
        repository=repository,
        revision="" if availability == EXCLUSION_UNREACHABLE else revision,
        availability=availability,
        http_status=http_status,
        checked_on="2026-08-29",
        evidence="fixture",
    )


def _gated(repository: str) -> ExclusionSource:
    """A source the hub does not answer for at all."""
    return _source(repository, availability=EXCLUSION_UNREACHABLE, http_status=GATED)


def _script_backed(repository: str) -> ExclusionSource:
    """A source that resolves at its sha and whose rows the pinned reader refuses."""
    return _source(repository, availability=EXCLUSION_UNREADABLE)


def _pins(sources: tuple[ExclusionSource, ...], *, required: tuple[str, ...] = ()) -> Pins:
    """A `Pins` whose `required_exclusion_sources()` is whatever the test says it is.

    Subclassed rather than assembled from lineage blocks: the point of every test below is what
    the filter does with the obligation, and building a whole loadable document to express "this
    source is required" would put the pins parser inside a test about the corpus.
    """

    class _Pins(Pins):
        def required_exclusion_sources(self) -> tuple[str, ...]:
            return required

    return _Pins(
        schema_version=2,
        verified_on="2026-08-29",
        verified_against="fixture",
        baselines=(),
        attack_datasets=(),
        exclusion_sources=sources,
        benign_frame=_frame(),
        path=None,  # type: ignore[arg-type]
    )


def _frame() -> BenignFrame:
    """A structurally valid frame. Nothing in this file reads it; `Pins` requires one.

    `declaration_digest` covers the exclusion declaration and nothing else, and the tests below
    assert exactly that -- so this fixture exists to satisfy the record's shape and is deliberately
    not varied by any of them.
    """
    return BenignFrame(
        declared_on="2026-08-29",
        sample_size_items=4,
        method=DRAW_SEEDED_RANDOM,
        seed=11,
        sort_key=None,
        frame_id="0" * 64,
        b_code=BenignCodeFrame(
            min_repositories=1,
            max_files_per_repository=1,
            eligibility=BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
            min_file_bytes=1,
            max_file_bytes=2,
            file_extensions=(".py",),
            repositories=(),
        ),
        b_chat=BenignChatFrame(hand_authored_items=0),
        confirmatory_cell=ConfirmatoryCell(
            declared_on="2026-08-29",
            baseline="fixture",
            dressing_chain="clean",
            benign_class="b_code",
        ),
    )


# --- the declared normalization ---------------------------------------------------------------


def test_the_normalization_is_the_three_declared_steps_and_nothing_else() -> None:
    assert normalize("  A  B\t\nc  ") == "a b c"
    assert NORMALIZATION == "nfkc-lower-collapse-whitespace"


@pytest.mark.parametrize(
    ("corpus_text", "source_text", "step"),
    [
        ("ＡＢＣ", "ABC", "NFKC"),
        ("ABC", "abc", "lowercase"),
        ("a  b\n", "a b", "whitespace collapse"),
        # NFKC turns U+00A0 into an ordinary space, so the collapse has to run AFTER it. Collapse
        # first and this pair stays two different rows.
        ("a b", "a b", "NFKC before the collapse"),
    ],
)
def test_each_declared_step_is_load_bearing(
    corpus_text: str, source_text: str, step: str
) -> None:
    """Drop the named step and the pair below stops matching. That is what makes it a step."""
    index = build_index({"source": [source_text]})
    result = filter_rows([corpus_text], index, text_of=lambda row: row)

    assert result.removed == (corpus_text,), step
    assert result.kept == ()


def test_a_blank_exclusion_cell_removes_nothing() -> None:
    """Most sources carry an empty column somewhere, and it must not empty the corpus.

    Without the guard, every row normalizing to nothing is removed and attributed to whichever
    source happened to have a blank cell.
    """
    index = build_index({"source": ["   ", " ", ""]})

    assert len(index) == 0
    assert filter_rows(["", "  "], index, text_of=lambda row: row).removed == ()


# --- the filter -------------------------------------------------------------------------------


def test_a_row_in_two_sources_is_removed_once_and_counted_twice() -> None:
    """The per-source counts do not add up to the total, on purpose, and both are published."""
    index = build_index({"first": ["shared", "only-first"], "second": ["shared"]})
    result = filter_rows(["shared", "survivor"], index, text_of=lambda row: row)

    assert result.removed == ("shared",)
    assert result.kept == ("survivor",)
    assert result.matches_by_source == {"first": 1, "second": 1}


def test_the_filter_preserves_order_and_repeats_the_same_answer() -> None:
    index = build_index({"first": ["b"]})
    rows = ["a", "b", "c", "b"]

    first = filter_rows(rows, index, text_of=lambda row: row)
    second = filter_rows(rows, index, text_of=lambda row: row)

    assert first.kept == ("a", "c") == second.kept
    assert first.matches_by_source == {"first": 2} == second.matches_by_source


def test_the_filter_reads_the_text_out_of_whatever_row_it_is_handed() -> None:
    """The corpus row does not exist yet, so the accessor is a parameter rather than a guess."""
    index = build_index({"first": ["payload"]})
    rows = [{"id": "one", "text": "payload"}, {"id": "two", "text": "clean"}]

    result = filter_rows(rows, index, text_of=lambda row: row["text"])

    assert [row["id"] for row in result.kept] == ["two"]


# --- the obligation, consumed rather than re-derived -------------------------------------------


def test_the_plan_marks_exactly_what_the_pins_require() -> None:
    """Decision D-C: the required set is read from `required_exclusion_sources()`, not rebuilt."""
    pins = _pins((_source(SEED_ONE), _source(SEED_TWO)), required=(SEED_TWO,))

    assert {entry.repository: entry.required for entry in plan(pins)} == {
        SEED_ONE: False,
        SEED_TWO: True,
    }


def test_the_plan_sees_through_the_spelling_of_a_repository_id() -> None:
    """The hub resolves ids case-insensitively, and the two blocks are written from two cards."""
    pins = _pins((_source("Example/Seed-One"),), required=("example/seed_one",))

    assert plan(pins)[0].required is True


def test_a_required_source_the_pins_never_declared_aborts() -> None:
    pins = _pins((_source(SEED_ONE),), required=(SEED_TWO,))

    with pytest.raises(ExclusionSetUnusable) as abort:
        plan(pins)

    assert SEED_TWO in str(abort.value)


# --- what the pins declare against what the run saw --------------------------------------------


def test_a_run_that_matches_the_declarations_passes() -> None:
    planned = plan(
        _pins(
            (_source(SEED_ONE), _gated(SEED_TWO)),
            required=(SEED_ONE,),
        )
    )

    verify_observations(
        planned,
        {
            "example-seed-one": Observation(HTTP_OK, loadable=True, texts_loaded=12),
            "example-seed-two": Observation(GATED),
        },
    )


def test_an_unreadable_source_that_is_not_required_is_reported_and_not_fatal() -> None:
    """The 401 case: named in the accounting, never treated as contributing zero."""
    pins = _pins((_source(SEED_ONE), _gated(SEED_TWO)), required=(SEED_ONE,))
    planned = plan(pins)
    observations = {
        "example-seed-one": Observation(HTTP_OK, loadable=True, texts_loaded=5),
        "example-seed-two": Observation(GATED),
    }
    verify_observations(planned, observations)

    report = ExclusionReport(
        normalization=NORMALIZATION,
        declaration_digest=declaration_digest(pins),
        rows_in=10,
        rows_removed=2,
        outcomes=outcomes_of(planned, observations, {"example-seed-one": 2}),
    )
    fields = report.as_run_fields()["exclusion"]

    assert fields["unread_sources"] == [SEED_TWO]
    unread = [row for row in fields["sources"] if row["repository"] == SEED_TWO][0]
    # Absent, not zero. Zero is a measurement and nobody measured this one.
    assert unread["matched_rows"] is None
    assert unread["declared_http_status"] == GATED == unread["observed_http_status"]
    assert fields["rows_kept"] == 8


def test_a_source_declared_available_that_no_longer_answers_aborts() -> None:
    planned = plan(_pins((_source(SEED_ONE),)))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(planned, {"example-seed-one": Observation(404)})

    assert "404" in str(abort.value)


def test_a_gated_source_that_opened_up_aborts_too() -> None:
    """The other direction, and it is not a happy surprise: the pins now understate the set."""
    planned = plan(_pins((_gated(SEED_ONE),)))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(
            planned, {"example-seed-one": Observation(HTTP_OK, loadable=True, texts_loaded=900)}
        )

    assert SEED_ONE in str(abort.value)


def test_a_required_source_the_pins_honestly_call_unreadable_still_aborts() -> None:
    """The gate D-C actually buys, and the one no other check covers.

    The declaration and the observation agree, so the drift check passes. The run still may not
    proceed: the published clean recall is an upper bound until exactly these rows are removed.
    """
    planned = plan(_pins((_gated(SEED_ONE),), required=(SEED_ONE,)))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(planned, {"example-seed-one": Observation(GATED)})

    assert "required" in str(abort.value)
    assert "upper bound" in str(abort.value)


def test_a_source_that_answered_200_and_yielded_no_text_aborts() -> None:
    """A schema change wearing the face of a source with no overlap."""
    planned = plan(_pins((_source(SEED_ONE),)))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(
            planned, {"example-seed-one": Observation(HTTP_OK, loadable=True, texts_loaded=0)}
        )

    assert "yielded no text" in str(abort.value)


def test_a_source_nobody_probed_aborts() -> None:
    planned = plan(_pins((_source(SEED_ONE), _source(SEED_TWO))))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(
            planned, {"example-seed-one": Observation(HTTP_OK, loadable=True, texts_loaded=3)}
        )

    assert "recorded no observation" in str(abort.value)


def test_a_probe_that_got_no_answer_at_all_fails_the_declared_status() -> None:
    """`NO_ANSWER` must compare unequal to every declared status, including 200."""
    planned = plan(_pins((_source(SEED_ONE),)))

    with pytest.raises(ExclusionSetUnusable):
        verify_observations(planned, {"example-seed-one": Observation(NO_ANSWER)})


def test_every_problem_is_collected_before_the_abort() -> None:
    planned = plan(_pins((_source(SEED_ONE), _source(SEED_TWO))))

    with pytest.raises(ExclusionSetUnusable) as abort:
        verify_observations(
            planned,
            {
                "example-seed-one": Observation(404),
                "example-seed-two": Observation(HTTP_OK, loadable=True, texts_loaded=0),
            },
        )

    assert len(abort.value.problems) == 2


# --- the declaration digest ---------------------------------------------------------------


def test_reordering_the_declaration_does_not_invent_a_new_corpus() -> None:
    first = _pins((_source(SEED_ONE), _source(SEED_TWO)))
    second = _pins((_source(SEED_TWO), _source(SEED_ONE)))

    assert declaration_digest(first) == declaration_digest(second)


@pytest.mark.parametrize(
    "moved",
    [
        (_source(SEED_ONE, revision=SHA_A),),
        (_gated(SEED_ONE),),
        (_script_backed(SEED_ONE),),
        (_source(SEED_ONE), _source(SEED_TWO)),
    ],
    ids=["revision", "unreachable", "unreadable", "an added source"],
)
def test_any_move_in_the_declaration_changes_the_digest(
    moved: tuple[ExclusionSource, ...],
) -> None:
    """It is offered to `build_id`, so a change nothing hashes is a corpus nobody can tell apart."""
    assert declaration_digest(_pins(moved)) != declaration_digest(_pins((_source(SEED_ONE),)))


def test_the_digest_covers_the_normalization_rule_itself() -> None:
    """A count without its rule is a number nobody can reproduce, so the rule is in the hash."""
    pins = _pins((_source(SEED_ONE),))
    import nbc.corpus.exclusion as module

    original = module.NORMALIZATION
    before = declaration_digest(pins)
    try:
        module.NORMALIZATION = "exact-text"
        assert declaration_digest(pins) != before
    finally:
        module.NORMALIZATION = original


# --- the abort itself -------------------------------------------------------------------------


def test_the_abort_carries_a_code_no_other_abort_declares() -> None:
    assert declared_exit_codes()[ExclusionSetUnusable.exit_code] is ExclusionSetUnusable
    assert issubclass(ExclusionSetUnusable, NbcError)


def test_the_plan_over_the_committed_pins_marks_the_obligation_the_file_grants() -> None:
    """The one assertion here that reads the real file, and the point of decision D-C.

    `required_exclusion_sources()` was published and consumed by nothing outside `pins.py`. This
    is the consumption: the plan marks exactly those sources, and it marks at least one -- a plan
    that marked nothing would satisfy every equality below while discharging no obligation at all.
    The repository ids are not written out; `tests/test_pins.py` characterizes which sources they
    are, structurally, where the pins fixtures live.
    """
    pins = load_pins(Path(__file__).resolve().parents[2])
    required = set(pins.required_exclusion_sources())

    assert required
    assert {entry.repository for entry in plan(pins) if entry.required} == required
    assert not all(entry.required for entry in plan(pins)), (
        "every declared source reads as required, so the required flag distinguishes nothing"
    )
