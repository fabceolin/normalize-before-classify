"""The shard algebra, with no model and no file in the process.

Every row of story 4.2's I/O matrix has a test here or in `test_run.py`, and every abort ships the
input that turns it red -- an abort nobody has seen fire is an abort nobody knows fires.

The load-bearing one is `test_the_merged_bytes_do_not_depend_on_how_the_work_was_split`: the whole
design is allowed only because splitting the pass across processes cannot change what it produces,
and that has to be checked rather than argued.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from nbc.errors import declared_exit_codes, exit_code_for
from nbc.harness.score import (
    SHARD_SCHEMA_VERSION,
    DeclaredPath,
    ExecutionPath,
    ScoreSetIncomplete,
    ShardFile,
    ShardHeader,
    agreement_problems,
    coverage_problems,
    expected_keys,
    key_of,
    merge,
    parse_shard,
    path_problems,
    render_record,
    render_shard,
    score_key,
    serialize,
    shard_of,
)
from nbc.schema import CANONICAL, CONDITIONS, RAW, CorpusItem, ItemScore
from tests.harness.corpus_fixtures import (
    attack_item,
    benign_item,
    digest_probability,
    small_corpus,
)

BASELINES = ("alpha", "beta")
REVISIONS = {"alpha": "a" * 40, "beta": "b" * 40}
BUILD_ID = "c" * 64
CPU = ("CPUExecutionProvider",)
CUDA = ("CUDAExecutionProvider",)

DECLARED = DeclaredPath(
    providers=CPU, intra_op_num_threads=1, batch_size=8, revisions=REVISIONS
)


def paths(
    *,
    providers: tuple[str, ...] = CPU,
    intra_op_num_threads: int = 1,
    batch_size: int = 8,
    revisions: dict[str, str] | None = None,
) -> tuple[ExecutionPath, ...]:
    chosen = revisions or REVISIONS
    return tuple(
        ExecutionPath(
            baseline_key=key,
            revision=chosen[key],
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            batch_size=batch_size,
        )
        for key in BASELINES
    )


def header(shards: int, shard: int, **kwargs) -> ShardHeader:
    build_id = kwargs.pop("build_id", BUILD_ID)
    recorded = kwargs.pop("paths", None)
    return ShardHeader(
        schema_version=SHARD_SCHEMA_VERSION,
        shards=shards,
        shard=shard,
        build_id=build_id,
        paths=paths(**kwargs) if recorded is None else recorded,
    )


def score_for(item: CorpusItem, baseline_key: str, condition: str) -> ItemScore:
    """One record, with numbers derived from the key so two records are never accidentally equal."""
    text = item.text if condition == RAW else item.text.replace("‍", "")
    return ItemScore(
        item_id=item.id,
        family=item.family,
        benign_class=item.benign_class,
        label=item.label,
        baseline_key=baseline_key,
        condition=condition,
        p_injection=digest_probability(baseline_key, text),
        n_windows=1 + len(text) // 64,
        max_depth_reached=0 if condition == CANONICAL else None,
        ceiling_hit=False if condition == CANONICAL else None,
    )


def every_score(items=None) -> tuple[ItemScore, ...]:
    return tuple(
        score_for(item, baseline_key, condition)
        for item in (items if items is not None else small_corpus())
        for baseline_key in BASELINES
        for condition in CONDITIONS
    )


def split(scores, shards: int) -> list[ShardFile]:
    """The scores partitioned exactly the way a real run would partition them."""
    return [
        ShardFile(
            name=f"scores-{shards}-{index}.jsonl",
            header=header(shards, index),
            scores=tuple(
                score for score in scores if shard_of(key_of(score), shards) == index
            ),
        )
        for index in range(shards)
    ]


def demanded(items=None) -> tuple[str, ...]:
    return expected_keys(
        list(items if items is not None else small_corpus()), list(BASELINES)
    )


def merged(files, expected=None, shards: int = 1, **kwargs):
    return merge(
        files,
        expected=expected if expected is not None else demanded(),
        build_id=kwargs.pop("build_id", BUILD_ID),
        declared=kwargs.pop("declared", DECLARED),
        shards=shards,
    )


# --- the key -----------------------------------------------------------------------------------


def test_the_key_of_one_cell_is_not_a_joined_string() -> None:
    """The failing input for a `::`-joined key, which corpus ids would collide under.

    A corpus id is `<payload id>::<chain>`, so the separator a naive key would use is already in
    one of the components. Two distinct cells mapping to one key is a coverage check that reports
    one item scored twice and another never scored, for two items each scored exactly once.
    """
    left = score_key("a::b", "c", RAW)
    right = score_key("a", "b::c", RAW)

    assert left != right
    assert "a::b::c::raw" == "::".join(["a::b", "c", RAW]) == "::".join(["a", "b::c", RAW])


def test_the_key_round_trips_to_the_three_things_it_names() -> None:
    """Structural, not textual: the key is a parsed value, so a reader never has to split it."""
    assert json.loads(score_key("id::clean", "alpha", CANONICAL)) == [
        "id::clean",
        "alpha",
        CANONICAL,
    ]


def test_the_key_of_a_record_is_the_key_of_its_three_fields() -> None:
    """`key_of` and `score_key` are one function with two front doors, and cannot drift."""
    record = score_for(attack_item(1), "alpha", RAW)

    assert key_of(record) == score_key(record.item_id, record.baseline_key, record.condition)


# --- membership --------------------------------------------------------------------------------


def test_membership_is_a_function_of_the_key_and_not_of_the_reading_order() -> None:
    """The reason `shard_of` hashes and does not take an index.

    The corpus is read in reverse and the partition is identical. Under `index % n` it would not
    be, and nothing downstream would notice: every key would still appear exactly once, in a
    different shard from the one that claimed it.
    """
    items = list(small_corpus())
    forward = {key: shard_of(key, 3) for key in demanded(items)}
    backward = {key: shard_of(key, 3) for key in demanded(list(reversed(items)))}

    assert forward == backward
    assert len(set(forward.values())) > 1, "every key landed in one shard; the split is not one"


def test_a_shard_count_that_is_not_a_count_is_refused() -> None:
    for bad in (0, -1, True, 2.0, "3"):
        with pytest.raises(ValueError, match="shards must be a positive int"):
            shard_of("k", bad)  # type: ignore[arg-type]


def test_every_key_lands_in_exactly_one_shard_at_every_count() -> None:
    keys = demanded()
    for shards in (1, 2, 3, 7, 13):
        owners = [shard_of(key, shards) for key in keys]
        assert all(0 <= owner < shards for owner in owners)
        assert len(owners) == len(keys)


# --- the demand set ----------------------------------------------------------------------------


def test_the_demand_set_is_the_corpus_crossed_with_the_baselines_and_the_conditions() -> None:
    items = list(small_corpus())
    keys = demanded(items)

    assert len(keys) == len(items) * len(BASELINES) * len(CONDITIONS)
    assert list(keys) == sorted(keys), "the demand set depends on the order its inputs arrived in"


def test_a_corpus_with_no_rows_is_refused_rather_than_scored() -> None:
    """A pass over nothing merges cleanly and publishes a rate over nothing."""
    with pytest.raises(ScoreSetIncomplete, match="carries no rows"):
        expected_keys([], list(BASELINES))


def test_a_pass_with_no_baselines_is_refused() -> None:
    with pytest.raises(ScoreSetIncomplete, match="no baseline was named"):
        expected_keys(list(small_corpus()), [])


def test_a_corpus_carrying_one_id_twice_is_refused() -> None:
    """Exactly-once is counted over keys, so a repeated id hides a row from the coverage check."""
    twice = [*small_corpus(), attack_item(1)]

    with pytest.raises(ScoreSetIncomplete, match="repeated item id"):
        expected_keys(twice, list(BASELINES))


def test_a_baseline_named_twice_is_refused() -> None:
    with pytest.raises(ScoreSetIncomplete, match="named more than once"):
        expected_keys(list(small_corpus()), ["alpha", "alpha"])


# --- the load-bearing property -------------------------------------------------------------------


@pytest.mark.parametrize("shards", [1, 3, 7])
def test_the_merged_bytes_do_not_depend_on_how_the_work_was_split(shards: int) -> None:
    """The claim the whole design rests on, at the algebra level.

    One score set, partitioned at 1, 3 and 7, merged, serialized -- and the bytes compared against
    the single-shard merge. If the merged file depended on the split, every number the table
    reports would depend on how many machines happened to be free.
    """
    scores = every_score()
    reference = serialize(merged(split(scores, 1), shards=1))

    assert serialize(merged(split(scores, shards), shards=shards)) == reference


def test_the_merged_bytes_do_not_depend_on_the_order_the_shard_files_were_read() -> None:
    """The other half of the same claim: the merge reads a set of files, not a sequence."""
    scores = every_score()
    files = split(scores, 3)

    assert serialize(merged(files, shards=3)) == serialize(
        merged(list(reversed(files)), shards=3)
    )


def test_the_split_invariance_test_can_fail() -> None:
    """The failing input for the property above, so it is not passing by comparing nothing.

    One record moved into the wrong shard file changes nothing about the merged bytes -- and that
    is exactly right, because the merge sorts. What must change the bytes is a changed *value*, so
    that is what this perturbs.
    """
    scores = list(every_score())
    reference = serialize(merged(split(tuple(scores), 1), shards=1))
    scores[0] = replace(scores[0], p_injection=scores[0].p_injection / 2)

    assert serialize(merged(split(tuple(scores), 1), shards=1)) != reference


def test_the_merged_file_is_one_terminated_json_object_per_record() -> None:
    text = serialize(every_score())
    lines = text.splitlines()

    assert text.endswith("\n")
    assert len(lines) == len(every_score())
    assert all(isinstance(json.loads(line), dict) for line in lines)


# --- coverage, in both directions -----------------------------------------------------------------


def test_a_shard_that_did_not_run_is_named_with_the_shard_it_belongs_to() -> None:
    """Matrix row "missing shard": the operator's next action is to re-run one shard."""
    files = split(every_score(), 3)
    without_one = [file for file in files if file.header.shard != 1]

    with pytest.raises(ScoreSetIncomplete) as abort:
        merged(without_one, shards=3)

    (problem,) = [line for line in abort.value.problems if "owes" in line]
    assert "shard 1 of 3" in problem


def test_a_key_two_shard_files_both_carry_is_refused_even_when_they_agree() -> None:
    """Matrix row "duplicated item": a key claimed twice means the split was not a partition."""
    scores = every_score()
    files = split(scores, 2)
    stolen = files[0].scores[0]
    files[1] = replace(files[1], scores=(*files[1].scores, stolen))

    with pytest.raises(ScoreSetIncomplete) as abort:
        merged(files, shards=2)

    assert any(
        "carried by" in problem and key_of(stolen) in problem
        for problem in abort.value.problems
    )


def test_a_key_nobody_demanded_is_refused() -> None:
    """A record for a row the corpus no longer has: the other direction of the coverage check."""
    files = split(every_score(), 1)
    ghost = score_for(benign_item(99), "alpha", RAW)
    files[0] = replace(files[0], scores=(*files[0].scores, ghost))

    with pytest.raises(ScoreSetIncomplete, match="demanded by nothing"):
        merged(files, shards=1)


def test_an_item_scored_under_one_condition_only_is_caught_and_the_condition_named() -> None:
    """Matrix row "item scored under one condition only": canonical present, raw missing."""
    files = split(every_score(), 1)
    dropped = next(score for score in files[0].scores if score.condition == RAW)
    files[0] = replace(
        files[0], scores=tuple(score for score in files[0].scores if score != dropped)
    )

    with pytest.raises(ScoreSetIncomplete) as abort:
        merged(files, shards=1)

    (problem,) = [line for line in abort.value.problems if "owes" in line]
    assert key_of(dropped) in problem
    assert f'"{RAW}"' in problem


def test_a_file_carrying_another_shards_key_is_refused() -> None:
    """The signature of a run whose membership was computed some other way.

    By itself it breaks nothing a reader of the merged file would see, which is exactly why it has
    to be an abort rather than something a reviewer might notice.
    """
    files = split(every_score(), 3)
    misplaced = files[0].scores[0]
    files[1] = replace(files[1], scores=(*files[1].scores, misplaced))
    files[0] = replace(files[0], scores=files[0].scores[1:])

    problems = coverage_problems(demanded(), files, shards=3)

    assert any("belong to another shard" in problem for problem in problems)


def test_a_complete_set_of_shards_has_no_coverage_problem() -> None:
    """The scan's passing input, so the failures above are not the only thing it can say."""
    assert coverage_problems(demanded(), split(every_score(), 3), shards=3) == ()


# --- disagreement -------------------------------------------------------------------------------


def test_two_shards_that_disagree_about_one_key_abort_naming_both_values() -> None:
    """Matrix row "disagreement". No mean, no first-wins, no last-wins."""
    scores = every_score()
    files = split(scores, 2)
    original = files[0].scores[0]
    other = replace(original, p_injection=original.p_injection / 2)
    files[1] = replace(files[1], scores=(*files[1].scores, other))

    problems = agreement_problems(files)

    (problem,) = problems
    assert key_of(original) in problem
    assert f"{original.p_injection:.17g}" in problem
    assert f"{other.p_injection:.17g}" in problem


def test_two_shards_that_agree_on_the_probability_and_not_on_the_windows_still_abort() -> None:
    """Same fault one step earlier: two shards that tokenized the document differently."""
    files = split(every_score(), 2)
    original = files[0].scores[0]
    files[1] = replace(
        files[1], scores=(*files[1].scores, replace(original, n_windows=original.n_windows + 1))
    )

    assert agreement_problems(files) != ()


def test_a_disagreement_names_values_that_differ_only_in_the_last_bit() -> None:
    """The message renders 17 significant digits, because that is where this fault lives.

    A crossed execution path moves a score in the last decimals. A message that printed six of
    them would report the two sides as the same number and send the reader looking elsewhere.
    """
    files = split(every_score(), 2)
    original = files[0].scores[0]
    nudged = replace(original, p_injection=nextafter_up(original.p_injection))
    files[1] = replace(files[1], scores=(*files[1].scores, nudged))

    (problem,) = agreement_problems(files)
    left, right = f"{original.p_injection:.17g}", f"{nudged.p_injection:.17g}"

    assert left != right
    assert left in problem and right in problem


def nextafter_up(value: float) -> float:
    import math

    return math.nextafter(value, 1.0)


def test_a_consistent_set_of_shards_has_no_agreement_problem() -> None:
    assert agreement_problems(split(every_score(), 3)) == ()


# --- the execution path ---------------------------------------------------------------------------


def test_one_shard_on_another_provider_aborts_naming_both() -> None:
    """Matrix row "crossed execution path": somebody ran one shard on the GPU box."""
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 1, providers=CUDA))

    problems = path_problems(files, build_id=BUILD_ID, declared=DECLARED)

    assert any("CUDAExecutionProvider" in problem for problem in problems)
    assert any("CPUExecutionProvider" in problem for problem in problems)


def test_a_whole_pass_on_another_provider_aborts_too() -> None:
    """The case shard-versus-shard agreement cannot see, and the one the gate is named for.

    Every shard ran on CUDA, so the shards agree perfectly with each other. They are compared
    against the declared path as well, which is why this is an abort and not a clean merge.
    """
    files = [replace(file, header=header(2, file.header.shard, providers=CUDA)) for file in split(every_score(), 2)]

    problems = path_problems(files, build_id=BUILD_ID, declared=DECLARED)

    assert problems
    assert all("CUDAExecutionProvider" in problem for problem in problems)


def test_one_shard_with_intra_op_threading_aborts() -> None:
    """Matrix row "crossed threading": a threaded float32 reduction does not add up twice."""
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 1, intra_op_num_threads=8))

    problems = path_problems(files, build_id=BUILD_ID, declared=DECLARED)

    assert any("intra_op_num_threads" in problem for problem in problems)


def test_one_shard_at_another_batch_size_aborts() -> None:
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 1, batch_size=16))

    assert any(
        "batch_size" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_two_shards_scored_against_different_revisions_abort_naming_the_baseline() -> None:
    """Matrix row "crossed revision"."""
    moved = {**REVISIONS, "beta": "d" * 40}
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 1, revisions=moved))

    problems = path_problems(files, build_id=BUILD_ID, declared=DECLARED)

    assert any("beta" in problem and "d" * 40 in problem for problem in problems)


def test_a_shard_scored_over_another_corpus_aborts() -> None:
    """Matrix row "corpus drift": the corpus was rebuilt after a shard ran."""
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 1, build_id="e" * 64))

    assert any(
        "build_id" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_a_pinned_column_no_shard_ran_is_named() -> None:
    """A pass that quietly dropped a baseline is a table with a missing column, not a short file."""
    files = [
        replace(file, header=header(1, 0, paths=paths()[:1]))
        for file in split(every_score(), 1)
    ]

    assert any(
        "beta" in problem and "did not run that column" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_a_shard_recording_a_baseline_the_pins_do_not_declare_is_refused() -> None:
    extra = (*paths(), ExecutionPath("gamma", "f" * 40, CPU, 1, 8))
    files = [replace(file, header=header(1, 0, paths=extra)) for file in split(every_score(), 1)]

    assert any(
        "gamma" in problem and "does not pin" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_two_files_declaring_different_shard_counts_do_not_compose() -> None:
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(3, 1))

    assert any(
        "different shard counts" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_two_files_declaring_the_same_shard_index_are_refused() -> None:
    files = split(every_score(), 2)
    files[1] = replace(files[1], header=header(2, 0))

    assert any(
        "declared by more than one file" in problem
        for problem in path_problems(files, build_id=BUILD_ID, declared=DECLARED)
    )


def test_a_merge_over_no_shard_file_at_all_is_refused() -> None:
    """An empty file set satisfies every pairwise check there is."""
    assert path_problems([], build_id=BUILD_ID, declared=DECLARED) != ()


def test_a_matching_set_of_shards_has_no_path_problem() -> None:
    assert path_problems(split(every_score(), 3), build_id=BUILD_ID, declared=DECLARED) == ()


# --- the file ----------------------------------------------------------------------------------


def test_a_shard_file_round_trips_through_its_own_renderer() -> None:
    scores = every_score()
    file = parse_shard("scores-1-0.jsonl", render_shard(header(1, 0), scores))

    assert file.header == header(1, 0)
    assert sorted(file.scores, key=key_of) == sorted(scores, key=key_of)


def test_the_appended_line_and_the_whole_file_render_agree() -> None:
    """Two spellings of one record would let a resumed file differ from a single-run one."""
    scores = every_score()
    piecewise = render_shard(header(1, 0), ()) + "".join(render_record(s) for s in scores)

    assert piecewise == render_shard(header(1, 0), scores)


def test_a_file_truncated_mid_record_is_refused_rather_than_parsed() -> None:
    """Matrix row "partial shard file": a process killed while writing."""
    text = render_shard(header(1, 0), every_score())
    killed = text[: len(text) - 40]

    with pytest.raises(ScoreSetIncomplete, match="ends without a line terminator"):
        parse_shard("scores-1-0.jsonl", killed)


def test_a_file_truncated_exactly_on_a_boundary_is_caught_by_coverage_instead() -> None:
    """The other half of the pair, stated so the gap in the first is not mistaken for a hole."""
    text = render_shard(header(1, 0), every_score())
    on_boundary = text[: text.index("\n", text.index("\n") + 1) + 1]
    file = parse_shard("scores-1-0.jsonl", on_boundary)

    assert len(file.scores) == 1
    assert coverage_problems(demanded(), [file], shards=1) != ()


def test_an_empty_shard_file_is_refused() -> None:
    with pytest.raises(ScoreSetIncomplete, match="is empty"):
        parse_shard("scores-1-0.jsonl", "")


@pytest.mark.parametrize(
    "first_line",
    [
        pytest.param("not json", id="not-json"),
        pytest.param("[1, 2, 3]", id="not-an-object"),
        pytest.param('{"schema_version":1,"shards":1,"shard":0,"paths":[]}', id="missing-build-id"),
        pytest.param('{"schema_version":1,"shards":1,"shard":0,"build_id":"x"}', id="no-paths"),
    ],
)
def test_a_shard_file_whose_first_line_is_not_a_header_is_refused(first_line: str) -> None:
    """Four shapes, because each escapes through a different exception.

    Bad JSON raises `ValueError`, a JSON array fails the type check, a missing key raises
    `KeyError` and an absent paths list raises neither on its own. A reader that caught only the
    first would report the other three as an unclassified crash in the merge rather than as the
    refusal they are.
    """
    with pytest.raises(ScoreSetIncomplete, match="is not a shard header"):
        parse_shard("scores-1-0.jsonl", f"{first_line}\n")


def test_a_shard_file_from_a_schema_this_reader_does_not_read_is_refused() -> None:
    """A shard file written by one checkout is merged by another; that is the point of sharding."""
    text = render_shard(header(1, 0), ()).replace(
        f'"schema_version":{SHARD_SCHEMA_VERSION}', '"schema_version":99'
    )

    with pytest.raises(ScoreSetIncomplete, match="schema_version"):
        parse_shard("scores-1-0.jsonl", text)


def test_a_record_missing_a_field_is_refused_rather_than_defaulted() -> None:
    """A record with no `condition` must not arrive as one of the two conditions."""
    scores = every_score()[:1]
    line = json.loads(render_record(scores[0]))
    del line["condition"]
    text = render_shard(header(1, 0), ()) + json.dumps(line) + "\n"

    with pytest.raises(ScoreSetIncomplete, match="is not a score record"):
        parse_shard("scores-1-0.jsonl", text)


def test_a_record_whose_family_and_label_disagree_is_refused_at_read_time() -> None:
    """The corpus is verified on read; the copy into a scores file is a second chance to go wrong."""
    scores = every_score()[:1]
    line = json.loads(render_record(scores[0]))
    line["label"] = 0
    text = render_shard(header(1, 0), ()) + json.dumps(line) + "\n"

    with pytest.raises(ScoreSetIncomplete, match="is not a score record"):
        parse_shard("scores-1-0.jsonl", text)


def test_a_header_that_records_no_execution_path_is_refused() -> None:
    with pytest.raises(ValueError, match="records the execution path"):
        header(1, 0, paths=())


def test_a_header_naming_a_shard_outside_its_own_split_is_refused() -> None:
    with pytest.raises(ValueError, match="outside 0..2"):
        header(3, 3)


# --- the merge, end to end ------------------------------------------------------------------------


def test_a_complete_pass_merges_to_one_record_per_cell_and_nothing_else() -> None:
    scores = merged(split(every_score(), 3), shards=3)

    assert len(scores) == len(demanded())
    assert sorted(key_of(score) for score in scores) == sorted(demanded())


def test_the_merge_reports_every_problem_it_found_rather_than_the_first() -> None:
    """A run that is wrong in two ways tells the operator both, rather than in two round trips."""
    files = split(every_score(), 2)
    files[1] = replace(
        files[1], header=header(2, 1, providers=CUDA), scores=files[1].scores[:-1]
    )

    with pytest.raises(ScoreSetIncomplete) as abort:
        merged(files, shards=2)

    assert any("CUDA" in problem for problem in abort.value.problems)
    assert any("owes" in problem for problem in abort.value.problems)


def test_the_abort_carries_the_exit_code_the_story_assigned_it() -> None:
    assert ScoreSetIncomplete.exit_code == 27
    assert exit_code_for(ScoreSetIncomplete("boom")) == 27
    assert declared_exit_codes()[27] is ScoreSetIncomplete


def test_the_abort_will_not_be_raised_with_nothing_to_say() -> None:
    with pytest.raises(ValueError, match="at least one problem"):
        ScoreSetIncomplete()
