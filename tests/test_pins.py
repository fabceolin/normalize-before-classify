"""Every remote artifact is named once, in `pins.toml`, and verified before any inference.

Three things are under test here and they are different claims:

1. the committed `pins.toml` says everything a pin has to say, down to the artifact path;
2. `pins.py` refuses a file that does not, and refuses a baseline set SC5 does not admit;
3. no repository id, revision or artifact path lives anywhere else in the code.

The resolver is injected in every test below, which is what lets the whole verification path be
covered by a suite that runs with no network at all.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator

import pytest

import nbc
from nbc import pins as pins_module
from nbc.pins import (
    LINEAGE_RELATIONSHIPS,
    MINIMUM_BASELINES,
    NOT_DECLARED,
    OQ2_KEPT,
    OQ2_OUTCOMES,
    PINNED_PRECISION,
    PINS_FILENAME,
    SCHEMA_VERSION,
    SEEDED_FROM_TRAINING_SOURCE,
    SHARED_WINDOW_POLICY,
    TRAINED_ON,
    TRAINING_SOURCE_RELATIONSHIPS,
    WINDOW_POLICIES,
    BaselineIneligible,
    BaselineSetInvalid,
    PinMismatch,
    PinsFileInvalid,
    CHECKED_AGAINST_CACHE,
    CHECKED_AGAINST_HUB,
    RemoteArtifact,
    Resolution,
    load_pins,
    resolve_from_cache,
    verify_revisions,
)

# --- fixtures ---------------------------------------------------------------------------------
#
# Every sha below is deliberately fake and structurally valid. A test that hard-coded a real
# revision would be a second home for the pin, which is the defect this file also tests for.

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_D = "d" * 40

# The pinned dataset's card names four seeds and one baseline declares training on two of them.
# The fixtures keep that shape -- two seeds, neither declared by default -- so a test that wants
# the one-hop reach has to say so, and a test that does not is not silently sitting on one.
SEED_ONE = "example/seed-one"
SEED_TWO = "example/seed-two"


def _baseline(
    key: str = "first",
    repository: str = "example/first-model",
    revision: str = SHA_A,
    architecture_family: str = "deberta-v2",
    tokenizer_family: str = "sentencepiece-unigram",
    **overrides: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "repository": repository,
        "revision": revision,
        "threshold": 0.5,
        "graph_path": "onnx/model.onnx",
        "precision": PINNED_PRECISION,
        "graph_bytes": 12345,
        "tokenizer_path": "onnx/tokenizer.json",
        "config_path": "onnx/config.json",
        "architecture_family": architecture_family,
        "tokenizer_family": tokenizer_family,
        "window_policy": SHARED_WINDOW_POLICY,
        "window": {
            "length": 512,
            "source": "onnx/config.json::max_position_embeddings",
            "confirmed_on": "2026-08-28",
            "confirmed_revision": revision,
        },
        "licence": {
            "identifier": "apache-2.0",
            "source": "card",
            "attribution": "someone",
            "redistributed": False,
        },
        "lineage": {
            "checked_on": "2026-08-28",
            "card_revision": revision,
            "attack_datasets": {"example/attacks": NOT_DECLARED},
            "training_sources": {SEED_ONE: NOT_DECLARED, SEED_TWO: NOT_DECLARED},
        },
        "oq2": {
            "outcome": OQ2_KEPT,
            "decided_on": "2026-08-29",
            "decided_revision": revision,
            "dataset_revision": SHA_D,
            "measured_at_threshold": 0.5,
            "hits": 90,
            "clean_recall": 0.9,
            "sample_size": 100,
            "overlap_rows": 0,
            "judged_sufficient_by": "fixture",
            "source": "spikes/oq2_clean_recall.py",
        },
    }
    entry.update(overrides)
    return entry


def _dataset(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": "attacks",
        "repository": "example/attacks",
        "revision": SHA_D,
        "splits": ["train", "test"],
        "attack_label": 1,
        "licence": {
            "identifier": NOT_DECLARED,
            "source": "nothing on the card",
            "attribution": "example/attacks",
            "redistributed": True,
            # Redistributed under an undeclared licence, so the file must record the open
            # question or refuse to load. The default fixture records it; the test below
            # removes it and asserts the refusal.
            "unresolved": "2026-08-29: fixture, question deliberately left open",
        },
        "provenance": {
            "checked_on": "2026-08-28",
            "card_revision": SHA_D,
            "seeds": [SEED_ONE, SEED_TWO],
        },
    }
    entry.update(overrides)
    return entry


def _document(
    baselines: list[dict[str, Any]] | None = None,
    datasets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "verified_on": "2026-08-28",
            "verified_against": "the live artifacts",
        },
        "baseline": baselines
        if baselines is not None
        else [
            _baseline(),
            _baseline(
                key="second",
                repository="example/second-model",
                revision=SHA_B,
                architecture_family="bert",
                tokenizer_family="wordpiece",
            ),
        ],
        "attack_dataset": datasets if datasets is not None else [_dataset()],
    }


def _dump(value: Any) -> str:
    """A minimal TOML writer. The fixtures are dicts, lists, strings, ints, floats and bools."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_dump(item) for item in value) + "]"
    raise TypeError(f"the fixture writer does not handle {type(value).__name__}")


def _write_table(lines: list[str], header: str, table: dict[str, Any], array: bool) -> None:
    lines.append(f"[[{header}]]" if array else f"[{header}]")
    nested: list[tuple[str, dict[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, dict):
            nested.append((key, value))
        else:
            lines.append(f"{_key(key)} = {_dump(value)}")
    for key, value in nested:
        _write_table(lines, f"{header}.{key}", value, array=False)


def _key(key: str) -> str:
    return key if key.replace("_", "").replace("-", "").isalnum() else f'"{key}"'


def write_pins(root: Path, document: dict[str, Any]) -> Path:
    """Render a fixture document to `pins.toml` under `root`."""
    lines: list[str] = []
    _write_table(lines, "meta", document["meta"], array=False)
    for entry in document.get("baseline", []):
        _write_table(lines, "baseline", entry, array=True)
    for entry in document.get("attack_dataset", []):
        _write_table(lines, "attack_dataset", entry, array=True)
    path = root / PINS_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def committed_pins() -> Any:
    return load_pins()


@pytest.fixture()
def committed_document(repo_root: Path) -> dict[str, Any]:
    return tomllib.loads((repo_root / PINS_FILENAME).read_text(encoding="utf-8"))


def echoing_resolver(artifact: RemoteArtifact) -> Resolution | None:
    """A world in which every pin still resolves to itself, and was asked."""
    return Resolution(artifact.revision, CHECKED_AGAINST_HUB)


# --- the fixture writer itself -----------------------------------------------------------


def test_the_fixture_writer_round_trips(tmp_path: Path) -> None:
    """Otherwise a fixture bug reads as a `pins.py` bug in every test below."""
    document = _document()
    path = write_pins(tmp_path, document)
    assert tomllib.loads(path.read_text(encoding="utf-8")) == document


# --- happy path -------------------------------------------------------------------------------


def test_a_well_formed_file_loads(tmp_path: Path) -> None:
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)

    assert [baseline.key for baseline in pins.baselines] == ["first", "second"]
    assert [dataset.key for dataset in pins.attack_datasets] == ["attacks"]
    assert pins.schema_version == SCHEMA_VERSION


def test_the_committed_file_loads_and_declares_the_floor(committed_pins: Any) -> None:
    assert len(committed_pins.baselines) >= MINIMUM_BASELINES
    assert len(committed_pins.attack_datasets) >= 1


def test_every_committed_baseline_carries_every_field_a_pin_has_to_carry(
    committed_pins: Any,
) -> None:
    """The AC's list, read off the loaded record rather than off the file's prose."""
    for baseline in committed_pins.baselines:
        assert baseline.repository and "/" in baseline.repository
        assert len(baseline.revision) == 40
        assert 0.0 <= baseline.threshold <= 1.0
        assert baseline.graph_path.endswith(".onnx")
        assert baseline.precision == PINNED_PRECISION
        assert baseline.tokenizer_path.endswith("tokenizer.json")
        assert baseline.config_path.endswith("config.json")
        assert baseline.window.length > 0
        assert baseline.config_path in baseline.window.source
        assert baseline.window.confirmed_on
        assert baseline.window.confirmed_revision == baseline.revision
        assert baseline.window_policy in WINDOW_POLICIES
        assert baseline.licence.identifier
        assert baseline.licence.attribution
        assert baseline.architecture_family
        assert baseline.tokenizer_family
        for dataset in committed_pins.attack_datasets:
            assert baseline.lineage.relationship_to(dataset.repository) is not None


def test_the_committed_baselines_span_two_families(committed_pins: Any) -> None:
    """SC5's independence claim, on the file that actually ships."""
    pairs = [baseline.family_pair for baseline in committed_pins.baselines]
    assert len(set(pairs)) == len(pairs)


def test_the_committed_pins_name_the_artifact_path_and_not_only_the_revision(
    committed_pins: Any,
) -> None:
    """One pinned repository ships two different `tokenizer.json` at the same commit.

    A pin naming only the revision leaves the choice between them to the loader, which is an
    unnamed source of variance inside a reproducibility promise.
    """
    for baseline in committed_pins.baselines:
        assert baseline.tokenizer_path
        assert baseline.graph_path
        assert baseline.config_path
        # Graph and tokenizer come from the same directory: pairing a root tokenizer with a
        # subdirectory graph pairs two artifacts the publisher never paired.
        assert (
            Path(baseline.graph_path).parent == Path(baseline.tokenizer_path).parent
            == Path(baseline.config_path).parent
        )


def test_no_committed_graph_is_a_reduced_precision_one(committed_pins: Any) -> None:
    for baseline in committed_pins.baselines:
        assert baseline.precision == PINNED_PRECISION
        lowered = baseline.graph_path.lower()
        for banned in ("fp16", "float16", "bf16", "mixed", "int8", "quant"):
            assert banned not in lowered


def test_a_missing_licence_is_recorded_as_a_finding_not_as_an_absent_key(
    committed_pins: Any,
) -> None:
    """An undeclared licence has to survive into the results as a finding.

    `not-declared` is a value, so a reader of `results.json` sees the absence rather than a
    missing key they can mistake for an oversight.
    """
    for source in (*committed_pins.baselines, *committed_pins.attack_datasets):
        assert source.licence.identifier
        assert source.licence.source
        if source.licence.identifier == NOT_DECLARED:
            assert not source.licence.declared


# --- structural refusals ----------------------------------------------------------------------


def test_a_missing_file_aborts_naming_the_path(tmp_path: Path) -> None:
    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert PINS_FILENAME in str(abort.value)
    assert str(tmp_path) in str(abort.value)


def test_unparseable_toml_aborts(tmp_path: Path) -> None:
    (tmp_path / PINS_FILENAME).write_text("[meta\nnot toml", encoding="utf-8")
    with pytest.raises(PinsFileInvalid, match="not valid TOML"):
        load_pins(tmp_path)


def test_a_missing_key_aborts_naming_the_baseline_and_the_key(tmp_path: Path) -> None:
    document = _document()
    del document["baseline"][0]["threshold"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert "threshold is missing" in message
    assert "first" in message


def test_a_wrong_type_aborts_naming_the_expected_type(tmp_path: Path) -> None:
    document = _document()
    document["baseline"][0]["revision"] = 90
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="must be a string, got int"):
        load_pins(tmp_path)


@pytest.mark.parametrize(
    "revision", ["not-a-sha", "90C9989B1A342275DD0D1A95AAD283C04E075671", "abc123", "a" * 39]
)
def test_a_revision_that_is_not_a_full_lowercase_sha_aborts(
    tmp_path: Path, revision: str
) -> None:
    """A short sha is ambiguous, and an uppercase one is not what the hub returns."""
    document = _document()
    document["baseline"][0]["revision"] = revision
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="40-character lowercase hex commit sha"):
        load_pins(tmp_path)


@pytest.mark.parametrize(
    "graph_path",
    [
        "onnx/model_fp16.onnx",
        "onnx/model_mixed.onnx",
        "onnx/model_int8.onnx",
        "model.quant.onnx",
    ],
)
def test_a_reduced_precision_graph_is_refused(tmp_path: Path, graph_path: str) -> None:
    """Reduced precision moves scores in the last decimals and the threshold flips the class."""
    document = _document()
    document["baseline"][0]["graph_path"] = graph_path
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="reduced-precision or quantized graph"):
        load_pins(tmp_path)


def test_a_precision_other_than_fp32_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["baseline"][0]["precision"] = "fp16"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="only 'fp32' may be pinned"):
        load_pins(tmp_path)


@pytest.mark.parametrize("threshold", [1.5, -0.1])
def test_a_threshold_outside_zero_to_one_is_refused(tmp_path: Path, threshold: float) -> None:
    document = _document()
    document["baseline"][0]["threshold"] = threshold
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match=r"threshold must lie in \[0, 1\]"):
        load_pins(tmp_path)


def test_two_baselines_sharing_a_repository_are_refused(tmp_path: Path) -> None:
    document = _document(
        baselines=[
            _baseline(),
            _baseline(
                key="second",
                revision=SHA_B,
                architecture_family="bert",
                tokenizer_family="wordpiece",
            ),
        ]
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="share repository"):
        load_pins(tmp_path)


def test_a_window_read_from_the_tokenizer_config_is_refused(tmp_path: Path) -> None:
    """`model_max_length` is a ~1e30 sentinel in the pinned repositories."""
    document = _document()
    document["baseline"][0]["window"]["source"] = "tokenizer_config.json::model_max_length"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="sentinel"):
        load_pins(tmp_path)


def test_a_window_confirmed_against_another_revision_is_refused(tmp_path: Path) -> None:
    """`config.json` is authoritative for capacity and a card can declare a smaller operative
    window; telling them apart is a human reading, so it is confirmed once per change of pin.
    A date alone would keep looking fresh while describing an artifact this file no longer pins.
    """
    document = _document()
    document["baseline"][0]["window"]["confirmed_revision"] = SHA_D
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="confirmed_revision"):
        load_pins(tmp_path)


def test_a_window_read_from_a_config_this_baseline_does_not_pin_is_refused(tmp_path: Path) -> None:
    """One pinned repository ships two files of one name at one revision. The pin names the path."""
    document = _document()
    document["baseline"][0]["window"]["source"] = "config.json::max_position_embeddings"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="does not pin"):
        load_pins(tmp_path)


def test_a_window_policy_no_strategy_implements_is_refused(tmp_path: Path) -> None:
    """A policy is a length, a stride and an aggregation together. A name is not a policy."""
    document = _document()
    document["baseline"][0]["window_policy"] = "publisher"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="admitted values"):
        load_pins(tmp_path)


def test_the_admitted_window_policies_are_not_empty() -> None:
    """A vacuous vocabulary would refuse every baseline instead of the wrong ones."""
    assert SHARED_WINDOW_POLICY in WINDOW_POLICIES


# --- OQ2: each baseline is worth a column, and the file says so ------------------------------


def test_an_oq2_result_decided_against_another_revision_is_refused(tmp_path: Path) -> None:
    """A recall measured against a revision this file no longer pins is a check nobody re-ran.

    The third declaration in this file to learn it: the date is metadata, the revision is the
    gate.
    """
    document = _document()
    document["baseline"][0]["oq2"]["decided_revision"] = SHA_D
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="decided_revision"):
        load_pins(tmp_path)


def test_an_oq2_outcome_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    """`dropped` is not a way out: SC5's floor is two and removal fails it outright."""
    document = _document()
    document["baseline"][0]["oq2"]["outcome"] = "dropped"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="admitted values"):
        load_pins(tmp_path)


@pytest.mark.parametrize("recall", [1.4, -0.1])
def test_an_oq2_clean_recall_outside_zero_to_one_is_refused(
    tmp_path: Path, recall: float
) -> None:
    document = _document()
    document["baseline"][0]["oq2"]["clean_recall"] = recall
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="must lie in"):
        load_pins(tmp_path)


def test_an_oq2_sample_size_of_zero_is_refused(tmp_path: Path) -> None:
    """A rate over no items is not a rate."""
    document = _document()
    document["baseline"][0]["oq2"]["sample_size"] = 0
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="sample_size"):
        load_pins(tmp_path)


def test_a_baseline_with_no_oq2_record_is_refused(tmp_path: Path) -> None:
    """A baseline nobody asked is not the same as a baseline that answered."""
    document = _document()
    del document["baseline"][0]["oq2"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="oq2 is missing"):
        load_pins(tmp_path)


def test_the_committed_baselines_each_carry_an_oq2_answer(committed_pins: Any) -> None:
    """OQ2 gates the epic, so every surviving column has a measurement behind it."""
    for baseline in committed_pins.baselines:
        assert baseline.oq2.outcome in OQ2_OUTCOMES
        assert baseline.oq2.decided_revision == baseline.revision
        assert baseline.oq2.sample_size > 0
        assert 0.0 <= baseline.oq2.clean_recall <= 1.0
        assert baseline.oq2.source


def test_the_run_fields_carry_the_oq2_answer(committed_pins: Any) -> None:
    """A reader of `results.json` can see why each column was worth having."""
    for baseline in committed_pins.as_run_fields()["pins"]["baselines"]:
        # Presence, not truthiness: a measured recall of 0.0 is a finding, not a missing field.
        assert isinstance(baseline["oq2"]["clean_recall"], float)
        assert baseline["oq2"]["decided_on"]
        assert baseline["oq2"]["source"]


def test_a_baseline_silent_about_a_pinned_attack_dataset_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"] = {}
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="declares no relationship to the pinned"):
        load_pins(tmp_path)


def test_a_file_pinning_no_attack_dataset_is_refused(tmp_path: Path) -> None:
    """With no dataset pinned, the lineage declaration is a check against an empty set."""
    write_pins(tmp_path, _document(datasets=[]))

    with pytest.raises(PinsFileInvalid, match="no \\[\\[attack_dataset\\]\\] is pinned"):
        load_pins(tmp_path)


def test_a_repository_shared_by_a_baseline_and_a_dataset_is_refused(tmp_path: Path) -> None:
    """Models and datasets live in separate namespaces on the hub, so one id can name both.

    `verify_revisions` reports its resolutions by repository id, and two artifacts sharing one
    would collapse into a single line -- leaving one pin unreported while the run looks
    verified.
    """
    document = _document(datasets=[_dataset(repository="example/first-model")])
    for entry in document["baseline"]:
        entry["lineage"]["attack_datasets"] = {"example/first-model": NOT_DECLARED}
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="share repository"):
        load_pins(tmp_path)


def test_a_relationship_to_an_unpinned_dataset_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/not-pinned"] = "trained-on"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="which is not a pinned attack dataset"):
        load_pins(tmp_path)


def test_a_relationship_outside_the_declared_vocabulary_is_refused(tmp_path: Path) -> None:
    """The eligibility rule reads this value, so free text here is a rule nobody enforces.

    Story 1.3 recorded the relationship and left refusing the baseline to the gate. The gate
    exists now, and closing the vocabulary is what lets it read a value instead of a sentence.
    """
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = (
        "declared-training-source"
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert "declared-training-source" in str(abort.value)
    for admitted in LINEAGE_RELATIONSHIPS:
        assert admitted in str(abort.value)


def test_a_training_source_may_not_claim_the_one_hop_relationship(tmp_path: Path) -> None:
    """`seeded-from-...` describes a reach to a dataset, not a card's own declaration."""
    document = _document()
    document["baseline"][0]["lineage"]["training_sources"][SEED_ONE] = (
        SEEDED_FROM_TRAINING_SOURCE
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert SEEDED_FROM_TRAINING_SOURCE not in TRAINING_SOURCE_RELATIONSHIPS
    assert "training_sources" in str(abort.value)


def test_a_foreign_schema_version_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["meta"]["schema_version"] = SCHEMA_VERSION + 1
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="schema_version"):
        load_pins(tmp_path)


def test_every_problem_in_a_bad_file_is_reported_at_once(tmp_path: Path) -> None:
    """A file wrong in three places should say so in one run, not over three."""
    document = _document()
    del document["baseline"][0]["threshold"]
    document["baseline"][0]["revision"] = "nope"
    document["baseline"][1]["precision"] = "fp16"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert len(abort.value.problems) >= 3


# --- AD-26 teeth 1 and 2: no baseline is scored over its own training text ---------------------
#
# Two teeth, one rule reaching two distances. Tooth 1 is what a model card says about the pinned
# pool. Tooth 2 is what the pool's own card says it was built from, which is the half no model
# card can show and the half that got through two rounds of reading.


def test_a_baseline_trained_on_a_pinned_attack_dataset_is_ineligible(tmp_path: Path) -> None:
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = TRAINED_ON
    write_pins(tmp_path, document)

    with pytest.raises(BaselineIneligible) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert "example/first-model" in message
    assert "example/attacks" in message


def test_the_ineligibility_message_names_the_card_revision_and_the_date(tmp_path: Path) -> None:
    """A reader has to be able to tell which reading of which card produced the refusal."""
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = TRAINED_ON
    write_pins(tmp_path, document)

    with pytest.raises(BaselineIneligible) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert SHA_A in message
    assert "2026-08-28" in message


def test_an_ineligible_baseline_is_told_to_be_replaced_rather_than_removed(
    tmp_path: Path,
) -> None:
    """SC5 sits exactly on its floor of two, so a removal breaks the claim outright.

    The replacement bar travels with the abort because the next person to read it is deciding
    what to do, and the four other conditions a replacement has to clear live nowhere else.
    """
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = TRAINED_ON
    write_pins(tmp_path, document)

    with pytest.raises(BaselineIneligible) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert "replacement is required, not a removal" in message
    for bar in ("ONNX graph", "tokenizer artifact", "id2label", "tokenizer family"):
        assert bar in message


def test_an_undeclared_one_hop_reach_is_caught_by_the_same_rule(tmp_path: Path) -> None:
    """The dataset's card names a seed; the baseline declares training on it; nothing else does.

    No model card mentions a dataset built downstream of it, so tooth 1 sees nothing here. This
    is the failure that reached a published measurement twice.
    """
    document = _document()
    document["baseline"][0]["lineage"]["training_sources"][SEED_ONE] = TRAINED_ON
    write_pins(tmp_path, document)

    with pytest.raises(BaselineIneligible) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert SEED_ONE in message
    assert "example/attacks" in message
    assert "one hop" in message


def test_a_seed_no_baseline_declares_produces_no_hop(tmp_path: Path) -> None:
    """A seed is only a reach when a baseline declares training on it."""
    write_pins(tmp_path, _document())

    pins = load_pins(tmp_path)
    assert pins.one_hop_reaches() == {}
    assert pins.required_exclusion_sources() == ()


def test_a_declared_one_hop_reach_loads_and_obliges_an_exclusion_source(
    tmp_path: Path,
) -> None:
    """The only way past the hop, and it is not free.

    Declaring it says the coincident rows are removed before anything is measured, and it turns
    every seed the reach runs through into a source the build has to remove the overlap with.
    """
    document = _document()
    document["baseline"][0]["lineage"]["training_sources"][SEED_ONE] = TRAINED_ON
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = (
        SEEDED_FROM_TRAINING_SOURCE
    )
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert pins.one_hop_reaches() == {("example/first-model", "example/attacks"): (SEED_ONE,)}
    assert pins.required_exclusion_sources() == (SEED_ONE,)


def test_a_declared_hop_the_seeds_do_not_carry_is_refused(tmp_path: Path) -> None:
    """A claim about nothing is how an exclusion source gets pinned for a reason that expired."""
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = (
        SEEDED_FROM_TRAINING_SOURCE
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="training on none of the seeds"):
        load_pins(tmp_path)


def test_a_baseline_silent_about_a_seed_is_refused(tmp_path: Path) -> None:
    """Silence would read as `no reach` while meaning `nobody looked`."""
    document = _document()
    del document["baseline"][0]["lineage"]["training_sources"][SEED_TWO]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert SEED_TWO in str(abort.value)
    assert "declares nothing about" in str(abort.value)


def test_a_baseline_with_no_training_sources_table_is_refused(tmp_path: Path) -> None:
    """The table is mandatory even when it would be empty: silence is not an answer."""
    document = _document()
    del document["baseline"][0]["lineage"]["training_sources"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="training_sources is missing"):
        load_pins(tmp_path)


def test_a_training_source_that_is_nobody_s_seed_is_admitted(tmp_path: Path) -> None:
    """The block records the card's `datasets:` list, not only the part a seed check needs.

    The measured-overlap filter draws its exclusion set from declared training sources, so
    refusing the ones that are nobody's seed today would mean deleting a read fact to satisfy
    a check -- the opposite of what recording lineage is for.
    """
    document = _document()
    document["baseline"][0]["lineage"]["training_sources"]["example/some-other-corpus"] = (
        TRAINED_ON
    )
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert pins.baselines[0].lineage.trains_on("example/some-other-corpus")
    assert pins.required_exclusion_sources() == ()


def test_a_dataset_with_no_provenance_block_is_refused(tmp_path: Path) -> None:
    """One hop is unmeasurable against a seed list nobody wrote down."""
    document = _document(datasets=[_dataset()])
    del document["attack_dataset"][0]["provenance"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="provenance"):
        load_pins(tmp_path)


def test_a_card_that_names_no_seed_is_a_fact_not_a_missing_declaration(
    tmp_path: Path,
) -> None:
    document = _document(
        datasets=[
            _dataset(
                provenance={"checked_on": "2026-08-28", "card_revision": SHA_D, "seeds": []}
            )
        ]
    )
    for entry in document["baseline"]:
        entry["lineage"]["training_sources"] = {}
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert pins.attack_datasets[0].provenance.seeds == ()
    assert pins.required_exclusion_sources() == ()


def test_a_dataset_naming_itself_as_its_own_seed_is_refused(tmp_path: Path) -> None:
    document = _document(
        datasets=[
            _dataset(
                provenance={
                    "checked_on": "2026-08-28",
                    "card_revision": SHA_D,
                    "seeds": ["example/attacks"],
                }
            )
        ]
    )
    for entry in document["baseline"]:
        entry["lineage"]["training_sources"] = {"example/attacks": NOT_DECLARED}
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="which is the dataset itself"):
        load_pins(tmp_path)


def test_a_lineage_check_run_against_another_card_revision_is_refused(tmp_path: Path) -> None:
    """A check never re-run after a pin moved is visible, not assumed to still hold."""
    document = _document()
    document["baseline"][0]["lineage"]["card_revision"] = SHA_B
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert SHA_B in message and SHA_A in message
    assert "no longer pins" in message


def test_a_provenance_check_run_against_another_card_revision_is_refused(
    tmp_path: Path,
) -> None:
    document = _document(
        datasets=[
            _dataset(
                provenance={
                    "checked_on": "2026-08-28",
                    "card_revision": SHA_B,
                    "seeds": [SEED_ONE, SEED_TWO],
                }
            )
        ]
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert SHA_B in str(abort.value) and SHA_D in str(abort.value)


def test_the_date_alone_does_not_keep_a_declaration_fresh(tmp_path: Path) -> None:
    """The revision is the gate. A pin can move on the same day the check was recorded."""
    document = _document()
    document["baseline"][0]["lineage"]["checked_on"] = "2026-08-28"
    document["baseline"][0]["lineage"]["card_revision"] = SHA_B
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid, match="no longer pins"):
        load_pins(tmp_path)


def test_the_committed_pins_clear_both_teeth(committed_pins: Any) -> None:
    """The file that actually ships, read through the rule rather than through its comments."""
    for baseline in committed_pins.baselines:
        for dataset in committed_pins.attack_datasets:
            assert baseline.lineage.relationship_to(dataset.repository) != TRAINED_ON
            for seed in dataset.provenance.seeds:
                assert seed in baseline.lineage.training_sources


def test_the_committed_one_hop_reach_is_the_one_the_architecture_records(
    committed_pins: Any,
) -> None:
    """Exactly one baseline reaches the pinned pool through its seeds, and it says so.

    The seeds it reaches through are the two the exclusion filter owes the corpus. The other
    baseline declares training on none of them, which is why it clears the gate outright.
    """
    reaches = committed_pins.one_hop_reaches()
    assert len(reaches) == 1
    (reaching_baseline, reached_dataset), seeds = next(iter(reaches.items()))
    assert len(seeds) == 2
    assert set(seeds) <= set(committed_pins.required_exclusion_sources())

    declaring = [b for b in committed_pins.baselines if b.repository == reaching_baseline]
    assert declaring[0].lineage.relationship_to(reached_dataset) == SEEDED_FROM_TRAINING_SOURCE
    for other in committed_pins.baselines:
        if other.repository != reaching_baseline:
            assert other.lineage.relationship_to(reached_dataset) == NOT_DECLARED


def test_the_committed_checks_were_run_against_the_pinned_revisions(
    committed_pins: Any,
) -> None:
    for baseline in committed_pins.baselines:
        assert baseline.lineage.card_revision == baseline.revision
        assert baseline.lineage.checked_on
    for dataset in committed_pins.attack_datasets:
        assert dataset.provenance.card_revision == dataset.revision
        assert dataset.provenance.checked_on


def test_the_run_fields_carry_the_lineage_the_gate_read(committed_pins: Any) -> None:
    """`results.json` has to show what was declared, not only that something was."""
    fields = committed_pins.as_run_fields()["pins"]
    assert fields["required_exclusion_sources"]
    for baseline in fields["baselines"]:
        assert baseline["lineage"]["training_sources"]
        assert baseline["lineage"]["card_revision"]
    for dataset in fields["attack_datasets"]:
        assert dataset["provenance"]["seeds"]
        assert dataset["provenance"]["card_revision"]


# --- the baseline set -------------------------------------------------------------------------


def test_fewer_than_two_baselines_aborts(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(baselines=[_baseline()]))

    with pytest.raises(BaselineSetInvalid) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert "1 baseline" in message
    assert str(MINIMUM_BASELINES) in message


def test_no_baselines_at_all_aborts(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(baselines=[]))
    with pytest.raises(BaselineSetInvalid):
        load_pins(tmp_path)


def test_two_baselines_declaring_the_same_family_pair_abort(tmp_path: Path) -> None:
    """SC5 checked, not asserted: two models that tokenize alike prove nothing."""
    document = _document(
        baselines=[
            _baseline(architecture_family="bert", tokenizer_family="wordpiece"),
            _baseline(
                key="second",
                repository="example/second-model",
                revision=SHA_B,
                architecture_family="bert",
                tokenizer_family="wordpiece",
            ),
        ]
    )
    write_pins(tmp_path, document)

    with pytest.raises(BaselineSetInvalid) as abort:
        load_pins(tmp_path)
    message = str(abort.value)
    assert "example/first-model" in message and "example/second-model" in message
    assert "bert/wordpiece" in message


def test_a_shared_tokenizer_family_under_two_architectures_is_admitted(tmp_path: Path) -> None:
    """The key is the pair, not either half, and this is the test that says so.

    Without it, the abort above would pass just as well against an implementation that keyed on
    the tokenizer family alone -- a stricter rule than SC5 states, refusing sets it admits.
    """
    document = _document(
        baselines=[
            _baseline(tokenizer_family="wordpiece"),
            _baseline(
                key="second",
                repository="example/second-model",
                revision=SHA_B,
                architecture_family="bert",
                tokenizer_family="wordpiece",
            ),
        ]
    )
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert len(pins.baselines) == 2


# --- verification -----------------------------------------------------------------------------


def test_verification_passes_when_every_pin_resolves_to_itself(tmp_path: Path) -> None:
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)

    resolved = verify_revisions(pins, echoing_resolver)
    # The sha AND what it was checked against. A run verified entirely from cache compared
    # nothing to the world, and results.json has to be able to say so.
    assert resolved == {
        "example/first-model": f"{SHA_A}@{CHECKED_AGAINST_HUB}",
        "example/second-model": f"{SHA_B}@{CHECKED_AGAINST_HUB}",
        "example/attacks": f"{SHA_D}@{CHECKED_AGAINST_HUB}",
    }


def test_a_moved_revision_aborts_naming_the_artifact_and_both_shas(tmp_path: Path) -> None:
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)
    moved = "c" * 40

    def resolver(artifact: RemoteArtifact) -> Resolution | None:
        sha = moved if artifact.repository == "example/first-model" else artifact.revision
        return Resolution(sha, CHECKED_AGAINST_HUB)

    with pytest.raises(PinMismatch) as abort:
        verify_revisions(pins, resolver)
    message = str(abort.value)
    assert "example/first-model" in message
    assert SHA_A in message
    assert moved in message


def test_a_moved_dataset_revision_aborts_too(tmp_path: Path) -> None:
    """Datasets are pinned as seriously as models; a moving pool moves the table."""
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)
    moved = "e" * 40

    def resolver(artifact: RemoteArtifact) -> Resolution | None:
        sha = moved if artifact.kind == "dataset" else artifact.revision
        return Resolution(sha, CHECKED_AGAINST_HUB)

    with pytest.raises(PinMismatch, match="example/attacks"):
        verify_revisions(pins, resolver)


def test_an_unresolvable_pin_aborts_rather_than_passing_quietly(tmp_path: Path) -> None:
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)

    with pytest.raises(PinMismatch) as abort:
        verify_revisions(pins, lambda artifact: None)
    assert len(abort.value.problems) == 3
    assert "could not be resolved" in str(abort.value)


def test_a_cached_snapshot_resolves_with_no_network(tmp_path: Path) -> None:
    """A cache hit resolves, and says it was checked against the cache and not the world."""
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    snapshot = tmp_path / artifact.cache_directory / "snapshots" / SHA_A
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    resolution = resolve_from_cache(artifact, cache_root=tmp_path)

    assert resolution == Resolution(SHA_A, CHECKED_AGAINST_CACHE)


def test_an_uncached_artifact_resolves_to_nothing_from_the_cache(tmp_path: Path) -> None:
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    assert resolve_from_cache(artifact, cache_root=tmp_path) is None


def test_a_snapshot_at_a_different_sha_does_not_satisfy_the_pin(tmp_path: Path) -> None:
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    (tmp_path / artifact.cache_directory / "snapshots" / SHA_B).mkdir(parents=True)
    assert resolve_from_cache(artifact, cache_root=tmp_path) is None


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("model", "models--example--first"), ("dataset", "datasets--example--first")],
)
def test_the_cache_directory_follows_the_hub_layout(kind: str, expected: str) -> None:
    assert RemoteArtifact(kind, "example/first", SHA_A).cache_directory == expected


def test_the_api_url_carries_the_pinned_revision() -> None:
    url = RemoteArtifact("model", "example/first", SHA_A).api_url
    assert url.endswith(f"/example/first/revision/{SHA_A}")
    assert "/api/models/" in url


# --- the file is the only home -----------------------------------------------------------


def _source_files() -> list[Path]:
    return sorted(Path(nbc.__file__).resolve().parent.rglob("*.py"))


def _string_constants(path: Path) -> Iterator[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _pinned_identifiers(document: dict[str, Any]) -> set[str]:
    """Every value in `pins.toml` that identifies a remote artifact or sizes a draw."""
    identifiers: set[str] = set()

    def walk(value: Any, key: str) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, child_key)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str):
            if key in {"repository", "revision", "card_revision"} or key.endswith("_path"):
                identifiers.add(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            if "sample_size" in key:
                identifiers.add(str(value))

    walk(document, "")
    for entry in document.get("baseline", []):
        lineage = entry.get("lineage", {})
        identifiers.update(lineage.get("attack_datasets", {}))
        identifiers.update(lineage.get("training_sources", {}))
    for entry in document.get("attack_dataset", []):
        identifiers.update(entry.get("provenance", {}).get("seeds", []))
    return {identifier for identifier in identifiers if identifier}


def test_no_pinned_identifier_appears_as_a_literal_in_the_source_tree(
    committed_document: dict[str, Any],
) -> None:
    """The pin has one home. A second copy is a pin that can drift without anyone noticing.

    The scan is over exact values, so it cannot catch a repository id that was *never* pinned.
    The sha scan below covers the revision half of that gap; the rest is code review's job.
    """
    identifiers = _pinned_identifiers(committed_document)
    assert identifiers, "the scan found nothing to look for, which would pass vacuously"

    offenders: list[str] = []
    for path in _source_files():
        for lineno, value in _string_constants(path):
            if value in identifiers:
                offenders.append(f"{path.name}:{lineno} {value!r}")

    assert not offenders, (
        f"these belong only in {PINS_FILENAME}: " + "; ".join(offenders)
    )


def test_no_commit_sha_appears_as_a_literal_in_the_source_tree() -> None:
    """Catches a revision that was hard-coded without ever being pinned."""
    offenders: list[str] = []
    for path in _source_files():
        for lineno, value in _string_constants(path):
            if len(value) == 40 and all(char in "0123456789abcdef" for char in value):
                offenders.append(f"{path.name}:{lineno} {value!r}")

    assert not offenders, "a commit sha is a pin: " + "; ".join(offenders)


def test_only_one_module_names_the_pins_file() -> None:
    """`pins.py` is the only reader, so a second module reading it is a second parser."""
    readers = [
        path.name
        for path in _source_files()
        if any(PINS_FILENAME == value for _, value in _string_constants(path))
    ]
    assert readers == ["pins.py"], readers


# --- the module stays a leaf -------------------------------------------------------------


def test_pins_imports_only_the_error_base_from_this_project() -> None:
    """Read from the parsed source, so an import inside a function body is caught too."""
    path = Path(pins_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nbc" or alias.name.startswith("nbc."):
                    offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                offenders.append(f"{path.name}:{node.lineno} relative import")
            elif node.module == "nbc" or (node.module or "").startswith("nbc."):
                if node.module != "nbc.errors":
                    offenders.append(f"{path.name}:{node.lineno} from {node.module} import ...")

    assert not offenders, "pins.py may import only nbc.errors: " + "; ".join(offenders)


def test_importing_pins_pulls_in_no_other_nbc_module() -> None:
    code = (
        "import sys, nbc.pins; "
        "print(sorted(m for m in sys.modules if m == 'nbc' or m.startswith('nbc.')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "['nbc', 'nbc.errors', 'nbc.pins']", completed.stdout


def test_importing_pins_does_not_import_the_inference_runtime() -> None:
    """Pins are verified before any inference, so verifying them must not start the runtime."""
    code = "import sys, nbc.pins; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout


def test_no_toml_library_other_than_the_standard_one_is_imported() -> None:
    source = Path(pins_module.__file__).read_text(encoding="utf-8")
    assert "import tomllib" in source
    for banned in ("import toml\n", "import tomlkit", "import tomli\n"):
        assert banned not in source


# --- the command line -------------------------------------------------------------------------


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nbc.pins", *args], capture_output=True, text=True
    )


def test_the_module_prints_the_run_fields_and_exits_zero() -> None:
    import json

    completed = _run_module()
    assert completed.returncode == 0, completed.stderr
    fields = json.loads(completed.stdout)
    assert "pins" in fields
    assert len(fields["pins"]["baselines"]) >= MINIMUM_BASELINES


def test_the_module_exits_with_the_declared_code_on_a_bad_file(tmp_path: Path) -> None:
    (tmp_path / PINS_FILENAME).write_text("[meta\n", encoding="utf-8")
    completed = _run_module("--root", str(tmp_path))
    assert completed.returncode == PinsFileInvalid.exit_code
    assert PINS_FILENAME in completed.stderr


def test_the_module_exits_with_the_declared_code_on_a_thin_baseline_set(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(baselines=[_baseline()]))
    completed = _run_module("--root", str(tmp_path))
    assert completed.returncode == BaselineSetInvalid.exit_code


def test_the_four_aborts_have_four_distinct_codes() -> None:
    """CI has to tell a malformed pin from a moved world from a baseline it may not score."""
    codes = {
        PinsFileInvalid.exit_code,
        BaselineSetInvalid.exit_code,
        PinMismatch.exit_code,
        BaselineIneligible.exit_code,
    }
    assert len(codes) == 4
    from nbc.platform import UnsupportedPlatform

    assert UnsupportedPlatform.exit_code not in codes


def test_the_module_exits_with_the_declared_code_on_an_ineligible_baseline(
    tmp_path: Path,
) -> None:
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = TRAINED_ON
    write_pins(tmp_path, document)

    completed = _run_module("--root", str(tmp_path))
    assert completed.returncode == BaselineIneligible.exit_code
    assert "example/attacks" in completed.stderr


# --- the world, once, over the network ----------------------------------------------------


@pytest.mark.smoke
def test_every_committed_pin_still_resolves_on_the_hub() -> None:
    """The only test here that touches a network, and it is excluded from the default run."""
    from nbc.pins import resolve_over_http

    pins = load_pins()
    resolved = verify_revisions(pins, resolve_over_http)
    assert len(resolved) == len(pins.remote_artifacts())


# --- Pass 1: what the pin file now refuses ------------------------------------------------------
#
# Every check below ships with the mutation that turns it red. A gate whose failing input nobody
# can name is not a gate, and this epic's review found that shape in production code as often as
# in tests. Each of these was verified to LOAD CLEAN before the corresponding fix.


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../../etc/passwd", "onnx/../../escape", "./onnx/model.onnx"],
)
def test_a_pinned_path_that_escapes_the_snapshot_is_refused(tmp_path: Path, path: str) -> None:
    """`Path("/snapshot") / "/etc/passwd"` is `/etc/passwd`: the left operand is discarded.

    Two modules join these paths, so the check lives where the pin is read rather than at either
    join. `graph_path = "/etc/passwd"` loaded clean before this.
    """
    write_pins(tmp_path, _document(baselines=[_baseline(graph_path=path), _baseline(
        key="second", repository="example/second-model", revision=SHA_B,
        architecture_family="bert", tokenizer_family="wordpiece")]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "graph_path" in str(caught.value)


def test_an_impossible_calendar_date_is_refused(tmp_path: Path) -> None:
    """`2026-13-45` satisfied the old shape check on every field recording when a human looked."""
    write_pins(tmp_path, _document(baselines=[
        _baseline(lineage={"checked_on": "2026-13-45", "card_revision": SHA_A,
                           "attack_datasets": {}, "training_sources": {}}),
        _baseline(key="second", repository="example/second-model", revision=SHA_B,
                  architecture_family="bert", tokenizer_family="wordpiece"),
    ]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "2026-13-45" in str(caught.value)


def test_a_repository_id_with_a_cyrillic_homoglyph_is_refused(tmp_path: Path) -> None:
    """`\\w` is Unicode-aware, so the old pattern admitted an id that reaches an API URL."""
    write_pins(tmp_path, _document(datasets=[_dataset(repository="еxample/attacks")]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "repository" in str(caught.value)


def test_a_repeated_split_is_refused(tmp_path: Path) -> None:
    """A split read twice doubles the pool, and the pool is a published denominator."""
    write_pins(tmp_path, _document(datasets=[_dataset(splits=["train", "train"])]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "repeats" in str(caught.value)


@pytest.mark.parametrize("label", [2, -1, 99])
def test_a_non_binary_attack_label_is_refused(tmp_path: Path, label: int) -> None:
    """A label matching no row yields a recall over an empty pool rather than an abort."""
    write_pins(tmp_path, _document(datasets=[_dataset(attack_label=label)]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "must be 0 or 1" in str(caught.value)


def test_redistributed_material_with_no_licence_and_no_open_question_is_refused(
    tmp_path: Path,
) -> None:
    """FR5.2 is a build abort in the corpus story; refusing here takes the decision first."""
    licence = {
        "identifier": NOT_DECLARED,
        "source": "nothing on the card",
        "attribution": "example/attacks",
        "redistributed": True,
    }
    write_pins(tmp_path, _document(datasets=[_dataset(licence=licence)]))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "undeclared licence" in str(caught.value)


def test_recording_the_open_question_lets_the_file_load_and_reaches_the_run_fields(
    tmp_path: Path,
) -> None:
    """The escape hatch is a record, not a waiver: it has to survive into `results.json`."""
    licence = {
        "identifier": NOT_DECLARED,
        "source": "nothing on the card",
        "attribution": "example/attacks",
        "redistributed": True,
        "unresolved": "2026-08-29: OPEN, publisher declares nothing",
    }
    write_pins(tmp_path, _document(datasets=[_dataset(licence=licence)]))

    pins = load_pins(tmp_path)
    dataset = pins.attack_datasets[0]

    assert dataset.licence.blocks_redistribution is True
    assert "OPEN" in dataset.as_run_fields()["licence"]["unresolved"]


def test_the_family_pair_check_sees_through_casing_and_separators(tmp_path: Path) -> None:
    """`DeBERTa-v2` and `deberta_v2` are one architecture declared twice, and SC5 must say so."""
    write_pins(tmp_path, _document(baselines=[
        _baseline(architecture_family="deberta-v2", tokenizer_family="sentencepiece-unigram"),
        _baseline(key="second", repository="example/second-model", revision=SHA_B,
                  architecture_family="DeBERTa_v2", tokenizer_family="SentencePiece Unigram"),
    ]))

    with pytest.raises(BaselineSetInvalid) as caught:
        load_pins(tmp_path)
    assert "families" in str(caught.value)


def test_the_one_hop_reach_joins_repository_ids_case_insensitively(tmp_path: Path) -> None:
    """Hugging Face resolves ids case-insensitively; a card spelling difference hid the reach."""
    baseline = _baseline(lineage={
        "checked_on": "2026-08-28",
        "card_revision": SHA_A,
        "attack_datasets": {},
        "training_sources": {SEED_ONE.upper(): TRAINED_ON},
    })
    write_pins(tmp_path, _document(baselines=[
        baseline,
        _baseline(key="second", repository="example/second-model", revision=SHA_B,
                  architecture_family="bert", tokenizer_family="wordpiece"),
    ]))

    with pytest.raises((PinsFileInvalid, BaselineSetInvalid)) as caught:
        load_pins(tmp_path)
    assert SEED_ONE.lower() in str(caught.value).lower()


def test_a_utf8_bom_does_not_read_as_broken_toml(tmp_path: Path) -> None:
    """A file saved by a Windows editor reported as 'not valid TOML at line 1 column 1'."""
    write_pins(tmp_path, _document())
    path = tmp_path / "pins.toml"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    assert load_pins(tmp_path).schema_version == SCHEMA_VERSION


# --- Pass 2: the OQ2 record now gates what it claims to gate ------------------------------------
#
# The story that published these numbers advertised, in four places, that a pin move forces a
# re-measurement. It did not: the record named the model and nothing else. Each mutation below
# was verified to LOAD CLEAN before this pass.


def _with_oq2(**overrides: Any) -> list[dict[str, Any]]:
    """The committed two-baseline set with one baseline's OQ2 block overridden."""
    first = _baseline()
    first["oq2"] = {**first["oq2"], **overrides}
    return [
        first,
        _baseline(
            key="second", repository="example/second-model", revision=SHA_B,
            architecture_family="bert", tokenizer_family="wordpiece",
        ),
    ]


def test_an_oq2_record_naming_an_unpinned_dataset_is_refused(tmp_path: Path) -> None:
    """A recall is a function of the model AND the rows; the record gated only the model."""
    write_pins(tmp_path, _document(baselines=_with_oq2(dataset_revision="b" * 40)))

    with pytest.raises(BaselineSetInvalid) as caught:
        load_pins(tmp_path)
    assert "no longer pins" in str(caught.value)


def test_an_oq2_record_measured_at_another_threshold_is_refused(tmp_path: Path) -> None:
    """Threshold is a repinnable parameter, and a recall counts items at or above one."""
    write_pins(tmp_path, _document(baselines=_with_oq2(measured_at_threshold=0.9)))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "measured_at_threshold" in str(caught.value)


def test_a_clean_recall_that_disagrees_with_its_own_integers_is_refused(tmp_path: Path) -> None:
    """The rate is derived from two integers this block records, so a typo is catchable."""
    write_pins(tmp_path, _document(baselines=_with_oq2(hits=90, clean_recall=0.75)))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "disagree" in str(caught.value)


def test_more_hits_than_items_scored_is_refused(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(baselines=_with_oq2(hits=101, clean_recall=1.01)))

    with pytest.raises(PinsFileInvalid) as caught:
        load_pins(tmp_path)
    assert "cannot count more items" in str(caught.value)


def test_two_baselines_measured_over_different_pools_are_refused(tmp_path: Path) -> None:
    """OQ2's whole reading is a comparison between the baselines, and pools must match."""
    write_pins(tmp_path, _document(baselines=_with_oq2(hits=45, sample_size=50, clean_recall=0.9)))

    with pytest.raises(BaselineSetInvalid) as caught:
        load_pins(tmp_path)
    assert "different sample sizes" in str(caught.value)


def test_keeping_a_baseline_without_naming_who_judged_it_is_refused(tmp_path: Path) -> None:
    """`outcome = "kept"` admitted any recall in [0, 1]; the floor is a judgement, so name it."""
    write_pins(tmp_path, _document(baselines=_with_oq2(judged_sufficient_by="")))

    # Refused twice over: the reader rejects the empty string, and the cross-entry check rejects
    # a kept baseline with no named judge. Either abort is the right answer.
    with pytest.raises((PinsFileInvalid, BaselineSetInvalid)) as caught:
        load_pins(tmp_path)
    assert "judged_sufficient_by" in str(caught.value)


def test_the_committed_records_publish_a_ceiling_and_its_floor(committed_pins: Any) -> None:
    """FR3.3's removal has not run, so the measured recall is an upper bound. Both are published.

    The floor assumes every overlapping row was a hit by memory rather than detection, which is
    the worst case the disclosed overlap allows. A reader who is handed only the ceiling has to
    derive this themselves from caveat 3d, and discovering unaided what the artifact did not
    report is the worst outcome available to a section about honesty.
    """
    by_key = {baseline.key: baseline.oq2 for baseline in committed_pins.baselines}

    protectai = by_key["protectai-deberta-v3"]
    assert protectai.overlap_rows == 515
    assert round(protectai.ceiling, 4) == 0.836
    assert round(protectai.floor, 4) == 0.8030
    assert protectai.floor < protectai.ceiling

    # Zero overlap is a measurement here, not a missing declaration: this baseline declares no
    # training on the pinned dataset nor on any seed its card names.
    testsavantai = by_key["testsavantai-bert-small"]
    assert testsavantai.overlap_rows == 0
    assert testsavantai.floor == testsavantai.ceiling


def test_the_committed_recalls_are_recomputable_from_their_own_integers(
    committed_pins: Any,
) -> None:
    """Nothing cross-checked the published numbers against anything. Now the loader does."""
    for baseline in committed_pins.baselines:
        oq2 = baseline.oq2
        places = len(str(oq2.clean_recall).partition(".")[2])
        assert round(oq2.hits / oq2.sample_size, places) == round(oq2.clean_recall, places)
        assert oq2.dataset_revision in {
            dataset.revision for dataset in committed_pins.attack_datasets
        }
        assert oq2.measured_at_threshold == baseline.threshold


# --- Pass 3: the pin is compared to the artifact, and the comparison can fail -------------------
#
# The #1 blocking finding: `resolve_from_cache` returned the pin, and the pin was what it was
# compared against, so on every machine that had fetched once `resolved != artifact.revision` was
# `x != x`. The module's docstring called that comparison the guarantee that the published
# numbers came from the pinned artifacts.


def test_a_cache_resolution_says_it_was_not_checked_against_the_world(tmp_path: Path) -> None:
    """The directory's existence is the pin spelled back off a directory name."""
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    snapshot = tmp_path / artifact.cache_directory / "snapshots" / SHA_A
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    resolution = resolve_from_cache(artifact, cache_root=tmp_path)

    assert resolution is not None
    assert resolution.checked_against == CHECKED_AGAINST_CACHE
    assert resolution.checked_against != CHECKED_AGAINST_HUB


def test_an_empty_snapshot_directory_does_not_resolve(tmp_path: Path) -> None:
    """An interrupted fetch leaves one behind, and it satisfied verification before this."""
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    (tmp_path / artifact.cache_directory / "snapshots" / SHA_A).mkdir(parents=True)

    assert resolve_from_cache(artifact, cache_root=tmp_path) is None


def test_the_run_fields_record_which_artifacts_were_asked_about(tmp_path: Path) -> None:
    """A run verified entirely from cache compared nothing to the world, and must say so."""
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)

    def cache_only(artifact: RemoteArtifact) -> Resolution | None:
        return Resolution(artifact.revision, CHECKED_AGAINST_CACHE)

    resolved = verify_revisions(pins, cache_only)

    assert all(value.endswith(f"@{CHECKED_AGAINST_CACHE}") for value in resolved.values())


def test_a_graph_whose_size_moved_aborts(tmp_path: Path, monkeypatch: Any) -> None:
    """`graph_bytes` was the declared evidence for `precision` and was read by nothing.

    An fp16 export is a fraction of its fp32 original, so the size is what catches a swapped
    graph that kept its filename -- the one substitution a revision check cannot see, because
    the revision still resolves.
    """
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)
    baseline = pins.baselines[0]

    cache = tmp_path / "hub"
    graph = cache / baseline.artifact.cache_directory / "snapshots" / SHA_A / baseline.graph_path
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"x" * (baseline.graph_bytes + 1))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    with pytest.raises(PinMismatch) as caught:
        verify_revisions(pins, echoing_resolver)
    assert "bytes and the file on this machine is" in str(caught.value)


def test_a_graph_whose_size_matches_verifies(tmp_path: Path, monkeypatch: Any) -> None:
    """The other half: the check has to pass on a file that is what the pin says it is."""
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)
    baseline = pins.baselines[0]

    cache = tmp_path / "hub"
    graph = cache / baseline.artifact.cache_directory / "snapshots" / SHA_A / baseline.graph_path
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"x" * baseline.graph_bytes)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    assert verify_revisions(pins, echoing_resolver)
