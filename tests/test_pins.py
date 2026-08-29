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
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator

import pytest

import nbc
from nbc import pins as pins_module
from nbc.corpus.heldout import HELD_OUT_FROM
from nbc.pins import (
    DRAW_HEAD,
    DRAW_METHODS,
    DRAW_SEEDED_RANDOM,
    DRAW_SORT_KEYS,
    EXCLUSION_AVAILABLE,
    EXCLUSION_UNREACHABLE,
    EXCLUSION_UNREADABLE,
    HTTP_OK,
    LINEAGE_RELATIONSHIPS,
    MINIMUM_BASELINES,
    NOT_DECLARED,
    BENIGN_CHAT_CLASS,
    BENIGN_CODE_CLASS,
    BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
    FRAME_ID_FIELD,
    MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY,
    MINIMUM_BENIGN_CODE_REPOSITORIES,
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
    frame_digest,
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
SHA_E = "e" * 40

# The pinned dataset's card names four seeds and one baseline declares training on two of them.
# The fixtures keep that shape -- two seeds, neither declared by default -- so a test that wants
# the one-hop reach has to say so, and a test that does not is not silently sitting on one.
SEED_ONE = "example/seed-one"
SEED_TWO = "example/seed-two"

SHA_F = "f" * 40


def _benign_code_repository(index: int) -> dict[str, Any]:
    return {
        "key": f"example-code-{index}",
        "repository": f"example/code-{index}",
        "revision": SHA_F,
        "licence": {
            "identifier": "MIT",
            "source": "fixture",
            "attribution": "fixture",
            "redistributed": True,
        },
    }


def _benign_frame(**overrides: Any) -> dict[str, Any]:
    """A loadable `[benign_frame]`, with `frame_id` computed from the block it ends up being.

    The digest is computed here rather than written as a literal, because a fixture carrying a
    frozen digest would have to be re-hashed by hand every time a test overrode a field -- and the
    tests that want the refusal to fire supply a WRONG id on purpose, which is the only way the two
    sides of that comparison come from different places.
    """
    frame: dict[str, Any] = {
        "declared_on": "2026-08-29",
        "sample_size_items": 4,
        "method": DRAW_SEEDED_RANDOM,
        "seed": 11,
        "b_code": {
            "min_repositories": MINIMUM_BENIGN_CODE_REPOSITORIES,
            "max_files_per_repository": MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY,
            "eligibility": BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
            "min_file_bytes": 200,
            "max_file_bytes": 4000,
            "file_extensions": [".py"],
            "repository": [
                _benign_code_repository(index)
                for index in range(MINIMUM_BENIGN_CODE_REPOSITORIES)
            ],
        },
        "b_chat": {"hand_authored_items": 1},
        "confirmatory_cell": {
            "declared_on": "2026-08-29",
            "baseline": "first",
            "dressing_chain": "base64+base64+base64+base64",
            "benign_class": "b_code",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(frame.get(key), dict):
            frame[key] = {**frame[key], **value}
        else:
            frame[key] = value
    frame[FRAME_ID_FIELD] = frame_digest(frame)
    return frame


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
        # The draw the fixture declares is deliberately tiny and deliberately not the committed
        # one: every value here is compared against `pins.toml`'s by the no-literals scans, and a
        # fixture that copied the real draw would be the second home those scans exist to forbid.
        "draw": {
            "declared_on": "2026-08-29",
            "sample_size_positives": 3,
            "method": DRAW_SEEDED_RANDOM,
            "seed": 7,
        },
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


def _exclusion_source(repository: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": repository.replace("/", "-"),
        "repository": repository,
        "revision": SHA_E,
        "availability": EXCLUSION_AVAILABLE,
        "http_status": HTTP_OK,
        "checked_on": "2026-08-29",
        "evidence": "fixture",
        # Every pinned source declares a licence (AD-34), an exclusion source included. Nothing of
        # one is redistributed -- its matches are removed from the corpus, never published -- so
        # the licence gate reads this and moves on. A fixture that omitted it would not load.
        "licence": {
            "identifier": "apache-2.0",
            "source": "fixture",
            "attribution": f"{repository}, apache-2.0, fixture",
            "redistributed": False,
        },
    }
    entry.update(overrides)
    return entry


def _derived_exclusions(
    baselines: list[dict[str, Any]], datasets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The array `load_pins` requires, computed from the fixture's own two lineage blocks.

    The fixture builds a *loadable* file, so it has to satisfy the derivation gate the same way
    `pins.toml` does. Every test that wants the gate to fire overrides `exclusion_source`
    explicitly -- which is the only way the two sides of that comparison come from different
    places.
    """
    names: list[str] = []
    for entry in baselines:
        for source, relationship in entry.get("lineage", {}).get(
            "training_sources", {}
        ).items():
            if relationship == TRAINED_ON and source not in names:
                names.append(source)
    for entry in datasets:
        for seed in entry.get("provenance", {}).get("seeds", []):
            if seed not in names:
                names.append(seed)
    return [_exclusion_source(name) for name in names]


def _document(
    baselines: list[dict[str, Any]] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    exclusion_sources: list[dict[str, Any]] | None = None,
    benign_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_baselines = (
        baselines
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
        ]
    )
    resolved_datasets = datasets if datasets is not None else [_dataset()]
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "verified_on": "2026-08-28",
            "verified_against": "the live artifacts",
        },
        "baseline": resolved_baselines,
        "attack_dataset": resolved_datasets,
        "exclusion_source": exclusion_sources
        if exclusion_sources is not None
        else _derived_exclusions(resolved_baselines, resolved_datasets),
        "benign_frame": benign_frame if benign_frame is not None else _benign_frame(),
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
    nested: list[tuple[str, Any]] = []
    for key, value in table.items():
        if isinstance(value, dict) or _is_table_array(value):
            # A nested array of TABLES is `[[parent.key]]`, one header per entry; a list of
            # scalars stays inline. The benign frame carries both -- its repository array and its
            # `file_extensions` -- so the writer has to tell them apart rather than guess by key.
            nested.append((key, value))
        else:
            lines.append(f"{_key(key)} = {_dump(value)}")
    for key, value in nested:
        if isinstance(value, dict):
            _write_table(lines, f"{header}.{key}", value, array=False)
            continue
        for entry in value:
            _write_table(lines, f"{header}.{key}", entry, array=True)


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


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
    for entry in document.get("exclusion_source", []):
        _write_table(lines, "exclusion_source", entry, array=True)
    if "benign_frame" in document:
        _write_table(lines, "benign_frame", document["benign_frame"], array=False)
    path = root / PINS_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def committed_pins(repo_root: Path) -> Any:
    """Resolved the same way `committed_document` resolves it.

    This called `load_pins()` with no argument, which walks up from the working directory, while
    its sibling fixture read `repo_root / PINS_FILENAME`. Two ways to find one file is two files
    a test can be reading, and every assertion pairing the two silently assumed they agreed.
    """
    return load_pins(repo_root)


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
    # Admitted, but not for free: a declared training source is an exclusion source whether or
    # not any dataset names it as a seed, so the array has to grow with it or the file is
    # refused. That is the obligation this record buys.
    document["exclusion_source"].append(_exclusion_source("example/some-other-corpus"))
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert "example/some-other-corpus" in pins.derived_exclusion_sources()
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

    # Pinned from the committed document, not re-derived. The assertion here used to be
    # `set(seeds) <= set(required_exclusion_sources())`, which is a tautology: the latter is
    # DEFINED as the union of the former, so it held whatever the reach turned out to be, and
    # named neither the baseline, nor the dataset, nor a single seed.
    document = tomllib.loads(
        (Path(nbc.__file__).resolve().parents[2] / PINS_FILENAME).read_text(encoding="utf-8")
    )
    declared_seeds = {
        seed
        for entry in document["attack_dataset"]
        for seed in entry["provenance"]["seeds"]
    }
    trained_on = {
        source
        for entry in document["baseline"]
        for source, relationship in entry["lineage"]["training_sources"].items()
        if relationship == TRAINED_ON
    }
    expected = declared_seeds & trained_on

    assert set(seeds) == expected, "the reach is not the one the committed file declares"
    assert expected, "a reach test that passes on an empty intersection tests nothing"
    assert set(committed_pins.required_exclusion_sources()) == expected

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
        # The exclusion sources are verified by the same gate as everything else: their
        # revisions decide which rows survive into the corpus, so a pin that stopped resolving
        # has to be as visible as a model that moved.
        SEED_ONE: f"{SHA_E}@{CHECKED_AGAINST_HUB}",
        SEED_TWO: f"{SHA_E}@{CHECKED_AGAINST_HUB}",
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
    # One problem per artifact the pins name, read from the pins rather than transcribed: a
    # count written here as a literal would stop tracking the set it claims to describe the
    # first time the file grew a section, which is exactly what it did.
    assert len(abort.value.problems) == len(pins.remote_artifacts())
    assert len(pins.remote_artifacts()) > len(pins.baselines)
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
    """Every Python file in the repository that could hold a pin, not only the package.

    Rooted at `src/nbc/` before, so `spikes/` and `tests/` could hold a repository id, a sha or a
    sample size freely -- and the OQ2 spike is exactly the kind of file that would, since it is
    the one that reads the pinned dataset. The check is about the pin file being the only place
    an artifact is named, and that claim is not scoped to one directory.
    """
    root = Path(nbc.__file__).resolve().parents[2]
    return sorted(
        path
        for directory in ("src", "spikes")
        for path in (root / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _test_files() -> list[Path]:
    root = Path(nbc.__file__).resolve().parents[2]
    return sorted(
        path for path in (root / "tests").rglob("*.py") if "__pycache__" not in path.parts
    )


def _unmistakable_pins(document: dict[str, Any]) -> set[str]:
    """The identifiers a test could never need by coincidence.

    The broad scan above cannot run over `tests/`: fixtures legitimately name `config.json` and
    `tokenizer.json`, which are also pinned paths, and every one of those would be noise. But a
    40-hex sha, a `namespace/name` repository id and a sample size are never coincidences -- a
    test carrying one is a second home for a pin, which is the whole thing this file forbids.
    """
    return {
        value
        for value in _pinned_identifiers(document)
        if (
            re.fullmatch(r"[0-9a-f]{40}", value)
            or re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", value)
            and "." not in value.rsplit("/", 1)[-1]
            or value.isdigit()
        )
    }


def _literal_constants(path: Path) -> Iterator[tuple[int, str]]:
    """String AND numeric literals.

    Numbers were collected from `pins.toml` and then compared against strings only, so the
    acceptance criterion "no sample size appears as a literal" was asserted by a scan that could
    not fire on a sample size. A sample size is an integer.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                yield node.lineno, node.value
            elif isinstance(node.value, int) and not isinstance(node.value, bool):
                yield node.lineno, str(node.value)


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
            elif key == FRAME_ID_FIELD:
                # The frame's digest is a pin like any other: a copy of it under `src/` would be a
                # second place the frame's identity lives, and the one thing `data/manifest.json`
                # is compared against must not have a twin in the code that does the comparing.
                identifiers.add(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            # Sizes and seeds both. A seed is a draw parameter exactly as a sample size is, and a
            # second copy of one reproduces a different sample while looking like the declared
            # draw -- which is the same defect, one field over.
            if "sample_size" in key or key == "seed":
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
        for lineno, value in _literal_constants(path):
            if value in identifiers:
                offenders.append(f"{path.name}:{lineno} {value!r}")

    assert not offenders, (
        f"these belong only in {PINS_FILENAME}: " + "; ".join(offenders)
    )


OWN_REVISIONS = frozenset({HELD_OUT_FROM.revision})
"""The commit shas that name **this** repository rather than a remote artifact.

The gate below is about a *pin*: a revision of somebody else's repository that was written into the
source instead of into `pins.toml`, where it would be verified against the hub before anything ran.
`heldout.HELD_OUT_FROM.revision` is not one of those. It is this repository's own commit, recorded
because AD-28 makes the held-out set a one-way door and requires the layer revision it was held out
from to travel with it, and the architecture's configuration table puts an AD-28 constant in the
module that owns the algorithm rather than in `pins.toml`. There is no hub to verify it against and
nothing to resolve it to; `pins.toml` could not check it.

So the allowance is granted to **one declared constant and no other value**, which is what keeps it
from becoming a hole: a second sha in `heldout.py`, or this sha copied into a second module, is
still an offender, because the membership test is equality with the constant rather than a filename
or a pattern. That the sha names a commit in this repository is checked separately, by
`tests/corpus/test_heldout.py`, against git.
"""


def test_no_commit_sha_appears_as_a_literal_in_the_source_tree() -> None:
    """Catches a revision that was hard-coded without ever being pinned."""
    offenders: list[str] = []
    for path in _source_files():
        for lineno, value in _literal_constants(path):
            if len(value) == 40 and all(char in "0123456789abcdef" for char in value):
                if value in OWN_REVISIONS:
                    continue
                offenders.append(f"{path.name}:{lineno} {value!r}")

    assert not offenders, "a commit sha is a pin: " + "; ".join(offenders)


def test_the_own_revision_allowance_still_catches_a_second_sha(tmp_path: Path) -> None:
    """The allowance's own failing input: a different sha, in the same file, is still an offender.

    Written against the scan rather than against the assertion above, because the assertion runs
    over the real tree and would pass vacuously the day the allowance swallowed everything.
    """
    module = tmp_path / "probe.py"
    module.write_text(
        f'A = "{HELD_OUT_FROM.revision}"\nB = "{"f" * 40}"\n', encoding="utf-8"
    )
    found = [
        value
        for _lineno, value in _literal_constants(module)
        if len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
        and value not in OWN_REVISIONS
    ]
    assert found == ["f" * 40]


def test_only_one_module_names_the_pins_file() -> None:
    """`pins.py` is the only reader, so a second module reading it is a second parser."""
    readers = [
        path.name
        for path in _source_files()
        if any(PINS_FILENAME == value for _, value in _literal_constants(path))
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


def test_no_sha_repository_id_or_sample_size_is_hardcoded_in_a_test(
    committed_document: dict[str, Any],
) -> None:
    """The broad scan skips `tests/`, so the unmistakable identifiers are checked here instead.

    The acceptance criterion says no sample size appears as a literal. The scan that asserted it
    collected numbers from pins.toml and then compared them against string constants only, so it
    could not fire on a sample size -- a sample size being an integer. Widening it to numeric
    literals found one immediately, in a test.
    """
    pinned = _unmistakable_pins(committed_document)
    offenders = [
        f"{path.name}:{line} {value!r}"
        for path in _test_files()
        for line, value in _literal_constants(path)
        if value in pinned
    ]

    assert not offenders, (
        "a pinned identifier is hardcoded in a test, so pins.toml is no longer the only place "
        f"it lives: {'; '.join(offenders)}"
    )


ACCESS_RESTRICTED_SOURCES = 1
"""How many declared exclusion sources the hub does not answer for at all, today.

Reviewed rather than derived: if this changes, the README's caveat about what the filter cannot
reach changes with it, and a test that read the number out of the file would let both move
together in silence.
"""

DECLARED_HOP_SEEDS = 2
"""How many seeds the one declared `seeded-from-declared-training-source` runs through, today.

Same reason. This is the size of the obligation the exemption bought, and the corpus build may
not proceed without discharging every one of them.
"""


# --- the exclusion set is derived, not chosen -------------------------------------------------
#
# The obligation `seeded-from-declared-training-source` buys is discharged by the corpus build,
# and until this section existed nothing forced anyone to discharge it: the exemption was granted
# in this file and `required_exclusion_sources()` was read by nothing outside it. The gate below
# is the other half -- the array of sources the build downloads cannot drift away from the two
# declaration blocks that create the obligation.


def test_the_committed_file_pins_every_source_its_declarations_derive(
    committed_pins: Any,
) -> None:
    declared = {source.repository for source in committed_pins.exclusion_sources}

    assert declared == set(committed_pins.derived_exclusion_sources())
    assert set(committed_pins.required_exclusion_sources()) <= declared


def test_the_committed_file_names_one_access_restricted_source_declared_by_both(
    committed_pins: Any,
) -> None:
    """Caveat 3d publishes an HTTP 401 on a source **both** baselines declare. Checked here.

    The repository id is deliberately not written out -- a test carrying one is a second home for
    a pin. The claim is structural instead, and it is the claim the README actually makes: there
    is exactly one unreadable source, it answers 401, it pins no revision because a 401 hands
    back none, and every pinned baseline declares training on it. That last part is what makes it
    unmeasurable for any corpus this experiment could have chosen rather than a limitation of one
    pin, and it is the half a reader has no way to check for themselves.
    """
    unreachable = [
        source
        for source in committed_pins.exclusion_sources
        if source.availability == EXCLUSION_UNREACHABLE
    ]

    assert len(unreachable) == ACCESS_RESTRICTED_SOURCES
    source = unreachable[0]
    assert source.http_status == 401
    assert source.revision == ""
    assert all(
        baseline.lineage.trains_on(source.repository)
        for baseline in committed_pins.baselines
    )
    assert len(committed_pins.baselines) >= MINIMUM_BASELINES


def test_the_required_exclusion_set_is_exactly_what_a_declared_hop_bought(
    committed_pins: Any,
) -> None:
    """Decision D-C's obligation, sized and characterized rather than transcribed.

    Every required source must be both a seed on a pinned dataset's card and a training source
    the baseline declares -- that conjunction is what "one hop" means. The count is asserted so
    that a lineage edit changing which sources the build must remove against makes a human look,
    which is the whole reason the number is reviewed rather than derived.
    """
    required = set(committed_pins.required_exclusion_sources())
    seeds = {
        seed
        for dataset in committed_pins.attack_datasets
        for seed in dataset.provenance.seeds
    }

    assert len(required) == DECLARED_HOP_SEEDS
    assert required < seeds, "a required source that is nobody's seed is not a one-hop reach"
    for name in required:
        assert any(
            baseline.lineage.trains_on(name) for baseline in committed_pins.baselines
        ), name
    assert required <= set(committed_pins.derived_exclusion_sources())


def test_a_derived_source_nobody_pinned_is_refused(tmp_path: Path) -> None:
    """The gate's failing input: drop one entry and the file stops loading."""
    document = _document()
    dropped = document["exclusion_source"].pop()
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert dropped["repository"] in str(abort.value)
    assert "no [[exclusion_source]] pins" in str(abort.value)


def test_an_exclusion_source_nothing_derives_is_refused(tmp_path: Path) -> None:
    """The other direction: a source pinned for a reason this file no longer states."""
    document = _document()
    document["exclusion_source"].append(_exclusion_source("example/nobody-declares-this"))
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "example/nobody-declares-this" in str(abort.value)


def test_a_new_seed_cannot_drift_away_from_the_exclusion_array(tmp_path: Path) -> None:
    """Decision D-C, as a mechanism: growing the reach without pinning it fails to load.

    This is the shape the defect actually takes. Somebody re-reads a dataset card, adds the seed
    it names, records the baseline's training on it -- and the build silently never downloads it,
    because the exclusion set was a list somebody maintained by hand.
    """
    document = _document()
    third = "example/seed-three"
    document["attack_dataset"][0]["provenance"]["seeds"].append(third)
    document["baseline"][0]["lineage"]["training_sources"][third] = TRAINED_ON
    document["baseline"][1]["lineage"]["training_sources"][third] = NOT_DECLARED
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = (
        SEEDED_FROM_TRAINING_SOURCE
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)
    assert third in str(abort.value)

    # And with the entry added, the same file loads and the new seed is required.
    document["exclusion_source"].append(_exclusion_source(third))
    write_pins(tmp_path, document)
    pins = load_pins(tmp_path)

    assert third in pins.required_exclusion_sources()


def test_the_two_blocks_are_joined_on_the_canonical_id_not_the_spelling(tmp_path: Path) -> None:
    """The hub resolves ids case-insensitively and the blocks are written from two cards."""
    document = _document()
    document["exclusion_source"][0]["repository"] = SEED_ONE.upper()
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)

    assert SEED_ONE.upper() in {source.repository for source in pins.exclusion_sources}


def test_an_available_source_without_a_revision_is_refused(tmp_path: Path) -> None:
    document = _document()
    del document["exclusion_source"][0]["revision"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "revision is missing" in str(abort.value)


def test_an_unreachable_source_carrying_a_revision_is_refused(tmp_path: Path) -> None:
    """A sha for a source the hub does not answer for came from somewhere this file cannot name."""
    document = _document()
    document["exclusion_source"][0].update(
        {"availability": EXCLUSION_UNREACHABLE, "http_status": 401}
    )
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "hands back no commit" in str(abort.value)


def test_an_available_source_declaring_a_failure_status_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["exclusion_source"][0]["http_status"] = 404
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "they differ in whether its rows load" in str(abort.value)


def test_an_unreachable_source_declaring_success_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["exclusion_source"][0].update(
        {"availability": EXCLUSION_UNREACHABLE, "http_status": HTTP_OK}
    )
    del document["exclusion_source"][0]["revision"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "is the status of a source the hub answers for" in str(abort.value)


def test_an_unreadable_source_keeps_its_revision_and_loads(tmp_path: Path) -> None:
    """The second way a source can be out of reach: it resolves and its rows will not load.

    Not a shade of unreachable. The sha is real, `verify_revisions` checks it like every other
    pin, and only `loadable` is false -- which is what tells the build to expect the refusal
    rather than to crash on it.
    """
    document = _document()
    document["exclusion_source"][0]["availability"] = EXCLUSION_UNREADABLE
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    source = pins.exclusion_sources[0]

    assert source.revision == SHA_E
    assert source.resolvable is True
    assert source.loadable is False
    assert source.artifact in pins.remote_artifacts()


def test_an_unreadable_source_without_a_revision_is_refused(tmp_path: Path) -> None:
    """It resolves, so it is pinned by its commit like everything else that resolves."""
    document = _document()
    document["exclusion_source"][0]["availability"] = EXCLUSION_UNREADABLE
    del document["exclusion_source"][0]["revision"]
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "revision is missing" in str(abort.value)


def test_an_availability_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    """Closed, because the build reads this value and compares it against the hub's answer."""
    document = _document()
    document["exclusion_source"][0]["availability"] = "probably-fine"
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "probably-fine" in str(abort.value)


def test_a_status_outside_the_http_range_is_refused(tmp_path: Path) -> None:
    document = _document()
    document["exclusion_source"][0]["http_status"] = 7
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "must be an HTTP status code" in str(abort.value)


def test_two_exclusion_sources_sharing_a_repository_are_refused(tmp_path: Path) -> None:
    """`verify_revisions` reports by repository id, where two entries collapse into one line."""
    document = _document()
    duplicate = dict(document["exclusion_source"][0])
    duplicate["key"] = "a-second-key"
    document["exclusion_source"].append(duplicate)
    write_pins(tmp_path, document)

    with pytest.raises(PinsFileInvalid) as abort:
        load_pins(tmp_path)

    assert "share repository" in str(abort.value)


def test_the_run_fields_publish_both_exclusion_sets(committed_pins: Any) -> None:
    """A reader recomputes the derived set from the two blocks; the array is what was pinned."""
    fields = committed_pins.as_run_fields()["pins"]

    assert [entry["repository"] for entry in fields["exclusion_sources"]] == [
        source.repository for source in committed_pins.exclusion_sources
    ]
    assert fields["derived_exclusion_sources"] == list(
        committed_pins.derived_exclusion_sources()
    )
    assert set(fields["required_exclusion_sources"]) <= set(
        fields["derived_exclusion_sources"]
    )


# --- the attack draw ---------------------------------------------------------------------


def _with_draw(**overrides: Any) -> list[dict[str, Any]]:
    """The fixture dataset with its `draw` sub-table amended.

    `None` removes a key, which is how the "declares no seed" and "declares no sort_key" cases
    are expressed without a second fixture builder.
    """
    dataset = _dataset()
    draw = dict(dataset["draw"])
    for key, value in overrides.items():
        if value is None:
            draw.pop(key, None)
        else:
            draw[key] = value
    dataset["draw"] = draw
    return [dataset]


def test_the_committed_draw_is_declared_in_positives_with_a_known_method(
    repo_root: Path,
) -> None:
    """The AC's list, read off the file rather than off a comment about it.

    No value is asserted here: `pins.toml` is the only home for the size and the seed, and this
    file may not hold a second copy of either. What is asserted is that each field is present, is
    the right kind of thing, and that the method's own companion field is the one that is set.
    """
    pins = load_pins(repo_root)
    assert pins.attack_datasets, "the scan found no attack dataset, which would pass vacuously"
    for dataset in pins.attack_datasets:
        draw = dataset.draw
        assert draw.sample_size_positives > 0
        assert draw.method in DRAW_METHODS
        assert draw.declared_on
        # Exactly one companion, decided by the method. Both set, or neither, is the state the
        # reader refuses.
        assert (draw.seed is None) != (draw.sort_key is None)
        if draw.method == DRAW_SEEDED_RANDOM:
            assert isinstance(draw.seed, int) and draw.sort_key is None
        else:
            assert draw.sort_key in DRAW_SORT_KEYS and draw.seed is None
        # Counts are over every split, so more than one must be declared for this pool.
        assert len(dataset.splits) > 1


def test_a_draw_declaring_an_unknown_method_is_refused(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(datasets=_with_draw(method="first_n", seed=None)))
    with pytest.raises(PinsFileInvalid, match="method"):
        load_pins(tmp_path)


def test_a_head_draw_with_no_sort_key_is_refused(tmp_path: Path) -> None:
    """The first N of an undeclared order is whatever the reader yielded first."""
    write_pins(tmp_path, _document(datasets=_with_draw(method=DRAW_HEAD, seed=None)))
    with pytest.raises(PinsFileInvalid, match="sort_key"):
        load_pins(tmp_path)


def test_a_head_draw_carrying_a_seed_is_refused(tmp_path: Path) -> None:
    """Refused rather than ignored: an ignored seed reads as a declared draw and is not one."""
    write_pins(
        tmp_path, _document(datasets=_with_draw(method=DRAW_HEAD, sort_key="text"))
    )
    with pytest.raises(PinsFileInvalid, match="seed"):
        load_pins(tmp_path)


def test_a_head_draw_with_an_unknown_sort_key_is_refused(tmp_path: Path) -> None:
    write_pins(
        tmp_path,
        _document(datasets=_with_draw(method=DRAW_HEAD, seed=None, sort_key="row_order")),
    )
    with pytest.raises(PinsFileInvalid, match="sort_key"):
        load_pins(tmp_path)


def test_a_seeded_draw_with_no_seed_is_refused(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(datasets=_with_draw(seed=None)))
    with pytest.raises(PinsFileInvalid, match="seed"):
        load_pins(tmp_path)


def test_a_seeded_draw_carrying_a_sort_key_is_refused(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(datasets=_with_draw(sort_key="text")))
    with pytest.raises(PinsFileInvalid, match="sort_key"):
        load_pins(tmp_path)


@pytest.mark.parametrize("size", [0, -1])
def test_a_draw_of_no_positives_is_refused(tmp_path: Path, size: int) -> None:
    """A draw of none publishes a rate over nothing."""
    write_pins(tmp_path, _document(datasets=_with_draw(sample_size_positives=size)))
    with pytest.raises(PinsFileInvalid, match="sample_size_positives"):
        load_pins(tmp_path)


def test_a_draw_dated_off_the_calendar_is_refused(tmp_path: Path) -> None:
    write_pins(tmp_path, _document(datasets=_with_draw(declared_on="2026-13-45")))
    with pytest.raises(PinsFileInvalid, match="declared_on"):
        load_pins(tmp_path)


def test_a_dataset_with_no_draw_at_all_is_refused(tmp_path: Path) -> None:
    """Absent rather than defaulted: a default draw is a draw nobody declared."""
    dataset = _dataset()
    del dataset["draw"]
    write_pins(tmp_path, _document(datasets=[dataset]))
    with pytest.raises(PinsFileInvalid, match="draw"):
        load_pins(tmp_path)


def test_the_draw_reaches_the_run_fields(repo_root: Path) -> None:
    """A declaration nothing publishes is a declaration a reader of the results cannot check."""
    fields = load_pins(repo_root).as_run_fields()["pins"]
    assert fields["attack_datasets"]
    for dataset in fields["attack_datasets"]:
        assert set(dataset["draw"]) == {
            "declared_on",
            "sample_size_positives",
            "method",
            "seed",
            "sort_key",
        }


# --- the benign frame (story 3.6) ----------------------------------------------------------


def test_a_frame_edited_without_its_digest_is_refused(tmp_path: Path) -> None:
    """The frame's whole content: it cannot be changed while the id that fixed it stays.

    The failing input is the realistic one -- a field edited and the digest left alone, which is
    exactly what growing a benign corpus until the number looked reasonable would look like.
    """
    frame = _benign_frame()
    frame["seed"] = frame["seed"] + 1
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any(FRAME_ID_FIELD in problem for problem in raised.value.problems)


def test_a_frame_re_declared_with_its_new_digest_loads(tmp_path: Path) -> None:
    """The other direction: re-declaring the frame deliberately is allowed and is a visible diff."""
    write_pins(tmp_path, _document(benign_frame=_benign_frame(seed=4242)))
    assert load_pins(tmp_path).benign_frame.seed == 4242


def test_the_committed_frame_id_is_the_digest_of_the_committed_block(
    committed_document: dict[str, Any],
) -> None:
    """CI's assertion, written here too: the file on disk hashes to the id it carries."""
    block = committed_document["benign_frame"]
    assert block[FRAME_ID_FIELD] == frame_digest(block)


def test_the_committed_frame_reaches_the_declared_repository_floor(
    committed_pins: Any,
) -> None:
    frame = committed_pins.benign_frame
    assert frame.b_code.min_repositories >= MINIMUM_BENIGN_CODE_REPOSITORIES
    assert len(frame.b_code.repositories) >= frame.b_code.min_repositories
    assert frame.b_code.max_files_per_repository <= MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY


def test_the_frame_s_class_names_are_the_schema_s(committed_document: dict[str, Any]) -> None:
    """`pins.py` may not import `nbc.schema`, so the duplication is checked rather than avoided."""
    from nbc.schema import BENIGN_CLASSES

    assert (BENIGN_CODE_CLASS, BENIGN_CHAT_CLASS) == BENIGN_CLASSES
    assert set(BENIGN_CLASSES) <= set(committed_document["benign_frame"])


def test_a_frame_below_the_requirement_floor_is_refused(tmp_path: Path) -> None:
    frame = _benign_frame(
        b_code={"min_repositories": MINIMUM_BENIGN_CODE_REPOSITORIES - 1}
    )
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("design effect" in problem for problem in raised.value.problems)


def test_a_frame_above_the_requirement_cap_is_refused(tmp_path: Path) -> None:
    frame = _benign_frame(
        b_code={"max_files_per_repository": MAXIMUM_FILES_PER_BENIGN_CODE_REPOSITORY + 1}
    )
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("cap" in problem for problem in raised.value.problems)


def test_a_frame_pinning_fewer_repositories_than_its_own_floor_is_refused(
    tmp_path: Path,
) -> None:
    frame = _benign_frame(
        b_code={"repository": [_benign_code_repository(index) for index in range(3)]}
    )
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("could only reach it" in problem for problem in raised.value.problems)


def test_an_eligibility_rule_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    frame = _benign_frame(b_code={"eligibility": "looks_like_base64"})
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("looks_like_base64" in problem for problem in raised.value.problems)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"file_extensions": []}, "vendored blobs"),
        ({"file_extensions": ["py"]}, "beginning with a dot"),
        ({"file_extensions": [".PY"]}, "beginning with a dot"),
        ({"file_extensions": [".py", ".py"]}, "repeats an entry"),
        ({"min_file_bytes": 0}, "not a corpus row"),
        ({"min_file_bytes": 900, "max_file_bytes": 400}, "band is empty"),
    ],
)
def test_a_malformed_b_code_frame_is_refused(
    tmp_path: Path, override: dict[str, Any], expected: str
) -> None:
    write_pins(tmp_path, _document(benign_frame=_benign_frame(b_code=override)))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any(expected in problem for problem in raised.value.problems), raised.value.problems


def test_a_hand_authored_allowance_that_swallows_the_class_is_refused(tmp_path: Path) -> None:
    """A class made entirely of hand-authored material is a corpus this repository wrote itself."""
    frame = _benign_frame(b_chat={"hand_authored_items": 4})
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("wrote for itself" in problem for problem in raised.value.problems)


def test_a_confirmatory_cell_naming_no_declared_baseline_is_refused(tmp_path: Path) -> None:
    frame = _benign_frame(confirmatory_cell={"baseline": "a-model-nobody-pinned"})
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("a-model-nobody-pinned" in problem for problem in raised.value.problems)


def test_a_benign_repository_that_is_also_a_pinned_source_is_refused(tmp_path: Path) -> None:
    """B-code is what the false-positive rate is measured over; it may not be exclusion material."""
    entry = _benign_code_repository(0)
    entry["repository"] = SEED_ONE
    frame = _benign_frame(
        b_code={
            "repository": [entry]
            + [
                _benign_code_repository(index)
                for index in range(1, MINIMUM_BENIGN_CODE_REPOSITORIES)
            ]
        }
    )
    write_pins(tmp_path, _document(benign_frame=frame))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("measures the filter rather than the layer" in p for p in raised.value.problems)


def test_two_benign_repositories_sharing_a_key_are_refused(tmp_path: Path) -> None:
    entries = [
        _benign_code_repository(index)
        for index in range(MINIMUM_BENIGN_CODE_REPOSITORIES)
    ]
    entries[1]["key"] = entries[0]["key"]
    write_pins(tmp_path, _document(benign_frame=_benign_frame(b_code={"repository": entries})))
    with pytest.raises(PinsFileInvalid) as raised:
        load_pins(tmp_path)
    assert any("share key" in problem for problem in raised.value.problems)


def test_the_frame_reaches_the_run_fields(committed_pins: Any) -> None:
    fields = committed_pins.as_run_fields()["pins"]["benign_frame"]
    assert fields[FRAME_ID_FIELD] == committed_pins.benign_frame.frame_id
    assert fields["confirmatory_cell"]["baseline"]
    assert len(fields[BENIGN_CODE_CLASS]["repositories"]) >= MINIMUM_BENIGN_CODE_REPOSITORIES


def test_the_frame_draw_uses_the_same_vocabulary_as_the_attack_draw(
    committed_pins: Any,
) -> None:
    """Not "the same words": the same reader, so the two cannot drift into two vocabularies."""
    from nbc.pins import DRAW_METHODS

    assert committed_pins.benign_frame.method in DRAW_METHODS
    frame = committed_pins.benign_frame
    if frame.method == DRAW_SEEDED_RANDOM:
        assert frame.seed is not None and frame.sort_key is None
    else:
        assert frame.sort_key is not None and frame.seed is None
