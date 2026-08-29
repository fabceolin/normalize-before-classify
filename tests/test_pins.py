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
    MINIMUM_BASELINES,
    NOT_DECLARED,
    PINNED_PRECISION,
    PINS_FILENAME,
    SCHEMA_VERSION,
    BaselineSetInvalid,
    PinMismatch,
    PinsFileInvalid,
    RemoteArtifact,
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
        "window_policy": "shared",
        "window": {
            "length": 512,
            "source": "onnx/config.json::max_position_embeddings",
            "confirmed_on": "2026-08-28",
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


def echoing_resolver(artifact: RemoteArtifact) -> str | None:
    """A world in which every pin still resolves to itself."""
    return artifact.revision


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


def test_a_declared_training_relationship_loads_and_is_left_to_the_lineage_gate(
    tmp_path: Path,
) -> None:
    """Recording the relationship is this module's job; refusing the baseline is the gate's.

    Splitting them keeps `load_pins` a reader of what is declared, so the eligibility rule has
    exactly one home instead of a half-implementation here and the real one elsewhere.
    """
    document = _document()
    document["baseline"][0]["lineage"]["attack_datasets"]["example/attacks"] = (
        "declared-training-source"
    )
    write_pins(tmp_path, document)

    pins = load_pins(tmp_path)
    assert (
        pins.baselines[0].lineage.relationship_to("example/attacks")
        == "declared-training-source"
    )


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
    assert resolved == {
        "example/first-model": SHA_A,
        "example/second-model": SHA_B,
        "example/attacks": SHA_D,
    }


def test_a_moved_revision_aborts_naming_the_artifact_and_both_shas(tmp_path: Path) -> None:
    write_pins(tmp_path, _document())
    pins = load_pins(tmp_path)
    moved = "c" * 40

    def resolver(artifact: RemoteArtifact) -> str | None:
        return moved if artifact.repository == "example/first-model" else artifact.revision

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

    def resolver(artifact: RemoteArtifact) -> str | None:
        return moved if artifact.kind == "dataset" else artifact.revision

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
    """The snapshot directory is named by the sha, so its existence is the resolution."""
    artifact = RemoteArtifact("model", "example/first-model", SHA_A)
    snapshot = tmp_path / artifact.cache_directory / "snapshots" / SHA_A
    snapshot.mkdir(parents=True)

    assert resolve_from_cache(artifact, cache_root=tmp_path) == SHA_A


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
        identifiers.update(entry.get("lineage", {}).get("attack_datasets", {}))
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


def test_the_three_aborts_have_three_distinct_codes() -> None:
    codes = {
        PinsFileInvalid.exit_code,
        BaselineSetInvalid.exit_code,
        PinMismatch.exit_code,
    }
    assert len(codes) == 3
    from nbc.platform import UnsupportedPlatform

    assert UnsupportedPlatform.exit_code not in codes


# --- the world, once, over the network ----------------------------------------------------


@pytest.mark.smoke
def test_every_committed_pin_still_resolves_on_the_hub() -> None:
    """The only test here that touches a network, and it is excluded from the default run."""
    from nbc.pins import resolve_over_http

    pins = load_pins()
    resolved = verify_revisions(pins, resolve_over_http)
    assert len(resolved) == len(pins.remote_artifacts())
