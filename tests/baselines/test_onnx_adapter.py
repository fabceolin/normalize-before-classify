"""The adapter is bound to CPU, feeds what the graph declares, and batches without flipping.

Every session here is a real `onnxruntime.InferenceSession` over a real ONNX graph emitted by
`onnx_fixtures`. A fake session would make each of these assertions a statement about the fake:
`get_providers()` would return what the double was told to return, and the "declared signature"
the feed is built from would be the one the test wrote at both ends.
"""

from __future__ import annotations

import ast

import json
import re
from pathlib import Path
from typing import Sequence

import pytest

import onnx_fixtures
from nbc import pins
from nbc.baselines import port
from nbc.baselines.onnx_adapter import (
    BATCH_SIZE,
    FEEDABLE_INPUTS,
    INTRA_OP_NUM_THREADS,
    PROVIDERS,
    DEVICE,
    InferenceSessionInvalid,
    observed_device,
    OnnxBaseline,
    open_baseline,
    read_id2label,
)
from nbc.baselines.port import PositiveClassUnresolved
from nbc.errors import NbcError, declared_exit_codes
from nbc.schema import Score

PINNED_MAPPING = {"0": "SAFE", "1": "INJECTION"}
DEBERTA_INPUTS = ("input_ids", "attention_mask")
"""What the pinned DeBERTa-family graph declares."""
BERT_INPUTS = ("input_ids", "attention_mask", "token_type_ids")
"""What the pinned BERT-family graph declares. The families genuinely differ here."""


def windows_of(*documents: Sequence[Sequence[int]], strict: bool = True) -> port.Windower:
    """A windower that hands back a fixed answer, so the adapter is the only thing under test.

    The real one arrives with the shared window policy; this seam is a parameter precisely so
    that the policy is applied identically for every adapter rather than grown inside one.
    """

    def windower(texts: Sequence[str]) -> list[list[port.TokenWindow]]:
        if strict:
            assert len(texts) == len(documents), "the double was built for a different call"
        return [[tuple(window) for window in document] for document in documents]

    return windower


SRC = Path(__file__).resolve().parents[2] / "src"
"""The package tree the scan below walks. One spelling, so a moved test does not silently
scan nothing."""

CPU_ONLY: tuple[str, ...] = ("CPUExecutionProvider",)
"""The path every fixture session in this module runs on. Not the published one, and named so.

`PROVIDERS` is CUDA since 2026-08-30. These sessions exist to exercise the adapter's contract on
whatever machine a reviewer has, so they declare CPU explicitly instead of inheriting a constant
that would make the whole module unrunnable off a GPU.
"""

def adapter(
    *,
    inputs: Sequence[str] = DEBERTA_INPUTS,
    id2label: dict = PINNED_MAPPING,
    windower: port.Windower | None = None,
    reduce: str = "sum",
    num_labels: int = 2,
    input_type: int = onnx_fixtures.INT64,
    rank: int = 2,
    key: str = "fixture",
    batch_size: int = BATCH_SIZE,
    providers: tuple[str, ...] = CPU_ONLY,
    device: str | None = None,
) -> OnnxBaseline:
    """Every fixture session runs on CPU, declared here rather than inherited from the constants.

    The published path moved to CUDA on 2026-08-30 and these tests are about the ADAPTER'S
    CONTRACT -- which inputs it can feed, which dtypes it refuses, which axis carries the labels,
    whether a document's score depends on its batch neighbours. None of that is a claim about
    hardware, and a contract test that could only run on a GPU is a contract nobody can check: the
    CI runners are `ubuntu-latest`. What protects the published table is not this default, it is
    `harness/score.py::path_problems` comparing what each shard RECORDED against `DeclaredPath`,
    plus the scan below that refuses any module under `src/` naming these two parameters.
    """
    return OnnxBaseline(
        key=key,
        graph=onnx_fixtures.classifier_graph(
            inputs=inputs,
            num_labels=num_labels,
            reduce=reduce,
            input_type=input_type,
            rank=rank,
        ),
        id2label=id2label,
        windower=windower if windower is not None else windows_of([[1, 2, 3]]),
        batch_size=batch_size,
        providers=providers,
        device=device,
    )


# -- the device is named, not defaulted ------------------------------------------------------


def test_the_published_path_names_cuda_first_and_cpu_as_the_declared_fallback() -> None:
    """AD-24's constant, re-pointed on 2026-08-30 and asserted rather than assumed.

    Two entries and this order. The CUDA provider does not implement every operator in the pinned
    graphs, so the session reports both as active; a single-entry tuple would make the observed
    value disagree with the declared one on the first real run, and `path_problems` would abort a
    correct pass. CPU-first would silently publish a CPU table from a GPU box.
    """
    assert PROVIDERS == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_the_published_device_is_named_with_its_compute_capability() -> None:
    """`providers` cannot separate two CUDA cards; this is the field that does.

    The capability is in the string because it is what selects the kernels: a Tesla P40 (6.1) and
    an RTX 3060 (8.6) both answer `CUDAExecutionProvider`, and the machine this table is computed
    on carries both.
    """
    assert DEVICE.endswith(")")
    name, _, capability = DEVICE.rpartition(" ")
    assert name
    major, _, minor = capability.strip("()").partition(".")
    assert major.isdigit() and minor.isdigit()


def test_the_active_provider_of_every_session_is_the_cpu_one() -> None:
    """AD-24's test. A session that picked up an accelerator produces numbers nobody can check."""
    for inputs in (DEBERTA_INPUTS, BERT_INPUTS):
        assert adapter(inputs=inputs).providers == ("CPUExecutionProvider",)


def test_the_adapter_reads_no_device_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every variable below steers a device somewhere in the stack. None of them is consulted."""
    for variable in ("CUDA_VISIBLE_DEVICES", "ORT_PROVIDERS", "NBC_DEVICE", "ONNXRUNTIME_DEVICE"):
        monkeypatch.setenv(variable, "gpu")
    assert adapter().providers == ("CPUExecutionProvider",)


def test_the_provider_list_is_named_at_construction_rather_than_defaulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The argument is asserted, not the outcome, because on this machine they agree.

    A CPU-only machine cannot tell a session that named `CPUExecutionProvider` from one that
    defaulted into it -- `get_providers()` says the same thing either way. That is exactly the
    failure AD-24 describes: the premise evaporates and every test still passes. So the call is
    what is checked, and it is checked wherever the run happens to execute.
    """
    import onnxruntime as ort

    named: list[object] = []
    threads: list[int] = []
    original = ort.InferenceSession.__init__

    def recording(self: ort.InferenceSession, *args: object, **kwargs: object) -> None:
        named.append(kwargs.get("providers"))
        threads.append(args[1].intra_op_num_threads)
        original(self, *args, **kwargs)

    monkeypatch.setattr(ort.InferenceSession, "__init__", recording)
    adapter()
    assert named == [["CPUExecutionProvider"]]
    assert threads == [INTRA_OP_NUM_THREADS], "the declared thread count reaches the session"


def test_no_session_anywhere_in_the_package_is_built_without_a_provider_list(
    repo_root: Path,
) -> None:
    """One construction site is checked above; this is the rule the next one inherits."""
    offenders: list[str] = []
    for source in sorted((repo_root / "src" / "nbc").rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(r"InferenceSession\(", text):
            statement = text[match.start() : text.find("\n\n", match.start())]
            if "providers=" not in statement:
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{source.relative_to(repo_root)}:{line}")
    assert not offenders, "; ".join(offenders)


def test_the_model_boundary_reads_nothing_from_the_environment(repo_root: Path) -> None:
    """The device is a property of the published run, not of the machine that happens to run it.

    `pins.hf_cache_root()` is the one place this project reads the environment, and it reads
    where somebody else's cache lives -- not a parameter of this run. Nothing at the model
    boundary gets that latitude.
    """
    offenders: list[str] = []
    for source in sorted((repo_root / "src" / "nbc" / "baselines").glob("*.py")):
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"os\.environ|getenv|environb", line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, "; ".join(offenders)


def test_a_three_class_repository_scores_over_its_full_label_axis() -> None:
    """`softmax` runs over every logit, so a third class takes mass off the other two."""
    two = adapter(windower=windows_of([[3, 3]]))
    three = adapter(
        id2label={"0": "safe", "1": "injection", "2": "benign"},
        num_labels=3,
        windower=windows_of([[3, 3]]),
    )
    assert three.positive_index == 1
    assert three.score(["x"])[0].p_injection < two.score(["x"])[0].p_injection


def test_a_session_the_runtime_moved_off_cpu_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider list is verified after construction, not trusted because it was passed.

    Naming the provider and never looking again is how the premise ends up living in a
    paragraph: `InferenceSession` is free to fall back, and a fallback is silent.
    """
    import onnxruntime as ort

    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CUDAExecutionProvider"]
    )
    with pytest.raises(InferenceSessionInvalid, match="CUDAExecutionProvider"):
        adapter()


# -- the feed comes from the declared signature ----------------------------------------------


def test_the_feed_carries_exactly_the_inputs_the_graph_declares() -> None:
    """The pinned families differ on `token_type_ids`, so a convention is right about one."""
    deberta = adapter(inputs=DEBERTA_INPUTS)
    bert = adapter(inputs=BERT_INPUTS)

    assert set(deberta._feed([(1, 2, 3)])) == set(DEBERTA_INPUTS)
    assert set(bert._feed([(1, 2, 3)])) == set(BERT_INPUTS)
    assert "token_type_ids" not in deberta._feed([(1, 2, 3)])


def test_both_declared_signatures_score_without_the_adapter_being_told_which_is_which() -> None:
    for inputs in (DEBERTA_INPUTS, BERT_INPUTS):
        scored = adapter(inputs=inputs).score(["one document"])
        assert len(scored) == 1 and isinstance(scored[0], Score)


def test_shorter_windows_are_padded_and_the_padding_is_masked() -> None:
    fed = adapter(inputs=BERT_INPUTS)._feed([(7, 7, 7), (7,)])
    assert fed["input_ids"] == [[7, 7, 7], [7, 0, 0]]
    assert fed["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert fed["token_type_ids"] == [[0, 0, 0], [0, 0, 0]]


def test_the_declared_input_order_is_the_graph_s_and_not_the_adapter_s() -> None:
    built = adapter(inputs=("input_ids", "token_type_ids", "attention_mask"))
    assert built.graph_inputs == ("input_ids", "token_type_ids", "attention_mask")


def test_an_input_the_adapter_cannot_produce_aborts_rather_than_being_zero_filled() -> None:
    """A zero tensor for `position_ids` runs, returns logits, and answers a different question."""
    with pytest.raises(InferenceSessionInvalid, match="position_ids"):
        adapter(inputs=("input_ids", "position_ids"))


def test_a_graph_with_nowhere_to_put_the_tokens_aborts() -> None:
    with pytest.raises(InferenceSessionInvalid, match="input_ids"):
        adapter(inputs=("attention_mask",))


def test_an_input_declared_with_another_dtype_aborts_instead_of_being_coerced() -> None:
    with pytest.raises(InferenceSessionInvalid, match="tensor\\(int32\\)"):
        adapter(input_type=onnx_fixtures.INT32)


def test_an_input_that_is_not_batch_by_sequence_aborts() -> None:
    with pytest.raises(InferenceSessionInvalid, match="shape"):
        adapter(rank=1)


def test_a_label_axis_narrower_than_the_declared_labels_aborts() -> None:
    """The resolved index would address an axis position the graph never emits."""
    with pytest.raises(InferenceSessionInvalid, match="label axis"):
        adapter(id2label={"0": "safe", "1": "injection", "2": "benign"}, num_labels=2)


def test_a_graph_onnxruntime_refuses_is_reported_as_a_session_problem() -> None:
    with pytest.raises(InferenceSessionInvalid, match="refused the pinned graph"):
        OnnxBaseline(
            key="not-a-graph",
            graph=b"this is not an onnx model",
            id2label=PINNED_MAPPING,
            windower=windows_of([[1]]),
        )


# -- the score is the port's, and the index is the repository's ------------------------------


def test_p_injection_follows_the_declared_mapping_and_not_the_position() -> None:
    """The same graph, two label declarations, complementary scores. Nobody guessed an index.

    The fixture puts the larger logit on the last axis position, so an adapter that hardcoded
    `1` would report the *same* number for both of these, and one of the two columns of the
    published table would be the complement of what it claims.
    """
    document = [[1, 2, 3]]
    declared = adapter(id2label={"0": "SAFE", "1": "INJECTION"}, windower=windows_of(document))
    reversed_ = adapter(id2label={"0": "INJECTION", "1": "SAFE"}, windower=windows_of(document))

    high = declared.score(["x"])[0].p_injection
    low = reversed_.score(["x"])[0].p_injection
    assert high > 0.5 > low
    assert high + low == pytest.approx(1.0, abs=1e-12)


def test_a_baseline_whose_positive_class_cannot_be_resolved_never_builds_a_session() -> None:
    """Ineligibility is cheaper than a session, and the abort should not depend on graph load."""
    with pytest.raises(PositiveClassUnresolved):
        adapter(id2label={"0": "LABEL_0", "1": "LABEL_1"})


def test_a_document_is_reduced_to_the_maximum_over_its_windows() -> None:
    """The reduction is the port's, so every adapter reduces a long document the same way."""
    built = adapter(windower=windows_of([[1], [5], [2]]))
    scored = built.score(["a long document"])
    assert scored[0].n_windows == 3
    # The fixture sums every declared input, so the busiest window scores 5 ids + 1 mask.
    assert scored[0].p_injection == pytest.approx(port.p_injection([-6.0, 6.0], 1), abs=1e-9)


def test_documents_come_back_in_the_order_they_went_in() -> None:
    built = adapter(windower=windows_of([[9, 9]], [[1]], [[4, 4, 4]]))
    scored = built.score(["big", "small", "middling"])
    assert [score.n_windows for score in scored] == [1, 1, 1]
    assert scored[0].p_injection > scored[2].p_injection > scored[1].p_injection


def test_the_windower_is_held_to_its_contract() -> None:
    with pytest.raises(ValueError, match="window lists"):
        adapter(windower=windows_of([[1]], strict=False)).score(["one", "two"])
    with pytest.raises(ValueError, match="at least one"):
        adapter(windower=windows_of([], strict=False)).score(["one"])
    with pytest.raises(ValueError, match="at least one token"):
        adapter(windower=windows_of([[]])).score(["one"])


# -- the inference parameters are declared, fixed, and harmless to the class -----------------


def test_batch_size_and_thread_count_are_declared_constants() -> None:
    assert isinstance(BATCH_SIZE, int) and BATCH_SIZE >= 1
    assert INTRA_OP_NUM_THREADS == 1, "a threaded float32 reduction does not promise NFR4"


def test_the_declared_parameters_reach_the_session() -> None:
    built = adapter()
    assert built.batch_size == BATCH_SIZE
    assert built.intra_op_num_threads == INTRA_OP_NUM_THREADS


@pytest.mark.parametrize("bad", [0, -1])
def test_a_batch_size_below_one_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        adapter(batch_size=bad)


def test_a_document_scored_alone_and_in_a_full_batch_yields_the_same_class() -> None:
    """AD-22's test, on a graph whose float genuinely moves when the batch pads.

    `reduce="mean"` divides by the padded width, so the same window scores differently
    depending on what it was batched with -- which is exactly what batched inference does to a
    real model through padding. The float is allowed to move. The class is not.
    """
    alone = adapter(reduce="mean", windower=windows_of([[4, 4, 4]]))
    together = adapter(
        reduce="mean",
        windower=windows_of([[4, 4, 4]], [[1] * 40]),
        batch_size=8,
    )

    solo = alone.score(["short"])[0].p_injection
    batched = together.score(["short", "much longer"])[0].p_injection

    assert solo != batched, "a padding-insensitive fixture would make this test a tautology"
    assert (solo >= 0.5) == (batched >= 0.5)


def test_every_window_is_scored_whatever_the_batch_size() -> None:
    """A window dropped at a batch boundary is a document scored on less than it contains."""
    document = [[index + 1] for index in range(BATCH_SIZE * 2 + 3)]
    for batch_size in (1, 2, BATCH_SIZE, BATCH_SIZE * 4):
        scored = adapter(windower=windows_of(document), batch_size=batch_size).score(["x"])
        assert scored[0].n_windows == len(document)


def test_batching_does_not_reorder_documents() -> None:
    documents = [[[index + 1]] for index in range(BATCH_SIZE * 2 + 1)]
    scored = adapter(windower=windows_of(*documents), batch_size=3).score(
        [f"doc {index}" for index in range(len(documents))]
    )
    values = [score.p_injection for score in scored]
    assert values == sorted(values), "document i has a larger token sum than document i-1"


# -- what the run records --------------------------------------------------------------------


def test_the_resolved_mapping_and_the_parameters_travel_into_the_run_block() -> None:
    fields = adapter(inputs=BERT_INPUTS, key="testsavantai-bert-small").as_run_fields()
    assert fields == {
        "key": "testsavantai-bert-small",
        "id2label": {"0": "SAFE", "1": "INJECTION"},
        "positive_index": 1,
        "providers": ["CPUExecutionProvider"],
        "device": None,
        "batch_size": BATCH_SIZE,
        "intra_op_num_threads": INTRA_OP_NUM_THREADS,
        "graph_inputs": list(BERT_INPUTS),
    }


def test_the_run_block_is_json_serializable() -> None:
    json.dumps(adapter().as_run_fields())


# -- opening a pinned baseline ---------------------------------------------------------------


def _cache(root: Path, baseline: pins.Baseline, *, id2label: dict | None = None) -> Path:
    """A Hugging Face cache laid out the way the hub lays one out, with a fixture graph in it."""
    snapshot = baseline.artifact.snapshot_dir(root)
    graph = snapshot / baseline.graph_path
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_bytes(onnx_fixtures.classifier_graph(inputs=DEBERTA_INPUTS))
    config = snapshot / baseline.config_path
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"id2label": id2label if id2label is not None else PINNED_MAPPING}),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def pinned() -> pins.Baseline:
    return pins.load_pins().baselines[0]


def test_a_pinned_baseline_opens_from_the_paths_the_pin_names(
    pinned: pins.Baseline, tmp_path: Path
) -> None:
    root = _cache(tmp_path, pinned)
    # The CPU path, named here for the reason `adapter` names it: this test is about which files
    # the pin resolves to, not about which device the published table comes from.
    built = open_baseline(
        pinned, windows_of([[1, 2]]), cache_root=root, providers=CPU_ONLY, device=None
    )
    assert built.key == pinned.key
    assert built.positive_index == 1
    assert built.providers == ("CPUExecutionProvider",)
    assert built.score(["x"])[0].n_windows == 1


def test_a_baseline_missing_from_the_cache_aborts_before_anything_is_scored(
    pinned: pins.Baseline, tmp_path: Path
) -> None:
    with pytest.raises(InferenceSessionInvalid, match="not in the Hugging Face cache"):
        open_baseline(pinned, windows_of([[1]]), cache_root=tmp_path)


def test_a_config_without_id2label_makes_the_baseline_ineligible(
    pinned: pins.Baseline, tmp_path: Path
) -> None:
    root = _cache(tmp_path, pinned, id2label={})
    with pytest.raises(PositiveClassUnresolved):
        open_baseline(pinned, windows_of([[1]]), cache_root=root)


def test_a_config_that_is_not_json_is_reported_as_a_session_problem(tmp_path: Path) -> None:
    broken = tmp_path / "config.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(InferenceSessionInvalid, match="not JSON"):
        read_id2label(broken)


def test_a_config_that_cannot_be_read_is_reported_as_a_session_problem(tmp_path: Path) -> None:
    with pytest.raises(InferenceSessionInvalid, match="could not be read"):
        read_id2label(tmp_path / "absent.json")


def test_read_id2label_reads_and_does_not_decide(tmp_path: Path) -> None:
    """Absence is the resolver's verdict to reach, not this reader's."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")
    assert read_id2label(config) == {}


# -- the aborts stay distinguishable ---------------------------------------------------------


def test_the_two_aborts_this_boundary_declares_have_two_distinct_codes() -> None:
    assert PositiveClassUnresolved.exit_code == 8
    assert InferenceSessionInvalid.exit_code == 9


def test_every_declared_abort_still_has_a_code_of_its_own() -> None:
    codes = declared_exit_codes()
    assert len(set(codes)) == len(codes)
    for code, declared in codes.items():
        assert issubclass(declared, NbcError)
    assert {8, 9} <= set(codes)


def test_an_invalid_session_abort_carries_its_problems() -> None:
    raised = InferenceSessionInvalid("first", "second")
    assert raised.problems == ("first", "second")
    assert "first" in str(raised) and "second" in str(raised)
    with pytest.raises(ValueError):
        InferenceSessionInvalid()


def test_the_feedable_inputs_are_the_ones_the_pinned_families_declare() -> None:
    assert set(BERT_INPUTS) <= set(FEEDABLE_INPUTS)
    assert set(DEBERTA_INPUTS) <= set(FEEDABLE_INPUTS)


# -- the pinned repositories, against the live cards ------------------------------------------


@pytest.mark.smoke
def test_every_pinned_baseline_declares_exactly_one_positive_class() -> None:
    """The only test here that touches a network, and it is excluded from the default run.

    Everything above proves the resolution rule behaves; this proves the rule has a subject.
    A pinned repository whose `config.json` resolves to zero or to two positive classes is
    ineligible, and the cheapest moment to learn that is now rather than after a corpus exists.
    Only `config.json` is fetched -- a few kilobytes each, not the graphs.
    """
    import urllib.request

    resolved: dict[str, int] = {}
    for baseline in pins.load_pins().baselines:
        url = (
            f"https://huggingface.co/{baseline.repository}/resolve/"
            f"{baseline.revision}/{baseline.config_path}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            config = json.loads(response.read().decode("utf-8"))
        resolved[baseline.key] = port.resolve_positive_index(
            config.get("id2label"), baseline=baseline.key
        )
    assert len(resolved) == len(pins.load_pins().baselines)


# -- batch composition is not a free parameter of a score ---------------------------------------
#
# The existing test above asserts the CLASS survives batching, which is what AD-22 requires. The
# review asked a harder question: does the VALUE survive? Measured on both pinned graphs, over
# 300 real attack payloads of 97 to 11869 characters, under chunk sizes 8, 32 and 32-reversed
# against a one-document-per-call reference:
#
#     max|delta| = 0.000e+00 and zero class flips, on both baselines.
#
# The padded width genuinely varied by 100x in that measurement -- the same document is padded to
# 5 tokens alone and 512 beside a long one -- so the zero is invariance, not an untested path.
# Both pinned graphs declare attention_mask and the adapter builds it correctly, which is why.
#
# That is an OBSERVATION about two pins. These tests make it a PROPERTY: a replacement baseline
# whose graph does not honour its mask fails here rather than quietly moving a published rate.


def test_a_documents_score_does_not_depend_on_its_batch_neighbours() -> None:
    """The same document, alone and beside a much longer one, scores identically.

    `_feed` pads to the longest window in the batch, and the batch mixes windows from different
    documents -- so the tensor a document is scored on is a function of its neighbours. On a
    graph that honours its mask that is invisible, and this asserts the invisibility exactly
    rather than to a tolerance: a float that moves in the last decimal moves a class at the
    threshold, which is the borderline encoded items this whole experiment measures.
    """
    document = [[1, 2, 3]]
    long_neighbour = [[index + 1 for index in range(400)]]

    alone = adapter(windower=windows_of(document)).score(["doc"])
    together = adapter(windower=windows_of(document, long_neighbour)).score(["doc", "long"])

    assert together[0].p_injection == alone[0].p_injection


def test_the_batch_size_constant_is_not_a_free_parameter_either() -> None:
    """Changing only BATCH_SIZE must not move a published number.

    The constant is documented as a memory knob. The spike slices its corpus by `--chunk`, which
    determines batch composition, so if composition moved scores then a memory knob would be a
    measurement parameter and two runs over identical inputs could report different recalls.
    """
    # Windows of DELIBERATELY UNEQUAL width. The first version of this test built every document
    # two tokens wide, so `_feed` never padded at any batch size and the assertion below held for
    # any graph at all -- including one with `attention_mask` replaced by all-ones. A test written
    # to prove a constant is not a free parameter, that could not observe the parameter.
    documents = [
        [[value + 1 for value in range(1 + (index % 7) * 60)]]
        for index in range(BATCH_SIZE * 2 + 1)
    ]
    texts = [f"doc {index}" for index in range(len(documents))]

    by_size = {
        size: [
            score.p_injection
            for score in adapter(windower=windows_of(*documents), batch_size=size).score(texts)
        ]
        for size in (1, 2, BATCH_SIZE, BATCH_SIZE * 4)
    }

    reference = by_size[1]
    for size, values in by_size.items():
        assert values == reference, f"batch_size={size} moved the scores away from batch_size=1"


def test_the_invariance_test_can_fail() -> None:
    """The same assertion on a graph that ignores its mask, so the test above is not a tautology.

    `reduce="mean"` divides by the padded width, which is what a graph feeding a zero-padded
    tensor without honouring `attention_mask` effectively does. This is also the fixture the
    review's 0.32-to-0.48 swings were measured on, and stating that here is why those numbers
    did not reproduce on the pinned models.
    """
    document = [[1, 2, 3]]
    long_neighbour = [[index + 1 for index in range(400)]]

    alone = adapter(reduce="mean", windower=windows_of(document)).score(["doc"])
    together = adapter(
        reduce="mean", windower=windows_of(document, long_neighbour)
    ).score(["doc", "long"])

    assert together[0].p_injection != alone[0].p_injection


# -- the device, and who is allowed to choose one --------------------------------------------


def session_choices(path: Path) -> list[str]:
    """Every `providers=`/`device=` keyword handed to a call that BUILDS a session, from the tree.

    Keyed on the callee and not on the keyword, and that distinction is the whole check:
    `ExecutionPath(device=...)` and `DeclaredPath(device=...)` record a path and appear all over
    `harness/`, while `OnnxBaseline(...)` and `open_baseline(...)` choose one. A scan that saw only
    the keyword flagged the recorders too, which is a scan somebody switches off.
    """
    builders = {"OnnxBaseline", "open_baseline"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Both spellings, so an import style is not a way around the rule.
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else None
        )
        if name in builders:
            found.extend(k.arg for k in node.keywords if k.arg in ("providers", "device"))
    return found


def test_a_process_that_cannot_reach_cuda_says_so_instead_of_naming_a_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`observed_device` aborts rather than returning something every comparison would accept.

    Driven through `ctypes.CDLL` so the input is the same on a GPU box and on a CI runner: a
    test that only turned red on machines without CUDA would be green on the one machine where
    the published table is produced, which is where it matters.
    """
    import ctypes

    def refuse(_name: str, *args: object, **kwargs: object) -> object:
        raise OSError("libcudart.so: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", refuse)
    with pytest.raises(InferenceSessionInvalid, match="CUDA runtime could not be loaded"):
        observed_device()


def test_no_module_under_src_chooses_its_own_provider_list_or_device() -> None:
    """The override exists for the tests. A second caller would be a second published path.

    `providers` and `device` are parameters with the declared constants as defaults, which is what
    lets this module exercise the adapter's contract on a CPU-only runner. That freedom has to stop
    at `src/`: a module that passed either keyword would be choosing the execution path of the
    published table somewhere other than where it is declared, and `DeclaredPath` would go on
    reading the constants and agreeing with itself.

    `open_baseline` is the one exception and it is exempt by NAME, not by pattern: it forwards the
    parameters it was given and adds nothing. The scan reads the syntax tree, so a keyword split
    across lines or hidden behind an alias is still a keyword.
    """
    allowed = {"nbc/baselines/onnx_adapter.py"}
    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        named = session_choices(path)
        if named and relative not in allowed:
            offenders[relative] = named
    assert offenders == {}
    # And the exemption is verified rather than asserted: the one allowed module really does
    # forward them, so `allowed` is not a name nobody would have flagged anyway.
    assert session_choices(SRC / "nbc/baselines/onnx_adapter.py")


def test_the_scan_above_reports_a_second_chooser(tmp_path: Path) -> None:
    """The scan, shown catching one, so it is not a check that passes by matching nothing."""
    module = tmp_path / "sneaky.py"
    module.write_text(
        "from nbc.baselines.onnx_adapter import open_baseline\n"
        "def go(b, w):\n"
        "    return open_baseline(b, w, providers=('CPUExecutionProvider',), device=None)\n",
        encoding="utf-8",
    )
    assert sorted(session_choices(module)) == ["device", "providers"]

    # The distractor, in the same shape: `ExecutionPath(device=...)` RECORDS a device, it does not
    # choose one, and `run.py` and `score.py` are full of it. A scan that keyed on the keyword
    # alone flagged both of them and would have been switched off within a day.
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        "from nbc.harness.score import ExecutionPath\n"
        "def record(observed):\n"
        "    return ExecutionPath(baseline_key='k', revision='r', providers=('CPUExecutionProvider',),\n"
        "                         intra_op_num_threads=1, batch_size=8, device=observed)\n",
        encoding="utf-8",
    )
    assert session_choices(recorder) == []
