"""One window policy, applied identically to every baseline, and nothing inherited from a file.

The test that carries this module is not the one that reads `tokenizer.truncation`: both pinned
files declare `null` today, so that assertion passes for the wrong reason on both of them. The one
that earns its place windows a document longer than the window through a tokenizer whose *file*
declares truncation, and counts the windows. Delete `no_truncation()` and the count silently drops
to one -- which is the confound in production, reproduced at fixture scale.

That the confound is real, and one `pins.toml` edit away, is the smoke test at the bottom: at the
pinned revision one repository ships `tokenizer.json` twice, at the root declaring
`truncation.max_length` and beside the ONNX graph declaring `null`. The pin names the second. A
pin that named only the repository would take the first.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

import nbc
import onnx_fixtures
import tokenizer_fixtures as fixtures
from nbc import pins
from nbc.baselines import port, tokenization
from nbc.baselines.onnx_adapter import OnnxBaseline
from nbc.baselines.tokenization import (
    SHARED,
    WINDOW_POLICIES,
    SpecialTokenFrame,
    WindowPolicyInvalid,
    derive_frame,
    load_tokenizer,
    open_windower,
)
from nbc.errors import NbcError, declared_exit_codes

FIXTURE_WINDOW = 8
"""A small window, so a "document longer than one window" is three words rather than a page.

The number is a property of this file's fixtures and of nothing else: no pinned window length is
written anywhere in the suite, and a test below scans the source tree to keep it that way.
"""

FIXTURE_FRAME = 2
CONTENT = FIXTURE_WINDOW - FIXTURE_FRAME


@pytest.fixture()
def committed() -> pins.Pins:
    return pins.load_pins()


def windower(
    tmp_path: Path,
    *,
    truncation: int | None = None,
    padding: int | None = None,
    tokenizer: Any = None,
    length: int = FIXTURE_WINDOW,
) -> tokenization.WindowedTokenizer:
    """A windower over a real tokenizer file, loaded through the shared path under test."""
    path = fixtures.write(
        tmp_path / "fixture.json", tokenizer, truncation=truncation, padding=padding
    )
    return tokenization.WindowedTokenizer(
        key="fixture",
        tokenizer=load_tokenizer(path, baseline="fixture"),
        window=fixtures.window_pin(length),
    )


# --- what the file declared does not survive the load ------------------------------------------


@pytest.mark.parametrize("declared", [None, 512, 2049])
def test_truncation_is_disabled_however_the_file_declared_it(tmp_path: Path, declared: int) -> None:
    """The field varies across published files and correlates with nothing.

    512 is what one pinned repository's *root* `tokenizer.json` declares while the file the pin
    names declares nothing; 2049 is what a third repository, since dropped, declared -- an
    off-by-one against its own published window. None of it can be assumed, so none of it is.
    """
    path = fixtures.write(tmp_path / "fixture.json", truncation=declared)
    assert load_tokenizer(path, baseline="fixture").truncation is None


@pytest.mark.parametrize("declared", [None, 16])
def test_padding_is_disabled_however_the_file_declared_it(tmp_path: Path, declared: int) -> None:
    path = fixtures.write(tmp_path / "fixture.json", padding=declared)
    assert load_tokenizer(path, baseline="fixture").padding is None


def test_a_declared_truncation_is_really_in_the_file_the_loader_reads(tmp_path: Path) -> None:
    """Otherwise the two tests above would be neutralizing something that was never there."""
    path = fixtures.write(tmp_path / "fixture.json", truncation=512, padding=16)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["truncation"]["max_length"] == 512
    assert document["padding"] is not None


def test_a_document_longer_than_the_window_is_windowed_and_not_truncated(tmp_path: Path) -> None:
    """The test that fails when `no_truncation()` is deleted.

    A tokenizer whose file declares truncation at the window length would silently cut this
    document to one window, and every assertion about *properties* would still pass.
    """
    policy = windower(tmp_path, truncation=FIXTURE_WINDOW)
    long_document = fixtures.document(CONTENT * 3)

    windows = policy.windows([long_document])[0]

    assert len(windows) == 3
    assert [len(window) for window in windows] == [FIXTURE_WINDOW] * 3


def test_padding_declared_in_the_file_does_not_reach_the_content_axis(tmp_path: Path) -> None:
    """Padding at encode time puts tokens in a window that the document does not contain."""
    policy = windower(tmp_path, padding=FIXTURE_WINDOW * 4)
    expected = fixtures.ids(fixtures.wordpiece(), fixtures.document(3))

    windows = policy.windows([fixtures.document(3)])[0]

    assert len(windows) == 1
    assert fixtures.flatten(windows, policy.frame) == expected


def test_a_tokenizer_whose_truncation_survives_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    """A future `tokenizers` that renamed the call must abort, not look like it neutralized."""

    class Deaf:
        truncation = {"max_length": 512}
        padding = None

        def no_truncation(self) -> None:
            pass

        def no_padding(self) -> None:
            pass

    path = fixtures.write(tmp_path / "fixture.json")
    monkeypatch.setattr(
        tokenization, "Tokenizer", type("T", (), {"from_file": staticmethod(lambda _: Deaf())})
    )

    with pytest.raises(WindowPolicyInvalid, match="survived no_truncation"):
        load_tokenizer(path, baseline="fixture")


def test_a_tokenizer_whose_padding_survives_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    class Deaf:
        truncation = None
        padding = {"length": 16}

        def no_truncation(self) -> None:
            pass

        def no_padding(self) -> None:
            pass

    path = fixtures.write(tmp_path / "fixture.json")
    monkeypatch.setattr(
        tokenization, "Tokenizer", type("T", (), {"from_file": staticmethod(lambda _: Deaf())})
    )

    with pytest.raises(WindowPolicyInvalid, match="survived no_padding"):
        load_tokenizer(path, baseline="fixture")


# --- the windows themselves --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_tokens", "expected_windows", "expected_widths"),
    [
        (0, 1, [FIXTURE_FRAME]),
        (1, 1, [FIXTURE_FRAME + 1]),
        (CONTENT - 1, 1, [FIXTURE_WINDOW - 1]),
        (CONTENT, 1, [FIXTURE_WINDOW]),
        (CONTENT + 1, 2, [FIXTURE_WINDOW, FIXTURE_FRAME + 1]),
        (2 * CONTENT + 1, 3, [FIXTURE_WINDOW, FIXTURE_WINDOW, FIXTURE_FRAME + 1]),
    ],
)
def test_a_document_occupies_the_windows_the_policy_says_it_does(
    tmp_path: Path, content_tokens: int, expected_windows: int, expected_widths: list[int]
) -> None:
    """Including the empty document, which occupies one window carrying the frame alone."""
    policy = windower(tmp_path)

    windows = policy.windows([fixtures.document(content_tokens)])[0]

    assert len(windows) == expected_windows
    assert [len(window) for window in windows] == expected_widths


@pytest.mark.parametrize("content_tokens", [0, 1, CONTENT, CONTENT + 1, 4 * CONTENT + 3])
def test_no_encoded_window_exceeds_the_pinned_window(tmp_path: Path, content_tokens: int) -> None:
    policy = windower(tmp_path)

    windows = policy.windows([fixtures.document(content_tokens)])[0]

    assert max(len(window) for window in windows) <= policy.max_length


@pytest.mark.parametrize("content_tokens", [1, CONTENT, CONTENT + 1, 3 * CONTENT + 2])
def test_the_windows_reproduce_the_documents_tokens_in_order(
    tmp_path: Path, content_tokens: int
) -> None:
    """Non-overlapping and lossless: every content token appears exactly once, in order."""
    policy = windower(tmp_path)
    text = fixtures.document(content_tokens)
    expected = fixtures.ids(fixtures.wordpiece(), text)

    windows = policy.windows([text])[0]

    assert fixtures.flatten(windows, policy.frame) == expected


def test_every_window_carries_the_tokenizers_own_frame(tmp_path: Path) -> None:
    policy = windower(tmp_path)
    framed = fixtures.ids(fixtures.wordpiece(), fixtures.document(1), specials=True)

    windows = policy.windows([fixtures.document(CONTENT + 1)])[0]

    assert all(window[0] == framed[0] and window[-1] == framed[-1] for window in windows)


def test_documents_are_windowed_independently_and_in_order(tmp_path: Path) -> None:
    policy = windower(tmp_path)
    texts = [fixtures.document(1), fixtures.document(2 * CONTENT), fixtures.document(0)]

    windows = policy.windows(texts)

    assert [len(document) for document in windows] == [1, 2, 1]


def test_encoded_and_dressed_text_is_windowed_by_the_same_arithmetic(tmp_path: Path) -> None:
    """Zero-width characters and homoglyphs are the material, not an exotic input.

    They are what makes an encoded document several times longer in tokens than its decoded form,
    which is the whole reason a window policy decides this experiment's numbers.
    """
    policy = windower(tmp_path)
    dressed = "\u200b".join("\u0430\u0440\u0435" for _ in range(20))
    expected = fixtures.ids(fixtures.wordpiece(), dressed)

    windows = policy.windows([dressed])[0]

    assert len(windows) > 1, "the dressed document should not fit one fixture window"
    assert fixtures.flatten(windows, policy.frame) == expected
    assert max(len(window) for window in windows) <= policy.max_length


def test_a_document_windows_the_same_way_wherever_it_sits(tmp_path: Path) -> None:
    """No state survives a call: `n_windows` rides on `Score`, never on the module."""
    policy = windower(tmp_path)
    text = fixtures.document(2 * CONTENT + 1)

    alone = policy.windows([text])
    beside = policy.windows([fixtures.document(CONTENT), text, fixtures.document(0)])
    again = policy.windows([text])

    assert alone[0] == beside[1] == again[0]


def test_windowing_no_documents_produces_no_windows(tmp_path: Path) -> None:
    assert windower(tmp_path).windows([]) == []


def test_the_windower_is_the_callable_the_port_declares(tmp_path: Path) -> None:
    """`Windower` is a call, not a class, so the adapter cannot depend on this type."""
    policy = windower(tmp_path)

    assert policy([fixtures.document(1)]) == policy.windows([fixtures.document(1)])


# --- the frame is measured, never named --------------------------------------------------------


@pytest.mark.parametrize("family", ["wordpiece", "unigram"])
def test_the_frame_is_measured_from_the_tokenizers_own_template(
    tmp_path: Path, family: str
) -> None:
    """Both pinned families wrap a sequence; neither is named in the source."""
    tokenizer = getattr(fixtures, family)()
    policy = windower(tmp_path, tokenizer=tokenizer)
    expected = fixtures.ids(getattr(fixtures, family)(), fixtures.document(1), specials=True)

    assert policy.frame.prefix == (expected[0],)
    assert policy.frame.suffix == (expected[-1],)
    assert policy.num_special_tokens == FIXTURE_FRAME
    assert policy.content_length == FIXTURE_WINDOW - FIXTURE_FRAME


def test_a_tokenizer_that_adds_no_special_tokens_gets_the_whole_window(tmp_path: Path) -> None:
    """A zero-token frame is legal; what is not legal is assuming a two-token one."""
    policy = windower(tmp_path, tokenizer=fixtures.unframed())

    assert policy.num_special_tokens == 0
    assert policy.content_length == FIXTURE_WINDOW
    assert len(policy.windows([fixtures.document(FIXTURE_WINDOW)])[0]) == 1


def test_a_template_that_is_not_a_wrap_is_refused() -> None:
    """This policy wraps a window in a prefix and a suffix. A template it cannot apply aborts."""

    class Interleaving:
        """Encodes `content` as content with a special token in the middle of it."""

        def num_special_tokens_to_add(self, is_pair: bool) -> int:
            return 1

        def encode(self, text: str, add_special_tokens: bool = True) -> Any:
            content = tuple(range(10, 10 + len(text.split())))
            if not add_special_tokens:
                return _Encoding(content)
            middle = len(content) // 2
            return _Encoding((*content[:middle], 99, *content[middle:]))

    with pytest.raises(WindowPolicyInvalid, match="no single prefix/suffix frame"):
        derive_frame(Interleaving(), baseline="fixture")


def test_a_frame_the_tokenizer_disagrees_about_is_refused() -> None:
    """The measured frame and the declared count have to be the same number."""

    class Miscounting:
        def num_special_tokens_to_add(self, is_pair: bool) -> int:
            return 4

        def encode(self, text: str, add_special_tokens: bool = True) -> Any:
            content = tuple(range(10, 10 + len(text.split())))
            return _Encoding((1, *content, 2) if add_special_tokens else content)

    with pytest.raises(WindowPolicyInvalid, match="declares it adds 4"):
        derive_frame(Miscounting(), baseline="fixture")


def test_a_probe_that_encodes_to_nothing_is_refused() -> None:
    class Empty:
        def num_special_tokens_to_add(self, is_pair: bool) -> int:
            return 0

        def encode(self, text: str, add_special_tokens: bool = True) -> Any:
            return _Encoding(())

    with pytest.raises(WindowPolicyInvalid, match="no content tokens"):
        derive_frame(Empty(), baseline="fixture")


class _Encoding:
    def __init__(self, ids: Sequence[int]) -> None:
        self.ids = list(ids)


def test_the_frame_probes_are_two_and_they_differ() -> None:
    """One probe would let a template that repeats its content pass as a clean wrap."""
    assert len(tokenization.FRAME_PROBES) >= 2
    assert len(set(tokenization.FRAME_PROBES)) == len(tokenization.FRAME_PROBES)


# --- what the window has to come from ----------------------------------------------------------


def test_a_window_with_no_room_for_content_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WindowPolicyInvalid, match="cannot carry a document"):
        windower(tmp_path, length=FIXTURE_FRAME)


def test_a_window_of_no_tokens_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WindowPolicyInvalid, match="at least one token"):
        windower(tmp_path, length=0)


def test_an_uncached_tokenizer_aborts_before_anything_is_windowed(
    tmp_path: Path, committed: pins.Pins
) -> None:
    baseline = committed.baselines[0]

    with pytest.raises(WindowPolicyInvalid, match="not in the Hugging Face cache"):
        open_windower(baseline, cache_root=tmp_path)


def test_a_file_that_is_not_a_tokenizer_aborts(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text("{not a tokenizer}", encoding="utf-8")

    with pytest.raises(WindowPolicyInvalid, match="could not be loaded as a tokenizer"):
        load_tokenizer(path, baseline="fixture")


def test_the_tokenizer_is_read_from_the_path_the_pin_names(
    tmp_path: Path, committed: pins.Pins
) -> None:
    """One pinned repository ships two files of that name at one revision, declaring different
    truncation. A loader convention picks one of them per reader, so the pin names the path.

    The decoy is planted at the repository root, where the convention would look, and it is a
    tokenizer with a different frame -- so a loader that took it would be visible in the arithmetic
    rather than only in a path.
    """
    nested = [line for line in committed.baselines if "/" in line.tokenizer_path]
    assert nested, "no pinned baseline keeps its tokenizer beside the graph any more"
    baseline = nested[0]

    snapshot = tmp_path / baseline.artifact.cache_directory / "snapshots" / baseline.revision
    fixtures.write(snapshot / Path(baseline.tokenizer_path).name, fixtures.unframed())
    fixtures.plant(tmp_path, baseline, tokenizer=fixtures.wordpiece())

    policy = open_windower(baseline, cache_root=tmp_path)

    assert policy.num_special_tokens == FIXTURE_FRAME
    assert policy.content_length == baseline.window.length - FIXTURE_FRAME


# --- every baseline, identically ---------------------------------------------------------------


def test_every_pinned_baseline_loads_with_truncation_and_padding_disabled(
    tmp_path: Path, committed: pins.Pins
) -> None:
    """The AC's "for every baseline", as far as an offline suite can carry it.

    Each pinned baseline gets a tokenizer planted at the path its own pin names, and the pair
    carries both shapes -- one declaring truncation, one declaring none -- so neither route reaches
    the policy with a setting of its own. The real files are loaded by the smoke test at the bottom
    of this file, which is where "every baseline" stops being a fixture.
    """
    assert len(committed.baselines) >= pins.MINIMUM_BASELINES

    for index, baseline in enumerate(committed.baselines):
        path = fixtures.plant(
            tmp_path,
            baseline,
            truncation=baseline.window.length if index % 2 == 0 else None,
            tokenizer=fixtures.wordpiece() if index % 2 == 0 else fixtures.unigram(),
        )
        assert path.is_file()

        policy = open_windower(baseline, cache_root=tmp_path)

        assert policy.tokenizer.truncation is None
        assert policy.tokenizer.padding is None
        assert policy.max_length == baseline.window.length
        assert policy.content_length == baseline.window.length - policy.num_special_tokens
        assert policy.num_special_tokens > 0


def test_every_pinned_baseline_windows_a_long_document_rather_than_truncating_it(
    tmp_path: Path, committed: pins.Pins
) -> None:
    """The same document, over every baseline, occupying more than one window in each."""
    for baseline in committed.baselines:
        fixtures.plant(tmp_path, baseline, truncation=baseline.window.length)
        policy = open_windower(baseline, cache_root=tmp_path)
        text = fixtures.document(2 * policy.content_length + 1)

        windows = policy.windows([text])[0]

        assert len(windows) == 3
        assert max(len(window) for window in windows) == baseline.window.length


def test_every_pinned_baseline_declares_a_policy_that_can_run(committed: pins.Pins) -> None:
    for baseline in committed.baselines:
        assert baseline.window_policy in WINDOW_POLICIES


def test_the_window_axis_is_part_of_the_recorded_key_from_the_first_run(
    tmp_path: Path, committed: pins.Pins
) -> None:
    """A key retro-fitted into a published envelope is a schema break; this one was always there."""
    baseline = committed.baselines[0]
    fixtures.plant(tmp_path, baseline)

    fields = open_windower(baseline, cache_root=tmp_path).as_run_fields()

    assert fields["window_policy"] == pins.SHARED_WINDOW_POLICY
    assert fields["policy"] == {"name": "shared", "stride": 0, "aggregation": port.REDUCTION_NAME}
    assert fields["window"] == baseline.window.as_run_fields()
    assert baseline.as_run_fields()["window_policy"] == fields["window_policy"]


# --- the policy is a declared strategy ---------------------------------------------------------


def test_the_pinnable_names_and_the_runnable_strategies_are_the_same_set() -> None:
    """A name with no strategy selects whatever the fallback happened to be."""
    assert set(WINDOW_POLICIES) == pins.WINDOW_POLICIES


def test_the_shared_policy_is_non_overlapping_and_max_reduced() -> None:
    assert SHARED.name == pins.SHARED_WINDOW_POLICY
    assert SHARED.stride == 0
    assert SHARED.aggregation == port.REDUCTION_NAME


def test_a_policy_no_strategy_implements_is_refused(tmp_path: Path, committed: pins.Pins) -> None:
    baseline = committed.baselines[0]
    fixtures.plant(tmp_path, baseline)
    unrunnable = dataclasses.replace(baseline, window_policy="publisher")

    with pytest.raises(WindowPolicyInvalid, match="policies that can run"):
        open_windower(unrunnable, cache_root=tmp_path)


def test_a_stride_the_walk_does_not_apply_is_refused(tmp_path: Path) -> None:
    """A policy is a length, a stride and an aggregation together, and all three are published.

    Overlapping windows are not implemented. A stride recorded in `results.json` and ignored by
    the walk would be a parameter the run never applied, which is worse than not offering it.
    """
    path = fixtures.write(tmp_path / "fixture.json")

    with pytest.raises(WindowPolicyInvalid, match="stride"):
        tokenization.WindowedTokenizer(
            key="fixture",
            tokenizer=load_tokenizer(path, baseline="fixture"),
            window=fixtures.window_pin(FIXTURE_WINDOW),
            policy=tokenization.WindowPolicy(name=SHARED.name, stride=1, aggregation="max"),
        )


def test_the_document_reduction_is_the_ports_and_carries_the_window_count(
    tmp_path: Path,
) -> None:
    """`n_windows` rides on `Score`, never on module state, and the reduction is the port's."""
    policy = windower(tmp_path)

    windows = policy.windows([fixtures.document(2 * CONTENT + 1)])[0]
    score = port.reduce_windows([0.1, 0.9, 0.4][: len(windows)])

    assert score.n_windows == len(windows) == 3
    assert score.p_injection == 0.9


# --- the seam closes ---------------------------------------------------------------------------


def test_an_adapter_scores_a_long_document_through_the_shared_policy(tmp_path: Path) -> None:
    """The whole boundary, end to end: a real tokenizer file, a real graph, one `Score`.

    Story 1.5 left `Windower` a parameter so a policy could not grow inside an adapter. This is
    the test that the parameter accepts the real policy and that `n_windows` arrives on `Score`.
    """
    policy = windower(tmp_path, truncation=FIXTURE_WINDOW)
    baseline = OnnxBaseline(
        key="fixture",
        graph=onnx_fixtures.classifier_graph(),
        id2label={"0": "SAFE", "1": "INJECTION"},
        windower=policy,
    )

    scores = baseline.score([fixtures.document(2 * CONTENT + 1), fixtures.document(1)])

    assert [score.n_windows for score in scores] == [3, 1]
    assert all(0.0 <= score.p_injection <= 1.0 for score in scores)


# --- nothing is hard-coded, and nothing else loads a tokenizer ---------------------------------


def _source_files() -> list[Path]:
    return sorted(Path(nbc.__file__).resolve().parent.rglob("*.py"))


def test_only_the_shared_module_constructs_a_tokenizer() -> None:
    """Every tokenizer in this project is neutralized on load, which needs one loader."""
    offenders: list[str] = []
    for path in _source_files():
        if path == Path(tokenization.__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for needle in ("Tokenizer(", "from_file(", "from_pretrained(", "import tokenizers"):
            if needle in source:
                offenders.append(f"{path.name}: {needle}")

    assert not offenders, "tokenizers are loaded in one place: " + "; ".join(offenders)


def _integer_constants(path: Path) -> Iterator[tuple[int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if not isinstance(node.value, bool):
                yield node.lineno, node.value


def test_no_pinned_window_length_appears_as_a_literal_in_the_source_tree(
    committed: pins.Pins,
) -> None:
    """The window is computed per baseline from its own pin, or it is not per baseline at all.

    Both the pinned length and the content window it implies are scanned: writing 510 into the
    code is the same defect as writing 512, one subtraction later.
    """
    forbidden: set[int] = set()
    for baseline in committed.baselines:
        forbidden.add(baseline.window.length)
        forbidden.add(baseline.window.length - FIXTURE_FRAME)
    assert forbidden, "the scan found nothing to look for, which would pass vacuously"

    offenders: list[str] = []
    for path in _source_files():
        for lineno, value in _integer_constants(path):
            if value in forbidden:
                offenders.append(f"{path.name}:{lineno} {value}")

    assert not offenders, "a window length comes from pins.toml: " + "; ".join(offenders)


def test_the_window_policy_module_does_not_import_the_inference_runtime() -> None:
    """The policy is read, tested and applied without starting an inference runtime."""
    code = "import sys, nbc.baselines.tokenization; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout


def test_the_window_abort_has_its_own_exit_code() -> None:
    assert issubclass(WindowPolicyInvalid, NbcError)
    assert declared_exit_codes()[WindowPolicyInvalid.exit_code] is WindowPolicyInvalid


def test_the_window_abort_reports_at_least_one_problem() -> None:
    with pytest.raises(ValueError, match="at least one problem"):
        WindowPolicyInvalid()


def test_the_frame_wraps_what_it_is_given() -> None:
    frame = SpecialTokenFrame(prefix=(1,), suffix=(2, 3))

    assert frame.size == 3
    assert frame.wrap((7, 8)) == (1, 7, 8, 2, 3)


# --- the live half -----------------------------------------------------------------------------


@pytest.mark.smoke
def test_every_pinned_window_is_the_one_its_config_declares() -> None:
    """AD-19's source of truth, checked against the artifact rather than against this file.

    A few kilobytes per baseline, and it is the only thing that makes `window.length` a reading
    rather than a number someone typed.
    """
    import urllib.request

    for baseline in pins.load_pins().baselines:
        url = (
            f"https://huggingface.co/{baseline.repository}/resolve/"
            f"{baseline.revision}/{baseline.config_path}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            config = json.loads(response.read().decode("utf-8"))

        assert config["max_position_embeddings"] == baseline.window.length, baseline.key
        assert baseline.window.source.startswith(f"{baseline.config_path}::")


@pytest.mark.smoke
def test_one_pinned_repository_ships_two_tokenizers_that_disagree_about_truncation() -> None:
    """The reason the pin names a *path*, read off the artifacts rather than off a document.

    Measured at the pinned shas on 2026-08-29 and **not** what this project's architecture note
    says: the disagreement is *inside* one repository, not between the two baselines. protectai
    publishes `tokenizer.json` twice at one revision -- at the root, declaring
    `truncation.max_length`, and beside the ONNX graph, declaring `null` -- and the pin names the
    second. Both baselines' *pinned* files therefore declare no truncation today.

    That does not soften `no_truncation()`; it relocates the trap. A pin that named only the
    repository, or a loader that resolved the file by convention, takes the root file and
    truncates every long document at 512 while every code test stays green. A pinned revision's
    files never change, so this inequality holds for as long as these shas are the pins.
    """
    declared: dict[str, object] = {}
    for baseline in pins.load_pins().baselines:
        pinned_path = Path(baseline.tokenizer_path)
        if pinned_path.parent == Path("."):
            continue
        for path in (baseline.tokenizer_path, pinned_path.name):
            declared[path] = _declared_truncation(
                _head_of(baseline.repository, baseline.revision, path), path
            )

    assert declared, "no pinned baseline keeps its tokenizer beside the graph any more"
    assert len(set(map(repr, declared.values()))) > 1, (
        f"the two files one repository ships at one revision now agree about truncation "
        f"({declared}); that agreement is why the pin names a path, so re-read the pins before "
        f"concluding the risk is gone"
    )


@pytest.mark.smoke
def test_every_pinned_tokenizer_loads_neutralized_and_windows_a_long_document(
    tmp_path: Path,
) -> None:
    """The AC's "every baseline", over the real files, once -- the half no fixture can carry.

    About 9 MB over both repositories, and it is the only place the shared loader, the measured
    frame and the content window ever meet a pinned artifact. Everything above proves the policy
    behaves; this proves it has the subjects it was written for.
    """
    observed: dict[str, tuple[int, int, int]] = {}
    for baseline in pins.load_pins().baselines:
        path = tmp_path / baseline.key / Path(baseline.tokenizer_path).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_fetch(baseline.repository, baseline.revision, baseline.tokenizer_path))

        policy = tokenization.WindowedTokenizer(
            key=baseline.key,
            tokenizer=load_tokenizer(path, baseline=baseline.key),
            window=baseline.window,
        )
        long_document = "word " * (4 * policy.content_length)
        windows = policy.windows([long_document])[0]

        assert policy.tokenizer.truncation is None, baseline.key
        assert policy.tokenizer.padding is None, baseline.key
        assert policy.content_length == baseline.window.length - policy.num_special_tokens
        assert len(windows) > 1, baseline.key
        assert max(len(window) for window in windows) <= baseline.window.length
        observed[baseline.key] = (
            policy.num_special_tokens,
            policy.content_length,
            len(windows),
        )

    assert len(observed) == len(pins.load_pins().baselines), observed


def _fetch(repository: str, revision: str, path: str) -> bytes:
    import urllib.request

    url = f"https://huggingface.co/{repository}/resolve/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return bytes(response.read())


def _head_of(repository: str, revision: str, path: str) -> str:
    """The first kilobytes of a file, which is where `tokenizer.json` declares truncation."""
    import urllib.request

    url = f"https://huggingface.co/{repository}/resolve/{revision}/{path}"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-16383"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def _declared_truncation(prefix: str, key: str) -> object:
    """The `truncation` value from the head of a `tokenizer.json`, parsed where it starts."""
    marker = '"truncation"'
    start = prefix.find(marker)
    assert start >= 0, f"{key}: no truncation field in the first bytes of the file"
    colon = prefix.index(":", start + len(marker))
    value, _ = json.JSONDecoder().raw_decode(prefix[colon + 1 :].lstrip())
    return value
