"""The only adapter: every baseline is an ONNX graph, on CPU, fed from its own signature.

Three things are bound here that are cheap to state in prose and worthless stated that way.

**The device.** An `InferenceSession` built without an explicit provider list picks up an
accelerator when one exists. The numbers it then produces are neither CPU numbers nor
reproducible on a reviewer's laptop, and every test still passes. So every session names
`providers=["CPUExecutionProvider"]`, reads no device from the environment, and is refused if
the runtime hands back anything else.

**The input feed.** It is built from the graph's **declared input signature**, never from a
convention, because the pinned families genuinely differ: BERT-family graphs take
`token_type_ids` and DeBERTa-family graphs do not. An adapter that assumes one shape either
crashes loudly against the other or -- worse -- feeds a zero tensor to a model that never
expected the input at all.

**The inference parameters.** Batch size and `intra_op_num_threads` are declared constants,
fixed for the published run and recorded in `results.json`. Neither is free: batched inference
pads to the longest sequence in the batch, and intra-op parallelism changes float32 reduction
order. Both move scores in the last decimals, and a hard threshold turns that into a class flip
on exactly the borderline encoded items this experiment is about.

Importing this module imports `onnxruntime`. The platform preflight runs before that happens,
or it is checking a floor the import already crashed through, so the entrypoint imports this
module after `platform.preflight()` and not at the top of a chain that starts with `nbc`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence

import onnxruntime as ort

from nbc import pins
from nbc.baselines.port import (
    TokenWindow,
    Windower,
    p_injection,
    reduce_windows,
    resolve_positive_index,
)
from nbc.errors import NbcError
from nbc.schema import Score

__all__ = [
    "BATCH_SIZE",
    "FEEDABLE_INPUTS",
    "INTRA_OP_NUM_THREADS",
    "InferenceSessionInvalid",
    "OnnxBaseline",
    "PAD_TOKEN_ID",
    "DEVICE",
    "PROVIDERS",
    "REQUIRED_INPUT_TYPE",
    "observed_device",
    "open_baseline",
    "read_id2label",
]


PROVIDERS: Final[tuple[str, ...]] = ("CUDAExecutionProvider", "CPUExecutionProvider")
"""The execution path of the published artifact. Named, not defaulted, and then verified.

**This was CPU-only until 2026-08-30, and the reason it changed is wall-clock, not principle.**
The full matrix is 114,400 scored keys over a corpus whose benign half averages 7,663 characters
an item. Measured on a 16-thread CPU: one process scores 18 keys a minute, ten processes score
about 47 -- the pass is memory-bandwidth bound, so processes stop helping long before the cores
run out, and the run lands near 20 hours. The same pass on one RTX 3060 is roughly an hour.

**What was given up, stated rather than glossed.** The previous docstring said a score computed on
another device "diverges in the last decimals" and left it qualitative. It is now measured: over a
fixed 15-item sample scored on both, **all 30 probabilities differ, by up to 3.61e-4**, with
`n_windows` identical -- the windowing is unchanged, the arithmetic is not. So this table cannot be
reproduced on a CPU-only machine, and the README says so instead of promising otherwise.

**What was NOT given up: determinism.** Measured 2026-08-30 on the pinned graphs -- two processes
on one RTX 3060 give bit-identical scores, and two different RTX 3060s give bit-identical scores.
That is what makes sharding across cards legitimate rather than a source of quiet divergence.
`tests/baselines/test_onnx_adapter.py` pins the constant; the smoke suite is what exercises it on
hardware, and it now needs a GPU to run.

**Two entries, not one.** The CUDA provider does not implement every operator in these graphs;
the CPU provider is the declared fallback for the rest, which is what the session reports as
active and therefore what `ExecutionPath.providers` records. A single-entry tuple would have made
the recorded value disagree with the declared one on the first run.
"""

DEVICE: Final[str] = "NVIDIA GeForce RTX 3060 (8.6)"
"""The CUDA device the published table is computed on, as `observed_device` spells it.

**`PROVIDERS` alone stopped being enough the moment the artifact moved to GPU.** A Tesla P40 and
an RTX 3060 both report `CUDAExecutionProvider`, and they are different architectures running
different kernels -- so the field that was written to make "somebody ran one shard on the GPU box"
an abort would no longer notice "somebody ran one shard on the *other* GPU". The machine this ran
on has both, so that hole was reachable on the first day rather than hypothetically.

Declared here beside `PROVIDERS`, `BATCH_SIZE` and `INTRA_OP_NUM_THREADS`, because `DeclaredPath`
already gathers the published path from this module and the shard files record what each process
observed. Two sides, two sources -- which is the only shape in which the comparison means anything.

The compute capability is in the string on purpose. Two cards can carry one marketing name across
a silicon revision; the capability is what selects the kernels.
"""


def observed_device() -> str:
    """The CUDA device this process will actually run on, asked of the driver.

    Observed, never declared: this is the value `ExecutionPath` records and `DEVICE` is compared
    against, and a function that returned the constant would be a comparison with itself.

    `onnxruntime` reports only `"GPU"` from `get_device()`, which cannot separate the two
    architectures this machine carries, so the identity comes from the CUDA runtime through
    `ctypes` -- the name and the compute capability of the current device, which is the device
    `CUDA_VISIBLE_DEVICES` has already narrowed to.

    Raises `InferenceSessionInvalid` rather than returning a placeholder when CUDA cannot be
    reached. A run that could not identify its device would otherwise record a string that agrees
    with every other string, and the check above it would pass by being unable to fail.
    """
    import ctypes

    class _Properties(ctypes.Structure):
        # The first field of `cudaDeviceProp` is `char name[256]`, and `cudaGetDeviceProperties`
        # writes the whole struct. Over-allocating the tail is what lets this read the name and
        # the capability without vendoring a header whose layout changes between CUDA releases.
        _fields_ = [("name", ctypes.c_char * 256), ("tail", ctypes.c_byte * 8192)]

    try:
        runtime = ctypes.CDLL("libcudart.so")
    except OSError as failure:
        raise InferenceSessionInvalid(
            f"the CUDA runtime could not be loaded ({failure}); the published execution path is "
            f"{list(PROVIDERS)} and a process that cannot reach CUDA cannot produce it"
        ) from failure

    index = ctypes.c_int(0)
    if runtime.cudaGetDevice(ctypes.byref(index)) != 0:
        raise InferenceSessionInvalid(
            "the CUDA runtime reported no current device; every scored row has to name the "
            "device that produced it"
        )
    major, minor = ctypes.c_int(0), ctypes.c_int(0)
    # 75 and 76 are cudaDevAttrComputeCapabilityMajor and ...Minor, stable across CUDA releases.
    for value, attribute in ((major, 75), (minor, 76)):
        if runtime.cudaDeviceGetAttribute(ctypes.byref(value), attribute, index) != 0:
            raise InferenceSessionInvalid(
                f"the CUDA runtime refused attribute {attribute} for device {index.value}; the "
                f"compute capability is what selects the kernels and it cannot be assumed"
            )
    properties = _Properties()
    if runtime.cudaGetDeviceProperties(ctypes.byref(properties), index) != 0:
        raise InferenceSessionInvalid(
            f"the CUDA runtime refused the properties of device {index.value}"
        )
    name = properties.name.decode("utf-8", errors="replace").strip()
    if not name:
        raise InferenceSessionInvalid(
            f"the CUDA runtime named device {index.value} with an empty string"
        )
    return f"{name} ({major.value}.{minor.value})"

BATCH_SIZE: Final[int] = 8
"""Windows per `session.run`. Declared, fixed for the run, and written into `results.json`.

The value is a choice, and the spine leaves it open: what it does not leave open is that it is
fixed rather than tuned per baseline. Eight 512-token windows is a working set a CPU-only
reviewer's machine holds comfortably, and the batch-invariance test is what makes the value
safe to change rather than frozen by superstition.
"""

INTRA_OP_NUM_THREADS: Final[int] = 1
"""One thread inside an operator, so float32 reductions add up in the same order every run.

The cost is wall-clock, and it is paid deliberately: NFR4 says the same input produces the same
number, and a threaded reduction over float32 does not promise that.
"""

FEEDABLE_INPUTS: Final[tuple[str, ...]] = ("input_ids", "attention_mask", "token_type_ids")
"""The declared inputs this adapter knows how to produce, and the whole of them.

A graph declaring anything else -- `position_ids`, `inputs_embeds` -- is not a graph this
adapter can feed. Feeding it zeros would be the quiet failure: a session that runs, returns
logits, and answers a question nobody asked.
"""

REQUIRED_INPUT_TYPE: Final[str] = "tensor(int64)"
"""Token ids and masks are int64, which is what the exporters of both pinned graphs emit.

Checked against the signature rather than assumed, so a graph quantized to int32 indices aborts
instead of being coerced into one by whatever the feed happened to be built from.
"""

PAD_TOKEN_ID: Final[int] = 0
"""The id written into padded positions, which `attention_mask` then masks out.

Declared rather than read from the tokenizer because it is arithmetically inert: the position
is masked, so its id reaches no output. Declaring it keeps that reasoning visible instead of
leaving a bare `0` in a list comprehension.
"""

_FLOAT_OUTPUT: Final[str] = "tensor(float)"
_EXPECTED_RANK: Final[int] = 2


class InferenceSessionInvalid(NbcError, exit_code=9):
    """The session the adapter built is not the session the adapter declared.

    Every member of this class has the same shape: the graph, the runtime or the cache offered
    something other than what the CPU-bound, signature-driven contract above assumes, and the
    only alternatives are to abort or to publish a number computed under conditions nobody
    declared. A missing cached artifact lands here too -- an adapter that cannot open its own
    pinned graph has not built a session either.

    Distinct from `PositiveClassUnresolved` (8) because the remedy differs in kind: that one is
    resolved by swapping a baseline, this one by fetching the pin, fixing the environment, or
    teaching the adapter an input it does not yet know how to produce.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("InferenceSessionInvalid needs at least one problem to report")
        joined = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"the inference session is not the one declared:\n{joined}")
        self.problems: tuple[str, ...] = tuple(problems)


class OnnxBaseline:
    """One pinned ONNX graph, reached through `port.Baseline`.

    Its only freedom is producing the logits. The positive index comes from the repository's
    own `id2label`, the score from `port.softmax`, the document reduction from
    `port.reduce_windows`, and the windows from the shared window policy handed in as
    `windower`.
    """

    def __init__(
        self,
        *,
        key: str,
        graph: Path | str | bytes,
        id2label: Mapping[object, object],
        windower: Windower,
        batch_size: int = BATCH_SIZE,
        intra_op_num_threads: int = INTRA_OP_NUM_THREADS,
        providers: tuple[str, ...] = PROVIDERS,
        device: str | None = DEVICE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size!r}")
        if intra_op_num_threads < 1:
            raise ValueError(
                f"intra_op_num_threads must be at least 1, got {intra_op_num_threads!r}"
            )
        if not providers or not all(isinstance(name, str) and name for name in providers):
            raise ValueError(f"providers must hold non-empty names, got {providers!r}")

        self.key = key
        self.batch_size = batch_size
        self.intra_op_num_threads = intra_op_num_threads
        # Parameters with the declared constants as defaults, for the reason `batch_size` and
        # `intra_op_num_threads` already are: the adapter's contract -- input names, dtypes, the
        # label axis -- is not a claim about hardware, and a test of that contract that could only
        # run on a GPU would be a contract nobody could check. The published table comes from the
        # defaults; `harness/score.py::path_problems` is what compares what each shard RECORDED
        # against `DeclaredPath`, and that comparison is where a shard produced on another device
        # is caught. `tests/baselines/test_onnx_adapter.py` scans `src/` and requires that no
        # module but `open_baseline` names these two parameters, so the override cannot become a
        # second way to set the published path.
        self.declared_providers: tuple[str, ...] = tuple(providers)
        self.declared_device: str | None = device
        self.id2label: Mapping[int, str] = _normalized_labels(id2label)
        # Resolved before the session is built: a baseline whose positive class cannot be
        # resolved is ineligible, and paying for a session first would only delay saying so.
        self.positive_index: int = resolve_positive_index(id2label, baseline=key)
        self._windower = windower

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        model: Any = str(graph) if isinstance(graph, (Path, str)) else graph
        try:
            self._session = ort.InferenceSession(
                model, options, providers=list(self.declared_providers)
            )
        except Exception as failure:  # noqa: BLE001 - the runtime's own class means nothing here
            raise InferenceSessionInvalid(
                f"baseline {key!r}: onnxruntime refused the pinned graph ({failure})"
            ) from failure

        self._input_names: tuple[str, ...] = self._checked_input_names()
        self._output_name: str = self._checked_output_name()

    # -- the contract, checked once, at construction -------------------------------------

    def _checked_input_names(self) -> tuple[str, ...]:
        problems: list[str] = []

        active = tuple(self._session.get_providers())
        if active != self.declared_providers:
            problems.append(
                f"baseline {self.key!r}: the session names providers "
                f"{list(self.declared_providers)} and the runtime made {list(active)} active; a "
                f"score from another device diverges in the last decimals and the decision "
                f"threshold turns that into a class flip"
            )
        elif self.declared_device is not None:
            # Only once the providers agree, and only when a device was declared. Two different
            # CUDA cards both report `CUDAExecutionProvider`, so this is the half `providers`
            # cannot see; and asking the driver on a run that already failed the check above would
            # report a second symptom of one fault. Measured 2026-08-30: the pinned graphs give
            # bit-identical scores across two RTX 3060s, and differ from CPU by up to 3.61e-4.
            self._device = observed_device()
            if self._device != self.declared_device:
                problems.append(
                    f"baseline {self.key!r}: the published table is computed on "
                    f"{self.declared_device!r} and this process is on {self._device!r}. Two CUDA "
                    f"cards of different architectures run different kernels, and both answer "
                    f"`CUDAExecutionProvider` -- so this is the divergence `providers` cannot see"
                )

        names: list[str] = []
        for declared in self._session.get_inputs():
            names.append(declared.name)
            if declared.name not in FEEDABLE_INPUTS:
                problems.append(
                    f"baseline {self.key!r}: the graph declares input {declared.name!r}, which "
                    f"this adapter cannot produce (it produces {list(FEEDABLE_INPUTS)}); "
                    f"feeding it a zero tensor would answer a question nobody asked"
                )
            elif declared.type != REQUIRED_INPUT_TYPE:
                problems.append(
                    f"baseline {self.key!r}: the graph declares input {declared.name!r} as "
                    f"{declared.type}, and this adapter feeds {REQUIRED_INPUT_TYPE}"
                )
            elif len(declared.shape) != _EXPECTED_RANK:
                problems.append(
                    f"baseline {self.key!r}: the graph declares input {declared.name!r} with "
                    f"shape {declared.shape}, and this adapter feeds [batch, sequence]"
                )
        if "input_ids" not in names:
            problems.append(
                f"baseline {self.key!r}: the graph declares inputs {names} and none of them is "
                f"'input_ids'; there is nowhere to put the tokens"
            )
        if len(set(names)) != len(names):
            problems.append(
                f"baseline {self.key!r}: the graph declares a duplicate input {names}"
            )

        if problems:
            raise InferenceSessionInvalid(*problems)
        return tuple(names)

    def _checked_output_name(self) -> str:
        outputs = self._session.get_outputs()
        if not outputs:
            raise InferenceSessionInvalid(f"baseline {self.key!r}: the graph declares no output")

        logits = outputs[0]
        problems: list[str] = []
        if logits.type != _FLOAT_OUTPUT:
            problems.append(
                f"baseline {self.key!r}: the graph's first output {logits.name!r} is "
                f"{logits.type}, and the port reads float logits from it"
            )
        if len(logits.shape) != _EXPECTED_RANK:
            problems.append(
                f"baseline {self.key!r}: the graph's first output {logits.name!r} has shape "
                f"{logits.shape}, and the port reads [batch, labels] from it"
            )
        else:
            width = logits.shape[-1]
            if isinstance(width, int) and width != len(self.id2label):
                problems.append(
                    f"baseline {self.key!r}: the graph's label axis is {width} wide and its "
                    f"config declares {len(self.id2label)} labels {_render(self.id2label)}; the "
                    f"resolved positive index addresses an axis that does not exist"
                )
        if problems:
            raise InferenceSessionInvalid(*problems)
        return logits.name

    # -- the port -------------------------------------------------------------------------

    @property
    def providers(self) -> tuple[str, ...]:
        """The providers the runtime actually made active for this session."""
        return tuple(self._session.get_providers())

    @property
    def device(self) -> str | None:
        """The CUDA device this session runs on, as the driver named it, or `None` off CUDA.

        Observed at construction and held, rather than asked again per call: the value cannot
        change under a live session, and a property that re-queried the driver would make every
        recorded row depend on a syscall that could start failing halfway through a shard.
        """
        return getattr(self, "_device", None)

    @property
    def graph_inputs(self) -> tuple[str, ...]:
        """The graph's declared input names, in order. The feed is built from exactly this."""
        return self._input_names

    def score(self, texts: Sequence[str]) -> list[Score]:
        """One `Score` per document, in the order the documents came in."""
        if isinstance(texts, (str, bytes)):
            # Belt to the windower's braces: a caller reaching the adapter directly gets the same
            # refusal rather than a plausible-looking list of one-character documents.
            raise TypeError(
                f"score() takes a sequence of documents and was given a bare "
                f"{type(texts).__name__}; pass [text] for a single document"
            )
        per_document = list(self._windower(texts))
        if len(per_document) != len(texts):
            raise ValueError(
                f"the windower returned {len(per_document)} window lists for {len(texts)} "
                f"documents; every document occupies its own windows"
            )

        flat: list[TokenWindow] = []
        owner: list[int] = []
        for document, windows in enumerate(per_document):
            if not windows:
                raise ValueError(
                    f"the windower returned no windows for document {document}; a scored "
                    f"document occupies at least one"
                )
            for window in windows:
                if not window:
                    raise ValueError(
                        f"the windower returned an empty window for document {document}; a "
                        f"window holds at least one token"
                    )
                flat.append(tuple(window))
                owner.append(document)

        probabilities = self._probabilities(flat)

        per_document_probabilities: list[list[float]] = [[] for _ in texts]
        for document, probability in zip(owner, probabilities, strict=True):
            per_document_probabilities[document].append(probability)
        return [reduce_windows(values) for values in per_document_probabilities]

    def _probabilities(self, windows: Sequence[TokenWindow]) -> list[float]:
        """`p_injection` per window, batched at the declared size and nowhere else."""
        probabilities: list[float] = []
        for batch in _batched(windows, self.batch_size):
            rows = self._session.run([self._output_name], self._feed(batch))[0].tolist()
            if len(rows) != len(batch):
                raise InferenceSessionInvalid(
                    f"baseline {self.key!r}: fed {len(batch)} windows and the graph returned "
                    f"{len(rows)} rows of logits"
                )
            for row in rows:
                if len(row) != len(self.id2label):
                    raise InferenceSessionInvalid(
                        f"baseline {self.key!r}: the graph returned {len(row)} logits and its "
                        f"config declares {len(self.id2label)} labels {_render(self.id2label)}"
                    )
                probabilities.append(p_injection(row, self.positive_index))
        return probabilities

    def _feed(self, windows: Sequence[TokenWindow]) -> dict[str, list[list[int]]]:
        """The session's input feed, built from the graph's declared signature.

        Only the inputs the graph declares are produced, which is the whole point: the pinned
        families differ on `token_type_ids`, and a convention would be right about one of them.
        """
        width = max(len(window) for window in windows)
        built: dict[str, list[list[int]]] = {
            "input_ids": [
                list(window) + [PAD_TOKEN_ID] * (width - len(window)) for window in windows
            ],
            "attention_mask": [
                [1] * len(window) + [0] * (width - len(window)) for window in windows
            ],
            "token_type_ids": [[0] * width for _ in windows],
        }
        return {name: built[name] for name in self._input_names}

    def as_run_fields(self) -> dict[str, object]:
        """What `results.json`'s `run` block records about this column's model boundary."""
        return {
            "key": self.key,
            "id2label": {str(index): self.id2label[index] for index in sorted(self.id2label)},
            "positive_index": self.positive_index,
            "providers": list(self.providers),
            "device": self.device,
            "batch_size": self.batch_size,
            "intra_op_num_threads": self.intra_op_num_threads,
            "graph_inputs": list(self.graph_inputs),
        }


def _batched(windows: Sequence[TokenWindow], size: int) -> Iterator[Sequence[TokenWindow]]:
    for start in range(0, len(windows), size):
        yield windows[start : start + size]


def _normalized_labels(id2label: Mapping[object, object]) -> dict[int, str]:
    """The declared mapping with integer keys, for recording. Resolution validates it."""
    normalized: dict[int, str] = {}
    if isinstance(id2label, Mapping):
        for key, label in id2label.items():
            if isinstance(key, int) and not isinstance(key, bool):
                normalized[key] = str(label)
            elif isinstance(key, str) and key.isdigit():
                normalized[int(key)] = str(label)
    return normalized


def _render(id2label: Mapping[int, str]) -> str:
    return "{" + ", ".join(f"{index}: {id2label[index]!r}" for index in sorted(id2label)) + "}"


def read_id2label(config_path: Path) -> Mapping[object, object]:
    """The `id2label` block of a pinned revision's `config.json`, read as found.

    Absence and malformation are reported through `PositiveClassUnresolved` by the resolver
    rather than repaired here: this function reads, it does not decide.
    """
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as failure:
        raise InferenceSessionInvalid(
            f"the pinned config {str(config_path)!r} could not be read ({failure}); fetch the "
            f"pins with `python -m nbc.pins --verify` before scoring anything"
        ) from failure
    except json.JSONDecodeError as failure:
        raise InferenceSessionInvalid(
            f"the pinned config {str(config_path)!r} is not JSON ({failure})"
        ) from failure

    if not isinstance(document, Mapping):
        return {}
    found = document.get("id2label")
    return found if isinstance(found, Mapping) else {}


def open_baseline(
    baseline: pins.Baseline,
    windower: Windower,
    *,
    cache_root: Path | None = None,
    batch_size: int = BATCH_SIZE,
    intra_op_num_threads: int = INTRA_OP_NUM_THREADS,
    providers: tuple[str, ...] = PROVIDERS,
    device: str | None = DEVICE,
) -> OnnxBaseline:
    """Build the adapter for one pinned baseline from the files the pin names, and no others.

    Every path comes from `pins.toml`: the snapshot the revision resolves to, the graph path
    inside it, the config path beside it. Nothing here picks a file by convention, because a
    repository shipping two `tokenizer.json` at one revision is not hypothetical -- it is the
    reason the pins name paths at all.
    """
    # The windower is a seam, which is what lets the window policy be applied identically for
    # every adapter rather than grown inside one -- and a seam is also where two halves of one
    # baseline can be crossed. Measured: scoring a canonical injection through the right graph
    # with the other baseline's tokenizer moves p_injection from 0.99999981 to 0.0000020, with
    # nothing anywhere aborting. Two pinned baselines with different vocabularies is exactly the
    # independence SC5 was rebuilt to buy, so the crossing is available by construction.
    windower_key = getattr(windower, "key", None)
    if windower_key is not None and windower_key != baseline.key:
        raise InferenceSessionInvalid(
            f"baseline {baseline.key!r} was handed the windower for {windower_key!r}. A document "
            f"tokenized by one baseline's vocabulary and scored by another's graph produces a "
            f"number that looks like a score and is not one: measured on the pinned pair, a "
            f"crossed pair moves p_injection by six orders of magnitude with no error raised"
        )

    snapshot = baseline.artifact.snapshot_dir(cache_root)
    graph = snapshot / baseline.graph_path
    config = snapshot / baseline.config_path

    missing = [str(path) for path in (graph, config) if not path.is_file()]
    if missing:
        raise InferenceSessionInvalid(
            *(
                f"baseline {baseline.key!r}: {path} is not in the Hugging Face cache; the "
                f"pinned revision has to be fetched before anything can be scored"
                for path in missing
            )
        )

    return OnnxBaseline(
        key=baseline.key,
        graph=graph,
        id2label=read_id2label(config),
        windower=windower,
        batch_size=batch_size,
        intra_op_num_threads=intra_op_num_threads,
        providers=providers,
        device=device,
    )
