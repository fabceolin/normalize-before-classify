"""The only place a tokenizer is loaded, and the one window policy every baseline is scored under.

Encoded text tokenizes into several times the tokens of its decoded form -- measured over 400 real
attack payloads with the pinned protectai tokenizer, the median is 21 tokens clean and 96 base64,
and 55 of 400 encoded payloads pass 510 tokens against 19 of 400 canonicalized. Every baseline
caps its sequence length. So how a long document is handled *is* the measurement, on exactly the
items the experiment is about: one story truncates at the cap, another windows and takes the
maximum, and the two report different numbers from the same model and the same layer.

Three failures are removed here rather than described.

**The inherited setting.** `Tokenizer.from_file` restores whatever the file declares, and what a
published file declares cannot be assumed. Read at the pinned shas on 2026-08-29: one pinned
repository ships `tokenizer.json` **twice** at one revision -- at the root, declaring
`"truncation": {"max_length": 512}`, and beside the ONNX graph, declaring `"truncation": null` --
and the pin names the second, so both baselines' pinned files happen to declare none today. That
is a fact about two shas, not a property of the ecosystem: a pin that named the repository instead
of the path, or a loader resolving the file by convention, takes the root file and silently
truncates every long document at 512 before windowing can fire, while the other baseline windows
it -- reinstating this module's confound through *data configuration* rather than through code,
with every code test still green. A third repository, since dropped, carried 2049, an off-by-one
against its own published window. So nothing about the field is assumed. `load_tokenizer` calls
`no_truncation()` and `no_padding()` immediately after every load, verifies both are off, and every
length decision after that is ours.

**The window length's origin.** It is the pinned baseline's own `window.length`, whose `source`
`pins.toml` records and whose reading a human confirmed at the pinned revision. It is never read
from `tokenizer_config.json`, whose `model_max_length` is a ~1e30 sentinel in both pinned
repositories, and it is never written as a literal here: this module receives a `WindowPin`, not
an integer, so a length can only arrive from the file that pins it.

**The special-token count.** A window has to carry the same frame the tokenizer would have added
itself, and the content window is `max_len - num_special_tokens` -- measured at 2 and 510 for both
pinned baselines, so a full window is exactly `max_len`. The two frames are *not* the same tokens:
the pinned pair wraps with ids `(1, 2)` and `(101, 102)` respectively, so a module that named the
frame instead of measuring it would corrupt every window of one baseline while passing every test
written against the other. The frame is therefore measured -- two probes encoded with and without
special tokens, the prefix and suffix cross-checked against each other and against the tokenizer's
own `num_special_tokens_to_add`. An off-by-two here moves every window boundary in the run.

The policy itself is AD-29's declared strategy -- a length, a stride and an aggregation together --
selected by name from `pins.toml`. `shared` is the only one, and `window_policy` is nonetheless
part of the cell key from the first run, because a key retro-fitted into a published envelope is a
schema break while a key that was always there costs a constant.

**Known limitation, carried into FR19 caveat 5.** Non-overlapping windows can split a decoded
instruction across a boundary so that no window sees the whole of it. Measured, that exposure is
small and points the safe way: the *encoded* condition is the one that spills into extra windows,
and max-over-windows hands the extra chances to the un-canonicalized route -- a bias that runs
against this project's thesis, which is the safe direction, but only while every column takes it.

This module imports no inference runtime. `tokenizers` is a pinned dependency and `onnxruntime` is
not imported here, so the window policy can be read, tested and applied without starting one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

from tokenizers import Tokenizer

from nbc import pins
from nbc.baselines.port import REDUCTION_NAME, TokenWindow
from nbc.errors import NbcError

__all__ = [
    "FRAME_PROBES",
    "SHARED",
    "SpecialTokenFrame",
    "WINDOW_POLICIES",
    "WindowPolicy",
    "WindowPolicyInvalid",
    "WindowedTokenizer",
    "derive_frame",
    "load_tokenizer",
    "open_windower",
]


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    """A length policy as AD-29 defines one: a stride and an aggregation, under a declared name.

    The window *length* is not a field here because it is per baseline and comes from `pins.toml`.
    What a policy fixes is the shape: how far the window advances between slices, and how the
    per-window scores become one document score.
    """

    name: str
    stride: int
    aggregation: str

    def as_run_fields(self) -> dict[str, object]:
        return {"name": self.name, "stride": self.stride, "aggregation": self.aggregation}


SHARED: Final[WindowPolicy] = WindowPolicy(
    name=pins.SHARED_WINDOW_POLICY, stride=0, aggregation=REDUCTION_NAME
)
"""AD-19's policy, and the one every baseline is scored under.

`stride = 0` is the declaration that the windows do not overlap: each one starts where the last
ended. `aggregation` is read from `port.REDUCTION_NAME` rather than spelled again, so the name
recorded in `results.json` and the reduction `port.reduce_windows` performs cannot drift apart.
"""

WINDOW_POLICIES: Final[Mapping[str, WindowPolicy]] = {SHARED.name: SHARED}
"""Every policy that can actually run, keyed by the name a pin selects it with."""

_unrunnable = sorted(pins.WINDOW_POLICIES - set(WINDOW_POLICIES))
if _unrunnable:  # pragma: no cover - the table is checked as it is defined
    raise ValueError(
        f"pins.toml admits window policies {_unrunnable} that no strategy here implements; a "
        f"policy name with nothing behind it selects whatever the fallback happened to be, for "
        f"every document the baseline that declares it ever scores"
    )

_unpinnable = sorted(set(WINDOW_POLICIES) - pins.WINDOW_POLICIES)
if _unpinnable:  # pragma: no cover - the table is checked as it is defined
    raise ValueError(
        f"window policies {_unpinnable} are implemented here and cannot be declared in "
        f"pins.toml; a strategy no pin can select is a strategy nothing runs"
    )


FRAME_PROBES: Final[tuple[str, ...]] = (
    "an ordinary sentence, tokenized to find the frame",
    "another, deliberately unlike the first",
)
"""The texts whose encodings reveal the tokenizer's own special-token frame.

Two, not one, and unalike: one probe would let a template that happens to repeat the content --
or a normalizer that injects a token near a boundary -- pass as a clean prefix/suffix wrap. Two
probes have to agree on the same frame before it is used, and the agreed frame is then checked
against the tokenizer's own `num_special_tokens_to_add`.
"""


class WindowPolicyInvalid(NbcError, exit_code=10):
    """The length handling that would be applied is not the one this project declares.

    Every member has the same consequence: the run would either not start or would score
    documents under a window policy nobody declared, which is the single confound AD-19 exists to
    remove. A pinned tokenizer that is not on this machine, a truncation setting that survived
    neutralization, a policy name with no strategy behind it, a special-token template this policy
    cannot apply, a window that came out longer than the model's own capacity -- all of them end
    with a table whose two columns are no longer comparable.

    Distinct from `InferenceSessionInvalid` (9) because that one is about the session an adapter
    built; this one is about the tokens fed into it, and it is raised by a module that never
    imports an inference runtime. Distinct from `PinsFileInvalid` (4) because the file can be
    perfectly well formed and the artifact it names still absent or unusable.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("WindowPolicyInvalid needs at least one problem to report")
        joined = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"the window policy is not the one declared:\n{joined}")
        self.problems: tuple[str, ...] = tuple(problems)


@dataclass(frozen=True, slots=True)
class SpecialTokenFrame:
    """The token ids a tokenizer wraps a single sequence in, measured rather than named."""

    prefix: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def wrap(self, content: Sequence[int]) -> TokenWindow:
        return (*self.prefix, *content, *self.suffix)


def load_tokenizer(path: Path, *, baseline: str) -> Tokenizer:
    """The pinned tokenizer, with its inherited truncation and padding removed. The only loader.

    The neutralization is the whole point of routing every load through here, and it is verified
    rather than trusted: a future `tokenizers` that renamed one of these calls would otherwise
    leave the confound in place while this function still looked like it removed it.
    """
    if not path.is_file():
        raise WindowPolicyInvalid(
            f"baseline {baseline!r}: {path} is not in the Hugging Face cache; the pinned "
            f"revision has to be fetched before any document can be windowed"
        )
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as failure:  # noqa: BLE001 - the library's own class means nothing here
        raise WindowPolicyInvalid(
            f"baseline {baseline!r}: {path} could not be loaded as a tokenizer ({failure})"
        ) from failure

    tokenizer.no_truncation()
    tokenizer.no_padding()

    problems: list[str] = []
    if tokenizer.truncation is not None:
        problems.append(
            f"baseline {baseline!r}: truncation survived no_truncation() as "
            f"{tokenizer.truncation!r}; a tokenizer that truncates cuts every long document "
            f"before this policy can window it, and the pinned files disagree about it"
        )
    if tokenizer.padding is not None:
        problems.append(
            f"baseline {baseline!r}: padding survived no_padding() as {tokenizer.padding!r}; "
            f"padding applied at encode time puts tokens in the content axis that the document "
            f"does not contain, and the adapter pads its own batches"
        )
    if problems:
        raise WindowPolicyInvalid(*problems)
    return tokenizer


def derive_frame(tokenizer: Tokenizer, *, baseline: str) -> SpecialTokenFrame:
    """The tokenizer's own single-sequence special-token frame, measured on two probes.

    Every probe is encoded twice, with and without special tokens; the frames consistent with both
    encodings are intersected across probes. Exactly one survivor is required, and its size must
    equal what the tokenizer says it adds. Anything else means this policy cannot wrap a window
    the way this tokenizer would have, and guessing is what moves every boundary in the run.
    """
    declared = int(tokenizer.num_special_tokens_to_add(False))

    candidates: set[tuple[tuple[int, ...], tuple[int, ...]]] | None = None
    for probe in FRAME_PROBES:
        bare = tuple(tokenizer.encode(probe, add_special_tokens=False).ids)
        framed = tuple(tokenizer.encode(probe, add_special_tokens=True).ids)
        if not bare:
            raise WindowPolicyInvalid(
                f"baseline {baseline!r}: the frame probe {probe!r} encodes to no content tokens, "
                f"so it cannot show where the special tokens sit"
            )
        found = {
            (framed[:start], framed[start + len(bare) :])
            for start in range(len(framed) - len(bare) + 1)
            if framed[start : start + len(bare)] == bare
        }
        candidates = found if candidates is None else candidates & found

    if not candidates:
        raise WindowPolicyInvalid(
            f"baseline {baseline!r}: encoding the frame probes with and without special tokens "
            f"yields no single prefix/suffix frame they agree on. This policy wraps every window "
            f"in the tokenizer's own frame, and a template that is not a wrap is one it cannot "
            f"apply"
        )
    if len(candidates) > 1:
        raise WindowPolicyInvalid(
            f"baseline {baseline!r}: the frame probes admit {len(candidates)} different "
            f"prefix/suffix frames {sorted(candidates)}; the frame has to be determined by the "
            f"tokenizer, not chosen from the possibilities"
        )

    prefix, suffix = candidates.pop()
    frame = SpecialTokenFrame(prefix=prefix, suffix=suffix)
    if frame.size != declared:
        raise WindowPolicyInvalid(
            f"baseline {baseline!r}: the measured frame adds {frame.size} tokens "
            f"({prefix} ... {suffix}) and the tokenizer declares it adds {declared}. The content "
            f"window is the model's window minus that count, so a disagreement here moves every "
            f"window boundary in the run"
        )
    return frame


class WindowedTokenizer:
    """One tokenizer under one policy: the `port.Windower` every adapter is handed.

    Constructed from a `pins.WindowPin` rather than an integer on purpose. The window length is a
    pinned, human-confirmed reading of one named file, and accepting a bare number here is exactly
    how it would come to be written somewhere else.
    """

    def __init__(
        self,
        *,
        key: str,
        tokenizer: Tokenizer,
        window: pins.WindowPin,
        policy: WindowPolicy = SHARED,
    ) -> None:
        if policy.name not in WINDOW_POLICIES:
            raise WindowPolicyInvalid(
                f"baseline {key!r}: window policy {policy.name!r} is not one of "
                f"{sorted(WINDOW_POLICIES)}"
            )
        if window.length <= 0:
            raise WindowPolicyInvalid(
                f"baseline {key!r}: the pinned window is {window.length}, and a window holds at "
                f"least one token"
            )
        if policy.stride != SHARED.stride:
            # A policy is a length, a stride and an aggregation *together*, and the three are
            # written into `results.json`. Overlapping windows are not implemented -- so a
            # non-zero stride must abort here rather than be recorded as a parameter of a run
            # that walked the document without it.
            raise WindowPolicyInvalid(
                f"baseline {key!r}: policy {policy.name!r} declares stride {policy.stride} and "
                f"only a stride of {SHARED.stride} -- fixed non-overlapping windows -- is "
                f"implemented. A stride recorded in the results and ignored by the walk is a "
                f"published parameter the run never applied"
            )

        self.key = key
        self.policy = policy
        self.window = window
        self._tokenizer = tokenizer
        self.frame: SpecialTokenFrame = derive_frame(tokenizer, baseline=key)
        self.num_special_tokens: int = self.frame.size
        self.content_length: int = window.length - self.num_special_tokens

        if self.content_length < 1:
            raise WindowPolicyInvalid(
                f"baseline {key!r}: its pinned window is {window.length} tokens "
                f"({window.source}) and its tokenizer adds {self.num_special_tokens} special "
                f"tokens, leaving {self.content_length} for content. A window that holds no "
                f"content cannot carry a document"
            )

    @property
    def tokenizer(self) -> Tokenizer:
        """The loaded tokenizer, already neutralized. Read-only: the load path is the only one."""
        return self._tokenizer

    @property
    def max_length(self) -> int:
        """The model's own window: no encoded window this object emits is longer."""
        return self.window.length

    def windows(self, texts: Sequence[str]) -> list[list[TokenWindow]]:
        """The windows each document occupies, in order, one inner list per document.

        A document shorter than one window occupies exactly one; so does an empty one, which is
        then the frame alone. That is what makes `n_windows` a count of windows rather than a
        count of overflows, and it is what the adapter's "at least one window" contract needs.
        """
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        # Nothing about a document survives this call: `n_windows` rides on `Score`, and two
        # identical documents window identically wherever they sit in the corpus.
        return [self._windows_of(tuple(encoding.ids)) for encoding in encodings]

    __call__ = windows

    def _windows_of(self, content: tuple[int, ...]) -> list[TokenWindow]:
        windows = [self.frame.wrap(chunk) for chunk in _chunks(content, self.content_length)]
        if not windows[0]:
            raise WindowPolicyInvalid(
                f"baseline {self.key!r}: a document with no content tokens and a tokenizer that "
                f"adds no special tokens produces an empty window, and there is nothing there to "
                f"score. Such an item belongs out of the corpus, not into a number"
            )
        oversized = [len(window) for window in windows if len(window) > self.max_length]
        if oversized:  # pragma: no cover - the arithmetic above forbids it; the check is the proof
            raise WindowPolicyInvalid(
                f"baseline {self.key!r}: produced windows of {oversized} tokens against a model "
                f"window of {self.max_length}; the model's positional axis stops there"
            )
        return windows

    def as_run_fields(self) -> dict[str, object]:
        """What `results.json` records about this column's length handling."""
        return {
            "key": self.key,
            "window_policy": self.policy.name,
            "policy": self.policy.as_run_fields(),
            "window": self.window.as_run_fields(),
            "num_special_tokens": self.num_special_tokens,
            "content_length": self.content_length,
        }


def _chunks(content: tuple[int, ...], size: int) -> Iterator[tuple[int, ...]]:
    """Fixed non-overlapping slices, and one empty slice when there is nothing to slice.

    The empty case is not a special case of the policy -- it is the statement that every document
    occupies at least one window, including the one with no content tokens at all.
    """
    if not content:
        yield ()
        return
    for start in range(0, len(content), size):
        yield content[start : start + size]


def open_windower(
    baseline: pins.Baseline,
    *,
    cache_root: Path | None = None,
) -> WindowedTokenizer:
    """The windower for one pinned baseline, from the tokenizer file the pin names, and no other.

    The path comes from `pins.toml`, never from a convention: one pinned repository ships two
    different tokenizer files at one revision, at the repository root and beside the ONNX graph,
    declaring different truncation. A loader convention would pick one of them per reader.
    """
    policy = WINDOW_POLICIES.get(baseline.window_policy)
    if policy is None:
        raise WindowPolicyInvalid(
            f"baseline {baseline.key!r} declares window policy "
            f"{baseline.window_policy!r} and the policies that can run are "
            f"{sorted(WINDOW_POLICIES)}"
        )

    path = baseline.artifact.snapshot_dir(cache_root) / baseline.tokenizer_path
    return WindowedTokenizer(
        key=baseline.key,
        tokenizer=load_tokenizer(path, baseline=baseline.key),
        window=baseline.window,
        policy=policy,
    )
