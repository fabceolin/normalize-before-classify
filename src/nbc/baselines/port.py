"""The one port every baseline is reached through, and the arithmetic every adapter shares.

A difference between two columns of the published table has to be a difference between two
models. It stops being that the moment two adapters are free to disagree about anything else:
one calling the second logit "injection" because that is where the first model put it, one
reading `P(injection)` off a sigmoid while the other reads it off a softmax, one truncating a
long document while the other windows it. Each of those produces a number that looks exactly
like a model difference and is not.

So the freedom is removed rather than documented. This module owns:

- `POSITIVE_CLASS_NAMES`, the single set every adapter resolves positivity against;
- `resolve_positive_index`, which reads the pinned revision's own `id2label` and refuses to
  guess -- zero matches aborts, and **more than one match also aborts**;
- `softmax` and `p_injection`, so the score is one scale across every column;
- `reduce_windows`, the window-to-document maximum, so a long document is reduced the same way
  for every baseline.

An adapter's only remaining freedom is producing the logits.

This module imports no inference runtime. `harness/` and `report/` read the shared arithmetic
from here, and neither should pay for `onnxruntime` to do it.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Final, Mapping, Protocol, Sequence, runtime_checkable

from nbc.errors import NbcError
from nbc.schema import Score

__all__ = [
    "POSITIVE_CLASS_NAMES",
    "REDUCTION_NAME",
    "Baseline",
    "PositiveClassUnresolved",
    "TokenWindow",
    "Windower",
    "p_injection",
    "reduce_windows",
    "resolve_positive_index",
    "softmax",
]


POSITIVE_CLASS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "injection",
        "prompt_injection",
        "prompt-injection",
        "jailbreak",
        "malicious",
        "unsafe",
        "attack",
    }
)
"""The names that denote "this input is an attack", matched case-insensitively.

Both pinned baselines publish `{0: SAFE, 1: INJECTION}`, so today only `injection` does any
work and the rest of the set is dead weight. It is kept short for the same reason it is kept at
all: a baseline is **replaced, never removed**, so the set has to survive a swap under OQ2 to a
repository that spells the class differently -- and every name added to it is a liability,
because a repository publishing two of them aborts rather than picking one.

Case-insensitivity is a property of the matching rule rather than of the current pins. Both
pinned repositories happen to shout their labels; nothing makes the next one do so.

Members are stored casefolded, so the fold happens on the observed label and nowhere else.
"""

_POSITIONALLY_MEANINGLESS: Final[re.Pattern[str]] = re.compile(
    r"\Alabel[_-]?\d+\Z", re.IGNORECASE
)
"""`LABEL_0`, `label-1`: the placeholder a model exports when nobody named its classes.

Such a name carries no meaning beyond the position it already sits at, so admitting one into
`POSITIVE_CLASS_NAMES` would turn "resolved from what the repository declares" back into
"hardcoded index", wearing the costume of a lookup.
"""

_meaningless = sorted(
    name for name in POSITIVE_CLASS_NAMES if _POSITIONALLY_MEANINGLESS.match(name)
)
if _meaningless:  # pragma: no cover - the constant is checked as it is defined
    raise ValueError(
        f"POSITIVE_CLASS_NAMES contains positionally-meaningless {_meaningless}; such a name "
        f"resolves to the index it already is and defeats the whole resolution"
    )

_unfolded = sorted(name for name in POSITIVE_CLASS_NAMES if name != name.casefold())
if _unfolded:  # pragma: no cover - the constant is checked as it is defined
    raise ValueError(
        f"POSITIVE_CLASS_NAMES members must be stored casefolded, got {_unfolded}; the fold "
        f"belongs on the observed label, applied in one place"
    )


REDUCTION_NAME: Final[str] = "max"
"""The name of what `reduce_windows` does, so the recorded run field cannot drift from the code.

A window policy is a length, a stride and an aggregation together (AD-29), and the aggregation is
the part a `results.json` reader has to take on trust. Naming it here, next to the function that
performs it, is what keeps the recorded name and the performed reduction the same thing.
"""


TokenWindow = tuple[int, ...]
"""One window's token ids, already bounded by the window policy. The adapter never tokenizes."""

Windower = Callable[[Sequence[str]], Sequence[Sequence[TokenWindow]]]
"""Documents in, the windows each one occupies out -- one inner sequence per input document.

This is the seam between the window policy (`baselines/tokenization.py`) and the adapters. It
is a parameter rather than an import so that the policy is applied identically for every
adapter and an adapter cannot quietly grow one of its own.
"""


class PositiveClassUnresolved(NbcError, exit_code=8):
    """A baseline's declared label mapping does not resolve to exactly one positive class.

    Zero matches and more than one match are the same failure with opposite shapes: in both,
    the index of `P(injection)` is not determined by what the repository declares, and the only
    ways forward are to guess it or to abort. A guessed index reports the second baseline's
    recall inverted, which looks like a finding.

    The remedy is a `pins.toml` edit -- a baseline that cannot resolve is ineligible under AD-7
    and is **replaced, never removed**, because SC5's floor is two and the run sits on it.
    """


def resolve_positive_index(id2label: Mapping[object, object], *, baseline: str) -> int:
    """The logit-axis index of the positive class, read from `id2label`, never guessed.

    `id2label` is the mapping in the **pinned revision's** `config.json`, whose keys are axis
    positions and whose values are the publisher's own names. JSON gives those keys as strings;
    both spellings are accepted, and nothing else is.

    Raises `PositiveClassUnresolved`, carrying the observed mapping, when the mapping is absent,
    is not an axis (keys other than `0..n-1`), matches no admitted name, or matches two.
    """
    observed = _axis(id2label, baseline=baseline)

    matches = [
        index
        for index, label in observed.items()
        if label.strip().casefold() in POSITIVE_CLASS_NAMES
    ]

    if len(matches) == 1:
        return matches[0]

    admitted = ", ".join(sorted(POSITIVE_CLASS_NAMES))
    if not matches:
        raise PositiveClassUnresolved(
            f"baseline {baseline!r} declares no positive class: none of its labels "
            f"{_render(observed)} is one of the admitted names ({admitted}). A repository whose "
            f"positive class cannot be resolved is ineligible as a baseline and is replaced, "
            f"never removed."
        )
    named = ", ".join(f"{index}={observed[index]!r}" for index in matches)
    raise PositiveClassUnresolved(
        f"baseline {baseline!r} declares {len(matches)} positive classes ({named}) in "
        f"{_render(observed)}; the port carries one `p_injection` per document, so picking one "
        f"of them would be a choice this code is not entitled to make. A repository whose "
        f"positive class cannot be resolved is ineligible as a baseline and is replaced, never "
        f"removed."
    )


def _axis(id2label: Mapping[object, object], *, baseline: str) -> dict[int, str]:
    """`id2label` as `{axis position: label}`, or an abort explaining why it is not one."""
    if not isinstance(id2label, Mapping) or not id2label:
        raise PositiveClassUnresolved(
            f"baseline {baseline!r} declares no usable `id2label` (got {id2label!r}); the "
            f"positive index is resolved from the pinned revision's own label mapping and "
            f"there is nothing else to resolve it from. A repository publishing no `id2label` "
            f"is ineligible as a baseline and is replaced, never removed."
        )

    observed: dict[int, str] = {}
    for key, label in id2label.items():
        index = _index(key)
        if index is None or not isinstance(label, str):
            raise PositiveClassUnresolved(
                f"baseline {baseline!r} declares an unreadable `id2label` entry "
                f"{key!r}: {label!r}; keys are logit-axis positions and values are names"
            )
        observed[index] = label

    expected = set(range(len(observed)))
    if set(observed) != expected:
        raise PositiveClassUnresolved(
            f"baseline {baseline!r} declares `id2label` keys {sorted(observed)}, which are not "
            f"the logit-axis positions {sorted(expected)}; a mapping that does not address the "
            f"axis cannot locate a class on it"
        )
    return observed


def _index(key: object) -> int | None:
    """An axis position from a JSON object key (`\"0\"`) or a TOML/Python one (`0`)."""
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key if key >= 0 else None
    if isinstance(key, str) and key.isdigit():
        return int(key)
    return None


def _render(observed: Mapping[int, str]) -> str:
    """The observed mapping, so the abort message carries the evidence rather than a verdict."""
    return "{" + ", ".join(f"{index}: {observed[index]!r}" for index in sorted(observed)) + "}"


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """`softmax` over the model's **full** label axis, in one place, for every adapter.

    Computed with the maximum subtracted, which is exact in the sense that matters here: it
    changes no result and removes the overflow that a large logit would otherwise produce.
    """
    if not logits:
        raise ValueError("softmax needs at least one logit")
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"logits must all be finite, got {values!r}")

    top = max(values)
    exponentials = [math.exp(value - top) for value in values]
    total = math.fsum(exponentials)
    return tuple(exponential / total for exponential in exponentials)


def p_injection(logits: Sequence[float], positive_index: int) -> float:
    """`softmax(logits)[positive_index]`, the one definition of the published score."""
    probabilities = softmax(logits)
    if not 0 <= positive_index < len(probabilities):
        raise ValueError(
            f"positive index {positive_index} is off a label axis of width "
            f"{len(probabilities)}"
        )
    return probabilities[positive_index]


def reduce_windows(probabilities: Sequence[float]) -> Score:
    """A document's score: the maximum over the windows it occupies, with how many there were.

    Shared here rather than in each adapter because the reduction is a property of the
    experiment, not of a model: an encoded payload occupies several times the windows of its
    decoded form, so max-over-windows hands the extra chances to the un-canonicalized route.
    That bias runs against this project's thesis, which is the safe direction -- but only if
    every column takes it.
    """
    if not probabilities:
        raise ValueError(
            "a scored document occupies at least one window; the windower returned none"
        )
    return Score(p_injection=max(probabilities), n_windows=len(probabilities))


@runtime_checkable
class Baseline(Protocol):
    """What the harness is allowed to know about a model: a key, and `score`.

    `harness/measure.py` never tokenizes and never sees a logit. It hands documents in and
    takes `Score`s out, which is what makes the threshold, the corpus and the report indifferent
    to which repository is behind the column.
    """

    key: str

    def score(self, texts: Sequence[str]) -> list[Score]:
        """One `Score` per input document, in the order the documents came in."""
        ...
