"""The port resolves the positive class rather than knowing it, and shares its arithmetic.

The failure these tests exist for is not a crash. It is a column of the published table being
the *complement* of what it claims to be, because the second baseline happened to order its
labels the other way and an adapter assumed index 1. That bug looks like a finding.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from nbc.baselines import port
from nbc.baselines.port import (
    POSITIVE_CLASS_NAMES,
    PositiveClassUnresolved,
    p_injection,
    reduce_windows,
    resolve_positive_index,
    softmax,
)
from nbc.errors import NbcError
from nbc.schema import Score

PINNED_MAPPING = {"0": "SAFE", "1": "INJECTION"}
"""What both pinned baselines publish, as `config.json` spells it: JSON keys are strings."""


# -- the name set is a constant, and a checked one -----------------------------------------


def test_the_positive_class_name_set_is_one_constant_in_the_port() -> None:
    assert isinstance(POSITIVE_CLASS_NAMES, frozenset)
    assert POSITIVE_CLASS_NAMES, "an empty set resolves nothing"


@pytest.mark.parametrize("meaningless", ["LABEL_0", "LABEL_1", "label_0", "label-1", "Label1"])
def test_the_name_set_admits_no_positionally_meaningless_name(meaningless: str) -> None:
    """`LABEL_1` resolves to the index it already is: a hardcoded index wearing a lookup."""
    assert meaningless.casefold() not in POSITIVE_CLASS_NAMES
    assert port._POSITIONALLY_MEANINGLESS.match(meaningless)


def test_every_admitted_name_is_stored_casefolded() -> None:
    """The fold belongs on the observed label, applied in one place."""
    assert all(name == name.casefold() for name in POSITIVE_CLASS_NAMES)


def test_no_adapter_module_spells_a_positive_index_of_its_own(repo_root: Path) -> None:
    """`positive_index` is assigned from the resolver, and from nothing else, anywhere.

    A literal `positive_index = 1` next to a resolver that also exists is the failure this
    story is named for: the lookup is present, correct, and not what produces the number.
    """
    offenders: list[str] = []
    for source in sorted((repo_root / "src" / "nbc" / "baselines").glob("*.py")):
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if "resolve_positive_index" in line or "positive_index" not in line:
                continue
            if re.search(r"=\s*-?\d", line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, "; ".join(offenders)


# -- resolution ------------------------------------------------------------------------------


def test_the_pinned_mapping_resolves_to_the_index_the_repository_declares() -> None:
    assert resolve_positive_index(PINNED_MAPPING, baseline="protectai-deberta-v3") == 1


def test_a_reversed_mapping_resolves_to_the_other_index() -> None:
    """The whole point: the index follows the declaration, not the position it usually sits at."""
    assert resolve_positive_index({"0": "INJECTION", "1": "SAFE"}, baseline="swapped") == 0


@pytest.mark.parametrize("spelling", ["INJECTION", "injection", "Injection", " Injection "])
def test_matching_is_case_insensitive_and_tolerates_surrounding_space(spelling: str) -> None:
    assert resolve_positive_index({0: "SAFE", 1: spelling}, baseline="cased") == 1


def test_integer_keys_and_string_keys_resolve_alike() -> None:
    """`config.json` gives string keys, a hand-built mapping integers. Neither is special."""
    assert resolve_positive_index({0: "safe", 1: "injection"}, baseline="ints") == 1


def test_zero_matches_aborts_and_carries_the_observed_mapping() -> None:
    with pytest.raises(PositiveClassUnresolved) as raised:
        resolve_positive_index({"0": "LABEL_0", "1": "LABEL_1"}, baseline="unnamed")
    message = str(raised.value)
    assert "unnamed" in message
    assert "'LABEL_0'" in message and "'LABEL_1'" in message
    assert "replaced, never removed" in message


def test_more_than_one_match_also_aborts_and_names_both() -> None:
    """Two matches is not a tie to be broken: the port carries one `p_injection` per document."""
    with pytest.raises(PositiveClassUnresolved) as raised:
        resolve_positive_index(
            {"0": "safe", "1": "injection", "2": "jailbreak"}, baseline="three-class"
        )
    message = str(raised.value)
    assert "1='injection'" in message and "2='jailbreak'" in message
    assert "2 positive classes" in message


@pytest.mark.parametrize("absent", [None, {}, "INJECTION", []])
def test_a_repository_publishing_no_id2label_is_ineligible(absent: object) -> None:
    with pytest.raises(PositiveClassUnresolved, match="no usable `id2label`"):
        resolve_positive_index(absent, baseline="unlabelled")  # type: ignore[arg-type]


def test_keys_that_are_not_the_logit_axis_abort() -> None:
    """A mapping that does not address `0..n-1` cannot locate a class on the axis."""
    with pytest.raises(PositiveClassUnresolved, match="not"):
        resolve_positive_index({"1": "safe", "2": "injection"}, baseline="off-by-one")


@pytest.mark.parametrize("entry", [{"x": "injection"}, {"0": 1}, {True: "injection"}])
def test_an_unreadable_entry_aborts(entry: dict) -> None:
    with pytest.raises(PositiveClassUnresolved):
        resolve_positive_index(entry, baseline="malformed")


def test_the_unresolved_abort_is_one_of_the_declared_ones() -> None:
    assert issubclass(PositiveClassUnresolved, NbcError)
    assert PositiveClassUnresolved.exit_code == 8


# -- the shared arithmetic -------------------------------------------------------------------


def test_softmax_sums_to_one_over_the_full_label_axis() -> None:
    probabilities = softmax([2.0, 1.0, 0.5])
    assert len(probabilities) == 3
    assert math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)


def test_softmax_does_not_overflow_on_a_large_logit() -> None:
    """The maximum is subtracted, so a logit a model can genuinely emit is not an `inf`."""
    probabilities = softmax([1000.0, 0.0])
    assert math.isclose(probabilities[0], 1.0, rel_tol=0, abs_tol=1e-12)


def test_softmax_is_over_every_logit_rather_than_a_sigmoid_on_one() -> None:
    """Two adapters, one on softmax and one on sigmoid, make a single threshold meaningless."""
    two = softmax([1.0, 0.0])
    assert math.isclose(two[1], 1.0 / (1.0 + math.e), rel_tol=0, abs_tol=1e-12)


@pytest.mark.parametrize("broken", [[], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_softmax_refuses_logits_it_cannot_reduce(broken: list[float]) -> None:
    with pytest.raises(ValueError):
        softmax(broken)


def test_p_injection_reads_the_resolved_index_and_not_the_last_one() -> None:
    logits = [3.0, 1.0]
    assert p_injection(logits, 0) > 0.5
    assert p_injection(logits, 1) < 0.5
    assert math.isclose(p_injection(logits, 0) + p_injection(logits, 1), 1.0, abs_tol=1e-12)


def test_p_injection_refuses_an_index_off_the_axis() -> None:
    with pytest.raises(ValueError, match="off a label axis"):
        p_injection([1.0, 2.0], 2)


def test_a_document_takes_the_maximum_over_its_windows() -> None:
    score = reduce_windows([0.1, 0.9, 0.4])
    assert score == Score(p_injection=0.9, n_windows=3)


def test_the_window_count_rides_on_the_score() -> None:
    """`n_windows` is part of what the run reports, so it travels with the number it produced."""
    assert reduce_windows([0.25]).n_windows == 1


def test_a_document_with_no_windows_is_a_contract_violation() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        reduce_windows([])


# -- the port keeps the inference runtime out ------------------------------------------------


def test_importing_the_port_does_not_import_the_inference_runtime() -> None:
    """`harness/` and `report/` read the shared arithmetic from here and must not pay for ORT."""
    import subprocess
    import sys

    code = "import sys, nbc.baselines.port; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout
