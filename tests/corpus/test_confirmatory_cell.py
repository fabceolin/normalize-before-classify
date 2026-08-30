"""Story 3.9: the pre-registered N1 cell has to be one N1 could have come out either way on.

The declaration and its hash already ship -- `pins.toml` fixes the cell and `frame_id` carries it
into `build_id`. What is asserted here is the half that decides whether the pre-registration was
worth making: a **bound** chain inside the layer's decode budget is recovered completely by story
3.4's contract, so `Delta recall` is at its maximum by construction and `D = FPR delta - recall
delta` could clear zero only through an impossibility. A cell declared there would report
`not_triggered` with every field present.

Two properties carry the module. The first is that the ceiling is genuinely **consulted**: the
shipped cell passes at the shipped ceiling and the *same* cell is refused under a ceiling that would
recover it, so the gate follows the layer rather than describing it. The second is that the refusal
**names a way out** -- a message that only says "no" leaves the reader to rediscover which of the
two admissible limbs exists in this corpus, and the suggestion is asserted to carry both.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nbc.canon.pipeline import DEFAULT_CEILING, canonicalize, default_context
from nbc.corpus.build import build_corpus
from nbc.corpus.dressings import dress
from nbc.corpus.manifest import (
    ConfirmatoryCellNotFalsifiable,
    confirmatory_cell_falsifiability_problems,
    confirmatory_cell_problems,
)
from nbc.corpus.matrix import (
    CHAIN_CLASS_BOUND,
    CHAINS,
    HELDOUT_CHAINS,
    CorpusMatrixInvalid,
    chain_class,
    encoding_depth,
    parse_chain,
    render_chain,
)
from nbc.errors import exit_code_for
from nbc.pins import load_pins

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pins():
    return load_pins(REPO_ROOT)


def _with_chain(pins, chain: str):
    cell = replace(pins.benign_frame.confirmatory_cell, dressing_chain=chain)
    return replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))


# --- the ceiling this gate is measured against --------------------------------------------------


def test_the_layer_itself_agrees_the_pre_registered_chain_is_past_its_budget(pins) -> None:
    """The gate computes `encoding_depth > ceiling`; the layer reports `ceiling_hit`. Compared.

    Two sides that do not come from the same place: one is arithmetic over the declared chain, the
    other is what `canonicalize` does to a document dressed through it at the shipped context. If
    the pre-registration were legitimate only on paper -- a depth the dressing does not actually
    spend -- this fails and the gate above would have been measuring its own definition.

    `BUDGET_PAYLOAD` carries no encoded run of its own, asserted in `tests/corpus/test_matrix.py`,
    so what the layer spends here is what the chain cost it and not what the payload did.
    """
    from tests.corpus.test_matrix import BUDGET_PAYLOAD

    context = default_context()
    links = parse_chain(pins.benign_frame.confirmatory_cell.dressing_chain)
    result = canonicalize(dress(BUDGET_PAYLOAD, links), context)

    assert result.ceiling_hit is True
    assert result.ceiling_hit is (encoding_depth(links) > context.ceiling)
    assert confirmatory_cell_falsifiability_problems(pins, ceiling=context.ceiling) == ()


def test_the_gate_would_have_seen_a_bound_chain_the_layer_finishes(pins) -> None:
    """The negative half, so the comparison above cannot pass by agreeing about nothing.

    A declared bound chain inside the budget: the layer finishes it -- no ceiling hit, the dressing
    undone -- and the gate refuses the cell. Both sides move together in the other direction.
    """
    from tests.corpus.test_matrix import BUDGET_PAYLOAD

    context = default_context()
    links = parse_chain("base64+base64")
    result = canonicalize(dress(BUDGET_PAYLOAD, links), context)

    assert result.ceiling_hit is False
    assert result.text == BUDGET_PAYLOAD
    assert confirmatory_cell_falsifiability_problems(
        _with_chain(pins, "base64+base64"), ceiling=context.ceiling
    )


# --- the shipped declaration ---------------------------------------------------------------------


def test_the_committed_cell_is_falsifiable_at_the_shipped_ceiling(pins) -> None:
    assert confirmatory_cell_falsifiability_problems(pins, ceiling=DEFAULT_CEILING) == ()


def test_the_committed_cell_is_bound_and_passes_on_the_ceiling_limb_alone(pins) -> None:
    """Which limb the shipped cell satisfies, asserted rather than left to the reader.

    If it ever became held out, the ceiling test below would still pass while testing nothing, so
    the limb is pinned here: the cell is *bound* and it clears the gate only by nesting past the
    budget.
    """
    links = parse_chain(pins.benign_frame.confirmatory_cell.dressing_chain)
    assert chain_class(links) == CHAIN_CLASS_BOUND
    assert encoding_depth(links) > DEFAULT_CEILING


def test_the_same_cell_is_refused_under_a_ceiling_that_would_recover_it(pins) -> None:
    """The ceiling is a parameter, not decoration: one number changes the verdict on one cell."""
    deeper = encoding_depth(parse_chain(pins.benign_frame.confirmatory_cell.dressing_chain))
    (problem,) = confirmatory_cell_falsifiability_problems(pins, ceiling=deeper)
    assert f"encoding_depth {deeper}" in problem
    assert f"ceiling of {deeper}" in problem


# --- the held-out limb ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chain", sorted(render_chain(chain) for chain in HELDOUT_CHAINS["b_code"])
)
def test_a_held_out_chain_is_accepted_whatever_its_depth(pins, chain: str) -> None:
    """Held out is sufficient on its own -- the layer was never written against these encodings.

    Depth is not consulted, and these chains all sit at depth 0, so a gate that had accidentally
    required *both* limbs would refuse every one of them.
    """
    assert confirmatory_cell_falsifiability_problems(_with_chain(pins, chain), ceiling=0) == ()


# --- the bound-and-inside-the-budget limb, which is the refusal -----------------------------------


BOUND_INSIDE_BUDGET = sorted(
    render_chain(chain)
    for chain in CHAINS["b_code"]
    if encoding_depth(chain) <= DEFAULT_CEILING
)


def test_the_corpus_declares_bound_chains_this_gate_must_refuse() -> None:
    """The red inputs below are real declared chains, not invented ones. Named so the list moves."""
    assert BOUND_INSIDE_BUDGET == [
        "base64",
        "base64+base64",
        "base64+homoglyph",
        "clean",
        "hex",
        "hex+zero_width",
        "homoglyph",
        "zero_width",
        "zero_width+base64",
    ]


@pytest.mark.parametrize("chain", BOUND_INSIDE_BUDGET)
def test_a_bound_chain_inside_the_decode_budget_is_refused(pins, chain: str) -> None:
    (problem,) = confirmatory_cell_falsifiability_problems(
        _with_chain(pins, chain), ceiling=DEFAULT_CEILING
    )
    assert repr(chain) in problem
    assert f"ceiling of {DEFAULT_CEILING}" in problem


def test_the_refusal_states_the_reason_rather_than_only_the_rule(pins) -> None:
    """FR5.4 asks for the reason in words, so the words are asserted.

    Not a substring of the rule -- the four clauses that make the refusal an argument: total
    recovery by the 3.4 contract, the maximum by construction, the impossibility the threshold would
    be hiding, and that the legible version of the mistake is the worse one.
    """
    (problem,) = confirmatory_cell_falsifiability_problems(
        _with_chain(pins, "base64"), ceiling=DEFAULT_CEILING
    )
    assert "round-trip contract" in problem
    assert "maximum by construction" in problem
    assert "impossibility dressed as a threshold" in problem
    assert "legibly, which is worse" in problem


def test_the_refusal_names_a_chain_from_each_admissible_limb(pins) -> None:
    """A check that names no way out is one whose reader has to rediscover the corpus.

    Both limbs, with the distractor present: the suggestion has to carry a held-out chain **and**
    the over-ceiling bound chain. A suggestion built from one registry would pass a test that only
    asked whether the list was non-empty.
    """
    (problem,) = confirmatory_cell_falsifiability_problems(
        _with_chain(pins, "base64"), ceiling=DEFAULT_CEILING
    )
    held_out = {render_chain(chain) for chain in HELDOUT_CHAINS["b_code"]}
    over_ceiling = {
        render_chain(chain)
        for chain in CHAINS["b_code"]
        if encoding_depth(chain) > DEFAULT_CEILING
    }
    assert held_out and over_ceiling
    assert any(name in problem for name in held_out)
    assert any(name in problem for name in over_ceiling)


# --- the precondition, and the gate it belongs behind ----------------------------------------------


def test_a_chain_no_registry_declares_reaches_the_declared_parser(pins) -> None:
    """This function is not a second copy of the shape gate, and does not repeat its message.

    `parse_chain` is the declared parser (house rule: structural identity over textual matching), so
    an undeclared chain raises its own abort here. The shape gate runs first in `build.py` and its
    message is the better one -- asserted in the build-order test below.
    """
    with pytest.raises(CorpusMatrixInvalid):
        confirmatory_cell_falsifiability_problems(
            _with_chain(pins, "base64+rot13"), ceiling=DEFAULT_CEILING
        )


def test_the_build_refuses_a_bound_cell_before_it_fetches_anything(pins) -> None:
    """Step 0, ahead of the first archive: the abort a wrong pre-registration deserves.

    `tests/conftest.py`'s offline guard is the second half of this assertion. If the gate did not
    fire here the build would reach `read_attack_pool` and raise a socket error instead, so the
    exception type is what proves the ordering.
    """
    with pytest.raises(ConfirmatoryCellNotFalsifiable) as raised:
        build_corpus(_with_chain(pins, "base64"))
    assert exit_code_for(raised.value) == 35
    assert any("base64" in problem for problem in raised.value.problems)


def test_the_shape_gate_runs_first_and_keeps_its_own_message(pins) -> None:
    """An undeclared chain is a shape problem, and the reader is told which registries were read."""
    from nbc.corpus.manifest import CorpusManifestMismatch

    moved = _with_chain(pins, "base32+rot13")
    assert confirmatory_cell_problems(moved)
    with pytest.raises(CorpusManifestMismatch) as raised:
        build_corpus(moved)
    assert any("not declared" in problem for problem in raised.value.problems)


# --- why one gate at build time is the whole door --------------------------------------------------


def test_a_cell_edited_after_the_build_is_caught_by_the_hash(tmp_path: Path, pins) -> None:
    """The argument for a build-time-only gate, asserted instead of asserted-about.

    The gate does not run on every `read_corpus`, and it does not need to: the cell is inside the
    frame, the frame is inside `build_id`, and the guarded read recomputes it. So a cell repointed
    at a bound chain after the corpus was drawn cannot be measured with that corpus, and repointing
    it and rebuilding meets the gate above. The two doors together are total.
    """
    from tests.corpus.test_manifest import _write

    from nbc.corpus.manifest import CorpusManifestMismatch, read_corpus

    _write(tmp_path, pins)
    with pytest.raises(CorpusManifestMismatch) as raised:
        read_corpus(_with_chain(pins, "base64"), tmp_path)
    assert any("build_id" in problem for problem in raised.value.problems)


def test_the_suggestion_says_so_when_no_registry_declares_a_chain_for_the_class(pins) -> None:
    """The branch `build.py` cannot reach, fired with a synthetic cell and declared vacuous.

    Behind the shape gate a class with no registry entry never arrives here, so this list is never
    empty in production. Reached deliberately: a refusal ending in a bare `[]` reads as "there is
    no way out", which is a stronger and wrong claim, and an unreachable branch that has never been
    executed is a branch nobody has read.
    """
    cell = replace(
        pins.benign_frame.confirmatory_cell, dressing_chain="base64", benign_class="b_email"
    )
    moved = replace(pins, benign_frame=replace(pins.benign_frame, confirmatory_cell=cell))
    (problem,) = confirmatory_cell_falsifiability_problems(moved, ceiling=DEFAULT_CEILING)
    assert "neither registry declares any for it" in problem
    assert "[]" not in problem
    # And the state is one the shape gate refuses first, which is why the branch is vacuous.
    assert confirmatory_cell_problems(moved)
