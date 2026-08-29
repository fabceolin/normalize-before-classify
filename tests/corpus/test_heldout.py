"""The held-out registry: the inverse import rule, the floor, and `probes` read off the layer.

Story 3.4 proved the layer undoes its own corpus. This module proves the other half of AD-28: that
there is a block of the corpus it was never taught, that the block cannot quietly acquire the
layer's alphabets, and that what each held-out chain claims about the layer is what the layer
actually does.

**The one claim in story 3.5 that could have been recorded beside a value and never compared to it
is `probes`.** `test_the_declared_probe_is_what_the_layer_actually_does` compares it, chain by
chain, against the decode-stage spans `canonicalize` reports over `heldout.PROBE_PAYLOADS`. The
first form of that comparison counted offered *spans*, which does not separate `partial` from
`none` at all; `test_counting_offered_spans_would_not_separate_partial_from_none` ships that
near-miss so the sharper measure cannot be quietly replaced by the blunt one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from nbc.canon.pipeline import DEFAULT_CEILING, canonicalize, default_context
from nbc.canon.stages import decode
from nbc.corpus.dressings import ALL_DRESSINGS, DRESSINGS, all_dressings, dress, dress_declared
from nbc.corpus.heldout import (
    HELD_OUT_FROM,
    HELDOUT_DRESSINGS,
    MIN_HELDOUT_CHAINS,
    ONE_WAY_DOOR,
    PROBE_NONE,
    PROBE_PAYLOADS,
    PROBES,
    TRIGGER_EXCLUSION_REASON,
    HeldOutEncoding,
    HeldOutRegistryInvalid,
    evaluated_by_trigger,
    excluded_from_trigger,
    heldout_problems,
    probes_for,
    to_rot13,
    validate_heldout,
)
from nbc.corpus.matrix import (
    CHAIN_CLASS_BOUND,
    CHAIN_CLASS_HELD_OUT,
    CHAINS,
    CORPUS_CLASSES,
    HELDOUT_CHAINS,
    CorpusMatrixInvalid,
    chain_class,
    declared_links,
    render_chain,
)
from nbc.corpus.roundtrip import bound_chains, in_scope
from nbc.errors import exit_code_for
from nbc.schema import BENIGN_CLASSES, FAMILY_ATTACK
from heldout_golden import GOLDEN, PAYLOAD

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
HELDOUT_MODULE = SRC / "nbc" / "corpus" / "heldout.py"

HELD_OUT_ATTACK_CHAINS = tuple(tuple(chain) for chain in HELDOUT_CHAINS[FAMILY_ATTACK])
DISTINCT_HELD_OUT = tuple(
    sorted({tuple(chain) for declared in HELDOUT_CHAINS.values() for chain in declared})
)


# --- AD-28's import rule, the exact inverse of story 3.4's ----------------------------------------


def imported_modules(path: Path) -> set[str]:
    """Every module name `path` imports, fully qualified, with relative imports resolved.

    Reads the syntax tree rather than the text, for `tests/canon/test_import_bound.py`'s reason: a
    name inside a docstring is not an import, and this module's docstring names `nbc.canon` several
    times on purpose. A grep would fire on every one of them, which is exactly the textual check
    this project keeps replacing with a structural one.
    """
    if path.is_relative_to(SRC):
        package = path.relative_to(SRC).with_suffix("").parts[:-1]
    else:
        # A synthetic probe written into `tmp_path`, which anchors no package. It uses absolute
        # imports, which is the shape the rule is about; a relative import from outside `src/`
        # resolves to nothing and is left as the empty prefix rather than guessed at.
        package = ()
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                prefix = ".".join(base)
                names.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                names.add(node.module)
    return names


def _module_path(name: str) -> Path | None:
    """The file an `nbc.` module name lives in, or `None` when it is not one of ours."""
    if name != "nbc" and not name.startswith("nbc."):
        return None
    candidate = SRC.joinpath(*name.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = SRC.joinpath(*name.split(".")) / "__init__.py"
    return package if package.is_file() else None


def transitive_nbc_imports(path: Path) -> set[str]:
    """Every `nbc` module reachable from `path` by following imports, `path`'s own module excluded.

    Transitive, and that is the whole point: AD-28 forbids `corpus/heldout.py` from importing
    anything under `nbc.canon`, and a module that imported `nbc.corpus.dressings` -- which imports
    the layer on purpose -- would satisfy a one-level check while reaching the layer anyway. The
    failing input is exactly that, and it ships below.
    """
    seen: set[str] = set()
    queue = [path]
    while queue:
        current = queue.pop()
        for name in imported_modules(current):
            if not (name == "nbc" or name.startswith("nbc.")):
                continue
            if name in seen:
                continue
            seen.add(name)
            target = _module_path(name)
            if target is not None:
                queue.append(target)
    return seen


def test_the_heldout_module_imports_nothing_under_the_layer() -> None:
    """AD-28's rule, statically, over the transitive closure."""
    reached = transitive_nbc_imports(HELDOUT_MODULE)
    offenders = sorted(
        name for name in reached if name == "nbc.canon" or name.startswith("nbc.canon.")
    )
    assert offenders == [], (
        f"corpus/heldout.py reaches {offenders} through {sorted(reached)}. AD-28: a held-out "
        f"encoding derived from the layer's own constants moves with the layer, and the block "
        f"stops being held out without anything changing in that file"
    )


def test_the_transitive_scan_fires_on_a_module_that_reaches_the_layer_indirectly(
    tmp_path: Path,
) -> None:
    """The scan's own failing input, in the shape that would defeat a one-level check.

    `probe.py` imports only `nbc.corpus.dressings`, which is bound to the layer by story 3.4 and
    names nothing under `nbc.canon` in the probe's own text. A check that looked one level deep
    would pass it, and the module would reach the layer anyway.
    """
    probe = tmp_path / "probe.py"
    probe.write_text("from nbc.corpus.dressings import dress\n", encoding="utf-8")
    reached = transitive_nbc_imports(probe)
    assert any(name.startswith("nbc.canon") for name in reached), sorted(reached)
    assert not any(
        name.startswith("nbc.canon") for name in imported_modules(probe)
    ), "the probe must be innocent one level deep, or it does not test transitivity"


def test_importing_the_heldout_module_pulls_in_no_part_of_the_layer() -> None:
    """The static scan reads what the file says; this reads what the interpreter did.

    Mirrors `tests/canon/test_import_bound.py`'s runtime check in the opposite direction. A module
    reached through some path the AST walk did not model shows up here and nowhere else.
    """
    code = (
        "import nbc.corpus.heldout, sys;"
        "print(' '.join(sorted(m for m in sys.modules if m.split('.')[0] == 'nbc')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(completed.stdout.split())
    assert loaded == {
        "nbc",
        "nbc.corpus",
        "nbc.corpus.heldout",
        "nbc.corpus.matrix",
        "nbc.errors",
        "nbc.schema",
    }, completed.stdout


def test_the_bound_module_still_imports_the_layer() -> None:
    """The other half of AD-28, asserted here so the pair is readable in one place.

    Story 3.4's rule and story 3.5's rule are inverses, and a repository that satisfied only one of
    them would have either a corpus the layer cannot undo or a held-out block that is not held out.
    """
    reached = transitive_nbc_imports(SRC / "nbc" / "corpus" / "dressings.py")
    assert any(name.startswith("nbc.canon") for name in reached), sorted(reached)


# --- two registries, disjoint ---------------------------------------------------------------------


def test_the_two_dressing_registries_are_disjoint() -> None:
    assert set(DRESSINGS).isdisjoint(set(HELDOUT_DRESSINGS)), (
        "a dressing cannot be bound and held out at once: the round-trip contract would require "
        "it to be recovered and AD-28 would require it not to be"
    )


def test_every_dressing_name_appears_in_exactly_one_registry() -> None:
    """AD-28's second assertion, over the **chains** rather than over the registries.

    The registries could be disjoint while a chain in `CHAINS` named a held-out encoding, which is
    the edit that would put a held-out row inside story 3.4's contract.
    """
    bound = declared_links(CHAINS)
    held = declared_links(HELDOUT_CHAINS)
    assert bound.isdisjoint(held), sorted(bound & held)
    assert bound | held == set(ALL_DRESSINGS)
    assert bound == set(DRESSINGS)
    assert held == set(HELDOUT_DRESSINGS)


def test_the_union_registry_is_the_disjoint_union() -> None:
    assert set(ALL_DRESSINGS) == set(DRESSINGS) | set(HELDOUT_DRESSINGS)
    assert len(ALL_DRESSINGS) == len(DRESSINGS) + len(HELDOUT_DRESSINGS)


def test_the_union_agrees_with_the_bound_registry_on_every_bound_chain() -> None:
    """`dress_declared` is `dress` on a bound chain, compared rather than asserted.

    If it were not, story 3.4's contract would be checking one rendering and the builder would be
    writing another, and the two would differ only on the chains nobody looked at.
    """
    for chain in bound_chains():
        for payload in PROBE_PAYLOADS.values():
            assert dress_declared(payload, chain) == dress(payload, chain), render_chain(chain)


def test_the_bound_fold_refuses_a_held_out_name() -> None:
    """`dress`'s registry default stayed the bound one, and that default is load-bearing.

    `roundtrip.in_scope` filters story 3.4's contract on `DRESSINGS`, so a default that quietly
    admitted a held-out encoding would put a held-out chain inside a contract it must fail. The
    failing input is this call succeeding.
    """
    with pytest.raises(CorpusMatrixInvalid) as caught:
        dress(PAYLOAD, ("rot13",))
    assert "rot13" in str(caught.value)
    assert dress_declared(PAYLOAD, ("rot13",)) == GOLDEN["rot13"]


def test_a_name_in_both_registries_aborts_rather_than_shadowing() -> None:
    """The union's failing input. `{**bound, **held}` would silently let one win."""
    collision = {
        "base64": HeldOutEncoding(
            name="base64", encode=to_rot13, probes=PROBE_NONE, mechanism="a collision"
        )
    }
    with pytest.raises(HeldOutRegistryInvalid) as caught:
        all_dressings(DRESSINGS, collision)
    assert "base64" in str(caught.value)
    assert exit_code_for(caught.value) == 20


def test_the_new_abort_code_is_distinct_from_every_other() -> None:
    assert HeldOutRegistryInvalid.exit_code == 20
    assert HeldOutRegistryInvalid.exit_code != CorpusMatrixInvalid.exit_code


# --- the floor, and the shape ---------------------------------------------------------------------


def test_the_declared_registry_validates() -> None:
    validate_heldout()
    assert heldout_problems(HELDOUT_CHAINS, HELDOUT_DRESSINGS) == ()


def test_the_held_out_set_clears_the_floor_and_the_floor_is_two() -> None:
    """AD-28's floor, read off the constant rather than counted by hand."""
    assert MIN_HELDOUT_CHAINS == 2
    assert len(DISTINCT_HELD_OUT) >= MIN_HELDOUT_CHAINS
    assert len(DISTINCT_HELD_OUT) == 3, DISTINCT_HELD_OUT


def test_a_one_chain_held_out_set_fails_the_floor_rather_than_a_non_emptiness_test() -> None:
    """The input the floor exists for: non-empty, and still refused.

    A one-chain held-out set passes every "is it empty" test while failing the reason the floor
    exists, which is why AD-28 states a number.
    """
    chains = {corpus_class: (("rot13",),) for corpus_class in CORPUS_CLASSES}
    problems = heldout_problems(chains, HELDOUT_DRESSINGS)
    assert any(str(MIN_HELDOUT_CHAINS) in problem for problem in problems), problems
    assert chains, "the input is non-empty, which is the whole point of this test"


def test_a_benign_class_dressed_differently_from_the_attack_family_is_refused() -> None:
    """AD-3 extended to the held-out block: benign items carry the same chains.

    Without it the held-out block would carry recall and no counter-metric, which invites the
    answer that non-recovery does not matter because its cost is unknown.
    """
    chains = {
        FAMILY_ATTACK: (("base32",), ("url_percent",), ("rot13",)),
        "b_code": (("base32",), ("url_percent",), ("rot13",)),
        "b_chat": (("base32",), ("url_percent",)),
    }
    problems = heldout_problems(chains, HELDOUT_DRESSINGS)
    assert any("b_chat" in problem and "rot13" in problem for problem in problems), problems


def test_a_chain_naming_an_unregistered_encoding_is_refused() -> None:
    chains = {corpus_class: (("base32",), ("rot47",)) for corpus_class in CORPUS_CLASSES}
    problems = heldout_problems(chains, HELDOUT_DRESSINGS)
    assert any("rot47" in problem for problem in problems), problems


def test_a_composed_held_out_chain_is_refused_because_its_probe_is_undeclared() -> None:
    """The reason is not tidiness: a composition has no measured classification.

    base32 of a percent-encoded document is neither `decode` nor `partial` until someone measures
    it, and a value assigned to it would decide N4 membership on a claim nothing checks.
    """
    chains = {
        corpus_class: (("base32",), ("url_percent",), ("base32", "rot13"))
        for corpus_class in CORPUS_CLASSES
    }
    problems = heldout_problems(chains, HELDOUT_DRESSINGS)
    assert any("base32+rot13" in problem for problem in problems), problems
    with pytest.raises(HeldOutRegistryInvalid):
        probes_for(("base32", "rot13"))


def test_a_probe_outside_the_vocabulary_is_refused() -> None:
    """A value nothing recognizes silently removes a chain from N4's trigger."""
    registry = dict(HELDOUT_DRESSINGS)
    registry["rot13"] = HeldOutEncoding(
        name="rot13", encode=to_rot13, probes="maybe", mechanism="a typo"
    )
    problems = heldout_problems(HELDOUT_CHAINS, registry)
    assert any("maybe" in problem for problem in problems), problems


def test_a_registry_in_which_every_encoding_declares_none_is_refused() -> None:
    """N4 would quantify over an empty set and be satisfied however the layer performed."""
    registry = {
        name: HeldOutEncoding(
            name=name, encode=entry.encode, probes=PROBE_NONE, mechanism=entry.mechanism
        )
        for name, entry in HELDOUT_DRESSINGS.items()
    }
    problems = heldout_problems(HELDOUT_CHAINS, registry)
    assert any("vacuously" in problem for problem in problems), problems


def test_validate_raises_with_every_problem_it_found() -> None:
    chains = {FAMILY_ATTACK: (("rot13",),)}
    with pytest.raises(HeldOutRegistryInvalid) as caught:
        validate_heldout(chains, HELDOUT_DRESSINGS)
    assert len(caught.value.problems) > 1, caught.value.problems


# --- chain_class, AD-2's cell key ----------------------------------------------------------------


@pytest.mark.parametrize("chain", CHAINS[FAMILY_ATTACK], ids=lambda c: render_chain(c))
def test_every_bound_chain_classifies_as_bound(chain: Sequence[str]) -> None:
    assert chain_class(chain) == CHAIN_CLASS_BOUND


@pytest.mark.parametrize("chain", HELD_OUT_ATTACK_CHAINS, ids=lambda c: render_chain(c))
def test_every_held_out_chain_classifies_as_held_out(chain: Sequence[str]) -> None:
    assert chain_class(chain) == CHAIN_CLASS_HELD_OUT


def test_a_chain_mixing_the_two_registries_has_no_cell_and_is_refused() -> None:
    with pytest.raises(CorpusMatrixInvalid) as caught:
        chain_class(("base64", "rot13"))
    assert "base64+rot13" in str(caught.value)


# --- the golden fixtures -------------------------------------------------------------------------


def test_every_held_out_chain_has_a_committed_fixture() -> None:
    assert set(GOLDEN) == {render_chain(chain) for chain in DISTINCT_HELD_OUT}


@pytest.mark.parametrize("name", sorted(GOLDEN), ids=lambda n: n)
def test_the_encoding_matches_its_independently_derived_literal(name: str) -> None:
    """Two sides: the literal came from coreutils and from RFC 3986 read by hand, not from here."""
    assert dress_declared(PAYLOAD, (name,)) == GOLDEN[name]


def test_rot13_is_its_own_inverse() -> None:
    """The cheapest possible check that the rotation table was built correctly."""
    for payload in PROBE_PAYLOADS.values():
        assert to_rot13(to_rot13(payload)) == payload


def test_every_held_out_encoding_is_deterministic() -> None:
    for name, entry in sorted(HELDOUT_DRESSINGS.items()):
        for payload in PROBE_PAYLOADS.values():
            assert entry.encode(payload) == entry.encode(payload), name


# --- `probes`, measured against the layer --------------------------------------------------------

DECODE_STAGES = frozenset({decode.NAME, decode.CEILING_NAME})
"""The two names step 4 stamps on a span it examined, whether or not it replaced it."""

CANDIDATE_FLOOR = max(test.min_encoded_chars for test in decode.ORDER)
"""The longest run the layer needs before it will look at one, read off its own constants.

`decode`'s quantifier is written over the payloads whose dressed text reaches this, because a
document shorter than any candidate floor is not evidence about an encoding -- it is evidence about
the floor. `PROBE_PAYLOADS["short"]` is below it on purpose, and a test asserts that at least one
entry is, so the guard is exercised rather than decorative.
"""


@dataclass(frozen=True, slots=True)
class Engagement:
    """What step 4 did with one document, at depth 0.

    Depth 0 only: an accepted decode canonicalizes the decoded segment as an independent document
    whose spans index into *that* document, so summing coverage across depths would add up offsets
    into different strings. What `probes` classifies is what the layer does with the dressed
    document it was handed.
    """

    length: int
    offered: int
    coverage: int
    accepted: int
    ceiling_hit: bool

    @property
    def whole(self) -> bool:
        return self.offered == 1 and self.coverage == self.length


def engagement(text: str, ceiling: int = DEFAULT_CEILING) -> Engagement:
    result = canonicalize(text, default_context(ceiling=ceiling))
    spans = [edit for edit in result.edits if edit.stage in DECODE_STAGES and edit.depth == 0]
    return Engagement(
        length=len(text),
        offered=len(spans),
        coverage=sum(end - start for start, end in (edit.span for edit in spans)),
        accepted=sum(1 for edit in spans if edit.before != edit.after),
        ceiling_hit=result.ceiling_hit,
    )


def _measure(chain: Sequence[str]) -> dict[str, tuple[Engagement, Engagement]]:
    """For each battery payload: what the layer did with it dressed, and what it did undressed."""
    return {
        name: (engagement(dress_declared(payload, chain)), engagement(payload))
        for name, payload in PROBE_PAYLOADS.items()
    }


@pytest.mark.parametrize("chain", DISTINCT_HELD_OUT, ids=lambda c: render_chain(c))
def test_the_declared_probe_is_what_the_layer_actually_does(chain: tuple[str, ...]) -> None:
    """AD-28's classification, compared against the layer over the declared battery.

    The three predicates are stated in `heldout.PROBES` and enforced here, so the constant that
    declares a probe and the test that measures it cannot drift into two different claims.
    """
    declared = probes_for(chain)
    measured = _measure(chain)

    if declared == "decode":
        above_floor = {
            name: pair for name, pair in measured.items() if pair[0].length >= CANDIDATE_FLOOR
        }
        assert above_floor, "no battery payload reaches the candidate floor; nothing was measured"
        for name, (dressed, _plain) in above_floor.items():
            assert dressed.whole, (
                f"{render_chain(chain)} on {name}: the layer offered {dressed.offered} span(s) "
                f"covering {dressed.coverage} of {dressed.length} characters, and probes=decode "
                f"claims the whole document is offered as one candidate"
            )
            assert dressed.accepted == 0, f"{render_chain(chain)} on {name} was decoded"

    elif declared == "partial":
        grew = [
            name
            for name, (dressed, plain) in measured.items()
            if dressed.coverage > plain.coverage
        ]
        assert grew, (
            f"{render_chain(chain)} never extended the layer's reach on any battery payload, so "
            f"nothing distinguishes probes=partial from probes=none"
        )
        for name, (dressed, _plain) in measured.items():
            assert not dressed.whole, f"{render_chain(chain)} on {name} was offered whole"
            assert dressed.accepted == 0, f"{render_chain(chain)} on {name} was decoded"

    elif declared == PROBE_NONE:
        for name, (dressed, plain) in measured.items():
            assert dressed.coverage == plain.coverage, (
                f"{render_chain(chain)} on {name}: the layer covered {dressed.coverage} "
                f"characters of the dressed text against {plain.coverage} of the payload, so the "
                f"encoding gave it purchase it did not have; probes=none says it gives none"
            )
            assert dressed.accepted == 0, f"{render_chain(chain)} on {name} was decoded"

    else:  # pragma: no cover - `heldout_problems` refuses this and a test supplies the input
        pytest.fail(f"{declared!r} is not one of {list(PROBES)}")


def test_counting_offered_spans_would_not_separate_partial_from_none() -> None:
    """The near-miss this measure replaced, kept as a test so it cannot come back.

    `url_percent` and `rot13` report the **same offered span count** on every battery payload. The
    quantity that separates them is coverage measured against the undressed payload: `url_percent`
    extends a run the payload already carried by the two hex digits of an adjacent escape, and
    `rot13` extends nothing because it is a bijection on letters.
    """
    partial = _measure(("url_percent",))
    none = _measure(("rot13",))
    assert [pair[0].offered for pair in partial.values()] == [
        pair[0].offered for pair in none.values()
    ], "the blunt measure was supposed to be blind here"

    grew_partial = [name for name, (d, p) in partial.items() if d.coverage > p.coverage]
    grew_none = [name for name, (d, p) in none.items() if d.coverage > p.coverage]
    assert grew_partial and not grew_none, (grew_partial, grew_none)


def test_the_battery_carries_payloads_the_layer_already_grips_and_payloads_it_does_not() -> None:
    """Anti-vacuity. Without both shapes, `none` and `partial` would be indistinguishable.

    A battery of payloads the layer has no purchase on undressed makes every `plain.coverage` zero,
    and then "coverage equals the payload's" and "coverage grew" collapse into the same statement
    for any encoding that creates no candidate of its own.
    """
    gripped = [name for name, payload in PROBE_PAYLOADS.items() if engagement(payload).coverage]
    untouched = [
        name for name, payload in PROBE_PAYLOADS.items() if not engagement(payload).coverage
    ]
    assert gripped, "no battery payload carries a run the layer offers undressed"
    assert untouched, "every battery payload is already gripped, so nothing measures a bare case"


def test_at_least_one_battery_payload_sits_below_the_candidate_floor() -> None:
    """`decode`'s guard is exercised rather than decorative."""
    below = [
        name
        for name, payload in PROBE_PAYLOADS.items()
        if len(dress_declared(payload, ("base32",))) < CANDIDATE_FLOOR
    ]
    assert below, "nothing exercises the floor clause in probes=decode"


@pytest.mark.parametrize("chain", DISTINCT_HELD_OUT, ids=lambda c: render_chain(c))
def test_no_held_out_chain_round_trips(chain: tuple[str, ...]) -> None:
    """The inverse of story 3.4's contract, and the one-way door's tripwire.

    A held-out chain that round-trips is not held out: either the encoding is not what it claims to
    be, or the layer has learned it. Both need a human, and both need a **new** held-out encoding
    and a complete re-run rather than an edit to this test.
    """
    ctx = default_context(ceiling=DEFAULT_CEILING)
    for name, payload in PROBE_PAYLOADS.items():
        dressed = canonicalize(dress_declared(payload, chain), ctx)
        plain = canonicalize(payload, ctx)
        assert not dressed.ceiling_hit, f"{render_chain(chain)} on {name} hit the ceiling"
        assert dressed.text != plain.text, (
            f"{render_chain(chain)} on {name} round-trips. {ONE_WAY_DOOR}"
        )


# --- the one-way door ----------------------------------------------------------------------------


def test_the_recorded_decoder_set_is_the_layer_s_decoder_set() -> None:
    """AD-28's one-way door, enforced through the capability rather than through a tree digest.

    The day someone teaches `canon/` base32, this fails and names the door. A reworded docstring
    in the layer does not, which is deliberate: a check that fired on prose would train the reflex
    of bumping the constant without reading the diff, and that reflex is what the record exists to
    prevent.
    """
    assert HELD_OUT_FROM.layer_decoders == {test.encoding for test in decode.ORDER}, ONE_WAY_DOOR


def test_no_held_out_encoding_is_among_the_layer_s_decoders() -> None:
    assert HELD_OUT_FROM.layer_decoders.isdisjoint(set(HELDOUT_DRESSINGS)), ONE_WAY_DOOR


def test_the_recorded_revision_has_the_shape_of_a_full_commit() -> None:
    """Full, not abbreviated: an abbreviated sha stops being unique as the history grows."""
    revision = HELD_OUT_FROM.revision
    assert len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), revision
    assert HELD_OUT_FROM.declared_on


def test_the_recorded_revision_is_a_commit_in_this_repository() -> None:
    """What makes the sha a record rather than a decoration, and the pins gate's other half.

    `tests/test_pins.py` grants this one value an allowance from the "a commit sha is a pin" gate,
    on the grounds that it names this repository rather than a remote artifact. That claim is
    checked here: git is asked whether the object exists and whether it is an ancestor of `HEAD`.
    """
    if not (REPO / ".git").exists():
        pytest.skip("no git metadata here; the ancestry half of the record is unchecked")
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{HELD_OUT_FROM.revision}^{{commit}}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, (
        f"{HELD_OUT_FROM.revision} is not a commit in this repository, so it is not the layer "
        f"revision anything was held out from: {probe.stderr}"
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", HELD_OUT_FROM.revision, "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert ancestry.returncode == 0, (
        f"{HELD_OUT_FROM.revision} is not an ancestor of HEAD; the held-out set was recorded "
        f"against a layer this branch never had"
    )


# --- N4's trigger exclusion ----------------------------------------------------------------------


def test_the_excluded_chains_are_exactly_the_ones_the_layer_cannot_grip() -> None:
    assert excluded_from_trigger() == ("rot13",)
    assert set(excluded_from_trigger()) == {
        render_chain(chain) for chain in DISTINCT_HELD_OUT if probes_for(chain) == PROBE_NONE
    }


def test_the_two_halves_partition_the_held_out_chains() -> None:
    """Neither can grow at the other's expense, and nothing falls out of both."""
    excluded = excluded_from_trigger()
    evaluated = evaluated_by_trigger()
    assert set(excluded).isdisjoint(evaluated)
    assert set(excluded) | set(evaluated) == {render_chain(c) for c in DISTINCT_HELD_OUT}
    assert evaluated, "N4 would have no held-out data at all"


def test_the_exclusion_is_computed_from_the_classification_and_not_from_a_list() -> None:
    """The failing input: a registry claiming the layer grips `rot13`. The exclusion disappears.

    Which is what should happen. A chain is excluded because of what the layer can do to it, never
    because someone put it on a list.
    """
    registry = dict(HELDOUT_DRESSINGS)
    registry["rot13"] = HeldOutEncoding(
        name="rot13",
        encode=to_rot13,
        probes="partial",
        mechanism="a claim that the layer grips it",
    )
    assert excluded_from_trigger(HELDOUT_CHAINS, registry) == ()
    assert set(evaluated_by_trigger(HELDOUT_CHAINS, registry)) == {
        render_chain(chain) for chain in DISTINCT_HELD_OUT
    }


def test_the_exclusion_reason_travels_with_the_exclusion() -> None:
    """AD-28 requires the reason rendered next to the verdict, so it has to exist beside the set."""
    assert f"probes: {PROBE_NONE}" in TRIGGER_EXCLUSION_REASON, TRIGGER_EXCLUSION_REASON
    assert "boundary" in TRIGGER_EXCLUSION_REASON, TRIGGER_EXCLUSION_REASON


# --- the bound contract is unchanged -------------------------------------------------------------


@pytest.mark.parametrize("chain", DISTINCT_HELD_OUT, ids=lambda c: render_chain(c))
def test_no_held_out_chain_is_in_the_round_trip_contract(chain: tuple[str, ...]) -> None:
    """Story 3.4 scoped its contract by a filter over the registries, before this one existed.

    `in_scope` refused a chain naming `rot13` when `rot13` was a name in a docstring. It now names
    a real encoding, and the filter still refuses it -- because of what it is, not because anyone
    added it to a list.
    """
    assert not in_scope(chain)
    assert chain not in bound_chains()


def test_the_round_trip_contract_still_covers_every_bound_dressing() -> None:
    """The complementary gate: narrowing the scope until the contract passes fails here."""
    links = {link for chain in bound_chains() for link in chain}
    assert links == set(DRESSINGS)


def test_the_benign_classes_carry_the_held_out_chains_too() -> None:
    """AD-3 extended, checked between three separately written declarations."""
    attack = {render_chain(chain) for chain in HELDOUT_CHAINS[FAMILY_ATTACK]}
    for corpus_class in BENIGN_CLASSES:
        assert {render_chain(chain) for chain in HELDOUT_CHAINS[corpus_class]} == attack


def test_the_held_out_registry_covers_three_distinct_mechanisms() -> None:
    """AD-28 chose the three for distinct interactions with the layer, not for variety."""
    declared = {entry.probes for entry in HELDOUT_DRESSINGS.values()}
    assert declared == set(PROBES), declared
    mechanisms = {entry.mechanism for entry in HELDOUT_DRESSINGS.values()}
    assert len(mechanisms) == len(HELDOUT_DRESSINGS)
