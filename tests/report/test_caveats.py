"""The honesty gate is a gate, proved by breaking the section every way it can break.

The repository's own README is the happy path and is checked as it ships. Every failure is
exercised against a synthetic README built from a fixture, because the interesting cases are the
ones the real file must never be in.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from nbc.errors import EXIT_OK, NbcError, exit_code_for
from nbc.report.caveats import (
    CAVEATS_HEADING,
    MIN_CAVEAT_BODY_CHARS,
    REQUIRED_LABELS,
    RESERVED_LABEL,
    RESULTS_END,
    RESULTS_START,
    Caveat,
    CaveatsSectionMissing,
    verify_caveats,
    verify_caveats_file,
)

BODY = "x" * (MIN_CAVEAT_BODY_CHARS + 20)


def caveat(label: str) -> str:
    return f"**{label}.** {BODY}"


def section(labels: tuple[str, ...] = REQUIRED_LABELS, reserved: str | None = RESERVED_LABEL) -> str:
    blocks = [caveat(label) for label in labels]
    if reserved is not None:
        blocks.append(f"**{reserved}.** *(reserved for what the run actually revealed)*")
    return CAVEATS_HEADING + "\n\n" + "\n\n".join(blocks) + "\n"


def readme(
    caveats: str | None = None,
    markers: str | None = None,
    trailing: str = "\n## License\n\nMIT.\n",
) -> str:
    if caveats is None:
        caveats = section()
    if markers is None:
        markers = f"{RESULTS_START}\n\n*No run has produced a table yet.*\n\n{RESULTS_END}\n"
    return f"# a repository\n\n## What gets measured\n\n{markers}\n{caveats}{trailing}"


# --- the happy path, including the file this repository actually ships ------------------------


def test_the_repositorys_own_readme_passes(repo_root: Path) -> None:
    report = verify_caveats_file(repo_root / "README.md")
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)
    assert report.section_chars > 0


def test_a_well_formed_section_reports_every_label_in_order() -> None:
    report = verify_caveats(readme())
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)
    assert all(isinstance(item, Caveat) for item in report.caveats)
    assert report.as_run_fields()["caveats_check"] == "ok"
    assert report.as_run_fields()["caveats_required"] == list(REQUIRED_LABELS)


def test_the_check_reads_the_real_readmes_caveats_and_not_the_generated_block(
    repo_root: Path,
) -> None:
    """The section must sit after the generated block, or a run would overwrite it."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert text.index(RESULTS_END) < text.index(CAVEATS_HEADING)
    assert text.count(RESULTS_START) == 1 and text.count(RESULTS_END) == 1


def test_caveat_three_states_the_baseline_count_the_pins_actually_declare(
    repo_root: Path,
) -> None:
    """Caveat 3 states the count as OQ2 decided it, and `pins.toml` is what decided it.

    A baseline swapped in later without the caveat being rewritten would publish a sentence about
    a baseline set the repository no longer has. There is no way to check the caveat's *prose*
    against the architecture workspace — it is not shipped — but the one fact in it that a machine
    can hold to account is the count, and this is that binding.
    """
    from nbc.pins import load_pins

    text = (repo_root / "README.md").read_text(encoding="utf-8")
    section = text[text.index(CAVEATS_HEADING) :]
    caveat_three = section[section.index("**3. ") : section.index("**3b.")]

    assert len(load_pins(repo_root).baselines) == 2
    assert "two baselines" in caveat_three
    assert "third baseline was pinned and then dropped" in caveat_three


def test_the_readme_says_normalization_is_not_a_new_idea_before_the_measurement(
    repo_root: Path,
) -> None:
    """FR19's related-work note: a reader must not conclude the author invented normalization."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    opening = text[: text.index("## What gets measured")]
    assert "not a new idea" in opening
    assert "measurement" in opening


# --- every figure a caveat prints, bound to the file it was read out of -------------------------
#
# The section is hand-written, so every number in it is typed by a person and is free to drift away
# from the file it came from. The precedent is `tests/report/test_readme.py`'s binding of the "63
# repositories" sentence to `data/manifest.json`: the figure stays typed, and a test re-derives it
# from the file and asserts the published sentence, so drift is a red test rather than a wrong page.
#
# One test per figure, and each asserts the derived value and the derived COORDINATE adjacent in a
# single string. Binding the value alone proves the arithmetic and leaves the sentence free to
# attribute a correct number to the wrong baseline, the wrong canon state or the wrong class -- a
# hole this suite was shown to have, by swapping the two baseline names in caveat 5 and staying
# green. Each test below is proved red by mutating an attribution, not a figure.


@pytest.fixture(scope="session")
def results(repo_root: Path) -> dict[str, Any]:
    """The committed results file, parsed. Read once; nothing here mutates it."""
    return json.loads((repo_root / "results" / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def manifest(repo_root: Path) -> dict[str, Any]:
    """The corpus build's own record of what it drew and what it removed."""
    return json.loads((repo_root / "data" / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def caveats_section(repo_root: Path) -> str:
    """The repository's own honesty section, from its heading to the next `## `."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    section = text[text.index(CAVEATS_HEADING) :]
    following = re.search(r"^## ", section[len(CAVEATS_HEADING) :], re.MULTILINE)
    if following is None:
        return section
    return section[: len(CAVEATS_HEADING) + following.start()]


def one_caveat(section: str, label: str) -> str:
    """One caveat's body, flattened to a single line.

    Flattened because the page is hard-wrapped: a sentence a test asserts on is split across lines
    at whatever column the prose reached, and an assertion that depended on where would fail the
    next time somebody reflowed a paragraph without changing a word of it.
    """
    start = section.index(f"**{label}.")
    following = re.search(r"^\*\*\d+[a-z]?\.", section[start + 3 :], re.MULTILINE)
    end = start + 3 + following.start() if following is not None else len(section)
    return " ".join(section[start:end].split())


NUMBER_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four"}
"""The prose spells small counts, so a test binding one has to spell it the same way."""


PAIRING_AXES: tuple[tuple[str, str], ...] = (
    ("baseline", "its baseline"),
    ("dressing_chain", "its dressing chain"),
    ("chain_class", "its chain class"),
    ("canon_on", "its canon state"),
    ("window_policy", "its window policy"),
    ("population", "its population"),
)
"""The axes a benign rate cell and its attack cell share, and how slot 8 names each of them.

Deliberately *not* `family` and `benign_class`: those are exactly the axes that differ between the
two cells being paired. The consequence is that the lookup built from these axes is many-to-one --
`b_code` and `b_chat` at the same chain both point at the same attack cell -- so the tests below
assert that no two attack cells collide on it rather than assuming it.
"""


def _coordinates(cell: dict[str, Any]) -> tuple[Any, ...]:
    """The axes a benign rate and its attack rate share, so the two can be read as one pair."""
    key = cell["key"]
    return tuple(key[axis] for axis, _ in PAIRING_AXES)


@pytest.fixture(scope="session")
def window_overflow(results: dict[str, Any]) -> dict[tuple[Any, ...], dict[Any, dict[str, Any]]]:
    """The `window_overflow` census counts, grouped so each group holds both canon states.

    The grouping refuses a duplicate rather than overwriting one: two cells at the same
    coordinates and the same canon state would mean the shares below were computed over a
    silently truncated population.
    """
    matched: dict[tuple[Any, ...], dict[Any, dict[str, Any]]] = {}
    for cell in results["cells"]:
        if cell["kind"] != "count" or cell.get("census") != "window_overflow":
            continue
        key = cell["key"]
        coordinates = (
            key["baseline"],
            key["dressing_chain"],
            key["chain_class"],
            key["family"],
            key["benign_class"],
            key["window_policy"],
            key["population"],
        )
        group = matched.setdefault(coordinates, {})
        assert key["canon_on"] not in group, (
            f"two window-overflow cells at {coordinates} with canon_on={key['canon_on']}; "
            f"one of them would be dropped and the share computed over the other"
        )
        group[key["canon_on"]] = cell
    assert matched, "results.json carries no window-overflow census at all"
    return matched


def overflow_share(
    matched: dict[tuple[Any, ...], dict[Any, dict[str, Any]]],
    select: Callable[[tuple[Any, ...]], bool],
) -> tuple[float, float]:
    """The layer-off and layer-on share of items needing more than one window, over one selection.

    One denominator, asserted rather than assumed: a group contributes only when the file carries
    it under both canon states, and the two states are asserted to count the same `n`. If they ever
    did not, `off` and `on` would be shares of different populations and the direction between them
    would not be a direction.
    """
    off = on = total = 0
    for coordinates, pair in matched.items():
        if False in pair and True in pair and select(coordinates):
            assert pair[False]["n"] == pair[True]["n"], (
                f"the two canon states at {coordinates} count different populations "
                f"({pair[False]['n']} and {pair[True]['n']}); one denominator is not available"
            )
            off += pair[False]["k"]
            on += pair[True]["k"]
            total += pair[False]["n"]
    assert total, "no matched window-overflow cells for that selection"
    return 100 * off / total, 100 * on / total


def test_the_reserved_slot_is_no_longer_the_seed_placeholder(
    repo_root: Path, caveats_section: str
) -> None:
    """`caveats.py:281-292` exempts slot 8 from the thinness floor, so this test is the only gate.

    The checker cannot tell a written slot 8 from a seeded one -- by design, since the seed has to
    pass before the run exists. Once the run exists, nothing in `caveats.py` notices if the seed
    comes back. This does, and it measures the quantity the exempted gate would have measured:
    `Caveat.body_chars` off the checker's own parse, not a separately sliced string.
    """
    report = verify_caveats_file(repo_root / "README.md")
    body_chars = {caveat.label: caveat.body_chars for caveat in report.caveats}

    assert "reserved for what the run actually revealed" not in one_caveat(
        caveats_section, RESERVED_LABEL
    )
    assert body_chars[RESERVED_LABEL] > MIN_CAVEAT_BODY_CHARS


def test_the_reserved_slots_saturation_figures_are_the_ones_results_json_carries(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """Slot 8's spine: benign cells whose attack cell saturates at the same coordinates.

    A false-positive rate and a recall are never one cell, so the pairing is stated in the prose
    and asserted here, including the property the lookup depends on: no two attack cells collide
    on the shared axes. Without that, a duplicate would overwrite unnoticed and the count of 19
    would be over a population nobody could reconstruct.
    """
    rates = [cell for cell in results["cells"] if cell["kind"] == "rate"]
    benign = [cell for cell in rates if cell["key"]["benign_class"] is not None]

    recall: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in rates:
        if cell["key"]["family"] != "attack":
            continue
        coordinates = _coordinates(cell)
        assert coordinates not in recall, (
            f"two attack rate cells share the paired axes at {coordinates}; the lookup would "
            f"drop one of them and the pairing slot 8 describes would not be well defined"
        )
        recall[coordinates] = cell

    saturated = [cell for cell in benign if cell["value"] >= 0.99]
    together = [
        cell
        for cell in saturated
        if _coordinates(cell) in recall and recall[_coordinates(cell)]["value"] >= 0.99
    ]
    exact = [
        cell
        for cell in together
        if cell["value"] == 1.0 and recall[_coordinates(cell)]["value"] == 1.0
    ]

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert f"Of the **{len(benign)}** published false-positive rates" in slot
    assert f"**{len(saturated)} sit at or above 0.99**" in slot
    assert (
        f"for **{len(together)} of those** the attack cell at the same baseline, chain and canon "
        f"state reports a recall at or above 0.99 as well" in slot
    )
    assert f"**{len(exact)} of the {len(together)}** read exactly 1.0000 on both" in slot

    # The pairing itself, named axis by axis in the order the lookup uses them.
    named = [name for _, name in PAIRING_AXES]
    assert (
        "the attack cell sharing " + ", ".join(named[:-1]) + " and " + named[-1] in slot
    )

    assert all(cell["key"]["canon_on"] is False for cell in together)
    assert f"Every one of the {len(together)} is the layer **off**" in slot


def test_the_reserved_slots_inverted_auc_is_the_one_results_json_carries(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """The threshold-free corroboration, which is a different measurement and not the same rates.

    Every coordinate the sentence attributes the two AUCs to -- the baseline, the benign class and
    the canon state -- is re-derived from the cells and asserted adjacent to the values, so moving
    the finding onto the canonicalized condition or onto the other baseline is a red test.
    """

    def auc_cell(chain: str) -> dict[str, Any]:
        found = [
            cell
            for cell in results["cells"]
            if cell["kind"] == "auc"
            and cell["key"]["baseline"] == "protectai-deberta-v3"
            and cell["key"]["dressing_chain"] == chain
            and cell["key"]["benign_class"] == "b_code"
            and cell["key"]["canon_on"] is False
            and cell["key"]["window_policy"] == "shared"
            and cell["key"]["population"] == "all"
        ]
        assert len(found) == 1, f"expected exactly one b_code AUC cell for {chain!r}, layer off"
        return found[0]

    first, second = auc_cell("hex"), auc_cell("base64+homoglyph")
    assert float(first["value"]) < 0.5 and float(second["value"]) < 0.5, (
        "the caveat calls these an inverted ordering"
    )

    key = first["key"]
    assert second["key"]["baseline"] == key["baseline"]
    assert second["key"]["benign_class"] == key["benign_class"]
    assert second["key"]["canon_on"] == key["canon_on"] is False
    state = "off" if key["canon_on"] is False else "on"

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert (
        f"**the AUC for attacks at `{key['baseline']}` against `{key['benign_class']}` with the "
        f"layer {state} is {float(first['value']):.4f} on `{key['dressing_chain']}` and "
        f"{float(second['value']):.4f} on `{second['key']['dressing_chain']}`**" in slot
    )


def test_the_reserved_slot_says_n3_triggered_and_by_how_much(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """N3 is the one condition that fired, and slot 8 said everything about it except that.

    The paragraph explained why the relative limb was inert, inside a passage headed "the ways the
    design itself came up short", and never stated the outcome. The outcome word and the condition
    name are both derived here, so attributing the trigger to another condition is a red test.
    """
    triggered = [
        entry["condition"] for entry in results["verdict"] if entry["outcome"] == "triggered"
    ]
    assert triggered == ["N3"], "slot 8 names exactly one triggered condition"
    verdict = {entry["condition"]: entry for entry in results["verdict"]}["N3"]
    computed = verdict["computed"]
    over = computed["layer_p95_ns"] / computed["ceiling_ns"]
    assert over > 1

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert f"**{triggered[0]} triggered, and its relative limb never bound.**" in slot
    assert (
        f"**{computed['layer_p95_ns']:,} ns against a ceiling of "
        f"{computed['ceiling_ns']:,.0f} ns, {over:.1f}× over it**" in slot
    )
    assert (
        f"of the {NUMBER_WORDS[len(results['verdict'])]} pre-registered falsification conditions "
        f"this is the one whose `outcome` reads `{verdict['outcome']}`" in slot
    )


def test_the_reserved_slots_latency_ceiling_formula_is_the_one_the_run_applied(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """The ceiling is a formula, and the formula's own constants are what a reader checks it by."""
    computed = {entry["condition"]: entry for entry in results["verdict"]}["N3"]["computed"]
    fraction = computed["share_ceiling_ns"] / computed["fastest_baseline_p95_ns"]
    absolute_ms = computed["absolute_ceiling_ns"] / 1e6
    assert computed["ceiling_ns"] == min(
        computed["share_ceiling_ns"], computed["absolute_ceiling_ns"]
    )
    assert computed["binding_ceiling"] == "the absolute"

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert f"`min({fraction:g} × fastest baseline p95, {absolute_ms:g} ms)`" in slot
    assert (
        f"`share_ceiling_ns` {computed['share_ceiling_ns'] / 1e6:.2f} ms against "
        f"`absolute_ceiling_ns` {computed['absolute_ceiling_ns'] / 1e6:.2f} ms "
        f"with `binding_ceiling` `{computed['binding_ceiling']}`" in slot
    )


def test_the_reserved_slot_names_the_confirmatory_cell_and_which_half_saturated(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """Slot 8 defines the saturation as a rate AND a recall both pinned; N1's cell is only half.

    `delta_recall` on that cell is 0.1658, so the recall moved. Saying it "landed inside the
    saturation described above" claims a property the record contradicts, and the narrowed claim
    is bound here together with the coordinates of the cell it is about.
    """
    computed = {entry["condition"]: entry for entry in results["verdict"]}["N1"]["computed"]
    cell = computed["confirmatory_cell"]
    pinned = computed["pinned_rates"][0]
    assert computed["cell_could_decide"] is False
    assert pinned["pinned_at"] == 1.0
    assert pinned["k_canon_off"] == pinned["k_canon_on"] == pinned["n"]
    assert abs(computed["delta_recall"]) >= 0.01, (
        "the recall on the confirmatory cell is saturated after all; the caveat's narrowing is stale"
    )

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert (
        f"**`{cell['baseline']}` on `{cell['dressing_chain']}` against `{cell['benign_class']}`**, "
        f"and its false-positive rate is pinned at {pinned['pinned_at']:.4f} with "
        f"{pinned['n']} of {pinned['n']} in both canon states" in slot
    )
    assert "`cell_could_decide` `no`" in slot
    assert "landed on the **false-positive half** of the saturation described above" in slot
    assert f"`delta_recall` there is **{computed['delta_recall']:.4f}**" in slot
    assert "landed inside the saturation described above" not in slot


def test_the_reserved_slot_says_which_held_out_encodings_the_condition_covers(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """N4's condition runs over two of the three held-out encodings, and the third is out by design.

    `held_out_chains` carries `base32` and `url_percent`; `rot13` sits in `excluded_probes_none`
    because caveat 7 holds it out to mark the boundary rather than to be recovered. "The three
    encodings held out for exactly that question" claimed a condition wider than the one that ran.
    """
    computed = {entry["condition"]: entry for entry in results["verdict"]}["N4"]["computed"]
    held_out = sorted(set(computed["held_out_chains"]))
    excluded = sorted(set(computed["excluded_probes_none"]))
    assert computed["held_out_chains_recovering"] == []
    assert len(held_out) == 2 and len(excluded) == 1
    assert not set(held_out) & set(excluded)

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert "`held_out_chains_recovering` is `none`" in slot
    assert (
        f"over **`{held_out[0]}` and `{held_out[1]}`, {NUMBER_WORDS[len(held_out)]} of the "
        f"{NUMBER_WORDS[len(held_out) + len(excluded)]} held out**" in slot
    )
    assert (
        f"because caveat 7's `{excluded[0]}` is there to mark the boundary of what normalization "
        f"can reach rather than to be recovered, and is excluded from the pass/fail condition"
        in slot
    )

    seven = one_caveat(caveats_section, "7")
    total = NUMBER_WORDS[len(held_out) + len(excluded)].capitalize()
    assert f"{total} encodings are held out" in seven


def test_the_reserved_slot_does_not_overstate_what_the_layer_failed_to_generalize(
    caveats_section: str, results: dict[str, Any]
) -> None:
    """"Generalized to nothing that could be measured" is wider than the record supports.

    `chains_recovering_off_distribution` carries one chain. The narrow true claim is about the
    held-out chains, and the wider one is corrected in the prose and bound here.
    """
    computed = {entry["condition"]: entry for entry in results["verdict"]}["N4"]["computed"]
    recovering = sorted(set(computed["chains_recovering_off_distribution"]))
    assert len(recovering) == 1, "the caveat's qualification names exactly one such chain"
    chain = recovering[0]
    assert chain in set(computed["over_ceiling_chains"])
    assert chain not in set(computed["held_out_chains"])

    slot = one_caveat(caveats_section, RESERVED_LABEL)
    assert (
        f"`chains_recovering_off_distribution` carries **`{chain}`**, a chain past the declared "
        f"recursion ceiling rather than an encoding the layer was never written against" in slot
    )
    assert "the layer generalized to nothing that could be measured" not in slot


def test_caveat_five_states_the_window_overflow_directions_the_run_measured(
    caveats_section: str,
    window_overflow: dict[tuple[Any, ...], dict[Any, dict[str, Any]]],
) -> None:
    """Caveat 5 asserted one direction for both baselines; the run measures opposite ones.

    Each share is asserted adjacent to the name of the baseline it was selected by, so swapping
    the two baseline names in the caveat -- the exact inversion this caveat exists to report --
    goes red instead of green.
    """
    body = one_caveat(caveats_section, "5")
    directions = (("protectai-deberta-v3", "shortens"), ("testsavantai-bert-small", "lengthens"))
    for baseline, direction in directions:
        off, on = overflow_share(window_overflow, lambda axes: axes[0] == baseline)
        if direction == "shortens":
            assert on < off, f"the caveat says the layer shortens for {baseline}"
        else:
            assert on > off, f"the caveat says the layer lengthens for {baseline}"
        assert f"**{off:.2f}% → {on:.2f}% for `{baseline}`**" in body


def test_caveat_fives_spot_cell_is_the_one_the_census_counted(
    caveats_section: str,
    window_overflow: dict[tuple[Any, ...], dict[Any, dict[str, Any]]],
) -> None:
    """The 3-of-500 to 395-of-500 cell, with every coordinate it is attributed to bound beside it."""
    baseline, chain, chain_class = "testsavantai-bert-small", "base64", "bound"
    family, benign_class = "benign", "b_code"
    pair = window_overflow[
        (baseline, chain, chain_class, family, benign_class, "shared", "all")
    ]
    assert pair[True]["k"] > pair[False]["k"], "the caveat reads this cell as the layer lengthening"

    body = one_caveat(caveats_section, "5")
    assert (
        f"`{baseline}` on `{benign_class}` with `{chain}` goes from "
        f"**{pair[False]['k']} of {pair[False]['n']}** items over one window with the layer "
        f"**off** to **{pair[True]['k']} of {pair[True]['n']}** with it **on**" in body
    )


def test_caveat_fives_b_code_half_is_the_share_the_census_counted(
    caveats_section: str,
    window_overflow: dict[tuple[Any, ...], dict[Any, dict[str, Any]]],
) -> None:
    """The half-corpus aggregate behind the spot cell, bound to both of its coordinates."""
    baseline, benign_class = "testsavantai-bert-small", "b_code"
    off, on = overflow_share(
        window_overflow, lambda axes: axes[0] == baseline and axes[4] == benign_class
    )
    assert on > off

    body = one_caveat(caveats_section, "5")
    assert (
        f"**`{baseline}`'s whole `{benign_class}` half moves {off:.2f}% → {on:.2f}%**" in body
    )


def test_caveat_five_counts_the_windows_matched_twins_the_file_actually_carries(
    caveats_section: str,
    results: dict[str, Any],
    window_overflow: dict[tuple[Any, ...], dict[Any, dict[str, Any]]],
) -> None:
    """The caveat published a universal the file refutes: 77 of 274 delta cells carry a twin.

    The quantifier is bound to the count rather than written as "every", and the single
    canon-on-versus-off rate delta without a twin is named with the census that explains it.
    """
    axes = (
        "baseline",
        "dressing_chain",
        "chain_class",
        "window_policy",
        "canon_on",
        "family",
        "benign_class",
        "contrast",
    )
    deltas = [cell for cell in results["cells"] if cell["kind"] == "delta"]
    whole = [cell for cell in deltas if cell["key"]["population"] == "all"]
    twins = {
        tuple(cell["key"][axis] for axis in axes)
        for cell in deltas
        if cell["key"]["population"] == "single_window"
    }
    twinned = [cell for cell in whole if tuple(cell["key"][axis] for axis in axes) in twins]
    rate_deltas = [
        cell
        for cell in whole
        if cell["key"]["contrast"] == "canon_on_vs_off" and cell["key"]["family"] is not None
    ]
    without = [
        cell for cell in rate_deltas if tuple(cell["key"][axis] for axis in axes) not in twins
    ]

    assert len(twinned) < len(whole), "every delta does carry a twin; the correction is stale"
    assert len(without) == 1, "the caveat names exactly one rate delta without a twin"
    key = without[0]["key"]
    census = window_overflow[
        (
            key["baseline"],
            key["dressing_chain"],
            key["chain_class"],
            key["family"],
            key["benign_class"],
            key["window_policy"],
            "all",
        )
    ]
    assert census[False]["k"] == census[False]["n"], (
        "the caveat explains the missing twin by every item overflowing with the layer off"
    )

    body = one_caveat(caveats_section, "5")
    assert (
        f"accompanies **{len(twinned)} of the {len(whole)}** delta cells the block above publishes"
        in body
    )
    assert (
        f"The twins cover **{len(twinned)} of the {len(rate_deltas)}** canon-on-versus-off rate "
        f"deltas" in body
    )
    assert (
        f"the one without is **`{key['baseline']}` on `{key['dressing_chain']}` against "
        f"`{key['benign_class']}`**, where **{census[False]['k']} of {census[False]['n']}** items "
        f"exceed one window with the layer **off**" in body
    )
    assert f"The remaining **{len(whole) - len(rate_deltas)}** delta cells" in body


def test_the_unkeepable_per_baseline_promise_is_gone_from_the_whole_section(
    caveats_section: str,
) -> None:
    """The block publishes `window_overflow` per cell and no aggregate over them.

    5.3 withdrew that promise from caveat 5 and left the same claim standing in 3b, which is where
    it had also been made. The absence is asserted over the section rather than over one caveat,
    so it cannot be reintroduced next door.
    """
    flat = " ".join(caveats_section.split())
    for promise in (
        "the share of items needing more than one window is given per baseline",
        "the per-baseline share of items that exceeded one window is reported alongside the rates",
    ):
        assert promise not in flat


def test_caveat_three_ds_exclusion_totals_are_the_ones_the_manifest_records(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """What the training-overlap filter removed, in total."""
    exclusion = manifest["reports"]["exclusion"]
    body = one_caveat(caveats_section, "3d")
    assert f"**{exclusion['rows_removed']:,} of {exclusion['rows_in']:,} rows**" in body
    assert f"leaving **{exclusion['rows_kept']:,}**" in body


def test_caveat_three_ds_benign_half_removal_is_the_one_the_manifest_records(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """The conversational benign half of that removal, bound to the half it is attributed to."""
    b_chat = manifest["reports"]["benign_draw"]["b_chat"]
    body = one_caveat(caveats_section, "3d")
    assert (
        f"**{b_chat['rows_removed_by_exclusion']:,} of {b_chat['rows_in']:,}** "
        f"on the conversational benign half" in body
    )


def test_caveat_three_ds_attack_half_removal_is_the_one_the_manifest_records(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """And the attack half, likewise bound to its half."""
    attack = manifest["reports"]["attack_draw"]
    body = one_caveat(caveats_section, "3d")
    assert (
        f"**{attack['removed_by_exclusion']:,} of {attack['unique_positives']:,}** "
        f"on the attack half" in body
    )


def test_caveat_three_d_says_why_its_two_benign_denominators_differ(
    caveats_section: str, manifest: dict[str, Any], repo_root: Path
) -> None:
    """3d prints 7,066 and 7,064 eleven lines apart for what reads as one population.

    They are two stages: a probe taken before the build, recorded in `pins.toml`, and the build's
    own `rows_in`. Both are labelled in the prose and each is bound to the file that carries it.
    """
    b_chat = manifest["reports"]["benign_draw"]["b_chat"]
    pins = (repo_root / "pins.toml").read_text(encoding="utf-8")
    recorded = re.search(
        r"`([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)` alone is (\d+) of (\d+) unique benign rows -- (\d+)%",
        pins,
    )
    assert recorded is not None
    probe_total = int(recorded.group(3))
    assert probe_total != b_chat["rows_in"], (
        "the two denominators agree now; the sentence explaining the difference is stale"
    )

    body = one_caveat(caveats_section, "3d")
    assert (
        f"**{probe_total:,}** is the 2026-08-24 probe recorded in `pins.toml`, "
        f"**{b_chat['rows_in']:,}** is `rows_in` for B-chat in the build that produced this table, "
        f"a difference of {probe_total - b_chat['rows_in']} rows" in body
    )


def test_caveat_three_ds_per_source_attribution_is_the_manifests(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """The two sources that attributed anything, each with its own count and a terminator.

    The counts are asserted with the words that follow them, because `47` on its own is an
    unbounded substring: editing the page to `470` left the suite green while the itemisation
    stopped adding up to the total the next clause discloses.

    The repository ids are read out of the manifest rather than typed: `test_pins.py`'s scan
    forbids a pinned `namespace/name` as a literal in a test, and it is right to -- a pin with a
    second home in the suite is a pin that can drift while the suite stays green.
    """
    exclusion = manifest["reports"]["exclusion"]
    attributing = sorted(
        (source for source in exclusion["sources"] if source["matched_rows"]),
        key=lambda source: source["matched_rows"],
        reverse=True,
    )
    assert len(attributing) == 2, "3d names the two sources that attributed anything"
    largest, second = attributing

    body = one_caveat(caveats_section, "3d")
    assert f"`{largest['repository']}` {largest['matched_rows']} matched rows" in body
    assert f"`{second['repository']}` {second['matched_rows']} matched rows" in body


def test_caveat_three_ds_attribution_gap_is_the_one_the_manifest_leaves(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """The headline gap between what the build acted on and what it can attribute."""
    exclusion = manifest["reports"]["exclusion"]
    attributed = sum(
        source["matched_rows"] for source in exclusion["sources"] if source["matched_rows"]
    )
    assert attributed < exclusion["rows_removed"], "the caveat discloses a gap; there is none"

    body = one_caveat(caveats_section, "3d")
    assert f"**{attributed:,} attributed against {exclusion['rows_removed']:,} removed**" in body


def test_caveat_three_d_does_not_average_the_two_halves_of_the_attribution_gap(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """562-against-4,061 averages a complete half and a near-empty one, and hides the empty one.

    On the attack half the attribution is exact. On the conversational benign half at most one
    source's count can apply, and it is two orders of magnitude short of what was removed.
    """
    exclusion = manifest["reports"]["exclusion"]
    attack = manifest["reports"]["attack_draw"]
    b_chat = manifest["reports"]["benign_draw"]["b_chat"]

    attributing = sorted(
        (source for source in exclusion["sources"] if source["matched_rows"]),
        key=lambda source: source["matched_rows"],
        reverse=True,
    )
    attack_source, benign_source = attributing
    assert attack_source["matched_rows"] == attack["removed_by_exclusion"], (
        "the attack half is no longer fully attributed; the caveat says it is"
    )
    assert benign_source["matched_rows"] * 10 < b_chat["rows_removed_by_exclusion"], (
        "the benign half is no longer near-total absence; the caveat says it is"
    )

    body = one_caveat(caveats_section, "3d")
    assert (
        f"**{attack['removed_by_exclusion']:,} attributed against "
        f"{attack_source['matched_rows']:,} removed**" in body
    )
    assert (
        f"**{b_chat['rows_removed_by_exclusion']:,} rows were removed and at most "
        f"{benign_source['matched_rows']} are attributed**" in body
    )
    assert (
        f"`{benign_source['repository']}` is the only non-zero source that is not the attack "
        f"half's own {attack_source['matched_rows']}" in body
    )


def test_caveat_three_d_names_the_two_sources_the_filter_could_not_read(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """The unmeasurable half of the disclosure, held to the count the manifest publishes."""
    exclusion = manifest["reports"]["exclusion"]
    assert len(exclusion["unread_sources"]) == 2, "3d says there are two unmeasurable sources"
    body = one_caveat(caveats_section, "3d")
    assert f"and there are {NUMBER_WORDS[len(exclusion['unread_sources'])]}" in body


def test_caveat_three_ds_attack_side_overlap_is_the_one_the_manifest_measured(
    caveats_section: str, manifest: dict[str, Any]
) -> None:
    """The 17% half of 3d's opening pair, held to the draw report it was read out of."""
    attack = manifest["reports"]["attack_draw"]
    exclusion = manifest["reports"]["exclusion"]
    percent = round(100 * attack["removed_by_exclusion"] / attack["unique_positives"])
    body = one_caveat(caveats_section, "3d")
    assert (
        f"**{attack['removed_by_exclusion']:,} of {attack['unique_positives']:,} "
        f"unique positives removed, {percent}%**" in body
    )

    # And the caveat names the source that reach runs through, read out of the manifest rather than
    # typed, for the reason the per-source test above gives.
    through = max(exclusion["sources"], key=lambda source: source["matched_rows"] or 0)
    assert through["matched_rows"] == attack["removed_by_exclusion"]
    assert f"the one-hop reach through `{through['repository']}`" in body


def test_caveat_three_ds_pre_filter_reach_is_the_figure_pins_toml_carries(
    caveats_section: str, repo_root: Path
) -> None:
    """The 48% half. It is a pre-filter measurement, and `pins.toml` is the only file that records it.

    It was published attributed to the wrong thing -- to the attack card's two one-hop seeds, whose
    measured reach the manifest puts in the hundreds -- while the figure it names is the pre-filter
    reach of one directly declared training source. The attribution is corrected in the caveat and
    the number is bound here, so the two cannot drift apart again. The repository id comes out of
    the regex rather than a literal, for the reason `test_pins.py`'s scan gives.
    """
    pins = (repo_root / "pins.toml").read_text(encoding="utf-8")
    recorded = re.search(
        r"`([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)` alone is (\d+) of (\d+) unique benign rows -- (\d+)%",
        pins,
    )
    assert recorded is not None, "pins.toml no longer records the pre-filter reach 3d prints"
    repository = recorded.group(1)
    reached, total, percent = (int(group) for group in recorded.groups()[1:])
    assert round(100 * reached / total) == percent

    body = one_caveat(caveats_section, "3d")
    assert (
        f"**`{repository}` alone at {reached:,} of {total:,} "
        f"unique benign rows, {percent}%**" in body
    )


def test_caveat_three_d_says_the_forty_eight_percent_source_attributed_nothing(
    caveats_section: str, manifest: dict[str, Any], repo_root: Path
) -> None:
    """The headline source of 3d's largest figure sits at `matched_rows: 0` in the same build.

    A reader who opens the manifest to check the 48% finds a zero. That is part of the same
    attribution gap the caveat already discloses, and the caveat now says so rather than leaving
    the reader to discover a contradiction that is not one.
    """
    pins = (repo_root / "pins.toml").read_text(encoding="utf-8")
    recorded = re.search(
        r"`([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)` alone is (\d+) of (\d+) unique benign rows -- (\d+)%",
        pins,
    )
    assert recorded is not None
    repository = recorded.group(1)

    exclusion = manifest["reports"]["exclusion"]
    source = next(
        entry for entry in exclusion["sources"] if entry["repository"] == repository
    )
    assert source["read"] is True, "the source was unreadable; the caveat says it was read"
    assert source["matched_rows"] == 0, "it no longer attributes zero; the caveat says it does"

    body = one_caveat(caveats_section, "3d")
    assert f"**`{repository}` is one of those zeros.**" in body
    assert f"{source['texts_loaded']:,} texts loaded" in body
    assert f"attributes **{source['matched_rows']} matched rows** to it" in body


def test_a_construct_validity_pointer_precedes_the_generated_block(repo_root: Path) -> None:
    """3c is the strongest objection on the page and its label keeps it sixth of twelve.

    Reordering the section is a `caveats.py` abort, so what moves is the reader's path: the pointer
    mechanism the page already uses for caveats 3 and 6, now used for 3c above the table a reader
    forms a verdict from.
    """
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    above = " ".join(text[: text.index(RESULTS_START)].split())
    assert 'caveat 3c in ["what this does not show"](#what-this-does-not-show)' in above
    assert "not whether an attack works" in above
    assert above.count("(#what-this-does-not-show)") >= 3


# --- every way the section can fail ------------------------------------------------------------


def test_an_absent_section_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=""))
    assert CAVEATS_HEADING in str(abort.value)


def test_an_empty_section_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=CAVEATS_HEADING + "\n\n   \n\n"))
    assert "empty" in str(abort.value)


@pytest.mark.parametrize("dropped", REQUIRED_LABELS)
def test_dropping_any_required_caveat_aborts_and_names_it(dropped: str) -> None:
    kept = tuple(label for label in REQUIRED_LABELS if label != dropped)
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(labels=kept)))
    assert f"caveat(s) {dropped} are missing" in str(abort.value)


def test_a_thin_caveat_aborts() -> None:
    thin = section().replace(caveat("6"), "**6.** the layer is a surface.")
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=thin))
    message = str(abort.value)
    assert "6 (" in message and "worse than none" in message


def test_caveats_published_out_of_the_prds_order_abort() -> None:
    swapped = list(REQUIRED_LABELS)
    here, there = swapped.index("3c"), swapped.index("3d")
    swapped[here], swapped[there] = swapped[there], swapped[here]
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(labels=tuple(swapped))))
    assert "not the PRD's" in str(abort.value)


def test_a_duplicated_label_aborts() -> None:
    doubled = section() + "\n" + caveat("3c") + "\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=doubled))
    assert "more than once" in str(abort.value)


def test_a_missing_reserved_slot_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section(reserved=None)))
    assert f"reserved slot {RESERVED_LABEL} is missing" in str(abort.value)


def test_a_reserved_slot_published_before_the_last_caveat_aborts() -> None:
    early = CAVEATS_HEADING + "\n\n" + f"**{RESERVED_LABEL}.** *(reserved)*\n\n"
    early += "\n\n".join(caveat(label) for label in REQUIRED_LABELS) + "\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=early))
    assert "is the last slot" in str(abort.value)


def test_a_section_inside_the_generated_block_aborts() -> None:
    inside = f"{RESULTS_START}\n\n{section()}\n{RESULTS_END}\n"
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats="", markers=inside))
    assert "generated and replaced on every run" in str(abort.value)


@pytest.mark.parametrize(
    "markers",
    [
        pytest.param("", id="no-markers"),
        pytest.param(f"{RESULTS_START}\n", id="start-only"),
        pytest.param(f"{RESULTS_END}\n", id="end-only"),
        pytest.param(f"{RESULTS_START}\n{RESULTS_START}\n{RESULTS_END}\n", id="doubled-start"),
    ],
)
def test_a_malformed_marker_pair_aborts(markers: str) -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(markers=markers))
    assert "not delimited exactly once" in str(abort.value)


def test_an_inverted_marker_pair_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(markers=f"{RESULTS_END}\n\n{RESULTS_START}\n"))
    assert "inverted" in str(abort.value)


def test_a_duplicated_heading_aborts() -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats(readme(caveats=section() + "\n" + section()))
    assert "appears 2 times" in str(abort.value)


def test_an_unreadable_readme_aborts_rather_than_crashing(tmp_path: Path) -> None:
    with pytest.raises(CaveatsSectionMissing) as abort:
        verify_caveats_file(tmp_path / "nope.md")
    assert "could not be read" in str(abort.value)


def test_a_section_that_runs_to_the_end_of_the_file_is_still_read() -> None:
    report = verify_caveats(readme(trailing=""))
    assert report.labels == (*REQUIRED_LABELS, RESERVED_LABEL)


# --- the abort itself ---------------------------------------------------------------------------


def test_the_abort_is_one_of_the_projects_declared_aborts() -> None:
    abort = CaveatsSectionMissing("because")
    assert isinstance(abort, NbcError)
    assert exit_code_for(abort) == CaveatsSectionMissing.exit_code == 11
    assert abort.failures == ("because",)


# --- the command line, and what it must not import ----------------------------------------------


def test_the_command_line_reports_the_sections_labels(repo_root: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.report.caveats", "--readme", str(repo_root / "README.md")],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert '"caveats_check": "ok"' in completed.stdout


def test_the_command_line_exits_with_the_aborts_own_code(tmp_path: Path, repo_root: Path) -> None:
    broken = tmp_path / "README.md"
    broken.write_text(readme(caveats=""), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "nbc.report.caveats", "--readme", str(broken)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert completed.returncode == CaveatsSectionMissing.exit_code == 11
    assert CAVEATS_HEADING in completed.stderr
    assert completed.stdout == ""


def test_the_check_does_not_import_the_inference_runtime() -> None:
    """AD-16 runs this before any inference; a check that imported the runtime would not be before it."""
    code = "import sys, nbc.report.caveats; print('onnxruntime' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout
