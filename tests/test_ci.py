"""What the two workflows promise, read from the files rather than trusted.

CI is where every claim this repository makes about itself is proved on a machine nobody
configured. That makes the workflows themselves load-bearing, and it makes two of their properties
worth asserting here rather than discovering at a push:

*The bandwidth decision holds.* Every checkout in `ci.yml` sets `lfs: false`, because the objects
cost about 130 MB per job per push against a 1 GB monthly allowance billed to the repository owner.
Exactly one checkout in the whole repository sets `lfs: true`, and it is the tag job whose subject
is the corpus itself.

*Every gate checks an exit code.* `if the command failed; then fail; fi` is green whether the abort
fired or the command died of a typo, an import error or a missing module -- a gate that passes on
its own failure. The workflows already do this everywhere; this is what keeps the next step from
being the exception.

Read as text and not as YAML on purpose: this repository does not depend on a YAML parser, and
adding one so a test can read a file whose failure mode is "a line was deleted" would be a
dependency bought for a check that a substring makes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
CORPUS = WORKFLOWS / "corpus.yml"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def settings(path: Path, key: str, value: str) -> list[int]:
    """Lines where `key: value` is an actual setting rather than a mention in a comment.

    The distinction is not pedantry: both workflows discuss `lfs: false` in prose -- the comment
    explaining the bandwidth decision names the setting it explains -- and counting those as
    settings made this test claim five checkouts where there are three.
    """
    found = []
    for number, line in enumerate(text(path).splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == f"{key}: {value}":
            found.append(number)
    return found


def test_both_workflows_exist() -> None:
    assert CI.is_file()
    assert CORPUS.is_file(), "the corpus determinism proof is its own workflow, on a tag"


def test_every_checkout_in_the_push_workflow_leaves_lfs_off() -> None:
    """The cost decision, asserted where deleting a line is what breaks it.

    Turning one of these on would not fail anything visibly -- it would spend the month's
    allowance and make the "nothing reads the corpus without its manifest" step green for a
    different reason than the one it claims.
    """
    body = text(CI)
    checkouts = body.count("actions/checkout@")
    assert checkouts == 3, f"ci.yml has {checkouts} checkouts; each needs its own lfs decision"
    assert len(settings(CI, "lfs", "false")) == checkouts
    assert settings(CI, "lfs", "true") == []


def test_exactly_one_checkout_in_the_repository_fetches_the_corpus() -> None:
    """And it is the tag job. A second one would be a second place the bandwidth is spent."""
    fetching = [
        path.name for path in sorted(WORKFLOWS.glob("*.yml")) if settings(path, "lfs", "true")
    ]
    assert fetching == [CORPUS.name]
    assert len(settings(CORPUS, "lfs", "true")) == 1


def test_the_corpus_workflow_runs_on_tags_and_not_on_every_push() -> None:
    body = text(CORPUS)
    assert "tags:" in body
    assert re.search(r"^\s+branches:", body, re.MULTILINE) is None, (
        "a branch trigger here would spend the LFS allowance on every push, which is the decision "
        "ci.yml records and this workflow exists to respect"
    )


def test_the_corpus_workflow_compares_the_rebuild_byte_for_byte() -> None:
    """`cmp` and not a row count: a rebuild that produced the same rows in a different order is a
    different file, and the stable order is part of AD-1's claim rather than an implementation
    detail."""
    body = text(CORPUS)
    assert "build-corpus" in body
    assert "cmp -s" in body
    assert "verify-corpus" in body


def test_the_corpus_workflow_proves_the_objects_were_actually_fetched() -> None:
    """Without it the rebuild could pass over LFS pointers, which are a few hundred bytes and would
    "match" if anybody ever compared pointers to pointers."""
    assert "is an LFS pointer rather than the file" in text(CORPUS)


def test_the_smoke_run_is_proved_not_to_touch_the_published_results() -> None:
    body = text(CORPUS)
    assert "the smoke run wrote the published results file" in body
    assert '--profile smoke' in body


def test_both_workflows_refuse_a_dirty_tree() -> None:
    """`git diff` cannot see a file that did not exist before the run, which is the shape a stray
    write actually takes -- so both check `git status --porcelain` as well."""
    for path in (CI, CORPUS):
        body = text(path)
        assert "git diff --exit-code" in body, path.name
        assert "git status --porcelain" in body, path.name


def exit_code_checks(body: str) -> list[str]:
    """Every step that captures a status and compares it to a number."""
    return re.findall(r'if \[ "\$status" -ne (\d+) \]', body)


def test_every_gate_compares_an_exit_code_to_a_number() -> None:
    """The shape that is NOT here is `if ! command; then fail; fi`, which is green whether the
    abort fired or the command died of a typo. Every captured status in both workflows is compared
    to a specific number."""
    for path in (CI, CORPUS):
        body = text(path)
        captured = body.count("status=$?")
        compared = len(exit_code_checks(body))
        assert captured == compared, (
            f"{path.name}: {captured} steps capture a status and {compared} compare it to a code"
        )


def test_the_push_workflow_asserts_the_codes_the_aborts_actually_declare() -> None:
    """The codes in the workflow against the codes the classes declare -- two different places, so
    a renumbered abort fails here rather than making a CI step assert a code nothing raises."""
    from nbc.errors import declared_exit_codes
    from nbc.harness.results import ResultsIncomplete
    from nbc.corpus.manifest import CorpusManifestMismatch
    from nbc.platform import UnsupportedPlatform
    from nbc.report.size_budget import SizeBudgetViolated
    from nbc.pins import PinsFileInvalid

    codes = declared_exit_codes()
    for expected in (
        UnsupportedPlatform,
        PinsFileInvalid,
        CorpusManifestMismatch,
        SizeBudgetViolated,
        ResultsIncomplete,
    ):
        assert codes[expected.exit_code] is expected
        assert str(expected.exit_code) in exit_code_checks(text(CI)), (
            f"ci.yml asserts no step against {expected.__name__} (exit {expected.exit_code})"
        )


def test_the_lockfile_gate_uses_locked_and_not_frozen() -> None:
    """`--frozen` exits 0 on a stale lock and installs it silently, verified against uv 0.12.5.
    `--frozen` stays as the documented reproduction command; the gate that refuses drift is
    `--locked`, and this is what keeps the two from being swapped by somebody tidying."""
    body = text(CI)
    assert "uv sync --locked" in body
    assert "DOES NOT VALIDATE" in body


@pytest.mark.parametrize("path", [CI, CORPUS])
def test_no_workflow_writes_the_published_results_file(path: Path) -> None:
    """The one thing CI must never do. A smoke run that overwrote the published table would ship a
    handful of items under the headline's name, with a small n as the only tell."""
    body = text(path)
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # the comments say the file must not be written, which is not writing it
        assert not re.search(r">\s*results/results\.json", stripped), stripped
        assert not re.search(r">\s*README\.md", stripped), stripped


def test_the_setting_scan_tells_a_setting_from_a_mention_of_one() -> None:
    """Its own red input, both ways: the comment above line 52 discusses `lfs: false` and is not
    one, and a test that could not tell them apart counted five checkouts where there are three."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "w.yml"
        path.write_text(
            "steps:\n  # turning lfs: true on would spend the allowance\n  - with:\n"
            "      lfs: false\n",
            encoding="utf-8",
        )
        assert settings(path, "lfs", "false") == [4]
        assert settings(path, "lfs", "true") == []
