"""The corpus can be pushed, and a clone that got pointers is told so.

Two gates that did not exist until 2026-08-30, and the reason they exist is one measurement: the
benign half is 130.4 MB, GitHub refuses a push over 100 MB, and nothing in the repository would
have said a word. The failure would have arrived at a server, after a commit.

Both are checked here against the **committed** `.gitattributes` as well as against synthetic
inputs, because a gate that only ever sees fixtures is a gate nobody has pointed at the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nbc.corpus import build
from nbc.corpus.manifest import (
    BENIGN_CORPUS_FILENAME,
    CORPUS_FILENAMES,
    DATA_DIRNAME,
    LFS_POINTER_VERSION,
    lfs_pointer_problem,
)
from nbc.corpus.storage import (
    GITATTRIBUTES_FILENAME,
    GITHUB_PUSH_REFUSES_BYTES,
    GITHUB_WARNS_BYTES,
    CorpusNotPublishable,
    covers,
    lfs_patterns,
    storage_problems,
)
from nbc.errors import declared_exit_codes, exit_code_for

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURED_BENIGN_BYTES = 130_382_057
"""What `build-corpus` actually wrote on 2026-08-30. The number this whole module is about."""


@pytest.fixture(scope="module")
def committed_gitattributes() -> str:
    return (REPO_ROOT / GITATTRIBUTES_FILENAME).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pins():
    from nbc.pins import load_pins

    return load_pins(REPO_ROOT)


def pointer(size: int = MEASURED_BENIGN_BYTES, oid: str = "b4044227" * 8) -> bytes:
    return (
        f"version {LFS_POINTER_VERSION}\noid sha256:{oid}\nsize {size}\n"
    ).encode("utf-8")


# --- the declaration this repository actually ships -------------------------------------------


def test_the_committed_declaration_tracks_the_half_that_needs_it(
    committed_gitattributes: str,
) -> None:
    """The gate pointed at the artifact rather than at a fixture.

    The two sides come from different files: the size is what the build measured, and the coverage
    is what a person wrote in `.gitattributes`. Comment the tracking line out and this goes red.
    """
    assert storage_problems(
        [(BENIGN_CORPUS_FILENAME, MEASURED_BENIGN_BYTES)],
        committed_gitattributes,
        directory=DATA_DIRNAME,
    ) == ()


def test_the_declaration_does_not_track_the_half_that_does_not_need_it(
    committed_gitattributes: str,
) -> None:
    """`attack.jsonl` is 7.2 MB and packs to 1.7 MB in ordinary git.

    Tracking it would spend LFS storage, which on the free tier does not reset and which GitHub
    does not garbage-collect, to make a diffable file undiffable. `data/*.jsonl` would have done
    exactly that, which is why the declaration names one path.
    """
    tracked = lfs_patterns(committed_gitattributes)

    assert tracked, "nothing is tracked at all; this module has lost its subject"
    assert not any(
        covers(pattern, f"{DATA_DIRNAME}/{name}")
        for pattern in tracked
        for name in CORPUS_FILENAMES
        if name != BENIGN_CORPUS_FILENAME
    )


def test_a_declaration_that_stopped_covering_the_corpus_aborts(
    committed_gitattributes: str,
) -> None:
    """The failing input for the test above, built by commenting the real line out.

    This is the input a person actually produces: the tracking line is still there, still readable,
    still says `filter=lfs`, and git no longer applies it.
    """
    commented = "\n".join(
        f"# {line}" if "filter=lfs" in line and not line.startswith("#") else line
        for line in committed_gitattributes.splitlines()
    )
    assert commented != committed_gitattributes, "the edit changed nothing"

    (problem,) = storage_problems(
        [(BENIGN_CORPUS_FILENAME, MEASURED_BENIGN_BYTES)], commented, directory=DATA_DIRNAME
    )

    assert "GitHub refuses a push" in problem
    assert "124.3 MB" in problem


# --- the size rule -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, expected",
    [
        pytest.param(GITHUB_WARNS_BYTES - 1, 0, id="under-the-warning"),
        pytest.param(GITHUB_WARNS_BYTES, 1, id="at-the-warning"),
        pytest.param(GITHUB_PUSH_REFUSES_BYTES, 1, id="at-the-refusal"),
    ],
)
def test_the_threshold_is_the_warning_and_not_the_refusal(size: int, expected: int) -> None:
    """Firing at the refusal would mean learning at 99 MB that nothing was ever checked."""
    assert len(storage_problems([("x.jsonl", size)], "", directory=DATA_DIRNAME)) == expected


def test_the_two_sizes_get_different_messages() -> None:
    """A push that will be refused and a push that will succeed are different problems."""
    (refused,) = storage_problems([("x", GITHUB_PUSH_REFUSES_BYTES)], "", directory=DATA_DIRNAME)
    (warned,) = storage_problems([("x", GITHUB_WARNS_BYTES)], "", directory=DATA_DIRNAME)

    assert "refuses a push" in refused
    assert "the push will succeed" in warned


def test_no_declaration_at_all_means_nothing_is_tracked() -> None:
    """An absent `.gitattributes` is not a reason to assume the best."""
    assert lfs_patterns("") == ()
    assert storage_problems(
        [(BENIGN_CORPUS_FILENAME, MEASURED_BENIGN_BYTES)], "", directory=DATA_DIRNAME
    ) != ()


def test_a_small_untracked_corpus_is_fine() -> None:
    """The rule is "small enough, or tracked", so the ordinary case has to pass."""
    assert storage_problems([("attack.jsonl", 7_241_240)], "", directory=DATA_DIRNAME) == ()


# --- parsing, not matching -----------------------------------------------------------------------


def test_a_commented_line_does_not_track_anything() -> None:
    assert lfs_patterns("# data/benign.jsonl filter=lfs diff=lfs\n") == ()


def test_the_word_lfs_somewhere_else_does_not_track_anything() -> None:
    """A substring search for "lfs" is satisfied by prose, by a path, and by another attribute."""
    assert lfs_patterns("lfs/notes.txt text\n") == ()
    assert lfs_patterns("*.bin diff=lfs\n") == (), "diff=lfs alone is not the storage filter"
    assert lfs_patterns("*.bin -filter\n") == ()


def test_a_real_tracking_line_is_found_whatever_else_it_carries() -> None:
    assert lfs_patterns(
        "*.md text eol=lf\n"
        "\n"
        "  data/benign.jsonl filter=lfs diff=lfs merge=lfs -text  \n"
    ) == ("data/benign.jsonl",)


@pytest.mark.parametrize(
    "pattern, path, matched",
    [
        pytest.param("data/benign.jsonl", "data/benign.jsonl", True, id="anchored-exact"),
        pytest.param("/data/benign.jsonl", "data/benign.jsonl", True, id="leading-slash"),
        pytest.param("data/*.jsonl", "data/benign.jsonl", True, id="anchored-glob"),
        pytest.param("*.jsonl", "data/benign.jsonl", True, id="basename-glob"),
        pytest.param("benign.jsonl", "data/benign.jsonl", True, id="basename-exact"),
        pytest.param("data/benign.jsonl", "other/benign.jsonl", False, id="wrong-directory"),
        pytest.param("data/attack.jsonl", "data/benign.jsonl", False, id="other-half"),
        pytest.param("*.json", "data/benign.jsonl", False, id="near-miss-extension"),
    ],
)
def test_pattern_coverage(pattern: str, path: str, matched: bool) -> None:
    assert covers(pattern, path) is matched


# --- the Git LFS pointer -----------------------------------------------------------------------


def test_a_pointer_is_reported_as_a_pointer_and_not_as_an_edit() -> None:
    """What a clone without the LFS filters holds, and what it needs to be told.

    Before this gate the content hash was what noticed, and it said "the file was edited after the
    build". The reader edited nothing, and rebuilding the corpus is not what fixes it.
    """
    problem = lfs_pointer_problem(BENIGN_CORPUS_FILENAME, pointer())

    assert problem is not None
    assert "git lfs pull" in problem
    assert str(MEASURED_BENIGN_BYTES) in problem
    assert "does not need rebuilding" in problem


def test_a_pointer_naming_the_object_the_manifest_expects_says_only_fetch_it() -> None:
    """Git LFS names objects by SHA-256, the same digest the manifest records.

    Measured against the real corpus on 2026-08-30: `git add` wrote
    `oid sha256:22f8ee6d44d5e2...` and the build had recorded `sha256: 22f8ee6d44d5e2...`. Two
    spellings of one fact, arriving from two different writers, so they can be compared.
    """
    digest = "22f8ee6d" * 8
    problem = lfs_pointer_problem(
        BENIGN_CORPUS_FILENAME,
        pointer(oid=digest),
        expected_sha256=digest,
        expected_bytes=MEASURED_BENIGN_BYTES,
    )

    assert problem is not None
    assert "the object this manifest expects" in problem
    assert "git lfs pull" in problem


def test_a_pointer_naming_another_object_is_drift_and_says_so() -> None:
    """The input that separates the two situations a reader must not confuse.

    A pointer to the wrong object is the committed corpus disagreeing with its manifest, and
    `git lfs pull` would hand over a corpus that is still not the one described. Telling this
    reader to fetch would be telling them to complete a mistake.
    """
    problem = lfs_pointer_problem(
        BENIGN_CORPUS_FILENAME,
        pointer(oid="aa" * 32),
        expected_sha256="22f8ee6d" * 8,
        expected_bytes=MEASURED_BENIGN_BYTES,
    )

    assert problem is not None
    assert "DIFFERENT object" in problem
    assert "not something `git lfs pull` fixes" in problem


def test_a_pointer_whose_size_contradicts_the_manifest_is_refused() -> None:
    digest = "22f8ee6d" * 8
    problem = lfs_pointer_problem(
        BENIGN_CORPUS_FILENAME,
        pointer(size=1, oid=digest),
        expected_sha256=digest,
        expected_bytes=MEASURED_BENIGN_BYTES,
    )

    assert problem is not None
    assert "describe different files" in problem


def test_the_corpus_itself_is_not_mistaken_for_a_pointer() -> None:
    """The other direction, and it matters here more than it usually would.

    This repository's corpus is a corpus **of prompt injections**, so a row quoting the LFS spec
    URL is exactly the kind of payload it carries on purpose. Recognition is positional and
    structural, never a search for the marker anywhere in the file.
    """
    rows = (
        '{"id":"a::clean","text":"version ' + LFS_POINTER_VERSION + '","label":1}\n'
    ).encode("utf-8")

    assert lfs_pointer_problem(BENIGN_CORPUS_FILENAME, rows) is None


@pytest.mark.parametrize(
    "payload, reason",
    [
        pytest.param(b"", "empty", id="empty"),
        pytest.param(b"\xff\xfe\x00\x01", "not utf-8", id="binary"),
        pytest.param(
            b"version " + LFS_POINTER_VERSION.encode() + b"\noid sha256:aa\n",
            "no size", id="missing-size",
        ),
        pytest.param(
            b"version " + LFS_POINTER_VERSION.encode() + b"\nsize 1\n",
            "no oid", id="missing-oid",
        ),
        pytest.param(
            b"version " + LFS_POINTER_VERSION.encode() + b"\nnowhitespace\nsize 1\n",
            "a line that is not key value", id="not-key-value",
        ),
        pytest.param(b"oid sha256:aa\nversion " + LFS_POINTER_VERSION.encode() + b"\n",
                     "version is not first", id="version-not-first"),
        pytest.param(b"version https://example.invalid/spec/v1\noid sha256:aa\nsize 1\n",
                     "another spec", id="wrong-spec-url"),
        pytest.param(b"a" * 2048, "too large to be a pointer", id="too-large"),
    ],
)
def test_what_is_not_a_pointer(payload: bytes, reason: str) -> None:
    assert lfs_pointer_problem(BENIGN_CORPUS_FILENAME, payload) is None, reason


def test_a_pointer_for_a_file_the_manifest_expects_is_caught_by_the_guarded_door(
    pins, tmp_path: Path
) -> None:
    """End to end through `read_corpus`, which is the only way anything reads the corpus."""
    from nbc.corpus.manifest import CorpusManifestMismatch, corpus_directory, read_corpus
    from tests.harness.corpus_fixtures import small_corpus, write_corpus

    write_corpus(pins, tmp_path, small_corpus())
    (corpus_directory(tmp_path) / BENIGN_CORPUS_FILENAME).write_bytes(pointer())

    with pytest.raises(CorpusManifestMismatch) as abort:
        read_corpus(pins, tmp_path)

    (problem,) = abort.value.problems
    assert "Git LFS pointer" in problem
    assert "was edited after the build" not in problem


# --- the abort ------------------------------------------------------------------------------------


def test_the_abort_carries_a_code_of_its_own() -> None:
    assert CorpusNotPublishable.exit_code == 28
    assert exit_code_for(CorpusNotPublishable("boom")) == 28
    assert declared_exit_codes()[28] is CorpusNotPublishable


def test_the_abort_will_not_be_raised_with_nothing_to_say() -> None:
    with pytest.raises(ValueError, match="at least one problem"):
        CorpusNotPublishable()


def test_the_builder_reads_the_declaration_from_the_repository_and_not_from_its_output_root(
    tmp_path: Path,
) -> None:
    """A scratch build still produces bytes that would have to be committed from this checkout.

    Reading `.gitattributes` from `--root` would make every build into a temporary directory
    report that nothing is tracked, which is the shape of a gate people learn to ignore.
    """
    assert build.read_gitattributes() == (REPO_ROOT / GITATTRIBUTES_FILENAME).read_text("utf-8")
    assert build.read_gitattributes(tmp_path) == ""


def test_the_builder_refuses_an_untracked_corpus_of_the_measured_size() -> None:
    """The gate as the build calls it, on the number the build actually produced."""
    build.refuse_an_unpublishable_corpus(
        [(BENIGN_CORPUS_FILENAME, MEASURED_BENIGN_BYTES)]
    )

    with pytest.raises(CorpusNotPublishable, match="refuses a push"):
        build.refuse_an_unpublishable_corpus([("untracked.jsonl", MEASURED_BENIGN_BYTES)])
