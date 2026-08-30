"""Whether the corpus this build wrote is a corpus this repository can actually publish.

The benign half of `data/` is **130.4 MB**, measured on 2026-08-30 against the committed pins,
and GitHub refuses any single file over 100 MB on push and warns above 50. Nothing in the
repository checked that. `report/size_budget.py` measures the canonicalization layer and says
nothing about the corpus, so the first `git add data/` would have produced a commit that cannot
be pushed, and the failure would have arrived at the push -- after the build, after the credits,
after the hashes -- rather than at the moment the bytes were produced. This module is that
missing check.

**The answer chosen was Git LFS, not compression.** Decided by Fabricio Ceolin on 2026-08-30. The
repository is experimental work for one reader rather than a public artifact a crowd clones, which
is what makes the free tier's bandwidth (about seven full clones a month at this size) a limit
nobody will reach. What it buys is that the benign half stays a **real text file** in the
working tree, which is the entire reason AD-1 commits the corpus at all: a reader opens `data/` and
checks the rows without running anything. Compression to 15.6 MB would have been smaller and would
have taken that away from every reader at once.

**So the rule here is not "small enough". It is "small enough, or tracked".** A corpus file over the
threshold has to be covered by an `lfs` filter in `.gitattributes`, and a file that is neither is an
abort. Stated that way rather than as a byte ceiling because the byte ceiling is not the property
anybody wants: a tracked half growing to 200 MB is fine, and an untracked half at 60 MB is not.

**Coverage is parsed, never matched as text.** `.gitattributes` is read as what it is -- a pattern
followed by attributes -- and a file is covered when a pattern that names it carries `filter=lfs`.
A substring search for "lfs" in the file would be satisfied by the word appearing in a comment,
which is precisely the shape of a `.gitattributes` somebody commented the rule out of.

This module is **pure**: the standard library, `nbc.errors` and `nbc.schema`'s neighbours only. It
opens no file and names no corpus path; `corpus/build.py` reads `.gitattributes` and the sizes and
hands them in, for the reason every other decision procedure in this package is separated from its
IO -- so the whole of it is covered by a suite that touches no disk.
"""

from __future__ import annotations

import fnmatch
import posixpath
from typing import Final, Sequence

from nbc.errors import NbcError

__all__ = [
    "GITATTRIBUTES_FILENAME",
    "GITHUB_PUSH_REFUSES_BYTES",
    "GITHUB_WARNS_BYTES",
    "LFS_FILTER",
    "CorpusNotPublishable",
    "covers",
    "lfs_patterns",
    "storage_problems",
]


class CorpusNotPublishable(NbcError, exit_code=28):
    """The corpus on disk cannot be committed and pushed, so writing it would be a trap.

    Code 28 because 3 through 27 are taken. An abort rather than a warning, and the reason is the
    shape of the failure it replaces: a build that succeeds, a corpus that hashes, credits that
    generate, and then a `git push` that is refused hours later by a server, with the operator
    holding a commit they have to rewrite history to undo.

    Two inputs produce it, and they are different problems:

    - a corpus file over GitHub's push limit that no `lfs` filter covers -- the push is refused;
    - a corpus file over GitHub's warning threshold that no `lfs` filter covers -- the push
      succeeds and the repository starts carrying a blob it will be told about later.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("CorpusNotPublishable must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__(
            "this corpus cannot be published as it stands:\n  - " + "\n  - ".join(problems)
        )


GITATTRIBUTES_FILENAME: Final[str] = ".gitattributes"
"""Where the tracking declaration lives, at the repository root. Named once, here."""

LFS_FILTER: Final[str] = "lfs"
"""The filter driver value Git LFS installs. `git lfs track` writes `filter=lfs`."""

GITHUB_PUSH_REFUSES_BYTES: Final[int] = 100 * 1024 * 1024
"""The size at which GitHub refuses a push outright. Not a number this project chose.

Spelled as a constant rather than inline so the two thresholds sit beside each other and a reader
can see that one is a refusal and the other is a warning.
"""

GITHUB_WARNS_BYTES: Final[int] = 50 * 1024 * 1024
"""The size at which GitHub starts warning. The gate fires here rather than at the refusal.

Half the limit is where a file stops being ordinary, and the gap between the two is exactly the
margin in which a corpus that grows a little on the next re-pin crosses from "accepted with a
warning nobody read" to "push refused". Learning that at the warning is learning it in time.
"""


def lfs_patterns(gitattributes: str) -> tuple[str, ...]:
    """Every path pattern in `.gitattributes` that carries `filter=lfs`, in file order.

    Parsed as the format actually is -- a pattern, then whitespace-separated attributes, with `#`
    starting a comment line -- rather than searched for a substring. The difference is the file
    somebody commented the rule out of: a commented tracking line still contains the word, and a
    text search would call the corpus tracked while git would not.

    Unset and unspecified forms (`-filter`, `!filter`) are not treated as tracking, because they
    are the opposite of it.
    """
    found: list[str] = []
    for line in gitattributes.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *attributes = stripped.split()
        if any(attribute == f"filter={LFS_FILTER}" for attribute in attributes):
            found.append(pattern)
    return tuple(found)


def covers(pattern: str, path: str) -> bool:
    """Whether a `.gitattributes` pattern names `path`, given as a repository-relative posix path.

    Enough of git's pattern language for the question this module asks, and no more: a pattern with
    a slash in it is anchored at the repository root, and one without matches on the basename at any
    depth. A leading slash is the explicit spelling of the anchored form.

    What it deliberately does **not** implement is `**`, negation, or directory-only patterns. A
    corpus file is named by one of two literal paths, so a fuller matcher would be code with no
    subject -- and a matcher that quietly got `**` wrong would report a corpus as tracked when git
    disagreed, which is the failure this whole module exists to prevent. `storage_problems` is the
    caller, and it is asking about the two halves `CORPUS_FILENAMES` declares.
    """
    cleaned = pattern.lstrip("/")
    if "/" in cleaned:
        return fnmatch.fnmatchcase(path, cleaned)
    return fnmatch.fnmatchcase(posixpath.basename(path), cleaned)


def storage_problems(
    files: Sequence[tuple[str, int]], gitattributes: str, *, directory: str
) -> tuple[str, ...]:
    """Every corpus file large enough to matter that no `lfs` filter covers.

    `files` is `(name, bytes)` per corpus half, `directory` is the corpus directory relative to the
    repository root, and `gitattributes` is the declaration's text -- empty when there is none,
    which correctly means nothing is tracked.

    The two sides come from different places on purpose: the sizes are the bytes the build just
    rendered, and the coverage is a file a person edits. A check built from one of them twice would
    agree with itself while the push was refused.
    """
    patterns = lfs_patterns(gitattributes)
    problems: list[str] = []
    for name, size in files:
        path = posixpath.join(directory, name)
        if size < GITHUB_WARNS_BYTES:
            continue
        if any(covers(pattern, path) for pattern in patterns):
            continue
        if size >= GITHUB_PUSH_REFUSES_BYTES:
            problems.append(
                f"{path} is {size / 1024 / 1024:.1f} MB and no filter={LFS_FILTER} pattern in "
                f"{GITATTRIBUTES_FILENAME} covers it ({list(patterns)}). GitHub refuses a push "
                f"carrying a file over {GITHUB_PUSH_REFUSES_BYTES // 1024 // 1024} MB, so "
                f"committing this would produce a commit that has to be unwound with a history "
                f"rewrite. Track it, or make it smaller"
            )
        else:
            problems.append(
                f"{path} is {size / 1024 / 1024:.1f} MB and no filter={LFS_FILTER} pattern in "
                f"{GITATTRIBUTES_FILENAME} covers it ({list(patterns)}). GitHub warns above "
                f"{GITHUB_WARNS_BYTES // 1024 // 1024} MB and refuses above "
                f"{GITHUB_PUSH_REFUSES_BYTES // 1024 // 1024}, and this file is between the two: "
                f"the push will succeed and the next re-pin that grows it will not"
            )
    return tuple(problems)
