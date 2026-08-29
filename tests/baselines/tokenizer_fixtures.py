"""Real `tokenizer.json` files, built in-process, so the window policy is tested with no download.

The claim under test is a claim about `tokenizers`: that a file's declared `truncation` survives
`Tokenizer.from_file`, that `no_truncation()` removes it, and that a document longer than the
window therefore reaches the windower whole. A fake tokenizer would turn every one of those into
a statement about the fake -- it would truncate exactly when the double was told to.

So the fixtures build real tokenizers with the library's own writer and save them to real files.
Two families, mirroring the pinned pair: WordPiece with a `[CLS] $A [SEP]` template, and Unigram
with the same frame, which is what the sentencepiece-based pin ships. Both use a whitespace
pre-tokenizer over single-letter words, so a test can say "a document of 17 content tokens" and
mean it.

The two truncation shapes are the load-bearing ones: `truncation` declared with a `max_length`,
as the protectai pin carries it, and `truncation` absent, as the testsavantai pin carries it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

from tokenizers import Tokenizer, models, pre_tokenizers, processors

from nbc import pins
from nbc.baselines.tokenization import SpecialTokenFrame

CONTENT_WORDS: Final[tuple[str, ...]] = ("a", "b", "c", "d")
"""One token each under the whitespace pre-tokenizer, so token counts are predictable."""

UNKNOWN: Final[str] = "[UNK]"
START: Final[str] = "[CLS]"
END: Final[str] = "[SEP]"

_SPECIALS: Final[tuple[str, ...]] = (UNKNOWN, START, END)


def _vocabulary() -> dict[str, int]:
    """The frame's ids sit *after* the content words on purpose.

    A vocabulary that put `[CLS]` at 1 and `[SEP]` at 2 would let a policy that guessed those two
    ids -- rather than measuring the tokenizer's own frame -- pass every test in the suite. Real
    tokenizers put them wherever their training left them.
    """
    ordered = (UNKNOWN, *CONTENT_WORDS, START, END)
    return {token: index for index, token in enumerate(ordered)}


def _framed(tokenizer: Tokenizer) -> Tokenizer:
    vocabulary = _vocabulary()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{START} $A {END}",
        pair=f"{START} $A {END} $B:1 {END}:1",
        special_tokens=[(START, vocabulary[START]), (END, vocabulary[END])],
    )
    return tokenizer


def wordpiece() -> Tokenizer:
    """The BERT-family shape: WordPiece with a two-token frame."""
    return _framed(Tokenizer(models.WordPiece(_vocabulary(), unk_token=UNKNOWN)))


def unigram() -> Tokenizer:
    """The sentencepiece-family shape: Unigram with the same two-token frame."""
    vocabulary = _vocabulary()
    scores = [(token, 0.0 if token in _SPECIALS else -1.0) for token in vocabulary]
    return _framed(Tokenizer(models.Unigram(scores, unk_id=vocabulary[UNKNOWN], byte_fallback=False)))


def unframed() -> Tokenizer:
    """A tokenizer that adds no special tokens at all: a zero-token frame is a legal one."""
    tokenizer = Tokenizer(models.WordPiece(_vocabulary(), unk_token=UNKNOWN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def write(
    path: Path,
    tokenizer: Tokenizer | None = None,
    *,
    truncation: int | None = None,
    padding: int | None = None,
) -> Path:
    """Save a tokenizer to `path`, optionally declaring truncation or padding *in the file*.

    Declared here rather than after loading, because the file is where the pinned pair disagrees
    and the file is what the shared loader has to neutralize.
    """
    tokenizer = wordpiece() if tokenizer is None else tokenizer
    if truncation is not None:
        tokenizer.enable_truncation(max_length=truncation)
    if padding is not None:
        tokenizer.enable_padding(length=padding)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


def document(content_tokens: int) -> str:
    """A document of exactly `content_tokens` content tokens under these fixtures."""
    return " ".join(CONTENT_WORDS[index % len(CONTENT_WORDS)] for index in range(content_tokens))


def plant(
    cache_root: Path,
    baseline: pins.Baseline,
    *,
    truncation: int | None = None,
    padding: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> Path:
    """Put a tokenizer where the pin says this baseline's tokenizer lives, in a fake cache.

    The path is built from the pin -- `snapshot_dir` plus `tokenizer_path` -- so a test exercises
    the same resolution the run does, including the one repository that ships two files of that
    name at one revision.
    """
    return write(
        cache_root / _relative(baseline),
        tokenizer,
        truncation=truncation,
        padding=padding,
    )


def _relative(baseline: pins.Baseline) -> Path:
    return (
        Path(baseline.artifact.cache_directory)
        / "snapshots"
        / baseline.revision
        / baseline.tokenizer_path
    )


def window_pin(length: int, *, source: str = "", revision: str = "0" * 40) -> pins.WindowPin:
    """A `WindowPin` for a fixture window, since the windower refuses a bare integer."""
    return pins.WindowPin(
        length=length,
        source=source or "a fixture, not a pinned artifact",
        confirmed_on="2026-08-29",
        confirmed_revision=revision,
    )


def ids(tokenizer: Tokenizer, text: str, *, specials: bool = False) -> tuple[int, ...]:
    """What a tokenizer makes of a text, with the test doing its own neutralization.

    Deliberately not routed through `load_tokenizer`: an expectation computed by the code under
    test is not an expectation.
    """
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tuple(tokenizer.encode(text, add_special_tokens=specials).ids)


def flatten(windows: Sequence[Sequence[int]], frame: SpecialTokenFrame) -> tuple[int, ...]:
    """The content of a document's windows, in order, with each window's own frame stripped."""
    out: list[int] = []
    for window in windows:
        end = len(window) - len(frame.suffix)
        out.extend(window[len(frame.prefix) : end])
    return tuple(out)
