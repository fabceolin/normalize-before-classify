"""Frames, repositories and files for the benign-draw tests, built in code rather than read.

A test that asserted the draw's behaviour against `pins.toml` would be comparing the frame with
itself. Everything here names its own numbers, so a gate that stopped firing is visible.
"""

from __future__ import annotations

from typing import Any

from nbc.corpus.benign import CodeFile, SourceFile
from nbc.pins import (
    BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
    DRAW_SEEDED_RANDOM,
    BenignChatFrame,
    BenignCodeFrame,
    BenignCodeRepository,
    BenignFrame,
    ConfirmatoryCell,
    Licence,
)

SHA = "c" * 40

ELIGIBLE_TEXT = (
    "const TOKEN = 'aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Q=';\n" + "let x = 1;\n" * 20
)
"""A file the layer's decode stage examines: the base64 run clears its declared candidate floor."""

PLAIN_TEXT = "def add(a, b):\n    return a + b\n" * 12
"""A file with nothing the decode stage will look at. Same size band, opposite eligibility."""


def repository(index: int = 0, key: str | None = None) -> BenignCodeRepository:
    return BenignCodeRepository(
        key=key or f"example-code-{index}",
        repository=f"example/code-{index}",
        revision=SHA,
        licence=Licence(
            identifier="MIT", source="fixture", attribution="fixture", redistributed=True
        ),
    )


def frame(**overrides: Any) -> BenignFrame:
    """A small but structurally complete frame. Every number here is deliberately not pins.toml's."""
    fields: dict[str, Any] = {
        "declared_on": "2026-08-29",
        "sample_size_items": 4,
        "method": DRAW_SEEDED_RANDOM,
        "seed": 11,
        "sort_key": None,
        "frame_id": "0" * 64,
        "b_code": BenignCodeFrame(
            min_repositories=2,
            max_files_per_repository=2,
            eligibility=BENIGN_CODE_ELIGIBILITY_DECODE_CANDIDATE,
            min_file_bytes=100,
            max_file_bytes=4000,
            file_extensions=(".js", ".py"),
            repositories=tuple(repository(index) for index in range(3)),
        ),
        "b_chat": BenignChatFrame(hand_authored_items=1),
        "confirmatory_cell": ConfirmatoryCell(
            declared_on="2026-08-29",
            baseline="example",
            dressing_chain="base64+base64+base64+base64",
            benign_class="b_code",
        ),
    }
    fields.update(overrides)
    return BenignFrame(**fields)


def unique_eligible(marker: str) -> str:
    """An eligible file whose text is unique, so two of them are two payloads."""
    return f"// {marker}\n{ELIGIBLE_TEXT}"


def source_file(marker: str, suffix: str = ".js") -> SourceFile:
    return SourceFile(path=f"src/{marker}{suffix}", text=unique_eligible(marker))


def code_file(repository_key: str, marker: str) -> CodeFile:
    return CodeFile(
        repository_key=repository_key,
        source=f"github.com/example/{repository_key}@{SHA}:src/{marker}.js",
        path=f"src/{marker}.js",
        text=unique_eligible(f"{repository_key}-{marker}"),
    )
