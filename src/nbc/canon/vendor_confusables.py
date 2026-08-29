"""Derives the vendored confusable mapping from the UTS-39 table, reproducibly.

The artifact under `data/` is not a blob someone pasted: this script produced it, this script can
reproduce it byte for byte, and its derivation is a pure function with an offline test over a
fixture in the real upstream format.

Run it::

    python -m nbc.canon.vendor_confusables --write            # fetch upstream, derive, write
    python -m nbc.canon.vendor_confusables --check            # fetch upstream, derive, compare
    python -m nbc.canon.vendor_confusables --check --source F # same, from a local copy

`--check` is what makes the committed file falsifiable: it re-renders from the upstream text and
compares byte for byte, so a hand-edited artifact fails rather than being trusted.

**The rules, all of them, in one place.** A row is kept when, and only when:

1. its source is exactly **one** code point,
2. that code point is **non-ASCII**, and
3. it lies in one of the blocks `confusables_table.SCOPED_BLOCKS` declares, and
4. every code point of its target is **ASCII**.

Nothing else is kept and nothing is transformed on the way through. In particular the upstream
prototype is taken as-is: UTS-39's table is transitively closed, so a Cyrillic code point whose
prototype is non-ASCII has no ASCII form at all, and chasing it further would be a second,
undeclared transform. `test_the_upstream_table_is_transitively_closed_for_the_rows_in_scope`
checks that closure against the live table, in the `smoke` tier, and is the input that would make
the assumption fail.

Rule 2 is the one that matters most, and it is the one a careless implementation drops. Upstream
carries `0031 ; 006C ; MA # ( 1 → l ) DIGIT ONE → LATIN SMALL LETTER L` and seven more ASCII
sources. Keeping them would fold `1` to `l` and `0` to `O` across ordinary source code, which is
how the benign-code counter-metric would quietly become a number about ASCII folding.

Network access lives in `fetch()`, reached only through the CLI when no `--source` is given.
`test_the_network_lives_in_one_function` reads this module's parsed source and asserts that
`urllib` is referenced nowhere else, so importing the module or calling `parse_upstream`,
`derive` or `render` opens nothing — which is what lets the offline unit suite use them at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from nbc.canon.confusables_table import (
    ASCII_LAST,
    ConfusablesTableInvalid,
    DATA_DIR,
    artifact_filename,
    declared_blocks,
    in_scope,
    revision_pattern,
)
from nbc.errors import EXIT_OK, exit_code_for

__all__ = (
    "RULE_VERSION",
    "UPSTREAM_URL_TEMPLATE",
    "Upstream",
    "derive",
    "fetch",
    "main",
    "parse_upstream",
    "render",
    "upstream_url",
)

UPSTREAM_URL_TEMPLATE: Final[str] = (
    "https://www.unicode.org/Public/security/{revision}/confusables.txt"
)

RULE_VERSION: Final[int] = 1
"""Bumped when the four derivation rules change, so a re-derivation under new rules is visible.

It is written into the artifact, and `ConfusablesTable.as_run_fields()` publishes it for whatever
eventually writes `results.json` — which does not exist yet. A mapping that changed because the
rules changed is a different treatment, and a run that does not say so is comparing two
experiments.
"""

_HTTP_TIMEOUT_SECONDS: Final[float] = 30.0

_VERSION_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^#\s*Version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$"
)
_NOTICE_MARKERS: Final[tuple[str, ...]] = ("©", "terms_of_use")


@dataclass(frozen=True, slots=True)
class Upstream:
    """The upstream table as parsed: what it says it is, what it maps, and its notice."""

    declared_revision: str
    notice: str
    rows: tuple[tuple[int, str], ...]
    """Every row whose source is a single code point, in file order, untouched by scoping."""


def upstream_url(revision: str) -> str:
    return UPSTREAM_URL_TEMPLATE.format(revision=revision)


def parse_upstream(text: str) -> Upstream:
    """Parse `confusables.txt` into its declared revision, its notice, and its single-source rows.

    Pure and offline. Malformed content aborts naming the line, because a table that parsed
    "mostly" would produce a mapping shorter than the one the artifact claims and nothing would
    say so.
    """
    declared_revision: str | None = None
    notice_lines: list[str] = []
    rows: list[tuple[int, str]] = []
    problems: list[str] = []

    # `split("\n")` and not `splitlines()`: the latter also breaks on U+2028, U+2029, U+000B,
    # U+000C, U+001C, U+001D and U+001E, so a comment containing one of them would be torn into a
    # fragment that no longer starts with `#` and would then be parsed as a data row.
    for number, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.lstrip("﻿").rstrip("\r")
        if line.lstrip().startswith("#"):
            header = _VERSION_HEADER.match(line.strip())
            if header is not None:
                if declared_revision is not None and declared_revision != header.group(1):
                    problems.append(
                        f"line {number}: the file declares two different versions, "
                        f"{declared_revision} and {header.group(1)}"
                    )
                declared_revision = header.group(1)
            if any(marker in line for marker in _NOTICE_MARKERS):
                notice_lines.append(line.lstrip("#").strip())
            continue

        body = line.split("#", 1)[0].strip()
        if not body:
            continue

        fields = [field.strip() for field in body.split(";")]
        if len(fields) < 3:
            problems.append(f"line {number}: {body!r} has {len(fields)} fields, expected 3")
            continue

        source_field, target_field = fields[0], fields[1]
        try:
            source_points = [int(token, 16) for token in source_field.split()]
            target_points = [int(token, 16) for token in target_field.split()]
        except ValueError:
            problems.append(f"line {number}: {body!r} does not hold hex code points")
            continue
        if not source_points or not target_points:
            problems.append(f"line {number}: {body!r} has an empty source or target")
            continue
        if any(point > 0x10FFFF for point in source_points + target_points):
            problems.append(f"line {number}: {body!r} names a code point above U+10FFFF")
            continue

        if len(source_points) != 1:
            # Kept out of `rows` rather than reported: a multi-code-point source is a legitimate
            # upstream row and simply cannot be applied per code point.
            continue
        rows.append((source_points[0], "".join(chr(point) for point in target_points)))

    if declared_revision is None:
        problems.append(
            "the upstream text carries no `# Version: N.N.N` header, so what revision it is "
            "cannot be established from the file itself"
        )
    if not notice_lines:
        problems.append(
            "the upstream text carries no copyright or terms-of-use line; the notice travels "
            "with the data derived from it"
        )
    if not rows:
        problems.append("the upstream text carries no single-code-point rows at all")

    if problems:
        raise ConfusablesTableInvalid(*problems)

    assert declared_revision is not None
    return Upstream(
        declared_revision=declared_revision,
        notice="\n".join(notice_lines),
        rows=tuple(rows),
    )


def derive(upstream: Upstream, *, revision: str) -> dict[str, str]:
    """Apply the four scoping rules to a parsed upstream table.

    `revision` is what the caller asked for; the parsed table says what it is. They are compared
    here rather than assumed, because a URL is a request and a file is an answer.
    """
    if upstream.declared_revision != revision:
        raise ConfusablesTableInvalid(
            f"the upstream text declares Unicode {upstream.declared_revision} but was requested "
            f"as {revision}; deriving under the wrong revision label is how the artifact's "
            f"filename stops meaning anything"
        )

    mapping: dict[str, str] = {}
    conflicts: list[str] = []
    for code_point, target in upstream.rows:
        if not in_scope(code_point):
            continue
        if any(ord(character) > ASCII_LAST for character in target):
            continue
        existing = mapping.get(chr(code_point))
        if existing is not None and existing != target:
            conflicts.append(
                f"U+{code_point:04X} maps to both {existing!r} and {target!r}; the derivation "
                f"refuses rather than letting file order decide"
            )
            continue
        mapping[chr(code_point)] = target

    if conflicts:
        raise ConfusablesTableInvalid(*conflicts)
    if not mapping:
        raise ConfusablesTableInvalid(
            "the derivation kept no rows at all; a table scoped to nothing would make the "
            "confusables stage a no-op that still claims to run"
        )
    return mapping


def render(
    mapping: Mapping[str, str],
    *,
    revision: str,
    source_url: str,
    source_sha256: str,
    source_bytes: int,
    source_notice: str,
) -> str:
    """The artifact as text: sorted by code point, ASCII-escaped, one trailing newline.

    Byte-stable by construction, so `--check` can compare bytes rather than parse and compare
    structures — the comparison a hand edit has to survive.
    """
    payload = {
        "unicode_revision": revision,
        "entry_count": len(mapping),
        # No `reason` here on purpose: `confusables_table.REVISION_REASON` is the one copy, and
        # it reaches `results.json` from the module. A second copy frozen into the artifact would
        # be a string that can disagree with the module it was quoted from.
        "derivation": {"rule_version": RULE_VERSION, "scoped_blocks": declared_blocks()},
        "source": {
            "url": source_url,
            "sha256": source_sha256,
            "bytes": source_bytes,
            "notice": source_notice,
        },
        "mapping": dict(sorted(mapping.items(), key=lambda item: ord(item[0]))),
    }
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False) + "\n"


def fetch(revision: str) -> bytes:
    """The one function in `canon/` that opens a socket, reached only from the CLI.

    Nothing on the canonicalization path calls it, and no unit test does either: the two tests
    that fetch are marked `smoke` and run in CI's smoke job alone.
    """
    url = upstream_url(revision)
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def _source_bytes(revision: str, source: Path | None) -> bytes:
    if source is None:
        return fetch(revision)
    try:
        return source.read_bytes()
    except OSError as error:
        raise ConfusablesTableInvalid(f"{source} could not be read: {error}") from error


def _render_from(revision: str, raw: bytes, source: Path | None) -> str:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        # A ValueError, not an OSError.
        raise ConfusablesTableInvalid(f"the upstream text is not valid UTF-8: {error}") from error

    upstream = parse_upstream(text)
    mapping = derive(upstream, revision=revision)
    return render(
        mapping,
        revision=revision,
        # The URL is recorded even when the bytes came off disk: it is where they are fetched
        # from, and a local copy is a convenience for `--check`, not a second provenance.
        source_url=upstream_url(revision),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        source_notice=upstream.notice,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nbc.canon.vendor_confusables",
        description="Derive the vendored confusable mapping from the UTS-39 table.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the artifact into --out")
    mode.add_argument(
        "--check", action="store_true", help="re-derive and compare, byte for byte, without writing"
    )
    parser.add_argument(
        "--revision",
        default=unicodedata.unidata_version,
        help="Unicode revision to derive (default: this interpreter's unicodedata.unidata_version)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="a local copy of confusables.txt; without it the table is fetched from unicode.org",
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="the data directory")
    arguments = parser.parse_args(argv)

    revision: str = arguments.revision
    if not revision_pattern().match(revision):
        parser.error(f"--revision {revision!r} is not a Unicode revision of the form N.N.N")

    try:
        rendered = _render_from(revision, _source_bytes(revision, arguments.source), arguments.source)
        target = arguments.out / artifact_filename(revision)

        if arguments.check:
            if not target.exists():
                raise ConfusablesTableInvalid(f"{target} does not exist, so there is nothing to check")
            committed = target.read_text(encoding="utf-8")
            if committed != rendered:
                raise ConfusablesTableInvalid(
                    f"{target.name} is not what this script derives from the upstream table at "
                    f"{revision}: {len(committed)} characters committed against {len(rendered)} "
                    f"derived. Re-vendor with --write rather than editing the artifact"
                )
            print(f"{target.name} matches the committed artifact ({revision})")
            return EXIT_OK

        arguments.out.mkdir(parents=True, exist_ok=True)
        stale = [
            path
            for path in sorted(arguments.out.glob("confusables-*.json"))
            if path.name != target.name
        ]
        if stale:
            raise ConfusablesTableInvalid(
                f"{arguments.out} already holds {', '.join(path.name for path in stale)}; "
                f"re-vendoring is a deliberate act, so delete the artifact being replaced first "
                f"rather than leaving two for a sort order to choose between"
            )
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target} ({revision})")
        return EXIT_OK
    except ConfusablesTableInvalid as error:
        print(f"error: {error}", file=sys.stderr)
        return exit_code_for(error)


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    raise SystemExit(main())
