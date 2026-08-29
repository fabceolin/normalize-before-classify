"""The artifact is reproducible from a committed script, and the script's rules are the tested part.

The fixture below is not invented: every row is copied verbatim from
`https://www.unicode.org/Public/security/15.1.0/confusables.txt`, tabs and all, so the parser is
tested against the format it will actually meet rather than against a tidied version of it.

The four derivation rules each appear twice — a row they keep and a row they drop — because a rule
whose rejecting input nobody wrote down is a rule nobody has checked.
"""

from __future__ import annotations

import ast
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from nbc.canon import vendor_confusables
from nbc.canon.confusables_table import (
    ConfusablesTableInvalid,
    DATA_DIR,
    artifact_filename,
    in_scope,
    load,
)
from nbc.canon.vendor_confusables import (
    RULE_VERSION,
    derive,
    main,
    parse_upstream,
    render,
    upstream_url,
)

# --- a fixture in the real upstream format ------------------------------------------------------

HEADER = """\
# confusables.txt
# Date: 2023-08-11, 17:46:40 GMT
# © 2023 Unicode®, Inc.
# For terms of use, see https://www.unicode.org/terms_of_use.html
#
# Unicode Security Mechanisms for UTS #39
# Version: {revision}
#
"""

# Verbatim upstream rows. The comment after `#` carries the character names and is ignored.
KEPT_ROWS = "\n".join(
    (
        "0430 ;\t0061 ;\tMA\t# ( а → a ) CYRILLIC SMALL LETTER A → LATIN SMALL LETTER A\t# ",
        "0391 ;\t0041 ;\tMA\t# ( Α → A ) GREEK CAPITAL LETTER ALPHA → LATIN CAPITAL LETTER A\t# ",
        "042B ;\t0062 006C ;\tMA\t# ( Ы → bl ) CYRILLIC CAPITAL LETTER YERU → LATIN SMALL LETTER B, LATIN SMALL LETTER L\t# ",
        "A699 ;\t006F 006F ;\tMA\t# ( ꚙ → oo ) CYRILLIC SMALL LETTER DOUBLE O → LATIN SMALL LETTER O, LATIN SMALL LETTER O\t# ",
    )
)

DROPPED_ROWS = "\n".join(
    (
        # Rule 2: an ASCII source. THIS is the row a full UTS-39 skeleton keeps, and keeping it
        # would fold `1` to `l` across every benign source file in the corpus.
        "0031 ;\t006C ;\tMA\t# ( 1 → l ) DIGIT ONE → LATIN SMALL LETTER L\t# ",
        "007C ;\t006C ;\tMA\t#* ( | → l ) VERTICAL LINE → LATIN SMALL LETTER L\t# ",
        # Rule 3: a non-ASCII source outside the declared Cyrillic and Greek blocks.
        "05AD ;\t0596 ;\tMA\t# ( ֭ → ֖ ) HEBREW ACCENT DEHI → HEBREW ACCENT TIPEHA\t# ",
        # Rule 4: an in-scope source whose upstream prototype is not all ASCII.
        "0498 ;\t0033 0326 ;\tMA\t# ( Ҙ → 3̦ ) CYRILLIC CAPITAL LETTER ZE WITH DESCENDER → DIGIT THREE, COMBINING COMMA BELOW\t# →З̧→",
        # Rule 1: a multi-code-point source cannot be applied per code point.
        "0431 0432 ;\t0062 ;\tMA\t# a synthetic multi-code-point source\t# ",
    )
)

EXPECTED = {"а": "a", "Α": "A", "Ы": "bl", "ꚙ": "oo"}


def upstream_text(revision: str = "15.1.0", *, rows: str | None = None) -> str:
    body = rows if rows is not None else f"{KEPT_ROWS}\n\n{DROPPED_ROWS}\n"
    return HEADER.format(revision=revision) + body


# --- parsing --------------------------------------------------------------------------------


def test_the_header_revision_is_read_from_the_file() -> None:
    parsed = parse_upstream(upstream_text("15.1.0"))
    assert parsed.declared_revision == "15.1.0"


def test_the_notice_travels_with_the_data() -> None:
    parsed = parse_upstream(upstream_text())
    assert "© 2023 Unicode" in parsed.notice
    assert "terms_of_use" in parsed.notice


def test_multi_code_point_sources_never_reach_the_rows() -> None:
    parsed = parse_upstream(upstream_text())
    assert len(parsed.rows) == 8  # nine rows, minus the one with a two-code-point source
    assert all(source <= 0x10FFFF for source, _ in parsed.rows)


def test_a_bom_and_crlf_line_endings_are_tolerated() -> None:
    text = "﻿" + upstream_text().replace("\n", "\r\n")
    assert derive(parse_upstream(text), revision="15.1.0") == EXPECTED


def test_a_file_with_no_version_header_aborts() -> None:
    text = upstream_text().replace("# Version: 15.1.0\n", "")
    with pytest.raises(ConfusablesTableInvalid, match="no `# Version"):
        parse_upstream(text)


def test_a_file_with_two_different_version_headers_aborts() -> None:
    text = upstream_text().replace("# Version: 15.1.0\n", "# Version: 15.1.0\n# Version: 9.9.9\n")
    with pytest.raises(ConfusablesTableInvalid, match="two different versions"):
        parse_upstream(text)


def test_a_file_with_no_notice_aborts() -> None:
    text = "# Version: 15.1.0\n" + KEPT_ROWS
    with pytest.raises(ConfusablesTableInvalid, match="no copyright or terms-of-use line"):
        parse_upstream(text)


def test_a_file_with_no_rows_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="no single-code-point rows"):
        parse_upstream(upstream_text(rows="\n"))


def test_a_two_field_line_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="2 fields, expected 3"):
        parse_upstream(upstream_text(rows="0430 ;\t0061\n"))


def test_a_line_that_is_not_hex_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="does not hold hex"):
        parse_upstream(upstream_text(rows="ZZZZ ;\t0061 ;\tMA\n"))


def test_a_code_point_above_the_unicode_range_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="above U\\+10FFFF"):
        parse_upstream(upstream_text(rows="110000 ;\t0061 ;\tMA\n"))


def test_an_empty_target_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="empty source or target"):
        parse_upstream(upstream_text(rows="0430 ; ;\tMA\n"))


# --- the four derivation rules ----------------------------------------------------------------


def test_the_derivation_keeps_exactly_the_rows_the_rules_admit() -> None:
    assert derive(parse_upstream(upstream_text()), revision="15.1.0") == EXPECTED


@pytest.mark.parametrize(
    ("row", "why"),
    [
        ("0031 ;\t006C ;\tMA\n", "an ASCII source folds ordinary text"),
        ("007C ;\t006C ;\tMA\n", "an ASCII source folds ordinary text"),
        ("05AD ;\t0596 ;\tMA\n", "outside the declared Cyrillic and Greek blocks"),
        ("0498 ;\t0033 0326 ;\tMA\n", "the prototype is not all ASCII"),
        ("0431 0432 ;\t0062 ;\tMA\n", "a multi-code-point source is not per code point"),
    ],
)
def test_each_rejecting_row_is_dropped_and_nothing_else_is(row: str, why: str) -> None:
    """On its own the row derives nothing, and the refusal comes from whichever of the two rules
    reaches it first: a multi-code-point source never becomes a row at all, and the other four
    survive parsing and are then dropped by the scope. Beside the kept rows it must change
    nothing.
    """
    with pytest.raises(ConfusablesTableInvalid, match="no rows at all|no single-code-point rows"):
        derive(parse_upstream(upstream_text(rows=row)), revision="15.1.0")

    with_row = derive(parse_upstream(upstream_text(rows=KEPT_ROWS + "\n" + row)), revision="15.1.0")
    assert with_row == EXPECTED, why


def test_a_revision_the_file_does_not_declare_aborts() -> None:
    with pytest.raises(ConfusablesTableInvalid, match="declares Unicode 15.1.0 but was requested"):
        derive(parse_upstream(upstream_text("15.1.0")), revision="16.0.0")


def test_conflicting_duplicate_sources_abort_rather_than_letting_file_order_decide() -> None:
    rows = KEPT_ROWS + "\n0430 ;\t0062 ;\tMA\n"
    with pytest.raises(ConfusablesTableInvalid, match="maps to both"):
        derive(parse_upstream(upstream_text(rows=rows)), revision="15.1.0")


def test_an_identical_duplicate_source_is_kept_once() -> None:
    rows = KEPT_ROWS + "\n0430 ;\t0061 ;\tMA\n"
    assert derive(parse_upstream(upstream_text(rows=rows)), revision="15.1.0") == EXPECTED


def test_every_derived_key_satisfies_the_loader_contract() -> None:
    """The derivation and the loader enforce the same domain, from the same declaration."""
    for key, value in derive(parse_upstream(upstream_text()), revision="15.1.0").items():
        assert len(key) == 1 and in_scope(ord(key))
        assert value and all(ord(character) < 0x80 for character in value)


# --- rendering ---------------------------------------------------------------------------------


def _render(mapping: dict[str, str]) -> str:
    return render(
        mapping,
        revision="15.1.0",
        source_url=upstream_url("15.1.0"),
        source_sha256="0" * 64,
        source_bytes=1,
        source_notice="© Unicode, Inc.",
    )


def test_the_render_is_byte_stable_and_independent_of_insertion_order() -> None:
    forward = _render(dict(EXPECTED))
    backward = _render(dict(reversed(list(EXPECTED.items()))))
    assert forward == backward
    assert forward.endswith("\n")


def test_the_render_is_sorted_by_code_point() -> None:
    payload = json.loads(_render(dict(EXPECTED)))
    keys = list(payload["mapping"])
    assert keys == sorted(keys, key=ord)


def test_the_render_is_pure_ascii_so_the_diff_is_readable() -> None:
    _render(dict(EXPECTED)).encode("ascii")


def test_the_render_declares_the_rule_version() -> None:
    payload = json.loads(_render(dict(EXPECTED)))
    assert payload["derivation"]["rule_version"] == RULE_VERSION
    assert payload["entry_count"] == len(EXPECTED)


# --- the CLI, offline --------------------------------------------------------------------------


def _local_source(tmp_path: Path, revision: str) -> Path:
    path = tmp_path / "confusables.txt"
    path.write_text(upstream_text(revision), encoding="utf-8")
    return path


def test_write_then_check_then_load_round_trips(tmp_path: Path) -> None:
    revision = unicodedata.unidata_version
    source = _local_source(tmp_path, revision)
    out = tmp_path / "data"

    assert main(["--write", "--revision", revision, "--source", str(source), "--out", str(out)]) == 0
    assert main(["--check", "--revision", revision, "--source", str(source), "--out", str(out)]) == 0
    assert dict(load(out).mapping) == EXPECTED


def test_check_refuses_an_artifact_that_was_edited_by_hand(tmp_path: Path) -> None:
    """The gate that makes the committed file falsifiable, with the edit that trips it."""
    revision = unicodedata.unidata_version
    source = _local_source(tmp_path, revision)
    out = tmp_path / "data"
    assert main(["--write", "--revision", revision, "--source", str(source), "--out", str(out)]) == 0

    artifact = out / artifact_filename(revision)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["mapping"]["б"] = "b"
    payload["entry_count"] = len(payload["mapping"])
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    assert main(["--check", "--revision", revision, "--source", str(source), "--out", str(out)]) != 0


def test_write_refuses_to_leave_two_artifacts_behind(tmp_path: Path) -> None:
    revision = unicodedata.unidata_version
    source = _local_source(tmp_path, revision)
    out = tmp_path / "data"
    out.mkdir()
    (out / artifact_filename("9.9.9")).write_text("{}", encoding="utf-8")

    assert main(["--write", "--revision", revision, "--source", str(source), "--out", str(out)]) != 0


def test_check_reports_a_missing_artifact(tmp_path: Path) -> None:
    revision = unicodedata.unidata_version
    source = _local_source(tmp_path, revision)
    assert (
        main(["--check", "--revision", revision, "--source", str(source), "--out", str(tmp_path)])
        != 0
    )


def test_a_malformed_revision_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--check", "--revision", "latest", "--out", str(tmp_path)])
    assert caught.value.code == 2


def test_the_failure_exit_code_is_the_declared_abort(tmp_path: Path) -> None:
    source = tmp_path / "absent.txt"
    status = main(["--check", "--revision", "15.1.0", "--source", str(source), "--out", str(tmp_path)])
    assert status == ConfusablesTableInvalid.exit_code


def test_the_network_lives_in_one_function() -> None:
    """`urllib` is referenced only inside `fetch`, checked in the parsed source.

    The failing input is a `urllib.request.urlopen` call moved into `parse_upstream` or `derive`:
    the layer would then reach the network on a path the offline suite exercises, and the guard's
    refusal would look like a test failure rather than like the design decision it is.
    """
    tree = ast.parse(
        Path(vendor_confusables.__file__).read_text(encoding="utf-8"),
        filename=vendor_confusables.__file__,
    )
    fetch_body = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fetch"
    )
    inside_fetch = {id(node) for node in ast.walk(fetch_body)}

    escaped = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "urllib"
        and id(node) not in inside_fetch
    ]
    assert not escaped, f"urllib is referenced outside fetch() at {escaped}"


def test_the_scan_that_bounds_the_network_can_fail(tmp_path: Path) -> None:
    """The gate above, applied to a module that does reach the network from the wrong place."""
    offending = tmp_path / "leaky.py"
    offending.write_text(
        "import urllib.request\n"
        "def fetch(url):\n"
        "    return urllib.request.urlopen(url).read()\n"
        "def derive(url):\n"
        "    return urllib.request.urlopen(url).read()\n",
        encoding="utf-8",
    )
    tree = ast.parse(offending.read_text(encoding="utf-8"))
    fetch_body = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fetch"
    )
    inside = {id(node) for node in ast.walk(fetch_body)}
    escaped = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "urllib" and id(node) not in inside
    ]
    assert escaped == [5]


def test_the_data_directory_documents_the_artifact_it_holds() -> None:
    """A reviewer auditing `canon/data/` reads this before anything else, so it must be current."""
    readme = (DATA_DIR / "README.md").read_text(encoding="utf-8")
    assert artifact_filename(unicodedata.unidata_version) in readme
    assert str(len(load().mapping)) in readme
    assert "identity on all of `U+0000..U+007F`" in readme
    assert "vendor_confusables --check" in readme
    assert "terms_of_use" in readme


# --- upstream, in the smoke tier ---------------------------------------------------------------
#
# The only place `source_sha256` and `source_bytes` are compared to anything. Recorded beside the
# artifact and never checked, they would be exactly the defect this repository keeps finding in
# itself: evidence written down next to a value and never read back.


@pytest.mark.smoke
def test_the_committed_artifact_is_what_upstream_derives_to() -> None:
    revision = unicodedata.unidata_version
    raw = vendor_confusables.fetch(revision)
    committed = json.loads(
        (DATA_DIR / artifact_filename(revision)).read_text(encoding="utf-8")
    )

    assert hashlib.sha256(raw).hexdigest() == committed["source"]["sha256"]
    assert len(raw) == committed["source"]["bytes"]
    assert upstream_url(revision) == committed["source"]["url"]

    upstream = parse_upstream(raw.decode("utf-8-sig"))
    assert upstream.declared_revision == revision
    assert upstream.notice == committed["source"]["notice"]
    assert derive(upstream, revision=revision) == committed["mapping"]

    # The whole file, byte for byte — which subsumes rule_version, the block set and the key
    # order, none of which the field-by-field assertions above would catch on their own.
    rendered = render(
        derive(upstream, revision=revision),
        revision=revision,
        source_url=upstream_url(revision),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        source_notice=upstream.notice,
    )
    assert rendered == (DATA_DIR / artifact_filename(revision)).read_text(encoding="utf-8")


@pytest.mark.smoke
def test_the_upstream_table_is_transitively_closed_for_the_rows_in_scope() -> None:
    """The assumption behind taking the prototype as-is, checked against the table itself.

    An in-scope source whose prototype is not ASCII is dropped. That is only correct while the
    prototype is itself final; if upstream mapped it onward to an ASCII form, dropping it would
    lose a real confusable and the layer would silently fail to undo its own dressing.
    """
    revision = unicodedata.unidata_version
    upstream = parse_upstream(vendor_confusables.fetch(revision).decode("utf-8-sig"))
    prototypes = dict(upstream.rows)

    unfinished = [
        (source, target)
        for source, target in upstream.rows
        if in_scope(source) and len(target) == 1 and ord(target) in prototypes
    ]
    assert not unfinished, f"upstream is not closed for {unfinished!r}"
