"""The vendored mapping is what it declares, and every rule here names the input that breaks it.

A rule the loader enforces is only a check if someone can construct the input that makes it fail.
Every gate in `confusables_table.load` therefore appears twice in this file: once as the committed
artifact passing it, and once as a hand-built payload that trips it. A gate with only the first
half is a comment.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from nbc.canon import confusables_table
from nbc.canon.confusables_table import (
    ASCII_LAST,
    Block,
    ConfusablesTableInvalid,
    DATA_DIR,
    SCOPED_BLOCKS,
    artifact_filename,
    declared_blocks,
    discover_revision,
    in_scope,
    load,
)
from nbc.errors import NbcError, declared_exit_codes
from nbc.platform import REQUIREMENTS

# --- the committed artifact ------------------------------------------------------------------


@pytest.fixture(scope="module")
def table() -> confusables_table.ConfusablesTable:
    return load()


def test_the_committed_artifact_loads() -> None:
    loaded = load()
    assert loaded.revision == unicodedata.unidata_version
    assert len(loaded.mapping) > 0
    assert loaded.rule_version >= 1


def test_every_key_is_a_single_non_ascii_cyrillic_or_greek_code_point(
    table: confusables_table.ConfusablesTable,
) -> None:
    for key in table.mapping:
        assert len(key) == 1, f"{key!r} is {len(key)} code points"
        assert ord(key) > ASCII_LAST, f"U+{ord(key):04X} is ASCII"
        assert in_scope(ord(key)), f"U+{ord(key):04X} is in no declared block"


def test_every_value_is_a_non_empty_ascii_string(
    table: confusables_table.ConfusablesTable,
) -> None:
    for key, value in table.mapping.items():
        assert value, f"U+{ord(key):04X} maps to the empty string"
        assert all(ord(character) <= ASCII_LAST for character in value), (
            f"U+{ord(key):04X} maps to {value!r}, which leaves ASCII"
        )


def test_the_mapping_is_the_identity_on_the_whole_of_ascii(
    table: confusables_table.ConfusablesTable,
) -> None:
    """AD-14's load-bearing property, asserted over all 128 code points rather than sampled.

    A full UTS-39 skeleton would fail here on `1`, `0`, `I`, `|`, `"` and `` ` ``: upstream really
    does carry `0031 ; 006C` (DIGIT ONE to LATIN SMALL LETTER L). Folding those across the
    benign-code corpus would replace the counter-metric with a number about ASCII folding, and
    would corrupt base64 and hex runs before the decode stage could look at them.
    """
    for code_point in range(ASCII_LAST + 1):
        character = chr(code_point)
        assert character not in table.mapping, f"U+{code_point:04X} is a key"
        assert code_point not in table.translate_table, f"U+{code_point:04X} is a translation"
        assert character.translate(table.translate_table) == character


def test_the_translate_table_applies_per_code_point(
    table: confusables_table.ConfusablesTable,
) -> None:
    """`str.translate` is per code point by construction; this asserts the table agrees with it."""
    for key, value in table.mapping.items():
        assert key.translate(table.translate_table) == value
        assert f"x{key}y".translate(table.translate_table) == f"x{value}y"


def test_a_known_homoglyph_line_canonicalizes_to_its_ascii_form(
    table: confusables_table.ConfusablesTable,
) -> None:
    """A golden case, so the artifact is checked against something a human recognizes."""
    disguised = "АВСЕН"  # Cyrillic А В С Е Н
    assert disguised.translate(table.translate_table) == "ABCEH"
    greek = "ΑΒΕΖ"  # Greek Α Β Ε Ζ
    assert greek.translate(table.translate_table) == "ABEZ"


def test_the_table_is_not_a_full_skeleton(table: confusables_table.ConfusablesTable) -> None:
    """The specific rows AD-14 refuses, named so the refusal is visible rather than implied."""
    for ascii_character in "10Ol|\"'`I":
        assert ascii_character.translate(table.translate_table) == ascii_character


def test_loading_twice_produces_equal_tables() -> None:
    """No cache, no module state, and no dependence on which call came first."""
    first, second = load(), load()
    assert dict(first.mapping) == dict(second.mapping)
    assert list(first.mapping) == list(second.mapping)
    assert first.as_run_fields() == second.as_run_fields()


def test_the_mapping_is_immutable(table: confusables_table.ConfusablesTable) -> None:
    with pytest.raises(TypeError):
        table.mapping["a"] = "b"  # type: ignore[index]
    with pytest.raises(TypeError):
        table.translate_table[0x61] = "b"  # type: ignore[index]


# --- the revision is pinned to the interpreter -------------------------------------------------


def test_the_vendored_revision_equals_the_interpreters_own() -> None:
    """The two sides come from different places: a filename on disk, and `unicodedata`.

    `discover_revision` exists so this comparison is possible at all — `load()` has already made
    it and would only ever hand back agreement.
    """
    assert discover_revision() == unicodedata.unidata_version


def test_the_artifact_is_named_for_the_revision_it_declares(table) -> None:
    assert (DATA_DIR / artifact_filename(table.revision)).is_file()


def test_the_platform_interpreter_pin_names_the_vendored_revision() -> None:
    """`platform.py` states this revision as the reason CPython 3.13 is pinned exactly.

    The reason is prose and can only be checked as prose; the structural half of the same claim is
    the assertion above, against the live interpreter. The failing input is a re-vendoring to
    16.0.0 that updates the filename and leaves the interpreter pin explaining 15.1.0.
    """
    assert discover_revision() in REQUIREMENTS.interpreter.reason
    assert REQUIREMENTS.interpreter.implementation == "CPython"


def test_the_abort_carries_an_exit_code_no_other_abort_uses() -> None:
    assert issubclass(ConfusablesTableInvalid, NbcError)
    code = ConfusablesTableInvalid.exit_code
    assert declared_exit_codes()[code] is ConfusablesTableInvalid


def test_the_abort_refuses_to_be_raised_with_no_problem() -> None:
    with pytest.raises(ValueError):
        raise ConfusablesTableInvalid()


# --- the declared domain -----------------------------------------------------------------------


def test_no_declared_block_reaches_ascii() -> None:
    """What makes the identity on ASCII structural rather than a property of the derived data."""
    for block in SCOPED_BLOCKS:
        assert block.first > ASCII_LAST
        assert block.first <= block.last


def test_a_block_that_reaches_ascii_is_refused() -> None:
    with pytest.raises(ValueError):
        Block("Basic Latin", 0x0000, 0x007F)
    with pytest.raises(ValueError):
        Block("Inverted", 0x0500, 0x0400)


def test_the_declared_blocks_are_disjoint() -> None:
    """Overlapping blocks would make `in_scope` true for a reason no single row explains."""
    ordered = sorted(SCOPED_BLOCKS, key=lambda block: block.first)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.last < later.first, f"{earlier.name} overlaps {later.name}"


def test_in_scope_admits_cyrillic_and_greek_and_refuses_everything_else() -> None:
    assert in_scope(0x0430)  # CYRILLIC SMALL LETTER A
    assert in_scope(0x0391)  # GREEK CAPITAL LETTER ALPHA
    assert in_scope(0xA699)  # Cyrillic Extended-B
    assert not in_scope(0x0061)  # LATIN SMALL LETTER A
    assert not in_scope(0x00E9)  # LATIN SMALL LETTER E WITH ACUTE
    assert not in_scope(0x05AD)  # HEBREW ACCENT DEHI
    assert not in_scope(0x4E00)  # CJK
    assert not in_scope(ASCII_LAST)


def test_run_fields_carry_the_revision_and_the_provenance(table) -> None:
    fields = table.as_run_fields()
    assert fields["unicode_revision"] == table.revision
    assert fields["entry_count"] == len(table.mapping)
    assert fields["scoped_blocks"] == declared_blocks()
    assert isinstance(fields["source"], dict)
    assert fields["source"]["bytes"] > 0


# --- every gate, with the input that trips it --------------------------------------------------


def _payload(**overrides: Any) -> dict[str, Any]:
    """A minimal payload that loads, so an override isolates exactly one defect."""
    payload: dict[str, Any] = {
        "unicode_revision": unicodedata.unidata_version,
        "entry_count": 1,
        "derivation": {"rule_version": 1, "scoped_blocks": declared_blocks()},
        "source": {
            "url": "https://www.unicode.org/Public/x/confusables.txt",
            "sha256": "0" * 64,
            "bytes": 1,
            "notice": "© Unicode, Inc.",
        },
        "mapping": {"а": "a"},
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: object, revision: str | None = None) -> Path:
    revision = revision or unicodedata.unidata_version
    path = tmp_path / artifact_filename(revision)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return tmp_path


def test_the_minimal_payload_loads(tmp_path: Path) -> None:
    """Without this, every negative case below could be passing for the wrong reason."""
    loaded = load(_write(tmp_path, _payload()))
    assert dict(loaded.mapping) == {"а": "a"}


def test_an_empty_data_directory_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="no vendored confusables artifact"):
        load(tmp_path)


def test_a_missing_data_directory_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="not a directory"):
        load(tmp_path / "absent")


def test_two_artifacts_abort_rather_than_letting_a_sort_order_choose(tmp_path: Path) -> None:
    _write(tmp_path, _payload())
    (tmp_path / artifact_filename("15.0.0")).write_text("{}", encoding="utf-8")
    with pytest.raises(ConfusablesTableInvalid, match="exactly"):
        load(tmp_path)


def test_a_revision_that_is_not_the_interpreters_aborts_and_says_to_re_vendor(
    tmp_path: Path,
) -> None:
    """The Python-minor-bump case, exercised on this interpreter by naming another revision.

    A file named for 15.0.0 on a 15.1.0 interpreter is the same inequality a contributor on
    CPython 3.12 would produce, and the message must send them to `--write`, not to a test.
    """
    other = "15.0.0" if unicodedata.unidata_version != "15.0.0" else "14.0.0"
    _write(tmp_path, _payload(unicode_revision=other), revision=other)
    with pytest.raises(ConfusablesTableInvalid) as caught:
        load(tmp_path)
    message = str(caught.value)
    assert other in message and unicodedata.unidata_version in message
    assert "vendor_confusables --write" in message


def test_a_filename_that_carries_no_revision_aborts(tmp_path: Path) -> None:
    (tmp_path / "confusables-latest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfusablesTableInvalid, match="N.N.N"):
        load(tmp_path)


def test_a_payload_revision_that_disagrees_with_the_filename_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="inside a file named for"):
        load(_write(tmp_path, _payload(unicode_revision="9.9.9")))


def test_malformed_json_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="not valid JSON"):
        load(_write(tmp_path, "{not json"))


def test_a_top_level_list_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="JSON object"):
        load(_write(tmp_path, []))


def test_invalid_utf8_aborts(tmp_path: Path) -> None:
    path = tmp_path / artifact_filename(unicodedata.unidata_version)
    path.write_bytes(b'{"unicode_revision": "\xff\xfe"}')
    with pytest.raises(ConfusablesTableInvalid, match="not valid UTF-8"):
        load(tmp_path)


def test_a_repeated_json_key_aborts(tmp_path: Path) -> None:
    """`json` keeps the last of a repeated key, so the mapping would depend on file order."""
    body = (
        '{"unicode_revision": "%s", "unicode_revision": "9.9.9"}' % unicodedata.unidata_version
    )
    with pytest.raises(ConfusablesTableInvalid, match="repeats the key"):
        load(_write(tmp_path, body))


def test_an_ascii_key_aborts(tmp_path: Path) -> None:
    """The defect AD-14 exists to prevent, supplied directly."""
    with pytest.raises(ConfusablesTableInvalid, match="ASCII key U\\+0031"):
        load(_write(tmp_path, _payload(mapping={"1": "l"})))


def test_a_non_ascii_key_outside_the_declared_blocks_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="U\\+00E9"):
        load(_write(tmp_path, _payload(mapping={"é": "e"})))


def test_a_multi_code_point_key_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="per code point"):
        load(_write(tmp_path, _payload(mapping={"аб": "ab"})))


def test_a_non_ascii_value_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="leaves"):
        load(_write(tmp_path, _payload(mapping={"а": "à"})))


def test_an_empty_value_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="empty string"):
        load(_write(tmp_path, _payload(mapping={"а": ""})))


def test_a_non_string_value_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="not a string"):
        load(_write(tmp_path, _payload(mapping={"а": 97})))


def test_an_empty_mapping_aborts(tmp_path: Path) -> None:
    with pytest.raises(ConfusablesTableInvalid, match="empty mapping"):
        load(_write(tmp_path, _payload(mapping={}, entry_count=0)))


def test_an_entry_count_that_disagrees_with_the_mapping_aborts(tmp_path: Path) -> None:
    """The count is recorded as the evidence for the mapping, so it is compared to it."""
    with pytest.raises(ConfusablesTableInvalid, match="entry_count 7 over a mapping of 1"):
        load(_write(tmp_path, _payload(entry_count=7)))


def test_a_scoped_block_set_the_loader_does_not_declare_aborts(tmp_path: Path) -> None:
    """A payload derived under a wider domain than the loader enforces."""
    wider = declared_blocks() + [["Basic Latin", "0000", "007F"]]
    with pytest.raises(ConfusablesTableInvalid, match="scoped_blocks"):
        load(_write(tmp_path, _payload(derivation={"rule_version": 1, "scoped_blocks": wider})))


def test_a_missing_source_block_aborts(tmp_path: Path) -> None:
    payload = _payload()
    del payload["source"]
    with pytest.raises(ConfusablesTableInvalid, match="declares no 'source'"):
        load(_write(tmp_path, payload))


def test_a_source_sha256_that_is_not_a_digest_aborts(tmp_path: Path) -> None:
    payload = _payload()
    payload["source"]["sha256"] = "not-a-digest"
    with pytest.raises(ConfusablesTableInvalid, match="64 hex digits"):
        load(_write(tmp_path, payload))


def test_a_source_byte_count_that_is_not_a_size_aborts(tmp_path: Path) -> None:
    payload = _payload()
    payload["source"]["bytes"] = 0
    with pytest.raises(ConfusablesTableInvalid, match="not a size"):
        load(_write(tmp_path, payload))


def test_an_empty_source_notice_aborts(tmp_path: Path) -> None:
    payload = _payload()
    payload["source"]["notice"] = "   "
    with pytest.raises(ConfusablesTableInvalid, match="empty source notice"):
        load(_write(tmp_path, payload))


def test_a_boolean_where_an_integer_belongs_aborts(tmp_path: Path) -> None:
    """`True` is an `int` in Python, and every other numeric gate in this repository refuses it."""
    with pytest.raises(ConfusablesTableInvalid, match="entry_count"):
        load(_write(tmp_path, _payload(entry_count=True)))


def test_every_problem_is_named_at_once(tmp_path: Path) -> None:
    """One abort, every reason: a loader that stops at the first is three re-runs to fix a file."""
    payload = _payload(unicode_revision="9.9.9", entry_count=7, mapping={"1": "l"})
    with pytest.raises(ConfusablesTableInvalid) as caught:
        load(_write(tmp_path, payload))
    assert len(caught.value.problems) >= 3
