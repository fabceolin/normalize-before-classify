"""The hand-authored allowance: closed, counted, and verified against what it declares.

Every check here has a failing input, because a verifier nobody has seen refuse anything is a
verifier nobody knows works. The real twenty are asserted clean, and then each of the four kinds is
handed material that should not pass.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from collections import Counter

import pytest

from nbc.corpus.sources.encoded_messages import (
    KIND_CONTENT_HASH,
    KIND_DATA_URI,
    KIND_JWT,
    KIND_SSH_PUBLIC_KEY,
    KINDS,
    MESSAGES,
    EncodedMessage,
    problems,
    texts,
)


def test_the_declared_material_is_what_it_says_it_is() -> None:
    """The whole point: the `kind` beside each text is measured against the text."""
    assert problems() == ()


def test_every_kind_is_represented_and_none_is_outside_the_vocabulary() -> None:
    counts = Counter(message.kind for message in MESSAGES)
    assert set(counts) == set(KINDS), sorted(set(counts).symmetric_difference(KINDS))
    assert min(counts.values()) >= 1, counts


def test_the_texts_are_the_message_texts_in_declared_order() -> None:
    assert texts() == tuple(message.text for message in MESSAGES)


def test_a_kind_outside_the_vocabulary_is_refused() -> None:
    (problem,) = problems([EncodedMessage(key="x", kind="pem_certificate", text="hello")])
    assert "pem_certificate" in problem and "fifth kind" in problem


# --- JWT ---------------------------------------------------------------------------------------


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_a_jwt_shaped_string_that_is_not_a_jwt_is_refused() -> None:
    """Three dot-separated base64url runs are not a token, which is why the header is decoded.

    The failing input is the one a regex would accept: a version string that happens to have the
    shape. `problems` decodes the first segment and asks for an `alg`.
    """
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_JWT, text="upgrading to vGVzdA.dGVzdA.dGVzdA today")]
    )
    assert KIND_JWT in problem and "alg" in problem


def test_a_jwt_whose_header_carries_no_alg_is_refused() -> None:
    header = _b64u(b'{"typ":"JWT"}')
    payload = _b64u(b'{"sub":"1"}')
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_JWT, text=f"token {header}.{payload}.c2ln")]
    )
    assert "alg" in problem


def test_a_well_formed_jwt_passes() -> None:
    header = _b64u(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64u(b'{"sub":"1"}')
    assert problems(
        [EncodedMessage(key="x", kind=KIND_JWT, text=f"token {header}.{payload}.c2ln")]
    ) == ()


# --- content hash ------------------------------------------------------------------------------


def test_a_content_hash_over_the_wrong_bytes_is_refused() -> None:
    """The one kind with no internal structure, checked by recomputing it from declared bytes."""
    digest = hashlib.sha256(b"something else").hexdigest()
    (problem,) = problems(
        [
            EncodedMessage(
                key="x",
                kind=KIND_CONTENT_HASH,
                hashed_source="the declared bytes",
                text=f"the digest is {digest}",
            )
        ]
    )
    assert "sha-256" in problem and "the declared bytes" in problem


def test_a_content_hash_with_no_declared_source_is_refused() -> None:
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_CONTENT_HASH, text="a" * 64)]
    )
    assert "hashed_source" in problem


def test_a_hashed_source_on_another_kind_is_refused() -> None:
    """An evidence field nothing consumes is an evidence field nobody compares."""
    found = problems(
        [
            EncodedMessage(
                key="x",
                kind=KIND_JWT,
                hashed_source="unused",
                text="no token here",
            )
        ]
    )
    assert any("hashed_source" in problem for problem in found)


# --- data URI ----------------------------------------------------------------------------------


def test_a_data_uri_whose_payload_is_not_strict_base64_is_refused() -> None:
    """`validate=True` is the point: the permissive decoder would accept the corrupted payload."""
    (problem,) = problems(
        [
            EncodedMessage(
                key="x",
                kind=KIND_DATA_URI,
                text="see data:text/plain;base64,***not base64***",
            )
        ]
    )
    assert KIND_DATA_URI in problem


def test_a_data_uri_with_no_media_type_is_refused() -> None:
    payload = base64.b64encode(b"hello").decode("ascii")
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_DATA_URI, text=f"data:;base64,{payload}")]
    )
    assert KIND_DATA_URI in problem


# --- SSH public key ----------------------------------------------------------------------------


def _ssh_blob(named: str, body: bytes) -> str:
    blob = struct.pack(">I", len(named)) + named.encode() + struct.pack(">I", len(body)) + body
    return base64.b64encode(blob).decode("ascii")


def test_an_ssh_key_whose_blob_names_another_algorithm_is_refused() -> None:
    """The wire format checking itself. No pattern over the text could tell the difference."""
    forged = _ssh_blob("ssh-dss", b"\x00" * 32)
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_SSH_PUBLIC_KEY, text=f"key ssh-ed25519 {forged} eve@x")]
    )
    assert KIND_SSH_PUBLIC_KEY in problem


def test_a_random_base64_run_behind_an_ssh_prefix_is_refused() -> None:
    noise = base64.b64encode(b"not an ssh key at all, just bytes").decode("ascii")
    (problem,) = problems(
        [EncodedMessage(key="x", kind=KIND_SSH_PUBLIC_KEY, text=f"ssh-ed25519 {noise}")]
    )
    assert KIND_SSH_PUBLIC_KEY in problem


def test_a_key_whose_blob_names_its_own_algorithm_passes() -> None:
    good = _ssh_blob("ssh-ed25519", b"\x01" * 32)
    assert problems(
        [EncodedMessage(key="x", kind=KIND_SSH_PUBLIC_KEY, text=f"ssh-ed25519 {good} me@host")]
    ) == ()


# --- the collection ----------------------------------------------------------------------------


def test_two_items_sharing_a_key_are_refused() -> None:
    good = _ssh_blob("ssh-ed25519", b"\x01" * 32)
    found = problems(
        [
            EncodedMessage(key="same", kind=KIND_SSH_PUBLIC_KEY, text=f"ssh-ed25519 {good} a"),
            EncodedMessage(key="same", kind=KIND_SSH_PUBLIC_KEY, text=f"ssh-ed25519 {good} b"),
        ]
    )
    assert any("share the key" in problem for problem in found)


def test_two_items_sharing_a_text_are_refused() -> None:
    """One payload under two entries is one corpus row where the frame counted two."""
    good = _ssh_blob("ssh-ed25519", b"\x01" * 32)
    text = f"ssh-ed25519 {good} me@host"
    found = problems(
        [
            EncodedMessage(key="a", kind=KIND_SSH_PUBLIC_KEY, text=text),
            EncodedMessage(key="b", kind=KIND_SSH_PUBLIC_KEY, text=text),
        ]
    )
    assert any("same text" in problem for problem in found)


def test_an_empty_field_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        EncodedMessage(key="", kind=KIND_JWT, text="x")
