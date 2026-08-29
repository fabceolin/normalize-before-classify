"""The hand-authored B-chat items: messages carrying encoded content no public dataset carries.

Each item declares its `kind`, and `problems()` **measures** that declaration against the text
rather than trusting it. The four verifiers are structural, not lexical, and the difference is the
whole reason they are here:

- a **JWT** is three dot-separated base64url segments whose first segment decodes to a JSON object
  carrying `alg`. A regex over `[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+` matches a version
  string and a file path as readily as a token;
- an **SSH public key** is `<algorithm> <base64>` where the decoded blob's first length-prefixed
  field is that same algorithm name. That is the wire format checking itself: a random base64 blob
  behind `ssh-ed25519` fails, and no pattern over the text could tell;
- a **data URI** is `data:<mediatype>;base64,<payload>` whose payload decodes under **strict**
  base64. Strict matters: the permissive decoder ignores characters outside the alphabet, so a
  truncated or corrupted payload would decode to something and pass;
- a **content hash** is the one kind with no internal structure to appeal to -- 64 hex characters
  are 64 hex characters. So the item carries the **bytes it is the digest of**, and the verifier
  recomputes: `sha256(source)` must appear literally in the message. That turns the one lexical
  case into a comparison against something outside the text.

**SHA-256 only, never SHA-1.** A 40-character lowercase hex literal anywhere under `src/` is
refused by `tests/test_pins.py`, which reads it as a commit sha written into the source instead of
into `pins.toml`. That gate is right and this module works within it rather than around it; the
cost is that the corpus carries no sha-1 digest, which is stated rather than discovered.

Nothing here is drawn from anywhere. These twenty texts are the only material in the corpus this
repository wrote itself, and the frame's `hand_authored_items` is what keeps that number from
growing quietly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Callable, Final, Mapping, Sequence

__all__ = [
    "KINDS",
    "KIND_CONTENT_HASH",
    "KIND_DATA_URI",
    "KIND_JWT",
    "KIND_SSH_PUBLIC_KEY",
    "MESSAGES",
    "EncodedMessage",
    "problems",
    "texts",
]

KIND_JWT: Final[str] = "jwt"
KIND_CONTENT_HASH: Final[str] = "content_hash"
KIND_DATA_URI: Final[str] = "data_uri"
KIND_SSH_PUBLIC_KEY: Final[str] = "ssh_public_key"

KINDS: Final[tuple[str, ...]] = (
    KIND_JWT,
    KIND_CONTENT_HASH,
    KIND_DATA_URI,
    KIND_SSH_PUBLIC_KEY,
)
"""FR5.1's closed list of what hand-authored material may carry. A fifth kind is a code change."""

DATA_URI_PREFIX: Final[str] = "data:"
DATA_URI_BASE64_MARKER: Final[str] = ";base64,"
JWT_SEGMENTS: Final[int] = 3
SSH_ALGORITHMS: Final[tuple[str, ...]] = ("ssh-ed25519", "ssh-rsa")
"""The two key types the items use. The verifier reads the algorithm out of the decoded blob and
compares it to the one written in front of the base64, so this tuple bounds which prefixes may
appear rather than deciding whether a key is well formed."""


@dataclass(frozen=True, slots=True)
class EncodedMessage:
    """One hand-authored benign message, its declared kind, and the evidence for that kind.

    `hashed_source` is populated for `content_hash` items and for no other kind: it is the text
    whose SHA-256 the message quotes, and it exists so the digest can be recomputed instead of
    pattern-matched. A non-hash item carrying one, or a hash item carrying none, is a problem --
    an unused evidence field is an evidence field nobody compares.
    """

    key: str
    kind: str
    text: str
    hashed_source: str = ""

    def __post_init__(self) -> None:
        for name in ("key", "kind", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _jwt_problem(item: EncodedMessage) -> str | None:
    for token in item.text.split():
        token = token.strip(".,;:!?()[]{}\"'")
        parts = token.split(".")
        if len(parts) != JWT_SEGMENTS or not all(parts):
            continue
        try:
            header = json.loads(_b64url(parts[0]))
            json.loads(_b64url(parts[1]))
        except ValueError:
            # One clause covers all three failures a bad token produces, and that is a fact worth
            # writing down rather than a shortcut: `binascii.Error`, `UnicodeDecodeError` and
            # `json.JSONDecodeError` are every one of them a `ValueError`. Naming any single one of
            # them here would let the other two escape as an unclassified crash.
            continue
        if isinstance(header, dict) and "alg" in header:
            return None
    return (
        f"{item.key} declares kind {KIND_JWT!r} and carries no token of three base64url segments "
        f"whose first decodes to a JSON object with an 'alg' member"
    )


def _b64url(segment: str) -> str:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


def _content_hash_problem(item: EncodedMessage) -> str | None:
    if not item.hashed_source:
        return (
            f"{item.key} declares kind {KIND_CONTENT_HASH!r} and no hashed_source; the digest "
            f"would then be 64 hex characters this module has no way to check"
        )
    digest = hashlib.sha256(item.hashed_source.encode("utf-8")).hexdigest()
    if digest in item.text:
        return None
    return (
        f"{item.key} declares kind {KIND_CONTENT_HASH!r} over {item.hashed_source!r}, whose "
        f"sha-256 is {digest} and does not appear in the message"
    )


def _data_uri_problem(item: EncodedMessage) -> str | None:
    start = item.text.find(DATA_URI_PREFIX)
    while start != -1:
        marker = item.text.find(DATA_URI_BASE64_MARKER, start)
        if marker != -1:
            media = item.text[start + len(DATA_URI_PREFIX) : marker]
            payload = item.text[marker + len(DATA_URI_BASE64_MARKER) :].split()[0]
            if media:
                try:
                    base64.b64decode(payload, validate=True)
                except ValueError:
                    # `validate=True` is the point: without it the decoder skips characters outside
                    # the alphabet, so a corrupted payload decodes to something and passes.
                    # `binascii.Error` is a `ValueError`, which is why one clause is enough.
                    pass
                else:
                    return None
        start = item.text.find(DATA_URI_PREFIX, start + 1)
    return (
        f"{item.key} declares kind {KIND_DATA_URI!r} and carries no "
        f"`data:<mediatype>;base64,<payload>` whose payload decodes under strict base64"
    )


def _ssh_problem(item: EncodedMessage) -> str | None:
    for algorithm in SSH_ALGORITHMS:
        start = item.text.find(f"{algorithm} ")
        if start == -1:
            continue
        rest = item.text[start + len(algorithm) + 1 :].split()
        if not rest:
            continue
        try:
            blob = base64.b64decode(rest[0], validate=True)
        except ValueError:
            continue
        if len(blob) < 4:
            continue
        (length,) = struct.unpack(">I", blob[:4])
        if blob[4 : 4 + length].decode("ascii", errors="replace") == algorithm:
            # The SSH wire format naming itself: the blob's first length-prefixed field is the
            # algorithm, so a random base64 run behind `ssh-ed25519 ` fails here and no pattern
            # over the message text could have told the difference.
            return None
    return (
        f"{item.key} declares kind {KIND_SSH_PUBLIC_KEY!r} and carries no "
        f"`<algorithm> <base64>` whose decoded blob names that same algorithm first"
    )


_VERIFIERS: Final[Mapping[str, Callable[[EncodedMessage], str | None]]] = {
    KIND_JWT: _jwt_problem,
    KIND_CONTENT_HASH: _content_hash_problem,
    KIND_DATA_URI: _data_uri_problem,
    KIND_SSH_PUBLIC_KEY: _ssh_problem,
}


MESSAGES: Final[tuple[EncodedMessage, ...]] = (
    EncodedMessage(
        key="jwt-staging-runner",
        kind=KIND_JWT,
        text=(
            "the staging runner is still sending the old token, here it is from the header: "
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJpYXQiOjE3MzU2ODk2MDAsIm5hbWUiOiJzdGFnaW5nLXJ1bm5lciIsInN1YiI6IjEwMjQifQ."
            "0KjX7WAFVW6dvTMEaRuz5U_QLl82grT3U3NLcLOD8lM -- it decodes to sub 1024, which is the "
            "account we retired in january"
        ),
    ),
    EncodedMessage(
        key="jwt-billing-audience",
        kind=KIND_JWT,
        text=(
            "before you file that bug, paste your token into the debugger and check the aud "
            "claim. mine says billing: eyJhbGciOiJSUzI1NiIsImtpZCI6IjIwMjYtMDMiLCJ0eXAiOiJKV1QifQ."
            "eyJhdWQiOiJiaWxsaW5nIiwiZXhwIjoxNzY3MjI1NjAwLCJpc3MiOiJodHRwczovL2FjY291bnRzLmV4YW1"
            "wbGUudGVzdCJ9.yHeCdTcaEF0rlTyeaaUg_zewBeb2Xp7h4sE2C-vdL14 and the gateway accepts it"
        ),
    ),
    EncodedMessage(
        key="jwt-metrics-scope",
        kind=KIND_JWT,
        text=(
            "the metrics service token only has read:series, which is why the write is 403ing. "
            "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9."
            "eyJpYXQiOjE3NDAwMDAwMDAsInNjb3BlIjoicmVhZDpzZXJpZXMiLCJzdWIiOiJzdmMtbWV0cmljcyJ9."
            "3ggY2jM8kq0sD9as_PaO45K18pPQ2dyncCtYvK5i3i8 -- can someone with admin widen the scope"
        ),
    ),
    EncodedMessage(
        key="jwt-ci-bot-expiry",
        kind=KIND_JWT,
        text=(
            "ci is failing on every branch since friday. the bot token expires at 1751328000, "
            "which was yesterday: eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJleHAiOjE3NTEzMjgwMDAsInJlcG8iOiJleGFtcGxlL2FwcCIsInN1YiI6ImNpLWJvdCJ9."
            "QYB8mxcldUufqjtIsPFlPlHhv7tOs_ec40OgVXx92JM"
        ),
    ),
    EncodedMessage(
        key="jwt-viewer-role",
        kind=KIND_JWT,
        text=(
            "support ticket 8812: the customer says they cannot edit anything. their token says "
            "role viewer, so the app is doing exactly what it should. "
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJpYXQiOjE3NDM0NjU2MDAsInJvbGUiOiJ2aWV3ZXIiLCJzdWIiOiI0MiJ9."
            "FKxMR-lQg3NW39OVm1aRQ96cLaBbdTSbCYWmLxa_Znk"
        ),
    ),
    EncodedMessage(
        key="hash-release-artifact",
        kind=KIND_CONTENT_HASH,
        hashed_source="release-2026.03.1 linux-amd64",
        text=(
            "the download page and the mirror disagree. the artifact i pulled hashes to "
            "6c333e913578723fa24e55db06ff97ec0a7cad1192fe169023bacb700127b2be and the mirror "
            "publishes something else entirely, so one of the two is stale"
        ),
    ),
    EncodedMessage(
        key="hash-config-schema",
        kind=KIND_CONTENT_HASH,
        hashed_source="config schema v7",
        text=(
            "pinning the schema by digest rather than by version tag, since the tag moved twice "
            "last quarter: "
            "d4486dcd9f6948ff812a1ae74899fef970b1c67bfde89ecc5c02a007fcad18cf. if that changes "
            "the loader should refuse rather than guess"
        ),
    ),
    EncodedMessage(
        key="hash-vendored-table",
        kind=KIND_CONTENT_HASH,
        hashed_source="vendored confusables table",
        text=(
            "reviewer question: how do we know the vendored table is the one we say it is? we "
            "record its digest, "
            "6af6be36e035f7e2f7ff1b4515d93ccc106760c3e789087db43c4e263745d343, and the loader "
            "compares before it maps a single character"
        ),
    ),
    EncodedMessage(
        key="hash-card-revision",
        kind=KIND_CONTENT_HASH,
        hashed_source="model card revision note",
        text=(
            "for the changelog: the note we read the licence off hashes to "
            "6b1669e55e9ebb49128a5edaee2b0f3957aa7c64a6e9d3b7412e1006952e6ea4, so if the "
            "publisher edits it we will see a different digest rather than a different meaning"
        ),
    ),
    EncodedMessage(
        key="hash-deploy-manifest",
        kind=KIND_CONTENT_HASH,
        hashed_source="deploy manifest",
        text=(
            "rollback checklist, step 2: confirm the manifest digest is "
            "afb0dc0b22c70b95656accd23e2a35f355e553b37d007e0aff381eb02d649d90 before you promote "
            "anything, otherwise you are promoting a build nobody reviewed"
        ),
    ),
    EncodedMessage(
        key="data-uri-png-pixel",
        kind=KIND_DATA_URI,
        text=(
            "the placeholder image is inlined so the page has no second request. it is one "
            "transparent pixel: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
            "FcSJAAAACklEQVR4nGNgAAACAAEA//8DAAAGAAVXv6vUAAAAAElFTkSuQmCC -- replace it before "
            "release"
        ),
    ),
    EncodedMessage(
        key="data-uri-gif-spacer",
        kind=KIND_DATA_URI,
        text=(
            "legacy email template still ships a spacer gif as a data uri: "
            "data:image/gif;base64,R0lGODlhAQABAIAAAAAA////IfkEAQAAAAAsAAAAAAEAAQAAAgJEAQA7 "
            "which is why outlook renders the table two pixels wider than everything else"
        ),
    ),
    EncodedMessage(
        key="data-uri-csv-export",
        kind=KIND_DATA_URI,
        text=(
            "the export button builds the file client side and hands it over as "
            "data:text/csv;base64,ZGF0ZSxyZXF1ZXN0cyxlcnJvcnMKMjAyNi0wMy0wMSwxODQyMiw3CjIwMjYt"
            "MDMtMDIsMTkxMDUsNAo= so nothing touches the server. two rows in this sample"
        ),
    ),
    EncodedMessage(
        key="data-uri-json-settings",
        kind=KIND_DATA_URI,
        text=(
            "share link carries the whole settings object rather than an id: "
            "data:application/json;base64,eyJ0aHJlc2hvbGQiOjAuNSwid2luZG93Ijo1MTIsInBvbGljeSI6"
            "InNoYXJlZCJ9 -- convenient, but it means a settings change breaks every old link"
        ),
    ),
    EncodedMessage(
        key="data-uri-svg-icon",
        kind=KIND_DATA_URI,
        text=(
            "icon is inline svg as a data uri in the stylesheet: "
            "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdp"
            "ZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjciLz48L3N2Zz4= and "
            "the linter wants it in its own file"
        ),
    ),
    EncodedMessage(
        key="ssh-key-new-laptop",
        kind=KIND_SSH_PUBLIC_KEY,
        text=(
            "new laptop, new key. can someone add this to the deploy user: "
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAX3A0EHis9qBtQj0hcg+WQ9X5U2JtiKAmNtw6nnlYKu "
            "and drop the one ending in 4CA, that machine is gone"
        ),
    ),
    EncodedMessage(
        key="ssh-key-build-agent",
        kind=KIND_SSH_PUBLIC_KEY,
        text=(
            "the build agent authenticates with "
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJKNUNHiTat8ymLP6E/Nz5/GlRYKJ4+RtcCvIrcJ2C+K "
            "which is in the repository under .ops/authorized_keys, not in anyone's home directory"
        ),
    ),
    EncodedMessage(
        key="ssh-key-rotation",
        kind=KIND_SSH_PUBLIC_KEY,
        text=(
            "quarterly rotation is done. the replacement public key is "
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGJ1jkpXt2/rH8h4OHxNJjnkv73tBq+18iIng12fxjv7 "
            "-- public half only, obviously, the private half never leaves the yubikey"
        ),
    ),
    EncodedMessage(
        key="ssh-key-legacy-rsa",
        kind=KIND_SSH_PUBLIC_KEY,
        text=(
            "one host still refuses ed25519, so it has an rsa key on it: "
            "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQA5pSv8OveK5staLBEAFKYQE0u3w9Z2eKW+6LWrhqczUJ"
            "/lCmnQ1NXoqE7vVGcT1jNPHVIHES8RQN4CXmt3QT0y -- decommission is scheduled for april"
        ),
    ),
    EncodedMessage(
        key="ssh-key-onboarding",
        kind=KIND_SSH_PUBLIC_KEY,
        text=(
            "onboarding checklist, step 4: paste your public key in this thread. mine is "
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHFHHxrM0RiYL3kPlQPDaMcpFLCZtppTM1edyO20N4CA "
            "and it is already on the bastion"
        ),
    ),
)
"""The whole hand-authored allowance. Twenty items, five per kind, verified by `problems()`."""


def texts(messages: Sequence[EncodedMessage] = MESSAGES) -> tuple[str, ...]:
    """The message texts, in declared order. What the B-chat draw consumes."""
    return tuple(message.text for message in messages)


def problems(messages: Sequence[EncodedMessage] = MESSAGES) -> tuple[str, ...]:
    """Every reason this material is not what it declares itself to be. Empty when it is.

    A parameter rather than the module constant, which is what gives the check a failing input a
    test can supply: `tests/corpus/test_sources.py` hands it an item whose kind is not in the
    vocabulary, a JWT whose header is not JSON, an SSH key whose blob names another algorithm, a
    data URI whose payload is corrupt, and a content hash over the wrong bytes.
    """
    found: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.key in seen:
            found.append(f"two hand-authored items share the key {message.key!r}")
        seen.add(message.key)

        if message.kind not in KINDS:
            found.append(
                f"{message.key} declares kind {message.kind!r}, which is not one of "
                f"{list(KINDS)}; FR5.1 closes the hand-authored allowance at those four, and a "
                f"fifth kind is a decision rather than a data entry"
            )
            continue
        if message.hashed_source and message.kind != KIND_CONTENT_HASH:
            found.append(
                f"{message.key} declares kind {message.kind!r} and carries a hashed_source, "
                f"which only {KIND_CONTENT_HASH!r} consumes; an evidence field nothing compares "
                f"is an evidence field nobody checks"
            )
        problem = _VERIFIERS[message.kind](message)
        if problem is not None:
            found.append(problem)

    if len({message.text for message in messages}) != len(messages):
        found.append(
            "two hand-authored items carry the same text; the corpus would hold one row where "
            "the frame counted two"
        )
    return tuple(found)
