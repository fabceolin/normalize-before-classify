"""The corpus matrix: which dressing chains are built, for which half of the table.

AD-20's rule is that the set of chains is **declared data in one constant**, because the dressing
axis of the headline table *is* this constant. Two builds that assembled their own chain lists
would produce two different tables and FR2's "every dressing" would be unverifiable.

**Why every class is written out separately.** `CHAINS` states the chains for attacks and for each
benign class as three independent literals, and a test compares them. Mapping all three keys to
one shared tuple would make "benign items are dressed with the same chain set as attacks" a
comparison of an object with itself -- the pattern this project has found in its own history
often enough to have a name for it. Written out, the failing input is a chain added to one class
and forgotten on another, and the test names it.

**Why this module imports nothing under `nbc.canon`.** `DECODED_LINKS` names the two encodings the
layer decodes, and it would be tempting to read them off `canon.stages.decode.ORDER`. It is
declared here instead and **compared** against the layer in `tests/corpus/test_matrix.py`. Story
3.5 put `HELDOUT_CHAINS` beside this constant, and `corpus/heldout.py` -- which holds the encodings
those chains name -- may not import anything under `nbc.canon`, transitively. It imports this
module, so this module's import surface is part of that rule and a transitive import here would
break it. The opposite rule governs `corpus/dressings.py`, which must share the layer's character
source; the two rules are what keep the bound and held-out halves of the table from collapsing
into each other.

**Where the chain and id vocabulary lives.** `CLEAN_CHAIN`, `CLEAN_CHAIN_NAME`,
`CHAIN_SEPARATOR`, `render_chain`, `PAYLOAD_ID_HEX`, `ID_SEPARATOR`, `payload_id` and `item_id` are
here rather than in `corpus/attack.py`, where story 3.2 first needed them. The benign builders and
the held-out registry need to know how a chain is spelled and how a row is addressed, and may not
reach into the attack module to find out -- an id scheme shared by both halves of the corpus and
owned by one of them is a second home waiting to happen. Story 3.6 moved the four id names here
when `corpus/benign.py` became the second caller.
"""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping, Sequence

from nbc.errors import NbcError
from nbc.schema import BENIGN_CLASSES, FAMILY_ATTACK

__all__ = [
    "CHAINS",
    "CHAIN_CLASSES",
    "CHAIN_CLASS_BOUND",
    "CHAIN_CLASS_HELD_OUT",
    "CHAIN_SEPARATOR",
    "CLEAN_CHAIN",
    "CLEAN_CHAIN_NAME",
    "CORPUS_CLASSES",
    "DECODED_LINKS",
    "HELDOUT_CHAINS",
    "ID_SEPARATOR",
    "PAYLOAD_ID_HEX",
    "CorpusMatrixInvalid",
    "chain_class",
    "chain_problems",
    "declared_links",
    "encoding_depth",
    "id_collisions",
    "item_id",
    "payload_id",
    "render_chain",
    "validate",
]


class CorpusMatrixInvalid(NbcError, exit_code=19):
    """A declared chain cannot be built, or the declaration is not the shape AD-20 requires.

    Code 19 because 3 through 18 are taken. An abort rather than a skip: a chain the builder
    cannot render is a column of the headline table that would silently not exist, and a table
    with a missing column looks exactly like a table whose column happened to be empty.

    Every problem is collected before raising, so a reviewer who mis-declared three chains sees
    three messages rather than fixing one at a time.
    """

    def __init__(self, *problems: str) -> None:
        if not problems:
            raise ValueError("CorpusMatrixInvalid must name at least one problem")
        self.problems: tuple[str, ...] = tuple(problems)
        super().__init__("the corpus matrix is not buildable:\n  - " + "\n  - ".join(problems))


# --- the chain vocabulary ---------------------------------------------------------------------

CLEAN_CHAIN: Final[tuple[str, ...]] = ()
"""The empty chain, which AD-3 declares to be `clean`.

It is the identity element of the fold `reduce(apply, chain, payload)`: dressing through no links
returns the payload, and `dressings.dress` returns the very object it was handed, so a `clean` row
carries the payload text rather than a copy that happened to compare equal.
"""

CLEAN_CHAIN_NAME: Final[str] = "clean"
CHAIN_SEPARATOR: Final[str] = "+"
"""AD-3's rendering: the dressing names joined by `+`, or the literal `clean` for the empty chain.

`+` is not a member of any dressing name, and `validate` refuses a name that contains it, so a
rendered chain can always be split back into its links. Without that refusal a dressing called
`base64+hex` would render the same string as the two-link chain and the dressing axis would name
two different documents with one label.
"""


def render_chain(chain: Sequence[str]) -> str:
    """The chain as it appears in an item id and as the dressing axis of the table.

    The **full** chain, never the last link: an axis naming only the outermost dressing would put
    `base64+base64` and `base64` in the same cell, which is the one distinction N4 is about.
    """
    return CHAIN_SEPARATOR.join(chain) if chain else CLEAN_CHAIN_NAME


# --- the item id ------------------------------------------------------------------------------

PAYLOAD_ID_HEX: Final[int] = 16
"""Hex characters of SHA-256 kept in a payload id.

64 bits. Over a pool of ten thousand payloads the birthday probability of a collision is about
3e-12, and a collision would be caught rather than silently merging two payloads: both builders
refuse a pool in which two distinct texts produce one id.
"""

ID_SEPARATOR: Final[str] = "::"
"""What separates the payload id from the chain in an item id."""


def payload_id(text: str) -> str:
    """A stable, content-derived id for one payload.

    Content-derived rather than positional because AD-1's stable order is `(source, payload_id,
    chain)`: an id that carried a row number would make the file's order a property of the read,
    which is the one thing the byte-identical claim cannot depend on.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:PAYLOAD_ID_HEX]


def item_id(payload: str, chain: Sequence[str] = CLEAN_CHAIN) -> str:
    """AD-3's item id: `<payload_id>::<chain>`, the chain joined by `+`, or the literal `clean`.

    The full chain, never the last link: a reported dressing axis that named only the outermost
    dressing would make `base64+base64` and `base64` the same cell.
    """
    return f"{payload}{ID_SEPARATOR}{render_chain(chain)}"


def id_collisions(pairs: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """One message per item id that two different payloads produced. Empty when there are none.

    A separate function so it has a failing input a test can supply: producing a real SHA-256
    prefix collision is not something a test can do, and a check nobody has seen fire is a check
    nobody knows fires. The tests hand it two payloads under one id directly.

    The consequence it prevents is silent: two distinct payloads under one id merge into one
    corpus row, the count drops by one, and every rate computed from it is over a pool that is not
    the pool the report describes. Both halves of the corpus run it, which is why it lives here
    beside `item_id` rather than in whichever builder needed it first.
    """
    seen: dict[str, str] = {}
    problems: list[str] = []
    for identifier, payload in pairs:
        first = seen.setdefault(identifier, payload)
        if first != payload:
            problems.append(
                f"payloads {first!r} and {payload!r} both produce item id {identifier}"
            )
    return tuple(problems)


# --- the declared matrix ----------------------------------------------------------------------

CORPUS_CLASSES: Final[tuple[str, ...]] = (FAMILY_ATTACK, *BENIGN_CLASSES)
"""The three rows of the matrix: the attack family and the two benign classes.

Derived from `schema.py`'s vocabulary rather than restated, because a benign class named here and
not there would be a class the harness cannot report and the aggregation cannot key.
"""

CHAINS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = MappingProxyType(
    {
        FAMILY_ATTACK: (
            CLEAN_CHAIN,
            ("base64",),
            ("hex",),
            ("homoglyph",),
            ("zero_width",),
            ("base64", "base64"),
            ("zero_width", "base64"),
            ("base64", "homoglyph"),
            ("hex", "zero_width"),
            ("base64", "base64", "base64", "base64"),
        ),
        "b_code": (
            CLEAN_CHAIN,
            ("base64",),
            ("hex",),
            ("homoglyph",),
            ("zero_width",),
            ("base64", "base64"),
            ("zero_width", "base64"),
            ("base64", "homoglyph"),
            ("hex", "zero_width"),
            ("base64", "base64", "base64", "base64"),
        ),
        "b_chat": (
            CLEAN_CHAIN,
            ("base64",),
            ("hex",),
            ("homoglyph",),
            ("zero_width",),
            ("base64", "base64"),
            ("zero_width", "base64"),
            ("base64", "homoglyph"),
            ("hex", "zero_width"),
            ("base64", "base64", "base64", "base64"),
        ),
    }
)
"""Every chain this corpus is built in, per corpus class. The bound half of FR2.

**The five singletons** are FR2's list: `clean`, `base64`, `hex`, `homoglyph`, `zero_width`.

**The four two-link chains** are AD-20's two (`base64+base64`, `zero_width+base64`), AD-3's worked
example of the fold's direction (`base64+homoglyph`, meaning `homoglyph(base64(payload))`), and
`hex+zero_width`. The fourth exists so `hex` appears in a composition at all and so `zero_width`
appears on both sides of an encoding link: with only `zero_width+base64`, the claim that a
character substitution consumes no level of the decode budget would be tested in one position
only.

**The four-link chain** is the smallest whose `encoding_depth` exceeds a ceiling of three, which
AD-20 requires the corpus to contain so N4's first limb — does the gain survive nesting past the
ceiling — has data. `tests/corpus/test_matrix.py` asserts the relation against `DEFAULT_CEILING`
rather than against the number three, so re-tuning the ceiling fails a test instead of silently
emptying that limb.

The three classes carry the same chains, per AD-3 and FR2: a benign column that was constant down
each row while the recall column varied would make the two halves of the table incomparable cell
for cell. They are written out three times so that agreement is a comparison between three
declarations rather than an identity. Naturally encoded benign material — JWTs, hashes, data URIs
— lives on the `clean` chain, which is story 3.6's assignment of sources to chains, not this
constant's.
"""

HELDOUT_CHAINS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = MappingProxyType(
    {
        FAMILY_ATTACK: (
            ("base32",),
            ("url_percent",),
            ("rot13",),
        ),
        "b_code": (
            ("base32",),
            ("url_percent",),
            ("rot13",),
        ),
        "b_chat": (
            ("base32",),
            ("url_percent",),
            ("rot13",),
        ),
    }
)
"""AD-28's second registry: the chains built from encodings the layer was never written against.

**Beside `CHAINS`, and disjoint from it.** `tests/corpus/test_heldout.py` asserts the two chain
sets are disjoint and that every dressing name appears in exactly one registry, so an encoding
cannot be bound and held out at once. Put one of these into `CHAINS` and story 3.4's round-trip
contract goes red, and the tempting repair -- delete the chain -- is the failure AD-28 exists to
prevent, which is why the separation is executable rather than conventional.

**The links are declared in `corpus/heldout.py`**, which imports nothing under `nbc.canon`. This
constant holds only their names, so it stays as canon-free as the rest of this module and can be
read by anything that needs the dressing axis without acquiring the layer.

**Three encodings, three mechanisms.** `base32`'s alphabet is a subset of base64's, so the layer is
offered the whole document and must refuse it. `url_percent` emits hex digits behind a character
the layer does not scan, so a decoding step can grip a fragment and never the document. `rot13`
carries no marker at all. Each declares `probes` in `heldout.py`, and the classification is
measured against the layer rather than asserted.

**Written out per class three times**, exactly as `CHAINS` is: benign items are dressed in the
held-out chains too, and that agreement is checked as a comparison between three declarations
rather than as an object compared with itself. A held-out block carrying recall and no counter
metric invites the answer that non-recovery does not matter because its cost is unknown.

The floor of two is `heldout.MIN_HELDOUT_CHAINS` and is enforced there, over the distinct chains
this constant declares.
"""

DECODED_LINKS: Final[frozenset[str]] = frozenset({"base64", "hex"})
"""The links that produce a segment the canonicalization layer must **decode**.

These consume a level of AD-6's per-branch recursion budget. `homoglyph` and `zero_width` are
character substitutions: the layer removes or maps them in steps 1 and 2 and never opens a new
document for them, so they consume none.

Declared here and **compared** against `canon.stages.decode.ORDER` by
`tests/corpus/test_matrix.py`, which is the only place the two meet. A third encoding taught to
the layer fails that comparison rather than silently making some chain's `encoding_depth` wrong.
"""


CHAIN_CLASS_BOUND: Final[str] = "bound"
CHAIN_CLASS_HELD_OUT: Final[str] = "held_out"
CHAIN_CLASSES: Final[tuple[str, ...]] = (CHAIN_CLASS_BOUND, CHAIN_CLASS_HELD_OUT)
"""AD-2's `chain_class`: which half of the table a cell belongs to.

Part of the cell key rather than a label, because AD-11 forbids any function from aggregating
across it: a held-out result averaged into bound ones destroys the only evidence of generalization
the artifact has, and it destroys it by producing a number that looks fine.
"""


def declared_links(chains: Mapping[str, Sequence[Sequence[str]]]) -> frozenset[str]:
    """Every dressing name `chains` mentions, across every corpus class.

    The two registries' link sets are compared with this in `tests/corpus/test_heldout.py`: they
    must be disjoint, and their union must be the union of the two dressing registries. A name in
    both would be a dressing that is bound and held out at once, which is a contradiction the round
    trip contract would resolve by going red on a chain nobody could safely delete.
    """
    return frozenset(link for declared in chains.values() for chain in declared for link in chain)


def chain_class(
    chain: Sequence[str],
    chains: Mapping[str, Sequence[Sequence[str]]] = CHAINS,
    heldout: Mapping[str, Sequence[Sequence[str]]] = HELDOUT_CHAINS,
) -> str:
    """Which half of the table `chain` belongs to, decided by which registry declares its links.

    Structural, not a lookup in a list of names: a chain is held out because every link it names is
    declared held out, so the classification cannot be changed by adding a chain to a set of
    exceptions. A chain mixing a bound link with a held-out one belongs to neither half and is
    refused -- there is no cell for it, and silently filing it under one class would put a row into
    a column that does not describe it.
    """
    links = tuple(chain)
    bound = declared_links(chains)
    held = declared_links(heldout)
    if not links or set(links) <= bound:
        # The empty chain is `clean`, which is a bound chain: AD-3 makes it the identity element of
        # the fold, and `CHAINS` declares it.
        return CHAIN_CLASS_BOUND
    if set(links) <= held:
        return CHAIN_CLASS_HELD_OUT
    raise CorpusMatrixInvalid(
        f"the chain {render_chain(links)!r} mixes links from both registries or names one neither "
        f"declares: bound links are {sorted(bound)} and held-out links are {sorted(held)}. "
        f"chain_class is part of the cell key (AD-2) and no function aggregates across it "
        f"(AD-11), so a chain belonging to neither half has no cell to be reported in"
    )


def encoding_depth(chain: Sequence[str]) -> int:
    """How many levels of the layer's per-branch decode budget this chain consumes.

    **Not `len(chain)`**, and the difference is load-bearing: `len` would call
    `homoglyph+zero_width` a depth-2 chain and `base64+homoglyph` a depth-2 chain, when the layer
    spends nothing on the first and one level on the second. Story 3.4 scopes its round-trip
    contract by this number, and the `len` reading would have exempted a chain that costs nothing
    -- a hiding place with no mechanism behind it.

    The claim is not left as prose. `tests/corpus/test_matrix.py` canonicalizes every chain in
    `CHAINS` and requires `max_depth_reached == min(encoding_depth(chain), ceiling)` and
    `ceiling_hit == (encoding_depth(chain) > ceiling)`.
    """
    return sum(1 for link in chain if link in DECODED_LINKS)


# --- validation -------------------------------------------------------------------------------


def chain_problems(
    chains: Mapping[str, Sequence[Sequence[str]]],
    registry: Mapping[str, Callable[[str], str]],
) -> tuple[str, ...]:
    """Every reason `chains` cannot be built with `registry`. Empty when there are none.

    Both arguments are parameters rather than the module constants, which is what gives this
    check a failing input a test can supply: `tests/corpus/test_matrix.py` hands it a mapping
    missing a class, a class carrying a chain twice, a chain naming an unregistered link, a link
    holding the separator, and a benign class whose chains differ from the attack class's. A gate
    that could only ever be called with the one input that passes is not a gate.
    """
    problems: list[str] = []

    declared = set(chains)
    expected = set(CORPUS_CLASSES)
    for missing in sorted(expected - declared):
        problems.append(
            f"the matrix declares no chains for corpus class {missing!r}; every class in "
            f"{list(CORPUS_CLASSES)} is a row of the table and a class with no chains is a row "
            f"that silently does not exist"
        )
    for extra in sorted(declared - expected):
        problems.append(
            f"the matrix declares chains for {extra!r}, which is not one of "
            f"{list(CORPUS_CLASSES)}; nothing downstream can key a cell on it"
        )

    for corpus_class in CORPUS_CLASSES:
        if corpus_class not in chains:
            continue
        seen: set[str] = set()
        for chain in chains[corpus_class]:
            rendered = render_chain(chain)
            if rendered in seen:
                problems.append(
                    f"{corpus_class} declares the chain {rendered!r} more than once; the second "
                    f"one would produce an item id that already exists"
                )
            seen.add(rendered)
            for link in chain:
                if not isinstance(link, str) or not link:
                    problems.append(
                        f"{corpus_class}: chain {rendered!r} holds the link {link!r}, which is "
                        f"not a dressing name"
                    )
                    continue
                if CHAIN_SEPARATOR in link:
                    problems.append(
                        f"{corpus_class}: the dressing name {link!r} holds "
                        f"{CHAIN_SEPARATOR!r}, so the rendered chain could not be split back "
                        f"into its links and two different chains could render the same axis"
                    )
                if link not in registry:
                    problems.append(
                        f"{corpus_class}: chain {rendered!r} names the dressing {link!r}, which "
                        f"is not in the registry ({sorted(registry)}); a new dressing is a new "
                        f"named function and nothing else"
                    )

    # AD-3: benign items are dressed with the same chain set as attacks. Compared between three
    # separately written declarations, which is the only reason writing them out three times is
    # worth the repetition.
    if FAMILY_ATTACK in chains:
        attack_set = {render_chain(chain) for chain in chains[FAMILY_ATTACK]}
        for corpus_class in BENIGN_CLASSES:
            if corpus_class not in chains:
                continue
            benign_set = {render_chain(chain) for chain in chains[corpus_class]}
            if benign_set != attack_set:
                only_attack = sorted(attack_set - benign_set)
                only_benign = sorted(benign_set - attack_set)
                problems.append(
                    f"{corpus_class} is dressed in a different chain set than "
                    f"{FAMILY_ATTACK}: {only_attack} are missing from it and {only_benign} are "
                    f"extra. AD-3 requires the same set, or the false-positive column is "
                    f"constant down each row while the recall column varies and the two halves "
                    f"of the table are not comparable cell for cell"
                )

    return tuple(problems)


def validate(
    chains: Mapping[str, Sequence[Sequence[str]]],
    registry: Mapping[str, Callable[[str], str]],
) -> None:
    """Abort unless every declared chain can be built. Called by the builder before it renders.

    **Both arguments are required, and `registry` in particular has no default.** It defaulted to
    `dressings.DRESSINGS` through an import inside the call until story 3.5, which is a deferred
    import and still an import: `corpus/heldout.py` imports this module, and AD-28 forbids it from
    reaching `nbc.canon` **transitively**, which is what an AST walk over this file's import
    statements sees whether or not the statement sits inside a function. Making the caller name the
    registry keeps this module a leaf over `nbc.errors` and `nbc.schema`, which is what its
    docstring has always claimed, and `tests/corpus/test_heldout.py` walks the closure to check it.
    """
    problems = chain_problems(chains, registry)
    if problems:
        raise CorpusMatrixInvalid(*problems)
