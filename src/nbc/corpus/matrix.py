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
3.5 puts `HELDOUT_CHAINS` beside this constant under an acceptance criterion that forbids
`corpus/heldout.py` from importing anything under `nbc.canon`, and a transitive import is still an
import. The opposite rule governs `corpus/dressings.py`, which must share the layer's character
source; the two rules are what keep the bound and held-out halves of the table from collapsing
into each other.

**Where the chain vocabulary lives.** `CLEAN_CHAIN`, `CLEAN_CHAIN_NAME`, `CHAIN_SEPARATOR` and
`render_chain` are here rather than in `corpus/attack.py`, where story 3.2 first needed them. The
benign builders and the held-out registry need to know how a chain is spelled and may not reach
into the attack module to find out.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Final, Mapping, Sequence

from nbc.errors import NbcError
from nbc.schema import BENIGN_CLASSES, FAMILY_ATTACK

__all__ = [
    "CHAINS",
    "CHAIN_SEPARATOR",
    "CLEAN_CHAIN",
    "CLEAN_CHAIN_NAME",
    "CORPUS_CLASSES",
    "DECODED_LINKS",
    "CorpusMatrixInvalid",
    "chain_problems",
    "encoding_depth",
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

DECODED_LINKS: Final[frozenset[str]] = frozenset({"base64", "hex"})
"""The links that produce a segment the canonicalization layer must **decode**.

These consume a level of AD-6's per-branch recursion budget. `homoglyph` and `zero_width` are
character substitutions: the layer removes or maps them in steps 1 and 2 and never opens a new
document for them, so they consume none.

Declared here and **compared** against `canon.stages.decode.ORDER` by
`tests/corpus/test_matrix.py`, which is the only place the two meet. A third encoding taught to
the layer fails that comparison rather than silently making some chain's `encoding_depth` wrong.
"""


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
    chains: Mapping[str, Sequence[Sequence[str]]] = CHAINS,
    registry: Mapping[str, Callable[[str], str]] | None = None,
) -> None:
    """Abort unless every declared chain can be built. Called by the builder before it renders.

    `registry` defaults to `dressings.DRESSINGS`, imported inside the call so this module keeps
    its own import surface -- `dressings.py` imports `nbc.canon` by story 3.4's rule, and the
    matrix must not acquire that import on behalf of `corpus/heldout.py`, which story 3.5
    forbids from having it.
    """
    if registry is None:
        from nbc.corpus.dressings import DRESSINGS

        registry = DRESSINGS
    problems = chain_problems(chains, registry)
    if problems:
        raise CorpusMatrixInvalid(*problems)
