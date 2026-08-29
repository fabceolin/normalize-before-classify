"""The OQ2 spike's arithmetic and its draw, offline.

The spike itself reaches the network and the models; nothing here does. What is tested is the
part that could quietly report a wrong number: the interval, and the rule that decides which
attack positives get scored.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

_SPIKE = Path(__file__).resolve().parents[2] / "spikes" / "oq2_clean_recall.py"


def _load() -> object:
    """Import the spike by path: `spikes/` is not a package, and must not become one."""
    name = "oq2_clean_recall_spike"
    specification = importlib.util.spec_from_file_location(name, _SPIKE)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    # Registered before execution because `@dataclass(slots=True)` resolves annotations
    # through `sys.modules[cls.__module__]`, which a path import would otherwise leave empty.
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


spike = _load()


def test_the_spike_owns_no_pinned_identity() -> None:
    """Every repository, revision and artifact path comes from `pins.toml` and nowhere else.

    A spike that spelled a sha of its own would answer OQ2 about an artifact the published run
    does not use, which is the one way this measurement could be both green and meaningless.
    """
    document = tomllib.loads((_SPIKE.parents[1] / "pins.toml").read_text(encoding="utf-8"))
    source = _SPIKE.read_text(encoding="utf-8")

    identifiers: set[str] = set()
    for entry in document["baseline"] + document["attack_dataset"]:
        identifiers.update(
            str(value)
            for key, value in entry.items()
            if isinstance(value, str)
            and (key in {"repository", "revision"} or key.endswith("_path"))
        )

    found = sorted(identifier for identifier in identifiers if identifier in source)
    assert not found, f"the spike spells pinned identities of its own: {found}"


def test_the_spike_reaches_the_models_only_through_the_shared_boundary() -> None:
    """Never its own tokenization, windowing or softmax: a baseline is not failed by our bug."""
    source = _SPIKE.read_text(encoding="utf-8")
    assert "open_windower" in source and "open_baseline" in source
    for forbidden in ("Tokenizer(", "from_file", "InferenceSession", "def softmax", "math.exp"):
        assert forbidden not in source, f"the spike reimplements {forbidden}"


# --- the interval -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hits", "n", "low", "high"),
    [
        # Hand-computed with z = 1.959964, z^2 = 3.841459. For 95/100:
        #   centre = (0.95 + 3.841459/200) / (1 + 0.03841459)          = 0.933352
        #   spread = 1.959964 * sqrt(0.0475/100 + 3.841459/40000) / (1.03841459)
        #          = 1.959964 * 0.0238964 / 1.03841459                 = 0.045103
        # giving [0.888249, 0.978455]. The other three follow the same two lines.
        (50, 100, 0.4038, 0.5962),
        (95, 100, 0.8882, 0.9785),
        (100, 100, 0.9630, 0.99998),
        (0, 100, 0.0000, 0.03699),
    ],
)
def test_the_wilson_interval_matches_a_hand_computed_value(
    hits: int, n: int, low: float, high: float
) -> None:
    interval = spike.wilson_interval(hits, n)
    assert interval.low == pytest.approx(low, abs=5e-5)
    assert interval.high == pytest.approx(high, abs=5e-5)


def test_the_interval_never_leaves_the_unit_range() -> None:
    """The reason it is Wilson: recall near 1.0 sends the normal interval past 1.0."""
    for n in (1, 7, 30, 3073):
        for hits in (0, n):
            interval = spike.wilson_interval(hits, n)
            assert 0.0 <= interval.low <= interval.high <= 1.0


def test_an_empty_sample_has_no_rate_and_no_interval() -> None:
    interval = spike.wilson_interval(0, 0)
    assert interval.low != interval.low  # NaN, rather than a division by zero
    assert interval.high != interval.high


def test_more_hits_than_items_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        spike.wilson_interval(5, 4)


# --- the draw ---------------------------------------------------------------------------------

ROWS = [
    ("attack one", 1),
    ("attack two", 1),
    ("attack one", 1),  # an exact duplicate, which must not weigh twice
    ("benign one", 0),
    ("", 1),  # an empty row carries no payload to recall
]


def test_only_attack_positives_are_scored_and_duplicates_collapse() -> None:
    selected = spike.select_positives(ROWS, attack_label=1, limit=0, seed=1)
    assert selected == ["attack one", "attack two"]


def test_the_attack_label_comes_from_the_pin_rather_than_from_a_convention() -> None:
    assert spike.select_positives(ROWS, attack_label=0, limit=0, seed=1) == ["benign one"]


def test_the_draw_is_deterministic_under_its_seed() -> None:
    pool = [(f"attack {index}", 1) for index in range(200)]
    first = spike.select_positives(pool, attack_label=1, limit=20, seed=7)
    second = spike.select_positives(pool, attack_label=1, limit=20, seed=7)
    other = spike.select_positives(pool, attack_label=1, limit=20, seed=8)

    assert first == second
    assert len(first) == 20
    assert first != other, "a seed that changes nothing is a draw nobody declared"


def test_a_limit_at_or_above_the_pool_scores_every_positive() -> None:
    pool = [(f"attack {index}", 1) for index in range(10)]
    assert len(spike.select_positives(pool, attack_label=1, limit=10, seed=7)) == 10
    assert len(spike.select_positives(pool, attack_label=1, limit=99, seed=7)) == 10


def test_the_selection_does_not_depend_on_the_order_rows_were_read_in() -> None:
    """Splits are read together; which parquet shard came first must not move the sample."""
    pool = [(f"attack {index}", 1) for index in range(50)]
    forward = spike.select_positives(pool, attack_label=1, limit=10, seed=3)
    backward = spike.select_positives(list(reversed(pool)), attack_label=1, limit=10, seed=3)
    assert forward == backward


# --- reading the pinned rows --------------------------------------------------------------


def _dataset(tmp_path: Path) -> "object":
    """A pinned dataset whose snapshot is not on this machine."""
    from nbc import pins

    return pins.AttackDataset(
        key="attacks",
        repository="example/attacks",
        revision="d" * 40,
        splits=("train",),
        attack_label=1,
        licence=pins.Licence(
            identifier=pins.NOT_DECLARED, source="s", attribution="a", redistributed=True
        ),
        provenance=pins.Provenance(checked_on="2026-08-29", card_revision="d" * 40, seeds=()),
    )


def test_a_missing_parquet_reader_says_which_extra_carries_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement runtime deliberately has no `pyarrow`; the message has to say so."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *rest: object, **kwargs: object) -> object:
        if name.startswith("pyarrow"):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *rest, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(SystemExit) as abort:
        spike.read_rows(_dataset(tmp_path))
    assert "build" in str(abort.value) and "pyarrow" in str(abort.value)


def test_a_pinned_split_with_no_shard_in_the_repository_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pins.toml` declares the splits; a split the repository does not ship is a broken pin."""
    import types

    stub = types.ModuleType("huggingface_hub")
    stub.list_repo_files = lambda *args, **kwargs: ["README.md"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    monkeypatch.setattr(spike.pins, "hf_cache_root", lambda: tmp_path)

    with pytest.raises(SystemExit) as abort:
        spike._dataset_files(_dataset(tmp_path), "train")
    assert "no parquet shard" in str(abort.value)


# --- the shard convention ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "split", "expected"),
    [
        ("data/train-00000-of-00001.parquet", "train", True),
        ("data/train-00001-of-00004.parquet", "train", True),
        ("data/train.parquet", "train", True),
        ("data/test-00000-of-00001.parquet", "train", False),
        ("data/train-00000-of-00001.parquet", "test", False),
        ("README.md", "train", False),
        ("data/training-notes.parquet", "train", False),
    ],
)
def test_a_split_is_read_whole_and_only_its_own_shards(
    name: str, split: str, expected: bool
) -> None:
    """A count over one shard of a split is the same error as a count over one split."""
    assert spike._is_shard_of(name, split) is expected
