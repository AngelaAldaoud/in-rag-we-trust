"""Tests for the context builder (C1, D122-D124)."""

import gzip
import json
from collections import Counter

import pytest

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.retrieval.contexts import (
    K, RHOS, TYPE_NAMES, build_context, closed_book_context,
    layout_for, n_poisoned, poisoned_slots,
)

# A hand-made record: every text says what it is, so we can check each passage.
TOY = {
    "id": "toy-question",
    "bridge_passage": "bridge text",
    "clean": [f"clean-{i}" for i in range(K)],
    "poisoned": {t: [f"{t}-{i}" for i in range(K)] for t in TYPE_NAMES},
}


def all_conditions():
    """The 17 controlled contexts per question: shared baseline + 4 types x 4 ratios."""
    yield None, 0.0
    for t in TYPE_NAMES:
        for rho in RHOS[1:]:
            yield t, rho


def test_layout_is_reproducible_and_valid():
    assert layout_for("q1") == layout_for("q1")
    lay = layout_for("q1")
    assert sorted(lay.poison_order) == sorted(lay.display_order) == list(range(K))
    assert 0 <= lay.bridge_pos <= K
    assert len({layout_for(f"q{i}") for i in range(50)}) > 1   # different questions differ


@pytest.mark.parametrize("ptype,rho", list(all_conditions()))
def test_context_structure(ptype, rho):
    ctx = build_context(TOY, ptype, rho)
    m = n_poisoned(rho)
    assert [p.position for p in ctx] == list(range(K + 1))
    assert Counter(p.role for p in ctx) == Counter(bridge=1, poisoned=m, clean=K - m)
    # every slot exactly once: a passage and its own twin can never both appear
    assert sorted(p.slot for p in ctx if p.role != "bridge") == list(range(K))
    for p in ctx:
        expected = {"bridge": "bridge text",
                    "clean": f"clean-{p.slot}",
                    "poisoned": f"{ptype}-{p.slot}"}[p.role]
        assert p.text == expected


def test_poisoned_slots_are_nested_across_rho():
    lay = layout_for(TOY["id"])
    sets = [poisoned_slots(lay, r) for r in RHOS]
    assert all(small < big for small, big in zip(sets, sets[1:]))


def test_order_is_fixed_across_conditions():
    orders = {tuple(p.slot for p in build_context(TOY, t, r)) for t, r in all_conditions()}
    assert len(orders) == 1


def test_same_poisoned_slots_for_every_type():
    for rho in RHOS[1:]:
        shapes = {tuple((p.role, p.slot) for p in build_context(TOY, t, rho))
                  for t in TYPE_NAMES}
        assert len(shapes) == 1


def test_guards():
    with pytest.raises(ValueError):
        build_context(TOY, "swap", 0.0)        # the baseline is shared: no type
    with pytest.raises(ValueError):
        build_context(TOY, None, 0.5)          # a poisoned context needs a type
    with pytest.raises(ValueError):
        build_context(TOY, "Authority", 0.5)   # names are lower case
    with pytest.raises(ValueError):
        build_context(TOY, "swap", 0.3)        # not a ratio of the design


def test_closed_book_is_empty():
    assert closed_book_context(TOY) == []


# ---- real data -------------------------------------------------------------
SAMPLE = PROCESSED_DIR / "sample.json"
POISONS = PROCESSED_DIR / "poisons.jsonl.gz"
real = pytest.mark.skipif(not (SAMPLE.exists() and POISONS.exists()),
                          reason="processed data not built")


@pytest.fixture(scope="module")
def sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


@real
def test_all_real_contexts_build(sample):
    ids = set(sample["pilot"] + sample["main"])
    with gzip.open(POISONS, "rt", encoding="utf-8") as fh:
        records = [r for r in map(json.loads, fh) if r["id"] in ids]
    assert len(records) == 550
    for rec in records:
        for t, rho in all_conditions():
            ctx = build_context(rec, t, rho)
            for p in ctx:
                if p.role == "poisoned":
                    assert p.text != rec["clean"][p.slot]   # the twin really differs


@real
def test_first_poison_is_balanced_over_positions_and_slots(sample):
    position, original = Counter(), 0
    for qid in sample["main"]:
        lay = layout_for(qid)
        first = lay.poison_order[0]
        position[lay.items().index(first)] += 1
        original += first == 0
    # 500 questions: about 100 per position (5 positions), about 125 originals (1 in 4)
    assert all(70 <= position[p] <= 130 for p in range(K + 1))
    assert 90 <= original <= 160