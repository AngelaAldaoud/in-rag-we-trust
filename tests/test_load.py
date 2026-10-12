"""Tests for the shared context loader (D147)."""

import pytest

from inragwetrust.retrieval import load
from inragwetrust.retrieval.load import N_CONTEXTS, load_contexts

CONTROLLED = [f"c{i}" for i in range(18)]
RETRIEVED = [f"bm25-{i}" for i in range(4)]


# --- real files ---------------------------------------------------------------

def test_real_files_join_to_22_contexts_per_question():
    rows = load_contexts()
    assert len(rows) == 550
    assert len({r["id"] for r in rows}) == 550
    assert all(len(r["contexts"]) == N_CONTEXTS for r in rows)


def test_every_question_has_the_same_context_ids():
    rows = load_contexts()
    ids = set(rows[0]["contexts"])
    assert all(set(r["contexts"]) == ids for r in rows)
    assert sum(cid.startswith("bm25-") for cid in ids) == 4
    assert {"closedbook", "clean-0.00"} <= ids


def test_closed_book_is_empty_and_other_contexts_have_five_passages():
    for row in load_contexts("pilot"):
        for cid, passages in row["contexts"].items():
            assert len(passages) == (0 if cid == "closedbook" else 5), (row["id"], cid)


def test_splits_have_the_right_size_and_do_not_overlap():
    pilot, main = load_contexts("pilot"), load_contexts("main")
    assert (len(pilot), len(main)) == (50, 500)
    assert {r["split"] for r in pilot} == {"pilot"}
    assert not {r["id"] for r in pilot} & {r["id"] for r in main}


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split"):
        load_contexts("pilt")


# --- fake files: every failure check must fire ----------------------------------

def row(qid, context_ids):
    return {"id": qid, "split": "main", "contexts": {cid: [] for cid in context_ids}}


def use_fake_files(monkeypatch, controlled, retrieved):
    data = {load.CONTROLLED_FILE: controlled, load.RETRIEVED_FILE: retrieved}
    monkeypatch.setattr(load, "read_rows", lambda name: data[name])


def test_fake_files_join(monkeypatch):
    use_fake_files(monkeypatch, [row("q1", CONTROLLED)], [row("q1", RETRIEVED)])
    assert len(load_contexts()[0]["contexts"]) == N_CONTEXTS


def test_question_missing_from_one_file_fails(monkeypatch):
    use_fake_files(monkeypatch, [row("q1", CONTROLLED), row("q2", CONTROLLED)],
                   [row("q1", RETRIEVED)])
    with pytest.raises(ValueError, match="only one context file"):
        load_contexts()


def test_context_id_in_both_files_fails(monkeypatch):
    use_fake_files(monkeypatch, [row("q1", CONTROLLED)], [row("q1", ["c0", *RETRIEVED[:3]])])
    with pytest.raises(ValueError, match="in both files"):
        load_contexts()


def test_wrong_number_of_contexts_fails(monkeypatch):
    use_fake_files(monkeypatch, [row("q1", CONTROLLED[:17])], [row("q1", RETRIEVED)])
    with pytest.raises(ValueError, match="expected 22"):
        load_contexts()