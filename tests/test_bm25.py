"""BM25 search and summary (C4, D141-D144)."""

import pytest

from inragwetrust.retrieval import bm25
from inragwetrust.retrieval.bm25 import BM25Index, Hit, summarize

DOCS = [
    {"doc_id": "q1:clean:0", "qid": "q1", "role": "clean", "text": "Paris is the capital of France."},
    {"doc_id": "q1:poison:0", "qid": "q1", "role": "poison", "text": "Lyon is the capital of France."},
    {"doc_id": "q1:bridge", "qid": "q1", "role": "bridge", "text": "The Eiffel Tower stands in the French capital."},
    {"doc_id": "q2:clean:0", "qid": "q2", "role": "clean", "text": "The Nile flows through Egypt."},
    {"doc_id": "q2:distractor:0", "qid": "q2", "role": "distractor", "text": "Mount Everest is the highest mountain."},
    {"doc_id": "q3:distractor:0", "qid": "q3", "role": "distractor", "text": "Bananas are yellow fruit."},
]


def test_best_match_first_and_k_results():
    hits = BM25Index(DOCS).search("Where does the NILE flow?", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].doc_id == "q2:clean:0"                     # case-insensitive match
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_ties_are_broken_fairly_and_reproducibly():
    docs = DOCS + [{"doc_id": "z:clean:0", "qid": "z", "role": "clean", "text": "alpha beta"},
                   {"doc_id": "a:clean:0", "qid": "a", "role": "clean", "text": "alpha beta"}]
    index = BM25Index(docs)
    assert index.search("alpha beta", k=2) == index.search("alpha beta", k=2)
    first = [index.search(f"alpha beta q{i}", k=1)[0].doc_id for i in range(200)]
    assert set(first) == {"a:clean:0", "z:clean:0"}
    assert 70 <= first.count("a:clean:0") <= 130          # about half each: no name advantage


def test_search_is_deterministic():
    index = BM25Index(DOCS)
    assert index.search("capital of France") == index.search("capital of France")


def hit(doc_id, qid, role):
    return Hit(1, doc_id, qid, role, 1.0, "")


def test_summary_counts_and_effective_rho():
    hits = [hit("q1:clean:0", "q1", "clean"), hit("q1:poison:0", "q1", "poison"),
            hit("q1:poison:1", "q1", "poison"), hit("q1:bridge", "q1", "bridge"),
            hit("q2:clean:0", "q2", "clean")]
    s = summarize(hits, "q1")
    assert s == {"gold": True, "own_clean": 1, "own_poison": 2, "bridge": True,
                 "other": 1, "effective_rho": 2 / 3}


def test_summary_without_answer_bearing_passages():
    s = summarize([hit("q2:clean:0", "q2", "clean"), hit("q1:bridge", "q1", "bridge")], "q1")
    assert s["effective_rho"] is None and s["gold"] is False and s["bridge"] is True


@pytest.mark.skipif(not (bm25.COLLECTION_DIR / "collection_swap.jsonl.gz").exists(),
                    reason="collections not built")
def test_real_collection_search():
    docs = bm25.load_collection("swap")
    assert len(docs) == 9301
    hits = BM25Index(docs).search(
        "Which American chain of fruit drink beverage stores has an eponymous beverage?")
    assert len(hits) == 5 and all(a.score >= b.score for a, b in zip(hits, hits[1:]))