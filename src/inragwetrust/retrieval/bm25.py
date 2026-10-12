"""BM25 retrieval over the poison-type collections (C4, D141-D144)."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

import hashlib

from inragwetrust.data.build import SEED

from inragwetrust.retrieval.collection import OUT_DIR as COLLECTION_DIR
from inragwetrust.retrieval.collection import tokenize

K = 5                                   # passages per retrieved context (D142)


@dataclass(frozen=True)
class Hit:
    rank: int                           # 1 = best match
    doc_id: str
    qid: str                            # the question this document belongs to
    role: str                           # clean, poison, bridge or distractor
    score: float
    text: str


def load_collection(poison_type: str, folder=COLLECTION_DIR) -> list[dict]:
    with gzip.open(folder / f"collection_{poison_type}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def tie_key(query: str, doc_id: str, seed: int = SEED) -> str:
    """A seeded random order among equal scores: fair to every document, reproducible (D146)."""
    return hashlib.sha256(f"{seed}|{query}|{doc_id}".encode()).hexdigest()

class BM25Index:
    """Standard BM25 (k1 = 1.5, b = 0.75) over one collection, Block B tokenisation (D141)."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    def search(self, query: str, k: int = K) -> list[Hit]:
        """Top-k documents by score; exact ties ordered by a seeded random key (D142, D146)."""
        scores = self.bm25.get_scores(tokenize(query))
        by_score = sorted(range(len(self.docs)), key=lambda i: -scores[i])
        cutoff = scores[by_score[min(k, len(by_score)) - 1]]
        candidates = [i for i in by_score if scores[i] >= cutoff]
        best = sorted(candidates,
                      key=lambda i: (-scores[i], tie_key(query, self.docs[i]["doc_id"])))[:k]
        return [Hit(rank, self.docs[i]["doc_id"], self.docs[i]["qid"], self.docs[i]["role"],
                    float(scores[i]), self.docs[i]["text"])
                for rank, i in enumerate(best, start=1)]

def summarize(hits: list[Hit], qid: str) -> dict:
    """The D144 measures for one question's result list."""
    own = [h for h in hits if h.qid == qid]
    n_clean = sum(h.role == "clean" for h in own)
    n_poison = sum(h.role == "poison" for h in own)
    return {
        "gold": any(h.doc_id == f"{qid}:clean:0" for h in hits),
        "own_clean": n_clean,
        "own_poison": n_poison,
        "bridge": any(h.role == "bridge" for h in own),
        "other": len(hits) - len(own),
        "effective_rho": n_poison / (n_clean + n_poison) if n_clean + n_poison else None,
    }