"""Data contract for the BM25 collections (D120)."""

import gzip
import json
from collections import Counter

import pytest

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.retrieval.collection import OUT_DIR, tokenize

pytestmark = pytest.mark.skipif(not (OUT_DIR / "collection_swap.jsonl.gz").exists(),
                                reason="collections not built")


def test_tokenize():
    assert tokenize("The U.S. Army, 1946!") == ["the", "u", "s", "army", "1946"]


@pytest.mark.parametrize("name", ["swap", "correct", "authority", "recency"])
def test_collection_contents(name):
    with gzip.open(OUT_DIR / f"collection_{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    assert len({d["doc_id"] for d in docs}) == len(docs)
    sample = json.loads((PROCESSED_DIR / "sample.json").read_text(encoding="utf-8"))
    per_q = Counter((d["qid"], d["role"]) for d in docs if d["role"] != "distractor")
    for qid in sample["pilot"] + sample["main"]:
        assert per_q[(qid, "clean")] == 4 and per_q[(qid, "poison")] == 4 and per_q[(qid, "bridge")] == 1