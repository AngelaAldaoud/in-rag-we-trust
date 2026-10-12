"""Data contract for data/processed/bm25_contexts.jsonl.gz (D145)."""

import gzip
import json

import pytest

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.hotpotqa import sha256_of

PATH = PROCESSED_DIR / "bm25_contexts.jsonl.gz"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="BM25 contexts not built")
ROLES = {"clean", "poisoned", "bridge", "distractor"}


@pytest.fixture(scope="module")
def rows():
    with gzip.open(PATH, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_same_questions_and_order_as_controlled_contexts(rows):
    with gzip.open(PROCESSED_DIR / "contexts.jsonl.gz", "rt", encoding="utf-8") as fh:
        ids = [json.loads(line)["id"] for line in fh]
    assert [r["id"] for r in rows] == ids


def test_four_ranked_contexts_of_five(rows):
    for r in rows:
        assert sorted(r["contexts"]) == ["bm25-authority", "bm25-correct", "bm25-recency", "bm25-swap"]
        for ctx in r["contexts"].values():
            assert [p["position"] for p in ctx] == list(range(5))
            assert [p["rank"] for p in ctx] == [1, 2, 3, 4, 5]
            assert all(a["score"] >= b["score"] for a, b in zip(ctx, ctx[1:]))
            assert all(p["role"] in ROLES for p in ctx)
            assert all((p["slot"] is not None) == (p["own"] and p["role"] in ("clean", "poisoned"))
                       for p in ctx)


def test_checksum_matches_meta():
    meta = json.loads((PROCESSED_DIR / "bm25_contexts_meta.json").read_text(encoding="utf-8"))
    assert meta["outputs"]["bm25_contexts.jsonl.gz"] == sha256_of(PATH)