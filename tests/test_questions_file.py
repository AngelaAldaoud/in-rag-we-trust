"""Data contract for data/processed/questions.jsonl.gz (D83, D94-D98)."""

import gzip
import json
from collections import Counter

import pytest

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.textmatch import contains_span, normalize

PATH = PROCESSED_DIR / "questions.jsonl.gz"
META = PROCESSED_DIR / "questions_meta.json"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="questions.jsonl.gz not built")


@pytest.fixture(scope="module")
def records():
    with gzip.open(PATH, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.fixture(scope="module")
def meta():
    return json.loads(META.read_text(encoding="utf-8"))


def text(p):
    return "".join(p["sentences"])


def has(needle, haystack):
    return bool(needle) and f" {needle} " in f" {haystack} "


def test_file_matches_meta(records, meta):
    info = meta["files"]["questions.jsonl.gz"]
    assert len(records) == info["records"]
    assert sha256_of(PATH) == info["sha256"]


def test_ids_splits_ranks(records):
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    sizes = Counter(r["split"] for r in records)
    assert sizes["pilot"] == 50 and sizes["main"] == 500
    assert sorted(r["rank"] for r in records) == list(range(len(records)))
    for r in records:
        expected = "pilot" if r["rank"] < 50 else "main" if r["rank"] < 550 else "reserve"
        assert r["split"] == expected


def test_answer_rules(records):
    for r in records:
        assert contains_span(r["gold_answer"], text(r["answer_passage"])), r["id"]
        assert not has(normalize(r["gold_answer"]), normalize(text(r["bridge_passage"]))), r["id"]


def test_target_candidates(records):
    for r in records:
        candidates = r["target_candidates"]
        assert 1 <= len(candidates) <= 30, r["id"]
        assert r["target_answer"] is None
        n_gold = normalize(r["gold_answer"])
        fields = [normalize(r["question"]),
                  normalize(text(r["answer_passage"])),
                  normalize(text(r["bridge_passage"]))]
        for t in candidates:
            n_t = normalize(t["answer"])
            assert not has(n_t, n_gold) and not has(n_gold, n_t), (r["id"], t)
            assert not any(has(n_t, f) for f in fields), (r["id"], t)