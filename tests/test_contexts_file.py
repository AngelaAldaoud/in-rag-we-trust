
"""Data contract for data/processed/contexts.jsonl.gz (D126)."""

import gzip
import json

import pytest

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.hotpotqa import sha256_of

PATH = PROCESSED_DIR / "contexts.jsonl.gz"
META = PROCESSED_DIR / "contexts_meta.json"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="contexts not built")


@pytest.fixture(scope="module")
def rows():
    with gzip.open(PATH, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_matches_frozen_sample(rows):
    sample = json.loads((PROCESSED_DIR / "sample.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == sample["pilot"] + sample["main"]


def test_poisoned_count_matches_rho(rows):
    for r in rows:
        assert len(r["contexts"]) == 18 and r["contexts"]["closedbook"] == []
        for cid, ctx in r["contexts"].items():
            if cid == "closedbook":
                continue
            rho = float(cid.rsplit("-", 1)[1])
            assert sum(p["role"] == "poisoned" for p in ctx) == round(rho * 4)


def test_checksum_matches_meta():
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["outputs"]["contexts.jsonl.gz"] == sha256_of(PATH)