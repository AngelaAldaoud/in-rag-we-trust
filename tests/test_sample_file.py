"""Data contract for data/processed/sample.json (D117)."""

import gzip
import json

import pytest

from inragwetrust.data.build import PROCESSED_DIR

SAMPLE = PROCESSED_DIR / "sample.json"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample.json not built")


@pytest.fixture(scope="module")
def sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sizes_unique_disjoint(sample):
    assert len(sample["pilot"]) == 50 and len(sample["main"]) == 500
    ids = sample["pilot"] + sample["main"]
    assert len(set(ids)) == 550


def test_all_ids_have_poisons(sample):
    with gzip.open(PROCESSED_DIR / "poisons.jsonl.gz", "rt", encoding="utf-8") as fh:
        poisoned = {json.loads(line)["id"] for line in fh}
    assert set(sample["pilot"] + sample["main"]) <= poisoned


def test_no_blocking_failures_in_final_sample(sample):
    assert sample["poison_checks_final_sample"]["blocking_failures"] == {}