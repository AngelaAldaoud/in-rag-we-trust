"""Shared loader: the 22 contexts of every question (18 controlled from C1, 4 retrieved from C4).

Used by the prompt examples (C2), the pilot (D1) and the main runs (E), so all of them
read the contexts in exactly the same way (D147).
"""

from __future__ import annotations

import gzip
import json

from inragwetrust.data.build import PROCESSED_DIR

CONTROLLED_FILE = "contexts.jsonl.gz"
RETRIEVED_FILE = "bm25_contexts.jsonl.gz"
N_CONTEXTS = 22  # 16 poisoned (4 types x 4 rho) + rho=0 + closed-book + 4 BM25
SPLITS = ("pilot", "main")


def read_rows(name: str) -> list[dict]:
    """Read one gzipped JSON-lines file from data/processed/ into a list of dicts."""
    with gzip.open(PROCESSED_DIR / name, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_contexts(split: str | None = None) -> list[dict]:
    """Controlled and retrieved contexts joined per question; optionally one split only.

    Fails loudly if the two files cover different questions, share a context id,
    or a question does not end up with exactly N_CONTEXTS contexts.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")

    rows = read_rows(CONTROLLED_FILE)
    retrieved = {r["id"]: r["contexts"] for r in read_rows(RETRIEVED_FILE)}

    missing = {r["id"] for r in rows} ^ retrieved.keys()
    if missing:
        raise ValueError(f"{len(missing)} questions are in only one context file, e.g. {sorted(missing)[:3]}")

    for row in rows:
        extra = retrieved[row["id"]]
        clash = row["contexts"].keys() & extra.keys()
        if clash:
            raise ValueError(f"{row['id']}: context ids in both files: {sorted(clash)}")
        row["contexts"] = row["contexts"] | extra
        if len(row["contexts"]) != N_CONTEXTS:
            raise ValueError(f"{row['id']}: {len(row['contexts'])} contexts, expected {N_CONTEXTS}")

    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows