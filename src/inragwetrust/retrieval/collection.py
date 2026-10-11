"""BM25 document collections, one per poison type (decisions D43, D120).

Run with:
    uv run python -m inragwetrust.retrieval.collection
"""

from __future__ import annotations

import gzip
import io
import json
import re
from collections import Counter

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.poisoning.build import read_jsonl_gz
from inragwetrust.poisoning.poisons import POISON_TYPES

OUT_DIR = PROCESSED_DIR / "bm25"
TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens: the tokenisation used for BM25 (Block C reuses it)."""
    return TOKEN.findall(text.lower())


def write_jsonl_gz(path, rows) -> None:
    """Reproducible compressed JSON Lines (D98)."""
    with open(path, "wb") as raw, \
         gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_collections(out_dir=OUT_DIR) -> dict:
    sample = json.loads((PROCESSED_DIR / "sample.json").read_text(encoding="utf-8"))
    ids = sample["pilot"] + sample["main"]
    poisons = {r["id"]: r for r in read_jsonl_gz(PROCESSED_DIR / "poisons.jsonl.gz")}
    questions = {q["id"]: q for q in read_jsonl_gz(PROCESSED_DIR / "questions.jsonl.gz")}

    # Documents shared by every collection: clean and bridge passages, then distractors.
    shared = []
    for qid in ids:
        for i, text in enumerate(poisons[qid]["clean"]):
            shared.append({"doc_id": f"{qid}:clean:{i}", "qid": qid, "role": "clean", "text": text})
        shared.append({"doc_id": f"{qid}:bridge", "qid": qid, "role": "bridge",
                       "text": poisons[qid]["bridge_passage"]})
    seen = {d["text"] for d in shared}
    for qid in ids:
        for j, p in enumerate(questions[qid]["distractors"]):
            text = "".join(p["sentences"])
            if text not in seen:
                seen.add(text)
                shared.append({"doc_id": f"{qid}:distractor:{j}", "qid": qid,
                               "role": "distractor", "text": text})

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"questions": len(ids), "collections": {}}
    for poison in POISON_TYPES:
        rows = list(shared) + [
            {"doc_id": f"{qid}:{poison.name}:{i}", "qid": qid, "role": "poison", "text": text}
            for qid in ids for i, text in enumerate(poisons[qid]["poisoned"][poison.name])
        ]
        path = out_dir / f"collection_{poison.name}.jsonl.gz"
        write_jsonl_gz(path, rows)
        meta["collections"][poison.name] = {
            "documents": len(rows),
            "by_role": dict(Counter(r["role"] for r in rows)),
            "sha256": sha256_of(path),
        }
    (out_dir / "collections_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(build_collections(), indent=2))