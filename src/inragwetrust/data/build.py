"""Build the B1 output: data/processed/questions.jsonl and questions_meta.json.

Schema version 1 (decision D97). Run with:
    uv run python -m inragwetrust.data.build
"""
import gzip
import io
import json
from collections import Counter

import spacy

from inragwetrust.data.entity_types import SPACY_MODEL, coarse_type, get_nlp, tag_answers
from inragwetrust.data.hotpotqa import DEV_SHA256, HF_REVISION, PROJECT_ROOT, load_dev, sha256_of
from inragwetrust.data.pool import build_pool
from inragwetrust.data.sampling import split_ids
from inragwetrust.data.targets import choose_candidates


SEED = 20261009
SCHEMA_VERSION = 1
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def paragraph_dict(p) -> dict:
    """A paragraph as plain JSON: its title and its list of sentences."""
    return {"title": p.title, "sentences": list(p.sentences)}


def build(seed: int = SEED, out_dir=PROCESSED_DIR) -> dict:
    """Run the whole B1 pipeline, write both files, return the metadata."""
    pool, funnel = build_pool(load_dev())
    fine = tag_answers([(q.answer, q.answer_passage.text) for q in pool])
    candidates = choose_candidates(pool, fine, seed)

    keep = [(q, f) for q, f in zip(pool, fine) if candidates[q.id]]
    funnel["has_target_candidate"] = len(keep)
    splits = split_ids([q.id for q, _ in keep], seed)

    records = []
    for q, f in keep:
        split, rank = splits[q.id]
        records.append({
            "schema_version": SCHEMA_VERSION,
            "id": q.id,
            "split": split,
            "rank": rank,
            "question": q.question,
            "level": q.level,
            "gold_answer": q.answer,
            "answer_entity_label": f,
            "answer_type": coarse_type(f),
            "target_candidates": [{"answer": t.answer, "source_id": t.source_id}
                                  for t in candidates[q.id]],
            "target_answer": None,  # chosen in B2 among the candidates (D95)
            "answer_passage": paragraph_dict(q.answer_passage),
            "bridge_passage": paragraph_dict(q.bridge_passage),
            "distractors": [paragraph_dict(p) for p in q.distractors],
        })
    records.sort(key=lambda r: r["rank"])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "questions.jsonl.gz"
    with open(path, "wb") as raw, \
         gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = out_dir / "questions.jsonl"
    if old.exists():
        old.unlink()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "files": {"questions.jsonl.gz": {"records": len(records), "sha256": sha256_of(path)}},
        "source": {"dataset": "hotpotqa/hotpot_qa (distractor, validation)",
                   "hf_revision": HF_REVISION, "sha256": DEV_SHA256},
        "spacy": {"version": spacy.__version__, "model": SPACY_MODEL,
                  "model_version": get_nlp().meta["version"]},
        "funnel": funnel,
        "split_sizes": dict(Counter(r["split"] for r in records)),
        "answer_types_by_split": {
            s: dict(Counter(r["answer_type"] for r in records if r["split"] == s))
            for s in ("pilot", "main", "reserve")
        },
    }
    (out_dir / "questions_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))