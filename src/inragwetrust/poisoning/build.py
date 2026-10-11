"""Build data/processed/poisons.jsonl.gz: clean and poisoned passages (B3, D114).

Run with:
    uv run python -m inragwetrust.poisoning.build
"""

from __future__ import annotations

import gzip
import io
import json
import random
from collections import Counter

from inragwetrust.data.build import PROCESSED_DIR, SEED
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.poisoning.poisons import POISON_TYPES

import re

MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
SHAPED_TYPES = ("date", "number")


def answer_shape(answer: str) -> tuple[bool, int, bool]:
    """(contains a month name, number of digit groups, is a range) — D115."""
    a = answer.lower()
    has_month = any(re.search(rf"\b{m}\b", a) for m in MONTHS)
    n_digits = len(re.findall(r"\d+(?:[.,]\d+)*", a))
    is_range = bool(re.search(r"\bto\b|\bbetween\b|[–-]", a))
    return has_month, n_digits, is_range


def choose_target(question: dict) -> tuple[str, bool]:
    """Target answer and whether the shape rule found a match (D107, D115)."""
    candidates = [c["answer"] for c in question["target_candidates"]]
    if question["answer_type"] in SHAPED_TYPES:
        shape = answer_shape(question["gold_answer"])
        for c in candidates:
            if answer_shape(c) == shape:
                return c, True
        return candidates[0], False
    return candidates[0], True

USED_SPLITS = ("pilot", "main", "reserve")


def read_jsonl_gz(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build(seed: int = SEED, out_dir=PROCESSED_DIR) -> dict:
    """Create 4 clean and 4 x 4 poisoned passages for every usable question."""
    questions = {q["id"]: q for q in read_jsonl_gz(out_dir / "questions.jsonl.gz")}
    usable = [r for r in read_jsonl_gz(out_dir / "paraphrases.jsonl.gz")
              if r["final_split"] in USED_SPLITS]

    records = []
    for r in sorted(usable, key=lambda x: x["rank"]):
        q = questions[r["id"]]
        target, shape_ok = choose_target(q)
        clean = ["".join(q["answer_passage"]["sentences"])] + [
            p["text"] for p in sorted(r["paraphrases"], key=lambda p: p["style"])
        ]
        poisoned = {
            poison.name: [
                poison.apply(text, q["gold_answer"], target,
                             random.Random(f"{seed}-{r['id']}-{i}-{poison.name}"))
                for i, text in enumerate(clean)
            ]
            for poison in POISON_TYPES
        }
        records.append({
            "id": r["id"],
            "rank": r["rank"],
            "final_split": r["final_split"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "target_answer": r["target_answer"],
            "answer_type": q["answer_type"],
            "bridge_passage": "".join(q["bridge_passage"]["sentences"]),
            "clean": clean,
            "poisoned": poisoned,
            "target_answer": target,
            "target_shape_matched": shape_ok,
        })

    path = out_dir / "poisons.jsonl.gz"
    with open(path, "wb") as raw, \
         gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "seed": seed,
        "target_rule": "first safe candidate; for date/number answers the first with the same shape (D107, D115)",
        "shape_fallbacks": sum(not r["target_shape_matched"] for r in records),
        "poison_types": {p.name: list(p.templates) for p in POISON_TYPES},
        "questions": dict(Counter(r["final_split"] for r in records)),
        "passages_per_question": {"clean": 4, "poisoned_per_type": 4},
        "files": {"poisons.jsonl.gz": {"records": len(records), "sha256": sha256_of(path)}},
    }
    (out_dir / "poisons_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))