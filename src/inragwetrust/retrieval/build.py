"""Build the controlled contexts and the condition table (C1, D126).

Run with:
    uv run python -m inragwetrust.retrieval.build
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from dataclasses import asdict

from inragwetrust.conditions import MODELS, condition_table, context_id, context_ids
from inragwetrust.data.build import PROCESSED_DIR, SEED
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.poisoning.build import read_jsonl_gz
from inragwetrust.retrieval.contexts import (
    RHOS, TYPE_NAMES, build_context, closed_book_context, layout_for,
)

CONFIGS_DIR = PROCESSED_DIR.parents[1] / "configs"


def controlled_contexts(rec: dict, layout) -> dict[str, list[dict]]:
    """The 18 controlled contexts of one question, named as in the condition table."""
    out = {context_id("closed_book"): [asdict(p) for p in closed_book_context(rec)],
           context_id("baseline"): [asdict(p) for p in build_context(rec, None, 0.0, layout)]}
    for ptype in TYPE_NAMES:
        for rho in RHOS[1:]:
            out[context_id("poisoned", ptype, rho)] = [
                asdict(p) for p in build_context(rec, ptype, rho, layout)]
    return out


def write_jsonl_gz(path, rows) -> None:
    """Compressed JSON Lines with no timestamp, so equal content gives equal bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    path.write_bytes(buf.getvalue())


def write_conditions_csv(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["condition_id", "run", "model", "model_hf", "prompt",
                    "poison_type", "rho", "context_id"])
        for c in condition_table():
            w.writerow([c.condition_id, c.run, c.model, MODELS[c.model], c.prompt,
                        c.poison_type or "", "" if c.rho is None else f"{c.rho:.2f}",
                        c.context_id])


def build(seed: int = SEED, out_dir=PROCESSED_DIR, configs_dir=CONFIGS_DIR) -> dict:
    sample = json.loads((out_dir / "sample.json").read_text(encoding="utf-8"))
    order = [("pilot", q) for q in sample["pilot"]] + [("main", q) for q in sample["main"]]
    records = {r["id"]: r for r in read_jsonl_gz(out_dir / "poisons.jsonl.gz")}
    expected = [c for c in context_ids() if not c.startswith("bm25-")]

    rows = []
    for split, qid in order:
        rec, layout = records[qid], layout_for(qid, seed)
        contexts = controlled_contexts(rec, layout)
        if list(contexts) != expected:
            raise ValueError(f"{qid}: contexts do not match the condition table")
        rows.append({
            "id": qid, "split": split, "question": rec["question"],
            "gold_answer": rec["gold_answer"], "target_answer": rec["target_answer"],
            "layout": asdict(layout), "contexts": contexts,
        })

    contexts_path = out_dir / "contexts.jsonl.gz"
    conditions_path = configs_dir / "conditions.csv"
    write_jsonl_gz(contexts_path, rows)
    write_conditions_csv(conditions_path)

    meta = {
        "schema_version": 1,
        "seed": seed,
        "questions": dict(Counter(split for split, _ in order)),
        "contexts_per_question": len(expected),
        "context_ids": expected,
        "bm25_contexts": "built in C4, separate file",
        "conditions": {"total": len(condition_table()),
                       **Counter(c.run for c in condition_table())},
        "inputs": {"sample.json": sha256_of(out_dir / "sample.json"),
                   "poisons.jsonl.gz": sha256_of(out_dir / "poisons.jsonl.gz")},
        "outputs": {"contexts.jsonl.gz": sha256_of(contexts_path),
                    "conditions.csv": sha256_of(conditions_path)},
    }
    (out_dir / "contexts_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    meta = build()
    print(json.dumps({k: v for k, v in meta.items() if k != "context_ids"}, indent=2))