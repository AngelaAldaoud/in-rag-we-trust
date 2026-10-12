"""Top-5 BM25 retrieval for every question and poison type (C4, D141-D145).

Run with:
    uv run python -m inragwetrust.retrieval.build_bm25
"""

from __future__ import annotations

import gzip
import json
import statistics
from collections import Counter

from inragwetrust.conditions import context_id
from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.retrieval.bm25 import COLLECTION_DIR, BM25Index, load_collection, summarize
from inragwetrust.retrieval.build import write_jsonl_gz
from inragwetrust.retrieval.contexts import TYPE_NAMES


def passage(hit, position: int, qid: str) -> dict:
    """A retrieved document in the format of a controlled passage, plus retrieval fields."""
    own = hit.qid == qid
    answer_bearing = own and hit.role in ("clean", "poison")
    return {"position": position,
            "role": "poisoned" if hit.role == "poison" else hit.role,
            "slot": int(hit.doc_id.rsplit(":", 1)[1]) if answer_bearing else None,
            "text": hit.text, "own": own, "doc_id": hit.doc_id,
            "rank": hit.rank, "score": round(hit.score, 4)}


def top1_role(ctx: list[dict]) -> str:
    first = ctx[0]
    return f"own {first['role']}" if first["own"] else "other"


def rates(rows: list[dict], ptype: str) -> dict:
    """Retrieval statistics for one poison type over the given rows (D144)."""
    s = [r["summary"][ptype] for r in rows]
    n = len(s)
    eff = [x["effective_rho"] for x in s if x["effective_rho"] is not None]
    return {
        "questions": n,
        "gold_hit": round(sum(x["gold"] for x in s) / n, 3),
        "bridge_hit": round(sum(x["bridge"] for x in s) / n, 3),
        "any_poison": round(sum(x["own_poison"] > 0 for x in s) / n, 3),
        "mean_own_clean": round(statistics.mean(x["own_clean"] for x in s), 2),
        "mean_own_poison": round(statistics.mean(x["own_poison"] for x in s), 2),
        "mean_other": round(statistics.mean(x["other"] for x in s), 2),
        "no_answer_bearing": round(sum(x["effective_rho"] is None for x in s) / n, 3),
        "own_poison_distribution": dict(sorted(Counter(x["own_poison"] for x in s).items())),
        "mean_effective_rho": round(statistics.mean(eff), 3) if eff else None,
        "top1_role": dict(Counter(top1_role(r["contexts"][context_id("retrieval", ptype)])
                                  for r in rows)),
    }


def build(out_dir=PROCESSED_DIR) -> dict:
    with gzip.open(out_dir / "contexts.jsonl.gz", "rt", encoding="utf-8") as fh:
        base = [json.loads(line) for line in fh]
    rows = [{k: r[k] for k in ("id", "split", "question", "gold_answer", "target_answer")}
            | {"contexts": {}, "summary": {}} for r in base]

    for ptype in TYPE_NAMES:
        index = BM25Index(load_collection(ptype))
        for row in rows:
            hits = index.search(row["question"])
            row["contexts"][context_id("retrieval", ptype)] = [
                passage(h, pos, row["id"]) for pos, h in enumerate(hits)]
            row["summary"][ptype] = summarize(hits, row["id"])

    path = out_dir / "bm25_contexts.jsonl.gz"
    write_jsonl_gz(path, rows)
    main = [r for r in rows if r["split"] == "main"]
    meta = {
        "schema_version": 1,
        "k": 5, "bm25": "rank_bm25.BM25Okapi, k1=1.5, b=0.75",
        "questions": dict(Counter(r["split"] for r in rows)),
        "rates_main": {t: rates(main, t) for t in TYPE_NAMES},
        "inputs": {"contexts.jsonl.gz": sha256_of(out_dir / "contexts.jsonl.gz"),
                   **{f"collection_{t}.jsonl.gz": sha256_of(COLLECTION_DIR / f"collection_{t}.jsonl.gz")
                      for t in TYPE_NAMES}},
        "outputs": {"bm25_contexts.jsonl.gz": sha256_of(path)},
    }
    (out_dir / "bm25_contexts_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    meta = build()
    for ptype, r in meta["rates_main"].items():
        print(f"\n== {ptype}")
        for key, value in r.items():
            print(f"  {key:24s} {value}")