"""Run conditions and save raw outputs in resumable shards (C3, D136-D137)."""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from inragwetrust.conditions import MODELS, Condition
from inragwetrust.models.chat import REVISIONS, render
from inragwetrust.models.generator import MAX_TOKENS, MODES, Request, request_seed
from inragwetrust.models.prompts import messages
from inragwetrust.retrieval import build as files

SCHEMA_VERSION = 1


def shard_path(out_dir, condition: Condition, mode: str) -> Path:
    return Path(out_dir) / mode / condition.model / f"{condition.condition_id}.jsonl.gz"


def write_atomic(path: Path, records: list[dict]) -> None:
    """Write to a temporary name, then rename: a final name always means a complete shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    files.write_jsonl_gz(tmp, records)
    os.replace(tmp, path)


def read_shard(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def context_texts(row: dict, context_id: str) -> list[str]:
    if context_id not in row["contexts"]:
        raise KeyError(f"{row['id']}: no context '{context_id}' (BM25 contexts come from C4)")
    return [p["text"] for p in row["contexts"][context_id]]


def run_shard(generator, tokenizer, rows, condition: Condition, mode: str,
              out_dir, env: dict) -> dict:
    """Run one condition on all rows, unless its shard already exists."""
    path = shard_path(out_dir, condition, mode)
    if path.exists():
        return {"condition_id": condition.condition_id, "status": "skipped"}

    rendered = [render(tokenizer, messages(r["question"],
                                           context_texts(r, condition.context_id),
                                           condition.prompt)) for r in rows]
    requests = [Request(condition.condition_id, r["id"], p.token_ids)
                for r, p in zip(rows, rendered)]

    start = time.perf_counter()
    outputs = generator.generate(requests, mode)
    seconds = round(time.perf_counter() - start, 3)
    if len(outputs) != len(requests):
        raise RuntimeError("the engine returned a different number of outputs")

    records = []
    for row, prompt, completions in zip(rows, rendered, outputs):
        records.append({
            "schema_version": SCHEMA_VERSION,
            "question_id": row["id"], "split": row["split"],
            "condition_id": condition.condition_id, "run": condition.run,
            "model": condition.model, "model_hf": MODELS[condition.model],
            "revision": REVISIONS[condition.model], "prompt": condition.prompt,
            "poison_type": condition.poison_type, "rho": condition.rho,
            "context_id": condition.context_id,
            "mode": mode, **MODES[mode], "max_tokens": MAX_TOKENS,
            "seed": request_seed(condition.condition_id, row["id"], mode),
            "prompt_hash": prompt.prompt_hash, "prompt_tokens": prompt.n_tokens,
            "completions": [asdict(c) for c in completions],
            "shard_seconds": seconds, "shard_size": len(rows), "env": env,
        })
    write_atomic(path, records)
    return {"condition_id": condition.condition_id, "status": "done",
            "prompts": len(rows), "seconds": seconds,
            "prompt_tokens": sum(p.n_tokens for p in rendered),
            "output_tokens": sum(c.n_tokens for outs in outputs for c in outs)}


def run_model(generator, tokenizer, rows, conditions, mode, out_dir, env, log=print) -> list[dict]:
    """All of one model's conditions, shard by shard, with a progress line per shard."""
    todo = [c for c in conditions if c.model == generator.model]
    summaries = []
    for i, condition in enumerate(todo, start=1):
        s = run_shard(generator, tokenizer, rows, condition, mode, out_dir, env)
        summaries.append(s)
        extra = (f"{s['prompts']} prompts, {s['seconds']:.1f} s, {s['output_tokens']} out-tokens"
                 if s["status"] == "done" else "")
        log(f"[{i}/{len(todo)}] {s['status']:7s} {condition.condition_id} {extra}")
    return summaries