"""Score every completion of a run folder: parsed answer, outcome, hallucination flags (D152-D157).

    python -m inragwetrust.evaluation.score --run runs/pilot --split pilot
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from inragwetrust.evaluation.outcomes import OUTCOMES, score
from inragwetrust.evaluation.parse import parse_answer
from inragwetrust.models.runs import read_shard
from inragwetrust.retrieval import build as files
from inragwetrust.retrieval.load import SPLITS, load_contexts

SCORED_DIR, SCORED_FILE = "_scored", "scored.jsonl.gz"
KEYS = ("question_id", "split", "condition_id", "run", "model", "prompt",
        "poison_type", "rho", "context_id", "mode")


def shard_paths(run_dir) -> list[Path]:
    """Shards only: <mode>/<model>/<condition>.jsonl.gz (not _logs, not _scored)."""
    return sorted(Path(run_dir).glob("*/*/*.jsonl.gz"))


def score_shards(paths, rows) -> list[dict]:
    """One row per completion, with the record's condition fields and the D2 results."""
    by_id = {r["id"]: r for r in rows}
    scored = []
    for path in paths:
        for record in read_shard(path):
            row = by_id[record["question_id"]]
            passages = [p["text"] for p in row["contexts"][record["context_id"]]]
            for i, completion in enumerate(record["completions"]):
                parsed = parse_answer(completion["text"], completion["finish_reason"])
                result = score(parsed, row["gold_answer"], row["target_answer"], passages)
                scored.append({**{k: record[k] for k in KEYS}, "sample": i,
                               "n_tokens": completion["n_tokens"],
                               "finish_reason": completion["finish_reason"],
                               "answer": parsed.answer, "position": parsed.position,
                               "truncated": parsed.truncated, "cut": parsed.cut,
                               "outcome": result.outcome, "hall": result.hall,
                               "hall_strict": result.hall_strict,
                               "missing_facts": (list(result.missing_facts)if result.missing_facts is not None else None)})
    return scored


def summary(scored: list[dict]) -> list[str]:
    """Outcome shares per model and mode; greedy Correct/Poisoned by rho (controlled contexts)."""
    lines = []
    groups = Counter((r["model"], r["mode"]) for r in scored)
    counts = Counter((r["model"], r["mode"], r["outcome"]) for r in scored)
    for (model, mode), n in sorted(groups.items()):
        shares = " | ".join(f"{o} {counts[(model, mode, o)] / n:6.1%}" for o in OUTCOMES)
        lines.append(f"{model:8s} {mode:8s} n={n:6d} | {shares}")
    greedy = [r for r in scored if r["mode"] == "greedy" and r["rho"] is not None]
    for model in sorted({r["model"] for r in greedy}):
        cells = []
        for rho in sorted({r["rho"] for r in greedy}):
            xs = [r for r in greedy if r["model"] == model and r["rho"] == rho]
            c = Counter(r["outcome"] for r in xs)
            cells.append(f"rho {rho:.2f}: C {c['Correct'] / len(xs):5.1%} P {c['Poisoned'] / len(xs):5.1%}")
        lines.append(f"{model:8s} greedy  " + " | ".join(cells))
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="run folder, e.g. runs/pilot")
    parser.add_argument("--split", required=True, choices=SPLITS)
    args = parser.parse_args(argv)

    paths = shard_paths(args.run)
    if not paths:
        raise SystemExit(f"no shards under {args.run}")
    scored = score_shards(paths, load_contexts(args.split))
    out = args.run / SCORED_DIR / SCORED_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    files.write_jsonl_gz(out, scored)
    print(f"{len(paths)} shards, {len(scored)} completions scored -> {out}")
    print("\n".join(summary(scored)))


if __name__ == "__main__":
    main()