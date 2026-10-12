"""Report tables from the scored file and the judge file (D4).

    python -m inragwetrust.evaluation.report --run runs/pilot --split pilot --out results/pilot
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from inragwetrust.evaluation.metrics import Rate, greedy_metrics, sc_metrics
from inragwetrust.models.runs import read_shard
from inragwetrust.retrieval.load import SPLITS, load_contexts

SCORED_PATH = Path("_scored") / "scored.jsonl.gz"
JUDGE_PATH = Path("_judge") / "judge.jsonl.gz"


def level(row: dict) -> str:
    if row["run"] == "closed_book":
        return "closed-book"
    if row["run"] == "retrieval":
        return "bm25"
    return f"{row['rho']:.2f}"


def group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(level(r) if k == "level" else r[k] for k in keys), []).append(r)
    return dict(sorted(out.items(), key=lambda kv: tuple(str(x) for x in kv[0])))


def _num(x: float):
    return "" if isinstance(x, float) and math.isnan(x) else round(x, 4)


def flatten(metrics: dict) -> dict:
    """Rates become value, _lo, _hi, _n columns; empty cells where n = 0."""
    flat = {}
    for name, v in metrics.items():
        if isinstance(v, Rate):
            flat[name], flat[f"{name}_lo"], flat[f"{name}_hi"] = _num(v.value), _num(v.lo), _num(v.hi)
            flat[f"{name}_n"] = v.n
        else:
            flat[name] = _num(v) if isinstance(v, float) else v
    return flat


def greedy_table(rows, keys, golds, flags, verdicts) -> list[dict]:
    return [dict(zip(keys, k)) | flatten(greedy_metrics(rs, golds, flags, verdicts))
            for k, rs in group(rows, keys).items()]


def sc_table(rows, keys) -> list[dict]:
    return [dict(zip(keys, k)) | flatten(sc_metrics(rs)) for k, rs in group(rows, keys).items()]


def format_table(rows: list[dict]) -> list[dict]:
    table = []
    for (model, mode), rs in group(rows, ("model", "mode")).items():
        n = len(rs)
        table.append({"model": model, "mode": mode, "n": n,
                      "line_start": _num(sum(r["position"] == "line-start" for r in rs) / n),
                      "mid_line": _num(sum(r["position"] == "mid-line" for r in rs) / n),
                      "no_answer_line": _num(sum(r["position"] is None for r in rs) / n),
                      "truncated": _num(sum(bool(r["truncated"]) for r in rs) / n),
                      "cut": _num(sum(bool(r["cut"]) for r in rs) / n)})
    return table


def judge_maps(judge_rows: list[dict]) -> tuple[dict, dict]:
    flags = {(r["condition_id"], r["question_id"]): r["label"] for r in judge_rows if r["task"] == "flag"}
    verdicts = {(r["condition_id"], r["question_id"]): r["label"] for r in judge_rows if r["task"] == "verdict"}
    return flags, verdicts


def build_tables(scored: list[dict], judge_rows: list[dict], golds: dict[str, str]) -> dict[str, list[dict]]:
    greedy = [r for r in scored if r["mode"] == "greedy"]
    samples = [r for r in scored if r["mode"] == "samples"]
    flags, verdicts = judge_maps(judge_rows)
    controlled = [r for r in greedy if r["run"] not in ("closed_book", "retrieval")]
    retrieval = [r for r in greedy if r["run"] == "retrieval"]
    closed = [r for r in greedy if r["run"] == "closed_book"]
    args = (golds, flags, verdicts)
    return {
        "by_rho": greedy_table(controlled, ("model", "prompt", "level"), *args),
        "by_rho_pooled": greedy_table(controlled, ("model", "level"), *args),
        "grid": greedy_table(controlled, ("model", "prompt", "poison_type", "level"), *args),
        "retrieval": greedy_table(retrieval, ("model", "prompt", "context_id"), *args),
        "closed_book": greedy_table(closed, ("model",), *args),
        "sc": sc_table(samples, ("model", "prompt", "level")),
        "format": format_table(scored),
    }


def write_tables(tables: dict[str, list[dict]], out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in tables.items():
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    golds = {r["id"]: r["gold_answer"] for r in load_contexts(args.split)}
    tables = build_tables(read_shard(args.run / SCORED_PATH), read_shard(args.run / JUDGE_PATH), golds)
    for path in write_tables(tables, args.out):
        print("wrote", path)
    print("\nmodel    level  Acc    ASR    no-usable  Flag   Acc+ (exploratory)")
    for r in tables["by_rho_pooled"]:
        print(f"{r['model']:8s} {r['level']:6s} {r['acc']:<6} {r['asr']:<6} {r['no_usable']:<10} "
              f"{r['flag']:<6} {r['acc_plus']}")


if __name__ == "__main__":
    main()