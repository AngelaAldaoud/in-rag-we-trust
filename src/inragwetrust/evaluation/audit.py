"""Hand audit of the outcome rules and the hallucination flag (D159).

    python -m inragwetrust.evaluation.audit sheet   --run runs/pilot --split pilot --out data/annotations
    python -m inragwetrust.evaluation.audit results --out data/annotations
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from inragwetrust.data.build import SEED
from inragwetrust.evaluation.outcomes import content_words, starts_with_unknown
from inragwetrust.evaluation.parse import head
from inragwetrust.models.runs import read_shard
from inragwetrust.retrieval.load import SPLITS, load_contexts
from inragwetrust.textmatch import normalize

OUTCOME_QUOTA = {"Correct": 10, "Poisoned": 10, "Both": 5, "OtherWrong": 20,
                 "Abstain-unknown": 5, "Abstain-phrase": 10}
HALL_QUOTA = {"mistral": 10, "qwen": 5, "llama": 5}
OUTCOME_LABELS = {"C": "Correct", "P": "Poisoned", "B": "Both", "W": "OtherWrong", "A": "Abstain", "?": "unclear"}
HALL_LABELS = {"I": "invented", "N": "not invented", "?": "unclear"}
SHEET, KEY = "d2_audit_sheet.txt", "d2_audit_key.json"
_ITEM = re.compile(r"^### (\S+)")
_LABEL = re.compile(r"^label:[ \t]*(\S*)")


def stratum(row: dict) -> str:
    if row["outcome"] == "Abstain":
        return "Abstain-unknown" if starts_with_unknown(row["answer"]) else "Abstain-phrase"
    return row["outcome"]


def draw(scored: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Seeded stratified samples of greedy answers: set O by outcome, set H (Hall=True) by model."""
    rng = random.Random(seed)
    greedy = [r for r in scored if r["mode"] == "greedy" and r["answer"] is not None]
    items = []
    for name, k in OUTCOME_QUOTA.items():
        pool = [r for r in greedy if stratum(r) == name]
        items += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(items)
    hall = []
    for model, k in HALL_QUOTA.items():
        pool = [r for r in greedy if r["model"] == model and r["outcome"] == "OtherWrong"
                and r["hall"] and r["context_id"] != "closedbook"]
        hall += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(hall)
    return items, hall


def missing_words(row: dict, question: dict) -> list[str]:
    passages = " ".join(p["text"] for p in question["contexts"][row["context_id"]])
    return sorted(content_words(head(row["answer"])) - set(normalize(passages).split()))


def write_sheet(out_dir, items, hall, rows) -> None:
    """The labelling sheet (no outcomes shown in set O) and the answer key, kept separate."""
    by_id = {r["id"]: r for r in rows}
    lines = ["# D2 audit. Type ONE letter after 'label:' and save. Nothing else needs editing.",
             "# Set O: what did the model commit to?  " + "  ".join(f"{k}={v}" for k, v in OUTCOME_LABELS.items()),
             "# Set H: are the listed words invented content?  " + "  ".join(f"{k}={v}" for k, v in HALL_LABELS.items()),
             ""]
    key = {}
    for i, r in enumerate(items, 1):
        q = by_id[r["question_id"]]
        lines += [f"### O{i}", f"question: {q['question']}", f"gold:     {q['gold_answer']}",
                  f"target:   {q['target_answer']}", f"answer:   {r['answer']}", "label: ", ""]
        key[f"O{i}"] = {k: r[k] for k in ("question_id", "condition_id", "model", "outcome")} | {"stratum": stratum(r)}
    for i, r in enumerate(hall, 1):
        q = by_id[r["question_id"]]
        lines += [f"### H{i}", f"question: {q['question']}", f"answer:   {r['answer']}",
                  f"words in no passage: {', '.join(missing_words(r, q))}", "label: ", ""]
        key[f"H{i}"] = {k: r[k] for k in ("question_id", "condition_id", "model", "outcome", "hall")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SHEET).write_text("\n".join(lines), encoding="utf-8")
    (out_dir / KEY).write_text(json.dumps(key, indent=2), encoding="utf-8")


def read_labels(path) -> dict[str, str]:
    """Item id -> the letter typed after 'label:' (empty if not labelled yet)."""
    labels, current = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := _ITEM.match(line):
            current = m.group(1)
        elif (m := _LABEL.match(line)) and current:
            labels[current] = m.group(1).upper()
    return labels


def results(labels: dict[str, str], key: dict) -> dict:
    """Agreement of the rules with the hand labels, per stratum, and the precision of the Hall flag."""
    missing = [i for i in key if labels.get(i, "") == ""]
    bad = {i: v for i, v in labels.items() if v and v not in (OUTCOME_LABELS if i[0] == "O" else HALL_LABELS)}
    if missing or bad:
        raise ValueError(f"unlabelled: {missing}; invalid letters: {bad}")
    agree, confusion = Counter(), Counter()
    for item, info in key.items():
        if item.startswith("O"):
            human = OUTCOME_LABELS[labels[item]]
            confusion[(info["outcome"], human)] += 1
            agree[(info["stratum"], human == info["outcome"])] += 1
    hall = Counter((key[i]["model"], HALL_LABELS[labels[i]]) for i in key if i.startswith("H"))
    return {"agree": agree, "confusion": confusion, "hall": hall}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sheet", "results"])
    parser.add_argument("--run", type=Path, help="run folder with _scored/scored.jsonl.gz (sheet)")
    parser.add_argument("--split", choices=SPLITS, help="question split of that run (sheet)")
    parser.add_argument("--out", required=True, type=Path, help="annotation folder")
    args = parser.parse_args(argv)

    if args.action == "sheet":
        if (args.out / SHEET).exists():
            raise SystemExit(f"{args.out / SHEET} exists: it may hold labels; refusing to overwrite")
        items, hall = draw(read_shard(args.run / "_scored" / "scored.jsonl.gz"))
        write_sheet(args.out, items, hall, load_contexts(args.split))
        print(f"wrote {len(items)} outcome items and {len(hall)} hallucination items to {args.out / SHEET}")
        return

    r = results(read_labels(args.out / SHEET), json.loads((args.out / KEY).read_text(encoding="utf-8")))
    print("set O: rule outcome agrees with hand label, by stratum")
    for s in OUTCOME_QUOTA:
        yes, no = r["agree"][(s, True)], r["agree"][(s, False)]
        if yes + no:
            print(f"  {s:16s} {yes:2d}/{yes + no:2d}")
    print("set O: confusion (rule outcome -> hand label)")
    for (rule, human), k in sorted(r["confusion"].items()):
        if rule != human:
            print(f"  {rule:10s} -> {human:10s} {k}")
    print("set H: Hall flag precision by model")
    for model in HALL_QUOTA:
        inv, n = r["hall"][(model, "invented")], sum(v for (m, _), v in r["hall"].items() if m == model)
        print(f"  {model:8s} invented {inv:2d}/{n:2d}")


if __name__ == "__main__":
    main()