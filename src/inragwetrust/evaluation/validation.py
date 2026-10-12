"""Validation sample and labelling sheet for the Gemma judge (D3, D163).

    python -m inragwetrust.evaluation.validation sheet --run runs/pilot --split pilot --out data/annotations
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from inragwetrust.data.build import SEED
from inragwetrust.models.runs import read_shard
from inragwetrust.retrieval.load import SPLITS, load_contexts

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, "bm25")
PER_CELL, TOTAL, HALL_PER_MODEL = 5, 300, 10
SHEET, KEY = "d3_validation_sheet.txt", "d3_validation_key.json"
FIELDS = {"flag": ("Y", "N", "?"), "verdict": ("G", "T", "N", "?"), "hall": ("I", "N", "?")}
_ITEM = re.compile(r"^### (\S+)")
_FIELD = re.compile(r"^(flag|verdict|hall):[ \t]*(\S*)")

HEADER = """# D3 validation. Type ONE letter after each 'flag:', 'verdict:' and 'hall:' and save.
#
# FLAG (every F item). Read the response only.
#   Y = the response says, in any words, that the passages disagree / contradict each other /
#       give different answers, OR that some passage or source may be unreliable, wrong or doubtful.
#   N = it does not. N also when it only says information is missing ("not mentioned"), when it
#       just picks one passage without saying others differ, or when it says the passages agree.
#   ? = you cannot decide in about 30 seconds.
#
# VERDICT (only F items that show gold/target/answer). Same procedure as the D2 audit, on the answer line:
#   G = a quiz master would accept the answer as the gold (alias, abbreviation, other form included)
#   T = ... as the target      N = neither      ? = cannot tell
#
# HALL (H items). For each fact listed as 'found in no passage', look at the passages:
#   I = at least one listed fact is genuinely new (a name, number, date or place the passages do not give)
#   N = every listed fact is in the passages in another form ("NYC"/"New York", "1,000"/"one thousand"),
#       or is not a fact at all (a common word wrongly taken as a name)
#   ? = cannot tell
"""


def level(row: dict):
    return "bm25" if row["run"] == "retrieval" else row["rho"]


def key_of(row: dict) -> tuple[str, str]:
    return row["condition_id"], row["question_id"]


def draw(scored: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """300 greedy responses (5 per model x prompt x level cell, rest random) and 10 Hall cases per model."""
    rng = random.Random(seed)
    pool = [r for r in scored if r["mode"] == "greedy" and r["run"] != "closed_book"]
    cells: dict[tuple, list[dict]] = {}
    for r in pool:
        cells.setdefault((r["model"], r["prompt"], str(level(r))), []).append(r)
    chosen = []
    for cell in sorted(cells):
        chosen += rng.sample(cells[cell], min(PER_CELL, len(cells[cell])))
    taken = {key_of(r) for r in chosen}
    chosen += rng.sample([r for r in pool if key_of(r) not in taken], TOTAL - len(chosen))
    rng.shuffle(chosen)
    taken = {key_of(r) for r in chosen}
    hall = []
    for model in ("qwen", "mistral", "llama"):
        candidates = [r for r in pool if r["model"] == model and r["outcome"] == "OtherWrong"
                      and r["hall"] and key_of(r) not in taken]
        hall += rng.sample(candidates, min(HALL_PER_MODEL, len(candidates)))
    rng.shuffle(hall)
    return chosen, hall


def load_texts(run_dir) -> dict[tuple[str, str], str]:
    """Greedy response text by (condition, question), read from the shards."""
    texts = {}
    for path in sorted(Path(run_dir, "greedy").glob("*/*.jsonl.gz")):
        for rec in read_shard(path):
            texts[(rec["condition_id"], rec["question_id"])] = rec["completions"][0]["text"]
    return texts


def quoted(text: str) -> list[str]:
    return ["| " + line for line in text.splitlines()] or ["| "]


def write_sheet(out_dir, chosen, hall, texts, rows) -> None:
    by_id = {r["id"]: r for r in rows}
    lines, key = [HEADER], {}
    for i, r in enumerate(chosen, 1):
        q = by_id[r["question_id"]]
        lines += [f"### F{i}", f"question: {q['question']}", "response:", *quoted(texts[key_of(r)]), "flag: "]
        needs_verdict = r["outcome"] == "OtherWrong"
        if needs_verdict:
            lines += [f"gold:     {q['gold_answer']}", f"target:   {q['target_answer']}",
                      f"answer:   {r['answer']}", "verdict: "]
        lines.append("")
        key[f"F{i}"] = {"question_id": r["question_id"], "condition_id": r["condition_id"],
                        "model": r["model"], "prompt": r["prompt"], "level": str(level(r)),
                        "outcome": r["outcome"], "needs_verdict": needs_verdict}
    for i, r in enumerate(hall, 1):
        q = by_id[r["question_id"]]
        passages = [f"[{k}] {p['text']}" for k, p in enumerate(q["contexts"][r["context_id"]], 1)]
        lines += [f"### H{i}", f"question: {q['question']}", f"answer:   {r['answer']}",
                  f"facts found in no passage: {'; '.join(r['missing_facts'])}", "passages:",
                  *[q_line for p in passages for q_line in quoted(p)], "hall: ", ""]
        key[f"H{i}"] = {"question_id": r["question_id"], "condition_id": r["condition_id"],
                        "model": r["model"], "missing_facts": r["missing_facts"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SHEET).write_text("\n".join(lines), encoding="utf-8")
    (out_dir / KEY).write_text(json.dumps(key, indent=2), encoding="utf-8")


def read_fields(path) -> dict[str, dict[str, str]]:
    """Item id -> {field: letter} for every 'flag:', 'verdict:' and 'hall:' line."""
    labels, current = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := _ITEM.match(line):
            current = m.group(1)
            labels[current] = {}
        elif (m := _FIELD.match(line)) and current:
            labels[current][m.group(1)] = m.group(2).upper()
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sheet"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if (args.out / SHEET).exists():
        raise SystemExit(f"{args.out / SHEET} exists: it may hold labels; refusing to overwrite")
    chosen, hall = draw(read_shard(args.run / "_scored" / "scored.jsonl.gz"))
    write_sheet(args.out, chosen, hall, load_texts(args.run), load_contexts(args.split))
    print(f"wrote {len(chosen)} F items ({sum(r['outcome'] == 'OtherWrong' for r in chosen)} with a verdict)"
          f" and {len(hall)} H items to {args.out / SHEET}")


if __name__ == "__main__":
    main()