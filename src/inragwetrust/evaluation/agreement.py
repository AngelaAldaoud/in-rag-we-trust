"""Agreement of the Gemma judge and of the Hall rule with the hand labels (D3, D167).

    python -m inragwetrust.evaluation.agreement flag    --judge runs/pilot/_judge_v1/judge.jsonl.gz
    python -m inragwetrust.evaluation.agreement hall
    python -m inragwetrust.evaluation.agreement verdict --judge runs/pilot/_judge/judge.jsonl.gz --scored runs/pilot/_scored/scored.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from inragwetrust.data.build import SEED
from inragwetrust.evaluation.validation import HALL_KEY, HALL_SHEET, KEY, SHEET, read_fields
from inragwetrust.models.runs import read_shard

ANNOTATIONS = Path("data/annotations")
BOOTSTRAP = 10_000          # resamples (D52)
KAPPA_MIN = 0.6             # acceptance threshold (D48)
INTERNAL_MIN = 0.95         # verdict internal check (D165)
Z = 1.959964                # 95% normal quantile


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement beyond chance between two raters; nan if both always give the same single label."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two non-empty label lists of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def bootstrap_kappa(a: list[str], b: list[str], n_boot: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of kappa over items resampled with replacement."""
    rng, n, values = random.Random(seed), len(a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(k):
            values.append(k)
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson 95% interval for a proportion k/n (D52)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def judge_labels(judge_rows: list[dict], task: str) -> dict[tuple[str, str], str | None]:
    return {(r["condition_id"], r["question_id"]): r["label"] for r in judge_rows if r["task"] == task}


def pairs(labels: dict, key: dict, judged: dict, field: str, only_verdict: bool = False):
    """(item, human, judge, info) for every F item with a usable human and judge label; counts of exclusions."""
    out, skipped = [], Counter()
    for item, info in key.items():
        if not item.startswith("F") or (only_verdict and not info.get("needs_verdict")):
            continue
        human = labels.get(item, {}).get(field, "")
        machine = judged.get((info["condition_id"], info["question_id"]))
        if human in ("", "?"):
            skipped["human ?"] += 1
        elif machine is None:
            skipped["judge missing/unparseable"] += 1
        else:
            out.append((item, human, machine, info))
    return out, skipped


def summarize(rows) -> dict:
    human, machine = [r[1] for r in rows], [r[2] for r in rows]
    return {"n": len(rows), "kappa": cohen_kappa(human, machine), "ci": bootstrap_kappa(human, machine),
            "agree": sum(h == m for h, m in zip(human, machine)) / len(rows),
            "human": Counter(human), "judge": Counter(machine), "table": Counter(zip(human, machine))}


def by_group(rows, field: str) -> dict[str, tuple[int, float]]:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[3][field], []).append(r)
    return {g: (len(rs), cohen_kappa([r[1] for r in rs], [r[2] for r in rs])) for g, rs in sorted(groups.items())}


def internal_check(scored: list[dict], judged: dict, exclude: set) -> dict[str, tuple[int, float]]:
    """Share of rule-Correct answers judged G and rule-Poisoned judged T, outside the validation sample."""
    want = {"Correct": "G", "Poisoned": "T"}
    out = {}
    for outcome, label in want.items():
        got = [judged[(r["condition_id"], r["question_id"])] for r in scored
               if r["mode"] == "greedy" and r["outcome"] == outcome
               and (r["condition_id"], r["question_id"]) not in exclude
               and (r["condition_id"], r["question_id"]) in judged]
        out[outcome] = (len(got), sum(g == label for g in got) / len(got) if got else float("nan"))
    return out


def hall_results(labels: dict, key: dict) -> dict[str, dict]:
    """Precision of the D166 flag (H items: I among I+N) and its miss rate (M items: M among M+K), by model."""
    out = {}
    for prefix, field, yes, no in (("H", "hall", "I", "N"), ("M", "miss", "M", "K")):
        per_model, total = Counter(), Counter()
        for item, info in key.items():
            if item.startswith(prefix):
                v = labels.get(item, {}).get(field, "")
                if v in (yes, no):
                    per_model[(info["model"], v == yes)] += 1
                    total[v == yes] += 1
                elif v == "?":
                    total["?"] += 1
        out[prefix] = {"yes": total[True], "n": total[True] + total[False], "unclear": total["?"],
                       "by_model": {m: (per_model[(m, True)], per_model[(m, True)] + per_model[(m, False)])
                                    for m in ("qwen", "mistral", "llama")}}
    return out


def print_summary(title: str, s: dict) -> None:
    verdict = "ACCEPT" if s["kappa"] >= KAPPA_MIN else f"BELOW {KAPPA_MIN}"
    print(f"{title}: n={s['n']} | kappa {s['kappa']:.3f} (95% CI {s['ci'][0]:.3f} to {s['ci'][1]:.3f})"
          f" | raw agreement {s['agree']:.1%} -> {verdict}")
    print(f"   human labels {dict(s['human'])} | judge labels {dict(s['judge'])}")
    print("   table (human, judge):", dict(sorted(s["table"].items())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["flag", "verdict", "hall"])
    parser.add_argument("--judge", type=Path, help="judge.jsonl.gz (flag, verdict)")
    parser.add_argument("--scored", type=Path, help="scored.jsonl.gz (verdict internal check)")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    args = parser.parse_args(argv)

    if args.action == "hall":
        r = hall_results(read_fields(args.annotations / HALL_SHEET),
                         json.loads((args.annotations / HALL_KEY).read_text(encoding="utf-8")))
        for prefix, name in (("H", "precision (flagged answers truly invented)"),
                             ("M", "miss rate (unflagged answers that do contain an invented fact)")):
            x = r[prefix]
            lo, hi = wilson(x["yes"], x["n"])
            print(f"Hall D166 {name}: {x['yes']}/{x['n']} = {x['yes'] / x['n']:.1%} "
                  f"(Wilson 95% {lo:.1%} to {hi:.1%}); unclear {x['unclear']}")
            print("   by model:", {m: f"{k}/{n}" for m, (k, n) in x["by_model"].items()})
        return

    labels = read_fields(args.annotations / SHEET)
    key = json.loads((args.annotations / KEY).read_text(encoding="utf-8"))
    judge_rows = read_shard(args.judge)
    field = args.action
    judged = judge_labels(judge_rows, field)
    rows, skipped = pairs(labels, key, judged, field, only_verdict=(field == "verdict"))
    print_summary(f"{field} vs hand labels", summarize(rows))
    print("   excluded:", dict(skipped))
    for group in ("prompt", "model"):
        print(f"   kappa by {group}:", {g: f"{k:.2f} (n={n})" for g, (n, k) in by_group(rows, group).items()})
    if field == "verdict":
        exclude = {(v["condition_id"], v["question_id"]) for v in key.values()}
        for outcome, (n, share) in internal_check(read_shard(args.scored), judged, exclude).items():
            target = "G" if outcome == "Correct" else "T"
            status = "PASS" if share >= INTERNAL_MIN else f"FAIL (< {INTERNAL_MIN:.0%})"
            print(f"   internal check: rule {outcome} judged {target}: {share:.1%} of {n} -> {status}")


if __name__ == "__main__":
    main()