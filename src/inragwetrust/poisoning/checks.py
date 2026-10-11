"""Automatic checks for poisoned passages (decision D116).

Blocking checks must all pass; partial leaks and length are measured and
reported. Run the full report with:
    uv run python -m inragwetrust.poisoning.checks
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from inragwetrust.textmatch import contains_normalized, contains_span

DIGITS = re.compile(r"\d+")
LETTERS = re.compile(r"[A-Za-z]+")


def leak_tokens(gold: str, target: str) -> tuple[set[str], str | None]:
    """Numbers of gold, and its last word (4+ letters), that do not also occur in target."""
    numbers = set(DIGITS.findall(gold)) - set(DIGITS.findall(target))
    words = [w for w in LETTERS.findall(gold) if len(w) >= 4 and w[0].isupper()]
    last = words[-1] if words else None 
    return numbers, last


def check_poison(clean: str, poisoned: str, swapped: str, gold: str, target: str,
                 name: str, templates: tuple[str, ...]) -> dict:
    """Blocking failures, partial leaks and length ratio of one poisoned passage."""
    blocking = []
    if not contains_span(target, poisoned):
        blocking.append("target_missing")
    if contains_normalized(gold, poisoned):
        blocking.append("gold_present")
    if name == "swap":
        well_formed = poisoned == swapped
    else:
        well_formed = any(poisoned == swapped.rstrip() + " " + t for t in templates)
    if not well_formed:
        blocking.append("framing")

    numbers, last = leak_tokens(gold, target)
    partial = []
    if any(re.search(rf"(?<!\d){n}(?!\d)", poisoned) for n in numbers):
        partial.append("number_leak")
    if last and re.search(rf"\b{re.escape(last)}\b", poisoned):
        partial.append("word_leak")
    return {"blocking": blocking, "partial": partial,
            "length_ratio": len(poisoned) / max(len(clean), 1)}


def report(path=None, ids=None) -> dict:
    """Check every poisoned passage (optionally only for the given question ids) and summarise."""
    from inragwetrust.data.build import PROCESSED_DIR
    from inragwetrust.poisoning.build import read_jsonl_gz
    from inragwetrust.poisoning.poisons import POISON_TYPES

    path = path or PROCESSED_DIR / "poisons.jsonl.gz"
    blocking, partial, ratios = Counter(), defaultdict(Counter), []
    bad_questions, leaky_questions = {}, defaultdict(set)
    records = [r for r in read_jsonl_gz(path) if ids is None or r["id"] in ids]
    for rec in records:
        for p in POISON_TYPES:
            for i, clean in enumerate(rec["clean"]):
                res = check_poison(clean, rec["poisoned"][p.name][i], rec["poisoned"]["swap"][i],
                                   rec["gold_answer"], rec["target_answer"], p.name, p.templates)
                blocking.update(res["blocking"])
                if res["blocking"]:
                    bad_questions.setdefault(rec["id"], set()).update(res["blocking"])
                for leak in res["partial"]:
                    partial[p.name][leak] += 1
                    leaky_questions[leak].add(rec["id"])
                if p.name == "swap":
                    ratios.append(res["length_ratio"])
    ratios.sort()
    return {
        "poisoned_passages": len(records) * 16,
        "blocking_failures": dict(blocking),
        "questions_with_blocking_failures": {
            q: sorted(v) for q, v in bad_questions.items()
        },
        "partial_leaks_per_type": {k: dict(v) for k, v in partial.items()},
        "questions_with_partial_leak": {k: len(v) for k, v in leaky_questions.items()},
        "swap_length_ratio": {"min": round(ratios[0], 2),
                              "median": round(ratios[len(ratios) // 2], 2),
                              "max": round(ratios[-1], 2)},
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
