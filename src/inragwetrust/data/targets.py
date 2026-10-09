"""Target (false) answer for each eligible question (decisions D13, D31, D94).

The target answer is the gold answer of another eligible question with the
same fine entity label (and, for untyped answers, the same length bucket),
drawn reproducibly per question from the project seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from inragwetrust.data.entity_types import NO_ENTITY
from inragwetrust.data.pool import PoolQuestion
from inragwetrust.textmatch import normalize


@dataclass(frozen=True)
class Target:
    """A false answer and the question it was taken from."""

    answer: str
    source_id: str


def length_bucket(answer: str) -> int:
    """Number of words, with 4 meaning four or more."""
    return min(len(answer.split()), 4)


def group_key(fine: str, answer: str) -> tuple[str, int]:
    """Answers can replace each other only within the same group."""
    return (fine, length_bucket(answer) if fine == NO_ENTITY else 0)


def _contains(needle: str, haystack: str) -> bool:
    """Whole-word containment between two already-normalised strings."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def is_valid_target(q: PoolQuestion, n_target: str, n_texts: dict[str, str]) -> bool:
    """D31 + D94 checks, on normalised strings."""
    n_gold = n_texts["answer"]
    if not n_target or n_target == n_gold:
        return False
    if _contains(n_target, n_gold) or _contains(n_gold, n_target):
        return False
    return not any(
        _contains(n_target, n_texts[k]) for k in ("question", "answer_passage", "bridge_passage")
    )

def choose_candidates(
    pool: list[PoolQuestion], fine_labels: list[str], seed: int, k: int = 30
) -> dict[str, list[Target]]:
    """Up to k valid target candidates per question, in seeded order (D94, D95).

    The first candidate equals a plain seeded random draw; the final target is
    chosen among the candidates in B2 (Gemma, plausibility), falling back to it.
    """
    groups: dict[tuple[str, int], list[tuple[str, str, str]]] = defaultdict(list)
    seen: set[tuple[tuple[str, int], str]] = set()
    for q, fine in zip(pool, fine_labels):
        key, n_ans = group_key(fine, q.answer), normalize(q.answer)
        if n_ans and (key, n_ans) not in seen:
            seen.add((key, n_ans))
            groups[key].append((q.answer, n_ans, q.id))

    result: dict[str, list[Target]] = {}
    for q, fine in zip(pool, fine_labels):
        n_texts = {
            "answer": normalize(q.answer),
            "question": normalize(q.question),
            "answer_passage": normalize(q.answer_passage.text),
            "bridge_passage": normalize(q.bridge_passage.text),
        }
        candidates = [c for c in groups[group_key(fine, q.answer)] if c[2] != q.id]
        rng = random.Random(f"{seed}-{q.id}")
        rng.shuffle(candidates)
        valid: list[Target] = []
        for answer, n_target, source in candidates:
            if is_valid_target(q, n_target, n_texts):
                valid.append(Target(answer, source))
                if len(valid) == k:
                    break
        result[q.id] = valid
    return result